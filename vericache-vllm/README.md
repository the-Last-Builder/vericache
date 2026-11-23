# VeriCache vLLM Phase 2

VeriCache adds a DRAM and optional NVMe KV prefix cache to vLLM 0.13.0. A prefix
hit requires consecutive full prompt blocks and all KV layers. vLLM still
computes at least the final prompt token to produce logits. The connector
supports eager execution, one full-attention KV cache group, and text prompts
without LoRA, multimodal inputs, or prompt embeddings.

## Native engine and worker pipeline

- Read leases expose the native DRAM frame via the verified `LocalEngine`
  lease APIs. The engine excludes leased frames from eviction and rejects
  explicit eviction or replacement of a leased key. The Python lease
  exposes a read-only `memoryview` and a zero-copy PyTorch tensor. Treat the
  tensor as read-only and keep the lease alive through any asynchronous copy.
  Exported views retain the frame even after the Python lease or engine closes.
- Worker loads are submitted one layer at a time. Each layer acquires its
  blocks in one native batch. CUDA copies go directly from the slab to the
  paged KV cache, with a load-stream event governing host-frame release and
  compute-stream ordering. When CUDA registration is unavailable, the
  engine uses 4096-byte-aligned DRAM and blocking copies. Contiguous CPU blocks
  save directly from the KV tensor. CUDA and strided CPU blocks use a bounded
  staging pool.
- Puts and explicit writeback polling can keep up to four `io_uring` spills
  outstanding per shard. The default eager watermark is full DRAM occupancy.
  Completed spills leave clean NVMe replicas that can be demoted without SSD
  I/O. NVMe hits enter a probationary LRU and a second hit promotes them to
  the protected LRU. Up to four reserve frames per NVMe shard let lease
  batches pipeline cold reads without displacing the protected working set.
- The 14 original C exports retain their signatures. The Python wrapper
  also exposes batch contains, put, get, acquire, and release, plus
  `poll_writeback()` and `extended_stats()`.
  `set_writeback_policy(percent)` sets the watermark from 1 to 100 and
  accepts 0 to disable eager writeback. In that mode, scalar puts wait for
  one spill at a time. Batched puts still pipeline their admission spills.

NVMe uses `O_DIRECT` and `io_uring`. Supply a dedicated `nvme_path` and a
positive `nvme_frames_per_shard` to enable it. Native storage frames round up
to 4096 bytes and the API reports the original payload length. Multiple
shards use `<nvme_path>.shard-0`, `<nvme_path>.shard-1`, and so on.
Directory metadata and KV contents are usable for the lifetime of the engine.

## Build and test

From this directory, with the repository Rust toolchain, Verus checkout,
Linux dependencies, and a Python environment with vLLM, PyTorch, and pytest:

```bash
env -u CUDA_HOME -u LD_LIBRARY_PATH cargo test --release --offline
env -u CUDA_HOME -u LD_LIBRARY_PATH cargo build --release --offline
env -u CUDA_HOME -u LD_LIBRARY_PATH PYTHONPATH="$PWD/python" python3 -m pytest -v tests
(cd .. && ./scripts/audit-tcb.sh)
```

The build produces `target/release/libvericache_vllm.so`. An installed
package can use `VERICACHE_VLLM_LIB_PATH` or pass `lib_path`. The checkout
`.cargo/config.toml` locates the CUDA runtime even for CPU tests. On a fresh
build directory, create its runtime linker alias as follows. If the CUDA
runtime wheel lives elsewhere, update the library path and rpath in
`.cargo/config.toml` to match.

```bash
cuda_runtime="$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("nvidia.cuda_runtime").origin).parent / "lib")')"
mkdir -p target/cuda-link/lib
ln -sfn "$cuda_runtime/libcudart.so.12" target/cuda-link/lib/libcudart.so
```

The NVMe tests require direct I/O and permission to create an `io_uring`.
CUDA tests run when PyTorch detects a CUDA device.

## Use with vLLM V1

With a `uni` executor, TP=1, and no `ipc_endpoint`, the scheduler and worker
share the native engine in-process. For separate processes or TP greater
than one, configure the same `ipc_endpoint` for the engine. The scheduler
uses the metadata server at that endpoint or starts one. Each worker has its
own native engine and KV keys. For NVMe, the worker adds `.rank-<tp_rank>`
before any shard suffix. A warm prefix requires all layers on every rank.
The metadata service carries presence information, while KV bytes stay in
the owning worker. The worker reports late misses through vLLM's load error
interface.

This example uses a local Unix domain socket and TP=2. For a TCP endpoint,
use `tcp://host:port`. vLLM supplies the TP rank and world size. Custom
launchers can set `tp_rank` on each worker and `world_size` on all roles.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export VERICACHE_VLLM_LIB_PATH="$PWD/target/release/libvericache_vllm.so"
vllm serve /path/to/model \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --disable-hybrid-kv-cache-manager \
  --block-size 16 \
  --kv-transfer-config '{
    "kv_connector": "VeriCacheV1Connector",
    "kv_connector_module_path": "vericache_vllm.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "ipc_endpoint": "unix:///tmp/vericache-example.sock",
      "num_shards": 4,
      "host_frames_per_shard": 1024,
      "frame_size": 65536,
      "prefer_cuda_pinned": true,
      "io_workers": 2,
      "staging_pool_size": 8,
      "enable_async_pipeline": true,
      "use_zero_copy_load": true
    }
  }'
```

Set `frame_size` to at least the largest per-layer KV block on a worker.
`io_workers=0` or `enable_async_pipeline=false` runs worker tasks inline.
`use_zero_copy_load=false` selects the staging-based load path.
Applications can also import `MetadataServer`, `MetadataClient`, and
`register_vericache_connector` from `vericache_vllm`.

## Benchmark

```bash
env -u CUDA_HOME -u LD_LIBRARY_PATH PYTHONPATH="$PWD/python" python3 bench/bench_phase2.py
env -u CUDA_HOME -u LD_LIBRARY_PATH PYTHONPATH="$PWD/python" python3 bench/bench_phase2.py --pressure
```

The default workload is 16 layers × 64 blocks × 64 KiB (64 MiB). The script
uses the preserved single-block ABI and staging calls as the Phase 1 transfer
reference, with synchronous QD=1 spills and eager writeback disabled. It
compares them with the Phase 2 connector, checks exact KV restoration, and
reports median save, load, and round-trip latency and throughput. The default
enables NVMe and sizes DRAM
to hold the entire workload. `--pressure` sizes DRAM for about
half the data to exercise demotion and promotion. Use `--without-nvme` for
a DRAM-only run and `--device cuda` on a CUDA host.
