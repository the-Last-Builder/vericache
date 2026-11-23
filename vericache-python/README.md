# VeriCache for Python and vLLM

The `vericache` Python package uses one Rust shared library,
`libvericache_python.so`. It provides two backends.

- **Distributed** exposes `VeriCacheClient` and `vericache.vllm_connector.VeriCacheConnector`.
  Each client owns a Rust worker thread for DRAM, GPU, NVMe, and optional RDMA access.
- **Local** exposes `NativeVeriCacheEngine`, `PinnedStagingBuffer`, and
  `vericache.vllm_v1_connector.VeriCacheV1Connector` for DRAM and optional NVMe
  KV prefix reuse in vLLM 0.13.0.

The first validated initialization claims its backend for the lifetime of
the process. Allocating a staging buffer also claims Local. Multiple handles
of the same backend can coexist, with a shared node ID for Distributed clients.
Use separate processes to run both backends. A plain `import vericache` works
without PyTorch and vLLM.

The Local connector supports eager execution, one full-attention KV cache
group, and text prompts without LoRA, multimodal inputs, or prompt embeddings.
A prefix hit requires consecutive full prompt blocks and all KV layers.
vLLM still computes at least the final prompt token to produce logits.

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
- The 28 existing Local C exports retain their signatures. The `*_key256`
  exports take four contiguous little-endian `u64` words per key. Python
  accepts full `ObjectKey` tuples and legacy integers, including in batches.
  `drain_evictions()` returns full key tuples. The engine also exposes
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

Run these commands from `vericache-python/` with the repository Rust toolchain,
the Verus checkout at `/opt/vericache/verus`, and the Linux build dependencies
for `vericache-transfer`. The vLLM tests use Python with PyTorch, vLLM 0.13.0,
and pytest.

With a CUDA toolkit, the transfer crate searches `CUDA_HOME`, then `CUDA_PATH`,
then `/usr/local/cuda`. When CUDA is available only through the
`nvidia.cuda_runtime` wheel, run the following in the same shell before building
or testing. The ignored symlink supplies the linker name and
`LD_LIBRARY_PATH` supplies the runtime search directory.

```bash
cuda_runtime="$(python3 -c 'from importlib.util import find_spec; print(next(iter(find_spec("nvidia.cuda_runtime").submodule_search_locations)) + "/lib")')"
mkdir -p target/cuda-link/lib
ln -sfn "$cuda_runtime/libcudart.so.12" target/cuda-link/lib/libcudart.so
export CUDA_HOME="$PWD/target/cuda-link"
export LD_LIBRARY_PATH="$cuda_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

```bash
cargo tree --offline --locked
cargo test --release --offline --locked
cargo build --release --offline --locked
readelf -d target/release/libvericache_python.so
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -v tests/test_local_vllm_connector.py tests/test_backend_mode.py
python3 -m pytest --collect-only -q tests/test_vllm_connector.py
(cd .. && ./scripts/audit-tcb.sh)
```

The loader accepts an explicit Local `lib_path`, then checks
`VERICACHE_PYTHON_LIB` and the legacy `VERICACHE_VLLM_LIB_PATH` fallback.
Both environment variables point to `libvericache_python.so`. Without an
override, the loader searches beside the Python package and under
`target/release/` and `target/debug/`. With an installed package and a
separately built library, set `VERICACHE_PYTHON_LIB` to the built file.

The NVMe tests require direct I/O and permission to create an `io_uring`.
The Local CUDA test runs when PyTorch detects a CUDA device. The preserved
Distributed E2E suite uses CUDA, NVMe, and RDMA. The command above checks that
it imports and collects on CPU hosts.

## Use with vLLM V1

With a `uni` executor, TP=1, and no `ipc_endpoint`, the scheduler and worker
share the native engine in-process. For separate processes or TP greater
than one, configure the same `ipc_endpoint` for the engine. The scheduler
uses the metadata server at that endpoint or starts one. Each worker has its
own native engine and KV keys. For NVMe, the worker adds `.rank-<tp_rank>`
before any shard suffix. A warm prefix requires all layers on every rank.
The metadata service carries presence information, while KV bytes stay in
the owning worker. Prefix hashes chain SHA-256 over complete token blocks.
IPC encodes the 32-byte prefix digests and keys as 64 lowercase hex characters.
The worker reports late misses through vLLM's load error interface.

This example uses a local Unix domain socket and TP=2. For a TCP endpoint,
use `tcp://host:port`. vLLM supplies the TP rank and world size. Custom
launchers can set `tp_rank` on each worker and `world_size` on all roles.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export VERICACHE_PYTHON_LIB="$PWD/target/release/libvericache_python.so"
vllm serve /path/to/model \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --disable-hybrid-kv-cache-manager \
  --block-size 16 \
  --kv-transfer-config '{
    "kv_connector": "VeriCacheV1Connector",
    "kv_connector_module_path": "vericache.vllm_v1_connector",
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
`register_vericache_connector` from `vericache`.

## Benchmark

```bash
PYTHONPATH="$PWD/python" python3 bench/bench_phase2.py --without-nvme --repeats 3
PYTHONPATH="$PWD/python" python3 bench/bench_phase2.py --pressure --repeats 3
```

The default workload is 16 layers × 64 blocks × 64 KiB (64 MiB). The script
uses scalar calls and staging as the Phase 1 transfer reference, with
synchronous QD=1 spills and eager writeback disabled. It compares that
path with the Phase 2 connector, checks exact KV restoration, and reports
median save, load, and round-trip latency and throughput. The default enables
NVMe and sizes DRAM to hold the entire workload. `--pressure` sizes DRAM for
about half the data to exercise demotion and promotion. Use `--without-nvme`
for a DRAM-only run and `--device cuda` on a CUDA host.
