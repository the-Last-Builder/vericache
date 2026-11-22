# VeriCache Benchmark Suite (Suites 1–5)

Reproducible end-to-end hardware evaluation comparing **VeriCache** (formally verified Verus kernel + hardware backends) against official **Mooncake `0.3.14`** (`mooncake-transfer-engine-cuda13`) and **LMCache `0.5.7.post1`** on identical TOML workload specifications. Optional NUMA pinning is controlled via `VERICACHE_NUMA_NODE` (e.g., `VERICACHE_NUMA_NODE=1`).

## Suites

| Suite | Spec | Systems | Hardware Path |
|---|---|---|---|
| **Suite 1** (`s1_local_tiers`) | `workloads/s1_local_tiers.toml` | VeriCache, LMCache | 1A: 16 non-contiguous GPU Paged KV blocks ↔ Pinned Host Frame (`64 KB–16 MB`); 1B: NVMe `O_DIRECT` write spill & cold read (`io_uring` vs `LocalDiskBackend`) |
| **Suite 2** (`s2_rdma_ablation`) | `workloads/s2_rdma_ablation.toml` | VeriCache (`raw_mw`, `dynamic_lease`), Mooncake | 2-process Donor (GPU 0) ↔ Initiator (GPU 1) over ConnectX RDMA (`VERICACHE_RDMA_DEVICE`): 2A Host→Host RDMA, 2B GPU→GPU GDR + BAR1 flush, 2C Isolation-Tax Ablation |
| **Suite 3** (`s3_trace_replay`) | `workloads/s3_trace_replay.toml` | VeriCache, Mooncake, LMCache | Official Mooncake FAST'25 `conversation_trace.jsonl` & `toolagent_trace.jsonl` + synthetic capacity-ratio sweep (`10%, 30%, 50%, 80%`) at `256 KB` & `4 MB`, closed-loop (Goodput GB/s) & open-loop (P99 µs) |
| **Suite 4** (`s4_safety_revoke`) | `workloads/s4_safety_revoke.toml` | VeriCache, Mooncake | Hotspot contention (`toolagent_trace.jsonl`, 15% capacity) with checksummed frames under `0.1%, 1%, 5%` forced unilateral revocations (`IBV_WC_REM_ACCESS_ERR` -> `abort_write` -> `reset_and_reconnect` vs static-`rkey` tearing) |
| **Suite 5** (`s5_scaleup_cluster`) | `workloads/s5_scaleup_cluster.toml` | VeriCache, Mooncake, LMCache | **600 GiB 2-Node 4-Tier Scale-Up Cluster**: `50 GiB` HBM + `200 GiB` Pinned DRAM + `50 GiB` NVMe per node (`8` shards/node, `8` RC QPs, `70/30` session-sticky routing, full FAST'25 `toolagent_trace` 23,608 reqs + `conversation_trace` 12,031 reqs at `2 MB`) |

## Reproducibility

1. Verify all proofs, AST audits, TCB whitelist, and formatting:
   ```bash
   ./image/dev.sh exec just check
   ```
2. Run suites serially on a PIX-paired GPU pair + RDMA HCA (set `VERICACHE_RDMA_GID_INDEX` and optionally `VERICACHE_NUMA_NODE` to match your host topology):
   ```bash
   ./image/dev.sh bench-hw <gpu0,gpu1> <uverbs_dev> <rdma_dev> python3 experiments/harness/orchestrate.py --all
   ```
3. Outputs are written to `experiments/results/<run-id>/` (gitignored):
   - `manifest.json`: hardware topology, git revision, tool versions, TOML specs
   - `summary.md`: full comparative markdown tables & key findings
   - `raw/*.jsonl`: raw per-job `MetricRecord` streams

## Schema

Every runner emits identical JSONL `MetricRecord` entries:
`system, suite, case, mode, chunk_bytes, num_blocks, concurrency, throughput_gbps, goodput_gbps, p50_us, p95_us, p99_us, hit_rate, cpu_pct, rss_mb, corrupted_frames, recovery_us, notes`.
