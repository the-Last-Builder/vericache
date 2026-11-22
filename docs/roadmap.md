# Roadmap

## Completed: Layers 0–2, Local Engine, and Milestone 3 (Distributed RDMA / GDR Engine)

- **Dual-SMT Verification (`Z3 ∧ CVC5`)**: All 1,210 proof items across `vericache-foundation` (92), `vericache-transfer` (343), and `vericache-core` (775, with `--no-cheating`) pass both Z3 4.16.0 and CVC5 1.3.4 (`--rlimit 60` on `vericache-core`).
- **Hardened Audit**: 10 `tree-sitter-verus` `ast-grep` rules + regression tests and `scripts/audit-tcb.sh` locking the exact 9-file FFI/C/build boundary.
- **Slot/Region Transfer & Kernel**: Single-key `SlotToken`/`ShareCap`/`RevokedToken` ownership over `SlotLoc + SlotRef + Seq<u8>`, cluster-wide write-once `VersionBinding`/`VersionRegistry`, 64-byte `FrameSyncMeta`, `SlidingBitmap`, `Catalog`, and `CacheCore` facade.
- **Pluggable Backends & Local Engine**:
  - `HostBackend`, `CudaBackend<'a>` (`PinnedHostSlab<'a>`, `CudaEvent` rings, boundary scatter/gather), `NvmeBackend` (`O_DIRECT` + `io_uring`), `RdmaBackend<'a>` (Type-2B MW `WindowPool` with `Quarantined` Option-A IBTA-strict revocation lifecycle, Host & GPUDirect RDMA MR registration with mandatory post-CQE `complete_gdr_slot` BAR1 visibility flush, batched RDMA reads/sends), and `compose` sibling-preservation summaries (`lemma_core_transition_preserves_{host,cuda,nvme,rdma}` and `lemma_peer_core_step_preserves_{host,idle_cuda,nvme}`).
  - `LocalEngine<'a>` with zero-copy `HostLease<'a>` / `GpuLease` and 1-hop / 2-hop routing (`GPU ↔ Host ↔ NVMe`).
- **Milestone 3 & Scale Wave 1 — 4-Tier Distributed RDMA / GDR + NVMe Engine**:
  - Authenticated peer protocol & session (`vericache-core/src/backends/rdma/{protocol,pull,peer}.rs`) using `Channel<T, VeriCacheAttachment>::recv_any`, constructive `VeriCacheAttachment`, `PullTarget::{Host, GpuGdr}`, cooperative `ReadRelease`, Option-A unilateral MW revocation (`LOCAL_INV` + `ibv_dealloc_mw` -> `Quarantined`), and QP `Error -> Reset -> Reconnect` recovery (`reset_and_reconnect`).
  - Verified 4-tier `DistributedEngine<'a, T>` (`vericache-core/src/engine/distributed.rs`) unifying `CacheCore`, `HostBackend`, optional `CudaBackend`, optional `NvmeBackend` (`io_uring` `O_DIRECT`), `RdmaBackend<'a>`, and `PeerSession<'a, T>` (`HBM + DRAM + Local NVMe + Peer RDMA/GDR`) with sync/async NVMe spill/warm and bounded `poll_nvme_completions` / `wait_nvme_completion` progress.
  - Real-hardware 2-node E2E suite (`vericache-core/tests/distributed_e2e.rs`) covering Host DRAM RDMA pull/release/reclaim, 2-GPU GPUDirect RDMA pull + BAR1 flush + D2H verification, evict/reclaim + re-pull `VersionBinding` agreement, Option-A unilateral revocation -> `IBV_WC_REM_ACCESS_ERR` -> `abort_write` -> `Error -> Reset -> Reconnect` -> clean re-fetch, and 4-tier HBM -> DRAM -> Local NVMe `io_uring` spill -> reclaim -> warm -> 2-GPU GPUDirect RDMA roundtrip.

---

## Intentionally Decoupled Outside the Verified Core

- **Placement & Eviction Policy (`policy.rs`)** and **Multi-Shard Routing / Cluster Scheduling (`engine/shard.rs`)** are intentionally kept decoupled outside the `vericache-core` verified core so heuristic policy and sharding layers can evolve freely without enlarging the verified safety kernel or weakening `CacheCore` / `ValueCertificate` invariants.
