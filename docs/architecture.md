# Architecture & Memory Model

VeriCache separates pure concurrency math from hardware FFI and storage management across three verified workspace crates plus a zero-`assume_new` benchmark suite:

```text
  vericache-bench        Multi-shard benchmark driver (ShardAllocator / ShardCapabilityBundle, zero assume_new)
         │
         ▼
  vericache-core         Hardware-free CacheCore (spec, provenance, timeline, frame, catalog, kernel)
                         + Pluggable backends (host, cuda, nvme, rdma, compose)
                         + LocalEngine<'a> & Multi-Peer DistributedEngine<'a, T>
         │
         ▼
  vericache-transfer     SlotLoc/SlotRef domain, NodeContext & ShardAllocator,
                         single-key SlotToken/ShareCap/RevokedToken/VersionBinding,
                         RAII-guarded Host / CUDA / io_uring / libibverbs wrappers,
                         CRC32-checked non-blocking Channel<T, A> over ByteTransport
         │
         ▼
  vericache-foundation   Domain trait, Access/Conflict/Effect, strict partial orders,
                         History state machine (race_free, every_event_isolated, read_returns_last_write),
                         HistoryTokens (VerusSync tokenized state machine)
```

---

## 1. Slot/Region & Share-Nothing Shard Model (`SlotLoc + SlotRef + ShardAllocator`)

Instead of tracking per-byte `IMap<PhysicalLoc, Slot>` tokens, VeriCache models physical storage at slot/region granularity and scales across CPU/GPU threads via share-nothing capability partitioning:

- **One key per slot (`SlotLoc`)**: Every physical frame or buffer slot maps 1-to-1 to a single `SlotLoc { node, space, slot_index }`, backed by `Value = Seq<u8>`.
- **Zero-splitting linear ownership**: Slab allocators (`alloc_slab`, `create_slab_at`) yield one single-key `SlotToken` per slot (`Slot::Owned`). Read sharing issues `ShareCap`, and unilateral RDMA fencing produces `RevokedToken`.
- **Share-nothing multi-shard concurrency (`ShardAllocator` / `ShardSources`)**: `NodeContext` splits into arbitrary independent `ShardSources`, carving disjoint `[version_seq_start, version_seq_limit)` `VersionSource` intervals, disjoint `AllocSource` ID streams, `EventSourceAllocator` streams, and `ShareSource` streams so independent `LocalEngine` or `DistributedEngine` shards run in parallel without cross-shard locking.
- **Multi-size KV tiers & boundary scatter/gather (`SlotRef`)**: Each registered `TierId` carries its own `frame_size` (`register_tier_with_frame_size`), supporting heterogeneous KV block sizes across tiers on one node. Sub-slice transfers (`put_gpu_scatter`, `get_into_gpu_scatter`, multi-SGE RDMA) operate over `SlotRef { loc, slot_len, sub_offset, sub_len }`.

---

## 2. Layer Responsibilities & End-to-End Guarantees

- **`vericache-foundation`**: Proves that if every submitted access holds either exclusive `Slot::Owned` or shared `ShareCap` tokens admitting its ordering frontier, then reachable histories are `race_free` and every read observes the value written by the unique causally preceding write (`read_returns_last_write`).
- **`vericache-transfer`**: Instantiates `TransferDomain` with concrete hardware resources and events. All `external_body` / `unsafe` FFI premises live strictly in the 6 `src/*/ffi.rs` files:
  - **RAII FFI safety & inline control WR cookies**: Raw streams, events, `io_uring` instances, device contexts, PDs, CQs, QPs, MRs, and Type-2B MWs carry `Drop` guards. Control WRs (`BIND_MW`, `LOCAL_INV`) pack 62-bit inline WR IDs (`< CONTROL_WR_ID_LIMIT`) via `verbs.c`'s `wide_cookie` union with zero per-WR heap allocation.
  - **Named/block `O_DIRECT` NVMe & channel framing**: `NvmeFile` supports temporary files, persistent named files, and raw block devices (`BLKGETSIZE64`). `Channel<T, A>` enforces protocol magic, trailing CRC32 integrity, duplicate-grant-tag rejection, and non-blocking `try_recv_any`.
- **`vericache-core`**:
  - **Hardware-free kernel (`spec`, `provenance`, `timeline`, `frame`, `catalog`, `kernel`)**: Enforced import-free of `transfer::{host,cuda,nvme,rdma,transport}` via `ast-grep`. Provides `CacheCore` with open `TierId(u16)`/`TierKind`, flat open-addressed `Catalog` with backward-shift deletion, 4096-slot `SlidingBitmap` (`BITMAP_CAPACITY = 4096`), per-frame `overflow_seqs` reclamation liveness (eliminating permanent hot-frame stranding), optional 64-bit FNV-1a payload integrity verification, and `ValueCertificate` / `VersionBinding` provenance (including `prebind_local_version` for async GPU puts).
  - **Service guarantees**:
    - *Value Integrity*: Write-once monotonic `ServiceSpec.versions` and cluster-wide `VersionBinding` agreement; every published replica carries a verified `ValueCertificate`.
    - *Execution Readiness*: `SyncPoint` (`HostComplete` or generation-checked `TimelineEvent`) orders downstream GPU work; `is_sync_point_ready` turns `true` only after explicit `poll_events` confirmation and slot retirement.
    - *Storage Protection*: Active `HostLease<'a>`, `GpuLease`, or remote RDMA Type-2B MW leases hold frames out of `ReusableForWrite`. Reclamation obeys **Observed Reclamation** (all write, in-window read, and `overflow_seqs` tickets retired) or **Ordered Handoff** (consumer stream ordering).

---

## 3. Borrow Architecture, Multi-Peer RDMA & Cross-Backend Composition

VeriCache achieves concurrent 4-tier access across CPU, GPU, NVMe, and $N$ RDMA peers with **zero `unsafe`** outside `vericache-transfer/src/*/ffi.rs` and zero self-referential structs:

1. `HostBackend` owns the 4096-aligned `HostBuffer` slab and exposes `&self`-shared `put_host_with` / `read_frame_into_with` using an external `&mut HostContext`.
2. `CudaBackend<'a>` optionally pins `&'a HostBackend`'s buffer as a `PinnedHostSlab<'a>` (`attach_pinned_host`).
3. `LocalEngine<'a>` coordinates `CacheCore`, `host: &'a HostBackend`, a disjoint `HostContext`, and optional `CudaBackend<'a>` / `NvmeBackend`.
4. `DistributedEngine<'a, T>` coordinates `CacheCore`, `host: &'a HostBackend`, `CudaAttachment<'a>` (`None`, `Borrowed(&'a CudaBackend<'a>)` for GPUDirect RDMA MRs, or `Owned(CudaBackend<'a>)` with `pending_gpu_bindings` for async `submit_put_gpu_async` / `poll_gpu_completions`), optional `NvmeBackend`, and `peers: Vec<PeerSession<'a, T>>`:
   - **Shared MRs & $N$-peer mesh**: Multiple `PeerSession`s share one PD's `RegisteredSlab` (`attach_shared_host_slab` / `attach_shared_gpu_slab`) while maintaining pairwise disjoint `rdma_pinned_frame_shares` across sibling peers.
   - **Single-QP full-duplex multiplexing**: `RdmaBackend::pump_cq` routes data CQEs into `completed_pulls` and control CQEs onto `active_leases` without dropping unrelated completions, enabling deadlock-free symmetric `fetch_from_peer_duplex` and `poll_all_peers_progress`.
   - **Cooperative-first MW revocation**: `revoke_grant_cooperative_first` requests zero-dealloc `SoftRevokeRequest` -> `ReadRelease` -> `LOCAL_INV` reuse first, falling back on timeout to Option-A IBTA-strict fenced `LOCAL_INV` + `ibv_dealloc_mw` + `RevokeNotice` and `reset_and_reconnect_peer`.

### Cross-Backend Dependency DAG

```text
  backends/host   ──▶ transfer::host
  backends/cuda   ──▶ backends/host (PinnedHostSlab<'a>), transfer::cuda
  backends/nvme   ──▶ backends/host (staging slab), transfer::{nvme, host::HostContext}
  backends/rdma   ──▶ transfer::{rdma, host::{HostContext, HostBuffer}, cuda::CudaBuffer, transport}
```

Acyclic backend boundaries are mechanically enforced by `no-cuda-dep.yml`, `no-nvme-dep.yml`, and `no-remote-dep.yml`.

### Cross-Backend Frame Composition (`backends/compose/`)

Every backend method ensures opaque step predicates (`core_transition` with `PreserveAllInit` / `PreserveUnparkedInit` / `PreserveParkedInit`, plus `peer_core_step` and `peer_pinned_shares_step`), letting `LocalEngine<'a>` and `DistributedEngine<'a, T>` compose arbitrary multi-tier CPU/GPU/NVMe/RDMA transitions modularly.
