# Architecture & Memory Model

VeriCache separates pure concurrency math from hardware FFI and storage management across three workspace crates:

```text
  vericache-core         Hardware-free CacheCore (spec, provenance, timeline, frame, catalog, kernel)
                         + Pluggable backends (host, cuda, nvme, rdma, compose) + LocalEngine<'a>
         │
         ▼
  vericache-transfer     SlotLoc/SlotRef domain, NodeContext & single-key SlotToken/ShareCap/RevokedToken,
                         Host / CUDA / io_uring / libibverbs wrappers, Channel<T> over ByteTransport
         │
         ▼
  vericache-foundation   Domain trait, Access/Conflict/Effect, strict partial orders,
                         History state machine (race_free, every_event_isolated, read_returns_last_write),
                         HistoryTokens (VerusSync tokenized state machine)
```

---

## 1. Slot/Region Model (`SlotLoc + SlotRef + Value = Seq<u8>`)

Instead of tracking per-byte `IMap<PhysicalLoc, Slot>` tokens, VeriCache models physical storage at slot/region granularity:

- **One key per slot (`SlotLoc`)**: Every physical frame or buffer slot maps 1-to-1 to a single `SlotLoc { node, space, slot_index }`, backed by `Value = Seq<u8>`.
- **Zero-splitting linear ownership**: Slab allocators (`alloc_slab`, `create_temp_slab`) yield one single-key `SlotToken` per slot (`Slot::Owned`). Read sharing issues `ShareCap`, and unilateral RDMA fencing produces `RevokedToken`.
- **Boundary scatter/gather (`SlotRef`)**: Sub-slice transfers (`put_gpu_scatter`, `get_into_gpu_scatter`, multi-SGE RDMA) operate over `SlotRef { loc, slot_len, sub_offset, sub_len }` within one `SlotLoc`, proving byte-range disjointness and `splice_bytes` / `copy_effect` equivalence without `forall`-heavy `IMap` domain splitting.

---

## 2. Layer Responsibilities & End-to-End Guarantees

- **`vericache-foundation`**: Proves that if every submitted access holds either exclusive `Slot::Owned` or shared `ShareCap` tokens admitting its ordering frontier, then reachable histories are `race_free` and every read observes the value written by the unique causally preceding write (`read_returns_last_write`).
- **`vericache-transfer`**: Instantiates `TransferDomain` with concrete hardware resources and events. All `external_body` / `unsafe` FFI premises live strictly in `src/*/ffi.rs`, while buffer/stream/ring/MR/MW/channel invariants stay verified in Rust/Verus.
- **`vericache-core`**:
  - **Hardware-free kernel (`spec`, `provenance`, `timeline`, `frame`, `catalog`, `kernel`)**: Enforced import-free of `transfer::{host,cuda,nvme,rdma,transport}` via `ast-grep`. Provides `CacheCore` with open `TierId(u16)`/`TierKind`, 64-byte `FrameSyncMeta`, `SlidingBitmap`, `Catalog`, and `ValueCertificate` provenance.
  - **Service guarantees**:
    - *Value Integrity*: Write-once monotonic `ServiceSpec.versions`; every published replica carries a verified `ValueCertificate`.
    - *Execution Readiness*: `SyncPoint` (`HostComplete` or generation-checked `TimelineEvent`) orders downstream GPU work; `is_sync_point_ready` turns `true` only after explicit `poll_events` confirmation and slot retirement.
    - *Storage Protection*: Active `HostLease<'a>`, `GpuLease`, or remote leases hold frames out of `ReusableForWrite`. Reclamation obeys either **Observed Reclamation** (all write and read tickets retired, including `FLAG_READ_OVERFLOW`) or **Ordered Handoff** (consumer stream ordering).

---

## 3. Borrow Architecture & Cross-Backend Composition

VeriCache achieves concurrent multi-tier access across CPU, GPU, and NVMe with **zero `unsafe`** and zero self-referential structs:

1. `HostBackend` owns the 4096-aligned `HostBuffer` slab and exposes `&self`-shared `put_host_with` / `read_frame_into_with` using an external `&mut HostContext`.
2. `CudaBackend<'a>` optionally pins `&'a HostBackend`'s buffer as a `PinnedHostSlab<'a>` (`attach_pinned_host`).
3. `LocalEngine<'a>` holds `core: CacheCore`, `host: &'a HostBackend`, a disjoint `host_ctx: HostContext`, and optional `CudaBackend<'a>` / `NvmeBackend`, letting `HostLease<'a>` slices coexist with `&mut LocalEngine<'a>` GPU/NVMe transfers.

### Cross-Backend Dependency DAG

```text
  backends/host   ──▶ transfer::host
  backends/cuda   ──▶ backends/host (PinnedHostSlab<'a>), transfer::cuda
  backends/nvme   ──▶ backends/host (staging slab), transfer::{nvme, host::HostContext}
  backends/rdma   ──▶ transfer::{rdma, host::{HostContext, HostBuffer}, cuda::CudaBuffer}
```

Acyclic backend boundaries are mechanically enforced by `no-cuda-dep.yml`, `no-nvme-dep.yml`, and `no-remote-dep.yml`.

### Cross-Backend Frame Composition (`backends/compose/`)

Every backend method ensures the opaque ghost predicate `core_transition(before, after, touched_timeline, allowed_source, mode)` with three modes:

- `PreserveAllInit`: Preserves all pre-existing initializing frames and internal leases (`HostBackend`, CUDA submit/get, NVMe staging, kernel evict/lease/reclaim).
- `PreserveUnparkedInit`: CUDA `poll_completions` / `synchronize` retires only parked GPU writes (`token is Some`), leaving NVMe hops intact.
- `PreserveParkedInit`: NVMe `poll_completions` / `wait_one_completion` retires only unparked hops (`token is None`), leaving parked GPU writes intact.

This lets `LocalEngine<'a>` verify 1-hop and 2-hop multi-tier routes (`GPU ↔ Host ↔ NVMe`) without duplicating backend proofs.
