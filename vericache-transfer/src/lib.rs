//! VeriCache-transfer: verified hardware-FFI and ghost-state layer connecting
//! `vericache-foundation` to Host CPU, CUDA, RDMA (`libibverbs`), NVMe, and
//! cross-node token transport.
//!
//! Modules, in dependency order:
//!
//! - `domain`: concrete `TransferDomain`, `SlotLoc`, `SlotRef`, and zero-`forall` `Effect`s.
//! - `context`: opaque `TransferInstance`, `NodeContext::bootstrap`, single-key
//!   slot/share tokens, and lock-free token supplies.
//! - `host`: host memory allocation (single-slot and slab) and synchronous CPU accesses.
//! - `cuda`: CUDA device/pinned memory (single-slot and slab), non-blocking streams, and events.
//! - `nvme`: batched direct file I/O through Linux io_uring (single-slot and slab files).
//! - `rdma`: `libibverbs` memory regions, Type-2 memory windows, CQs, and RC QPs.
//! - `transport`: decoupled `ByteTransport` trait and cross-node `Channel<T>`.
pub mod context;
pub mod cuda;
pub mod domain;
pub mod host;
pub mod nvme;
pub mod rdma;
pub mod transport;
