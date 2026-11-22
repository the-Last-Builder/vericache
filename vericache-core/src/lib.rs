//! VeriCache-core: backend-agnostic KV cache kernel, pluggable hardware
//! backends, and composable local/distributed engines (`LocalEngine`, `DistributedEngine`).
//!
//! Modules, in dependency order:
//!
//! - `spec`: open `TierId`, `VersionId`, `Ticket`, `SyncPoint`, and `CoreError`.
//! - `provenance`: `KnowledgeLedger` and inductive `ValueCertificate` chains.
//! - `timeline`: open `TimelineTable` and O(1) `SlidingBitmap` prefix completion.
//! - `frame`: 64-byte `FrameSyncMeta` and single-key `SlotToken`/`ShareCap` table.
//! - `catalog`: open-addressing hash directory and caller-prioritized replica lookup.
//! - `kernel`: hardware-free `CacheCore` facade unifying tiers, frames, leases, and catalog.
//! - `backends`: pluggable `host`, `cuda`, `nvme`, `rdma` backends and `compose` summaries.
//! - `engine`: composable `LocalEngine<'a>`, `DistributedEngine<'a, T>`, `HostLease<'a>`, and `GpuLease`.
pub mod backends;
pub mod catalog;
pub mod engine;
pub mod frame;
pub mod kernel;
pub mod provenance;
pub mod spec;
pub mod timeline;
