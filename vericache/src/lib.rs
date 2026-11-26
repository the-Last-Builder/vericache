#![forbid(unsafe_code)]
//! # `vericache` — Ergonomic, Location-Transparent Distributed KV Cache Facade
//!
//! `vericache` sits above the formally verified `vericache-core`, `vericache-transfer`,
//! and `vericache-foundation` crates with **zero `unsafe`** and **zero `assume_new`**,
//! embodying **Unverified Policy + Verified Enforcement**:
//!
//! - **Location-transparent unified API**: callers invoke `cache.get(key, dst)` and
//!   `cache.put(key, src)` (or explicit `get_into_host`, `get_into_gpu`, `put_host`,
//!   `put_gpu`, `lease_host`, `lease_gpu`) without knowing whether `key` currently
//!   resides in Local GPU HBM, Local Pinned Host DRAM, Local NVMe (`io_uring`), or
//!   any remote RDMA / GPUDirect RDMA peer.
//! - **Composable small policy traits**: [`EvictionPolicy`], [`TieringPolicy`], and
//!   [`PeerRoutingPolicy`] are orthogonal unverified hint providers combined in
//!   [`PolicySet`].
//! - **Verified enforcement**: even if a custom policy suggests out-of-bounds, stale,
//!   wrong-tier, or actively-leased `FrameId`s, the verified `DistributedEngine` and
//!   `CacheCore` enforce bounds, local lease protection, cooperative-first + Option-A
//!   fenced RDMA Type-2B MW revocation, and cluster-wide `VersionBinding` agreement.
//!
//! The cache facade requires the `rdma` feature. Enable `nvme` to attach local
//! NVMe storage and `cuda` to use the CUDA device runtime.

#[cfg(feature = "rdma")]
pub mod cache;
#[cfg(all(feature = "cuda", feature = "rdma"))]
mod cuda_runtime;
#[cfg(feature = "rdma")]
pub mod device;
#[cfg(feature = "rdma")]
pub mod lease;
pub mod policy;
#[cfg(feature = "rdma")]
pub mod resources;
#[cfg(feature = "rdma")]
pub mod target;
pub mod transport;

use std::fmt;
use vericache_core::spec::{CoreError, Ticket, VersionId};

#[cfg(feature = "rdma")]
pub use cache::{BatchReadSource, VeriCache};
#[cfg(all(feature = "cuda", feature = "rdma"))]
pub use cuda_runtime::{CudaRuntime, GpuSlot, GpuStream};
#[cfg(feature = "rdma")]
pub use device::{CpuOnlyRuntime, DeviceRuntime, NoDeviceSlot, NoDeviceStream};
#[cfg(feature = "rdma")]
pub use lease::{DeviceLeaseGuard, GpuLeaseGuard, HostLeaseGuard};
pub use policy::{
    CascadeTiering, ConsistentHashPeers, DataDomain, DefaultPolicy, EvictionPolicy, HitLocation,
    LruEviction, NoEviction, NoTiering, PeerMeta, PeerRoutingPolicy, PolicySet, ProbeAllPeers,
    SpillWatermark, TieringPolicy,
};
#[cfg(feature = "rdma")]
pub use resources::{
    AttachedStorage, GpuConfig, NodeStorage, NvmeConfig, PeerLink, RdmaConfig, StorageConfig,
};
#[cfg(feature = "rdma")]
pub use target::{DeviceDst, DeviceSrc, GpuDst, GpuSrc, ReadTarget, WriteSource};
pub use transport::{NoTransport, TcpTransport};
pub use vericache_core::spec::{FrameId, ObjectKey, TierKind};

/// Runtime used by the facade's default constructors and compatibility types.
#[cfg(all(feature = "cuda", feature = "rdma"))]
pub type DefaultDeviceRuntime = crate::cuda_runtime::CudaRuntime;
/// Runtime used by the facade's default constructors and compatibility types.
#[cfg(all(not(feature = "cuda"), feature = "rdma"))]
pub type DefaultDeviceRuntime = crate::device::CpuOnlyRuntime;

/// Unified error type returned by the cache facade.
#[derive(Debug)]
pub enum CacheError {
    Core(CoreError),
    Init(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::Core(e) => write!(f, "vericache core error: {e:?}"),
            CacheError::Init(msg) => write!(f, "vericache init error: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<CoreError> for CacheError {
    fn from(e: CoreError) -> Self {
        CacheError::Core(e)
    }
}

/// Metadata returned on a synchronous Host read hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadHit {
    pub version: VersionId,
    pub location: HitLocation,
}

/// Metadata returned on an asynchronous GPU stream read hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuReadHit {
    pub version: VersionId,
    pub location: HitLocation,
    pub ticket: Ticket,
}

/// Summary of work performed during a cache poll.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollSummary {
    pub nvme_spills_retired: usize,
    pub peer_cq_processed: usize,
    pub peer_grants_completed: usize,
    pub peer_windows_replenished: usize,
    pub reclaimed_frames: usize,
}

/// Observable per-cache counters (including verified enforcement rejections of buggy policies).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub puts: u64,
    pub gpu_hits: u64,
    pub dram_hits: u64,
    pub nvme_hits: u64,
    pub remote_hits: u64,
    pub misses: u64,
    pub policy_victims_rejected: u64,
    pub policy_victims_pinned: u64,
}
