//! Zero-copy RAII read lease guards over verified `HostLease` and `GpuLease`.
//!
//! Dropping a guard automatically returns its `ShareCap` to the verified
//! `DistributedEngine`, ensuring frames cannot be prematurely reclaimed while
//! leased and cannot leak leases on early return.

use transfer::cuda::CudaSlice;
use transfer::host::HostSlice;
use transfer::transport::ByteTransport;
use vericache_core::engine::{GpuLease, HostLease};
use vericache_core::spec::{FrameId, SyncPoint, Ticket, VersionId};

use crate::cache::VeriCache;
use crate::policy::{EvictionPolicy, HitLocation, PeerRoutingPolicy, TieringPolicy};

/// Zero-copy read-only lease on a published Host DRAM frame.
pub struct HostLeaseGuard<'cache, 'storage, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    cache: &'cache mut VeriCache<'storage, T, E, P, R>,
    lease: Option<HostLease<'storage>>,
    location: HitLocation,
}

impl<'cache, 'storage, T, E, P, R> HostLeaseGuard<'cache, 'storage, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    pub(crate) fn new(
        cache: &'cache mut VeriCache<'storage, T, E, P, R>,
        lease: HostLease<'storage>,
        location: HitLocation,
    ) -> Self {
        Self {
            cache,
            lease: Some(lease),
            location,
        }
    }

    pub fn version(&self) -> VersionId {
        self.lease.as_ref().expect("active host lease").version
    }

    pub fn frame(&self) -> FrameId {
        self.lease.as_ref().expect("active host lease").handle.frame
    }

    pub fn location(&self) -> HitLocation {
        self.location
    }

    pub fn slice(&self) -> &HostSlice<'storage> {
        &self.lease.as_ref().expect("active host lease").slice
    }

    /// Synchronously copy the leased frame's bytes into `dst` using the cache's
    /// verified `HostBackend` read path.
    pub fn copy_into(&mut self, dst: &mut [u8]) -> Result<(), crate::CacheError> {
        let frame = self.frame();
        self.cache.read_host_frame_raw(frame, dst)?;
        Ok(())
    }
}

impl<'cache, 'storage, T, E, P, R> Drop for HostLeaseGuard<'cache, 'storage, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            let _ = self
                .cache
                .release_local_lease_raw(lease.handle, lease.cap, Ticket::NONE);
            self.cache.reclaim_transient_remote_frames([]);
        }
    }
}

/// Zero-copy read-only lease on a published GPU VRAM frame.
pub struct GpuLeaseGuard<'cache, 'storage, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    cache: &'cache mut VeriCache<'storage, T, E, P, R>,
    lease: Option<GpuLease>,
    location: HitLocation,
    release_ticket: Ticket,
}

impl<'cache, 'storage, T, E, P, R> GpuLeaseGuard<'cache, 'storage, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    pub(crate) fn new(
        cache: &'cache mut VeriCache<'storage, T, E, P, R>,
        lease: GpuLease,
        location: HitLocation,
    ) -> Self {
        Self {
            cache,
            lease: Some(lease),
            location,
            release_ticket: Ticket::NONE,
        }
    }

    pub fn version(&self) -> VersionId {
        self.lease.as_ref().expect("active gpu lease").version
    }

    pub fn frame(&self) -> FrameId {
        self.lease.as_ref().expect("active gpu lease").frame
    }

    pub fn sync_point(&self) -> SyncPoint {
        self.lease.as_ref().expect("active gpu lease").sync_point
    }

    pub fn location(&self) -> HitLocation {
        self.location
    }

    /// Borrow the underlying `CudaSlice` for the leased GPU frame.
    pub fn cuda_slice(&self) -> Option<CudaSlice<'_>> {
        self.cache.engine_gpu_frame_slice(self.frame())
    }

    /// Raw CUDA device address of the leased GPU frame (for zero-copy D2D read
    /// into external PyTorch/vLLM paged KV tensors).
    pub fn device_addr(&self) -> Option<usize> {
        self.cuda_slice().map(|s| s.device_addr())
    }

    /// Record a trailing timeline `Ticket` to be handed to Observed Reclamation on drop.
    pub fn set_release_ticket(&mut self, ticket: Ticket) {
        self.release_ticket = ticket;
    }
}

impl<'cache, 'storage, T, E, P, R> Drop for GpuLeaseGuard<'cache, 'storage, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            let _ =
                self.cache
                    .release_local_lease_raw(lease.handle, lease.cap, self.release_ticket);
            self.cache.reclaim_transient_remote_frames([]);
        }
    }
}
