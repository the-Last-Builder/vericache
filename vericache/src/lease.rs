//! Zero-copy RAII read lease guards over verified `HostLease` and `DeviceLease`.
//!
//! Dropping a guard automatically returns its `ShareCap` to the verified
//! `DistributedEngine`, ensuring frames cannot be prematurely reclaimed while
//! leased and cannot leak leases on early return.

use transfer::host::HostSlice;
use transfer::transport::ByteTransport;
use vericache_core::backends::{DeviceBackend, DeviceLease, DeviceSlice};
use vericache_core::engine::HostLease;
use vericache_core::spec::{FrameId, SyncPoint, Ticket, VersionId};

use crate::DefaultDeviceRuntime;
use crate::cache::VeriCache;
use crate::device::DeviceRuntime;
use crate::policy::{EvictionPolicy, HitLocation, PeerRoutingPolicy, TieringPolicy};

/// Zero-copy read-only lease on a published Host DRAM frame.
pub struct HostLeaseGuard<'cache, 'storage, T, E, P, R, RT: DeviceRuntime = DefaultDeviceRuntime>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    cache: &'cache mut VeriCache<'storage, T, E, P, R, RT>,
    lease: Option<HostLease<'storage>>,
    location: HitLocation,
}

impl<'cache, 'storage, T, E, P, R, RT: DeviceRuntime>
    HostLeaseGuard<'cache, 'storage, T, E, P, R, RT>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    pub(crate) fn new(
        cache: &'cache mut VeriCache<'storage, T, E, P, R, RT>,
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
        self.lease
            .as_ref()
            .expect("active host lease")
            .handle
            .frame()
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

impl<'cache, 'storage, T, E, P, R, RT: DeviceRuntime> Drop
    for HostLeaseGuard<'cache, 'storage, T, E, P, R, RT>
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
        }
    }
}

/// Zero-copy read-only lease on a published device frame.
pub struct DeviceLeaseGuard<'cache, 'storage, T, E, P, R, RT: DeviceRuntime = DefaultDeviceRuntime>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    cache: &'cache mut VeriCache<'storage, T, E, P, R, RT>,
    lease: Option<DeviceLease>,
    location: HitLocation,
    release_ticket: Ticket,
}

impl<'cache, 'storage, T, E, P, R, RT: DeviceRuntime>
    DeviceLeaseGuard<'cache, 'storage, T, E, P, R, RT>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    pub(crate) fn new(
        cache: &'cache mut VeriCache<'storage, T, E, P, R, RT>,
        lease: DeviceLease,
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
        self.lease.as_ref().expect("active device lease").version
    }

    pub fn frame(&self) -> FrameId {
        self.lease.as_ref().expect("active device lease").frame
    }

    pub fn sync_point(&self) -> SyncPoint {
        self.lease.as_ref().expect("active device lease").sync_point
    }

    pub fn location(&self) -> HitLocation {
        self.location
    }

    /// Borrow the device slice for the lifetime of this lease guard.
    pub fn device_slice(
        &self,
    ) -> Option<<RT::Backend<'storage> as DeviceBackend<'storage>>::Slice<'_>> {
        self.cache.engine().device_frame_slice(self.frame())
    }

    /// Raw device address of the leased frame for external DMA.
    pub fn device_addr(&self) -> Option<usize> {
        self.device_slice().map(|slice| slice.device_addr())
    }

    /// Record a trailing timeline `Ticket` to be handed to Observed Reclamation on drop.
    pub fn set_release_ticket(&mut self, ticket: Ticket) {
        self.release_ticket = ticket;
    }
}

impl<'cache, 'storage, T, E, P, R, RT: DeviceRuntime> Drop
    for DeviceLeaseGuard<'cache, 'storage, T, E, P, R, RT>
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
        }
    }
}

/// Compatibility alias for a device lease guard, defaulting to the CUDA runtime.
pub type GpuLeaseGuard<'cache, 'storage, T, E, P, R, RT = DefaultDeviceRuntime> =
    DeviceLeaseGuard<'cache, 'storage, T, E, P, R, RT>;
