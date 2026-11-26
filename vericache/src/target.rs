//! Unified polymorphic `ReadTarget` and `WriteSource` traits enabling a single
//! location-transparent `cache.get(key, dst)` and `cache.put(key, src)` entry
//! point with strong, domain-accurate return types (zero enum matching).

use transfer::transport::ByteTransport;
use vericache_core::spec::{ObjectKey, VersionId};

use crate::cache::VeriCache;
use crate::device::DeviceRuntime;
use crate::policy::{EvictionPolicy, PeerRoutingPolicy, TieringPolicy};
use crate::{CacheError, DefaultDeviceRuntime, GpuReadHit, ReadHit};

/// Destination buffer passed to [`VeriCache::get`].
pub trait ReadTarget<RT: DeviceRuntime = DefaultDeviceRuntime> {
    type Output;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<Option<Self::Output>, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy;
}

/// Source payload passed to [`VeriCache::put`].
pub trait WriteSource<RT: DeviceRuntime = DefaultDeviceRuntime> {
    type Output;

    fn store<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<Self::Output, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy;
}

/// Device destination descriptor pairing a caller stream and slot.
pub struct DeviceDst<'a, RT: DeviceRuntime> {
    pub stream: &'a mut RT::Stream,
    pub slot: &'a mut RT::Slot,
}

impl<'a, RT: DeviceRuntime> DeviceDst<'a, RT> {
    pub fn new(stream: &'a mut RT::Stream, slot: &'a mut RT::Slot) -> Self {
        Self { stream, slot }
    }
}

/// Device source descriptor pairing a caller stream and slot.
pub struct DeviceSrc<'a, RT: DeviceRuntime> {
    pub stream: &'a mut RT::Stream,
    pub slot: &'a mut RT::Slot,
}

impl<'a, RT: DeviceRuntime> DeviceSrc<'a, RT> {
    pub fn new(stream: &'a mut RT::Stream, slot: &'a mut RT::Slot) -> Self {
        Self { stream, slot }
    }
}

/// Destination descriptor for the default GPU runtime.
pub type GpuDst<'a> = DeviceDst<'a, DefaultDeviceRuntime>;

/// Source descriptor for the default GPU runtime.
pub type GpuSrc<'a> = DeviceSrc<'a, DefaultDeviceRuntime>;

impl<RT: DeviceRuntime> ReadTarget<RT> for &mut [u8] {
    type Output = ReadHit;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<Option<ReadHit>, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy,
    {
        cache.get_into_host(key, self)
    }
}

impl<RT: DeviceRuntime> ReadTarget<RT> for &mut Vec<u8> {
    type Output = ReadHit;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<Option<ReadHit>, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy,
    {
        cache.get_into_host(key, self.as_mut_slice())
    }
}

impl<RT: DeviceRuntime> ReadTarget<RT> for DeviceDst<'_, RT> {
    type Output = GpuReadHit;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<Option<GpuReadHit>, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy,
    {
        cache.get_into_gpu(key, self.stream, self.slot)
    }
}

impl<RT: DeviceRuntime> WriteSource<RT> for &[u8] {
    type Output = VersionId;

    fn store<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<VersionId, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy,
    {
        cache.put_host(key, self)
    }
}

impl<RT: DeviceRuntime> WriteSource<RT> for DeviceSrc<'_, RT> {
    type Output = VersionId;

    fn store<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R, RT>,
        key: ObjectKey,
    ) -> Result<VersionId, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy,
    {
        cache.put_gpu(key, self.stream, self.slot)
    }
}
