//! Unified polymorphic `ReadTarget` and `WriteSource` traits enabling a single
//! location-transparent `cache.get(key, dst)` and `cache.put(key, src)` entry
//! point with strong, domain-accurate return types (zero enum matching).

use transfer::transport::ByteTransport;
use vericache_core::spec::{ObjectKey, VersionId};

use crate::cache::VeriCache;
use crate::policy::{EvictionPolicy, PeerRoutingPolicy, TieringPolicy};
use crate::resources::{GpuSlot, GpuStream};
use crate::{CacheError, GpuReadHit, ReadHit};

/// Destination buffer passed to [`VeriCache::get`].
pub trait ReadTarget {
    type Output;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
        key: ObjectKey,
    ) -> Result<Option<Self::Output>, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy;
}

/// Source payload passed to [`VeriCache::put`].
pub trait WriteSource {
    type Output;

    fn store<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
        key: ObjectKey,
    ) -> Result<Self::Output, CacheError>
    where
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy;
}

/// GPU destination descriptor pairing a caller [`GpuStream`] and managed [`GpuSlot`].
pub struct GpuDst<'a> {
    pub stream: &'a mut GpuStream,
    pub slot: &'a mut GpuSlot,
}

impl<'a> GpuDst<'a> {
    pub fn new(stream: &'a mut GpuStream, slot: &'a mut GpuSlot) -> Self {
        Self { stream, slot }
    }
}

/// GPU source descriptor pairing a caller [`GpuStream`] and managed [`GpuSlot`].
pub struct GpuSrc<'a> {
    pub stream: &'a mut GpuStream,
    pub slot: &'a mut GpuSlot,
}

impl<'a> GpuSrc<'a> {
    pub fn new(stream: &'a mut GpuStream, slot: &'a mut GpuSlot) -> Self {
        Self { stream, slot }
    }
}

impl ReadTarget for &mut [u8] {
    type Output = ReadHit;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
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

impl ReadTarget for &mut Vec<u8> {
    type Output = ReadHit;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
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

impl ReadTarget for GpuDst<'_> {
    type Output = GpuReadHit;

    fn load<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
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

impl WriteSource for &[u8] {
    type Output = VersionId;

    fn store<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
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

impl WriteSource for GpuSrc<'_> {
    type Output = VersionId;

    fn store<T, E, P, R>(
        self,
        cache: &mut VeriCache<'_, T, E, P, R>,
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
