//! Runtime resources used by the facade to call the verified device backend.

use transfer::context::SlotToken;
use transfer::domain::DeviceId;
use transfer::rdma::{AccessFlags, ProtectionDomain};
use transfer::transport::ByteTransport;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::backends::{DeviceBackend, DeviceSlice, NoDevice};
use vericache_core::engine::DistributedEngine;
use vericache_core::kernel::CacheCore;
use vericache_core::provenance::ValueCertificate;
use vericache_core::spec::{CoreError, FrameId, TimelineId};
use vstd::prelude::{Ghost, Tracked};

use crate::CacheError;
use crate::resources::{GpuConfig, ShardCaps};

/// Borrow of a caller slot and its capabilities for verified device reads and writes.
pub struct DeviceSlotParts<'storage, 'slot, D: DeviceBackend<'storage>> {
    pub slice: D::Slice<'slot>,
    pub token: &'slot mut Tracked<SlotToken>,
    pub certificate: Option<Ghost<ValueCertificate>>,
}

/// Adapts storage initialization and caller-owned DMA resources to a verified backend.
pub trait DeviceRuntime: 'static {
    type Backend<'a>: DeviceBackend<'a> + 'a;
    type Slot;
    type Stream;
    type Staging;

    fn new_backend<'a>(
        core: &mut CacheCore,
        caps: &mut ShardCaps,
        config: &GpuConfig,
    ) -> Result<Self::Backend<'a>, CacheError>;

    fn attach_pinned_host<'a>(
        backend: &mut Self::Backend<'a>,
        core: &CacheCore,
        host: &'a HostBackend,
    ) -> Result<(), CacheError>;

    /// Shorten a backend's storage lifetime to the lifetime of a cache session.
    fn borrow_backend<'a, 'storage: 'a>(
        backend: &'a Self::Backend<'storage>,
    ) -> &'a Self::Backend<'a>;

    /// Create a stream and register its timeline with the engine.
    fn new_stream<'a, T: ByteTransport>(
        engine: &mut DistributedEngine<'a, T, Self::Backend<'a>>,
        caps: &mut ShardCaps,
    ) -> Result<Self::Stream, CacheError>;

    fn new_staging(caps: &mut ShardCaps, frame_size: usize) -> Result<Self::Staging, CacheError>;

    fn alloc_slot(
        caps: &mut ShardCaps,
        device: DeviceId,
        frame_size: usize,
    ) -> Result<Self::Slot, CacheError>;

    fn slot_len(slot: &Self::Slot) -> usize;

    fn upload_slot<'a, T: ByteTransport>(
        engine: &mut DistributedEngine<'a, T, Self::Backend<'a>>,
        caps: &mut ShardCaps,
        staging: &mut Self::Staging,
        stream: &mut Self::Stream,
        slot: &mut Self::Slot,
        data: &[u8],
    ) -> Result<(), CacheError>;

    fn download_slot<'a, T: ByteTransport>(
        engine: &mut DistributedEngine<'a, T, Self::Backend<'a>>,
        caps: &mut ShardCaps,
        staging: &mut Self::Staging,
        stream: &mut Self::Stream,
        slot: &mut Self::Slot,
        dst: &mut [u8],
    ) -> Result<(), CacheError>;

    /// Borrow the stream and timeline for an engine operation or synchronization.
    fn stream_parts<'a, 'stream>(
        stream: &'stream mut Self::Stream,
    ) -> Result<
        (
            &'stream mut <Self::Backend<'a> as DeviceBackend<'a>>::Stream,
            TimelineId,
        ),
        CacheError,
    >;

    /// Borrow the slice, token, and provenance for engine reads and writes.
    /// Asynchronous writes require an owned device attachment in the engine.
    fn slot_parts<'a, 'slot>(
        slot: &'slot mut Self::Slot,
    ) -> Result<DeviceSlotParts<'a, 'slot, Self::Backend<'a>>, CacheError>;

    /// Record provenance after a caller finishes writing to a slot from an external stream.
    fn mark_external_write(slot: &mut Self::Slot) -> Result<(), CacheError>;

    fn register_device_slab<'a>(
        backend: &'a Self::Backend<'a>,
        rdma: &mut RdmaBackend<'a>,
        core: &CacheCore,
        pd: &'a ProtectionDomain<'a>,
        flags: AccessFlags,
    ) -> Result<(), CacheError> {
        rdma.register_gpu_slab(
            core,
            pd,
            backend.device_buffer(),
            backend.device_tier_id(),
            backend.device_first_frame(),
            backend.device_num_frames(),
            flags,
        )
        .map_err(CacheError::Core)
    }

    fn device_frame_addr<'a, T: ByteTransport>(
        engine: &DistributedEngine<'a, T, Self::Backend<'a>>,
        frame: FrameId,
    ) -> Option<usize> {
        engine
            .device_frame_slice(frame)
            .map(|slice| slice.device_addr())
    }
}

/// Runtime for host, NVMe, and host RDMA without a device tier.
pub struct CpuOnlyRuntime;

pub struct NoDeviceSlot;
pub struct NoDeviceStream;

impl DeviceRuntime for CpuOnlyRuntime {
    type Backend<'a> = NoDevice;
    type Slot = NoDeviceSlot;
    type Stream = NoDeviceStream;
    type Staging = ();

    fn new_backend<'a>(
        _core: &mut CacheCore,
        _caps: &mut ShardCaps,
        _config: &GpuConfig,
    ) -> Result<Self::Backend<'a>, CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn attach_pinned_host<'a>(
        _backend: &mut NoDevice,
        _core: &CacheCore,
        _host: &'a HostBackend,
    ) -> Result<(), CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn borrow_backend<'a, 'storage: 'a>(backend: &'a NoDevice) -> &'a NoDevice {
        backend
    }

    fn new_stream<'a, T: ByteTransport>(
        _engine: &mut DistributedEngine<'a, T, NoDevice>,
        _caps: &mut ShardCaps,
    ) -> Result<NoDeviceStream, CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn new_staging(_caps: &mut ShardCaps, _frame_size: usize) -> Result<(), CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn alloc_slot(
        _caps: &mut ShardCaps,
        _device: DeviceId,
        _frame_size: usize,
    ) -> Result<NoDeviceSlot, CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn slot_len(_slot: &NoDeviceSlot) -> usize {
        0
    }

    fn upload_slot<'a, T: ByteTransport>(
        _engine: &mut DistributedEngine<'a, T, NoDevice>,
        _caps: &mut ShardCaps,
        _staging: &mut (),
        _stream: &mut NoDeviceStream,
        _slot: &mut NoDeviceSlot,
        _data: &[u8],
    ) -> Result<(), CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn download_slot<'a, T: ByteTransport>(
        _engine: &mut DistributedEngine<'a, T, NoDevice>,
        _caps: &mut ShardCaps,
        _staging: &mut (),
        _stream: &mut NoDeviceStream,
        _slot: &mut NoDeviceSlot,
        _dst: &mut [u8],
    ) -> Result<(), CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn stream_parts<'a, 'stream>(
        _stream: &'stream mut NoDeviceStream,
    ) -> Result<
        (
            &'stream mut <NoDevice as DeviceBackend<'a>>::Stream,
            TimelineId,
        ),
        CacheError,
    > {
        Err(CoreError::UnsupportedTier.into())
    }

    fn slot_parts<'a, 'slot>(
        _slot: &'slot mut NoDeviceSlot,
    ) -> Result<DeviceSlotParts<'a, 'slot, NoDevice>, CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }

    fn mark_external_write(_slot: &mut NoDeviceSlot) -> Result<(), CacheError> {
        Err(CoreError::UnsupportedTier.into())
    }
}
