//! DMA between externally supplied addresses and cache buffers.

#[derive(Clone, Copy)]
pub(crate) enum CopyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

pub(crate) trait CopyBatch {
    fn new() -> Self;
    fn clear(&mut self);
    fn is_empty(&self) -> bool;
    fn push(
        &mut self,
        dst: usize,
        src: usize,
        len: usize,
        direction: CopyDirection,
    ) -> Result<(), i32>;
}

pub(crate) trait BlockingEvent: Sized {
    fn new() -> Result<Self, i32>;
    fn record(&self, stream: usize) -> Result<(), i32>;
    fn synchronize(&self) -> Result<(), i32>;

    fn record_and_sync(&self, stream: usize) -> Result<(), i32>;
}

/// Reusable staging memory whose allocation is released on drop.
pub(crate) trait PinnedHostBuffer {
    fn new() -> Self;
    fn ensure_capacity(&mut self, bytes: usize) -> Result<(), i32>;
    fn addr(&self) -> usize;
    // Borrow only initialized bytes after the stream has completed its writes.
    fn as_slice(&self, len: usize) -> &[u8];
    fn as_mut_slice(&mut self, len: usize) -> &mut [u8];
}

pub(crate) trait ExternalDeviceCopy {
    type Batch: CopyBatch;
    type Event: BlockingEvent;
    type Staging: PinnedHostBuffer;

    fn set_device(device_id: i32) -> Result<(), i32>;

    fn memcpy_async(
        dst: usize,
        src: usize,
        len: usize,
        direction: CopyDirection,
        stream: usize,
    ) -> Result<(), i32>;

    /// Return whether any copy was enqueued. An error's boolean indicates
    /// whether the stream must be drained before releasing buffers or leases.
    fn memcpy_batch_async(batch: &Self::Batch, stream: usize) -> Result<bool, (i32, bool)>;

    fn stream_synchronize(stream: usize) -> Result<(), i32>;

    fn memcpy_d2d_sync(dst: usize, src: usize, len: usize, stream: usize) -> Result<(), i32> {
        Self::memcpy_async(dst, src, len, CopyDirection::DeviceToDevice, stream)?;
        Self::stream_synchronize(stream)
    }
}

#[cfg(feature = "cuda")]
pub(crate) type DeviceCopyBackend = crate::cuda_copy::CudaDeviceCopy;
#[cfg(not(feature = "cuda"))]
pub(crate) type DeviceCopyBackend = cpu_only::CpuOnlyDeviceCopy;
pub(crate) type BatchCopyBuf = <DeviceCopyBackend as ExternalDeviceCopy>::Batch;
pub(crate) type BlockingEventHandle = <DeviceCopyBackend as ExternalDeviceCopy>::Event;
pub(crate) type PinnedHostBuf = <DeviceCopyBackend as ExternalDeviceCopy>::Staging;

#[cfg(not(feature = "cuda"))]
mod cpu_only {
    use super::*;

    pub(crate) struct CpuOnlyDeviceCopy;
    pub(crate) struct NoDeviceBatch;
    pub(crate) struct NoDeviceEvent;
    pub(crate) struct NoPinnedHostBuffer;

    impl ExternalDeviceCopy for CpuOnlyDeviceCopy {
        type Batch = NoDeviceBatch;
        type Event = NoDeviceEvent;
        type Staging = NoPinnedHostBuffer;

        fn set_device(_device_id: i32) -> Result<(), i32> {
            Err(-1)
        }

        fn memcpy_async(
            _dst: usize,
            _src: usize,
            _len: usize,
            _direction: CopyDirection,
            _stream: usize,
        ) -> Result<(), i32> {
            Err(-1)
        }

        fn memcpy_batch_async(_batch: &NoDeviceBatch, _stream: usize) -> Result<bool, (i32, bool)> {
            Err((-1, false))
        }

        fn stream_synchronize(_stream: usize) -> Result<(), i32> {
            Err(-1)
        }
    }

    impl CopyBatch for NoDeviceBatch {
        fn new() -> Self {
            Self
        }

        fn clear(&mut self) {}

        fn is_empty(&self) -> bool {
            true
        }

        fn push(
            &mut self,
            _dst: usize,
            _src: usize,
            _len: usize,
            _direction: CopyDirection,
        ) -> Result<(), i32> {
            Err(-1)
        }
    }

    impl BlockingEvent for NoDeviceEvent {
        fn new() -> Result<Self, i32> {
            Err(-1)
        }

        fn record(&self, _stream: usize) -> Result<(), i32> {
            Err(-1)
        }

        fn synchronize(&self) -> Result<(), i32> {
            Err(-1)
        }

        fn record_and_sync(&self, _stream: usize) -> Result<(), i32> {
            Err(-1)
        }
    }

    impl PinnedHostBuffer for NoPinnedHostBuffer {
        fn new() -> Self {
            Self
        }

        fn ensure_capacity(&mut self, _bytes: usize) -> Result<(), i32> {
            Err(-1)
        }

        fn addr(&self) -> usize {
            0
        }

        fn as_slice(&self, _len: usize) -> &[u8] {
            &[]
        }

        fn as_mut_slice(&mut self, _len: usize) -> &mut [u8] {
            &mut []
        }
    }
}
