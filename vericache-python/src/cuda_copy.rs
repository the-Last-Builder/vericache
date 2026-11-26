//! Minimal CUDA runtime async DMA & pinned host staging helpers bridging
//! external GPU tensor pointers (PyTorch / vLLM paged KV caches) and VeriCache
//! managed GPU HBM / CUDA-pinned Host DRAM frames.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::device_copy::{
    BlockingEvent, CopyBatch, CopyDirection, ExternalDeviceCopy, PinnedHostBuffer,
};

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

const CUDA_EVENT_BLOCKING_SYNC: c_uint = 0x01;
const CUDA_EVENT_DISABLE_TIMING: c_uint = 0x02;

const CUDA_MEM_LOCATION_TYPE_NONE: c_int = 0;
const CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM: c_int = 1;
const CUDA_MEMCPY_SRC_ACCESS_ORDER_ANY: c_int = 3;
const CUDA_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE: c_uint = 0x1;
const CUDA_ERROR_NOT_SUPPORTED: c_int = 801;

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaMemLocation {
    kind: c_int,
    id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaMemcpyAttributes {
    src_access_order: c_int,
    src_loc_hint: CudaMemLocation,
    dst_loc_hint: CudaMemLocation,
    flags: c_uint,
}

type CudaMemcpyBatchAsyncFn = unsafe extern "C" fn(
    dsts: *const *mut c_void,
    srcs: *const *const c_void,
    sizes: *const usize,
    count: usize,
    attrs: *mut CudaMemcpyAttributes,
    attrs_idxs: *mut usize,
    num_attrs: usize,
    stream: *mut c_void,
) -> c_int;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;

    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaEventSynchronize(event: *mut c_void) -> c_int;

    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;

    fn cudaRuntimeGetVersion(runtime_version: *mut c_int) -> c_int;
    fn cudaDriverGetVersion(driver_version: *mut c_int) -> c_int;
    fn cudaSetDevice(device: c_int) -> c_int;
}

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

static BATCH_COPY_FN: OnceLock<Option<CudaMemcpyBatchAsyncFn>> = OnceLock::new();
static BATCH_COPY_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

fn resolve_batch_copy_fn() -> Option<CudaMemcpyBatchAsyncFn> {
    *BATCH_COPY_FN.get_or_init(|| {
        let mut rt_ver: c_int = 0;
        let mut drv_ver: c_int = 0;
        let rc_rt = unsafe { cudaRuntimeGetVersion(&mut rt_ver) };
        let rc_drv = unsafe { cudaDriverGetVersion(&mut drv_ver) };
        // CUDA 13.0+ uses the 8-argument `cudaMemcpyBatchAsync` ABI (removed `failIdx`).
        // Require both runtime and host driver >= 13.0 before invoking it.
        if rc_rt != 0 || rc_drv != 0 || rt_ver < 13_000 || drv_ver < 13_000 {
            return None;
        }
        // On Linux/glibc, `RTLD_DEFAULT` is NULL.
        let sym = unsafe { dlsym(core::ptr::null_mut(), c"cudaMemcpyBatchAsync".as_ptr()) };
        if sym.is_null() {
            None
        } else {
            Some(unsafe { core::mem::transmute::<*mut c_void, CudaMemcpyBatchAsyncFn>(sym) })
        }
    })
}

/// Reusable blocking-sync CUDA event so waiting on async DMA yields the CPU
/// instead of spinning at 100% CPU on the worker thread.
pub(crate) struct BlockingCudaEvent {
    ev: *mut c_void,
}

impl BlockingEvent for BlockingCudaEvent {
    fn new() -> Result<Self, i32> {
        let mut ev: *mut c_void = core::ptr::null_mut();
        let rc = unsafe {
            cudaEventCreateWithFlags(
                &mut ev,
                CUDA_EVENT_BLOCKING_SYNC | CUDA_EVENT_DISABLE_TIMING,
            )
        };
        if rc != 0 || ev.is_null() {
            return Err(if rc != 0 { rc } else { -1 });
        }
        Ok(Self { ev })
    }

    fn record(&self, cuda_stream_ptr: usize) -> Result<(), i32> {
        let stream = cuda_stream_ptr as *mut c_void;
        let rc = unsafe { cudaEventRecord(self.ev, stream) };
        if rc != 0 { Err(rc) } else { Ok(()) }
    }

    fn synchronize(&self) -> Result<(), i32> {
        let rc = unsafe { cudaEventSynchronize(self.ev) };
        if rc != 0 { Err(rc) } else { Ok(()) }
    }

    fn record_and_sync(&self, cuda_stream_ptr: usize) -> Result<(), i32> {
        let stream = cuda_stream_ptr as *mut c_void;
        let rc = unsafe { cudaEventRecord(self.ev, stream) };
        if rc != 0 {
            return Err(rc);
        }
        let rc2 = unsafe { cudaEventSynchronize(self.ev) };
        if rc2 != 0 {
            return Err(rc2);
        }
        Ok(())
    }
}

impl Drop for BlockingCudaEvent {
    fn drop(&mut self) {
        if !self.ev.is_null() {
            unsafe {
                let _ = cudaEventDestroy(self.ev);
            }
            self.ev = core::ptr::null_mut();
        }
    }
}

/// Reusable CUDA page-locked host staging buffer for batched D2H DMA.
pub(crate) struct PinnedHostBuf {
    ptr: *mut u8,
    cap: usize,
}

impl PinnedHostBuffer for PinnedHostBuf {
    fn new() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            cap: 0,
        }
    }

    fn ensure_capacity(&mut self, bytes: usize) -> Result<(), i32> {
        if bytes <= self.cap {
            return Ok(());
        }
        self.free();
        let mut raw: *mut c_void = core::ptr::null_mut();
        let rc = unsafe { cudaHostAlloc(&mut raw, bytes, 0) };
        if rc != 0 || raw.is_null() {
            return Err(if rc != 0 { rc } else { -1 });
        }
        self.ptr = raw as *mut u8;
        self.cap = bytes;
        Ok(())
    }

    fn addr(&self) -> usize {
        self.ptr as usize
    }

    fn as_slice(&self, len: usize) -> &[u8] {
        debug_assert!(len <= self.cap);
        unsafe { core::slice::from_raw_parts(self.ptr, len) }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        debug_assert!(len <= self.cap);
        unsafe { core::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl PinnedHostBuf {
    fn free(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cudaFreeHost(self.ptr as *mut c_void);
            }
            self.ptr = core::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        self.free();
    }
}

/// Reusable descriptor arrays for submitting batched hardware Copy Engine DMA
/// transfers via CUDA 13.0 `cudaMemcpyBatchAsync` (with automatic fallback to
/// `cudaMemcpyAsync` for single copies, legacy stream 0, or pre-13.0 runtimes).
pub(crate) struct BatchCopyBuf {
    dsts: Vec<*mut c_void>,
    srcs: Vec<*const c_void>,
    sizes: Vec<usize>,
    kinds: Vec<c_int>,
    /// True iff every queued copy is `CUDA_MEMCPY_HOST_TO_DEVICE` (read-only leased
    /// pinned host frames with no prior stream writers, allowing `srcAccessOrder = Any`).
    all_h2d: bool,
}

impl CopyBatch for BatchCopyBuf {
    fn new() -> Self {
        Self {
            dsts: Vec::new(),
            srcs: Vec::new(),
            sizes: Vec::new(),
            kinds: Vec::new(),
            all_h2d: true,
        }
    }

    fn clear(&mut self) {
        self.dsts.clear();
        self.srcs.clear();
        self.sizes.clear();
        self.kinds.clear();
        self.all_h2d = true;
    }

    fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    fn push(
        &mut self,
        dst_addr: usize,
        src_addr: usize,
        len: usize,
        direction: CopyDirection,
    ) -> Result<(), i32> {
        if len == 0 {
            return Ok(());
        }
        if dst_addr == 0 || src_addr == 0 {
            return Err(-1);
        }
        self.dsts.push(dst_addr as *mut c_void);
        self.srcs.push(src_addr as *const c_void);
        self.sizes.push(len);
        let kind = cuda_memcpy_kind(direction);
        self.kinds.push(kind);
        if kind != CUDA_MEMCPY_HOST_TO_DEVICE {
            self.all_h2d = false;
        }
        Ok(())
    }
}

#[inline]
fn cuda_memcpy_kind(direction: CopyDirection) -> c_int {
    match direction {
        CopyDirection::HostToDevice => CUDA_MEMCPY_HOST_TO_DEVICE,
        CopyDirection::DeviceToHost => CUDA_MEMCPY_DEVICE_TO_HOST,
        CopyDirection::DeviceToDevice => CUDA_MEMCPY_DEVICE_TO_DEVICE,
    }
}

pub(crate) struct CudaDeviceCopy;

impl ExternalDeviceCopy for CudaDeviceCopy {
    type Batch = BatchCopyBuf;
    type Event = BlockingCudaEvent;
    type Staging = PinnedHostBuf;

    fn set_device(device_id: i32) -> Result<(), i32> {
        let rc = unsafe { cudaSetDevice(device_id as c_int) };
        if rc != 0 { Err(rc) } else { Ok(()) }
    }

    fn memcpy_async(
        dst_addr: usize,
        src_addr: usize,
        len: usize,
        direction: CopyDirection,
        cuda_stream_ptr: usize,
    ) -> Result<(), i32> {
        if len == 0 {
            return Ok(());
        }
        if dst_addr == 0 || src_addr == 0 {
            return Err(-1);
        }
        let stream = cuda_stream_ptr as *mut c_void;
        let rc = unsafe {
            cudaMemcpyAsync(
                dst_addr as *mut c_void,
                src_addr as *const c_void,
                len,
                cuda_memcpy_kind(direction),
                stream,
            )
        };
        if rc != 0 {
            return Err(rc);
        }
        Ok(())
    }

    fn memcpy_batch_async(
        batch: &BatchCopyBuf,
        cuda_stream_ptr: usize,
    ) -> Result<bool, (i32, bool)> {
        let count = batch.sizes.len();
        if count == 0 {
            return Ok(false);
        }
        let stream = cuda_stream_ptr as *mut c_void;

        // CUDA 13.0 `cudaMemcpyBatchAsync` requires a non-NULL stream and amortizes
        // launch overhead when submitting multiple copies in one driver call.
        if count > 1 && cuda_stream_ptr != 0 && !BATCH_COPY_UNSUPPORTED.load(Ordering::Relaxed) {
            if let Some(batch_fn) = resolve_batch_copy_fn() {
                // Pure H2D LOAD from verified read-leased PinnedHost frames has no prior
                // GPU writers on `stream`, so `Any` lets CUDA 13.0 relax source ordering.
                // SAVE (D2H) and any batch containing GPU HBM/GDR sources (D2D) MUST stay
                // strictly `Stream`-ordered.
                let src_access_order = if batch.all_h2d {
                    CUDA_MEMCPY_SRC_ACCESS_ORDER_ANY
                } else {
                    CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM
                };
                let mut attr = CudaMemcpyAttributes {
                    src_access_order,
                    src_loc_hint: CudaMemLocation {
                        kind: CUDA_MEM_LOCATION_TYPE_NONE,
                        id: 0,
                    },
                    dst_loc_hint: CudaMemLocation {
                        kind: CUDA_MEM_LOCATION_TYPE_NONE,
                        id: 0,
                    },
                    flags: CUDA_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE,
                };
                let mut attr_idx: usize = 0;
                let rc = unsafe {
                    batch_fn(
                        batch.dsts.as_ptr(),
                        batch.srcs.as_ptr(),
                        batch.sizes.as_ptr(),
                        count,
                        &mut attr,
                        &mut attr_idx,
                        1,
                        stream,
                    )
                };
                if rc == 0 {
                    return Ok(true);
                }
                if rc == CUDA_ERROR_NOT_SUPPORTED {
                    BATCH_COPY_UNSUPPORTED.store(true, Ordering::Relaxed);
                } else {
                    return Err((rc, true));
                }
            }
        }

        let mut submitted = false;
        for i in 0..count {
            let rc = unsafe {
                cudaMemcpyAsync(
                    batch.dsts[i],
                    batch.srcs[i],
                    batch.sizes[i],
                    batch.kinds[i],
                    stream,
                )
            };
            if rc != 0 {
                return Err((rc, submitted));
            }
            submitted = true;
        }
        Ok(submitted)
    }

    fn stream_synchronize(cuda_stream_ptr: usize) -> Result<(), i32> {
        let stream = cuda_stream_ptr as *mut c_void;
        let rc = unsafe { cudaStreamSynchronize(stream) };
        if rc != 0 {
            return Err(rc);
        }
        Ok(())
    }
}
