//! Minimal CUDA runtime D2D copy helper bridging external GPU tensor pointers
//! (e.g., PyTorch / vLLM paged KV cache tensors) and VeriCache managed GPU slots.

use core::ffi::{c_int, c_void};

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

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
}

/// Asynchronously copy `len` bytes from pinned host memory to GPU HBM on
/// `cuda_stream_ptr` (or default stream 0 if null).
pub(crate) fn memcpy_h2d_async(
    dst_gpu: usize,
    src_host: usize,
    len: usize,
    cuda_stream_ptr: usize,
) -> Result<(), i32> {
    if len == 0 {
        return Ok(());
    }
    if dst_gpu == 0 || src_host == 0 {
        return Err(-1);
    }
    let rc = unsafe {
        cudaMemcpyAsync(
            dst_gpu as *mut c_void,
            src_host as *const c_void,
            len,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            cuda_stream_ptr as *mut c_void,
        )
    };
    if rc != 0 { Err(rc) } else { Ok(()) }
}

/// Asynchronously copy `len` bytes from GPU HBM to pinned host memory on
/// `cuda_stream_ptr` (or default stream 0 if null).
pub(crate) fn memcpy_d2h_async(
    dst_host: usize,
    src_gpu: usize,
    len: usize,
    cuda_stream_ptr: usize,
) -> Result<(), i32> {
    if len == 0 {
        return Ok(());
    }
    if dst_host == 0 || src_gpu == 0 {
        return Err(-1);
    }
    let rc = unsafe {
        cudaMemcpyAsync(
            dst_host as *mut c_void,
            src_gpu as *const c_void,
            len,
            CUDA_MEMCPY_DEVICE_TO_HOST,
            cuda_stream_ptr as *mut c_void,
        )
    };
    if rc != 0 { Err(rc) } else { Ok(()) }
}

/// Synchronously copy `len` bytes from `src_addr` to `dst_addr` in GPU HBM on
/// `cuda_stream_ptr` (or default stream 0 if null).
pub(crate) fn memcpy_d2d_sync(
    dst_addr: usize,
    src_addr: usize,
    len: usize,
    cuda_stream_ptr: usize,
) -> Result<(), i32> {
    if len == 0 {
        return Ok(());
    }
    if dst_addr == 0 || src_addr == 0 {
        return Err(-1);
    }
    let stream = cuda_stream_ptr as *mut c_void;
    // SAFETY: Caller provides valid GPU device pointers of at least `len` bytes
    // (one side is a VeriCache-owned GpuSlot / leased HBM frame, the other side
    // is a caller-owned PyTorch CUDA tensor slice). Synchronization ensures the
    // copy completes before returning ownership or releasing the lease guard.
    let rc = unsafe {
        cudaMemcpyAsync(
            dst_addr as *mut c_void,
            src_addr as *const c_void,
            len,
            CUDA_MEMCPY_DEVICE_TO_DEVICE,
            stream,
        )
    };
    if rc != 0 {
        return Err(rc);
    }
    let sync_rc = unsafe { cudaStreamSynchronize(stream) };
    if sync_rc != 0 {
        return Err(sync_rc);
    }
    Ok(())
}
