//! Minimal CUDA runtime D2D copy helper bridging external GPU tensor pointers
//! (e.g., PyTorch / vLLM paged KV cache tensors) and VeriCache managed GPU slots.

use core::ffi::{c_int, c_void};

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
