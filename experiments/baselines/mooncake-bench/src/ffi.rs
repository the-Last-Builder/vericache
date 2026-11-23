use std::ffi::{CString, c_char, c_int, c_void};

#[repr(C)]
struct McTeHandle {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn mc_te_create(rdma_dev: *const c_char, out_rpc_port: *mut c_int) -> *mut McTeHandle;
    fn mc_te_destroy(h: *mut McTeHandle);
    fn mc_te_register_memory(h: *mut McTeHandle, addr: *mut c_void, len: usize) -> c_int;
    fn mc_te_unregister_memory(h: *mut McTeHandle, addr: *mut c_void) -> c_int;
    fn mc_te_open_segment(h: *mut McTeHandle, seg_name: *const c_char) -> i64;
    fn mc_te_close_segment(h: *mut McTeHandle, seg_id: u64) -> c_int;
    fn mc_te_sync_read(
        h: *mut McTeHandle,
        seg_id: u64,
        local_ptr: *mut c_void,
        remote_ptr: u64,
        len: usize,
    ) -> c_int;
    fn mc_te_batch_sync_read(
        h: *mut McTeHandle,
        seg_id: u64,
        local_ptrs: *const *mut c_void,
        remote_ptrs: *const u64,
        lens: *const usize,
        count: usize,
    ) -> c_int;

    fn mc_cuda_set_device(dev: c_int) -> c_int;
    fn mc_cuda_malloc(bytes: usize, out_ptr: *mut *mut c_void) -> c_int;
    fn mc_cuda_free(ptr: *mut c_void) -> c_int;
    fn mc_cuda_pinned_alloc(bytes: usize, out_ptr: *mut *mut c_void) -> c_int;
    fn mc_cuda_pinned_free(ptr: *mut c_void) -> c_int;
    fn mc_cuda_memset(ptr: *mut c_void, val: c_int, bytes: usize) -> c_int;
    fn mc_cuda_stream_create(out_stream: *mut *mut c_void) -> c_int;
    fn mc_cuda_stream_destroy(stream: *mut c_void) -> c_int;
    fn mc_cuda_stream_sync(stream: *mut c_void) -> c_int;
    fn mc_cuda_device_sync() -> c_int;
    fn mc_cuda_memcpy_async_d2d(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn mc_cuda_memcpy_async_h2d(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn mc_cuda_memcpy_async_d2h(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;

    fn mc_odirect_open_unlinked(path: *const c_char, size_bytes: u64) -> c_int;
    fn mc_odirect_pread(fd: c_int, buf: *mut c_void, len: usize, offset: u64) -> c_int;
    fn mc_odirect_pwrite(fd: c_int, buf: *const c_void, len: usize, offset: u64) -> c_int;
    fn mc_odirect_close(fd: c_int) -> c_int;
}

pub struct TransferEngine {
    raw: *mut McTeHandle,
    rpc_port: u16,
}

// SAFETY: Each TransferEngine is owned by a single shard/process; Mooncake TransferEngine is internally synchronized.
unsafe impl Send for TransferEngine {}

impl TransferEngine {
    pub fn new(rdma_dev: &str) -> Result<Self, String> {
        let c_dev = CString::new(rdma_dev).map_err(|e| e.to_string())?;
        let mut port: c_int = 0;
        let raw = unsafe { mc_te_create(c_dev.as_ptr(), &mut port) };
        if raw.is_null() {
            return Err(format!("mc_te_create failed for rdma_dev={rdma_dev}"));
        }
        Ok(Self {
            raw,
            rpc_port: port as u16,
        })
    }

    #[allow(dead_code)]
    pub fn rpc_port(&self) -> u16 {
        self.rpc_port
    }

    pub fn segment_name(&self) -> String {
        format!("127.0.0.1:{}", self.rpc_port)
    }

    pub fn register_memory(&self, addr: u64, len: usize) -> Result<(), i32> {
        let rc = unsafe { mc_te_register_memory(self.raw, addr as *mut c_void, len) };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    pub fn unregister_memory(&self, addr: u64) -> Result<(), i32> {
        let rc = unsafe { mc_te_unregister_memory(self.raw, addr as *mut c_void) };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    pub fn open_segment(&self, seg_name: &str) -> Result<u64, i64> {
        let c_seg = CString::new(seg_name).expect("valid seg_name");
        let id = unsafe { mc_te_open_segment(self.raw, c_seg.as_ptr()) };
        if id < 0 { Err(id) } else { Ok(id as u64) }
    }

    #[allow(dead_code)]
    pub fn close_segment(&self, seg_id: u64) {
        let _ = unsafe { mc_te_close_segment(self.raw, seg_id) };
    }

    #[inline]
    pub fn sync_read(
        &self,
        seg_id: u64,
        local_ptr: u64,
        remote_ptr: u64,
        len: usize,
    ) -> Result<(), i32> {
        let rc =
            unsafe { mc_te_sync_read(self.raw, seg_id, local_ptr as *mut c_void, remote_ptr, len) };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }

    #[inline]
    pub fn batch_sync_read(
        &self,
        seg_id: u64,
        local_ptrs: &[u64],
        remote_ptrs: &[u64],
        lens: &[usize],
    ) -> Result<(), i32> {
        assert_eq!(local_ptrs.len(), remote_ptrs.len());
        assert_eq!(local_ptrs.len(), lens.len());
        let ptrs: Vec<*mut c_void> = local_ptrs.iter().map(|&p| p as *mut c_void).collect();
        let rc = unsafe {
            mc_te_batch_sync_read(
                self.raw,
                seg_id,
                ptrs.as_ptr(),
                remote_ptrs.as_ptr(),
                lens.as_ptr(),
                ptrs.len(),
            )
        };
        if rc == 0 { Ok(()) } else { Err(rc) }
    }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { mc_te_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

pub fn cuda_set_device(dev: u32) {
    let rc = unsafe { mc_cuda_set_device(dev as c_int) };
    assert_eq!(rc, 0, "cudaSetDevice({dev}) failed rc={rc}");
}

pub fn cuda_device_sync() {
    let rc = unsafe { mc_cuda_device_sync() };
    assert_eq!(rc, 0, "cudaDeviceSynchronize failed rc={rc}");
}

pub struct CudaDevBuf {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for CudaDevBuf {}

impl CudaDevBuf {
    pub fn alloc(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { mc_cuda_malloc(len, &mut ptr) };
        assert_eq!(rc, 0, "cudaMalloc({len}) failed rc={rc}");
        Self { ptr, len }
    }

    pub fn fill(&self, byte: u8) {
        let rc = unsafe { mc_cuda_memset(self.ptr, byte as c_int, self.len) };
        assert_eq!(rc, 0, "cudaMemset failed rc={rc}");
    }

    #[inline]
    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }

    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for CudaDevBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { mc_cuda_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

pub struct CudaPinnedBuf {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for CudaPinnedBuf {}

impl CudaPinnedBuf {
    pub fn alloc(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { mc_cuda_pinned_alloc(len, &mut ptr) };
        assert_eq!(rc, 0, "mc_cuda_pinned_alloc({len}) failed rc={rc}");
        Self { ptr, len }
    }

    pub fn fill(&mut self, byte: u8) {
        unsafe {
            std::ptr::write_bytes(self.ptr as *mut u8, byte, self.len);
        }
    }

    #[inline]
    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }

    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_u64_slice_mut(&mut self) -> &mut [u64] {
        assert_eq!(self.len % 8, 0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u64, self.len / 8) }
    }

    #[inline]
    pub fn as_u64_slice(&self) -> &[u64] {
        assert_eq!(self.len % 8, 0);
        unsafe { std::slice::from_raw_parts(self.ptr as *const u64, self.len / 8) }
    }
}

impl Drop for CudaPinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { mc_cuda_pinned_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

pub struct CudaStream {
    raw: *mut c_void,
}

unsafe impl Send for CudaStream {}

impl CudaStream {
    pub fn new() -> Self {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { mc_cuda_stream_create(&mut raw) };
        assert_eq!(rc, 0, "cudaStreamCreateWithFlags failed rc={rc}");
        Self { raw }
    }

    #[inline]
    pub fn memcpy_d2d_async(&self, dst: u64, src: u64, bytes: usize) {
        let rc = unsafe {
            mc_cuda_memcpy_async_d2d(dst as *mut c_void, src as *const c_void, bytes, self.raw)
        };
        assert_eq!(rc, 0, "cudaMemcpyAsync D2D failed rc={rc}");
    }

    #[inline]
    pub fn memcpy_h2d_async(&self, dst: u64, src: u64, bytes: usize) {
        let rc = unsafe {
            mc_cuda_memcpy_async_h2d(dst as *mut c_void, src as *const c_void, bytes, self.raw)
        };
        assert_eq!(rc, 0, "cudaMemcpyAsync H2D failed rc={rc}");
    }

    #[inline]
    pub fn memcpy_d2h_async(&self, dst: u64, src: u64, bytes: usize) {
        let rc = unsafe {
            mc_cuda_memcpy_async_d2h(dst as *mut c_void, src as *const c_void, bytes, self.raw)
        };
        assert_eq!(rc, 0, "cudaMemcpyAsync D2H failed rc={rc}");
    }

    #[inline]
    pub fn sync(&self) {
        let rc = unsafe { mc_cuda_stream_sync(self.raw) };
        assert_eq!(rc, 0, "cudaStreamSynchronize failed rc={rc}");
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { mc_cuda_stream_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

pub struct ODirectFile {
    fd: c_int,
}

impl ODirectFile {
    pub fn create_unlinked(path: &str, size_bytes: u64) -> Self {
        let c_path = CString::new(path).expect("valid path");
        let fd = unsafe { mc_odirect_open_unlinked(c_path.as_ptr(), size_bytes) };
        assert!(
            fd >= 0,
            "mc_odirect_open_unlinked({path}, {size_bytes}) failed rc={fd}"
        );
        Self { fd }
    }

    #[inline]
    pub fn pread(&self, buf_addr: u64, len: usize, offset: u64) {
        let rc = unsafe { mc_odirect_pread(self.fd, buf_addr as *mut c_void, len, offset) };
        assert_eq!(
            rc, 0,
            "O_DIRECT pread(fd={}, len={len}, off={offset}) failed rc={rc}",
            self.fd
        );
    }

    #[inline]
    pub fn pwrite(&self, buf_addr: u64, len: usize, offset: u64) {
        let rc = unsafe { mc_odirect_pwrite(self.fd, buf_addr as *const c_void, len, offset) };
        assert_eq!(
            rc, 0,
            "O_DIRECT pwrite(fd={}, len={len}, off={offset}) failed rc={rc}",
            self.fd
        );
    }
}

impl Drop for ODirectFile {
    fn drop(&mut self) {
        if self.fd >= 0 {
            let _ = unsafe { mc_odirect_close(self.fd) };
            self.fd = -1;
        }
    }
}
