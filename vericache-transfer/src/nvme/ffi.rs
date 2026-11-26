//! Coverage of file and buffer ranges, CQE observation and full-length copy behavior.
use super::*;
use std::ffi::{c_char, CString};
use std::ptr;

unsafe extern "C" {
    fn vc_nvme_create(dir: *const c_char, len: u64) -> i32;
    fn vc_nvme_open_path(path: *const c_char, len: u64, create_if_missing: i32) -> i32;
    #[link_name = "close"]
    fn linux_close(fd: i32) -> i32;
    fn vc_uring_new(entries: u32, ring: *mut *mut u8, sq: *mut u32, cq: *mut u32) -> i32;
    fn vc_uring_destroy(ring: *mut u8);
    fn vc_uring_stage_io(
        ring: *mut u8,
        fd: i32,
        offset: u64,
        buf: *mut u8,
        len: u32,
        id: u64,
        write: i32,
    ) -> i32;
    fn vc_uring_submit(ring: *mut u8, count: u32) -> i32;
    fn vc_uring_poll(ring: *mut u8, id: *mut u64, len: *mut u32, res: *mut i32) -> i32;
    fn vc_uring_wait(ring: *mut u8) -> i32;
}

fn open_cstr(path: &str, f: impl FnOnce(*const c_char) -> i32) -> Result<i32, NvmeError> {
    let cstr = CString::new(path).map_err(|_| NvmeError(22))?;
    let fd = f(cstr.as_ptr());
    if fd < 0 {
        Err(NvmeError(-fd))
    } else {
        Ok(fd)
    }
}

verus! {

#[verifier::external_body]
pub(super) struct NvmeFdGuard(pub(super) i32);

impl NvmeFdGuard {
    #[verifier::external_body]
    pub(super) fn new(fd: i32) -> Self {
        Self(fd)
    }

    #[verifier::external_body]
    pub(super) fn fd(&self) -> i32 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_fd(mut self) -> i32 {
        let fd = self.0;
        self.0 = -1;
        std::mem::forget(self);
        fd
    }
}

impl Drop for NvmeFdGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        if self.0 >= 0 {
            unsafe {
                linux_close(self.0);
            }
            self.0 = -1;
        }
    }
}

#[verifier::external_body]
pub(super) struct IoUringGuard(pub(super) *mut u8);

impl IoUringGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = self.0;
        self.0 = ptr::null_mut();
        std::mem::forget(self);
        raw
    }
}

impl Drop for IoUringGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        if !self.0.is_null() {
            unsafe {
                vc_uring_destroy(self.0);
            }
            self.0 = ptr::null_mut();
        }
    }
}

/// Coverage: fresh, unlinked O_DIRECT file of exactly len bytes.
#[verifier::external_body]
pub(super) fn create(dir: &str, len: u64) -> Result<i32, NvmeError>
    requires
        len > 0 && len % DIRECT_IO_ALIGNMENT == 0,
{
    open_cstr(dir, |p| unsafe { vc_nvme_create(p, len) })
}

/// Coverage: open or create a persistent O_DIRECT file or block device of at least len bytes.
#[verifier::external_body]
pub(super) fn open_path(path: &str, len: u64, create_if_missing: bool) -> Result<i32, NvmeError>
    requires
        len > 0 && len % DIRECT_IO_ALIGNMENT == 0,
{
    open_cstr(path, |p| unsafe { vc_nvme_open_path(p, len, create_if_missing as i32) })
}

/// Linux releases the fd even when close reports a late writeback error.
#[verifier::external_body]
pub(super) fn close(file: NvmeFile) -> (r: Tracked<CompletedAccess>)
    requires
        file.well_formed(),
    ensures
        r@.access() == reclaim_slab_access(file.node(), file.space(), file.num_slots() as nat),
        r@.effect() == havoc_effect(),
{
    let fd = file.fd.into_fd();
    if fd >= 0 {
        unsafe {
            linux_close(fd);
        }
    }
    unsafe {
        CompletedAccess::from_completed_access(
            Ghost(reclaim_slab_access(file.node(), file.space(), file.num_slots() as nat)),
            Ghost(havoc_effect()),
        )
    }
}

#[verifier::external_body]
pub(super) fn new(entries: u32) -> (r: Result<(IoUringGuard, u32, u32), NvmeError>)
    requires
        entries > 0,
    ensures
        r matches Ok((_, sq, cq)) ==> 0 < sq <= cq,
{
    let mut raw = ptr::null_mut();
    let (mut sq, mut cq) = (0, 0);
    let err = unsafe { vc_uring_new(entries, &mut raw, &mut sq, &mut cq) };
    if err == 0 {
        Ok((IoUringGuard::new(raw), sq, cq))
    } else {
        Err(NvmeError(err))
    }
}

#[verifier::external_body]
pub(super) fn destroy(ring: IoUringGuard) {
    let raw = ring.into_raw();
    if !raw.is_null() {
        unsafe {
            vc_uring_destroy(raw);
        }
    }
}

/// Coverage: success stages this exact range and direction without publishing the SQ tail.
#[verifier::external_body]
pub(super) fn stage(
    ring: &IoUring,
    file: &NvmeSlice,
    buf: *mut u8,
    len: u32,
    id: u64,
    write: bool,
) -> Result<(), NvmeError>
    requires
        ring.well_formed(),
        file.well_formed(),
        len == file.slot_ref().sub_len,
        0 < len <= MAX_IO_LEN,
        buf.addr() % (DIRECT_IO_ALIGNMENT as usize) == 0,
        file.file_offset() % DIRECT_IO_ALIGNMENT == 0,
        file.slot_ref().sub_len % DIRECT_IO_ALIGNMENT == 0,
        !ring.pending().contains_key(id),
        ring.staged_count() < ring.sq_capacity(),
        ring.pending_count() < ring.max_in_flight(),
{
    let err = unsafe {
        vc_uring_stage_io(
            ring.raw.as_ptr(),
            file.file.fd.fd(),
            file.start,
            buf,
            len,
            id,
            write as i32,
        )
    };
    if err == 0 {
        Ok(())
    } else {
        Err(NvmeError(err))
    }
}

/// The kernel consumes a FIFO prefix and reports its exact size.
#[verifier::external_body]
pub(super) fn submit(ring: &IoUring) -> (r: Result<usize, NvmeError>)
    requires
        ring.well_formed(),
    ensures
        r matches Ok(n) ==> n <= ring.staged_count(),
{
    let n = unsafe { vc_uring_submit(ring.raw.as_ptr(), ring.staged_count as u32) };
    if n < 0 {
        Err(NvmeError(-n))
    } else {
        Ok(n as usize)
    }
}

/// Observation: the DMA identified by this CQE has terminated. Behavior:
/// its signed result gives the transferred byte count or a negative errno.
#[verifier::external_body]
pub(super) fn poll(ring: &IoUring) -> (r: Option<(u64, u32, i32, Tracked<ObservedEvents>)>)
    requires
        ring.well_formed(),
    ensures
        r matches Some((id, len, res, evidence)) ==> {
            let event = ring.pending()[id].event;
            &&& ring.pending().contains_key(id) && !ring.staged().contains(id)
            &&& ring.pending()[id].expected_len == len
            &&& evidence@.observed(ring.instance_id(), ring.node(), ISet::empty().insert(event))
            &&& res == len as i32 ==> evidence@.succeeded(ring.instance_id(), event)
        },
{
    let (mut id, mut len, mut res) = (0, 0, 0);
    let found = unsafe { vc_uring_poll(ring.raw.as_ptr(), &mut id, &mut len, &mut res) };
    if found == 0 {
        None
    } else {
        let ghost event = ring.pending()[id].event;
        let ghost completed = ISet::empty().insert(event);
        let ghost successful = if res == len as i32 {
            completed
        } else {
            ISet::empty()
        };
        let evidence = unsafe {
            ObservedEvents::from_completion(
                Ghost(ring.instance_id()),
                Ghost(ring.node()),
                Ghost(completed),
                Ghost(successful),
            )
        };
        Some((id, len, res, evidence))
    }
}

#[verifier::external_body]
pub(super) fn wait(ring: &IoUring) -> Result<(), NvmeError>
    requires
        ring.well_formed(),
        ring.in_flight() > 0,
{
    let err = unsafe { vc_uring_wait(ring.raw.as_ptr()) };
    if err == 0 {
        Ok(())
    } else {
        Err(NvmeError(err))
    }
}

} // verus!
