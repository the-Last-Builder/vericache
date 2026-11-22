//! Coverage of file and buffer ranges, CQE observation and full-length copy behavior.
use super::*;
use crate::host::CompletedAccess;
use std::ffi::{c_char, CString};
use std::ptr;

unsafe extern "C" {
    fn vc_nvme_create(dir: *const c_char, len: u64) -> i32;
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

verus! {

/// Coverage: fresh, unlinked O_DIRECT file of exactly len bytes.
#[verifier::external_body]
pub(super) fn create(dir: &str, len: u64) -> Result<i32, NvmeError>
    requires
        len > 0 && len % DIRECT_IO_ALIGNMENT == 0,
{
    let path = CString::new(dir).map_err(|_| NvmeError(22))?;
    let fd = unsafe { vc_nvme_create(path.as_ptr(), len) };
    if fd < 0 {
        Err(NvmeError(-fd))
    } else {
        Ok(fd)
    }
}

/// Linux releases the fd even when close reports a late writeback error.
#[verifier::external_body]
pub(super) fn close(file: NvmeFile) -> (r: Tracked<CompletedAccess>)
    requires
        file.well_formed(),
    ensures
        r@.access == reclaim_slab_access(file.node(), file.space(), file.num_slots() as nat),
        r@.effect == havoc_effect(),
{
    unsafe {
        linux_close(file.fd);
    }
    Tracked(
        CompletedAccess {
            access: reclaim_slab_access(file.node(), file.space(), file.num_slots() as nat),
            effect: havoc_effect(),
        },
    )
}

#[verifier::external_body]
pub(super) fn new(entries: u32) -> (r: Result<(*mut u8, u32, u32), NvmeError>)
    requires
        entries > 0,
    ensures
        r matches Ok((_, sq, cq)) ==> 0 < sq <= cq,
{
    let mut raw = ptr::null_mut();
    let (mut sq, mut cq) = (0, 0);
    let err = unsafe { vc_uring_new(entries, &mut raw, &mut sq, &mut cq) };
    if err == 0 {
        Ok((raw, sq, cq))
    } else {
        Err(NvmeError(err))
    }
}

#[verifier::external_body]
pub(super) fn destroy(ring: &IoUring)
    requires
        ring.well_formed(),
        ring.pending().is_empty(),
{
    unsafe {
        vc_uring_destroy(ring.raw);
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
        vc_uring_stage_io(ring.raw, file.file.fd, file.start, buf, len, id, write as i32)
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
    let n = unsafe { vc_uring_submit(ring.raw, ring.staged_count as u32) };
    if n < 0 {
        Err(NvmeError(-n))
    } else {
        Ok(n as usize)
    }
}

/// Observation: the DMA identified by this CQE has terminated. Behavior:
/// its signed result gives the transferred byte count or a negative errno.
#[verifier::external_body]
pub(super) fn poll(ring: &IoUring) -> (r: Option<(u64, u32, i32)>)
    requires
        ring.well_formed(),
    ensures
        r matches Some((id, len, _)) ==> ring.pending().contains_key(id) && !ring.staged().contains(
            id,
        ) && ring.pending()[id].expected_len == len,
{
    let (mut id, mut len, mut res) = (0, 0, 0);
    let found = unsafe { vc_uring_poll(ring.raw, &mut id, &mut len, &mut res) };
    if found == 0 {
        None
    } else {
        Some((id, len, res))
    }
}

#[verifier::external_body]
pub(super) fn wait(ring: &IoUring) -> Result<(), NvmeError>
    requires
        ring.well_formed(),
        ring.in_flight() > 0,
{
    let err = unsafe { vc_uring_wait(ring.raw) };
    if err == 0 {
        Ok(())
    } else {
        Err(NvmeError(err))
    }
}

} // verus!
