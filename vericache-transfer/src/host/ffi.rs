//! Synchronous CPU memory operations. The verified caller establishes the
//! allocation lifetime and checks access permissions before each call.
#[cfg(verus_only)]
use super::is_power_of_two_usize;
use super::{CompletedAccess, HostBuffer, HostSlice};
#[cfg(verus_only)]
use crate::domain::{
    copy_access, copy_effect, havoc_effect, read_access, read_bytes_effect, reclaim_slab_access,
    splice_bytes, write_access,
};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::{self, NonNull};
use vstd::prelude::*;

verus! {

/// Coverage: success yields a distinct live block, with every byte initialized.
/// An empty allocation has a non-null sentinel and needs no deallocation.
#[verifier::external_body]
pub(super) fn raw_alloc_aligned(len: usize, align: usize) -> (result: Option<*mut u8>)
    requires
        is_power_of_two_usize(align),
    ensures
        result matches Some(ptr) ==> ptr.addr() != 0 && ptr.addr() % align == 0 && ptr.addr() + len
            <= usize::MAX,
{
    if len == 0 {
        return Some(align as *mut u8);
    }
    let layout = Layout::from_size_align(len, align).ok()?;
    NonNull::new(unsafe { alloc_zeroed(layout) }).map(NonNull::as_ptr)
}

/// Coverage and observation: reclaim exactly the live base allocation before return.
#[verifier::external_body]
pub(super) fn raw_free(buf: HostBuffer) -> (result: Tracked<CompletedAccess>)
    requires
        buf.well_formed(),
    ensures
        result@.access == reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
        result@.effect == havoc_effect(),
{
    let len = buf.len;
    let base = buf.ptr;
    if len > 0 {
        unsafe {
            dealloc(base, Layout::from_size_align(len, buf.align).unwrap());
        }
    }
    Tracked(
        CompletedAccess {
            access: reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
            effect: havoc_effect(),
        },
    )
}

/// Coverage, behavior, observation: move exactly `len` bytes, synchronously.
/// memmove semantics also handle copies between overlapping subregions.
#[verifier::external_body]
pub(super) fn raw_copy(dst: &HostSlice, src: &HostSlice) -> (result: Tracked<CompletedAccess>)
    requires
        src.well_formed(),
        dst.well_formed(),
        src.slot_ref().sub_len == dst.slot_ref().sub_len,
    ensures
        result@.access == copy_access(src.slot_ref(), dst.slot_ref()),
        result@.effect == copy_effect(src.slot_ref(), dst.slot_ref()),
{
    unsafe {
        ptr::copy(src.ptr, dst.ptr, src.len);
    }
    Tracked(
        CompletedAccess {
            access: copy_access(src.slot_ref(), dst.slot_ref()),
            effect: copy_effect(src.slot_ref(), dst.slot_ref()),
        },
    )
}

/// On return the destination range contains the concrete input bytes.
#[verifier::external_body]
pub(super) fn raw_write(dst: &HostSlice, data: &[u8]) -> (result: Tracked<CompletedAccess>)
    requires
        dst.well_formed(),
        data@.len() == dst.slot_ref().sub_len,
    ensures
        result@.access == write_access(dst.slot_ref()),
        result@.effect == splice_bytes(dst.slot_ref(), data@),
{
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), dst.ptr, data.len());
    }
    Tracked(
        CompletedAccess {
            access: write_access(dst.slot_ref()),
            effect: splice_bytes(dst.slot_ref(), data@),
        },
    )
}

/// On return the output contains the bytes present in the source range.
#[verifier::external_body]
pub(super) fn raw_read(src: &HostSlice, out: &mut [u8]) -> (result: Tracked<CompletedAccess>)
    requires
        src.well_formed(),
        old(out)@.len() == src.slot_ref().sub_len,
    ensures
        final(out)@.len() == old(out)@.len(),
        result@.access == read_access(src.slot_ref()),
        result@.effect == read_bytes_effect(src.slot_ref(), final(out)@),
{
    unsafe {
        ptr::copy_nonoverlapping(src.ptr, out.as_mut_ptr(), out.len());
    }
    Tracked(
        CompletedAccess {
            access: read_access(src.slot_ref()),
            effect: read_bytes_effect(src.slot_ref(), out@),
        },
    )
}

} // verus!
