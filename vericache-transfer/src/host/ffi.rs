//! Synchronous CPU memory operations. The verified caller establishes the
//! allocation lifetime and checks access permissions before each call.
#[cfg(verus_only)]
use super::is_power_of_two_usize;
use super::{CompletedAccess, DmaAllocation, HostBuffer, HostSlice};
#[cfg(verus_only)]
use crate::domain::{
    copy_access, copy_effect, havoc_effect, read_access, read_bytes_effect, reclaim_slab_access,
    splice_bytes, write_access,
};
use crate::memory::HostDmaSlice;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::{self, NonNull};
use vstd::prelude::*;

verus! {

/// Coverage: success yields a distinct live block, with every byte initialized.
/// An empty allocation has a non-null sentinel and needs no deallocation.
#[verifier::external_body]
pub(super) fn raw_alloc_aligned(len: usize, align: usize) -> (result: Option<DmaAllocation>)
    requires
        is_power_of_two_usize(align),
    ensures
        result matches Some(a) ==> a.well_formed() && a.total_len() == len && a.address() % align
            == 0 && a.device_spec().is_none(),
{
    let ptr = if len == 0 {
        align as *mut u8
    } else {
        let layout = Layout::from_size_align(len, align).ok()?;
        NonNull::new(unsafe { alloc_zeroed(layout) })?.as_ptr()
    };
    Some(unsafe { DmaAllocation::from_live_allocation(ptr, len, None) })
}

/// Coverage and observation: reclaim exactly the live base allocation before return.
#[verifier::external_body]
pub(super) fn raw_free(buf: HostBuffer) -> (result: Tracked<CompletedAccess>)
    requires
        buf.well_formed(),
    ensures
        result@.access() == reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
        result@.effect() == havoc_effect(),
{
    let (base, len) = buf.allocation.raw_parts();
    if len > 0 {
        unsafe {
            dealloc(base, Layout::from_size_align(len, buf.align).unwrap());
        }
    }
    unsafe {
        CompletedAccess::from_completed_access(
            Ghost(reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat)),
            Ghost(havoc_effect()),
        )
    }
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
        result@.access() == copy_access(src.slot_ref(), dst.slot_ref()),
        result@.effect() == copy_effect(src.slot_ref(), dst.slot_ref()),
{
    let (src_ptr, len) = src.dma_raw_parts();
    let (dst_ptr, _) = dst.dma_raw_parts();
    unsafe {
        ptr::copy(src_ptr, dst_ptr, len);
    }
    unsafe {
        CompletedAccess::from_completed_access(
            Ghost(copy_access(src.slot_ref(), dst.slot_ref())),
            Ghost(copy_effect(src.slot_ref(), dst.slot_ref())),
        )
    }
}

/// On return the destination range contains the concrete input bytes.
#[verifier::external_body]
pub(super) fn raw_write(dst: &HostSlice, data: &[u8]) -> (result: Tracked<CompletedAccess>)
    requires
        dst.well_formed(),
        data@.len() == dst.slot_ref().sub_len,
    ensures
        result@.access() == write_access(dst.slot_ref()),
        result@.effect() == splice_bytes(dst.slot_ref(), data@),
{
    let (ptr, len) = dst.dma_raw_parts();
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
    }
    unsafe {
        CompletedAccess::from_completed_access(
            Ghost(write_access(dst.slot_ref())),
            Ghost(splice_bytes(dst.slot_ref(), data@)),
        )
    }
}

/// Observe the bytes of a completed external write while the caller owns the
/// destination exclusively. The caller drains DMA before this observation.
#[inline]
#[verifier::external_body]
pub(super) fn raw_observe_write(dst: &HostSlice) -> (result: (
    Tracked<CompletedAccess>,
    Ghost<Seq<u8>>,
))
    requires
        dst.well_formed(),
        dst.slot_ref().loc.space is HostAlloc,
    ensures
        result.0@.access() == write_access(dst.slot_ref()),
        result.0@.effect() == splice_bytes(dst.slot_ref(), result.1@),
        result.1@.len() == dst.slot_ref().sub_len,
{
    // A temporary slice view supplies the observed bytes in the erased proof.
    let (ptr, len) = dst.dma_raw_parts();
    let _bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    (
        unsafe {
            CompletedAccess::from_completed_access(
                Ghost(write_access(dst.slot_ref())),
                Ghost(splice_bytes(dst.slot_ref(), _bytes@)),
            )
        },
        Ghost(_bytes@),
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
        result@.access() == read_access(src.slot_ref()),
        result@.effect() == read_bytes_effect(src.slot_ref(), final(out)@),
{
    let (ptr, len) = src.dma_raw_parts();
    unsafe {
        ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    }
    unsafe {
        CompletedAccess::from_completed_access(
            Ghost(read_access(src.slot_ref())),
            Ghost(read_bytes_effect(src.slot_ref(), out@)),
        )
    }
}

} // verus!
