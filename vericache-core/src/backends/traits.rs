#[cfg(verus_only)]
use super::slab_bound_to_core;
#[cfg(verus_only)]
use crate::kernel::CacheCore;
#[cfg(verus_only)]
use crate::spec::TierKind;
use crate::spec::{FrameId, TierId};
#[cfg(verus_only)]
use transfer_core::domain::{SlotLoc, SlotRef};
use transfer_core::memory::{HostDmaSlab, HostSlice};
use vstd::prelude::*;

verus! {

/// The contiguous host slab used by device DMA and NVMe direct I/O.
pub trait HostSlabBackend: Sized {
    type Buffer: HostDmaSlab;

    spec fn host_tier_spec(&self) -> TierId;

    spec fn host_first_frame_spec(&self) -> FrameId;

    spec fn host_num_frames_spec(&self) -> u32;

    spec fn host_buffer_spec(&self) -> Self::Buffer;

    spec fn host_alignment_spec(&self) -> usize;

    open spec fn host_owns(&self, frame: FrameId) -> bool {
        self.host_first_frame_spec().0 <= frame.0 && frame.0 < self.host_first_frame_spec().0
            + self.host_num_frames_spec()
    }

    spec fn host_well_formed(&self, core: &CacheCore) -> bool;

    proof fn lemma_host_layout(&self, core: &CacheCore)
        requires
            self.host_well_formed(core),
        ensures
            host_slab_layout(
                core,
                self.host_buffer_spec(),
                self.host_tier_spec(),
                self.host_first_frame_spec(),
                self.host_num_frames_spec(),
                self.host_alignment_spec(),
            ),
            forall|frame: FrameId| #[trigger]
                self.host_owns(frame) <==> {
                    self.host_first_frame_spec().0 <= frame.0 && frame.0
                        < self.host_first_frame_spec().0 + self.host_num_frames_spec()
                },
    ;

    fn host_tier_id(&self) -> (tier: TierId)
        ensures
            tier == self.host_tier_spec(),
    ;

    fn host_first_frame(&self) -> (frame: FrameId)
        ensures
            frame == self.host_first_frame_spec(),
    ;

    fn host_num_frames(&self) -> (count: u32)
        ensures
            count == self.host_num_frames_spec(),
    ;

    fn host_buffer(&self) -> (buffer: &Self::Buffer)
        ensures
            *buffer == self.host_buffer_spec(),
    ;

    fn host_owns_frame(&self, frame: FrameId) -> (owns: bool)
        ensures
            owns == self.host_owns(frame),
    ;

    fn host_frame_slice(&self, frame: FrameId) -> (slice: HostSlice<'_>)
        requires
            self.host_buffer_spec().slab_well_formed(),
            self.host_buffer_spec().slab_num_slots() == self.host_num_frames_spec() as usize,
            self.host_owns(frame),
        ensures
            slice.well_formed(),
            slice.instance_id() == self.host_buffer_spec().slab_instance_id(),
            ({
                let buffer = self.host_buffer_spec();
                let index = (frame.0 - self.host_first_frame_spec().0) as nat;
                &&& slice.address() == buffer.slab_address() + index * buffer.slab_slot_len()
                &&& slice.slot_ref() == SlotRef::full(
                    SlotLoc {
                        node: buffer.slab_node(),
                        space: buffer.slab_space(),
                        slot_index: index,
                    },
                    buffer.slab_slot_len() as u64,
                )
            }),
    ;
}

pub open spec fn host_slab_layout<B: HostDmaSlab>(
    core: &CacheCore,
    buffer: B,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    alignment: usize,
) -> bool {
    &&& buffer.slab_well_formed()
    &&& buffer.slab_instance_id() == core.instance_id()
    &&& buffer.slab_node() == core.node()
    &&& !buffer.slab_space().is_gpu()
    &&& buffer.slab_num_slots() == num_frames as usize
    &&& buffer.slab_slot_len() == core.frames_spec().tier_frame_size(tier)
    &&& alignment > 0
    &&& buffer.slab_address() % alignment == 0
    &&& tier.valid(core.tiers_spec().len())
    &&& core.tiers_spec()[tier.index() as int].kind == TierKind::HostDram
    &&& slab_bound_to_core(core, tier, first_frame, num_frames, buffer.slab_space())
}

pub open spec fn host_backend_layout<H: HostSlabBackend>(host: &H, core: &CacheCore) -> bool {
    &&& host_slab_layout(
        core,
        host.host_buffer_spec(),
        host.host_tier_spec(),
        host.host_first_frame_spec(),
        host.host_num_frames_spec(),
        host.host_alignment_spec(),
    )
    &&& forall|frame: FrameId| #[trigger]
        host.host_owns(frame) <==> {
            host.host_first_frame_spec().0 <= frame.0 && frame.0 < host.host_first_frame_spec().0
                + host.host_num_frames_spec()
        }
}

} // verus!
