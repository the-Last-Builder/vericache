use super::*;
use transfer::host::{self, HostBuffer, HostContext, HostSlice};

mod ops;
mod shared;

verus! {

/// Host-DRAM storage backend owning one slab of `HostBuffer` slots registered
/// as a `TierKind::HostDram` tier on `CacheCore`.
pub struct HostBackend {
    pub ctx: HostContext,
    pub buffer: HostBuffer,
    pub tier: TierId,
    pub timeline: u8,
    pub first_frame: FrameId,
    pub num_frames: u32,
}

impl HostBackend {
    pub open spec fn well_formed(&self, core: &CacheCore) -> bool {
        &&& self.buffer.well_formed()
        &&& self.buffer.align() > 0
        &&& self.buffer.address() % self.buffer.align() == 0
        &&& self.buffer.instance_id() == core.instance_id()
        &&& self.buffer.node() == core.node()
        &&& !(self.buffer.space() is GpuAlloc)
        &&& self.buffer.num_slots() == self.num_frames as usize
        &&& self.buffer.slot_len() == core.frames.tier_frame_size(self.tier)
        &&& self.ctx.ready(core.instance_id(), core.node())
        &&& source_fresh_for_ledger(self.ctx.source@, core.ledger@)
        &&& self.tier.valid(core.tiers.len())
        &&& core.tiers[self.tier.index() as int].kind == TierKind::HostDram
        &&& self.timeline < core.timelines.count
        &&& core.timelines.count <= MAX_TIMELINES
        &&& core.timelines.kinds[self.timeline as int] == TimelineKind::Host
        &&& core.timelines.confirmed_seq[self.timeline as int] + 1
            == core.timelines.next_seq[self.timeline as int]
        &&& slab_bound_to_core(
            core,
            self.tier,
            self.first_frame,
            self.num_frames,
            self.buffer.space(),
        )
    }

    pub open spec fn owns(&self, frame: FrameId) -> bool {
        self.first_frame.0 <= frame.0 && frame.0 < self.first_frame.0 + self.num_frames
    }

    /// Allocate a `HostBuffer` slab (`align` must be a power of two; pass `4096`
    /// when pairing with NVMe `O_DIRECT`) and register its tier and host timeline
    /// into `core` using the node default `frame_size`.
    pub fn new(
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(source): Tracked<EventSourceToken>,
        num_frames: u32,
        align: usize,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            source.well_formed(),
            source.id() == old(core).instance_id(),
            source.node() == old(core).node(),
            source.source() is Host,
            source_fresh_for_ledger(source, old(core).ledger@),
        ensures
            final(core).well_formed(),
            final(alloc).well_formed(),
            final(alloc).id() == old(alloc).id(),
            final(alloc).node() == old(alloc).node(),
            alloc_fresh_for_core(*final(alloc), final(core)),
            final(core).node_id == old(core).node_id,
            final(core).ledger == old(core).ledger,
            final(core).spec == old(core).spec,
            match r {
                Err(_) => *final(core) == *old(core) && *final(alloc) == *old(alloc),
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.buffer.align() == align
                    &&& b.num_frames == num_frames
                    &&& b.tier == TierId(old(core).tiers.len() as u16)
                    &&& b.first_frame.0 == old(core).frames.metas.len()
                    &&& b.timeline == old(core).timelines.count
                    &&& final(core).frames.tier_frame_size(b.tier) == old(core).frames.frame_size
                },
            },
    {
        let frame_size = core.frames.frame_size;
        Self::new_with_frame_size(
            core,
            Tracked(instance),
            Tracked(alloc),
            Tracked(source),
            num_frames,
            align,
            frame_size,
        )
    }

    fn finish_init(
        core: &mut CacheCore,
        Tracked(alloc): Tracked<&AllocSource>,
        ctx: HostContext,
        buffer: HostBuffer,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
        Ghost(before_alloc): Ghost<AllocSource>,
        align: usize,
        frame_size: usize,
        num_frames: u32,
    ) -> (r: Self)
        requires
            old(core).well_formed(),
            before_alloc.well_formed(),
            before_alloc.id() == old(core).instance_id(),
            before_alloc.node() == old(core).node(),
            alloc_fresh_for_core(before_alloc, old(core)),
            alloc.well_formed(),
            alloc.id() == old(core).instance_id(),
            alloc.node() == old(core).node(),
            alloc.names() == before_alloc.names().tail(),
            !alloc.names().contains(before_alloc.next_alloc_id()),
            buffer.well_formed(),
            buffer.align() == align,
            align > 0,
            buffer.address() % buffer.align() == 0,
            buffer.instance_id() == old(core).instance_id(),
            buffer.node() == old(core).node(),
            !(buffer.space() is GpuAlloc),
            buffer.space().alloc_id() == before_alloc.next_alloc_id(),
            buffer.num_slots() == num_frames as usize,
            buffer.slot_len() == frame_size,
            0 < frame_size,
            alloc.allocated_slab(before_alloc, buffer.space(), num_frames as usize, tokens),
            ctx.ready(old(core).instance_id(), old(core).node()),
            source_fresh_for_ledger(ctx.source@, old(core).ledger@),
            num_frames > 0,
            old(core).tiers.len() < MAX_TIERS,
            old(core).timelines.count < MAX_TIMELINES,
            old(core).frames.metas.len() + (num_frames as usize) <= u32::MAX,
        ensures
            final(core).well_formed(),
            alloc_fresh_for_core(*alloc, final(core)),
            final(core).node_id == old(core).node_id,
            final(core).ledger == old(core).ledger,
            final(core).spec == old(core).spec,
            r.well_formed(final(core)),
            r.buffer.align() == align,
            r.num_frames == num_frames,
            r.tier == TierId(old(core).tiers.len() as u16),
            r.first_frame.0 == old(core).frames.metas.len(),
            r.timeline == old(core).timelines.count,
            final(core).frames.tier_frame_size(r.tier) == frame_size,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let count = num_frames as usize;
        let ghost slots;
        proof {
            slots =
            lemma_allocated_slab_disjoint_with_size(
                before_alloc,
                *alloc,
                core,
                buffer.space(),
                count,
                frame_size,
                tokens,
            );
        }
        let timeline = core.register_timeline(TimelineKind::Host).unwrap();
        let ghost after_tl = *core;
        let desc = TierDescriptor {
            id: TierId(0),
            kind: TierKind::HostDram,
            device_id: None,
            host_accessible: true,
            rdma_capable: true,
        };
        let (tier, first_frame) = core.register_tier_with_frame_size(
            desc,
            frame_size,
            Ghost(slots),
            num_frames,
            Tracked(tokens),
        ).unwrap();
        proof {
            lemma_core_wf_facts(&after_tl);
            lemma_core_wf_facts(core);
            lemma_after_register_tier(
                before_alloc,
                *alloc,
                &after_tl,
                core,
                tier,
                first_frame,
                num_frames,
                buffer.space(),
                slots,
            );
        }
        Self { ctx, buffer, tier, timeline, first_frame, num_frames }
    }

    /// Allocate a `HostBuffer` slab (`align` must be a power of two; pass `4096`
    /// when pairing with NVMe `O_DIRECT`) with an explicit per-tier `frame_size`,
    /// registering its tier and host timeline into `core`.
    pub fn new_with_frame_size(
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(source): Tracked<EventSourceToken>,
        num_frames: u32,
        align: usize,
        frame_size: usize,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            source.well_formed(),
            source.id() == old(core).instance_id(),
            source.node() == old(core).node(),
            source.source() is Host,
            source_fresh_for_ledger(source, old(core).ledger@),
        ensures
            final(core).well_formed(),
            final(alloc).well_formed(),
            final(alloc).id() == old(alloc).id(),
            final(alloc).node() == old(alloc).node(),
            alloc_fresh_for_core(*final(alloc), final(core)),
            final(core).node_id == old(core).node_id,
            final(core).ledger == old(core).ledger,
            final(core).spec == old(core).spec,
            match r {
                Err(_) => *final(core) == *old(core) && *final(alloc) == *old(alloc),
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.buffer.align() == align
                    &&& b.num_frames == num_frames
                    &&& b.tier == TierId(old(core).tiers.len() as u16)
                    &&& b.first_frame.0 == old(core).frames.metas.len()
                    &&& b.timeline == old(core).timelines.count
                    &&& final(core).frames.tier_frame_size(b.tier) == frame_size
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if frame_size == 0 || num_frames == 0 || align == 0 || (align & (align - 1)) != 0 {
            return Err(CoreError::InvalidArgument);
        }
        let count = num_frames as usize;
        proof {
            lemma_core_wf_facts(core);
        }
        if count > usize::MAX / frame_size {
            return Err(CoreError::InvalidArgument);
        }
        if core.tiers.len() >= MAX_TIERS || core.timelines.count >= MAX_TIMELINES || count > (
        u32::MAX as usize) - core.frames.metas.len() {
            return Err(CoreError::CapacityExhausted);
        }
        proof {
            assert(count * frame_size <= usize::MAX) by (nonlinear_arith)
                requires
                    frame_size > 0,
                    count <= usize::MAX / frame_size,
            ;
        }
        let ctx = HostContext::new(Tracked(source));
        let ghost before_alloc = *alloc;
        let Some((buffer, Tracked(tokens))) = host::alloc_slab(
            &ctx,
            Tracked(instance),
            Tracked(alloc),
            count,
            frame_size,
            align,
        ) else {
            return Err(CoreError::CapacityExhausted);
        };
        Ok(
            Self::finish_init(
                core,
                Tracked(alloc),
                ctx,
                buffer,
                Tracked(tokens),
                Ghost(before_alloc),
                align,
                frame_size,
                num_frames,
            ),
        )
    }

    pub fn buffer(&self) -> (b: &HostBuffer)
        ensures
            b == &self.buffer,
    {
        &self.buffer
    }

    pub fn tier_id(&self) -> (t: TierId)
        ensures
            t == self.tier,
    {
        self.tier
    }

    pub fn timeline_id(&self) -> (t: u8)
        ensures
            t == self.timeline,
    {
        self.timeline
    }

    pub fn first_frame(&self) -> (f: FrameId)
        ensures
            f == self.first_frame,
    {
        self.first_frame
    }

    pub fn num_frames(&self) -> (n: u32)
        ensures
            n == self.num_frames,
    {
        self.num_frames
    }

    pub fn owns_frame(&self, frame: FrameId) -> (r: bool)
        ensures
            r == self.owns(frame),
    {
        frame.0 >= self.first_frame.0 && frame.0 - self.first_frame.0 < self.num_frames
    }

    pub fn frame_slice(&self, frame: FrameId) -> (s: HostSlice<'_>)
        requires
            self.buffer.well_formed(),
            self.buffer.num_slots() == self.num_frames as usize,
            self.owns(frame),
        ensures
            s.well_formed(),
            s.slot_ref() == self.buffer.slot_ref((frame.0 - self.first_frame.0) as nat),
            s.address() == self.buffer.address() + ((frame.0 - self.first_frame.0) as usize)
                * self.buffer.slot_len(),
            s.instance_id() == self.buffer.instance_id(),
    {
        let idx = (frame.0 - self.first_frame.0) as usize;
        self.buffer.slot_slice(idx)
    }
}

} // verus!
