use super::*;
use transfer::host::{self, HostBuffer, HostContext, HostSlice};

mod ops;
mod shared;

#[cfg(verus_only)]
mod preserve;
#[cfg(verus_only)]
pub use preserve::*;

verus! {

/// Host-DRAM storage backend owning one slab of `HostBuffer` slots registered
/// as a `TierKind::HostDram` tier on `CacheCore`.
pub struct HostBackend {
    pub(crate) core_id: Option<u64>,
    pub(crate) ctx: HostContext,
    pub(crate) buffer: HostBuffer,
    pub(crate) tier: TierId,
    pub(crate) timeline: TimelineId,
    pub(crate) first_frame: FrameId,
    pub(crate) num_frames: u32,
}

impl HostBackend {
    pub open(crate) spec fn ctx_spec(&self) -> HostContext {
        self.ctx
    }

    pub open(crate) spec fn timeline_spec(&self) -> TimelineId {
        self.timeline
    }

    pub open(crate) spec fn buffer_spec(&self) -> HostBuffer {
        self.buffer
    }

    pub open(crate) spec fn tier_spec(&self) -> TierId {
        self.tier
    }

    pub open(crate) spec fn first_frame_spec(&self) -> FrameId {
        self.first_frame
    }

    pub open(crate) spec fn num_frames_spec(&self) -> u32 {
        self.num_frames
    }

    pub open(crate) spec fn well_formed(&self, core: &CacheCore) -> bool {
        &&& self.buffer.well_formed()
        &&& self.buffer.align() > 0
        &&& self.buffer.address() % self.buffer.align() == 0
        &&& self.buffer.instance_id() == core.instance_id()
        &&& self.buffer.node() == core.node()
        &&& !(self.buffer.space().is_gpu())
        &&& self.buffer.num_slots() == self.num_frames as usize
        &&& self.buffer.slot_len() == core.frames_spec().tier_frame_size(self.tier)
        &&& self.ctx.ready(core.instance_id(), core.node())
        &&& source_fresh_for_ledger(self.ctx.source@, core.ledger_spec()@)
        &&& self.tier.valid(core.tiers_spec().len())
        &&& core.tiers_spec()[self.tier.index() as int].kind == TierKind::HostDram
        &&& self.timeline < core.timelines_spec().count_spec()
        &&& core.timelines_spec().count_spec() <= MAX_TIMELINES
        &&& core.timelines_spec().kinds_spec()[self.timeline as int] == TimelineKind::Host
        &&& core.timelines_spec().confirmed_seq_spec()[self.timeline as int] + 1
            == core.timelines_spec().next_seq_spec()[self.timeline as int]
        &&& slab_bound_to_core(
            core,
            self.tier,
            self.first_frame,
            self.num_frames,
            self.buffer.space(),
        )
    }

    pub open(crate) spec fn owns(&self, frame: FrameId) -> bool {
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
            source.source().is_host(),
            source_fresh_for_ledger(source, old(core).ledger_spec()@),
        ensures
            final(core).well_formed(),
            final(alloc).well_formed(),
            final(alloc).id() == old(alloc).id(),
            final(alloc).node() == old(alloc).node(),
            alloc_fresh_for_core(*final(alloc), final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            match r {
                Err(_) => *final(core) == *old(core) && *final(alloc) == *old(alloc),
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.ctx_spec().source@ == source
                    &&& b.buffer_spec().align() == align
                    &&& b.num_frames_spec() == num_frames
                    &&& b.tier_spec() == TierId(old(core).tiers_spec().len() as u16)
                    &&& b.first_frame_spec().0 == old(core).frames_spec().metas_spec().len()
                    &&& b.timeline_spec() == old(core).timelines_spec().count_spec()
                    &&& final(core).frames_spec().tier_frame_size(b.tier_spec()) == old(
                        core,
                    ).frames_spec().frame_size_spec()
                },
            },
    {
        let frame_size = core.frame_size();
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
            !(buffer.space().is_gpu()),
            buffer.space().alloc_id() == before_alloc.next_alloc_id(),
            buffer.num_slots() == num_frames as usize,
            buffer.slot_len() == frame_size,
            0 < frame_size,
            alloc.allocated_slab(before_alloc, buffer.space(), num_frames as usize, tokens),
            ctx.ready(old(core).instance_id(), old(core).node()),
            source_fresh_for_ledger(ctx.source@, old(core).ledger_spec()@),
            num_frames > 0,
            old(core).tiers_spec().len() < MAX_TIERS,
            old(core).timelines_spec().count_spec() < MAX_TIMELINES,
            old(core).frames_spec().metas_spec().len() + (num_frames as usize) <= u32::MAX,
        ensures
            final(core).well_formed(),
            alloc_fresh_for_core(*alloc, final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            r.well_formed(final(core)),
            r.ctx == ctx,
            r.buffer.align() == align,
            r.num_frames == num_frames,
            r.tier == TierId(old(core).tiers_spec().len() as u16),
            r.first_frame.0 == old(core).frames_spec().metas_spec().len(),
            r.timeline == old(core).timelines_spec().count_spec(),
            final(core).frames_spec().tier_frame_size(r.tier) == frame_size,
    {
        hide(CacheCore::well_formed);
        hide(FrameTable::shape);
        hide(AllocSource::allocated_slab);
        hide(alloc_fresh_for_core);
        hide(source_fresh_for_ledger);
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
            assert(after_tl.frames_spec().tier_frame_sizes_spec().len()
                == after_tl.tiers_spec().len()) by {
                reveal(FrameTable::shape);
                lemma_core_wf_facts(&after_tl);
            }
            assert(core.frames_spec().tier_frame_size(tier) == frame_size);
            assert(alloc_fresh_for_core(before_alloc, &after_tl)) by {
                reveal(alloc_fresh_for_core);
            }
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
        Self { core_id: core.runtime_id(), ctx, buffer, tier, timeline, first_frame, num_frames }
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
            source.source().is_host(),
            source_fresh_for_ledger(source, old(core).ledger_spec()@),
        ensures
            final(core).well_formed(),
            final(alloc).well_formed(),
            final(alloc).id() == old(alloc).id(),
            final(alloc).node() == old(alloc).node(),
            alloc_fresh_for_core(*final(alloc), final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            match r {
                Err(_) => *final(core) == *old(core) && *final(alloc) == *old(alloc),
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.ctx_spec().source@ == source
                    &&& b.buffer_spec().align() == align
                    &&& b.num_frames_spec() == num_frames
                    &&& b.tier_spec() == TierId(old(core).tiers_spec().len() as u16)
                    &&& b.first_frame_spec().0 == old(core).frames_spec().metas_spec().len()
                    &&& b.timeline_spec() == old(core).timelines_spec().count_spec()
                    &&& final(core).frames_spec().tier_frame_size(b.tier_spec()) == frame_size
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
        if core.tier_count() >= MAX_TIERS || core.timeline_count() >= MAX_TIMELINES || count > (
        u32::MAX as usize) - core.frame_count() {
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

    pub fn context(&self) -> (ctx: &HostContext)
        ensures
            *ctx == self.ctx_spec(),
    {
        &self.ctx
    }

    pub fn buffer(&self) -> (b: &HostBuffer)
        ensures
            b == &self.buffer_spec(),
    {
        &self.buffer
    }

    pub fn tier_id(&self) -> (t: TierId)
        ensures
            t == self.tier_spec(),
    {
        self.tier
    }

    pub fn timeline_id(&self) -> (t: TimelineId)
        ensures
            t == self.timeline_spec(),
    {
        self.timeline
    }

    pub fn first_frame(&self) -> (f: FrameId)
        ensures
            f == self.first_frame_spec(),
    {
        self.first_frame
    }

    pub fn num_frames(&self) -> (n: u32)
        ensures
            n == self.num_frames_spec(),
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
            self.buffer_spec().well_formed(),
            self.buffer_spec().num_slots() == self.num_frames_spec() as usize,
            self.owns(frame),
        ensures
            s.well_formed(),
            s.slot_ref() == self.buffer_spec().slot_ref(
                (frame.0 - self.first_frame_spec().0) as nat,
            ),
            s.address() == self.buffer_spec().address() + ((frame.0
                - self.first_frame_spec().0) as usize) * self.buffer_spec().slot_len(),
            s.instance_id() == self.buffer_spec().instance_id(),
    {
        let idx = (frame.0 - self.first_frame.0) as usize;
        self.buffer.slot_slice(idx)
    }
}

impl HostSlabBackend for HostBackend {
    type Buffer = HostBuffer;

    open spec fn host_tier_spec(&self) -> TierId {
        self.tier_spec()
    }

    open spec fn host_first_frame_spec(&self) -> FrameId {
        self.first_frame_spec()
    }

    open spec fn host_num_frames_spec(&self) -> u32 {
        self.num_frames_spec()
    }

    open spec fn host_buffer_spec(&self) -> Self::Buffer {
        self.buffer_spec()
    }

    open spec fn host_alignment_spec(&self) -> usize {
        self.buffer_spec().align()
    }

    open spec fn host_well_formed(&self, core: &CacheCore) -> bool {
        self.well_formed(core)
    }

    proof fn lemma_host_layout(&self, core: &CacheCore) {
    }

    fn host_tier_id(&self) -> TierId {
        self.tier_id()
    }

    fn host_first_frame(&self) -> FrameId {
        self.first_frame()
    }

    fn host_num_frames(&self) -> u32 {
        self.num_frames()
    }

    fn host_buffer(&self) -> &Self::Buffer {
        self.buffer()
    }

    fn host_owns_frame(&self, frame: FrameId) -> bool {
        self.owns_frame(frame)
    }

    fn host_frame_slice(&self, frame: FrameId) -> HostSlice<'_> {
        self.frame_slice(frame)
    }
}

/// Lightweight ghost witness tying a completed host read (`bytes`) to a published
/// `ValueCertificate` for `version` in `core.spec_spec()@.versions` via `read_bytes_effect` (M9).
#[cfg(verus_only)]
pub open(crate) spec fn host_read_witness(
    core: &CacheCore,
    read_ev: TransferEvent,
    cert: ValueCertificate,
    version: VersionId,
    bytes: Seq<u8>,
) -> bool {
    &&& core.spec_spec()@.versions.contains_key(version)
    &&& cert.bytes == core.spec_spec()@.versions[version]
    &&& bytes.len() == core.spec_spec()@.versions[version].len()
    &&& cert.slot.well_formed()
    &&& replica_valid(core.ledger_spec()@, core.node(), cert)
    &&& event_succeeded(core.ledger_spec()@, core.node(), read_ev)
    &&& core.ledger_spec()@.submitted.dom().contains(read_ev)
    &&& core.ledger_spec()@.submitted.map()[read_ev].access.reads.contains(cert.slot.loc)
    &&& effect_implies(
        core.ledger_spec()@.submitted.map()[read_ev].effect,
        read_bytes_effect(cert.slot, bytes),
    )
}

/// Certify a single synchronous full-slot host byte write into a `ValueCertificate`
/// with minimal proof context (CVC5-friendly).
pub proof fn lemma_certify_host_write(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    writer: TransferEvent,
    slot: SlotRef,
    bytes: Seq<u8>,
) -> (cert: ValueCertificate)
    requires
        complete_receipts(before),
        !before.submitted.dom().contains(writer),
        before.submitted.map().submap_of(after.submitted.map()),
        revoked_submap(before.revoked, after.revoked),
        after.submitted.dom() == before.submitted.dom().insert(writer),
        slot.well_formed(),
        bytes.len() == slot.sub_len,
        after.citations(node).contains(writer),
        writer.source.succeeds_on_observe(),
        after.submitted.map()[writer].access == write_access(slot),
        after.submitted.map()[writer].effect == splice_bytes(slot, bytes),
    ensures
        replica_valid(after, node, cert),
        cert.writer == writer,
        cert.slot == slot,
        cert.bytes == bytes,
        cert.proof == ValueProof::HostWrite,
{
    assert(after.submitted.map()[writer].access.writes.contains(slot.loc));
    assert(event_succeeded(after, node, writer));
    assert(effect_implies(after.submitted.map()[writer].effect, splice_bytes(slot, bytes)));
    certify_host_replica(before, after, node, writer, slot, bytes)
}

} // verus!
