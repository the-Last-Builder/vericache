use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::host::HostContext;
use transfer::nvme::{self, IoUring, NvmeFile, NvmeSlice, DIRECT_IO_ALIGNMENT, MAX_IO_LEN};

mod poll;
#[cfg(verus_only)]
mod proofs;
mod stage;
mod stage_io;
#[cfg(verus_only)]
mod window_proofs;

#[cfg(verus_only)]
use proofs::*;
#[cfg(verus_only)]
use window_proofs::*;

verus! {

/// Non-dropping NVMe completion progress counters (M8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvmeProgress {
    pub completed: usize,
    pub failed: usize,
}

impl NvmeProgress {
    pub fn empty() -> (r: Self)
        ensures
            r.completed == 0,
            r.failed == 0,
    {
        Self { completed: 0, failed: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvmeOpKind {
    DramToNvme,
    NvmeToDram,
}

pub struct InFlightNvmeOp {
    pub kind: NvmeOpKind,
    pub hop: CopyHopHandle,
    pub ticket: Ticket,
    pub src_cap: Tracked<ShareCap>,
    pub dst_tok: Tracked<SlotToken>,
    pub copy: Ghost<ReplicaCopy>,
}

/// Retention invariant for an in-flight NVMe copy hop. Unlike `ReplicaCopy::retained`,
/// destination-writer recency is proved at completion time via `TransferInstance::slot_writer_is_latest`
/// on `InFlightNvmeOp::dst_tok`, allowing arbitrary sibling RDMA/peer ledger merges while
/// NVMe `io_uring` operations are in flight.
pub open spec fn nvme_copy_retained(copy: ReplicaCopy, l: KnowledgeLedger, node: NodeId) -> bool {
    let d = copy.destination;
    let info = l.submitted.map()[copy.writer];
    &&& copy.before.valid(l.submitted.instance_id(), node)
    &&& copy.before.submitted.map().submap_of(l.submitted.map())
    &&& revoked_submap(copy.before.revoked, l.revoked)
    &&& copy.before.succeeded.dom().subset_of(l.succeeded.dom())
    &&& copy.before.citations(node).subset_of(l.citations(node))
    &&& replica_valid(copy.before, node, copy.source)
    &&& l.submitted.dom().contains(copy.writer)
    &&& source_slice_valid(copy.source, copy.source_slot)
    &&& d.well_formed() && copy.source_slot.sub_len == d.sub_len
    &&& info.access.reads.contains(copy.source_slot.loc) && info.access.writes.contains(d.loc)
    &&& effect_implies(info.effect, copy_effect(copy.source_slot, d))
    &&& info.preds.subset_of(copy.before.submitted.dom())
    &&& copy.before.citations(node).subset_of(info.preds)
}

impl InFlightNvmeOp {
    pub open spec fn valid(&self, core: &CacheCore, timeline: u8, slot_idx: int) -> bool {
        let s = self.ticket.sequence();
        &&& self.ticket != Ticket::NONE
        &&& self.ticket.timeline_id() == timeline
        &&& (s % BITMAP_CAPACITY) as int == slot_idx
        &&& core.timelines.issued(timeline, s)
        &&& core.timelines.events@[(timeline, s)] == self.copy@.writer
        &&& self.hop.src_frame != self.hop.dst_frame
        &&& self.hop.src_frame.0 < core.frames.metas.len()
        &&& self.hop.dst_frame.0 < core.frames.metas.len()
        &&& self.hop.src_slot@ == core.frames.frame(self.hop.src_frame).slot
        &&& self.hop.dst_slot@ == core.frames.frame(self.hop.dst_frame).slot
        &&& self.hop.src_slot@.loc != self.hop.dst_slot@.loc
        &&& core.frames.meta(self.hop.dst_frame).state == STATE_INITIALIZING
        &&& !core.frames.meta(self.hop.dst_frame).queued()
        &&& core.frames.frame(self.hop.dst_frame).tier == self.hop.dst_tier
        &&& core.frames.frame(self.hop.dst_frame).bound_key == Some(self.hop.key)
        &&& core.frames.frame(self.hop.dst_frame).bound_version == Some(self.hop.version)
        &&& core.frames.frame(self.hop.dst_frame).token is None
        &&& core.frames.versions@.contains_key(self.hop.version)
        &&& core.frames.versions@[self.hop.version] == self.hop.src_cert@.bytes
        &&& (self.hop.version.origin_node != core.node_id || self.hop.version.seq
            < core.next_version_seq || self.hop.version.seq >= core.version_seq_limit)
        &&& self.hop.src_lease.frame == self.hop.src_frame
        &&& core.active_lease_matches(self.hop.src_lease)
        &&& core.active_leases[self.hop.src_lease.slot as int]->Some_0.internal
        &&& self.src_cap@.share()
            == core.active_leases[self.hop.src_lease.slot as int]->Some_0.share@
        &&& self.src_cap@.is_local()
        &&& self.src_cap@.instance_id() == core.instance_id()
        &&& self.src_cap@.loc() == self.hop.src_slot@.loc
        &&& self.src_cap@.frontier().readers == ISet::empty().insert(self.copy@.writer)
        &&& slot_owned(
            self.dst_tok@,
            core.instance_id(),
            self.hop.dst_slot@.loc,
            Frontier { last_writer: Some(self.copy@.writer), readers: ISet::empty() },
        )
        &&& self.copy@.destination == self.hop.dst_slot@
        &&& self.copy@.source == self.hop.src_cert@
        &&& self.copy@.source_slot == self.hop.src_slot@
        &&& self.copy@.source.slot == self.copy@.source_slot
        &&& self.copy@.source.bytes.len() == self.copy@.destination.slot_len
        &&& self.copy@.source.bytes.len() == self.copy@.source_slot.sub_len
        &&& nvme_copy_retained(self.copy@, core.ledger@, core.node())
    }
}

/// Local NVMe storage backend owning an `O_DIRECT` `NvmeFile` slab (4KB aligned)
/// and an `io_uring` submission/completion ring.
pub struct NvmeBackend {
    pub file: NvmeFile,
    pub ring: IoUring,
    pub tier: TierId,
    pub timeline: u8,
    pub first_frame: FrameId,
    pub num_frames: u32,
    pub bitmap: SlidingBitmap,
    pub in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pub pending_transfers: Vec<Option<InFlightNvmeOp>>,
}

#[verifier::opaque]
pub open spec fn nvme_pending_disjoint(pending_transfers: Seq<Option<InFlightNvmeOp>>) -> bool {
    &&& pending_transfers.len() == BITMAP_CAPACITY
    &&& forall|i: int, j: int|
        0 <= i < BITMAP_CAPACITY && 0 <= j < BITMAP_CAPACITY && i != j ==> match (#[trigger]
            pending_transfers[i],
            #[trigger] pending_transfers[j],
        ) {
            (Some(a), Some(b)) => a.hop.dst_frame != b.hop.dst_frame && a.hop.dst_slot@.loc
                != b.hop.dst_slot@.loc && a.hop.src_lease.slot != b.hop.src_lease.slot,
            _ => true,
        }
}

#[verifier::opaque]
pub open spec fn nvme_pending_valid(
    pending_transfers: Seq<Option<InFlightNvmeOp>>,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    ring: IoUring,
    core: &CacheCore,
    timeline: u8,
) -> bool {
    &&& pending_transfers.len() == BITMAP_CAPACITY
    &&& nvme_pending_disjoint(pending_transfers)
    &&& forall|i: int|
        0 <= i < BITMAP_CAPACITY ==> (#[trigger] in_flight_ops[i]).is_some()
            == pending_transfers[i].is_some()
    &&& forall|i: int|
        0 <= i < BITMAP_CAPACITY ==> (#[trigger] pending_transfers[i] matches Some(op) ==> {
            &&& op.valid(core, timeline, i)
            &&& ring.pending().contains_key(op.ticket.sequence())
            &&& ring.pending()[op.ticket.sequence()].event == op.copy@.writer
        })
}

pub open spec fn ring_event_map(ring: IoUring) -> Map<u64, TransferEvent> {
    Map::new(ring.pending().dom(), |s: u64| ring.pending()[s].event)
}

#[verifier::opaque]
pub open spec fn nvme_window_wf(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    ring: IoUring,
    core: &CacheCore,
    timeline: u8,
) -> bool {
    &&& bitmap.well_formed()
    &&& bitmap.observations(&core.timelines, timeline, core.node(), &core.ledger@)
    &&& window_matches(in_flight_ops, bitmap, &core.timelines, timeline, ring_event_map(ring))
}

impl NvmeBackend {
    pub open spec fn well_formed(&self, core: &CacheCore) -> bool {
        &&& self.file.well_formed()
        &&& self.file.instance_id() == core.instance_id()
        &&& self.file.node() == core.node()
        &&& self.file.num_slots() == self.num_frames as usize
        &&& self.file.slot_len() == core.frames.tier_frame_size(self.tier) as u64
        &&& 0 < self.file.slot_len() <= MAX_IO_LEN
        &&& self.file.slot_len() % DIRECT_IO_ALIGNMENT == 0
        &&& self.ring.well_formed()
        &&& self.ring.instance_id() == core.instance_id()
        &&& self.ring.node() == core.node()
        &&& self.ring.source().source() is IoUring
        &&& self.tier.valid(core.tiers.len())
        &&& core.tiers[self.tier.index() as int].kind == TierKind::LocalNvme
        &&& self.timeline < core.timelines.count
        &&& core.timelines.count <= MAX_TIMELINES
        &&& core.timelines.kinds[self.timeline as int] == TimelineKind::IoUring
        &&& slab_bound_to_core(
            core,
            self.tier,
            self.first_frame,
            self.num_frames,
            self.file.space(),
        )
        &&& self.pending_transfers.len() == BITMAP_CAPACITY
        &&& nvme_window_wf(self.bitmap, self.in_flight_ops, self.ring, core, self.timeline)
        &&& nvme_pending_valid(
            self.pending_transfers@,
            self.in_flight_ops,
            self.ring,
            core,
            self.timeline,
        )
    }

    pub open spec fn owns(&self, frame: FrameId) -> bool {
        self.first_frame.0 <= frame.0 && frame.0 < self.first_frame.0 + self.num_frames
    }

    fn validate_init(core: &CacheCore, num_frames: u32, ring_entries: u32, frame_size: usize) -> (r:
        BackendResult<(usize, u64)>)
        requires
            core.well_formed(),
        ensures
            r matches Ok((count, slot_len)) ==> {
                &&& num_frames > 0 && ring_entries > 0
                &&& count == num_frames as usize
                &&& slot_len == frame_size as u64
                &&& 0 < slot_len <= MAX_IO_LEN
                &&& slot_len % DIRECT_IO_ALIGNMENT == 0
                &&& (count as nat) * (slot_len as nat) <= u64::MAX
                &&& core.tiers.len() < MAX_TIERS
                &&& core.timelines.count < MAX_TIMELINES
                &&& core.frames.metas.len() + count <= u32::MAX
            },
    {
        hide(CacheCore::well_formed);
        proof {
            lemma_core_wf_facts(core);
        }
        if frame_size == 0 || num_frames == 0 || ring_entries == 0 || frame_size > MAX_IO_LEN || (
        frame_size as u64) % DIRECT_IO_ALIGNMENT != 0 {
            return Err(CoreError::InvalidArgument);
        }
        let count = num_frames as usize;
        let slot_len = frame_size as u64;
        if (count as u64) > u64::MAX / slot_len {
            return Err(CoreError::InvalidArgument);
        }
        if core.tiers.len() >= MAX_TIERS || core.timelines.count >= MAX_TIMELINES || count > (
        u32::MAX as usize) - core.frames.metas.len() {
            return Err(CoreError::CapacityExhausted);
        }
        proof {
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(u64::MAX as int, slot_len as int);
            vstd::arithmetic::mul::lemma_mul_inequality(
                count as int,
                (u64::MAX / slot_len) as int,
                slot_len as int,
            );
            assert((count as nat) * (slot_len as nat) <= u64::MAX);
        }
        Ok((count, slot_len))
    }

    fn finish_init(
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&AllocSource>,
        file: NvmeFile,
        ring: IoUring,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
        Ghost(before_alloc): Ghost<AllocSource>,
        frame_size: usize,
        num_frames: u32,
    ) -> (r: Self)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            before_alloc.well_formed(),
            before_alloc.id() == old(core).instance_id(),
            before_alloc.node() == old(core).node(),
            alloc_fresh_for_core(before_alloc, old(core)),
            alloc.well_formed(),
            alloc.id() == old(core).instance_id(),
            alloc.node() == old(core).node(),
            alloc.names() == before_alloc.names().tail(),
            !alloc.names().contains(before_alloc.next_alloc_id()),
            file.well_formed(),
            file.instance_id() == old(core).instance_id(),
            file.node() == old(core).node(),
            file.space().alloc_id() == before_alloc.next_alloc_id(),
            file.num_slots() == num_frames as usize,
            file.slot_len() == frame_size as u64,
            0 < file.slot_len() <= MAX_IO_LEN,
            file.slot_len() % DIRECT_IO_ALIGNMENT == 0,
            alloc.allocated_slab(before_alloc, file.space(), num_frames as usize, tokens),
            ring.well_formed(),
            ring.instance_id() == old(core).instance_id(),
            ring.node() == old(core).node(),
            ring.source().source() is IoUring,
            ring.pending() == Map::empty(),
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
                file.space(),
                count,
                frame_size,
                tokens,
            );
        }
        let timeline = core.register_timeline(TimelineKind::IoUring).unwrap();
        let ghost after_tl = *core;
        let desc = TierDescriptor {
            id: TierId(0),
            kind: TierKind::LocalNvme,
            device_id: None,
            host_accessible: false,
            rdma_capable: false,
        };
        let (tier, first_frame) = core.register_tier_with_frame_size(
            desc,
            frame_size,
            Ghost(slots),
            num_frames,
            Tracked(tokens),
        ).unwrap();
        let bitmap = SlidingBitmap::new();
        let in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS] = [None;BITMAP_SLOTS];
        let mut pending_transfers: Vec<Option<InFlightNvmeOp>> = Vec::with_capacity(BITMAP_SLOTS);
        let mut i: usize = 0;
        while i < BITMAP_SLOTS
            invariant
                i <= BITMAP_SLOTS,
                pending_transfers.len() == i,
                forall|k: int| 0 <= k < i ==> (#[trigger] pending_transfers[k]).is_none(),
            decreases BITMAP_SLOTS - i,
        {
            pending_transfers.push(None);
            i += 1;
        }
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
                file.space(),
                slots,
            );
            lemma_nvme_init_state(bitmap, in_flight_ops, pending_transfers@, ring, core, timeline);
        }
        Self {
            file,
            ring,
            tier,
            timeline,
            first_frame,
            num_frames,
            bitmap,
            in_flight_ops,
            pending_transfers,
        }
    }

    /// Create an `O_DIRECT` temporary NVMe slab file using the node default `frame_size`,
    /// initialize its `io_uring`, and register its `TierKind::LocalNvme` tier and
    /// `TimelineKind::IoUring` timeline in `core`.
    pub fn new(
        core: &mut CacheCore,
        host_ctx: &HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(ring_source): Tracked<EventSourceToken>,
        dir: &str,
        num_frames: u32,
        ring_entries: u32,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            host_ctx.ready(old(core).instance_id(), old(core).node()),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            ring_source.well_formed(),
            ring_source.id() == old(core).instance_id(),
            ring_source.node() == old(core).node(),
            ring_source.source() is IoUring,
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
            host_ctx,
            Tracked(instance),
            Tracked(alloc),
            Tracked(ring_source),
            dir,
            num_frames,
            ring_entries,
            frame_size,
        )
    }

    /// Create an `O_DIRECT` temporary NVMe slab file with an explicit per-tier
    /// `frame_size`, initialize its `io_uring`, and register its tier in `core`.
    pub fn new_with_frame_size(
        core: &mut CacheCore,
        host_ctx: &HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(ring_source): Tracked<EventSourceToken>,
        dir: &str,
        num_frames: u32,
        ring_entries: u32,
        frame_size: usize,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            host_ctx.ready(old(core).instance_id(), old(core).node()),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            ring_source.well_formed(),
            ring_source.id() == old(core).instance_id(),
            ring_source.node() == old(core).node(),
            ring_source.source() is IoUring,
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
                    &&& b.num_frames == num_frames
                    &&& b.tier == TierId(old(core).tiers.len() as u16)
                    &&& b.first_frame.0 == old(core).frames.metas.len()
                    &&& b.timeline == old(core).timelines.count
                    &&& final(core).frames.tier_frame_size(b.tier) == frame_size
                },
            },
    {
        hide(CacheCore::well_formed);
        let (count, slot_len) = Self::validate_init(core, num_frames, ring_entries, frame_size)?;
        let ring = match IoUring::new(Tracked(ring_source), ring_entries) {
            Ok(ring) => ring,
            Err((e, _)) => return Err(nvme_error(e)),
        };
        let ghost before_alloc = *alloc;
        let (file, Tracked(tokens)) = match NvmeFile::create_temp_slab(
            host_ctx,
            Tracked(instance),
            Tracked(alloc),
            dir,
            count,
            slot_len,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                let _ = ring.destroy();
                return Err(nvme_error(e));
            },
        };
        Ok(
            Self::finish_init(
                core,
                Tracked(instance),
                Tracked(alloc),
                file,
                ring,
                Tracked(tokens),
                Ghost(before_alloc),
                frame_size,
                num_frames,
            ),
        )
    }

    /// Open or create an `O_DIRECT` NVMe slab file at an explicit filesystem `path`
    /// with per-tier `frame_size`, initialize its `io_uring`, and register its tier in `core`.
    pub fn new_with_path(
        core: &mut CacheCore,
        host_ctx: &HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(ring_source): Tracked<EventSourceToken>,
        path: &str,
        num_frames: u32,
        ring_entries: u32,
        create_if_missing: bool,
        frame_size: usize,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            host_ctx.ready(old(core).instance_id(), old(core).node()),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            ring_source.well_formed(),
            ring_source.id() == old(core).instance_id(),
            ring_source.node() == old(core).node(),
            ring_source.source() is IoUring,
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
                    &&& b.num_frames == num_frames
                    &&& b.tier == TierId(old(core).tiers.len() as u16)
                    &&& b.first_frame.0 == old(core).frames.metas.len()
                    &&& b.timeline == old(core).timelines.count
                    &&& final(core).frames.tier_frame_size(b.tier) == frame_size
                },
            },
    {
        hide(CacheCore::well_formed);
        let (count, slot_len) = Self::validate_init(core, num_frames, ring_entries, frame_size)?;
        let ring = match IoUring::new(Tracked(ring_source), ring_entries) {
            Ok(ring) => ring,
            Err((e, _)) => return Err(nvme_error(e)),
        };
        let ghost before_alloc = *alloc;
        let (file, Tracked(tokens)) = match NvmeFile::open_path_slab(
            host_ctx,
            Tracked(instance),
            Tracked(alloc),
            path,
            count,
            slot_len,
            create_if_missing,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                let _ = ring.destroy();
                return Err(nvme_error(e));
            },
        };
        Ok(
            Self::finish_init(
                core,
                Tracked(instance),
                Tracked(alloc),
                file,
                ring,
                Tracked(tokens),
                Ghost(before_alloc),
                frame_size,
                num_frames,
            ),
        )
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

    pub fn owns_frame(&self, frame: FrameId) -> (r: bool)
        ensures
            r == self.owns(frame),
    {
        frame.0 >= self.first_frame.0 && frame.0 - self.first_frame.0 < self.num_frames
    }

    pub fn frame_slice(&self, frame: FrameId) -> (s: NvmeSlice<'_>)
        requires
            self.file.well_formed(),
            self.file.num_slots() == self.num_frames as usize,
            self.owns(frame),
        ensures
            s.well_formed(),
            s.slot_ref() == self.file.slot_ref((frame.0 - self.first_frame.0) as nat),
            s.file_offset() == ((frame.0 - self.first_frame.0) as nat) * (
            self.file.slot_len() as nat),
            s.file_offset() % DIRECT_IO_ALIGNMENT == 0,
            s.instance_id() == self.file.instance_id(),
    {
        let idx = (frame.0 - self.first_frame.0) as usize;
        self.file.slot_slice(idx)
    }

    pub fn staged_len(&self) -> (n: usize)
        requires
            self.ring.well_formed(),
        ensures
            n == self.ring.staged_count(),
    {
        self.ring.staged_len()
    }

    pub fn in_flight_len(&self) -> (n: usize)
        requires
            self.ring.well_formed(),
        ensures
            n == self.ring.in_flight(),
    {
        self.ring.in_flight_len()
    }

    /// Submit staged `io_uring` SQEs to the kernel.
    pub fn submit(&mut self, Ghost(core): Ghost<&CacheCore>) -> (r: BackendResult<usize>)
        requires
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            r matches Ok(n) ==> n <= old(self).ring.staged_count() && final(self).ring.staged()
                == old(self).ring.staged().skip(n as int),
    {
        match self.ring.submit() {
            Ok(n) => {
                proof {
                    lemma_nvme_after_submit(
                        self.bitmap,
                        self.in_flight_ops,
                        self.pending_transfers@,
                        old(self).ring,
                        self.ring,
                        core,
                        self.timeline,
                    );
                }
                Ok(n)
            },
            Err(e) => Err(nvme_error(e)),
        }
    }
}

} // verus!
