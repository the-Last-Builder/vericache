#[cfg(verus_only)]
use crate::frame::*;
#[cfg(not(verus_only))]
use crate::spec::CoreError;
#[cfg(verus_only)]
use crate::{kernel::*, provenance::*, spec::*, timeline::*};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, knowledge_of, untouched};
#[cfg(verus_only)]
use transfer_core::{context::*, domain::*};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

mod progress;

pub use progress::ProgressReport;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

pub type BackendResult<T> = Result<T, CoreError>;

/// Partial or complete progress on a device timeline and its pending writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevicePollReport {
    pub events_completed: usize,
    pub writes_published: usize,
    pub writes_aborted: usize,
}

impl DevicePollReport {
    pub fn empty() -> (r: Self)
        ensures
            r.events_completed == 0,
            r.writes_published == 0,
            r.writes_aborted == 0,
    {
        Self { events_completed: 0, writes_published: 0, writes_aborted: 0 }
    }
}

/// Every frame registered in `core` came from an `AllocId` no longer in `source`.
pub open spec fn alloc_fresh_for_core(source: AllocSource, core: &CacheCore) -> bool {
    forall|i: nat|
        i < core.frames_spec().metas_spec().len() ==> !source.names().contains(
            (#[trigger] core.frames_spec().ghosts_spec()@[i].slot).loc.space.alloc_id(),
        )
}

/// Every event recorded in `ledger` from `source`'s hardware stream has `seq < source.next_seq()`.
pub open spec fn source_fresh_for_ledger(
    source: EventSourceToken,
    ledger: KnowledgeLedger,
) -> bool {
    forall|e: TransferEvent| #[trigger]
        ledger.submitted.dom().contains(e) && e.source == source.source() && e.observer
            == source.node() ==> e.seq < source.next_seq()
}

/// Contiguous one-to-one binding between a hardware slab's `0..num_frames` slots
/// and the exclusive frames `[first_frame, first_frame + num_frames)` of `tier`.
#[verifier::opaque]
pub open spec fn slab_bound_to_core(
    core: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
) -> bool {
    &&& tier.valid(core.tiers_spec().len())
    &&& num_frames > 0
    &&& (first_frame.0 as nat) + (num_frames as nat) <= core.frames_spec().metas_spec().len()
    &&& forall|j: nat|
        j < num_frames ==> {
            &&& (#[trigger] core.frames_spec().frame(FrameId((first_frame.0 + j) as u32))).tier
                == tier
            &&& core.frames_spec().frame(FrameId((first_frame.0 + j) as u32)).slot == SlotRef::full(
                SlotLoc { node: core.node(), space, slot_index: j },
                core.frames_spec().tier_frame_size(tier) as u64,
            )
        }
    &&& forall|f: FrameId|
        f.0 < core.frames_spec().metas_spec().len() && (#[trigger] core.frames_spec().frame(f)).tier
            == tier ==> first_frame.0 <= f.0 && f.0 < first_frame.0 + num_frames
}

pub proof fn lemma_slab_owns_tier_frame(
    core: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
    f: FrameId,
)
    requires
        slab_bound_to_core(core, tier, first_frame, num_frames, space),
        f.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().frame(f).tier == tier,
    ensures
        first_frame.0 <= f.0 && f.0 < first_frame.0 + num_frames,
{
    reveal(slab_bound_to_core);
}

pub proof fn lemma_slab_frame_slot(
    core: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
    f: FrameId,
)
    requires
        slab_bound_to_core(core, tier, first_frame, num_frames, space),
        first_frame.0 <= f.0 && f.0 < first_frame.0 + num_frames,
    ensures
        f.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().frame(f).tier == tier,
        core.frames_spec().frame(f).slot == SlotRef::full(
            SlotLoc { node: core.node(), space, slot_index: (f.0 - first_frame.0) as nat },
            core.frames_spec().tier_frame_size(tier) as u64,
        ),
{
    reveal(slab_bound_to_core);
    let j = (f.0 - first_frame.0) as nat;
    assert(FrameId((first_frame.0 + j) as u32) == f);
}

/// Show that a freshly allocated slab from `AllocSource` is disjoint from every
/// existing frame in `core` and satisfies `CacheCore::register_tier_with_frame_size` preconditions.
pub proof fn lemma_allocated_slab_disjoint_with_size(
    before_alloc: AllocSource,
    after_alloc: AllocSource,
    core: &CacheCore,
    space: MemorySpace,
    num_frames: usize,
    frame_size: usize,
    tokens: Map<usize, SlotToken>,
) -> (slots: Seq<SlotRef>)
    requires
        core.well_formed(),
        0 < frame_size,
        before_alloc.well_formed(),
        before_alloc.id() == core.instance_id(),
        before_alloc.node() == core.node(),
        alloc_fresh_for_core(before_alloc, core),
        space.alloc_id() == before_alloc.next_alloc_id(),
        after_alloc.allocated_slab(before_alloc, space, num_frames, tokens),
    ensures
        slots.len() == num_frames,
        forall|i: int|
            0 <= i < num_frames ==> (#[trigger] slots[i]) == SlotRef::full(
                SlotLoc { node: core.node(), space, slot_index: i as nat },
                frame_size as u64,
            ),
        slab_slots_ready(
            tokens,
            slots,
            0,
            num_frames,
            core.instance_id(),
            core.node(),
            Some(frame_size),
        ),
        forall|i: int, j: int|
            0 <= i < j < num_frames ==> (#[trigger] slots[i]).loc != (#[trigger] slots[j]).loc,
        forall|i: nat, j: int|
            i < core.frames_spec().metas_spec().len() && 0 <= j < num_frames ==> (
            #[trigger] core.frames_spec().ghosts_spec()@[i].slot).loc != (#[trigger] slots[j]).loc,
{
    let slots = lemma_slab_slots_valid(
        core.instance_id(),
        core.node(),
        space,
        num_frames,
        frame_size,
        tokens,
    );
    assert(before_alloc.names().contains(space.alloc_id())) by {
        assert((before_alloc.names().name)(0) == before_alloc.next_alloc_id());
    }
    assert forall|i: nat, j: int|
        i < core.frames_spec().metas_spec().len() && 0 <= j < num_frames implies (
    #[trigger] core.frames_spec().ghosts_spec()@[i].slot).loc != (#[trigger] slots[j]).loc by {
        let existing = core.frames_spec().ghosts_spec()@[i].slot.loc;
        assert(!before_alloc.names().contains(existing.space.alloc_id()));
        assert(slots[j].loc.space == space);
        assert(existing.space.alloc_id() != space.alloc_id());
    }
    slots
}

/// Show that a freshly allocated slab from `AllocSource` is disjoint from every
/// existing frame in `core` and satisfies `CacheCore::register_tier` preconditions.
pub proof fn lemma_allocated_slab_disjoint(
    before_alloc: AllocSource,
    after_alloc: AllocSource,
    core: &CacheCore,
    space: MemorySpace,
    num_frames: usize,
    tokens: Map<usize, SlotToken>,
) -> (slots: Seq<SlotRef>)
    requires
        core.well_formed(),
        before_alloc.well_formed(),
        before_alloc.id() == core.instance_id(),
        before_alloc.node() == core.node(),
        alloc_fresh_for_core(before_alloc, core),
        space.alloc_id() == before_alloc.next_alloc_id(),
        after_alloc.allocated_slab(before_alloc, space, num_frames, tokens),
    ensures
        slots.len() == num_frames,
        forall|i: int|
            0 <= i < num_frames ==> (#[trigger] slots[i]) == SlotRef::full(
                SlotLoc { node: core.node(), space, slot_index: i as nat },
                core.frames_spec().frame_size_spec() as u64,
            ),
        slab_slots_ready(
            tokens,
            slots,
            0,
            num_frames,
            core.instance_id(),
            core.node(),
            Some(core.frames_spec().frame_size_spec()),
        ),
        forall|i: int, j: int|
            0 <= i < j < num_frames ==> (#[trigger] slots[i]).loc != (#[trigger] slots[j]).loc,
        forall|i: nat, j: int|
            i < core.frames_spec().metas_spec().len() && 0 <= j < num_frames ==> (
            #[trigger] core.frames_spec().ghosts_spec()@[i].slot).loc != (#[trigger] slots[j]).loc,
{
    lemma_allocated_slab_disjoint_with_size(
        before_alloc,
        after_alloc,
        core,
        space,
        num_frames,
        core.frames_spec().frame_size_spec(),
        tokens,
    )
}

/// Preserve `alloc_fresh_for_core` and establish `slab_bound_to_core` after
/// `CacheCore::register_tier` / `CacheCore::register_tier_with_frame_size`.
pub proof fn lemma_after_register_tier(
    before_alloc: AllocSource,
    after_alloc: AllocSource,
    before_core: &CacheCore,
    after_core: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
    slots: Seq<SlotRef>,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        before_alloc.well_formed(),
        alloc_fresh_for_core(before_alloc, before_core),
        after_alloc.names() == before_alloc.names().tail(),
        !after_alloc.names().contains(before_alloc.next_alloc_id()),
        space.alloc_id() == before_alloc.next_alloc_id(),
        tier == TierId(before_core.tiers_spec().len() as u16),
        after_core.tiers_spec().len() == before_core.tiers_spec().len() + 1,
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.frames_spec().frame_size_spec() == before_core.frames_spec().frame_size_spec(),
        num_frames > 0,
        first_frame.0 == before_core.frames_spec().metas_spec().len(),
        after_core.frames_spec().metas_spec().len() == before_core.frames_spec().metas_spec().len()
            + num_frames,
        slots.len() == num_frames,
        forall|j: int|
            0 <= j < num_frames ==> (#[trigger] slots[j]) == SlotRef::full(
                SlotLoc { node: after_core.node(), space, slot_index: j as nat },
                after_core.frames_spec().tier_frame_size(tier) as u64,
            ),
        forall|j: nat|
            j < before_core.frames_spec().metas_spec().len()
                ==> #[trigger] after_core.frames_spec().ghosts_spec()@[j]
                == before_core.frames_spec().ghosts_spec()@[j],
        forall|j: int|
            0 <= j < num_frames ==> (#[trigger] after_core.frames_spec().frame(
                FrameId((first_frame.0 + j) as u32),
            )).tier == tier && after_core.frames_spec().frame(
                FrameId((first_frame.0 + j) as u32),
            ).slot == slots[j],
    ensures
        alloc_fresh_for_core(after_alloc, after_core),
        slab_bound_to_core(after_core, tier, first_frame, num_frames, space),
{
    reveal(slab_bound_to_core);
    assert forall|i: nat|
        i < after_core.frames_spec().metas_spec().len() implies !after_alloc.names().contains(
        (#[trigger] after_core.frames_spec().ghosts_spec()@[i].slot).loc.space.alloc_id(),
    ) by {
        if i < before_core.frames_spec().metas_spec().len() {
            let aid = after_core.frames_spec().ghosts_spec()@[i].slot.loc.space.alloc_id();
            assert(after_core.frames_spec().ghosts_spec()@[i]
                == before_core.frames_spec().ghosts_spec()@[i]);
            assert(!before_alloc.names().contains(aid));
            if after_alloc.names().contains(aid) {
                let k = choose|k: nat| #[trigger] (after_alloc.names().name)(k) == aid;
                assert((before_alloc.names().name)(k + 1) == aid);
            }
        } else {
            let j = (i - before_core.frames_spec().metas_spec().len()) as int;
            assert(FrameId((first_frame.0 + j) as u32).0 as nat == i);
            assert(after_core.frames_spec().frame(FrameId((first_frame.0 + j) as u32)).slot
                == slots[j]);
        }
    }
    assert forall|f: FrameId|
        f.0 < after_core.frames_spec().metas_spec().len() && (
        #[trigger] after_core.frames_spec().frame(f)).tier == tier implies first_frame.0 <= f.0
        && f.0 < first_frame.0 + num_frames by {
        if f.0 < before_core.frames_spec().metas_spec().len() {
            assert(before_core.frames_spec().valid_frame(f, before_core.kernel_view()));
            assert(before_core.frames_spec().frame(f).tier.valid(before_core.tiers_spec().len()));
            assert(after_core.frames_spec().ghosts_spec()@[f.0 as nat]
                == before_core.frames_spec().ghosts_spec()@[f.0 as nat]);
        }
    }
}

/// Preserve `slab_bound_to_core` whenever all existing frames keep their `(tier, slot)`
/// and no new frames are appended (e.g., `same_directory`, `finish_write`, `evict_frame`).
pub proof fn lemma_slab_bound_preserved(
    before: &CacheCore,
    after: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
)
    requires
        slab_bound_to_core(before, tier, first_frame, num_frames, space),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec().len() == before.tiers_spec().len(),
        after.frames_spec().frame_size_spec() == before.frames_spec().frame_size_spec(),
        after.frames_spec().tier_frame_size(tier) == before.frames_spec().tier_frame_size(tier),
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
        forall|f: FrameId|
            f.0 < before.frames_spec().metas_spec().len() ==> (#[trigger] after.frames_spec().frame(
                f,
            )).tier == before.frames_spec().frame(f).tier && after.frames_spec().frame(f).slot
                == before.frames_spec().frame(f).slot,
    ensures
        slab_bound_to_core(after, tier, first_frame, num_frames, space),
{
    reveal(slab_bound_to_core);
    assert forall|j: nat| j < num_frames implies {
        &&& (#[trigger] after.frames_spec().frame(FrameId((first_frame.0 + j) as u32))).tier == tier
        &&& after.frames_spec().frame(FrameId((first_frame.0 + j) as u32)).slot == SlotRef::full(
            SlotLoc { node: after.node(), space, slot_index: j },
            after.frames_spec().tier_frame_size(tier) as u64,
        )
    } by {
        let f = FrameId((first_frame.0 + j) as u32);
        assert(before.frames_spec().frame(f).tier == tier);
    }
    assert forall|f: FrameId|
        f.0 < after.frames_spec().metas_spec().len() && (#[trigger] after.frames_spec().frame(
            f,
        )).tier == tier implies first_frame.0 <= f.0 && f.0 < first_frame.0 + num_frames by {
        assert(before.frames_spec().frame(f).tier == tier);
    }
}

/// Preserve `slab_bound_to_core` when `frames`, `tiers`, and `node_id` are identical.
pub proof fn lemma_slab_bound_same_frames(
    before: &CacheCore,
    after: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
)
    requires
        slab_bound_to_core(before, tier, first_frame, num_frames, space),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec().len() == before.tiers_spec().len(),
        after.frames_spec() == before.frames_spec(),
    ensures
        slab_bound_to_core(after, tier, first_frame, num_frames, space),
{
    reveal(slab_bound_to_core);
}

/// Preserve `slab_bound_to_core` directly from `FrameTable::same_layout`.
pub proof fn lemma_slab_bound_same_layout(
    before: &CacheCore,
    after: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
)
    requires
        slab_bound_to_core(before, tier, first_frame, num_frames, space),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec().len() == before.tiers_spec().len(),
        after.frames_spec().same_layout(&before.frames_spec()),
    ensures
        slab_bound_to_core(after, tier, first_frame, num_frames, space),
        after.frames_spec().frame_size_spec() == before.frames_spec().frame_size_spec(),
        after.frames_spec().tier_frame_sizes_spec() == before.frames_spec().tier_frame_sizes_spec(),
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
{
    lemma_slab_bound_preserved(before, after, tier, first_frame, num_frames, space);
}

/// Preserve `slab_bound_to_core` directly from `FrameTable::same_directory`.
pub proof fn lemma_slab_bound_same_directory(
    before: &CacheCore,
    after: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
)
    requires
        slab_bound_to_core(before, tier, first_frame, num_frames, space),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec().len() == before.tiers_spec().len(),
        after.frames_spec().same_directory(&before.frames_spec()),
    ensures
        slab_bound_to_core(after, tier, first_frame, num_frames, space),
        after.frames_spec().frame_size_spec() == before.frames_spec().frame_size_spec(),
        after.frames_spec().tier_frame_sizes_spec() == before.frames_spec().tier_frame_sizes_spec(),
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
{
    lemma_slab_bound_preserved(before, after, tier, first_frame, num_frames, space);
}

/// Extract `core.catalog_spec().indexed()` from `core.well_formed()` without exposing
/// the full `CacheCore::well_formed` body.
pub proof fn lemma_core_catalog_indexed(core: &CacheCore)
    requires
        core.well_formed(),
    ensures
        core.catalog_spec().indexed(),
{
}

/// Show that `core.citations()` satisfies the closure and knowledge preconditions
/// of `vericache-transfer` operations with empty `after`.
pub proof fn lemma_citations_closure(core: &CacheCore)
    requires
        core.well_formed(),
    ensures
        core.citations().subset_of(core.ledger_spec()@.submitted.dom()),
        knowledge_of(core.node(), core.citations()).subset_of(core.ledger_spec()@.knows.set()),
        core.citations().subset_of(
            closure_from(core.ledger_spec()@.submitted.map(), core.citations()),
        ),
        closure_from(core.ledger_spec()@.submitted.map(), core.citations()).subset_of(
            core.ledger_spec()@.submitted.dom(),
        ),
{
    let ordered = closure_from(core.ledger_spec()@.submitted.map(), core.citations());
    assert(core.citations().subset_of(ordered));
    assert forall|e: TransferEvent| #[trigger]
        ordered.contains(e) implies core.ledger_spec()@.submitted.dom().contains(e) by {
        if !core.citations().contains(e) {
            let s = choose|s: TransferEvent|
                core.citations().contains(s) && (
                #[trigger] core.ledger_spec()@.submitted.map()[s]).preds.contains(e);
            assert(core.ledger_spec()@.submitted.dom().contains(s));
            assert(core.ledger_spec()@.submitted.map()[s].preds.subset_of(
                core.ledger_spec()@.submitted.dom(),
            ));
        }
    }
}

/// After a synchronous or already-observed read event `ev` has been recorded into
/// `core.citations()`, the updated `ShareCap` satisfies `returned_reads(..., Ticket::NONE, ...)`.
pub proof fn lemma_returned_reads_observed(
    core: &CacheCore,
    frame: FrameId,
    cap: ShareCap,
    ev: TransferEvent,
)
    requires
        core.well_formed(),
        frame.0 < core.frames_spec().metas_spec().len(),
        cap.well_formed(),
        cap.instance_id() == core.instance_id(),
        cap.loc() == core.frames_spec().frame(frame).slot.loc,
        cap.frontier().readers == ISet::empty().insert(ev),
        core.citations().contains(ev),
    ensures
        core.frames_spec().returned_reads(frame, cap, Ticket::NONE, core.kernel_view()),
{
}

/// When every read event recorded in `cap.frontier().readers` was issued on
/// `ticket.timeline_id()` at sequence `<= ticket.sequence()`, `returned_reads` holds.
pub proof fn lemma_returned_reads_by_ticket(
    core: &CacheCore,
    frame: FrameId,
    cap: ShareCap,
    ticket: Ticket,
)
    requires
        core.well_formed(),
        frame.0 < core.frames_spec().metas_spec().len(),
        cap.well_formed(),
        cap.instance_id() == core.instance_id(),
        cap.loc() == core.frames_spec().frame(frame).slot.loc,
        ticket != Ticket::NONE,
        core.timelines_spec().valid_ticket(ticket),
        forall|e: TransferEvent| #[trigger]
            cap.frontier().readers.contains(e) ==> core.citations().contains(e) || exists|s: u64|
                1 <= s <= ticket.sequence() && #[trigger] core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == e,
    ensures
        core.frames_spec().returned_reads(frame, cap, ticket, core.kernel_view()),
{
}

/// Any newly submitted event from a fresh `EventSourceToken` is absent from `ledger`.
pub proof fn lemma_fresh_source_event_absent(source: EventSourceToken, ledger: KnowledgeLedger)
    requires
        source_fresh_for_ledger(source, ledger),
    ensures
        !ledger.submitted.dom().contains(source.next_event()),
{
    let ev = source.next_event();
    if ledger.submitted.dom().contains(ev) {
        assert(ev.seq < source.next_seq());
    }
}

/// Preserve `source_fresh_for_ledger` after inserting `ev = old_source.next_event()`
/// and advancing `source` by one step.
pub proof fn lemma_source_fresh_stepped(
    old_source: EventSourceToken,
    new_source: EventSourceToken,
    old_ledger: KnowledgeLedger,
    new_ledger: KnowledgeLedger,
    ev: TransferEvent,
)
    requires
        source_fresh_for_ledger(old_source, old_ledger),
        new_source.source() == old_source.source(),
        new_source.node() == old_source.node(),
        new_source.next_seq() == old_source.next_seq() + 1,
        ev == old_source.next_event(),
        new_ledger.submitted.dom() == old_ledger.submitted.dom().insert(ev),
    ensures
        source_fresh_for_ledger(new_source, new_ledger),
{
    assert forall|e: TransferEvent| #[trigger]
        new_ledger.submitted.dom().contains(e) && e.source == new_source.source() && e.observer
            == new_source.node() implies e.seq < new_source.next_seq() by {
        if e != ev {
            assert(old_ledger.submitted.dom().contains(e));
        }
    }
}

/// Extract the few facts backends need from `CacheCore::well_formed()` so backend
/// exec methods can `hide(CacheCore::well_formed)` for fast CVC5/Z3 proofs.
pub proof fn lemma_core_wf_facts(core: &CacheCore)
    requires
        core.well_formed(),
    ensures
        complete_receipts(core.ledger_spec()@),
        core.ledger_spec()@.valid(core.instance_id(), core.node()),
        core.ledger_spec()@.knows.instance_id() == core.instance_id(),
        core.timelines_spec().bounds(),
        core.frames_spec().shape(),
        core.tiers_spec().len() == core.frames_spec().num_tiers_spec(),
        core.frames_spec().frame_size_spec() > 0,
        core.frames_spec().frame_size_spec() == core.spec_spec()@.frame_size,
        core.frames_spec().versions_spec()@ == core.spec_spec()@.versions,
        core.version_seq_start_spec() <= core.next_version_seq_spec()
            <= core.version_seq_limit_spec(),
        core.frames_spec().metas_spec().len() <= u32::MAX,
        core.active_leases_spec().len() <= u32::MAX,
{
}

/// Extract `FrameTable::separated` only where pairwise frame-slot distinctness is needed.
pub proof fn lemma_core_frames_separated(core: &CacheCore)
    requires
        core.well_formed(),
    ensures
        core.frames_spec().separated(),
{
}

/// Extract `TimelineTable::well_formed` from `CacheCore::well_formed` without
/// unfolding the full core invariant in the caller.
pub proof fn lemma_core_timelines_wf(core: &CacheCore)
    requires
        core.well_formed(),
    ensures
        core.timelines_spec().well_formed(core.instance_id(), core.node(), &core.ledger_spec()@),
{
}

/// Extract slot & certificate shape for a published frame without unfolding
/// `CacheCore::well_formed()` in the caller.
pub proof fn lemma_published_frame_slot_valid(core: &CacheCore, f: FrameId)
    requires
        core.well_formed(),
        f.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(f).state == STATE_PUBLISHED,
    ensures
        core.frames_spec().frame(f).tier.valid(core.tiers_spec().len()),
        core.frames_spec().frame(f).slot.is_full(),
        core.frames_spec().frame(f).slot.well_formed(),
        core.frames_spec().frame(f).slot.slot_len == core.frames_spec().tier_frame_size(
            core.frames_spec().frame(f).tier,
        ),
        core.frames_spec().frame(f).bound_version is Some,
        core.frames_spec().frame(f).bound_version.unwrap().well_formed(),
        core.frames_spec().frame(f).bound_version.unwrap() != VersionId::NONE,
        core.frames_spec().frame(f).bound_version.unwrap().origin_node != core.node_id_spec()
            || core.frames_spec().frame(f).bound_version.unwrap().seq < core.next_version_seq_spec()
            || core.frames_spec().frame(f).bound_version.unwrap().seq
            >= core.version_seq_limit_spec(),
        core.frames_spec().frame(f).cert is Some,
        core.frames_spec().frame(f).cert.unwrap().slot == core.frames_spec().frame(f).slot,
        core.frames_spec().frame(f).cert.unwrap().bytes.len() == core.frames_spec().frame(
            f,
        ).slot.sub_len,
{
    assert(core.frames_spec().valid_frame(f, core.kernel_view()));
}

/// Extract `replica_valid` and version-byte agreement for a published frame
/// without unfolding all of `CacheCore::well_formed()` in callers (M9).
pub proof fn lemma_published_frame_replica_valid(core: &CacheCore, f: FrameId)
    requires
        core.well_formed(),
        f.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(f).state == STATE_PUBLISHED,
    ensures
        core.frames_spec().frame(f).bound_version is Some,
        core.frames_spec().frame(f).cert is Some,
        core.frames_spec().versions_spec()@.contains_key(
            core.frames_spec().frame(f).bound_version.unwrap(),
        ),
        core.frames_spec().versions_spec()@[core.frames_spec().frame(f).bound_version.unwrap()]
            == core.frames_spec().frame(f).cert.unwrap().bytes,
        core.ledger_spec()@.citations(core.node()).contains(
            core.frames_spec().frame(f).cert.unwrap().writer,
        ),
        replica_valid(core.ledger_spec()@, core.node(), core.frames_spec().frame(f).cert.unwrap()),
{
}

/// Extract slot shape for an initializing frame without unfolding `CacheCore::well_formed()`.
pub proof fn lemma_initializing_frame_slot_valid(core: &CacheCore, f: FrameId)
    requires
        core.well_formed(),
        f.0 < core.frames_spec().metas_spec().len(),
    ensures
        core.frames_spec().frame(f).tier.valid(core.tiers_spec().len()),
        core.frames_spec().frame(f).slot.is_full(),
        core.frames_spec().frame(f).slot.well_formed(),
        core.frames_spec().frame(f).slot.slot_len == core.frames_spec().tier_frame_size(
            core.frames_spec().frame(f).tier,
        ),
        core.frames_spec().frame(f).slot.loc.node == core.node(),
{
    assert(core.frames_spec().valid_frame(f, core.kernel_view()));
}

/// Show that after inserting `ev` into `submitted` and unioning `knows` containing
/// `(node, ev)`, `ev` is cited and `after.submitted.dom() == before.submitted.dom().insert(ev)`.
pub proof fn lemma_observed_event_in_citations(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    ev: TransferEvent,
    val: foundation::tracked::EventInfo<TransferDomain>,
    knows: Knows,
    preds: ISet<TransferEvent>,
)
    requires
        after.submitted.map() == before.submitted.map().insert(ev, val),
        after.knows.set() == before.knows.set().union(knows.set()),
        knows.set() == knowledge_of(node, preds.insert(ev)),
    ensures
        after.submitted.dom() == before.submitted.dom().insert(ev),
        after.submitted.map()[ev] == val,
        after.citations(node).contains(ev),
        before.citations(node).subset_of(after.citations(node)),
        ISet::<TransferEvent>::empty().is_empty(),
{
    assert(preds.insert(ev).contains(ev));
    assert(knows.set().contains((node, ev)));
    assert(after.knows.set().contains((node, ev)));
    assert(after.submitted.dom() =~= before.submitted.dom().insert(ev));
}

/// Prove `write_access(slot).writes` is disjoint from published frames for an
/// initializing frame `target`, without exposing `ISet` lemmas to the caller.
pub proof fn lemma_write_disjoint_from_published(
    core: &CacheCore,
    target: FrameId,
    writes: ISet<SlotLoc>,
)
    requires
        core.well_formed(),
        target.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(target).state != STATE_PUBLISHED,
        writes == ISet::empty().insert(core.frames_spec().frame(target).slot.loc),
    ensures
        core.disjoint_from_published(writes),
{
    core.lemma_non_published_write_disjoint(target, writes);
}

/// Certify a single full-slot copy event with minimal proof context.
pub proof fn lemma_certify_single_copy(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    src_cert: ValueCertificate,
    src_slot: SlotRef,
    dst_slot: SlotRef,
    writer: TransferEvent,
) -> (cert: ValueCertificate)
    requires
        before.valid(after.submitted.instance_id(), node),
        complete_receipts(before),
        !before.submitted.dom().contains(writer),
        before.submitted.map().submap_of(after.submitted.map()),
        revoked_submap(before.revoked, after.revoked),
        after.submitted.dom() == before.submitted.dom().insert(writer),
        before.succeeded.dom().subset_of(after.succeeded.dom()),
        before.citations(node).subset_of(after.citations(node)),
        replica_valid(before, node, src_cert),
        src_cert.slot == src_slot,
        src_slot.well_formed(),
        src_slot.is_full(),
        src_cert.bytes.len() == src_slot.sub_len,
        dst_slot.well_formed(),
        dst_slot.is_full(),
        src_slot.sub_len == dst_slot.sub_len,
        after.submitted.map()[writer].access.reads.contains(src_slot.loc),
        after.submitted.map()[writer].access.writes.contains(dst_slot.loc),
        after.submitted.map()[writer].effect == copy_effect(src_slot, dst_slot),
        after.submitted.map()[writer].preds.subset_of(before.submitted.dom()),
        before.citations(node).subset_of(after.submitted.map()[writer].preds),
        after.citations(node).contains(writer),
        event_succeeded(after, node, writer),
    ensures
        cert == copied_slice_certificate(src_cert, src_slot, writer, dst_slot),
        replica_valid(after, node, cert),
        cert.slot == dst_slot,
        cert.writer == writer,
        cert.bytes == src_cert.bytes,
{
    let copy = ReplicaCopy {
        before,
        source: src_cert,
        source_slot: src_slot,
        writer,
        destination: dst_slot,
    };
    assert(source_slice_valid(src_cert, src_slot));
    assert(effect_implies(after.submitted.map()[writer].effect, copy_effect(src_slot, dst_slot)));
    assert(copy.retained(after, node));
    let cert = copy.certify(after, node);
    assert(source_slice_bytes(src_cert, src_slot) =~= src_cert.bytes);
    cert
}

/// Transitivity of `ServiceSpec::versions_monotonic` without broadcasting map
/// lemmas into exec-heavy backend modules.
pub proof fn lemma_versions_monotonic_trans(
    a: crate::spec::ServiceSpec,
    b: crate::spec::ServiceSpec,
    c: crate::spec::ServiceSpec,
)
    requires
        a.versions_monotonic(b),
        b.versions_monotonic(c),
    ensures
        a.versions_monotonic(c),
{
    vstd::map_lib::lemma_submap_of_trans(a.versions, b.versions, c.versions);
}

} // verus!
