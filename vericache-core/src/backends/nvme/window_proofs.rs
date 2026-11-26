use super::*;
use foundation::tracked::knowledge_of;
use transfer::nvme::IoUring;
use vstd::prelude::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

pub(super) proof fn lemma_nvme_init_state(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pending_transfers: Seq<Option<InFlightNvmeOp>>,
    ring: IoUring,
    core: &CacheCore,
    timeline: TimelineId,
)
    requires
        bitmap.well_formed(),
        forall|i: int| !#[trigger] bitmap.bit_set(i),
        ring.pending() == Map::empty(),
        timeline < core.timelines_spec().count_spec() <= MAX_TIMELINES,
        core.timelines_spec().next_seq_spec()[timeline as int] == 1,
        core.timelines_spec().confirmed_seq_spec()[timeline as int] == 0,
        pending_transfers.len() == BITMAP_CAPACITY,
        forall|k: int| 0 <= k < BITMAP_CAPACITY ==> (#[trigger] in_flight_ops[k]).is_none(),
        forall|k: int| 0 <= k < BITMAP_CAPACITY ==> (#[trigger] pending_transfers[k]).is_none(),
    ensures
        nvme_window_wf(bitmap, in_flight_ops, ring, core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, ring, core, timeline),
{
    assert(ring_event_map(ring) =~= Map::empty());
    reveal(nvme_window_wf);
    reveal(nvme_pending_disjoint);
    reveal(nvme_pending_valid);
}

pub(super) proof fn lemma_nvme_after_submit(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pending_transfers: Seq<Option<InFlightNvmeOp>>,
    old_ring: IoUring,
    new_ring: IoUring,
    core: &CacheCore,
    timeline: TimelineId,
)
    requires
        nvme_window_wf(bitmap, in_flight_ops, old_ring, core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, old_ring, core, timeline),
        new_ring.pending() == old_ring.pending(),
    ensures
        nvme_window_wf(bitmap, in_flight_ops, new_ring, core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, new_ring, core, timeline),
{
    assert(ring_event_map(new_ring) =~= ring_event_map(old_ring));
    reveal(nvme_window_wf);
    reveal(nvme_pending_valid);
}

pub(super) proof fn lemma_nvme_state_after_hop_rollback(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pending_transfers: Seq<Option<InFlightNvmeOp>>,
    ring: IoUring,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
    aborted_dst: Option<FrameId>,
    aborted_lease_slot: Option<u32>,
)
    requires
        nvme_window_wf(bitmap, in_flight_ops, ring, before_core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, ring, before_core, timeline),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.ledger_spec() == before_core.ledger_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.timelines_spec().only_watermarks(&before_core.timelines_spec()),
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().preserves_initializing(&before_core.frames_spec(), aborted_dst),
        aborted_dst matches Some(d) ==> before_core.frames_spec().meta(d).state
            != STATE_INITIALIZING,
        aborted_lease_slot matches Some(sl) ==> (sl as usize)
            < before_core.active_leases_spec().len()
            && before_core.active_leases_spec()[sl as int].is_none(),
        after_core.active_leases_spec().len() == before_core.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before_core.active_leases_spec().len() && aborted_lease_slot != Some(s as u32)
                ==> #[trigger] after_core.active_leases_spec()[s]
                == before_core.active_leases_spec()[s],
    ensures
        nvme_window_wf(bitmap, in_flight_ops, ring, after_core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, ring, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    reveal(nvme_window_wf);
    reveal(nvme_pending_valid);
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (
    #[trigger] pending_transfers[i] matches Some(op) ==> {
        &&& op.valid(after_core, timeline, i)
        &&& ring.pending().contains_key(op.ticket.sequence())
        &&& ring.pending()[op.ticket.sequence()].event == op.copy@.writer
    }) by {
        if let Some(op) = pending_transfers[i] {
            assert(op.valid(before_core, timeline, i));
            assert(aborted_dst != Some(op.hop.dst_frame));
            assert(aborted_lease_slot != Some(op.hop.src_lease.slot_spec()));
            assert(after_core.frames_spec().meta(op.hop.dst_frame)
                == before_core.frames_spec().meta(op.hop.dst_frame));
            assert(after_core.frames_spec().frame(op.hop.dst_frame)
                == before_core.frames_spec().frame(op.hop.dst_frame));
        }
    }
}

pub(super) proof fn lemma_nvme_seq_not_pending(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    ring: IoUring,
    core: &CacheCore,
    timeline: TimelineId,
    seq: u64,
)
    requires
        nvme_window_wf(bitmap, in_flight_ops, ring, core, timeline),
        timeline < core.timelines_spec().count_spec() <= MAX_TIMELINES,
        seq == core.timelines_spec().next_seq_spec()[timeline as int],
    ensures
        !ring.pending().contains_key(seq),
{
    reveal(nvme_window_wf);
    if ring.pending().contains_key(seq) {
        assert(ring_event_map(ring).contains_key(seq));
    }
}

pub(super) proof fn lemma_nvme_stage_unfold_window(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pending_transfers: Seq<Option<InFlightNvmeOp>>,
    ring: IoUring,
    core: &CacheCore,
    timeline: TimelineId,
    seq: u64,
)
    requires
        nvme_window_wf(bitmap, in_flight_ops, ring, core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, ring, core, timeline),
        timeline < core.timelines_spec().count_spec() <= MAX_TIMELINES,
        seq == core.timelines_spec().next_seq_spec()[timeline as int],
        seq <= core.timelines_spec().confirmed_seq_spec()[timeline as int] + BITMAP_CAPACITY,
    ensures
        bitmap.well_formed(),
        bitmap.observations(&core.timelines_spec(), timeline, core.node(), &core.ledger_spec()@),
        window_matches(
            in_flight_ops,
            bitmap,
            &core.timelines_spec(),
            timeline,
            ring_event_map(ring),
        ),
        !ring.pending().contains_key(seq),
        pending_transfers[(seq % BITMAP_CAPACITY) as int].is_none(),
{
    reveal(nvme_window_wf);
    reveal(nvme_pending_valid);
    if ring.pending().contains_key(seq) {
        assert(ring_event_map(ring).contains_key(seq));
    }
    let idx = (seq % BITMAP_CAPACITY) as int;
    if let Some(p) = in_flight_ops[idx] {
        assert(ring_event_map(ring).contains_key(p.sequence));
        assert(core.timelines_spec().confirmed_seq_spec()[timeline as int] < p.sequence < seq);
        assert(p.sequence % BITMAP_CAPACITY != seq % BITMAP_CAPACITY) by {
            assert(p.sequence == BITMAP_CAPACITY * (p.sequence / BITMAP_CAPACITY) + p.sequence
                % BITMAP_CAPACITY);
            assert(seq == BITMAP_CAPACITY * (seq / BITMAP_CAPACITY) + seq % BITMAP_CAPACITY);
        }
    }
}

pub(super) proof fn lemma_nvme_pack_window(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    ring: IoUring,
    core: &CacheCore,
    timeline: TimelineId,
)
    requires
        bitmap.well_formed(),
        bitmap.observations(&core.timelines_spec(), timeline, core.node(), &core.ledger_spec()@),
        window_matches(
            in_flight_ops,
            bitmap,
            &core.timelines_spec(),
            timeline,
            ring_event_map(ring),
        ),
    ensures
        nvme_window_wf(bitmap, in_flight_ops, ring, core, timeline),
{
    reveal(nvme_window_wf);
}

pub(super) proof fn lemma_nvme_poll_unfold(
    bitmap: SlidingBitmap,
    in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pending_transfers: Seq<Option<InFlightNvmeOp>>,
    before_ring: IoUring,
    after_ring: IoUring,
    core: &CacheCore,
    timeline: TimelineId,
    seq: u64,
    ev: TransferEvent,
    knows: Knows,
)
    requires
        core.well_formed(),
        nvme_window_wf(bitmap, in_flight_ops, before_ring, core, timeline),
        nvme_pending_valid(pending_transfers, in_flight_ops, before_ring, core, timeline),
        timeline < core.timelines_spec().count_spec() <= MAX_TIMELINES,
        before_ring.pending().contains_key(seq),
        before_ring.pending()[seq].event == ev,
        after_ring.pending() == before_ring.pending().remove(seq),
        knows.set() == knowledge_of(core.node(), before_ring.pending()[seq].preds.insert(ev)),
    ensures
        bitmap.well_formed(),
        bitmap.observations(&core.timelines_spec(), timeline, core.node(), &core.ledger_spec()@),
        window_matches(
            in_flight_ops,
            bitmap,
            &core.timelines_spec(),
            timeline,
            ring_event_map(before_ring),
        ),
        core.timelines_spec().well_formed(core.instance_id(), core.node(), &core.ledger_spec()@),
        ring_event_map(before_ring).contains_key(seq),
        ring_event_map(after_ring) == ring_event_map(before_ring).remove(seq),
        core.timelines_spec().issued(timeline, seq),
        core.timelines_spec().events_spec()@[(timeline, seq)] == ev,
        core.timelines_spec().observed(timeline, seq, core.node(), knows),
        in_flight_ops[(seq % BITMAP_CAPACITY) as int].is_some(),
        pending_transfers[(seq % BITMAP_CAPACITY) as int].is_some(),
        pending_transfers[(seq % BITMAP_CAPACITY) as int]->Some_0.valid(
            core,
            timeline,
            (seq % BITMAP_CAPACITY) as int,
        ),
        pending_transfers[(seq % BITMAP_CAPACITY) as int]->Some_0.ticket.sequence() == seq,
{
    reveal(nvme_window_wf);
    reveal(nvme_pending_valid);
    lemma_nvme_ring_pending_remove(before_ring, after_ring, seq);
    assert(ring_event_map(before_ring).contains_key(seq));
    assert(before_ring.pending()[seq].preds.insert(ev).contains(ev));
    assert(knows.set().contains((core.node(), ev)));
    let idx = (seq % BITMAP_CAPACITY) as int;
    let op = pending_transfers[idx]->Some_0;
    assert(op.valid(core, timeline, idx));
    assert(in_flight_ops[idx]->Some_0.sequence == seq);
    assert(core.timelines_spec().confirmed_seq_spec()[timeline as int] < op.ticket.sequence()
        < core.timelines_spec().next_seq_spec()[timeline as int]);
    assert(core.timelines_spec().confirmed_seq_spec()[timeline as int] < seq
        < core.timelines_spec().next_seq_spec()[timeline as int]);
    assert(op.ticket.sequence() == BITMAP_CAPACITY * (op.ticket.sequence() / BITMAP_CAPACITY)
        + op.ticket.sequence() % BITMAP_CAPACITY);
    assert(seq == BITMAP_CAPACITY * (seq / BITMAP_CAPACITY) + seq % BITMAP_CAPACITY);
}

} // verus!
