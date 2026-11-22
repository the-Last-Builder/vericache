use super::*;
use foundation::tracked::{closure_from, EventInfo, Frontier};
use transfer::cuda::CudaStream;
use vstd::prelude::*;

verus! {

/// Build a `PendingPutProof` for a single CUDA async copy (used by `put_gpu`
/// and `copy_hop_cuda`).
pub(super) proof fn lemma_cuda_single_put_proof(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    src_cert: ValueCertificate,
    src_slot: SlotRef,
    dst_slot: SlotRef,
    writer: TransferEvent,
    val: EventInfo<TransferDomain>,
) -> (p: PendingPutProof)
    requires
        before.valid(before.submitted.instance_id(), node),
        complete_receipts(before),
        complete_receipts(after),
        after.submitted.instance_id() == before.submitted.instance_id(),
        after.submitted.map() == before.submitted.map().insert(writer, val),
        before.submitted.map().submap_of(after.submitted.map()),
        after.knows == before.knows,
        after.succeeded == before.succeeded,
        after.revoked == before.revoked,
        !before.submitted.dom().contains(writer),
        replica_valid(before, node, src_cert),
        source_slice_valid(src_cert, src_slot),
        src_cert.slot == src_slot,
        src_cert.bytes.len() == src_slot.sub_len,
        dst_slot.is_full(),
        dst_slot.slot_len == src_slot.sub_len,
        src_slot.loc != dst_slot.loc,
        val.access == copy_access(src_slot, dst_slot),
        val.effect == copy_effect(src_slot, dst_slot),
        val.preds.subset_of(before.submitted.dom()),
        before.citations(node).subset_of(val.preds),
    ensures
        p.certificate.slot == dst_slot,
        p.certificate.writer == writer,
        p.certificate.bytes == src_cert.bytes,
        p.events == seq![writer],
        p.pending(after, node),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    let copy = ReplicaCopy {
        before,
        source: src_cert,
        source_slot: src_slot,
        writer,
        destination: dst_slot,
    };
    assert(after.submitted.dom().contains(writer));
    assert(after.submitted.map()[writer] == val);
    assert(before.citations(node).subset_of(after.citations(node)));
    assert(effect_implies(val.effect, copy_effect(src_slot, dst_slot)));
    assert forall|e: TransferEvent| #[trigger]
        after.submitted.dom().contains(e) && !before.submitted.dom().contains(e) && e
            != writer implies !after.submitted.map()[e].access.writes.contains(dst_slot.loc) by {
        assert(e == writer);
    }
    assert(copy.retained(after, node));
    let p = PendingPutProof::single(copy, after, node);
    assert(source_slice_bytes(src_cert, src_slot) =~= src_cert.bytes);
    p
}

/// Opaque wrapper for timeline event prefix preservation (prevents CVC5 quantifier blowup).
#[verifier::opaque]
pub(super) open spec fn cuda_timeline_events_preserved(
    before_events: Map<(u8, u64), TransferEvent>,
    after_events: Map<(u8, u64), TransferEvent>,
    timeline: u8,
    old_next: u64,
) -> bool {
    forall|s: u64|
        1 <= s < old_next ==> #[trigger] after_events[(timeline, s)] == before_events[(timeline, s)]
}

/// Opaque wrapper for disjoint-write ledger growth (prevents CVC5 `IMap::submap_of`
/// and `ISet::subset_of` quantifier blowup in callers).
#[verifier::opaque]
pub(super) open spec fn cuda_ledger_writes_only(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    disjoint_write_loc: Option<SlotLoc>,
) -> bool {
    &&& before.submitted.map().submap_of(after.submitted.map())
    &&& revoked_submap(before.revoked, after.revoked)
    &&& before.knows.set().subset_of(after.knows.set())
    &&& before.succeeded.dom().subset_of(after.succeeded.dom())
    &&& forall|e: TransferEvent| #[trigger]
        after.submitted.dom().contains(e) && !before.submitted.dom().contains(e)
            ==> match disjoint_write_loc {
            None => false,
            Some(loc) => after.submitted.map()[e].access.writes == ISet::empty().insert(loc),
        }
}

/// Pure constructors for `cuda_timeline_events_preserved` and `cuda_ledger_writes_only`
/// so callers never expand `IMap::submap_of`, `ISet::subset_of`, or `forall|s|`/`forall|e|`!
pub(super) proof fn lemma_cuda_events_preserved_same(
    events: Map<(u8, u64), TransferEvent>,
    timeline: u8,
    old_next: u64,
)
    ensures
        cuda_timeline_events_preserved(events, events, timeline, old_next),
{
    reveal(cuda_timeline_events_preserved);
}

pub(super) proof fn lemma_cuda_events_preserved_insert(
    before_events: Map<(u8, u64), TransferEvent>,
    after_events: Map<(u8, u64), TransferEvent>,
    timeline: u8,
    old_next: u64,
    ins_tl: u8,
    ins_seq: u64,
    ev: TransferEvent,
)
    requires
        after_events == before_events.insert((ins_tl, ins_seq), ev),
        ins_tl != timeline || ins_seq >= old_next,
    ensures
        cuda_timeline_events_preserved(before_events, after_events, timeline, old_next),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(cuda_timeline_events_preserved);
}

pub(super) proof fn lemma_cuda_events_preserved_step(
    before_events: Map<(u8, u64), TransferEvent>,
    mid_events: Map<(u8, u64), TransferEvent>,
    after_events: Map<(u8, u64), TransferEvent>,
    timeline: u8,
    old_next: u64,
    ins_tl: u8,
    ins_seq: u64,
    ev: TransferEvent,
)
    requires
        cuda_timeline_events_preserved(before_events, mid_events, timeline, old_next),
        after_events == mid_events.insert((ins_tl, ins_seq), ev),
        ins_tl != timeline || ins_seq >= old_next,
    ensures
        cuda_timeline_events_preserved(before_events, after_events, timeline, old_next),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(cuda_timeline_events_preserved);
}

pub(super) proof fn lemma_cuda_ledger_writes_only_same(
    ledger: KnowledgeLedger,
    loc: Option<SlotLoc>,
)
    ensures
        cuda_ledger_writes_only(ledger, ledger, loc),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    reveal(cuda_ledger_writes_only);
}

pub(super) proof fn lemma_cuda_ledger_writes_only_insert(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    ev: TransferEvent,
    val: EventInfo<TransferDomain>,
    loc: SlotLoc,
)
    requires
        after.submitted.map() == before.submitted.map().insert(ev, val),
        before.submitted.map().submap_of(after.submitted.map()),
        after.knows == before.knows,
        after.succeeded == before.succeeded,
        after.revoked == before.revoked,
        val.access.writes == ISet::empty().insert(loc),
    ensures
        cuda_ledger_writes_only(before, after, Some(loc)),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    reveal(cuda_ledger_writes_only);
}

/// Pure timeline-event coverage preservation for a `PendingPutProof` (no `CacheCore` in scope).
proof fn lemma_put_events_covered_after_step(
    events: Seq<TransferEvent>,
    before_events: Map<(u8, u64), TransferEvent>,
    after_events: Map<(u8, u64), TransferEvent>,
    timeline: u8,
    ticket_seq: u64,
    old_next: u64,
)
    requires
        ticket_seq < old_next,
        cuda_timeline_events_preserved(before_events, after_events, timeline, old_next),
        cuda_put_events_covered(events, before_events, timeline, ticket_seq),
    ensures
        cuda_put_events_covered(events, after_events, timeline, ticket_seq),
{
    reveal(cuda_timeline_events_preserved);
    reveal(cuda_put_events_covered);
    assert forall|p: int| 0 <= p < events.len() implies cuda_event_at_prefix(
        #[trigger] events[p],
        after_events,
        timeline,
        ticket_seq,
    ) by {
        assert(cuda_event_at_prefix(events[p], before_events, timeline, ticket_seq));
        let q = choose|q: u64|
            1 <= q <= ticket_seq && (#[trigger] before_events[(timeline, q)]) == events[p]
                && events[p].source.succeeds_on_observe();
        assert(1 <= q < old_next);
        assert(after_events[(timeline, q)] == before_events[(timeline, q)]);
    }
}

/// Pure helper proving `cuda_put_events_covered` for a 1-event `PendingPutProof`.
pub(super) proof fn lemma_cuda_single_event_covered(
    events: Seq<TransferEvent>,
    ev_map: Map<(u8, u64), TransferEvent>,
    timeline: u8,
    seq: u64,
    ev: TransferEvent,
)
    requires
        events == seq![ev],
        1 <= seq,
        ev_map[(timeline, seq)] == ev,
        ev.source.succeeds_on_observe(),
    ensures
        cuda_put_events_covered(events, ev_map, timeline, seq),
{
    reveal(cuda_put_events_covered);
    assert forall|p: int| 0 <= p < events.len() implies cuda_event_at_prefix(
        #[trigger] events[p],
        ev_map,
        timeline,
        seq,
    ) by {
        assert(p == 0);
        assert(events[0] == ev);
        assert(1 <= seq <= seq && ev_map[(timeline, seq)] == events[p]
            && events[p].source.succeeds_on_observe());
    }
}

/// Pure helper proving `timelines.ticket_covers(ticket, ev)` when `ev` is at `ticket.sequence()`.
pub(super) proof fn lemma_cuda_ticket_covers_exact(
    timelines: &TimelineTable,
    ticket: Ticket,
    ev: TransferEvent,
)
    requires
        ticket != Ticket::NONE,
        1 <= ticket.sequence(),
        timelines.events@[(ticket.timeline_id(), ticket.sequence())] == ev,
    ensures
        timelines.ticket_covers(ticket, ev),
{
    let s = ticket.sequence();
    assert(1 <= s <= ticket.sequence() && timelines.events@[(ticket.timeline_id(), s)] == ev);
}

/// Pure abort-writer coverage preservation (no `CacheCore` in scope).
proof fn lemma_abort_writer_covered_after_step(
    w: TransferEvent,
    before_table: &TimelineTable,
    after_table: &TimelineTable,
    ticket: Ticket,
)
    requires
        ticket != Ticket::NONE,
        before_table.count <= MAX_TIMELINES,
        ticket.timeline_id() < before_table.count,
        after_table.count == before_table.count,
        ticket.sequence() < before_table.next_seq[ticket.timeline_id() as int],
        cuda_timeline_events_preserved(
            before_table.events@,
            after_table.events@,
            ticket.timeline_id(),
            before_table.next_seq[ticket.timeline_id() as int],
        ),
        before_table.ticket_covers(ticket, w),
    ensures
        after_table.ticket_covers(ticket, w),
{
    reveal(cuda_timeline_events_preserved);
    let old_next = before_table.next_seq[ticket.timeline_id() as int];
    assert forall|q: u64|
        1 <= q <= ticket.sequence() && #[trigger] before_table.events@[(ticket.timeline_id(), q)]
            == w implies after_table.events@[(ticket.timeline_id(), q)] == w by {
        assert(1 <= q < old_next);
        assert(after_table.events@[(ticket.timeline_id(), q)] == before_table.events@[(
            ticket.timeline_id(),
            q,
        )]);
    }
}

/// Preserve both GPU and optional pinned-host slab bindings across any
/// `FrameTable::same_directory` step (eliminates `if let Some(h)` branching in callers).
pub(super) proof fn lemma_cuda_slabs_same_directory<'a>(
    backend: &CudaBackend<'a>,
    before: &CacheCore,
    after: &CacheCore,
)
    requires
        slab_bound_to_core(
            before,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
        ),
        backend.pinned_host matches Some(h) ==> h.well_formed(before) && h.tier != backend.gpu_tier,
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.tiers == before.tiers,
        after.frames.same_directory(&before.frames),
    ensures
        slab_bound_to_core(
            after,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
        ),
        backend.pinned_host matches Some(h) ==> h.well_formed(after) && h.tier != backend.gpu_tier,
        after.frames.frame_size == before.frames.frame_size,
        after.frames.metas.len() == before.frames.metas.len(),
{
    lemma_slab_bound_same_directory(
        before,
        after,
        backend.gpu_tier,
        backend.gpu_first_frame,
        backend.gpu_num_frames,
        backend.gpu_buffer.space(),
    );
    if let Some(h) = backend.pinned_host {
        lemma_slab_bound_same_directory(
            before,
            after,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
}

/// Preserve both GPU and optional pinned-host slab bindings when `frames` is unchanged.
pub(super) proof fn lemma_cuda_slabs_same_frames<'a>(
    backend: &CudaBackend<'a>,
    before: &CacheCore,
    after: &CacheCore,
)
    requires
        slab_bound_to_core(
            before,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
        ),
        backend.pinned_host matches Some(h) ==> h.well_formed(before) && h.tier != backend.gpu_tier,
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.tiers == before.tiers,
        after.frames == before.frames,
    ensures
        slab_bound_to_core(
            after,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
        ),
        backend.pinned_host matches Some(h) ==> h.well_formed(after) && h.tier != backend.gpu_tier,
{
    lemma_slab_bound_same_frames(
        before,
        after,
        backend.gpu_tier,
        backend.gpu_first_frame,
        backend.gpu_num_frames,
        backend.gpu_buffer.space(),
    );
    if let Some(h) = backend.pinned_host {
        lemma_slab_bound_same_frames(
            before,
            after,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
}

/// Pure `ISet::subset_of` transitivity without dragging `CacheCore` into `iset` broadcasts.
proof fn lemma_iset_subset_trans(
    a: ISet<TransferEvent>,
    b: ISet<TransferEvent>,
    c: ISet<TransferEvent>,
)
    requires
        a.subset_of(b),
        b.subset_of(c),
    ensures
        a.subset_of(c),
{
    broadcast use vstd::iset::group_iset_lemmas;

}

/// Pure ledger preservation wrapper using `cuda_ledger_writes_only`.
proof fn lemma_ledger_preserves_dest_opaque(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    loc: SlotLoc,
    disjoint_write_loc: Option<SlotLoc>,
)
    requires
        before.submitted.instance_id() == after.submitted.instance_id(),
        cuda_ledger_writes_only(before, after, disjoint_write_loc),
        disjoint_write_loc != Some(loc),
    ensures
        before.citations(node).subset_of(after.citations(node)),
        preserves_destination(before, after, node, loc),
{
    reveal(cuda_ledger_writes_only);
    lemma_ledger_preserves_dest_disjoint(before, after, node, loc, disjoint_write_loc);
}

/// Preserve all existing `PendingGpuWrite` entries when `core` runs a non-write
/// or disjoint write (such as `begin_write`, `abort_write`, `begin_copy_hop`,
/// `acquire_local_lease`, `release_local_lease`, or `enqueue_memcpy` on another
/// frame).
proof fn lemma_cuda_put_valid_after_disjoint_step(
    entry: &PendingGpuWrite,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: u8,
    new_frame: Option<FrameId>,
    disjoint_write_loc: Option<SlotLoc>,
)
    requires
        entry.valid(before_core, timeline),
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.next_version_seq >= before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.next_seq[timeline as int]
            >= before_core.timelines.next_seq[timeline as int],
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            after_core.timelines.events@,
            timeline,
            before_core.timelines.next_seq[timeline as int],
        ),
        cuda_ledger_writes_only(before_core.ledger@, after_core.ledger@, disjoint_write_loc),
        disjoint_write_loc != Some(entry.slot@.loc),
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.preserves_initializing(&before_core.frames, new_frame),
        new_frame != Some(entry.frame),
    ensures
        entry.valid(after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(FrameTable::same_directory);
    hide(FrameTable::preserves_initializing);
    hide(TimelineTable::ticket_covers);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(after_core);
    assert(after_core.frames.metas.len() == before_core.frames.metas.len()
        && after_core.frames.versions@ == before_core.frames.versions@ && after_core.frames.meta(
        entry.frame,
    ) == before_core.frames.meta(entry.frame) && after_core.frames.frame(entry.frame)
        == before_core.frames.frame(entry.frame)) by {
        reveal(FrameTable::same_directory);
        reveal(FrameTable::preserves_initializing);
    }
    lemma_ledger_preserves_dest_opaque(
        before_core.ledger@,
        after_core.ledger@,
        after_core.node(),
        entry.slot@.loc,
        disjoint_write_loc,
    );
    let tok = after_core.frames.frame(entry.frame).token.unwrap();
    lemma_iset_subset_trans(
        slot_frontier(tok).readers,
        before_core.citations(),
        after_core.citations(),
    );
    assert(after_core.timelines.issued(timeline, entry.ticket.sequence()));
    assert(after_core.timelines.events@[(timeline, entry.ticket.sequence())]
        == before_core.timelines.events@[(timeline, entry.ticket.sequence())]) by {
        reveal(cuda_timeline_events_preserved);
    }
    match entry.action {
        PendingGpuAction::Abort => {
            if let Some(w) = slot_frontier(tok).last_writer {
                if !before_core.citations().contains(w) {
                    lemma_abort_writer_covered_after_step(
                        w,
                        &before_core.timelines,
                        &after_core.timelines,
                        entry.ticket,
                    );
                }
            }
        },
        PendingGpuAction::Publish { proof, .. } => {
            proof@.preserve(before_core.ledger@, after_core.ledger@, after_core.node());
            lemma_put_events_covered_after_step(
                proof@.events,
                before_core.timelines.events@,
                after_core.timelines.events@,
                timeline,
                entry.ticket.sequence(),
                before_core.timelines.next_seq[timeline as int],
            );
        },
    }
}

/// Preserve `cuda_puts_valid` when `core` takes a step that does not touch any
/// existing pending put's frame or slot location.
pub(super) proof fn lemma_cuda_puts_after_disjoint_step(
    puts: Seq<PendingGpuWrite>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: u8,
    touched_frame: Option<FrameId>,
    disjoint_write_loc: Option<SlotLoc>,
)
    requires
        cuda_puts_valid(puts, before_core, timeline),
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.next_version_seq >= before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.next_seq[timeline as int]
            >= before_core.timelines.next_seq[timeline as int],
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            after_core.timelines.events@,
            timeline,
            before_core.timelines.next_seq[timeline as int],
        ),
        cuda_ledger_writes_only(before_core.ledger@, after_core.ledger@, disjoint_write_loc),
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.preserves_initializing(&before_core.frames, touched_frame),
        touched_frame matches Some(f) ==> f.0 < before_core.frames.metas.len()
            && before_core.frames.meta(f).state != STATE_INITIALIZING,
        disjoint_write_loc matches Some(loc) ==> forall|f: FrameId|
            f.0 < before_core.frames.metas.len() && (#[trigger] before_core.frames.meta(f)).state
                == STATE_INITIALIZING ==> loc != before_core.frames.frame(f).slot.loc,
    ensures
        cuda_puts_valid(puts, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(PendingGpuWrite::valid);
    lemma_core_wf_facts(before_core);
    reveal(cuda_puts_valid);
    assert forall|i: int| 0 <= i < puts.len() implies (#[trigger] puts[i]).valid(
        after_core,
        timeline,
    ) by {
        assert(puts[i].valid(before_core, timeline));
        assert(touched_frame != Some(puts[i].frame)) by {
            reveal(PendingGpuWrite::valid);
        }
        assert(disjoint_write_loc != Some(puts[i].slot@.loc)) by {
            reveal(PendingGpuWrite::valid);
        }
        lemma_cuda_put_valid_after_disjoint_step(
            &puts[i],
            before_core,
            after_core,
            timeline,
            touched_frame,
            disjoint_write_loc,
        );
    }
}

/// Push a newly parked `PendingGpuWrite` onto `active_gpu_puts`, preserving
/// `cuda_puts_valid` and `cuda_puts_disjoint`.
pub(super) proof fn lemma_cuda_puts_push(
    old_puts: Seq<PendingGpuWrite>,
    new_puts: Seq<PendingGpuWrite>,
    entry: PendingGpuWrite,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: u8,
)
    requires
        cuda_puts_valid(old_puts, before_core, timeline),
        before_core.well_formed(),
        after_core.well_formed(),
        new_puts == old_puts.push(entry),
        entry.valid(after_core, timeline),
        entry.frame.0 < before_core.frames.metas.len(),
        before_core.frames.meta(entry.frame).state != STATE_INITIALIZING,
        before_core.frames.frame(entry.frame).slot == entry.slot@,
        entry.action matches PendingGpuAction::Publish { version, only_if_current: false, .. }
            ==> version.origin_node == before_core.node_id && version.seq
            == before_core.next_version_seq && version.seq < before_core.version_seq_limit,
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.next_version_seq >= before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.next_seq[timeline as int]
            >= before_core.timelines.next_seq[timeline as int],
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            after_core.timelines.events@,
            timeline,
            before_core.timelines.next_seq[timeline as int],
        ),
        cuda_ledger_writes_only(before_core.ledger@, after_core.ledger@, Some(entry.slot@.loc)),
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.preserves_initializing(&before_core.frames, Some(entry.frame)),
    ensures
        cuda_puts_valid(new_puts, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(PendingGpuWrite::valid);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(after_core);
    lemma_core_frames_separated(before_core);
    assert forall|f: FrameId|
        f.0 < before_core.frames.metas.len() && (#[trigger] before_core.frames.meta(f)).state
            == STATE_INITIALIZING implies entry.slot@.loc != before_core.frames.frame(
        f,
    ).slot.loc by {
        assert(f != entry.frame);
        if f.0 < entry.frame.0 {
            assert(before_core.frames.ghosts@[f.0 as nat].slot.loc
                != before_core.frames.ghosts@[entry.frame.0 as nat].slot.loc);
        } else {
            assert(before_core.frames.ghosts@[entry.frame.0 as nat].slot.loc
                != before_core.frames.ghosts@[f.0 as nat].slot.loc);
        }
    }
    lemma_cuda_puts_after_disjoint_step(
        old_puts,
        before_core,
        after_core,
        timeline,
        Some(entry.frame),
        Some(entry.slot@.loc),
    );
    reveal(cuda_puts_valid);
    reveal(cuda_puts_disjoint);
    let n = old_puts.len();
    assert forall|i: int, j: int|
        0 <= i < new_puts.len() && 0 <= j < new_puts.len() && i != j implies {
        &&& (#[trigger] new_puts[i]).frame != (#[trigger] new_puts[j]).frame
        &&& new_puts[i].slot@.loc != new_puts[j].slot@.loc
        &&& match (new_puts[i].action, new_puts[j].action) {
            (
                PendingGpuAction::Publish { version: v1, only_if_current: false, .. },
                PendingGpuAction::Publish { version: v2, only_if_current: false, .. },
            ) => v1 != v2,
            _ => true,
        }
    } by {
        if i == n || j == n {
            let k = if i == n {
                j
            } else {
                i
            };
            assert(old_puts[k].valid(before_core, timeline));
            reveal(PendingGpuWrite::valid);
            assert(old_puts[k].frame != entry.frame);
            if old_puts[k].frame.0 < entry.frame.0 {
                assert(before_core.frames.ghosts@[old_puts[k].frame.0 as nat].slot.loc
                    != before_core.frames.ghosts@[entry.frame.0 as nat].slot.loc);
            } else {
                assert(before_core.frames.ghosts@[entry.frame.0 as nat].slot.loc
                    != before_core.frames.ghosts@[old_puts[k].frame.0 as nat].slot.loc);
            }
        }
    }
}

/// Preserve `CudaTimeline::well_formed` across non-timeline `CacheCore` steps
/// (writes, leases, copy-hop reservation, or external-timeline tickets).
pub(super) proof fn lemma_cuda_timeline_preserved(
    tl: &CudaTimeline,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        tl.well_formed(&before_core.timelines, before_core.node(), &before_core.ledger@),
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.kinds == before_core.timelines.kinds,
        after_core.timelines.next_seq[tl.timeline_id as int]
            == before_core.timelines.next_seq[tl.timeline_id as int],
        after_core.timelines.confirmed_seq[tl.timeline_id as int]
            >= before_core.timelines.confirmed_seq[tl.timeline_id as int],
        forall|s: u64|
            1 <= s < before_core.timelines.next_seq[tl.timeline_id as int]
                ==> #[trigger] after_core.timelines.events@[(tl.timeline_id, s)]
                == before_core.timelines.events@[(tl.timeline_id, s)],
        before_core.ledger@.submitted.map().submap_of(after_core.ledger@.submitted.map()),
        forall|e: TransferEvent| #[trigger]
            after_core.ledger@.submitted.dom().contains(e)
                && !before_core.ledger@.submitted.dom().contains(e) ==> e.source
                != tl.stream.source().source() || e.observer != tl.stream.source().node(),
    ensures
        tl.well_formed(&after_core.timelines, after_core.node(), &after_core.ledger@),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(before_core);
    let t = tl.timeline_id;
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before_core.ledger@.submitted.map(),
        after_core.ledger@.submitted.map(),
    );
    assert(source_fresh_for_ledger(tl.stream.source(), after_core.ledger@)) by {
        assert forall|e: TransferEvent| #[trigger]
            after_core.ledger@.submitted.dom().contains(e) && e.source
                == tl.stream.source().source() && e.observer
                == tl.stream.source().node() implies e.seq < tl.stream.source().next_seq() by {
            assert(before_core.ledger@.submitted.dom().contains(e));
        }
    }
    reveal(timeline::event_ring_valid);
    assert forall|j: int| 0 <= j < tl.event_ring.len() implies #[trigger] tl.event_ring[j].valid(
        &after_core.timelines,
        t,
        tl.device,
        after_core.node(),
        &after_core.ledger@,
    ) by {
        assert(tl.event_ring[j].valid(
            &before_core.timelines,
            t,
            tl.device,
            before_core.node(),
            &before_core.ledger@,
        ));
        vstd::imap_lib::lemma_submap_of_trans(
            tl.event_ring[j].event.infos().map(),
            before_core.ledger@.submitted.map(),
            after_core.ledger@.submitted.map(),
        );
        assert forall|q: u64| 1 <= q <= tl.event_ring[j].recorded_seq implies closure_from(
            tl.event_ring[j].event.infos().map(),
            tl.event_ring[j].event.recorded(),
        ).contains(#[trigger] after_core.timelines.events@[(t, q)]) by {
            assert(q < before_core.timelines.next_seq[t as int]);
            assert(after_core.timelines.events@[(t, q)] == before_core.timelines.events@[(t, q)]);
        }
    }
    assert forall|q: u64|
        1 <= q <= (after_core.timelines.next_seq[t as int]
            - 1) as u64 implies tl.closure().contains(
        #[trigger] after_core.timelines.events@[(t, q)],
    ) by {
        assert(q < before_core.timelines.next_seq[t as int]);
        assert(after_core.timelines.events@[(t, q)] == before_core.timelines.events@[(t, q)]);
    }
}

/// Helper proving `read_admitted` / `write_admitted` against a stream's
/// `closure_from(submitted, stream.tail().union(core.citations()))` whenever
/// admitted by `core.citations()`.
pub(super) proof fn lemma_citations_admit_stream_closure(
    core: &CacheCore,
    stream: CudaStream,
    fr: Frontier<TransferDomain>,
)
    requires
        core.well_formed(),
        stream.infos().map().submap_of(core.ledger@.submitted.map()),
        stream.tail().subset_of(stream.infos().dom()),
    ensures
        read_admitted(fr, core.citations()) ==> read_admitted(
            fr,
            closure_from(core.ledger@.submitted.map(), stream.tail().union(core.citations())),
        ),
        write_admitted(fr, core.citations()) ==> write_admitted(
            fr,
            closure_from(core.ledger@.submitted.map(), stream.tail().union(core.citations())),
        ),
{
    core.lemma_citations_admit_closure(fr, stream.tail());
}

} // verus!
