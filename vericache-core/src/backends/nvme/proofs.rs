use super::*;
use foundation::tracked::{knowledge_of, EventInfo, Frontier};
use transfer::nvme::{IoUring, DIRECT_IO_ALIGNMENT};
use vstd::prelude::*;

verus! {

pub(super) proof fn lemma_aligned_host_slot(addr: usize, align: usize, idx: usize, slot_len: usize)
    requires
        align > 0,
        align % (DIRECT_IO_ALIGNMENT as usize) == 0,
        addr % align == 0,
        slot_len % (DIRECT_IO_ALIGNMENT as usize) == 0,
        addr + idx * slot_len <= usize::MAX,
    ensures
        ((addr + idx * slot_len) as usize) % (DIRECT_IO_ALIGNMENT as usize) == 0,
{
    let d = DIRECT_IO_ALIGNMENT as int;
    let a = align as int;
    let p = addr as int;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(p, a);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(a, d);
    assert(p == (p / a) * (a / d) * d) by (nonlinear_arith)
        requires
            p == a * (p / a),
            a == d * (a / d),
    ;
    vstd::arithmetic::div_mod::lemma_mul_mod_noop_right((p / a) * (a / d), d, d);
    assert(p % d == 0);
    vstd::arithmetic::div_mod::lemma_mul_mod_noop_right(idx as int, slot_len as int, d);
    vstd::arithmetic::div_mod::lemma_add_mod_noop(p, (idx * slot_len) as int, d);
}

pub(super) proof fn lemma_retain_copy_after_disjoint_write(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    copy: ReplicaCopy,
    ev: TransferEvent,
    val: EventInfo<TransferDomain>,
)
    requires
        nvme_copy_retained(copy, before, node),
        after.submitted.instance_id() == before.submitted.instance_id(),
        after.submitted.map() == before.submitted.map().insert(ev, val),
        before.submitted.map().submap_of(after.submitted.map()),
        revoked_submap(before.revoked, after.revoked),
        before.succeeded.dom().subset_of(after.succeeded.dom()),
        before.citations(node).subset_of(after.citations(node)),
        !val.access.writes.contains(copy.destination.loc),
    ensures
        nvme_copy_retained(copy, after, node),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    vstd::imap_lib::lemma_submap_of_trans(
        copy.before.submitted.map(),
        before.submitted.map(),
        after.submitted.map(),
    );
    assert(after.submitted.dom().contains(copy.writer));
    assert(after.submitted.map()[copy.writer] == before.submitted.map()[copy.writer]);
}

pub(super) proof fn lemma_retain_copy_after_knows_growth(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    copy: ReplicaCopy,
)
    requires
        nvme_copy_retained(copy, before, node),
        after.submitted == before.submitted,
        revoked_submap(before.revoked, after.revoked),
        before.succeeded.dom().subset_of(after.succeeded.dom()),
        before.citations(node).subset_of(after.citations(node)),
    ensures
        nvme_copy_retained(copy, after, node),
{
}

pub(super) proof fn lemma_nvme_preserve_core_after_submission(
    before: &CacheCore,
    after: &CacheCore,
    dst_frame: FrameId,
    timeline: TimelineId,
    seq: u64,
    ev: TransferEvent,
    val: EventInfo<TransferDomain>,
)
    requires
        before.well_formed(),
        dst_frame.0 < before.frames_spec().metas_spec().len(),
        before.frames_spec().meta(dst_frame).state == STATE_INITIALIZING,
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec() == before.frames_spec(),
        after.catalog_spec() == before.catalog_spec(),
        after.version_seq_start_spec() == before.version_seq_start_spec(),
        after.next_version_seq_spec() == before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.active_leases_spec() == before.active_leases_spec(),
        after.free_lease_slots_spec() == before.free_lease_slots_spec(),
        after.lease_generation_spec() == before.lease_generation_spec(),
        after.spec_spec() == before.spec_spec(),
        after.version_registry_spec() == before.version_registry_spec(),
        timeline < before.timelines_spec().count_spec(),
        seq == before.timelines_spec().next_seq_spec()[timeline as int],
        after.timelines_spec().well_formed(
            after.instance_id(),
            after.node(),
            &after.ledger_spec()@,
        ),
        after.timelines_spec().count_spec() == before.timelines_spec().count_spec(),
        after.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
        after.timelines_spec().confirmed_seq_spec() == before.timelines_spec().confirmed_seq_spec(),
        after.timelines_spec().next_seq_spec()@ == before.timelines_spec().next_seq_spec()@.update(
            timeline as int,
            (seq + 1) as u64,
        ),
        after.timelines_spec().events_spec()@ == before.timelines_spec().events_spec()@.insert(
            (timeline, seq),
            ev,
        ),
        after.ledger_spec()@.submitted.instance_id() == before.instance_id(),
        after.ledger_spec()@.submitted.map() == before.ledger_spec()@.submitted.map().insert(
            ev,
            val,
        ),
        before.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map()),
        after.ledger_spec()@.knows == before.ledger_spec()@.knows,
        after.ledger_spec()@.succeeded == before.ledger_spec()@.succeeded,
        after.ledger_spec()@.revoked == before.ledger_spec()@.revoked,
        val.preds.subset_of(before.ledger_spec()@.submitted.dom()),
        val.access.writes == ISet::empty().insert(before.frames_spec().frame(dst_frame).slot.loc),
    ensures
        after.well_formed(),
        after.kernel_view().extends(before.kernel_view()),
{
    assert(before.ledger_spec()@.citations(after.node()).subset_of(
        after.ledger_spec()@.citations(after.node()),
    ));
    assert forall|i: int|
        0 <= i
            < before.timelines_spec().count_spec() implies #[trigger] after.timelines_spec().confirmed_seq_spec()[i]
        >= before.timelines_spec().confirmed_seq_spec()[i]
        && after.timelines_spec().next_seq_spec()[i] >= before.timelines_spec().next_seq_spec()[i]
        && after.timelines_spec().kinds_spec()[i] == before.timelines_spec().kinds_spec()[i] by {}
    assert forall|u: TimelineId, s: u64|
        before.timelines_spec().issued(
            u,
            s,
        ) implies #[trigger] after.timelines_spec().events_spec()@[(u, s)]
        == before.timelines_spec().events_spec()@[(u, s)] by {}
    assert(after.kernel_view().extends(before.kernel_view()));
    assert forall|e: TransferEvent| #[trigger]
        after.ledger_spec()@.submitted.dom().contains(
            e,
        ) implies after.ledger_spec()@.submitted.map()[e].preds.subset_of(
        after.ledger_spec()@.submitted.dom(),
    ) by {
        if before.ledger_spec()@.submitted.dom().contains(e) {
            assert(before.ledger_spec()@.submitted.map()[e]
                == after.ledger_spec()@.submitted.map()[e]);
        } else {
            assert(e == ev);
        }
    }
    before.lemma_non_published_write_disjoint(dst_frame, val.access.writes);
    assert forall|e: TransferEvent, f: FrameId|
        after.ledger_spec()@.submitted.dom().contains(e)
            && !before.ledger_spec()@.submitted.dom().contains(e) && f.0
            < after.frames_spec().metas_spec().len() && (#[trigger] after.frames_spec().meta(
            f,
        )).state == STATE_PUBLISHED implies !(
    #[trigger] after.ledger_spec()@.submitted.map()[e]).access.writes.contains(
        after.frames_spec().frame(f).slot.loc,
    ) by {
        assert(e == ev);
        assert(before.frames_spec().meta(f).state == STATE_PUBLISHED);
    }
    after.preserve_after_ledger_growth(*before);
}

pub(super) proof fn lemma_nvme_preserve_core_after_complete(
    before: &CacheCore,
    after: &CacheCore,
    timeline: TimelineId,
)
    requires
        before.well_formed(),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec() == before.frames_spec(),
        after.catalog_spec() == before.catalog_spec(),
        after.version_seq_start_spec() == before.version_seq_start_spec(),
        after.next_version_seq_spec() == before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.active_leases_spec() == before.active_leases_spec(),
        after.free_lease_slots_spec() == before.free_lease_slots_spec(),
        after.lease_generation_spec() == before.lease_generation_spec(),
        after.spec_spec() == before.spec_spec(),
        after.version_registry_spec() == before.version_registry_spec(),
        after.timelines_spec().well_formed(
            after.instance_id(),
            after.node(),
            &after.ledger_spec()@,
        ),
        after.timelines_spec().polled(&before.timelines_spec(), timeline),
        after.ledger_spec()@.submitted == before.ledger_spec()@.submitted,
        after.ledger_spec()@.revoked == before.ledger_spec()@.revoked,
        before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
        before.ledger_spec()@.succeeded.dom().subset_of(after.ledger_spec()@.succeeded.dom()),
    ensures
        after.well_formed(),
        after.kernel_view().extends(before.kernel_view()),
{
    assert(before.ledger_spec()@.citations(after.node()).subset_of(
        after.ledger_spec()@.citations(after.node()),
    ));
    assert(after.kernel_view().extends(before.kernel_view()));
    after.preserve_after_ledger_growth(*before);
}

pub(super) proof fn lemma_nvme_stage_copy_retained(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    src_cert: ValueCertificate,
    src_slot: SlotRef,
    dst_slot: SlotRef,
    writer: TransferEvent,
    val: EventInfo<TransferDomain>,
) -> (copy: ReplicaCopy)
    requires
        before.valid(after.submitted.instance_id(), node),
        before.submitted.map().submap_of(after.submitted.map()),
        after.submitted.map() == before.submitted.map().insert(writer, val),
        after.knows == before.knows,
        after.succeeded == before.succeeded,
        after.revoked == before.revoked,
        replica_valid(before, node, src_cert),
        src_cert.slot == src_slot,
        src_slot.well_formed(),
        src_slot.is_full(),
        src_cert.bytes.len() == src_slot.sub_len,
        dst_slot.well_formed(),
        dst_slot.is_full(),
        src_slot.sub_len == dst_slot.sub_len,
        val.access == copy_access(src_slot, dst_slot),
        val.effect == copy_effect(src_slot, dst_slot),
        val.preds.subset_of(before.submitted.dom()),
        before.citations(node).subset_of(val.preds),
    ensures
        copy == (ReplicaCopy {
            before,
            source: src_cert,
            source_slot: src_slot,
            writer,
            destination: dst_slot,
        }),
        nvme_copy_retained(copy, after, node),
{
    let copy = ReplicaCopy {
        before,
        source: src_cert,
        source_slot: src_slot,
        writer,
        destination: dst_slot,
    };
    assert(after.submitted.dom() =~= before.submitted.dom().insert(writer));
    assert(before.citations(node).subset_of(after.citations(node)));
    assert(source_slice_valid(src_cert, src_slot));
    assert(effect_implies(after.submitted.map()[writer].effect, copy_effect(src_slot, dst_slot)));
    assert(nvme_copy_retained(copy, after, node));
    copy
}

pub(super) proof fn lemma_nvme_ring_pending_insert(
    old_ring: IoUring,
    new_ring: IoUring,
    seq: u64,
    ev: TransferEvent,
)
    requires
        !old_ring.pending().contains_key(seq),
        new_ring.pending().contains_key(seq),
        new_ring.pending()[seq].event == ev,
        new_ring.pending() == old_ring.pending().insert(seq, new_ring.pending()[seq]),
    ensures
        ring_event_map(new_ring).contains_key(seq),
        ring_event_map(new_ring)[seq] == ev,
        ring_event_map(new_ring).remove(seq) == ring_event_map(old_ring),
{
    assert(ring_event_map(new_ring).remove(seq) =~= ring_event_map(old_ring));
}

pub(super) proof fn lemma_nvme_ring_pending_remove(old_ring: IoUring, new_ring: IoUring, seq: u64)
    requires
        old_ring.pending().contains_key(seq),
        new_ring.pending() == old_ring.pending().remove(seq),
    ensures
        ring_event_map(old_ring).contains_key(seq),
        ring_event_map(old_ring)[seq] == old_ring.pending()[seq].event,
        ring_event_map(new_ring) == ring_event_map(old_ring).remove(seq),
{
    assert(ring_event_map(new_ring) =~= ring_event_map(old_ring).remove(seq));
}

pub(super) proof fn lemma_nvme_completion_observed(
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
    seq: u64,
    ev: TransferEvent,
    knows: Knows,
    preds: ISet<TransferEvent>,
    op: &InFlightNvmeOp,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id_spec() == before_core.node_id_spec(),
        before_core.timelines_spec().issued(timeline, seq),
        before_core.timelines_spec().events_spec()@[(timeline, seq)] == ev,
        after_core.ledger_spec()@.submitted == before_core.ledger_spec()@.submitted,
        after_core.ledger_spec()@.knows.set() == before_core.ledger_spec()@.knows.set().union(
            knows.set(),
        ),
        knows.set() == knowledge_of(before_core.node(), preds.insert(ev)),
        op.copy@.writer == ev,
        nvme_copy_retained(op.copy@, before_core.ledger_spec()@, before_core.node()),
        op.src_cap@.well_formed(),
        op.src_cap@.instance_id() == after_core.instance_id(),
        op.src_cap@.loc() == after_core.frames_spec().frame(op.hop.src_frame).slot.loc,
        op.src_cap@.frontier().readers == ISet::empty().insert(ev),
        op.hop.src_frame.0 < after_core.frames_spec().metas_spec().len(),
        slot_owned(
            op.dst_tok@,
            after_core.instance_id(),
            op.hop.dst_slot@.loc,
            Frontier { last_writer: Some(ev), readers: ISet::empty() },
        ),
    ensures
        before_core.timelines_spec().observed(timeline, seq, before_core.node(), knows),
        after_core.citations().contains(ev),
        after_core.frames_spec().returned_reads(
            op.hop.src_frame,
            op.src_cap@,
            op.ticket,
            after_core.kernel_view(),
        ),
        write_admitted(slot_frontier(op.dst_tok@), after_core.citations()),
        slot_frontier(op.dst_tok@).readers.is_empty(),
{
    assert(before_core.ledger_spec()@.submitted.dom().contains(ev));
    assert(preds.insert(ev).contains(ev));
    assert(knows.set().contains((before_core.node(), ev)));
    assert(after_core.citations().contains(ev));
}

pub(super) proof fn lemma_nvme_dst_latest(
    writer: TransferEvent,
    dst_loc: SlotLoc,
    tracked submitted: &transfer::context::SubmittedMap,
    tracked tok: &mut SlotToken,
    tracked instance: &TransferInstance,
)
    requires
        submitted.instance_id() == instance.id(),
        submitted.dom().contains(writer),
        old(tok).well_formed(),
        old(tok).instance_id() == instance.id(),
        old(tok).loc() == dst_loc,
        foundation::tracked::slot_base_frontier(old(tok).value()).last_writer == Some(writer),
    ensures
        final(tok).well_formed(),
        final(tok).instance_id() == old(tok).instance_id(),
        final(tok).loc() == old(tok).loc(),
        final(tok).value() == old(tok).value(),
        forall|e: TransferEvent| #[trigger]
            submitted.dom().contains(e) && submitted.map()[e].access.writes.contains(dst_loc) && e
                != writer ==> submitted.map()[writer].preds.contains(e),
{
    if exists|e: TransferEvent|
        submitted.dom().contains(e) && (#[trigger] submitted.map()[e]).access.writes.contains(
            dst_loc,
        ) && e != writer && !submitted.map()[writer].preds.contains(e) {
        let e = choose|e: TransferEvent|
            submitted.dom().contains(e) && (#[trigger] submitted.map()[e]).access.writes.contains(
                dst_loc,
            ) && e != writer && !submitted.map()[writer].preds.contains(e);
        instance.slot_writer_is_latest(writer, e, tok, submitted);
        assert(false);
    }
}

pub(super) proof fn lemma_nvme_certify_completed_copy(
    copy: ReplicaCopy,
    ledger: KnowledgeLedger,
    node: NodeId,
) -> (cert: ValueCertificate)
    requires
        ledger.valid(ledger.submitted.instance_id(), node),
        complete_receipts(ledger),
        nvme_copy_retained(copy, ledger, node),
        ledger.citations(node).contains(copy.writer),
        event_succeeded(ledger, node, copy.writer),
        copy.source.slot == copy.source_slot,
        copy.source.bytes.len() == copy.source_slot.sub_len,
        forall|e: TransferEvent| #[trigger]
            ledger.submitted.dom().contains(e) && ledger.submitted.map()[e].access.writes.contains(
                copy.destination.loc,
            ) && e != copy.writer ==> ledger.submitted.map()[copy.writer].preds.contains(e),
    ensures
        replica_valid(ledger, node, cert),
        cert.writer == copy.writer,
        cert.slot == copy.destination,
        cert.bytes == copy.source.bytes,
{
    let w = copy.writer;
    let cert = copied_slice_certificate(copy.source, copy.source_slot, w, copy.destination);
    assert(source_slice_bytes(copy.source, copy.source_slot) =~= copy.source.bytes);
    assert forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && ledger_consistent_with_history(
            ledger,
            node,
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
        cert,
        ledger,
        node,
        h,
    ) by {
        lemma_completed_replica_copy(
            copy.before,
            ledger,
            node,
            copy.source,
            copy.source_slot,
            w,
            copy.destination,
            h,
            run,
        );
        assert forall|e|
            ledger.submitted.dom().contains(e) && (#[trigger] (h.access)(e)).writes.contains(
                cert.slot.loc,
            ) && e != w implies (h.happens_before)(e, w) by {
            assert(ledger.submitted.map()[e].access == (h.access)(e));
            assert(ledger.submitted.map()[w].preds.contains(e));
        }
    }
    cert
}

proof fn lemma_nvme_disjoint_after_insert(
    old_transfers: Seq<Option<InFlightNvmeOp>>,
    new_transfers: Seq<Option<InFlightNvmeOp>>,
    idx: int,
    new_op: InFlightNvmeOp,
)
    requires
        nvme_pending_disjoint(old_transfers),
        0 <= idx < BITMAP_CAPACITY,
        new_transfers == old_transfers.update(idx, Some(new_op)),
        forall|k: int|
            0 <= k < BITMAP_CAPACITY && k != idx ==> (#[trigger] old_transfers[k] matches Some(b)
                ==> new_op.hop.dst_frame != b.hop.dst_frame && new_op.hop.dst_slot@.loc
                != b.hop.dst_slot@.loc && new_op.hop.src_lease.slot_spec()
                != b.hop.src_lease.slot_spec()),
    ensures
        nvme_pending_disjoint(new_transfers),
{
    reveal(nvme_pending_disjoint);
    assert forall|i: int, j: int|
        0 <= i < BITMAP_CAPACITY && 0 <= j < BITMAP_CAPACITY && i != j implies match (#[trigger]
        new_transfers[i],
        #[trigger] new_transfers[j],
    ) {
        (Some(a), Some(b)) => a.hop.dst_frame != b.hop.dst_frame && a.hop.dst_slot@.loc
            != b.hop.dst_slot@.loc && a.hop.src_lease.slot_spec() != b.hop.src_lease.slot_spec(),
        _ => true,
    } by {
        if i != idx && j != idx {
            assert(new_transfers[i] == old_transfers[i]);
            assert(new_transfers[j] == old_transfers[j]);
        } else if i == idx {
            assert(new_transfers[j] == old_transfers[j]);
        } else {
            assert(new_transfers[i] == old_transfers[i]);
        }
    }
}

proof fn lemma_nvme_disjoint_after_remove(
    old_transfers: Seq<Option<InFlightNvmeOp>>,
    new_transfers: Seq<Option<InFlightNvmeOp>>,
    idx: int,
)
    requires
        nvme_pending_disjoint(old_transfers),
        0 <= idx < BITMAP_CAPACITY,
        new_transfers == old_transfers.update(idx, None),
    ensures
        nvme_pending_disjoint(new_transfers),
{
    reveal(nvme_pending_disjoint);
    assert forall|i: int, j: int|
        0 <= i < BITMAP_CAPACITY && 0 <= j < BITMAP_CAPACITY && i != j implies match (#[trigger]
        new_transfers[i],
        #[trigger] new_transfers[j],
    ) {
        (Some(a), Some(b)) => a.hop.dst_frame != b.hop.dst_frame && a.hop.dst_slot@.loc
            != b.hop.dst_slot@.loc && a.hop.src_lease.slot_spec() != b.hop.src_lease.slot_spec(),
        _ => true,
    } by {
        if i != idx && j != idx {
            assert(new_transfers[i] == old_transfers[i]);
            assert(new_transfers[j] == old_transfers[j]);
        }
    }
}

proof fn lemma_nvme_disjoint_pair(
    transfers: Seq<Option<InFlightNvmeOp>>,
    i: int,
    j: int,
    a: InFlightNvmeOp,
    b: InFlightNvmeOp,
)
    requires
        nvme_pending_disjoint(transfers),
        0 <= i < BITMAP_CAPACITY,
        0 <= j < BITMAP_CAPACITY,
        i != j,
        transfers[i] == Some(a),
        transfers[j] == Some(b),
    ensures
        a.hop.dst_frame != b.hop.dst_frame,
        a.hop.dst_slot@.loc != b.hop.dst_slot@.loc,
        a.hop.src_lease.slot_spec() != b.hop.src_lease.slot_spec(),
{
    reveal(nvme_pending_disjoint);
}

proof fn lemma_nvme_op_valid_after_stage(
    op: InFlightNvmeOp,
    before_core: &CacheCore,
    after_hop_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
    i: int,
    new_op: InFlightNvmeOp,
    ev: TransferEvent,
    val: EventInfo<TransferDomain>,
)
    requires
        op.valid(before_core, timeline, i),
        after_hop_core.frames_spec().separated(),
        after_hop_core.node_id_spec() == before_core.node_id_spec(),
        after_hop_core.ledger_spec() == before_core.ledger_spec(),
        after_hop_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_hop_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_hop_core.timelines_spec().only_watermarks(&before_core.timelines_spec()),
        after_hop_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_hop_core.frames_spec().preserves_initializing(&before_core.frames_spec(), None),
        before_core.frames_spec().meta(new_op.hop.dst_frame).state != STATE_INITIALIZING,
        before_core.active_leases_spec()[new_op.hop.src_lease.slot_spec() as int].is_none(),
        after_hop_core.active_leases_spec().len() == before_core.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before_core.active_leases_spec().len() && s
                != new_op.hop.src_lease.slot_spec() as int
                ==> #[trigger] after_hop_core.active_leases_spec()[s]
                == before_core.active_leases_spec()[s],
        after_core.node_id_spec() == after_hop_core.node_id_spec(),
        after_core.frames_spec() == after_hop_core.frames_spec(),
        after_core.active_leases_spec() == after_hop_core.active_leases_spec(),
        after_core.next_version_seq_spec() == after_hop_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == after_hop_core.version_seq_limit_spec(),
        after_core.timelines_spec().count_spec() == after_hop_core.timelines_spec().count_spec(),
        after_core.timelines_spec().next_seq_spec()[timeline as int]
            == after_hop_core.timelines_spec().next_seq_spec()[timeline as int] + 1,
        after_core.timelines_spec().events_spec()@
            == after_hop_core.timelines_spec().events_spec()@.insert(
            (timeline, new_op.ticket.sequence()),
            ev,
        ),
        new_op.ticket.sequence()
            == after_hop_core.timelines_spec().next_seq_spec()[timeline as int],
        new_op.hop.dst_frame.0 < after_hop_core.frames_spec().metas_spec().len(),
        new_op.hop.dst_slot@ == after_hop_core.frames_spec().frame(new_op.hop.dst_frame).slot,
        after_core.ledger_spec()@.submitted.instance_id()
            == after_hop_core.ledger_spec()@.submitted.instance_id(),
        after_core.ledger_spec()@.submitted.map()
            == after_hop_core.ledger_spec()@.submitted.map().insert(ev, val),
        after_hop_core.ledger_spec()@.submitted.map().submap_of(
            after_core.ledger_spec()@.submitted.map(),
        ),
        after_core.ledger_spec()@.knows == after_hop_core.ledger_spec()@.knows,
        after_core.ledger_spec()@.succeeded == after_hop_core.ledger_spec()@.succeeded,
        after_core.ledger_spec()@.revoked == after_hop_core.ledger_spec()@.revoked,
        val.access.writes == ISet::empty().insert(new_op.hop.dst_slot@.loc),
    ensures
        op.valid(after_core, timeline, i),
        op.hop.dst_frame != new_op.hop.dst_frame,
        op.hop.dst_slot@.loc != new_op.hop.dst_slot@.loc,
        op.hop.src_lease.slot_spec() != new_op.hop.src_lease.slot_spec(),
        op.ticket.sequence() < new_op.ticket.sequence(),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    assert(nvme_copy_retained(op.copy@, before_core.ledger_spec()@, before_core.node()));
    assert(op.hop.dst_frame != new_op.hop.dst_frame);
    assert(op.hop.src_lease.slot_spec() != new_op.hop.src_lease.slot_spec());
    if op.hop.dst_frame.0 < new_op.hop.dst_frame.0 {
        assert(after_hop_core.frames_spec().ghosts_spec()@[op.hop.dst_frame.0 as nat].slot.loc
            != after_hop_core.frames_spec().ghosts_spec()@[new_op.hop.dst_frame.0 as nat].slot.loc);
    } else {
        assert(after_hop_core.frames_spec().ghosts_spec()@[new_op.hop.dst_frame.0 as nat].slot.loc
            != after_hop_core.frames_spec().ghosts_spec()@[op.hop.dst_frame.0 as nat].slot.loc);
    }
    assert(after_hop_core.frames_spec().meta(op.hop.dst_frame) == before_core.frames_spec().meta(
        op.hop.dst_frame,
    ));
    assert(after_hop_core.frames_spec().frame(op.hop.dst_frame) == before_core.frames_spec().frame(
        op.hop.dst_frame,
    ));
    assert(op.ticket.sequence() < new_op.ticket.sequence());
    assert(after_core.timelines_spec().events_spec()@[(timeline, op.ticket.sequence())]
        == op.copy@.writer) by {
        broadcast use vstd::map::group_map_lemmas;

    }
    assert(before_core.ledger_spec()@.citations(before_core.node()).subset_of(
        after_core.ledger_spec()@.citations(after_core.node()),
    )) by {
        broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    }
    lemma_retain_copy_after_disjoint_write(
        before_core.ledger_spec()@,
        after_core.ledger_spec()@,
        after_core.node(),
        op.copy@,
        ev,
        val,
    );
}

proof fn lemma_nvme_op_valid_after_reap(
    op: InFlightNvmeOp,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
    i: int,
    idx: int,
    seq: u64,
    removed_op: InFlightNvmeOp,
)
    requires
        op.valid(before_core, timeline, i),
        0 <= i < BITMAP_CAPACITY,
        0 <= idx < BITMAP_CAPACITY,
        i != idx,
        (seq % BITMAP_CAPACITY) as int == idx,
        op.hop.dst_frame != removed_op.hop.dst_frame,
        op.hop.src_lease.slot_spec() != removed_op.hop.src_lease.slot_spec(),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
        after_core.timelines_spec().next_seq_spec() == before_core.timelines_spec().next_seq_spec(),
        after_core.timelines_spec().events_spec() == before_core.timelines_spec().events_spec(),
        after_core.ledger_spec()@.submitted == before_core.ledger_spec()@.submitted,
        after_core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
        before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
        before_core.ledger_spec()@.succeeded.dom().subset_of(
            after_core.ledger_spec()@.succeeded.dom(),
        ),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        after_core.frames_spec().preserves_initializing(
            &before_core.frames_spec(),
            Some(removed_op.hop.dst_frame),
        ),
        before_core.frames_spec().versions_spec()@.submap_of(
            after_core.frames_spec().versions_spec()@,
        ),
        after_core.active_leases_spec().len() == before_core.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before_core.active_leases_spec().len() && s
                != removed_op.hop.src_lease.slot_spec() as int
                ==> #[trigger] after_core.active_leases_spec()[s]
                == before_core.active_leases_spec()[s],
    ensures
        op.valid(after_core, timeline, i),
        op.ticket.sequence() != seq,
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(nvme_copy_retained);
    assert(before_core.ledger_spec()@.citations(before_core.node()).subset_of(
        after_core.ledger_spec()@.citations(after_core.node()),
    )) by {
        assert forall|e: TransferEvent|
            before_core.ledger_spec()@.citations(before_core.node()).contains(
                e,
            ) implies after_core.ledger_spec()@.citations(after_core.node()).contains(e) by {
            assert(before_core.ledger_spec()@.knows.set().contains((before_core.node(), e)));
        }
    }
    assert(op.ticket.sequence() != seq) by {
        assert((op.ticket.sequence() % BITMAP_CAPACITY) as int == i);
        assert((seq % BITMAP_CAPACITY) as int == idx);
    }
    assert(after_core.frames_spec().meta(op.hop.dst_frame) == before_core.frames_spec().meta(
        op.hop.dst_frame,
    ));
    assert(after_core.frames_spec().frame(op.hop.dst_frame) == before_core.frames_spec().frame(
        op.hop.dst_frame,
    ));
    assert(before_core.frames_spec().versions_spec()@.dom().contains(op.hop.version));
    assert(after_core.frames_spec().versions_spec()@.dom().contains(op.hop.version));
    assert(after_core.frames_spec().versions_spec()@[op.hop.version] == op.hop.src_cert@.bytes);
    lemma_retain_copy_after_knows_growth(
        before_core.ledger_spec()@,
        after_core.ledger_spec()@,
        after_core.node(),
        op.copy@,
    );
}

pub(super) proof fn lemma_nvme_pending_preserved_after_stage(
    old_transfers: Seq<Option<InFlightNvmeOp>>,
    new_transfers: Seq<Option<InFlightNvmeOp>>,
    old_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    new_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    old_ring: IoUring,
    new_ring: IoUring,
    before_core: &CacheCore,
    after_hop_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
    idx: int,
    new_op: InFlightNvmeOp,
    ev: TransferEvent,
    val: EventInfo<TransferDomain>,
)
    requires
        nvme_pending_valid(old_transfers, old_ops, old_ring, before_core, timeline),
        before_core.well_formed(),
        after_hop_core.well_formed(),
        after_core.well_formed(),
        0 <= idx < BITMAP_CAPACITY,
        old_transfers[idx].is_none(),
        new_transfers == old_transfers.update(idx, Some(new_op)),
        new_ops@ == old_ops@.update(idx, new_ops[idx]),
        new_ops[idx].is_some(),
        new_op.valid(after_core, timeline, idx),
        new_ring.pending() == old_ring.pending().insert(
            new_op.ticket.sequence(),
            new_ring.pending()[new_op.ticket.sequence()],
        ),
        new_ring.pending()[new_op.ticket.sequence()].event == new_op.copy@.writer,
        after_hop_core.node_id_spec() == before_core.node_id_spec(),
        after_hop_core.ledger_spec() == before_core.ledger_spec(),
        after_hop_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_hop_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_hop_core.timelines_spec().only_watermarks(&before_core.timelines_spec()),
        after_hop_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_hop_core.frames_spec().preserves_initializing(&before_core.frames_spec(), None),
        before_core.frames_spec().meta(new_op.hop.dst_frame).state != STATE_INITIALIZING,
        before_core.active_leases_spec()[new_op.hop.src_lease.slot_spec() as int].is_none(),
        after_hop_core.active_leases_spec().len() == before_core.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before_core.active_leases_spec().len() && s
                != new_op.hop.src_lease.slot_spec() as int
                ==> #[trigger] after_hop_core.active_leases_spec()[s]
                == before_core.active_leases_spec()[s],
        after_core.node_id_spec() == after_hop_core.node_id_spec(),
        after_core.frames_spec() == after_hop_core.frames_spec(),
        after_core.active_leases_spec() == after_hop_core.active_leases_spec(),
        after_core.next_version_seq_spec() == after_hop_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == after_hop_core.version_seq_limit_spec(),
        after_core.timelines_spec().count_spec() == after_hop_core.timelines_spec().count_spec(),
        after_core.timelines_spec().next_seq_spec()[timeline as int]
            == after_hop_core.timelines_spec().next_seq_spec()[timeline as int] + 1,
        after_core.timelines_spec().events_spec()@
            == after_hop_core.timelines_spec().events_spec()@.insert(
            (timeline, new_op.ticket.sequence()),
            ev,
        ),
        new_op.ticket.sequence()
            == after_hop_core.timelines_spec().next_seq_spec()[timeline as int],
        after_core.ledger_spec()@.submitted.instance_id()
            == after_hop_core.ledger_spec()@.submitted.instance_id(),
        after_core.ledger_spec()@.submitted.map()
            == after_hop_core.ledger_spec()@.submitted.map().insert(ev, val),
        after_hop_core.ledger_spec()@.submitted.map().submap_of(
            after_core.ledger_spec()@.submitted.map(),
        ),
        after_core.ledger_spec()@.knows == after_hop_core.ledger_spec()@.knows,
        after_core.ledger_spec()@.succeeded == after_hop_core.ledger_spec()@.succeeded,
        after_core.ledger_spec()@.revoked == after_hop_core.ledger_spec()@.revoked,
        val.access.writes == ISet::empty().insert(new_op.hop.dst_slot@.loc),
    ensures
        nvme_pending_valid(new_transfers, new_ops, new_ring, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(InFlightNvmeOp::valid);
    lemma_core_wf_facts(after_hop_core);
    lemma_core_frames_separated(after_hop_core);
    lemma_core_wf_facts(after_core);
    reveal(nvme_pending_valid);
    assert(new_op.hop.dst_frame.0 < after_hop_core.frames_spec().metas_spec().len()
        && new_op.hop.dst_slot@ == after_hop_core.frames_spec().frame(new_op.hop.dst_frame).slot)
        by {
        reveal(InFlightNvmeOp::valid);
    }
    assert(after_core.ledger_spec()@.submitted.instance_id()
        == after_hop_core.ledger_spec()@.submitted.instance_id()) by {
        assert(before_core.ledger_spec()@.submitted.instance_id()
            == after_hop_core.ledger_spec()@.submitted.instance_id());
    }
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (
    #[trigger] new_transfers[i] matches Some(op) ==> {
        &&& op.valid(after_core, timeline, i)
        &&& new_ring.pending().contains_key(op.ticket.sequence())
        &&& new_ring.pending()[op.ticket.sequence()].event == op.copy@.writer
    }) by {
        if i != idx {
            if let Some(op) = new_transfers[i] {
                assert(old_transfers[i] == Some(op));
                assert(op.valid(before_core, timeline, i));
                lemma_nvme_op_valid_after_stage(
                    op,
                    before_core,
                    after_hop_core,
                    after_core,
                    timeline,
                    i,
                    new_op,
                    ev,
                    val,
                );
            }
        }
    }
    assert forall|k: int| 0 <= k < BITMAP_CAPACITY && k != idx implies (
    #[trigger] old_transfers[k] matches Some(b) ==> new_op.hop.dst_frame != b.hop.dst_frame
        && new_op.hop.dst_slot@.loc != b.hop.dst_slot@.loc && new_op.hop.src_lease.slot_spec()
        != b.hop.src_lease.slot_spec()) by {
        if let Some(b) = old_transfers[k] {
            assert(b.valid(before_core, timeline, k));
            lemma_nvme_op_valid_after_stage(
                b,
                before_core,
                after_hop_core,
                after_core,
                timeline,
                k,
                new_op,
                ev,
                val,
            );
        }
    }
    lemma_nvme_disjoint_after_insert(old_transfers, new_transfers, idx, new_op);
}

pub(super) proof fn lemma_nvme_pending_preserved_after_reap(
    old_transfers: Seq<Option<InFlightNvmeOp>>,
    new_transfers: Seq<Option<InFlightNvmeOp>>,
    old_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    new_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    old_ring: IoUring,
    new_ring: IoUring,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
    idx: int,
    seq: u64,
    removed_op: InFlightNvmeOp,
)
    requires
        nvme_pending_valid(old_transfers, old_ops, old_ring, before_core, timeline),
        before_core.well_formed(),
        after_core.well_formed(),
        0 <= idx < BITMAP_CAPACITY,
        old_transfers[idx] == Some(removed_op),
        removed_op.ticket.sequence() == seq,
        new_transfers == old_transfers.update(idx, None),
        new_ops@ == old_ops@.update(idx, None),
        new_ring.pending() == old_ring.pending().remove(seq),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
        after_core.timelines_spec().next_seq_spec() == before_core.timelines_spec().next_seq_spec(),
        after_core.timelines_spec().events_spec() == before_core.timelines_spec().events_spec(),
        after_core.ledger_spec()@.submitted == before_core.ledger_spec()@.submitted,
        after_core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
        before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
        before_core.ledger_spec()@.succeeded.dom().subset_of(
            after_core.ledger_spec()@.succeeded.dom(),
        ),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        after_core.frames_spec().preserves_initializing(
            &before_core.frames_spec(),
            Some(removed_op.hop.dst_frame),
        ),
        before_core.frames_spec().versions_spec()@.submap_of(
            after_core.frames_spec().versions_spec()@,
        ),
        after_core.active_leases_spec().len() == before_core.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before_core.active_leases_spec().len() && s
                != removed_op.hop.src_lease.slot_spec() as int
                ==> #[trigger] after_core.active_leases_spec()[s]
                == before_core.active_leases_spec()[s],
    ensures
        nvme_pending_valid(new_transfers, new_ops, new_ring, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(InFlightNvmeOp::valid);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(after_core);
    reveal(nvme_pending_valid);
    assert((seq % BITMAP_CAPACITY) as int == idx) by {
        reveal(InFlightNvmeOp::valid);
        assert(removed_op.valid(before_core, timeline, idx));
    }
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (
    #[trigger] new_transfers[i] matches Some(op) ==> {
        &&& op.valid(after_core, timeline, i)
        &&& new_ring.pending().contains_key(op.ticket.sequence())
        &&& new_ring.pending()[op.ticket.sequence()].event == op.copy@.writer
    }) by {
        if i != idx {
            if let Some(op) = new_transfers[i] {
                assert(old_transfers[i] == Some(op));
                assert(op.valid(before_core, timeline, i));
                lemma_nvme_disjoint_pair(old_transfers, i, idx, op, removed_op);
                lemma_nvme_op_valid_after_reap(
                    op,
                    before_core,
                    after_core,
                    timeline,
                    i,
                    idx,
                    seq,
                    removed_op,
                );
            }
        }
    }
    lemma_nvme_disjoint_after_remove(old_transfers, new_transfers, idx);
}

} // verus!
