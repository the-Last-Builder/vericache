use super::*;
use foundation::{
    history::{reachable, reachable_is_well_formed, History},
    order::is_last,
    space::{restrict, Access, Effect},
    value::*,
};

verus! {

pub type TransferHistory = History<TransferDomain>;

pub type TransferRun = Run<TransferDomain>;

pub open spec fn submissions_match_history(ledger: KnowledgeLedger, h: TransferHistory) -> bool {
    forall|e| #[trigger]
        ledger.submitted.dom().contains(e) ==> {
            let info = ledger.submitted.map()[e];
            &&& h.events.contains(e)
            &&& (h.access)(e) == info.access
            &&& (h.effect)(e) == info.effect
            &&& (h.read_guards)(e) == info.read_guards
            &&& forall|p| #[trigger] info.preds.contains(p) == (h.happens_before)(p, e)
        }
}

pub open spec fn revocations_match_history(ledger: KnowledgeLedger, h: TransferHistory) -> bool {
    forall|k: (SlotLoc, TransferShare)| #[trigger]
        ledger.revoked.dom().contains(k) ==> h.revoked.dom().contains(k) && h.revoked[k]
            == ledger.revoked.map()[k]
}

pub open spec fn ledger_consistent_with_history(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
) -> bool {
    &&& submissions_match_history(ledger, h)
    &&& revocations_match_history(ledger, h)
    &&& forall|e| #[trigger]
        event_succeeded(ledger, node, e) ==> ledger.submitted.dom().contains(e) && (run.succeeded)(
            e,
        ) && (run.end)(e).is_some()
}

/// With only observations from event sources that may fail on completion,
/// the ledger allows a run in which all events failed and memory stayed initial.
pub proof fn lemma_completion_without_success(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    initial_slot: Seq<u8>,
) -> (run: TransferRun)
    requires
        submissions_match_history(ledger, h),
        revocations_match_history(ledger, h),
        ledger.succeeded.dom().is_empty(),
        forall|e: TransferEvent| #[trigger]
            ledger.citations(node).contains(e) ==> !e.source.succeeds_on_observe(),
    ensures
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        forall|e| #[trigger] (run.end)(e) == Some(0) && !(run.succeeded)(e),
        forall|t, loc| #[trigger] (run.memory)(t)(loc) == initial_slot,
{
    let run = Run::<TransferDomain> {
        memory: |_t: nat| |_loc: SlotLoc| initial_slot,
        start: |_e: TransferEvent| 0,
        end: |_e: TransferEvent| Some(0),
        succeeded: |_e: TransferEvent| false,
        rejected_read: |_e: TransferEvent, _loc: SlotLoc| true,
    };
    assert forall|a, b| #[trigger] (h.happens_before)(a, b) implies finished_by(
        run,
        a,
        (run.start)(b),
    ) by {
        assert((run.end)(a) == Some(0));
    }
    assert forall|e| !#[trigger] ledger.succeeded.contains(e) by {
        assert(!ledger.succeeded.dom().contains(e));
    }
    run
}

/// A failed completion from a fallible source cannot certify its own write.
pub proof fn failed_io_cannot_certify(ledger: KnowledgeLedger, node: NodeId, cert: ValueCertificate)
    requires
        !cert.writer.source.succeeds_on_observe(),
        ledger.citations(node).contains(cert.writer),
        !ledger.succeeded.contains(cert.writer),
    ensures
        !event_succeeded(ledger, node, cert.writer),
        !replica_valid(ledger, node, cert),
{
}

pub proof fn ledger_history_restrict(
    earlier: KnowledgeLedger,
    later: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
)
    requires
        forall|e| #[trigger] earlier.succeeded.contains(e) ==> earlier.submitted.dom().contains(e),
        earlier.submitted.map().submap_of(later.submitted.map()),
        revoked_submap(earlier.revoked, later.revoked),
        earlier.succeeded.dom().subset_of(later.succeeded.dom()),
        earlier.citations(node).subset_of(later.citations(node)),
        ledger_consistent_with_history(later, node, h, run),
    ensures
        ledger_consistent_with_history(earlier, node, h, run),
{
    assert forall|e| #[trigger] earlier.submitted.dom().contains(e) implies {
        let info = earlier.submitted.map()[e];
        &&& h.events.contains(e) && (h.access)(e) == info.access && (h.effect)(e) == info.effect
            && (h.read_guards)(e) == info.read_guards
        &&& forall|p| #[trigger] info.preds.contains(p) == (h.happens_before)(p, e)
    } by {
        assert(later.submitted.dom().contains(e));
        assert(earlier.submitted.map()[e] == later.submitted.map()[e]);
    }
    assert forall|k: (SlotLoc, TransferShare)| #[trigger]
        earlier.revoked.dom().contains(k) implies h.revoked.dom().contains(k) && h.revoked[k]
        == earlier.revoked.map()[k] by {
        assert(later.revoked.dom().contains(k));
        assert(earlier.revoked.map()[k] == later.revoked.map()[k]);
    }
    assert forall|e| #[trigger]
        event_succeeded(earlier, node, e) implies earlier.submitted.dom().contains(e) && (
    run.succeeded)(e) && (run.end)(e).is_some() by {
        assert(event_succeeded(later, node, e));
    }
}

pub open spec fn slot_holds_bytes(
    run: TransferRun,
    time: nat,
    slot: SlotRef,
    bytes: Seq<u8>,
) -> bool {
    &&& slot.well_formed()
    &&& bytes.len() == slot.sub_len
    &&& (run.memory)(time)(slot.loc).len() == slot.slot_len
    &&& (run.memory)(time)(slot.loc).subrange(
        slot.sub_offset as int,
        (slot.sub_offset + slot.sub_len) as int,
    ) == bytes
}

pub open spec fn event_completed_in(
    e: TransferEvent,
    h: TransferHistory,
    run: TransferRun,
) -> bool {
    h.events.contains(e) && (run.succeeded)(e) && (run.end)(e).is_some()
}

pub open spec fn source_slice_valid(source: ValueCertificate, source_slot: SlotRef) -> bool {
    &&& source_slot.well_formed()
    &&& source_slot.loc == source.slot.loc
    &&& source_slot.slot_len == source.slot.slot_len
    &&& source.slot.sub_offset <= source_slot.sub_offset
    &&& source_slot.sub_offset + source_slot.sub_len <= source.slot.sub_offset + source.slot.sub_len
}

pub open spec fn source_slice_bytes(source: ValueCertificate, source_slot: SlotRef) -> Seq<u8> {
    let rel = (source_slot.sub_offset - source.slot.sub_offset) as int;
    source.bytes.subrange(rel, rel + source_slot.sub_len as int)
}

pub open spec fn gather_pieces_contiguous(dst: SlotRef, pieces: Seq<GatherPiece>) -> bool {
    &&& pieces.len() > 0
    &&& pieces[0].dst_offset == 0
    &&& pieces.last().dst_offset + pieces.last().source_slot.sub_len == dst.sub_len
    &&& forall|p: int|
        0 <= p < pieces.len() ==> (#[trigger] pieces[p]).dst_offset + pieces[p].source_slot.sub_len
            <= dst.sub_len
    &&& forall|p: int, q: int|
        0 <= p && q == p + 1 && q < pieces.len() ==> (#[trigger] pieces[p]).dst_offset
            + pieces[p].source_slot.sub_len == (#[trigger] pieces[q]).dst_offset
}

pub open spec fn gather_is_single_event(pieces: Seq<GatherPiece>, writer: TransferEvent) -> bool {
    forall|p: int| 0 <= p < pieces.len() ==> (#[trigger] pieces[p]).writer == writer
}

pub open spec fn gather_sequential_chain(pieces: Seq<GatherPiece>, h: TransferHistory) -> bool {
    forall|p: int, q: int|
        0 <= p && q == p + 1 && q < pieces.len() ==> (h.happens_before)(
            (#[trigger] pieces[p]).writer,
            (#[trigger] pieces[q]).writer,
        )
}

pub open spec fn gather_sequential_last_writers(
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    h: TransferHistory,
) -> bool {
    forall|p: int, q: int|
        0 <= p && q == p + 1 && q < pieces.len() ==> is_last(
            h.happens_before,
            writers_before(h, (#[trigger] pieces[q]).writer, dst.loc),
            (#[trigger] pieces[p]).writer,
        )
}

pub open spec fn proof_succeeded(
    cert: ValueCertificate,
    h: TransferHistory,
    run: TransferRun,
) -> bool
    decreases cert,
{
    match cert.proof {
        ValueProof::HostWrite => {
            &&& event_completed_in(cert.writer, h, run)
            &&& (h.access)(cert.writer).writes.contains(cert.slot.loc)
            &&& effect_implies((h.effect)(cert.writer), splice_bytes(cert.slot, cert.bytes))
        },
        ValueProof::CopyHop { source, source_slot } => {
            &&& event_completed_in(cert.writer, h, run)
            &&& source_slice_valid(*source, source_slot)
            &&& source_slot.sub_len == cert.slot.sub_len
            &&& (h.access)(cert.writer).reads.contains(source_slot.loc)
            &&& (h.access)(cert.writer).writes.contains(cert.slot.loc)
            &&& effect_implies((h.effect)(cert.writer), copy_effect(source_slot, cert.slot))
            &&& proof_succeeded(*source, h, run)
        },
        ValueProof::GatherHop { pieces } => {
            let single = gather_is_single_event(pieces, cert.writer);
            &&& cert.slot.is_full()
            &&& gather_pieces_contiguous(cert.slot, pieces)
            &&& pieces.last().writer == cert.writer
            &&& (!single ==> gather_sequential_chain(pieces, h))
            &&& forall|p: int|
                0 <= p < pieces.len() ==> {
                    let piece = #[trigger] pieces[p];
                    let dst_sub = cert.slot.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
                    &&& source_slice_valid(piece.source, piece.source_slot)
                    &&& event_completed_in(piece.writer, h, run)
                    &&& (h.access)(piece.writer).reads.contains(piece.source_slot.loc)
                    &&& (h.access)(piece.writer).writes.contains(cert.slot.loc)
                    &&& proof_succeeded(piece.source, h, run)
                    &&& if single {
                        effect_implies(
                            (h.effect)(piece.writer),
                            subrange_copy_effect(piece.source_slot, dst_sub),
                        )
                    } else {
                        &&& effect_implies(
                            (h.effect)(piece.writer),
                            copy_effect(piece.source_slot, dst_sub),
                        )
                        &&& (p > 0 && !dst_sub.is_full() ==> (h.access)(
                            piece.writer,
                        ).reads.contains(cert.slot.loc))
                    }
                }
        },
    }
}

/// Zero-quantifier slot certificate validity: `cert.writer` establishes `cert.bytes`
/// in `cert.slot` at its completion.
pub open spec fn certificate_valid_in(
    cert: ValueCertificate,
    h: TransferHistory,
    run: TransferRun,
) -> bool {
    &&& proof_succeeded(cert, h, run)
    &&& event_completed_in(cert.writer, h, run)
    &&& slot_holds_bytes(run, (run.end)(cert.writer).unwrap(), cert.slot, cert.bytes)
}

pub open spec fn successful_submission(
    ledger: KnowledgeLedger,
    node: NodeId,
    event: TransferEvent,
    access: Access<TransferDomain>,
    effect: Effect<TransferDomain>,
) -> bool {
    &&& event_succeeded(ledger, node, event)
    &&& ledger.submitted.dom().contains(event)
    &&& ledger.submitted.map()[event].access == access
    &&& ledger.submitted.map()[event].effect == effect
}

pub open spec fn last_writer_certified(
    h: TransferHistory,
    reader: TransferEvent,
    cert: ValueCertificate,
) -> bool {
    is_last(h.happens_before, writers_before(h, reader, cert.slot.loc), cert.writer)
}

pub proof fn lemma_certified_read_observes_slot(
    cert: ValueCertificate,
    h: TransferHistory,
    run: TransferRun,
    reader: TransferEvent,
)
    requires
        reachable(h),
        valid_run(h, run),
        certificate_valid_in(cert, h, run),
        h.events.contains(reader),
        (h.access)(reader).reads.contains(cert.slot.loc),
        !(run.rejected_read)(reader, cert.slot.loc),
        last_writer_certified(h, reader, cert),
    ensures
        (run.memory)((run.start)(reader))(cert.slot.loc) == (run.memory)(
            (run.end)(cert.writer).unwrap(),
        )(cert.slot.loc),
        slot_holds_bytes(run, (run.start)(reader), cert.slot, cert.bytes),
        cert.slot.is_full() ==> (run.memory)((run.start)(reader))(cert.slot.loc) == cert.bytes,
{
    reachable_is_well_formed(h);
    let loc = cert.slot.loc;
    let writers = writers_before(h, reader, loc);
    read_returns_last_write(h, run, reader, loc);
    assert(writers.contains(cert.writer));
    assert(writers != ISet::empty());
    let (last, end) = choose|last: TransferEvent, end: nat| #[trigger]
        is_last(h.happens_before, writers, last) && (run.end)(last) == Some(end) && value_at_start(
            run,
            reader,
            loc,
        ) == #[trigger] (run.memory)(end)(loc);
    if last != cert.writer {
        assert((h.happens_before)(last, cert.writer));
        assert((h.happens_before)(cert.writer, last));
        assert(!(h.happens_before)(last, last));
    }
    assert((run.memory)((run.start)(reader))(loc) == (run.memory)((run.end)(cert.writer).unwrap())(
        loc,
    ));
    if cert.slot.is_full() {
        let mem = (run.memory)((run.start)(reader))(loc);
        assert(mem.subrange(0, cert.slot.slot_len as int) =~= mem);
    }
}

pub proof fn lemma_certified_read_observes_slice(
    cert: ValueCertificate,
    source_slot: SlotRef,
    h: TransferHistory,
    run: TransferRun,
    reader: TransferEvent,
)
    requires
        reachable(h),
        valid_run(h, run),
        certificate_valid_in(cert, h, run),
        source_slice_valid(cert, source_slot),
        h.events.contains(reader),
        (h.access)(reader).reads.contains(source_slot.loc),
        !(run.rejected_read)(reader, source_slot.loc),
        last_writer_certified(h, reader, cert),
    ensures
        slot_holds_bytes(
            run,
            (run.start)(reader),
            source_slot,
            source_slice_bytes(cert, source_slot),
        ),
{
    lemma_certified_read_observes_slot(cert, h, run, reader);
    let mem = (run.memory)((run.start)(reader))(source_slot.loc);
    let rel = (source_slot.sub_offset - cert.slot.sub_offset) as int;
    assert(mem.subrange(
        source_slot.sub_offset as int,
        (source_slot.sub_offset + source_slot.sub_len) as int,
    ) =~= cert.bytes.subrange(rel, rel + source_slot.sub_len as int));
}

/// The bytes supplied by a read receipt equal the certified value on a successful read.
pub proof fn lemma_read_returns_certificate(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    cert: ValueCertificate,
    reader: TransferEvent,
    observed: Seq<u8>,
)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        certificate_valid_in(cert, h, run),
        last_writer_certified(h, reader, cert),
        event_succeeded(ledger, node, reader),
        ledger.submitted.dom().contains(reader),
        ledger.submitted.map()[reader].access.reads.contains(cert.slot.loc),
        effect_implies(
            ledger.submitted.map()[reader].effect,
            read_bytes_effect(cert.slot, observed),
        ),
    ensures
        cert.bytes == observed,
{
    assert(!(run.rejected_read)(reader, cert.slot.loc));
    lemma_certified_read_observes_slot(cert, h, run, reader);
    every_event_isolated(h, run, reader);
    let input = restrict::<TransferDomain>(
        (run.memory)((run.start)(reader)),
        (h.access)(reader).reads,
    );
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(reader).unwrap()),
        (h.access)(reader).writes,
    );
    assert(((h.effect)(reader))(input, output));
    assert(read_bytes_effect(cert.slot, observed)(input, output));
    assert(input[cert.slot.loc] == (run.memory)((run.start)(reader))(cert.slot.loc));
    assert(cert.bytes =~= observed);
}

pub proof fn lemma_host_write_certifies(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    event: TransferEvent,
    slot: SlotRef,
    bytes: Seq<u8>,
) -> (cert: ValueCertificate)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        slot.well_formed(),
        slot.sub_len == bytes.len(),
        event_succeeded(ledger, node, event),
        ledger.submitted.dom().contains(event),
        ledger.submitted.map()[event].access.writes.contains(slot.loc),
        effect_implies(ledger.submitted.map()[event].effect, splice_bytes(slot, bytes)),
    ensures
        certificate_valid_in(cert, h, run),
        cert.slot == slot,
        cert.bytes == bytes,
        cert.writer == event,
        cert.proof == ValueProof::HostWrite,
{
    every_event_isolated(h, run, event);
    let input = restrict::<TransferDomain>(
        (run.memory)((run.start)(event)),
        (h.access)(event).reads,
    );
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(event).unwrap()),
        (h.access)(event).writes,
    );
    assert(((h.effect)(event))(input, output));
    assert(splice_bytes(slot, bytes)(input, output));
    assert(output[slot.loc] == (run.memory)((run.end)(event).unwrap())(slot.loc));
    ValueCertificate { slot, writer: event, bytes, proof: ValueProof::HostWrite }
}

/// Universal whole-slot and sub-slice copy preservation: covers Host memcpy,
/// CUDA H2D/D2H/D2D, NVMe read/write, single-SGE RDMA READ/WRITE, scatter
/// slices, and any future copy backend.
pub proof fn lemma_copy_hop_preserves_value(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    source: ValueCertificate,
    source_slot: SlotRef,
    event: TransferEvent,
    dst: SlotRef,
) -> (cert: ValueCertificate)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        certificate_valid_in(source, h, run),
        source_slice_valid(source, source_slot),
        dst.well_formed(),
        source_slot.sub_len == dst.sub_len,
        event_succeeded(ledger, node, event),
        ledger.submitted.dom().contains(event),
        ledger.submitted.map()[event].access.reads.contains(source_slot.loc),
        ledger.submitted.map()[event].access.writes.contains(dst.loc),
        effect_implies(ledger.submitted.map()[event].effect, copy_effect(source_slot, dst)),
        last_writer_certified(h, event, source),
    ensures
        certificate_valid_in(cert, h, run),
        cert.slot == dst,
        cert.writer == event,
        cert.bytes == source_slice_bytes(source, source_slot),
        cert.proof == (ValueProof::CopyHop { source: Box::new(source), source_slot }),
{
    assert(!(run.rejected_read)(event, source_slot.loc));
    lemma_certified_read_observes_slice(source, source_slot, h, run, event);
    every_event_isolated(h, run, event);
    let input = restrict::<TransferDomain>(
        (run.memory)((run.start)(event)),
        (h.access)(event).reads,
    );
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(event).unwrap()),
        (h.access)(event).writes,
    );
    assert(((h.effect)(event))(input, output));
    assert(copy_effect(source_slot, dst)(input, output));
    assert(input[source_slot.loc] == (run.memory)((run.start)(event))(source_slot.loc));
    assert(output[dst.loc] == (run.memory)((run.end)(event).unwrap())(dst.loc));
    ValueCertificate {
        slot: dst,
        writer: event,
        bytes: source_slice_bytes(source, source_slot),
        proof: ValueProof::CopyHop { source: Box::new(source), source_slot },
    }
}

/// Guarded remote copy preservation: a succeeded read guarded by `share` on
/// `source_slot.loc` observes `source.writer`'s value even when the donor
/// later revoked `share` and overwrote `source_slot.loc`, using
/// `lemma_guarded_read_observes_writer` and the share-observation bound.
pub proof fn lemma_guarded_copy_hop_preserves_value(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    source: ValueCertificate,
    source_slot: SlotRef,
    share: TransferShare,
    event: TransferEvent,
    dst: SlotRef,
) -> (cert: ValueCertificate)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        certificate_valid_in(source, h, run),
        source_slice_valid(source, source_slot),
        dst.well_formed(),
        source_slot.sub_len == dst.sub_len,
        event_succeeded(ledger, node, event),
        ledger.submitted.dom().contains(event),
        ledger.submitted.dom().contains(source.writer),
        ledger.submitted.map()[event].access.reads.contains(source_slot.loc),
        ledger.submitted.map()[event].access.writes.contains(dst.loc),
        effect_implies(ledger.submitted.map()[event].effect, copy_effect(source_slot, dst)),
        ledger.submitted.map()[event].preds.contains(source.writer),
        ledger.submitted.map()[event].preds.subset_of(ledger.submitted.dom()),
        ledger.submitted.map()[event].read_guards.dom().contains(source_slot.loc),
        ledger.submitted.map()[event].read_guards[source_slot.loc] == share,
        forall|e: TransferEvent| #[trigger]
            ledger.submitted.dom().contains(e) && ledger.submitted.map()[e].access.writes.contains(
                source_slot.loc,
            ) && e != source.writer ==> (ledger.submitted.map()[source.writer].preds.contains(e)
                || (ledger.revoked.dom().contains((source_slot.loc, share))
                && ledger.submitted.map()[e].preds.contains(
                ledger.revoked.map()[(source_slot.loc, share)],
            ))),
    ensures
        certificate_valid_in(cert, h, run),
        cert.slot == dst,
        cert.writer == event,
        cert.bytes == source_slice_bytes(source, source_slot),
        cert.proof == (ValueProof::CopyHop { source: Box::new(source), source_slot }),
{
    let loc = source_slot.loc;
    let w = source.writer;
    assert(h.events.contains(event));
    assert(h.events.contains(w));
    assert((h.access)(w).writes.contains(loc));
    assert((h.access)(event).reads.contains(loc));
    assert((run.succeeded)(event));
    assert((h.happens_before)(w, event));
    assert forall|e: TransferEvent| #[trigger]
        writers_before(h, event, loc).contains(e) && e != w implies ((h.happens_before)(e, w)
        || foundation::history::read_fenced_before(h, event, e, loc)) by {
        assert((h.happens_before)(e, event));
        assert(ledger.submitted.map()[event].preds.contains(e));
        assert(ledger.submitted.dom().contains(e));
        assert((h.access)(e) == ledger.submitted.map()[e].access);
        assert(ledger.submitted.map()[e].access.writes.contains(loc));
        if ledger.submitted.map()[w].preds.contains(e) {
            assert((h.happens_before)(e, w));
        } else {
            let key = (loc, share);
            assert(ledger.revoked.dom().contains(key));
            let fence = ledger.revoked.map()[key];
            assert(h.revoked.dom().contains(key));
            assert(h.revoked[key] == fence);
            assert((h.read_guards)(event) == ledger.submitted.map()[event].read_guards);
            assert((h.read_guards)(event).dom().contains(loc));
            assert((h.read_guards)(event)[loc] == share);
            assert(ledger.submitted.map()[e].preds.contains(fence));
            assert((h.happens_before)(fence, e));
            assert(foundation::history::read_fenced_before(h, event, e, loc));
        }
    }
    lemma_guarded_read_observes_writer(h, run, event, w, loc);
    let mem = (run.memory)((run.start)(event))(loc);
    let rel = (source_slot.sub_offset - source.slot.sub_offset) as int;
    assert(mem.subrange(
        source_slot.sub_offset as int,
        (source_slot.sub_offset + source_slot.sub_len) as int,
    ) =~= source.bytes.subrange(rel, rel + source_slot.sub_len as int));
    every_event_isolated(h, run, event);
    let input = restrict::<TransferDomain>(
        (run.memory)((run.start)(event)),
        (h.access)(event).reads,
    );
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(event).unwrap()),
        (h.access)(event).writes,
    );
    assert(((h.effect)(event))(input, output));
    assert(copy_effect(source_slot, dst)(input, output));
    assert(input[loc] == mem);
    assert(output[dst.loc] == (run.memory)((run.end)(event).unwrap())(dst.loc));
    ValueCertificate {
        slot: dst,
        writer: event,
        bytes: source_slice_bytes(source, source_slot),
        proof: ValueProof::CopyHop { source: Box::new(source), source_slot },
    }
}

proof fn lemma_gather_prefix_single_event(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
    k: int,
)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        dst.is_full(),
        dst.slot_len == bytes.len(),
        gather_pieces_contiguous(dst, pieces),
        gather_is_single_event(pieces, pieces.last().writer),
        1 <= k <= pieces.len(),
        forall|p: int|
            0 <= p < pieces.len() ==> {
                let piece = #[trigger] pieces[p];
                let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
                &&& source_slice_valid(piece.source, piece.source_slot)
                &&& certificate_valid_in(piece.source, h, run)
                &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
                    piece.dst_offset as int,
                    (piece.dst_offset + piece.source_slot.sub_len) as int,
                )
                &&& event_succeeded(ledger, node, piece.writer)
                &&& ledger.submitted.dom().contains(piece.writer)
                &&& ledger.submitted.map()[piece.writer].access.reads.contains(
                    piece.source_slot.loc,
                )
                &&& ledger.submitted.map()[piece.writer].access.writes.contains(dst.loc)
                &&& effect_implies(
                    ledger.submitted.map()[piece.writer].effect,
                    subrange_copy_effect(piece.source_slot, dst_sub),
                )
                &&& last_writer_certified(h, piece.writer, piece.source)
            },
    ensures
        ({
            let w = pieces.last().writer;
            let end_off = (pieces[k - 1].dst_offset + pieces[k - 1].source_slot.sub_len) as int;
            let mem = (run.memory)((run.end)(w).unwrap())(dst.loc);
            mem.len() == dst.slot_len && mem.subrange(0, end_off) == bytes.subrange(0, end_off)
        }),
    decreases k,
{
    let w = pieces.last().writer;
    let piece = pieces[k - 1];
    assert(piece.writer == w);
    let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
    assert(!(run.rejected_read)(w, piece.source_slot.loc));
    lemma_certified_read_observes_slice(piece.source, piece.source_slot, h, run, w);
    every_event_isolated(h, run, w);
    let input = restrict::<TransferDomain>((run.memory)((run.start)(w)), (h.access)(w).reads);
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(w).unwrap()),
        (h.access)(w).writes,
    );
    assert(((h.effect)(w))(input, output));
    assert(subrange_copy_effect(piece.source_slot, dst_sub)(input, output));
    let mem = (run.memory)((run.end)(w).unwrap())(dst.loc);
    assert(mem == output[dst.loc]);
    let off = piece.dst_offset as int;
    let end_off = (piece.dst_offset + piece.source_slot.sub_len) as int;
    assert(mem.subrange(off, end_off) == bytes.subrange(off, end_off));
    if k == 1 {
        assert(off == 0);
    } else {
        lemma_gather_prefix_single_event(ledger, node, h, run, pieces, dst, bytes, k - 1);
        let prev = pieces[k - 2];
        assert(prev.dst_offset + prev.source_slot.sub_len == piece.dst_offset);
        assert(mem.subrange(0, off) == bytes.subrange(0, off));
        assert forall|idx: int| 0 <= idx < end_off implies mem.subrange(0, end_off)[idx]
            == bytes.subrange(0, end_off)[idx] by {
            if idx < off {
                assert(mem.subrange(0, off)[idx] == bytes.subrange(0, off)[idx]);
            } else {
                assert(mem.subrange(off, end_off)[idx - off] == bytes.subrange(off, end_off)[idx
                    - off]);
            }
        }
        assert(mem.subrange(0, end_off) =~= bytes.subrange(0, end_off));
    }
}

proof fn lemma_gather_prefix_sequential(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
    k: int,
)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        dst.is_full(),
        dst.slot_len == bytes.len(),
        gather_pieces_contiguous(dst, pieces),
        gather_sequential_chain(pieces, h),
        gather_sequential_last_writers(pieces, dst, h),
        1 <= k <= pieces.len(),
        forall|p: int|
            0 <= p < pieces.len() ==> {
                let piece = #[trigger] pieces[p];
                let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
                &&& source_slice_valid(piece.source, piece.source_slot)
                &&& certificate_valid_in(piece.source, h, run)
                &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
                    piece.dst_offset as int,
                    (piece.dst_offset + piece.source_slot.sub_len) as int,
                )
                &&& event_succeeded(ledger, node, piece.writer)
                &&& ledger.submitted.dom().contains(piece.writer)
                &&& ledger.submitted.map()[piece.writer].access.reads.contains(
                    piece.source_slot.loc,
                )
                &&& ledger.submitted.map()[piece.writer].access.writes.contains(dst.loc)
                &&& effect_implies(
                    ledger.submitted.map()[piece.writer].effect,
                    copy_effect(piece.source_slot, dst_sub),
                )
                &&& last_writer_certified(h, piece.writer, piece.source)
                &&& (p > 0 && !dst_sub.is_full()
                    ==> ledger.submitted.map()[piece.writer].access.reads.contains(dst.loc))
            },
    ensures
        ({
            let w = pieces[k - 1].writer;
            let end_off = (pieces[k - 1].dst_offset + pieces[k - 1].source_slot.sub_len) as int;
            let mem = (run.memory)((run.end)(w).unwrap())(dst.loc);
            mem.len() == dst.slot_len && mem.subrange(0, end_off) == bytes.subrange(0, end_off)
        }),
    decreases k,
{
    let piece = pieces[k - 1];
    let w = piece.writer;
    let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
    assert(!(run.rejected_read)(w, piece.source_slot.loc));
    lemma_certified_read_observes_slice(piece.source, piece.source_slot, h, run, w);
    every_event_isolated(h, run, w);
    let input = restrict::<TransferDomain>((run.memory)((run.start)(w)), (h.access)(w).reads);
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(w).unwrap()),
        (h.access)(w).writes,
    );
    assert(((h.effect)(w))(input, output));
    assert(copy_effect(piece.source_slot, dst_sub)(input, output));
    let mem = (run.memory)((run.end)(w).unwrap())(dst.loc);
    assert(mem == output[dst.loc]);
    let off = piece.dst_offset as int;
    let end_off = (piece.dst_offset + piece.source_slot.sub_len) as int;
    assert(mem.subrange(off, end_off) == bytes.subrange(off, end_off));
    if k == 1 {
        assert(off == 0);
    } else {
        lemma_gather_prefix_sequential(ledger, node, h, run, pieces, dst, bytes, k - 1);
        let prev = pieces[k - 2];
        assert(prev.dst_offset + prev.source_slot.sub_len == piece.dst_offset);
        assert((h.happens_before)(prev.writer, piece.writer));
        assert(is_last(h.happens_before, writers_before(h, piece.writer, dst.loc), prev.writer));
        if off > 0 {
            assert(!dst_sub.is_full());
            assert((h.access)(w).reads.contains(dst.loc));
            assert(!(run.rejected_read)(w, dst.loc));
            reachable_is_well_formed(h);
            read_returns_last_write(h, run, w, dst.loc);
            let writers = writers_before(h, w, dst.loc);
            assert(writers.contains(prev.writer));
            assert(writers != ISet::empty());
            let (last, end) = choose|last: TransferEvent, end: nat| #[trigger]
                is_last(h.happens_before, writers, last) && (run.end)(last) == Some(end)
                    && value_at_start(run, w, dst.loc) == #[trigger] (run.memory)(end)(dst.loc);
            if last != prev.writer {
                assert((h.happens_before)(last, prev.writer));
                assert((h.happens_before)(prev.writer, last));
                assert(!(h.happens_before)(last, last));
            }
            assert(input[dst.loc] == (run.memory)((run.end)(prev.writer).unwrap())(dst.loc));
            assert(input[dst.loc].len() == dst.slot_len);
            assert(mem.subrange(0, off) == input[dst.loc].subrange(0, off));
            assert(mem.subrange(0, off) == bytes.subrange(0, off));
            assert forall|idx: int| 0 <= idx < end_off implies mem.subrange(0, end_off)[idx]
                == bytes.subrange(0, end_off)[idx] by {
                if idx < off {
                    assert(mem.subrange(0, off)[idx] == bytes.subrange(0, off)[idx]);
                } else {
                    assert(mem.subrange(off, end_off)[idx - off] == bytes.subrange(off, end_off)[idx
                        - off]);
                }
            }
            assert(mem.subrange(0, end_off) =~= bytes.subrange(0, end_off));
        } else {
            assert(mem.subrange(0, end_off) =~= bytes.subrange(0, end_off));
        }
    }
}

/// Universal multi-piece gather preservation: covers both sequential multi-event
/// gathers (Host/CUDA/NVMe) and single-event multi-SGE transfers (RDMA SG).
pub proof fn lemma_gather_hop_preserves_value(
    ledger: KnowledgeLedger,
    node: NodeId,
    h: TransferHistory,
    run: TransferRun,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
) -> (cert: ValueCertificate)
    requires
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(ledger, node, h, run),
        dst.is_full(),
        dst.slot_len == bytes.len(),
        gather_pieces_contiguous(dst, pieces),
        !gather_is_single_event(pieces, pieces.last().writer) ==> gather_sequential_chain(pieces, h)
            && gather_sequential_last_writers(pieces, dst, h),
        forall|p: int|
            0 <= p < pieces.len() ==> {
                let piece = #[trigger] pieces[p];
                let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
                &&& source_slice_valid(piece.source, piece.source_slot)
                &&& certificate_valid_in(piece.source, h, run)
                &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
                    piece.dst_offset as int,
                    (piece.dst_offset + piece.source_slot.sub_len) as int,
                )
                &&& event_succeeded(ledger, node, piece.writer)
                &&& ledger.submitted.dom().contains(piece.writer)
                &&& ledger.submitted.map()[piece.writer].access.reads.contains(
                    piece.source_slot.loc,
                )
                &&& ledger.submitted.map()[piece.writer].access.writes.contains(dst.loc)
                &&& last_writer_certified(h, piece.writer, piece.source)
                &&& if gather_is_single_event(pieces, pieces.last().writer) {
                    effect_implies(
                        ledger.submitted.map()[piece.writer].effect,
                        subrange_copy_effect(piece.source_slot, dst_sub),
                    )
                } else {
                    &&& effect_implies(
                        ledger.submitted.map()[piece.writer].effect,
                        copy_effect(piece.source_slot, dst_sub),
                    )
                    &&& (p > 0 && !dst_sub.is_full()
                        ==> ledger.submitted.map()[piece.writer].access.reads.contains(dst.loc))
                }
            },
    ensures
        certificate_valid_in(cert, h, run),
        cert.slot == dst,
        cert.bytes == bytes,
        cert.writer == pieces.last().writer,
        cert.proof == (ValueProof::GatherHop { pieces }),
{
    let last_idx = pieces.len() - 1;
    assert(pieces[last_idx] == pieces.last());
    let cert = ValueCertificate {
        slot: dst,
        writer: pieces.last().writer,
        bytes,
        proof: ValueProof::GatherHop { pieces },
    };
    if gather_is_single_event(pieces, pieces.last().writer) {
        lemma_gather_prefix_single_event(
            ledger,
            node,
            h,
            run,
            pieces,
            dst,
            bytes,
            pieces.len() as int,
        );
    } else {
        lemma_gather_prefix_sequential(
            ledger,
            node,
            h,
            run,
            pieces,
            dst,
            bytes,
            pieces.len() as int,
        );
    }
    let mem = (run.memory)((run.end)(cert.writer).unwrap())(dst.loc);
    assert(mem.subrange(0, dst.slot_len as int) == bytes.subrange(0, dst.slot_len as int));
    assert(bytes.subrange(0, dst.slot_len as int) =~= bytes);
    assert(proof_succeeded(cert, h, run)) by {
        assert forall|p: int| 0 <= p < pieces.len() implies {
            let piece = #[trigger] pieces[p];
            proof_succeeded(piece.source, h, run)
        } by {
            assert(certificate_valid_in(pieces[p].source, h, run));
        }
    }
    cert
}

} // verus!
