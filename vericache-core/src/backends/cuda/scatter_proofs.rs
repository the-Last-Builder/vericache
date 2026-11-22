use super::*;
use transfer::cuda::CudaSlice;
use vstd::prelude::*;

verus! {

/// Induction proving every offset in a valid contiguous scatter schedule is
/// bounded by `total_len`.
pub(super) proof fn lemma_scatter_offsets_bounded(
    srcs: Seq<CudaSlice<'_>>,
    offsets: Seq<usize>,
    total_len: usize,
    k: int,
)
    requires
        put::scatter_offsets_valid(srcs, offsets, total_len),
        0 <= k <= srcs.len(),
    ensures
        offsets[k] <= total_len,
    decreases srcs.len() - k,
{
    if k < srcs.len() {
        assert(srcs[k].well_formed());
        lemma_scatter_offsets_bounded(srcs, offsets, total_len, k + 1);
    }
}

proof fn lemma_scatter_offsets_bounded_pieces(
    pieces: Seq<GatherPiece>,
    offsets: Seq<usize>,
    total_len: u64,
    k: int,
)
    requires
        pieces.len() >= 1,
        offsets.len() == pieces.len() + 1,
        offsets[pieces.len() as int] == total_len,
        forall|j: int|
            0 <= j < pieces.len() ==> offsets[j] + (#[trigger] pieces[j]).source_slot.sub_len
                == offsets[j + 1],
        0 <= k <= pieces.len(),
    ensures
        offsets[k] <= total_len,
    decreases pieces.len() - k,
{
    if k < pieces.len() {
        assert(offsets[k] + pieces[k].source_slot.sub_len == offsets[k + 1]);
        lemma_scatter_offsets_bounded_pieces(pieces, offsets, total_len, k + 1);
    }
}

proof fn lemma_copy_effect_subrange_and_access(src: SlotRef, dst_full: SlotRef, off: u64)
    requires
        dst_full.is_full(),
        off + src.sub_len <= dst_full.slot_len,
    ensures
        ({
            let dst_sub = dst_full.sub_slice(off, src.sub_len);
            &&& copy_access(src, dst_sub).reads.contains(src.loc)
            &&& copy_access(src, dst_sub).writes.contains(dst_full.loc)
            &&& (!dst_sub.is_full() ==> copy_access(src, dst_sub).reads.contains(dst_full.loc))
            &&& effect_implies(copy_effect(src, dst_sub), copy_effect(src, dst_sub))
            &&& effect_implies(copy_effect(src, dst_sub), subrange_copy_effect(src, dst_sub))
        }),
{
    broadcast use vstd::iset::group_iset_lemmas;

}

proof fn lemma_scatter_submitted_gather(
    before: KnowledgeLedger,
    staged: KnowledgeLedger,
    node: NodeId,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
    offsets: Seq<usize>,
)
    requires
        before.valid(before.submitted.instance_id(), node),
        complete_receipts(before),
        complete_receipts(staged),
        before.submitted.map().submap_of(staged.submitted.map()),
        revoked_submap(before.revoked, staged.revoked),
        before.citations(node).subset_of(staged.citations(node)),
        before.succeeded.dom().subset_of(staged.succeeded.dom()),
        dst.is_full(),
        dst.slot_len == bytes.len(),
        pieces.len() >= 1,
        offsets.len() == pieces.len() + 1,
        offsets[0] == 0,
        offsets[pieces.len() as int] == dst.slot_len,
        forall|k: int|
            0 <= k < pieces.len() ==> {
                let piece = #[trigger] pieces[k];
                let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
                &&& piece.dst_offset == offsets[k] as u64
                &&& piece.source_slot.sub_len > 0
                &&& offsets[k] + piece.source_slot.sub_len == offsets[k + 1]
                &&& source_slice_valid(piece.source, piece.source_slot)
                &&& piece.source.slot.well_formed()
                &&& piece.source.bytes.len() == piece.source.slot.sub_len
                &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
                    offsets[k] as int,
                    offsets[k + 1] as int,
                )
                &&& piece.source_slot.loc != dst.loc
                &&& replica_valid(before, node, piece.source)
                &&& staged.submitted.dom().contains(piece.writer)
                &&& !before.submitted.dom().contains(piece.writer)
                &&& staged.submitted.map()[piece.writer].access == copy_access(
                    piece.source_slot,
                    dst_sub,
                )
                &&& staged.submitted.map()[piece.writer].effect == copy_effect(
                    piece.source_slot,
                    dst_sub,
                )
                &&& before.citations(node).subset_of(staged.submitted.map()[piece.writer].preds)
            },
        forall|p: int, q: int|
            0 <= p && q == p + 1 && q < pieces.len() ==> {
                let prev = #[trigger] pieces[p];
                let curr = #[trigger] pieces[q];
                &&& staged.submitted.map()[curr.writer].preds.contains(prev.writer)
                &&& forall|x: TransferEvent| #[trigger]
                    staged.submitted.map()[curr.writer].preds.contains(x)
                        && !before.submitted.dom().contains(x) && x != prev.writer
                        ==> staged.submitted.map()[prev.writer].preds.contains(x)
            },
        forall|x: TransferEvent| #[trigger]
            staged.submitted.dom().contains(x) && !before.submitted.dom().contains(x)
                ==> staged.submitted.map()[x].access.writes == ISet::empty().insert(dst.loc) && (x
                != pieces.last().writer
                ==> staged.submitted.map()[pieces.last().writer].preds.contains(x)),
    ensures
        submitted_gather(before, staged, node, pieces, dst, bytes),
{
    hide(replica_valid);
    hide(complete_receipts);
    hide(effect_implies);
    hide(copy_effect);
    hide(subrange_copy_effect);
    hide(copy_access);
    hide(source_slice_bytes);
    hide(source_slice_valid);
    assert(gather_pieces_contiguous(dst, pieces)) by {
        assert(pieces[0].dst_offset == 0);
        let last = pieces.len() - 1;
        assert(pieces[last].dst_offset + pieces[last].source_slot.sub_len == offsets[last + 1]);
        assert forall|p: int| 0 <= p < pieces.len() implies (#[trigger] pieces[p]).dst_offset
            + pieces[p].source_slot.sub_len <= dst.sub_len by {
            assert(offsets[p] + pieces[p].source_slot.sub_len == offsets[p + 1]);
            lemma_scatter_offsets_bounded_pieces(pieces, offsets, dst.slot_len, p + 1);
        }
        assert forall|p: int, q: int| 0 <= p && q == p + 1 && q < pieces.len() implies (
        #[trigger] pieces[p]).dst_offset + pieces[p].source_slot.sub_len == (
        #[trigger] pieces[q]).dst_offset by {
            assert(offsets[p] + pieces[p].source_slot.sub_len == offsets[q]);
            assert(pieces[q].dst_offset == offsets[q] as u64);
        }
    }
    assert forall|k: int| 0 <= k < pieces.len() implies {
        let piece = #[trigger] pieces[k];
        let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
        &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
            piece.dst_offset as int,
            (piece.dst_offset + piece.source_slot.sub_len) as int,
        )
        &&& staged.submitted.map()[piece.writer].access.reads.contains(piece.source_slot.loc)
        &&& staged.submitted.map()[piece.writer].access.writes.contains(dst.loc)
        &&& effect_implies(
            staged.submitted.map()[piece.writer].effect,
            copy_effect(piece.source_slot, dst_sub),
        )
        &&& effect_implies(
            staged.submitted.map()[piece.writer].effect,
            subrange_copy_effect(piece.source_slot, dst_sub),
        )
        &&& (k > 0 && !dst_sub.is_full()
            ==> staged.submitted.map()[piece.writer].access.reads.contains(dst.loc))
    } by {
        let piece = pieces[k];
        assert(offsets[k] + piece.source_slot.sub_len == offsets[k + 1]);
        assert((piece.dst_offset + piece.source_slot.sub_len) as int == offsets[k + 1] as int);
        lemma_scatter_offsets_bounded_pieces(pieces, offsets, dst.slot_len, k + 1);
        lemma_copy_effect_subrange_and_access(piece.source_slot, dst, piece.dst_offset);
    }
    assert forall|p: int, q: int| 0 <= p && q == p + 1 && q < pieces.len() implies {
        let prev = #[trigger] pieces[p];
        let curr = #[trigger] pieces[q];
        &&& staged.submitted.map()[curr.writer].preds.contains(prev.writer)
        &&& forall|x: TransferEvent| #[trigger]
            staged.submitted.map()[curr.writer].preds.contains(x)
                && !before.submitted.dom().contains(x) && x != prev.writer
                ==> !staged.submitted.map()[x].access.writes.contains(dst.loc)
                || staged.submitted.map()[prev.writer].preds.contains(x)
    } by {
        assert(pieces[p] == pieces[p]);
    }
    assert forall|x: TransferEvent| #[trigger]
        staged.submitted.dom().contains(x) && !before.submitted.dom().contains(
            x,
        ) implies staged.submitted.map()[x].access.writes.subset_of(ISet::empty().insert(dst.loc))
        && (x != pieces.last().writer ==> (!staged.submitted.map()[x].access.writes.contains(
        dst.loc,
    ) || staged.submitted.map()[pieces.last().writer].preds.contains(x))) by {
        broadcast use vstd::iset::group_iset_lemmas;

    }
}

/// Opaque scatter loop accumulator invariant (keeps CVC5 queries small in `put_gpu_scatter`).
#[verifier::opaque]
pub(super) open spec fn cuda_scatter_pieces_wf(
    before_ledger: KnowledgeLedger,
    curr_ledger: KnowledgeLedger,
    node: NodeId,
    timelines: &TimelineTable,
    tl: u8,
    dst: SlotRef,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    pieces: Seq<GatherPiece>,
    piece_seqs: Seq<u64>,
    last_ticket: Ticket,
    i: int,
) -> bool {
    &&& pieces.len() == i
    &&& piece_seqs.len() == i
    &&& before_ledger.submitted.map().submap_of(curr_ledger.submitted.map())
    &&& forall|e: TransferEvent| #[trigger]
        curr_ledger.submitted.dom().contains(e) && !before_ledger.submitted.dom().contains(e)
            ==> curr_ledger.submitted.map()[e].access.writes == ISet::empty().insert(dst.loc) && (i
            > 0 && e != pieces[i - 1].writer ==> curr_ledger.submitted.map()[pieces[i
            - 1].writer].preds.contains(e))
    &&& forall|p: int|
        0 <= p < i ==> {
            let piece = #[trigger] pieces[p];
            let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
            &&& piece.source == src_certs[p]
            &&& piece.source_slot == srcs[p].slot_ref()
            &&& piece.dst_offset == offsets[p] as u64
            &&& 1 <= piece_seqs[p] <= last_ticket.sequence()
            &&& timelines.events@[(tl, piece_seqs[p])] == piece.writer
            &&& piece.writer.source.succeeds_on_observe()
            &&& curr_ledger.submitted.dom().contains(piece.writer)
            &&& !before_ledger.submitted.dom().contains(piece.writer)
            &&& curr_ledger.submitted.map()[piece.writer].access == copy_access(
                piece.source_slot,
                dst_sub,
            )
            &&& curr_ledger.submitted.map()[piece.writer].effect == copy_effect(
                piece.source_slot,
                dst_sub,
            )
            &&& before_ledger.citations(node).subset_of(
                curr_ledger.submitted.map()[piece.writer].preds,
            )
        }
    &&& forall|p: int, q: int|
        0 <= p && q == p + 1 && q < i ==> {
            let prev = #[trigger] pieces[p];
            let curr = #[trigger] pieces[q];
            &&& curr_ledger.submitted.map()[curr.writer].preds.contains(prev.writer)
            &&& forall|x: TransferEvent| #[trigger]
                curr_ledger.submitted.map()[curr.writer].preds.contains(x)
                    && !before_ledger.submitted.dom().contains(x) && x != prev.writer
                    ==> curr_ledger.submitted.map()[prev.writer].preds.contains(x)
        }
}

pub(super) proof fn lemma_cuda_scatter_pieces_init(
    before_ledger: KnowledgeLedger,
    node: NodeId,
    timelines: &TimelineTable,
    tl: u8,
    dst: SlotRef,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
)
    ensures
        cuda_scatter_pieces_wf(
            before_ledger,
            before_ledger,
            node,
            timelines,
            tl,
            dst,
            srcs,
            src_certs,
            offsets,
            Seq::empty(),
            Seq::empty(),
            Ticket::NONE,
            0,
        ),
{
    broadcast use vstd::imap::group_imap_lemmas;

    reveal(cuda_scatter_pieces_wf);
}

pub(super) proof fn lemma_cuda_scatter_pieces_step(
    before_ledger: KnowledgeLedger,
    step_ledger: KnowledgeLedger,
    after_ledger: KnowledgeLedger,
    node: NodeId,
    step_timelines: &TimelineTable,
    after_timelines: &TimelineTable,
    tl: u8,
    dst: SlotRef,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    pieces: Seq<GatherPiece>,
    piece_seqs: Seq<u64>,
    last_ticket: Ticket,
    i: int,
    ev: TransferEvent,
    ticket: Ticket,
    stream_tail: ISet<TransferEvent>,
)
    requires
        0 <= i < srcs.len(),
        i < src_certs.len(),
        i + 1 < offsets.len(),
        cuda_scatter_pieces_wf(
            before_ledger,
            step_ledger,
            node,
            step_timelines,
            tl,
            dst,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i,
        ),
        step_ledger.knows == before_ledger.knows,
        after_ledger.knows == step_ledger.knows,
        step_ledger.submitted.map().submap_of(after_ledger.submitted.map()),
        after_ledger.submitted.map() == step_ledger.submitted.map().insert(
            ev,
            after_ledger.submitted.map()[ev],
        ),
        !step_ledger.submitted.dom().contains(ev),
        ev.source.succeeds_on_observe(),
        after_ledger.submitted.map()[ev].access == copy_access(
            srcs[i].slot_ref(),
            dst.sub_slice(offsets[i] as u64, srcs[i].slot_ref().sub_len),
        ),
        after_ledger.submitted.map()[ev].effect == copy_effect(
            srcs[i].slot_ref(),
            dst.sub_slice(offsets[i] as u64, srcs[i].slot_ref().sub_len),
        ),
        after_ledger.submitted.map()[ev].preds == closure_from(
            step_ledger.submitted.map(),
            stream_tail.union(step_ledger.citations(node)),
        ),
        after_ledger.submitted.map()[ev].preds.subset_of(step_ledger.submitted.dom()),
        i == 0 ==> step_ledger.submitted.map() == before_ledger.submitted.map(),
        i > 0 ==> stream_tail == ISet::empty().insert(pieces[i - 1].writer)
            && last_ticket.sequence() < ticket.sequence(),
        after_timelines.events@ == step_timelines.events@.insert((tl, ticket.sequence()), ev),
        1 <= ticket.sequence(),
        i > 0 ==> ticket.sequence() > last_ticket.sequence(),
        srcs[i].slot_ref().sub_len > 0,
        offsets[i] + srcs[i].slot_ref().sub_len <= dst.slot_len,
        dst.is_full(),
    ensures
        ({
            let new_piece = GatherPiece {
                source: src_certs[i],
                source_slot: srcs[i].slot_ref(),
                dst_offset: offsets[i] as u64,
                writer: ev,
            };
            cuda_scatter_pieces_wf(
                before_ledger,
                after_ledger,
                node,
                after_timelines,
                tl,
                dst,
                srcs,
                src_certs,
                offsets,
                pieces.push(new_piece),
                piece_seqs.push(ticket.sequence()),
                ticket,
                i + 1,
            )
        }),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(cuda_scatter_pieces_wf);
    vstd::imap_lib::lemma_submap_of_trans(
        before_ledger.submitted.map(),
        step_ledger.submitted.map(),
        after_ledger.submitted.map(),
    );
    let new_piece = GatherPiece {
        source: src_certs[i],
        source_slot: srcs[i].slot_ref(),
        dst_offset: offsets[i] as u64,
        writer: ev,
    };
    let next_pieces = pieces.push(new_piece);
    let next_seqs = piece_seqs.push(ticket.sequence());
    assert(before_ledger.citations(node).subset_of(step_ledger.citations(node)));
    assert(before_ledger.citations(node).subset_of(after_ledger.submitted.map()[ev].preds));
    assert(after_ledger.submitted.map()[ev].access.writes =~= ISet::empty().insert(dst.loc));
    if i > 0 {
        let prev_w = pieces[i - 1].writer;
        assert(stream_tail.contains(prev_w));
        assert(after_ledger.submitted.map()[ev].preds.contains(prev_w));
        assert forall|x: TransferEvent| #[trigger]
            after_ledger.submitted.map()[ev].preds.contains(x)
                && !before_ledger.submitted.dom().contains(x) && x
                != prev_w implies after_ledger.submitted.map()[prev_w].preds.contains(x) by {
            assert(step_ledger.submitted.dom().contains(x));
        }
        assert forall|e: TransferEvent| #[trigger]
            after_ledger.submitted.dom().contains(e) && !before_ledger.submitted.dom().contains(e)
                && e != ev implies after_ledger.submitted.map()[ev].preds.contains(e) by {
            assert(step_ledger.submitted.dom().contains(e));
            if e != prev_w {
                assert(step_ledger.submitted.map()[prev_w].preds.contains(e));
                assert(stream_tail.union(step_ledger.citations(node)).contains(prev_w));
            }
        }
    }
    assert forall|p: int| 0 <= p < i + 1 implies {
        let piece = #[trigger] next_pieces[p];
        let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
        &&& piece.source == src_certs[p]
        &&& piece.source_slot == srcs[p].slot_ref()
        &&& piece.dst_offset == offsets[p] as u64
        &&& 1 <= next_seqs[p] <= ticket.sequence()
        &&& after_timelines.events@[(tl, next_seqs[p])] == piece.writer
        &&& piece.writer.source.succeeds_on_observe()
        &&& after_ledger.submitted.dom().contains(piece.writer)
        &&& !before_ledger.submitted.dom().contains(piece.writer)
        &&& after_ledger.submitted.map()[piece.writer].access == copy_access(
            piece.source_slot,
            dst_sub,
        )
        &&& after_ledger.submitted.map()[piece.writer].effect == copy_effect(
            piece.source_slot,
            dst_sub,
        )
        &&& before_ledger.citations(node).subset_of(
            after_ledger.submitted.map()[piece.writer].preds,
        )
    } by {
        if p < i {
            assert(next_pieces[p] == pieces[p]);
            assert(next_seqs[p] == piece_seqs[p]);
            assert(piece_seqs[p] != ticket.sequence());
            assert(step_ledger.submitted.dom().contains(pieces[p].writer));
        }
    }
    assert forall|p: int, q: int| 0 <= p && q == p + 1 && q < i + 1 implies {
        let prev = #[trigger] next_pieces[p];
        let curr = #[trigger] next_pieces[q];
        &&& after_ledger.submitted.map()[curr.writer].preds.contains(prev.writer)
        &&& forall|x: TransferEvent| #[trigger]
            after_ledger.submitted.map()[curr.writer].preds.contains(x)
                && !before_ledger.submitted.dom().contains(x) && x != prev.writer
                ==> after_ledger.submitted.map()[prev.writer].preds.contains(x)
    } by {
        if q < i {
            assert(next_pieces[p] == pieces[p]);
            assert(next_pieces[q] == pieces[q]);
            assert(step_ledger.submitted.dom().contains(pieces[p].writer));
            assert(step_ledger.submitted.dom().contains(pieces[q].writer));
        }
    }
}

/// Opaque wrapper for the per-source validity properties in `put_gpu_scatter`
/// (keeps `replica_valid` and `source_slice_bytes` out of the scatter `while` loop).
#[verifier::opaque]
pub(super) open spec fn cuda_scatter_srcs_valid(
    before_ledger: KnowledgeLedger,
    id: InstanceId,
    node: NodeId,
    dst_loc: SlotLoc,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    bytes: Seq<u8>,
) -> bool {
    forall|k: int|
        0 <= k < srcs.len() ==> {
            &&& (#[trigger] srcs[k]).instance_id() == id
            &&& srcs[k].slot_ref().loc.node == node
            &&& srcs[k].slot_ref().loc != dst_loc
            &&& replica_valid(before_ledger, node, src_certs[k])
            &&& source_slice_valid(src_certs[k], srcs[k].slot_ref())
            &&& src_certs[k].slot.well_formed()
            &&& src_certs[k].bytes.len() == src_certs[k].slot.sub_len
            &&& source_slice_bytes(src_certs[k], srcs[k].slot_ref()) == bytes.subrange(
                offsets[k] as int,
                offsets[k + 1] as int,
            )
        }
}

pub(super) proof fn lemma_cuda_scatter_src_at(
    before_ledger: KnowledgeLedger,
    id: InstanceId,
    node: NodeId,
    dst_loc: SlotLoc,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    bytes: Seq<u8>,
    i: int,
)
    requires
        0 <= i < srcs.len(),
        cuda_scatter_srcs_valid(before_ledger, id, node, dst_loc, srcs, src_certs, offsets, bytes),
    ensures
        srcs[i].instance_id() == id,
        srcs[i].slot_ref().loc.node == node,
        srcs[i].slot_ref().loc != dst_loc,
{
    reveal(cuda_scatter_srcs_valid);
}

proof fn lemma_cuda_event_at_prefix_intro(
    ev: TransferEvent,
    ev_map: Map<(u8, u64), TransferEvent>,
    tl: u8,
    s: u64,
    q: u64,
)
    requires
        1 <= q <= s,
        ev_map[(tl, q)] == ev,
        ev.source.succeeds_on_observe(),
    ensures
        cuda_event_at_prefix(ev, ev_map, tl, s),
{
}

/// Build a `PendingPutProof` + `cuda_put_events_covered` from `cuda_scatter_pieces_wf`.
pub(super) proof fn lemma_cuda_scatter_build_proof_opaque(
    before: KnowledgeLedger,
    staged: KnowledgeLedger,
    node: NodeId,
    timelines: &TimelineTable,
    tl: u8,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    pieces: Seq<GatherPiece>,
    piece_seqs: Seq<u64>,
    last_ticket: Ticket,
    dst: SlotRef,
    bytes: Seq<u8>,
    offsets: Seq<usize>,
) -> (p: PendingPutProof)
    requires
        before.valid(before.submitted.instance_id(), node),
        complete_receipts(before),
        complete_receipts(staged),
        staged.knows == before.knows,
        staged.succeeded == before.succeeded,
        staged.revoked == before.revoked,
        dst.is_full(),
        dst.slot_len == bytes.len(),
        bytes.len() <= usize::MAX,
        put::scatter_offsets_valid(srcs, offsets, bytes.len() as usize),
        src_certs.len() == srcs.len(),
        cuda_scatter_pieces_wf(
            before,
            staged,
            node,
            timelines,
            tl,
            dst,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            srcs.len() as int,
        ),
        cuda_scatter_srcs_valid(
            before,
            before.submitted.instance_id(),
            node,
            dst.loc,
            srcs,
            src_certs,
            offsets,
            bytes,
        ),
    ensures
        p == gather_put_proof(pieces, dst, bytes),
        p.certificate.slot == dst,
        p.certificate.writer == pieces.last().writer,
        p.certificate.bytes == bytes,
        p.events.len() == pieces.len(),
        p.pending(staged, node),
        cuda_put_events_covered(p.events, timelines.events@, tl, last_ticket.sequence()),
{
    hide(replica_valid);
    hide(complete_receipts);
    reveal(cuda_scatter_pieces_wf);
    reveal(cuda_scatter_srcs_valid);
    reveal(cuda_put_events_covered);
    lemma_ledger_citations_monotone(before, staged, node);
    assert(before.succeeded.dom().subset_of(staged.succeeded.dom())) by {
        broadcast use vstd::iset::group_iset_lemmas;

    }
    assert forall|k: int| 0 <= k < pieces.len() implies {
        let piece = #[trigger] pieces[k];
        let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
        &&& piece.dst_offset == offsets[k] as u64
        &&& piece.source_slot.sub_len > 0
        &&& offsets[k] + piece.source_slot.sub_len == offsets[k + 1]
        &&& source_slice_valid(piece.source, piece.source_slot)
        &&& piece.source.slot.well_formed()
        &&& piece.source.bytes.len() == piece.source.slot.sub_len
        &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
            offsets[k] as int,
            offsets[k + 1] as int,
        )
        &&& piece.source_slot.loc != dst.loc
        &&& replica_valid(before, node, piece.source)
        &&& staged.submitted.dom().contains(piece.writer)
        &&& !before.submitted.dom().contains(piece.writer)
        &&& staged.submitted.map()[piece.writer].access == copy_access(piece.source_slot, dst_sub)
        &&& staged.submitted.map()[piece.writer].effect == copy_effect(piece.source_slot, dst_sub)
        &&& before.citations(node).subset_of(staged.submitted.map()[piece.writer].preds)
    } by {
        assert(srcs[k].well_formed());
    }
    lemma_scatter_submitted_gather(before, staged, node, pieces, dst, bytes, offsets);
    stage_gather_proof(before, staged, node, pieces, dst, bytes);
    let p = gather_put_proof(pieces, dst, bytes);
    assert forall|k: int| 0 <= k < p.events.len() implies cuda_event_at_prefix(
        #[trigger] p.events[k],
        timelines.events@,
        tl,
        last_ticket.sequence(),
    ) by {
        assert(p.events[k] == pieces[k].writer);
        let q = piece_seqs[k];
        assert(1 <= q <= last_ticket.sequence());
        assert(timelines.events@[(tl, q)] == p.events[k]);
        assert(p.events[k].source.succeeds_on_observe());
        lemma_cuda_event_at_prefix_intro(
            p.events[k],
            timelines.events@,
            tl,
            last_ticket.sequence(),
            q,
        );
    }
    p
}

/// Extract `cuda_ledger_writes_only` from `cuda_scatter_pieces_wf`.
pub(super) proof fn lemma_cuda_scatter_writes_only(
    before_ledger: KnowledgeLedger,
    curr_ledger: KnowledgeLedger,
    node: NodeId,
    timelines: &TimelineTable,
    tl: u8,
    dst: SlotRef,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    pieces: Seq<GatherPiece>,
    piece_seqs: Seq<u64>,
    last_ticket: Ticket,
    i: int,
)
    requires
        curr_ledger.knows == before_ledger.knows,
        curr_ledger.succeeded == before_ledger.succeeded,
        curr_ledger.revoked == before_ledger.revoked,
        cuda_scatter_pieces_wf(
            before_ledger,
            curr_ledger,
            node,
            timelines,
            tl,
            dst,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i,
        ),
    ensures
        cuda_ledger_writes_only(before_ledger, curr_ledger, Some(dst.loc)),
{
    broadcast use vstd::iset::group_iset_lemmas;

    reveal(cuda_scatter_pieces_wf);
    reveal(cuda_ledger_writes_only);
}

/// Opaque reader-ticket coverage invariant for `get_into_gpu_scatter` (prevents
/// CVC5 `forall ... exists` blowup inside the scatter-get loop).
#[verifier::opaque]
pub(super) open spec fn cuda_readers_covered(
    readers: ISet<TransferEvent>,
    core: &CacheCore,
    dst_timeline: u8,
    last_ticket: Ticket,
) -> bool {
    forall|e: TransferEvent| #[trigger]
        readers.contains(e) ==> core.citations().contains(e) || exists|s: u64|
            1 <= s <= last_ticket.sequence() && #[trigger] core.timelines.events@[(dst_timeline, s)]
                == e
}

pub(super) proof fn lemma_cuda_readers_covered_empty(
    readers: ISet<TransferEvent>,
    core: &CacheCore,
    dst_timeline: u8,
    last_ticket: Ticket,
)
    requires
        readers.is_empty(),
    ensures
        cuda_readers_covered(readers, core, dst_timeline, last_ticket),
{
    broadcast use vstd::iset::group_iset_lemmas;

    reveal(cuda_readers_covered);
}

pub(super) proof fn lemma_cuda_readers_covered_step(
    before_readers: ISet<TransferEvent>,
    after_readers: ISet<TransferEvent>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    dst_timeline: u8,
    prev_ticket: Ticket,
    new_ticket: Ticket,
    ev: TransferEvent,
    first: bool,
)
    requires
        after_readers == before_readers.insert(ev),
        first ==> before_readers.is_empty(),
        !first ==> cuda_readers_covered(before_readers, before_core, dst_timeline, prev_ticket)
            && prev_ticket.sequence() < new_ticket.sequence(),
        before_core.ledger@.submitted.map().submap_of(after_core.ledger@.submitted.map()),
        after_core.ledger@.knows == before_core.ledger@.knows,
        after_core.node_id == before_core.node_id,
        after_core.timelines.events@ == before_core.timelines.events@.insert(
            (dst_timeline, new_ticket.sequence()),
            ev,
        ),
        1 <= new_ticket.sequence(),
    ensures
        cuda_readers_covered(after_readers, after_core, dst_timeline, new_ticket),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(cuda_readers_covered);
    assert(before_core.citations().subset_of(after_core.citations()));
    assert forall|e: TransferEvent| #[trigger]
        after_readers.contains(e) implies after_core.citations().contains(e) || exists|s: u64|
        1 <= s <= new_ticket.sequence() && #[trigger] after_core.timelines.events@[(
            dst_timeline,
            s,
        )] == e by {
        if e == ev {
            let s = new_ticket.sequence();
            assert(1 <= s <= new_ticket.sequence());
            assert(after_core.timelines.events@[(dst_timeline, s)] == ev);
        } else {
            assert(!first);
            assert(before_readers.contains(e));
            if !before_core.citations().contains(e) {
                let s = choose|s: u64|
                    1 <= s <= prev_ticket.sequence() && before_core.timelines.events@[(
                        dst_timeline,
                        s,
                    )] == e;
                assert(s < new_ticket.sequence());
                assert(after_core.timelines.events@[(dst_timeline, s)] == e);
            }
        }
    }
}

pub(super) proof fn lemma_cuda_readers_covered_to_returned(
    core: &CacheCore,
    frame: FrameId,
    cap: ShareCap,
    last_ticket: Ticket,
)
    requires
        core.well_formed(),
        frame.0 < core.frames.metas.len(),
        cap.well_formed(),
        cap.instance_id() == core.instance_id(),
        cap.loc() == core.frames.frame(frame).slot.loc,
        last_ticket != Ticket::NONE,
        core.timelines.valid_ticket(last_ticket),
        cuda_readers_covered(cap.frontier().readers, core, last_ticket.timeline_id(), last_ticket),
    ensures
        core.frames.returned_reads(frame, cap, last_ticket, core.kernel_view()),
{
    reveal(cuda_readers_covered);
    lemma_returned_reads_by_ticket(core, frame, cap, last_ticket);
}

pub(super) proof fn lemma_cuda_scatter_citations_monotone(
    before_ledger: KnowledgeLedger,
    curr_ledger: KnowledgeLedger,
    node: NodeId,
    timelines: &TimelineTable,
    tl: u8,
    dst: SlotRef,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    pieces: Seq<GatherPiece>,
    piece_seqs: Seq<u64>,
    last_ticket: Ticket,
    i: int,
)
    requires
        curr_ledger.knows == before_ledger.knows,
        cuda_scatter_pieces_wf(
            before_ledger,
            curr_ledger,
            node,
            timelines,
            tl,
            dst,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i,
        ),
    ensures
        before_ledger.citations(node).subset_of(curr_ledger.citations(node)),
{
    reveal(cuda_scatter_pieces_wf);
    lemma_ledger_citations_monotone(before_ledger, curr_ledger, node);
}

} // verus!
