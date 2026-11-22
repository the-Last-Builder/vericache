use super::put::scatter_offsets_valid;
use super::*;
use foundation::tracked::Frontier;
use transfer::cuda::CudaSlice;
use vstd::prelude::*;

verus! {

pub(super) proof fn lemma_scatter_piece_pre_enqueue(
    core: &CacheCore,
    before_core: &CacheCore,
    stream: transfer::cuda::CudaStream,
    tl: u8,
    dst_frame: FrameId,
    dst_slot: SlotRef,
    srcs: Seq<CudaSlice<'_>>,
    src_certs: Seq<ValueCertificate>,
    offsets: Seq<usize>,
    bytes: Seq<u8>,
    pieces: Seq<GatherPiece>,
    piece_seqs: Seq<u64>,
    last_ticket: Ticket,
    i: int,
    src_frontier: Frontier<TransferDomain>,
    dst_frontier: Frontier<TransferDomain>,
)
    requires
        0 <= i < srcs.len(),
        pieces.len() == i,
        piece_seqs.len() == i,
        scatter_offsets_valid(srcs, offsets, core.frames.frame_size),
        bytes.len() == core.frames.frame_size,
        src_certs.len() == srcs.len(),
        core.well_formed(),
        before_core.well_formed(),
        stream.well_formed(),
        stream.infos().map().submap_of(core.ledger@.submitted.map()),
        stream.tail().subset_of(stream.infos().dom()),
        core.node_id == before_core.node_id,
        dst_frame.0 < core.frames.metas.len(),
        core.frames.meta(dst_frame).state == STATE_INITIALIZING,
        core.frames.frame(dst_frame).slot == dst_slot,
        core.ledger@.knows == before_core.ledger@.knows,
        cuda_scatter_pieces_wf(
            before_core.ledger@,
            core.ledger@,
            core.node(),
            &core.timelines,
            tl,
            dst_slot,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i,
        ),
        cuda_scatter_srcs_valid(
            before_core.ledger@,
            core.instance_id(),
            core.node(),
            dst_slot.loc,
            srcs,
            src_certs,
            offsets,
            bytes,
        ),
        read_admitted(src_frontier, core.citations()),
        dst_frontier.readers.subset_of(core.citations()),
        i == 0 ==> write_admitted(dst_frontier, core.citations()),
        i > 0 ==> dst_frontier == (Frontier {
            last_writer: Some(pieces[i - 1].writer),
            readers: ISet::empty(),
        }) && stream.tail() == ISet::empty().insert(pieces[i - 1].writer),
    ensures
        offsets[i] <= offsets[i + 1] <= core.frames.frame_size,
        srcs[i].well_formed(),
        srcs[i].instance_id() == core.instance_id(),
        srcs[i].slot_ref().loc.node == core.node(),
        srcs[i].slot_ref().loc != dst_slot.loc,
        read_admitted(
            src_frontier,
            foundation::tracked::closure_from(
                core.ledger@.submitted.map(),
                stream.tail().union(core.citations()),
            ),
        ),
        write_admitted(
            dst_frontier,
            foundation::tracked::closure_from(
                core.ledger@.submitted.map(),
                stream.tail().union(core.citations()),
            ),
        ),
        core.disjoint_from_published(ISet::empty().insert(dst_slot.loc)),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(core);
    lemma_scatter_offsets_bounded(srcs, offsets, core.frames.frame_size, i + 1);
    lemma_cuda_scatter_src_at(
        before_core.ledger@,
        core.instance_id(),
        core.node(),
        dst_slot.loc,
        srcs,
        src_certs,
        offsets,
        bytes,
        i,
    );
    assert(srcs[i].well_formed());
    lemma_citations_admit_stream_closure(core, stream, src_frontier);
    if i == 0 {
        lemma_citations_admit_stream_closure(core, stream, dst_frontier);
    } else {
        assert(stream.tail().contains(pieces[i - 1].writer));
        assert(before_core.node() == core.node());
        lemma_cuda_scatter_citations_monotone(
            before_core.ledger@,
            core.ledger@,
            core.node(),
            &core.timelines,
            tl,
            dst_slot,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i,
        );
        assert(before_core.citations().subset_of(core.citations()));
        lemma_iset_subset_closure_union(
            core.ledger@.submitted.map(),
            stream.tail(),
            core.citations(),
            before_core.citations(),
            pieces[i - 1].writer,
        );
    }
    lemma_write_disjoint_from_published(core, dst_frame, ISet::empty().insert(dst_slot.loc));
}

pub(super) proof fn lemma_scatter_piece_post_enqueue(
    before_core: &CacheCore,
    before_step_core: &CacheCore,
    core: &CacheCore,
    tl: u8,
    dst_slot: SlotRef,
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
        before_step_core.timelines.next_seq[tl as int] >= before_core.timelines.next_seq[tl as int],
        ticket != Ticket::NONE,
        ticket.timeline_id() == tl,
        ticket.sequence() == before_step_core.timelines.next_seq[tl as int],
        1 <= ticket.sequence(),
        core.timelines.events@ == before_step_core.timelines.events@.insert(
            (tl, ticket.sequence()),
            ev,
        ),
        core.timelines.events@[(tl, ticket.sequence())] == ev,
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            before_step_core.timelines.events@,
            tl,
            before_core.timelines.next_seq[tl as int],
        ),
        cuda_scatter_pieces_wf(
            before_core.ledger@,
            before_step_core.ledger@,
            core.node(),
            &before_step_core.timelines,
            tl,
            dst_slot,
            srcs,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i,
        ),
        before_step_core.ledger@.knows == before_core.ledger@.knows,
        core.ledger@.knows == before_step_core.ledger@.knows,
        before_step_core.ledger@.submitted.map().submap_of(core.ledger@.submitted.map()),
        core.ledger@.submitted.map() == before_step_core.ledger@.submitted.map().insert(
            ev,
            core.ledger@.submitted.map()[ev],
        ),
        !before_step_core.ledger@.submitted.dom().contains(ev),
        ev.source.succeeds_on_observe(),
        core.ledger@.submitted.map()[ev].access == copy_access(
            srcs[i].slot_ref(),
            dst_slot.sub_slice(offsets[i] as u64, srcs[i].slot_ref().sub_len),
        ),
        core.ledger@.submitted.map()[ev].effect == copy_effect(
            srcs[i].slot_ref(),
            dst_slot.sub_slice(offsets[i] as u64, srcs[i].slot_ref().sub_len),
        ),
        core.ledger@.submitted.map()[ev].preds == foundation::tracked::closure_from(
            before_step_core.ledger@.submitted.map(),
            stream_tail.union(before_step_core.citations()),
        ),
        core.ledger@.submitted.map()[ev].preds.subset_of(before_step_core.ledger@.submitted.dom()),
        before_step_core.node() == core.node(),
        i == 0 ==> before_step_core.ledger@.submitted.map() == before_core.ledger@.submitted.map(),
        i > 0 ==> stream_tail == ISet::empty().insert(pieces[i - 1].writer)
            && last_ticket.sequence() < ticket.sequence(),
        srcs[i].slot_ref().sub_len > 0,
        offsets[i] + srcs[i].slot_ref().sub_len <= dst_slot.slot_len,
        dst_slot.is_full(),
    ensures
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            core.timelines.events@,
            tl,
            before_core.timelines.next_seq[tl as int],
        ),
        core.timelines.ticket_covers(ticket, ev),
        ({
            let new_piece = GatherPiece {
                source: src_certs[i],
                source_slot: srcs[i].slot_ref(),
                dst_offset: offsets[i] as u64,
                writer: ev,
            };
            cuda_scatter_pieces_wf(
                before_core.ledger@,
                core.ledger@,
                core.node(),
                &core.timelines,
                tl,
                dst_slot,
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
    lemma_cuda_events_preserved_step(
        before_core.timelines.events@,
        before_step_core.timelines.events@,
        core.timelines.events@,
        tl,
        before_core.timelines.next_seq[tl as int],
        tl,
        ticket.sequence(),
        ev,
    );
    lemma_cuda_scatter_pieces_step(
        before_core.ledger@,
        before_step_core.ledger@,
        core.ledger@,
        core.node(),
        &before_step_core.timelines,
        &core.timelines,
        tl,
        dst_slot,
        srcs,
        src_certs,
        offsets,
        pieces,
        piece_seqs,
        last_ticket,
        i,
        ev,
        ticket,
        stream_tail,
    );
    lemma_cuda_ticket_covers_exact(&core.timelines, ticket, ev);
}

} // verus!
