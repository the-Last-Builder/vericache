#[cfg(verus_only)]
use super::put::scatter_offsets_valid;
use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::cuda::{CudaBuffer, CudaSlice};

verus! {

/// Enqueue one scatter piece `srcs[i]` into `dst_frame` at byte `offset`
/// and advance `cuda_scatter_pieces_wf` by 1 step.
pub(super) fn scatter_enqueue_one(
    gpu_buffer: &CudaBuffer,
    gpu_first_frame: FrameId,
    gpu_num_frames: u32,
    timeline: &mut CudaTimeline,
    core: &mut CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    dst_frame: FrameId,
    Ghost(dst_slot): Ghost<SlotRef>,
    srcs: &[CudaSlice<'_>],
    i: usize,
    offset: usize,
    last_ticket: Ticket,
    Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
    Tracked(dst_tok): Tracked<&mut SlotToken>,
    Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
    Ghost(offsets): Ghost<Seq<usize>>,
    Ghost(bytes): Ghost<Seq<u8>>,
    Ghost(before_core): Ghost<CacheCore>,
    Ghost(pieces): Ghost<Seq<GatherPiece>>,
    Ghost(piece_seqs): Ghost<Seq<u64>>,
) -> (r: BackendResult<(Ghost<TransferEvent>, Ticket, Ghost<GatherPiece>)>)
    requires
        i < srcs.len(),
        offset == offsets[i as int],
        pieces.len() == i,
        piece_seqs.len() == i,
        scatter_offsets_valid(srcs@, offsets, dst_slot.slot_len as usize),
        bytes.len() == dst_slot.slot_len,
        src_certs.len() == srcs@.len(),
        gpu_buffer.well_formed(),
        gpu_buffer.instance_id() == old(core).instance_id(),
        gpu_buffer.node() == old(core).node(),
        gpu_buffer.num_slots() == gpu_num_frames as usize,
        gpu_buffer.space() is GpuAlloc && gpu_buffer.space()->GpuAlloc_device == old(
            timeline,
        ).device,
        gpu_buffer.slot_len() == dst_slot.slot_len,
        gpu_first_frame.0 <= dst_frame.0 && dst_frame.0 < gpu_first_frame.0 + gpu_num_frames,
        gpu_buffer.slot_ref((dst_frame.0 - gpu_first_frame.0) as nat) == dst_slot,
        before_core.well_formed(),
        old(core).well_formed(),
        old(timeline).well_formed(&old(core).timelines, old(core).node(), &old(core).ledger@),
        instance.id() == old(core).instance_id(),
        before_core.node_id == old(core).node_id,
        old(core).instance_id() == before_core.instance_id(),
        old(core).tiers == before_core.tiers,
        old(core).spec == before_core.spec,
        dst_frame.0 < old(core).frames.metas.len(),
        old(core).frames.meta(dst_frame).state == STATE_INITIALIZING,
        old(core).frames.frame(dst_frame).slot == dst_slot,
        dst_slot.is_full(),
        dst_slot.slot_len == old(core).frames.tier_frame_size(
            old(core).frames.frame(dst_frame).tier,
        ),
        dst_slot.loc.node == old(core).node(),
        old(core).timelines.count == before_core.timelines.count,
        old(core).timelines.next_seq[old(timeline).timeline_id as int]
            >= before_core.timelines.next_seq[old(timeline).timeline_id as int],
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            old(core).timelines.events@,
            old(timeline).timeline_id,
            before_core.timelines.next_seq[old(timeline).timeline_id as int],
        ),
        old(core).ledger@.knows == before_core.ledger@.knows,
        old(core).ledger@.succeeded == before_core.ledger@.succeeded,
        old(core).ledger@.revoked == before_core.ledger@.revoked,
        cuda_scatter_pieces_wf(
            before_core.ledger@,
            old(core).ledger@,
            old(core).node(),
            &old(core).timelines,
            old(timeline).timeline_id,
            dst_slot,
            srcs@,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i as int,
        ),
        cuda_scatter_srcs_valid(
            before_core.ledger@,
            old(core).instance_id(),
            old(core).node(),
            dst_slot.loc,
            srcs@,
            src_certs,
            offsets,
            bytes,
        ),
        slot_is_owned(*old(dst_tok), old(core).instance_id(), dst_slot.loc),
        slot_frontier(*old(dst_tok)).readers.subset_of(old(core).citations()),
        i == 0 ==> write_admitted(slot_frontier(*old(dst_tok)), old(core).citations())
            && last_ticket == Ticket::NONE && old(core).ledger@ == before_core.ledger@,
        i > 0 ==> {
            let tl = old(timeline).timeline_id;
            &&& last_ticket != Ticket::NONE
            &&& last_ticket.timeline_id() == tl
            &&& last_ticket.sequence() == piece_seqs[i - 1]
            &&& old(core).timelines.issued(tl, last_ticket.sequence())
            &&& old(core).timelines.events@[(tl, last_ticket.sequence())] == pieces[i - 1].writer
            &&& old(core).timelines.ticket_covers(last_ticket, pieces[i - 1].writer)
            &&& slot_frontier(*old(dst_tok)) == (Frontier {
                last_writer: Some(pieces[i - 1].writer),
                readers: ISet::empty(),
            })
            &&& old(timeline).stream.tail() == ISet::empty().insert(pieces[i - 1].writer)
        },
        forall|k: int|
            i <= k < srcs.len() ==> {
                &&& old(src_perms).dom().contains(k)
                &&& old(src_perms)[k].valid_local(
                    old(core).instance_id(),
                    (#[trigger] srcs@[k]).slot_ref().loc,
                )
                &&& read_admitted(old(src_perms)[k].frontier(), old(core).citations())
                &&& (srcs@[k].slot_ref().loc.space is GpuAlloc
                    && srcs@[k].slot_ref().loc.space->GpuAlloc_device != old(timeline).device ==> (
                old(src_perms)[k].frontier().last_writer matches Some(w) ==> !(
                w.source is RdmaQueuePair)))
            },
        dst_frame.0 < before_core.frames.metas.len(),
        before_core.frames.meta(dst_frame).state == STATE_EMPTY,
        before_core.frames.frame(dst_frame).slot == dst_slot,
        core_transition(
            &before_core,
            old(core),
            Some(old(timeline).timeline_id),
            Some(old(timeline).stream.source().source()),
            FrameTouchMode::PreserveAllInit,
        ),
    ensures
        final(core).well_formed(),
        final(timeline).well_formed(
            &final(core).timelines,
            final(core).node(),
            &final(core).ledger@,
        ),
        core_transition(
            &before_core,
            final(core),
            Some(old(timeline).timeline_id),
            Some(old(timeline).stream.source().source()),
            FrameTouchMode::PreserveAllInit,
        ),
        final(timeline).timeline_id == old(timeline).timeline_id,
        final(timeline).device == old(timeline).device,
        final(timeline).stream.source().source() == old(timeline).stream.source().source(),
        final(core).node_id == old(core).node_id,
        final(core).instance_id() == old(core).instance_id(),
        final(core).tiers == old(core).tiers,
        final(core).frames == old(core).frames,
        final(core).catalog == old(core).catalog,
        final(core).spec == old(core).spec,
        final(core).next_version_seq == old(core).next_version_seq,
        final(core).version_seq_limit == old(core).version_seq_limit,
        final(core).timelines.count == before_core.timelines.count,
        final(core).timelines.next_seq[final(timeline).timeline_id as int]
            >= before_core.timelines.next_seq[final(timeline).timeline_id as int],
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            final(core).timelines.events@,
            final(timeline).timeline_id,
            before_core.timelines.next_seq[final(timeline).timeline_id as int],
        ),
        final(core).ledger@.knows == before_core.ledger@.knows,
        final(core).ledger@.succeeded == before_core.ledger@.succeeded,
        final(core).ledger@.revoked == before_core.ledger@.revoked,
        match r {
            Err(_) => *final(core) == *old(core) && *final(timeline) == *old(timeline)
                && *final(dst_tok) == *old(dst_tok),
            Ok((ev, ticket, new_piece)) => {
                let tl = final(timeline).timeline_id;
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == tl
                &&& final(core).timelines.issued(tl, ticket.sequence())
                &&& final(core).timelines.events@[(tl, ticket.sequence())] == ev@
                &&& final(core).timelines.ticket_covers(ticket, ev@)
                &&& new_piece@ == (GatherPiece {
                    source: src_certs[i as int],
                    source_slot: srcs@[i as int].slot_ref(),
                    dst_offset: offset as u64,
                    writer: ev@,
                })
                &&& cuda_scatter_pieces_wf(
                    before_core.ledger@,
                    final(core).ledger@,
                    final(core).node(),
                    &final(core).timelines,
                    tl,
                    dst_slot,
                    srcs@,
                    src_certs,
                    offsets,
                    pieces.push(new_piece@),
                    piece_seqs.push(ticket.sequence()),
                    ticket,
                    (i + 1) as int,
                )
                &&& slot_owned(
                    *final(dst_tok),
                    final(core).instance_id(),
                    dst_slot.loc,
                    Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                )
                &&& final(timeline).stream.tail() == ISet::empty().insert(ev@)
                &&& forall|k: int|
                    (i + 1) <= k < srcs.len() ==> {
                        &&& final(src_perms).dom().contains(k)
                        &&& final(src_perms)[k].valid_local(
                            final(core).instance_id(),
                            (#[trigger] srcs@[k]).slot_ref().loc,
                        )
                        &&& final(src_perms)[k] == old(src_perms)[k]
                        &&& read_admitted(final(src_perms)[k].frontier(), final(core).citations())
                    }
            },
        },
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(copy_effect);
    hide(effect_implies);
    hide(FrameTable::same_directory);
    hide(FrameTable::preserves_initializing);
    hide(FrameTable::same_other_pins);
    hide(CacheCore::disjoint_from_published);
    hide(TimelineTable::ticket_covers);
    let _ = gpu_num_frames;
    let _ = last_ticket;
    let ghost tl = timeline.timeline_id;
    proof {
        lemma_scatter_piece_pre_enqueue(
            core,
            &before_core,
            timeline.stream,
            tl,
            dst_frame,
            dst_slot,
            srcs@,
            src_certs,
            offsets,
            bytes,
            pieces,
            piece_seqs,
            last_ticket,
            i as int,
            src_perms[i as int].frontier(),
            slot_frontier(*dst_tok),
        );
    }
    let slice_len = srcs[i].len();
    let dst_sub = gpu_buffer.slot_slice((dst_frame.0 - gpu_first_frame.0) as usize).sub_slice(
        offset,
        slice_len,
    );
    let tracked mut p = src_perms.tracked_remove(i as int);
    let ghost before_step_core = *core;
    let ghost before_stream = timeline.stream;
    let step_res = timeline.enqueue_memcpy(
        core,
        Tracked(instance),
        &dst_sub,
        &srcs[i],
        Tracked(&mut p),
        Tracked(dst_tok),
    );
    proof {
        src_perms.tracked_insert(i as int, p);
    }
    let (Ghost(ev), ticket) = match step_res {
        Ok(pair) => pair,
        Err(e) => return Err(e),
    };
    let ghost new_piece = GatherPiece {
        source: src_certs[i as int],
        source_slot: srcs@[i as int].slot_ref(),
        dst_offset: offset as u64,
        writer: ev,
    };
    proof {
        lemma_ledger_citations_monotone(before_step_core.ledger@, core.ledger@, core.node());
        lemma_scatter_piece_post_enqueue(
            &before_core,
            &before_step_core,
            core,
            tl,
            dst_slot,
            srcs@,
            src_certs,
            offsets,
            pieces,
            piece_seqs,
            last_ticket,
            i as int,
            ev,
            ticket,
            before_stream.tail(),
        );
        lemma_core_transition_after_single_enqueue(
            &before_core,
            &before_step_core,
            core,
            tl,
            timeline.stream.source().source(),
            dst_frame,
            dst_slot,
            ev,
            ticket.sequence(),
        );
    }
    Ok((Ghost(ev), ticket, Ghost(new_piece)))
}

impl<'a> CudaBackend<'a> {
    /// Build the gather `PendingPutProof` and park the completed scatter write.
    pub(super) fn scatter_commit(
        &mut self,
        core: &mut CacheCore,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        last_ticket: Ticket,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(srcs): Ghost<Seq<CudaSlice<'_>>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
        Ghost(pieces): Ghost<Seq<GatherPiece>>,
        Ghost(piece_seqs): Ghost<Seq<u64>>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            scatter_offsets_valid(
                srcs,
                offsets,
                old(core).frames.tier_frame_size(old(self).gpu_tier),
            ),
            bytes.len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            src_certs.len() == srcs.len(),
            old(self).owns_gpu(handle.frame),
            old(core).node_id == before_core.node_id,
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers == before_core.tiers,
            old(core).spec == before_core.spec,
            old(core).next_version_seq == before_core.next_version_seq + 1,
            old(core).version_seq_limit == before_core.version_seq_limit,
            old(core).timelines.count == before_core.timelines.count,
            old(core).timelines.next_seq[old(self).timeline.timeline_id as int]
                >= before_core.timelines.next_seq[old(self).timeline.timeline_id as int],
            cuda_timeline_events_preserved(
                before_core.timelines.events@,
                old(core).timelines.events@,
                old(self).timeline.timeline_id,
                before_core.timelines.next_seq[old(self).timeline.timeline_id as int],
            ),
            old(core).ledger@.knows == before_core.ledger@.knows,
            old(core).ledger@.succeeded == before_core.ledger@.succeeded,
            old(core).ledger@.revoked == before_core.ledger@.revoked,
            old(core).frames.same_directory(&before_core.frames),
            old(core).frames.preserves_initializing(&before_core.frames, None),
            handle.frame.0 < before_core.frames.metas.len(),
            before_core.frames.meta(handle.frame).state == STATE_EMPTY,
            handle.frame.0 < old(core).frames.metas.len(),
            old(core).frames.meta(handle.frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(handle.frame).queued(),
            old(core).frames.frame(handle.frame).token is None,
            old(core).frames.frame(handle.frame).tier == handle.tier,
            old(core).frames.frame(handle.frame).slot == handle.slot@,
            old(core).frames.frame(handle.frame).bound_key == Some(handle.key),
            old(core).frames.frame(handle.frame).bound_version == Some(handle.version),
            handle.version == (VersionId {
                origin_node: before_core.node_id,
                seq: before_core.next_version_seq,
            }),
            handle.version.well_formed() && handle.version != VersionId::NONE,
            !before_core.frames.versions@.contains_key(handle.version),
            handle.slot@.is_full(),
            handle.slot@.slot_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            handle.slot@.loc.node == old(core).node(),
            last_ticket != Ticket::NONE,
            last_ticket.timeline_id() == old(self).timeline.timeline_id,
            old(core).timelines.issued(old(self).timeline.timeline_id, last_ticket.sequence()),
            old(core).timelines.valid_ticket(last_ticket),
            pieces.len() == srcs.len(),
            piece_seqs.len() == srcs.len(),
            old(core).timelines.events@[(old(self).timeline.timeline_id, last_ticket.sequence())]
                == pieces.last().writer,
            old(core).timelines.ticket_covers(last_ticket, pieces.last().writer),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                handle.slot@.loc,
                Frontier { last_writer: Some(pieces.last().writer), readers: ISet::empty() },
            ),
            cuda_scatter_pieces_wf(
                before_core.ledger@,
                old(core).ledger@,
                old(core).node(),
                &old(core).timelines,
                old(self).timeline.timeline_id,
                handle.slot@,
                srcs,
                src_certs,
                offsets,
                pieces,
                piece_seqs,
                last_ticket,
                srcs.len() as int,
            ),
            cuda_scatter_srcs_valid(
                before_core.ledger@,
                old(core).instance_id(),
                old(core).node(),
                handle.slot@.loc,
                srcs,
                src_certs,
                offsets,
                bytes,
            ),
            core_transition(
                &before_core,
                old(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).same_pinned_layout(old(self)),
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            core_transition(
                &before_core,
                final(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((version, frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == old(self).timeline.timeline_id
                &&& version.origin_node == before_core.node_id
                &&& version.seq == before_core.next_version_seq
                &&& final(self).owns_gpu(frame)
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(TimelineTable::ticket_covers);
        let _ = last_ticket;
        let ghost tl = self.timeline.timeline_id;
        let ghost put_proof;
        proof {
            lemma_core_wf_facts(&before_core);
            lemma_core_wf_facts(core);
            put_proof =
            lemma_cuda_scatter_build_proof_opaque(
                before_core.ledger@,
                core.ledger@,
                core.node(),
                &core.timelines,
                tl,
                srcs,
                src_certs,
                pieces,
                piece_seqs,
                last_ticket,
                handle.slot@,
                bytes,
                offsets,
            );
            lemma_cuda_scatter_writes_only(
                before_core.ledger@,
                core.ledger@,
                core.node(),
                &core.timelines,
                tl,
                handle.slot@,
                srcs,
                src_certs,
                offsets,
                pieces,
                piece_seqs,
                last_ticket,
                srcs.len() as int,
            );
            assert(core.frames.preserves_initializing(&before_core.frames, Some(handle.frame))) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            assert(core.frames.versions@ == before_core.frames.versions@) by {
                reveal(FrameTable::same_directory);
            }
        }
        let h_frame = handle.frame;
        let h_version = handle.version;
        let sync_point = self.commit_pending_gpu_put(
            core,
            handle,
            Tracked(dst_tok),
            last_ticket,
            Ghost(put_proof),
            false,
            Ghost(before_core),
        )?;
        Ok((h_version, h_frame, last_ticket, sync_point))
    }

    /// Cleanly abort or park-abort a scatter write after a mid-loop error.
    pub(super) fn scatter_abort(
        &mut self,
        core: &mut CacheCore,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        done_count: usize,
        last_ticket: Ticket,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(srcs): Ghost<Seq<CudaSlice<'_>>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(pieces): Ghost<Seq<GatherPiece>>,
        Ghost(piece_seqs): Ghost<Seq<u64>>,
    )
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            old(core).node_id == before_core.node_id,
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers == before_core.tiers,
            old(core).spec == before_core.spec,
            old(core).next_version_seq >= before_core.next_version_seq,
            old(core).version_seq_limit == before_core.version_seq_limit,
            old(core).timelines.count == before_core.timelines.count,
            old(core).timelines.next_seq[old(self).timeline.timeline_id as int]
                >= before_core.timelines.next_seq[old(self).timeline.timeline_id as int],
            cuda_timeline_events_preserved(
                before_core.timelines.events@,
                old(core).timelines.events@,
                old(self).timeline.timeline_id,
                before_core.timelines.next_seq[old(self).timeline.timeline_id as int],
            ),
            old(core).ledger@.knows == before_core.ledger@.knows,
            old(core).ledger@.succeeded == before_core.ledger@.succeeded,
            old(core).ledger@.revoked == before_core.ledger@.revoked,
            old(core).frames.same_directory(&before_core.frames),
            old(core).frames.preserves_initializing(&before_core.frames, None),
            handle.frame.0 < before_core.frames.metas.len(),
            before_core.frames.meta(handle.frame).state == STATE_EMPTY,
            handle.frame.0 < old(core).frames.metas.len(),
            old(core).frames.meta(handle.frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(handle.frame).queued(),
            old(core).frames.frame(handle.frame).token is None,
            old(core).frames.frame(handle.frame).tier == handle.tier,
            old(core).frames.frame(handle.frame).slot == handle.slot@,
            old(core).frames.frame(handle.frame).bound_key == Some(handle.key),
            old(core).frames.frame(handle.frame).bound_version == Some(handle.version),
            slot_is_owned(dst_tok, old(core).instance_id(), handle.slot@.loc),
            slot_frontier(dst_tok).readers.subset_of(old(core).citations()),
            done_count == 0 ==> old(core).ledger@ == before_core.ledger@ && write_admitted(
                slot_frontier(dst_tok),
                old(core).citations(),
            ),
            done_count > 0 ==> {
                &&& last_ticket != Ticket::NONE
                &&& last_ticket.timeline_id() == old(self).timeline.timeline_id
                &&& old(core).timelines.issued(
                    old(self).timeline.timeline_id,
                    last_ticket.sequence(),
                )
                &&& old(core).timelines.valid_ticket(last_ticket)
                &&& (slot_frontier(dst_tok).last_writer matches Some(w) ==> old(
                    core,
                ).citations().contains(w) || old(core).timelines.ticket_covers(last_ticket, w))
                &&& cuda_scatter_pieces_wf(
                    before_core.ledger@,
                    old(core).ledger@,
                    old(core).node(),
                    &old(core).timelines,
                    old(self).timeline.timeline_id,
                    handle.slot@,
                    srcs,
                    src_certs,
                    offsets,
                    pieces,
                    piece_seqs,
                    last_ticket,
                    done_count as int,
                )
            },
            core_transition(
                &before_core,
                old(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).same_pinned_layout(old(self)),
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            core_transition(
                &before_core,
                final(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(TimelineTable::ticket_covers);
        let _ = last_ticket;
        let ghost tl = self.timeline.timeline_id;
        let ghost before_abort_core = *core;
        if done_count == 0 {
            core.abort_write(handle, Tracked(dst_tok));
            proof {
                lemma_cuda_slabs_same_directory(self, &before_abort_core, core);
                assert(core.frames.same_directory(&before_core.frames)) by {
                    reveal(FrameTable::same_directory);
                }
                assert(core.frames.preserves_initializing(&before_core.frames, Some(handle.frame)))
                    by {
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::preserves_initializing);
                }
                lemma_cuda_ledger_writes_only_same(core.ledger@, None);
                lemma_cuda_puts_after_disjoint_step(
                    self.active_gpu_puts@,
                    &before_core,
                    core,
                    tl,
                    Some(handle.frame),
                    None,
                );
                lemma_core_transition_step_no_event(
                    &before_core,
                    &before_abort_core,
                    core,
                    Some(tl),
                    Some(self.timeline.stream.source().source()),
                    Some(handle.frame),
                    None,
                );
            }
        } else {
            proof {
                assert(core.frames.preserves_initializing(&before_core.frames, Some(handle.frame)))
                    by {
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::preserves_initializing);
                }
                lemma_cuda_scatter_writes_only(
                    before_core.ledger@,
                    core.ledger@,
                    core.node(),
                    &core.timelines,
                    tl,
                    handle.slot@,
                    srcs,
                    src_certs,
                    offsets,
                    pieces,
                    piece_seqs,
                    last_ticket,
                    done_count as int,
                );
            }
            self.park_aborted_gpu_put(
                core,
                handle,
                Tracked(dst_tok),
                last_ticket,
                Ghost(before_core),
            );
        }
    }
}

} // verus!
