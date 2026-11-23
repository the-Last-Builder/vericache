#[cfg(verus_only)]
use super::put::scatter_offsets_valid;
use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::cuda::CudaSlice;

verus! {

impl<'a> CudaBackend<'a> {
    /// Asynchronously gather multiple `CudaSlice`s sequentially into ONE GPU
    /// frame (`1 Frame = 1 SlotLoc`, zero `IMap` splitting!) via `sub_slice` +
    /// `cuda::memcpy_async` and `stage_gather_proof`. On mid-loop failure at
    /// `i == 0`, immediately aborts; at `i > 0`, parks with `last_ticket` and
    /// `PendingGpuAction::Abort` so the frame is safely recycled once the
    /// in-flight DMA finishes.
    pub fn put_gpu_scatter(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        srcs: &[CudaSlice<'_>],
        Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            scatter_offsets_valid(
                srcs@,
                offsets,
                old(core).frames.tier_frame_size(old(self).gpu_tier),
            ),
            bytes.len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            src_certs.len() == srcs@.len(),
            forall|i: int|
                0 <= i < srcs@.len() ==> {
                    &&& (#[trigger] srcs@[i]).instance_id() == old(core).instance_id()
                    &&& srcs@[i].slot_ref().loc.node == old(core).node()
                    &&& old(src_perms).dom().contains(i)
                    &&& old(src_perms)[i].valid_local(
                        old(core).instance_id(),
                        srcs@[i].slot_ref().loc,
                    )
                    &&& read_admitted(old(src_perms)[i].frontier(), old(core).citations())
                    &&& (srcs@[i].slot_ref().loc.space is GpuAlloc
                        && srcs@[i].slot_ref().loc.space->GpuAlloc_device != old(self).device ==> (
                    old(src_perms)[i].frontier().last_writer matches Some(w) ==> !(
                    w.source is RdmaQueuePair)))
                    &&& replica_valid(old(core).ledger@, old(core).node(), src_certs[i])
                    &&& source_slice_valid(src_certs[i], srcs@[i].slot_ref())
                    &&& src_certs[i].slot.well_formed()
                    &&& src_certs[i].bytes.len() == src_certs[i].slot.sub_len
                    &&& source_slice_bytes(src_certs[i], srcs@[i].slot_ref()) == bytes.subrange(
                        offsets[i] as int,
                        offsets[i + 1] as int,
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(core).frames.metas.len() && (#[trigger] old(core).frames.meta(
                            f,
                        )).state == STATE_EMPTY ==> srcs@[i].slot_ref().loc != old(
                            core,
                        ).frames.frame(f).slot.loc
                },
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
                old(core),
                final(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((version, frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == old(self).timeline.timeline_id
                &&& version.origin_node == old(core).node_id
                &&& version.seq == old(core).next_version_seq
                &&& final(self).owns_gpu(frame)
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
        let ghost before_core = *core;
        let (handle, Tracked(dst_tok)) = match core.begin_write(self.gpu_tier, key) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(self.timeline.timeline_id),
                        Some(self.timeline.stream.source().source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_core_wf_facts(&before_core);
            lemma_core_wf_facts(core);
            lemma_initializing_frame_slot_valid(core, handle.frame);
            assert(before_core.frames.frame(handle.frame).slot == handle.slot@) by {
                reveal(FrameTable::same_directory);
            }
            lemma_core_transition_no_event(
                &before_core,
                core,
                Some(self.timeline.timeline_id),
                Some(self.timeline.stream.source().source()),
            );
            lemma_cuda_slabs_same_directory(self, &before_core, core);
            assert(before_core.frames.meta(handle.frame).state == STATE_EMPTY);
            assert(cuda_scatter_srcs_valid(
                before_core.ledger@,
                core.instance_id(),
                core.node(),
                handle.slot@.loc,
                srcs@,
                src_certs,
                offsets,
                bytes,
            )) by {
                reveal(cuda_scatter_srcs_valid);
                assert forall|k: int| 0 <= k < srcs@.len() implies (
                #[trigger] srcs@[k]).slot_ref().loc != handle.slot@.loc by {
                    assert(srcs@[k].slot_ref().loc != before_core.frames.frame(
                        handle.frame,
                    ).slot.loc);
                }
            }
            lemma_slab_owns_tier_frame(
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                handle.frame,
            );
            lemma_slab_frame_slot(
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                handle.frame,
            );
            lemma_write_disjoint_from_published(
                core,
                handle.frame,
                ISet::empty().insert(handle.slot@.loc),
            );
        }
        self.execute_put_gpu_scatter(
            core,
            Tracked(instance),
            handle,
            Tracked(dst_tok),
            srcs,
            Tracked(src_perms),
            Ghost(src_certs),
            Ghost(offsets),
            Ghost(bytes),
            Ghost(before_core),
        )
    }

    fn execute_put_gpu_scatter(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        srcs: &[CudaSlice<'_>],
        Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            instance.id() == old(core).instance_id(),
            scatter_offsets_valid(
                srcs@,
                offsets,
                old(core).frames.tier_frame_size(old(self).gpu_tier),
            ),
            bytes.len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            src_certs.len() == srcs@.len(),
            old(self).owns_gpu(handle.frame),
            old(self).gpu_buffer.slot_ref((handle.frame.0 - old(self).gpu_first_frame.0) as nat)
                == handle.slot@,
            old(core).node_id == before_core.node_id,
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers == before_core.tiers,
            old(core).spec == before_core.spec,
            old(core).next_version_seq == before_core.next_version_seq + 1,
            old(core).version_seq_limit == before_core.version_seq_limit,
            old(core).timelines == before_core.timelines,
            old(core).ledger == before_core.ledger,
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
            slot_is_owned(dst_tok, old(core).instance_id(), handle.slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(core).citations()),
            cuda_scatter_srcs_valid(
                before_core.ledger@,
                old(core).instance_id(),
                old(core).node(),
                handle.slot@.loc,
                srcs@,
                src_certs,
                offsets,
                bytes,
            ),
            forall|k: int|
                0 <= k < srcs@.len() ==> {
                    &&& old(src_perms).dom().contains(k)
                    &&& old(src_perms)[k].valid_local(
                        old(core).instance_id(),
                        (#[trigger] srcs@[k]).slot_ref().loc,
                    )
                    &&& read_admitted(old(src_perms)[k].frontier(), old(core).citations())
                    &&& (srcs@[k].slot_ref().loc.space is GpuAlloc
                        && srcs@[k].slot_ref().loc.space->GpuAlloc_device != old(self).device ==> (
                    old(src_perms)[k].frontier().last_writer matches Some(w) ==> !(
                    w.source is RdmaQueuePair)))
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
        hide(copy_effect);
        hide(effect_implies);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(CacheCore::disjoint_from_published);
        hide(TimelineTable::ticket_covers);
        let tracked mut dst_tok = dst_tok;
        proof {
            assert(before_core.frames.frame(handle.frame).slot == handle.slot@) by {
                reveal(FrameTable::same_directory);
            }
        }
        let (done_count, last_ticket, err_opt, Ghost(pieces), Ghost(piece_seqs)) =
            self.scatter_enqueue_loop(
            core,
            Tracked(instance),
            handle.frame,
            Ghost(handle.slot@),
            srcs,
            Tracked(src_perms),
            Tracked(&mut dst_tok),
            Ghost(src_certs),
            Ghost(offsets),
            Ghost(bytes),
            Ghost(before_core),
        );
        if let Some(e) = err_opt {
            self.scatter_abort(
                core,
                handle,
                Tracked(dst_tok),
                done_count,
                last_ticket,
                Ghost(before_core),
                Ghost(srcs@),
                Ghost(src_certs),
                Ghost(offsets),
                Ghost(pieces),
                Ghost(piece_seqs),
            );
            return Err(e);
        }
        self.scatter_commit(
            core,
            handle,
            Tracked(dst_tok),
            last_ticket,
            Ghost(before_core),
            Ghost(srcs@),
            Ghost(src_certs),
            Ghost(offsets),
            Ghost(bytes),
            Ghost(pieces),
            Ghost(piece_seqs),
        )
    }

    fn scatter_enqueue_loop(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_frame: FrameId,
        Ghost(dst_slot): Ghost<SlotRef>,
        srcs: &[CudaSlice<'_>],
        Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: (usize, Ticket, Option<CoreError>, Ghost<Seq<GatherPiece>>, Ghost<Seq<u64>>))
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            instance.id() == old(core).instance_id(),
            scatter_offsets_valid(
                srcs@,
                offsets,
                old(core).frames.tier_frame_size(old(self).gpu_tier),
            ),
            bytes.len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            src_certs.len() == srcs@.len(),
            old(self).owns_gpu(dst_frame),
            old(self).gpu_buffer.slot_ref((dst_frame.0 - old(self).gpu_first_frame.0) as nat)
                == dst_slot,
            dst_frame.0 < old(core).frames.metas.len(),
            old(core).frames.meta(dst_frame).state == STATE_INITIALIZING,
            old(core).frames.frame(dst_frame).slot == dst_slot,
            dst_slot.is_full(),
            dst_slot.slot_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            dst_slot.loc.node == old(core).node(),
            old(core).node_id == before_core.node_id,
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers == before_core.tiers,
            old(core).spec == before_core.spec,
            old(core).timelines == before_core.timelines,
            old(core).ledger == before_core.ledger,
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst_slot.loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(core).citations()),
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
            forall|k: int|
                0 <= k < srcs@.len() ==> {
                    &&& old(src_perms).dom().contains(k)
                    &&& old(src_perms)[k].valid_local(
                        old(core).instance_id(),
                        (#[trigger] srcs@[k]).slot_ref().loc,
                    )
                    &&& read_admitted(old(src_perms)[k].frontier(), old(core).citations())
                    &&& (srcs@[k].slot_ref().loc.space is GpuAlloc
                        && srcs@[k].slot_ref().loc.space->GpuAlloc_device != old(self).device ==> (
                    old(src_perms)[k].frontier().last_writer matches Some(w) ==> !(
                    w.source is RdmaQueuePair)))
                },
            dst_frame.0 < before_core.frames.metas.len(),
            before_core.frames.meta(dst_frame).state == STATE_EMPTY,
            before_core.frames.frame(dst_frame).slot == dst_slot,
            core_transition(
                &before_core,
                old(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
        ensures
            final(core).well_formed(),
            cuda_backend_pre_park_wf(final(self), final(core), &before_core),
            core_transition(
                &before_core,
                final(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).same_pinned_layout(old(self)),
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(self).timeline.stream.source().source() == old(
                self,
            ).timeline.stream.source().source(),
            final(self).active_gpu_puts == old(self).active_gpu_puts,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).timelines.count == before_core.timelines.count,
            final(core).timelines.next_seq[final(self).timeline.timeline_id as int]
                >= before_core.timelines.next_seq[final(self).timeline.timeline_id as int],
            cuda_timeline_events_preserved(
                before_core.timelines.events@,
                final(core).timelines.events@,
                final(self).timeline.timeline_id,
                before_core.timelines.next_seq[final(self).timeline.timeline_id as int],
            ),
            final(core).ledger@.knows == before_core.ledger@.knows,
            final(core).ledger@.succeeded == before_core.ledger@.succeeded,
            final(core).ledger@.revoked == before_core.ledger@.revoked,
            ({
                let (done, last_ticket, err_opt, pieces, piece_seqs) = r;
                let tl = final(self).timeline.timeline_id;
                &&& done <= srcs.len()
                &&& err_opt.is_none() ==> done == srcs.len()
                &&& err_opt.is_some() ==> done < srcs.len()
                &&& pieces@.len() == done
                &&& piece_seqs@.len() == done
                &&& slot_is_owned(*final(dst_tok), final(core).instance_id(), dst_slot.loc)
                &&& slot_frontier(*final(dst_tok)).readers.subset_of(final(core).citations())
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
                    pieces@,
                    piece_seqs@,
                    last_ticket,
                    done as int,
                )
                &&& (done == 0 ==> final(core).ledger@ == before_core.ledger@ && write_admitted(
                    slot_frontier(*final(dst_tok)),
                    final(core).citations(),
                ))
                &&& (done > 0 ==> {
                    &&& last_ticket != Ticket::NONE
                    &&& last_ticket.timeline_id() == tl
                    &&& last_ticket.sequence() == piece_seqs@[done - 1]
                    &&& final(core).timelines.issued(tl, last_ticket.sequence())
                    &&& final(core).timelines.valid_ticket(last_ticket)
                    &&& final(core).timelines.events@[(tl, last_ticket.sequence())] == pieces@[done
                        - 1].writer
                    &&& final(core).timelines.ticket_covers(last_ticket, pieces@[done - 1].writer)
                    &&& slot_owned(
                        *final(dst_tok),
                        final(core).instance_id(),
                        dst_slot.loc,
                        Frontier {
                            last_writer: Some(pieces@[done - 1].writer),
                            readers: ISet::empty(),
                        },
                    )
                })
            }),
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
        let ghost tl = self.timeline.timeline_id;
        proof {
            lemma_cuda_events_preserved_same(
                core.timelines.events@,
                tl,
                before_core.timelines.next_seq[tl as int],
            );
            lemma_cuda_scatter_pieces_init(
                before_core.ledger@,
                core.node(),
                &core.timelines,
                tl,
                dst_slot,
                srcs@,
                src_certs,
                offsets,
            );
        }
        let mut i: usize = 0;
        let mut offset: usize = 0;
        let mut last_ticket: Ticket = Ticket::NONE;
        let mut err_opt: Option<CoreError> = None;
        let ghost mut pieces: Seq<GatherPiece> = Seq::empty();
        let ghost mut piece_seqs: Seq<u64> = Seq::empty();
        while i < srcs.len()
            invariant_except_break
                err_opt.is_none(),
                forall|k: int|
                    i <= k < srcs.len() ==> {
                        &&& src_perms.dom().contains(k)
                        &&& src_perms[k].valid_local(
                            core.instance_id(),
                            (#[trigger] srcs@[k]).slot_ref().loc,
                        )
                        &&& read_admitted(src_perms[k].frontier(), core.citations())
                        &&& (srcs@[k].slot_ref().loc.space is GpuAlloc
                            && srcs@[k].slot_ref().loc.space->GpuAlloc_device != self.device ==> (
                        src_perms[k].frontier().last_writer matches Some(w) ==> !(
                        w.source is RdmaQueuePair)))
                    },
            invariant
                i <= srcs.len(),
                offset == offsets[i as int],
                pieces.len() == i,
                piece_seqs.len() == i,
                scatter_offsets_valid(srcs@, offsets, core.frames.tier_frame_size(self.gpu_tier)),
                bytes.len() == core.frames.tier_frame_size(self.gpu_tier),
                src_certs.len() == srcs@.len(),
                tl == self.timeline.timeline_id == old(self).timeline.timeline_id,
                self.device == old(self).device,
                self.gpu_buffer == old(self).gpu_buffer,
                self.gpu_tier == old(self).gpu_tier,
                self.gpu_first_frame == old(self).gpu_first_frame,
                self.gpu_num_frames == old(self).gpu_num_frames,
                self.same_pinned_layout(old(self)),
                self.active_gpu_puts == old(self).active_gpu_puts,
                self.owns_gpu(dst_frame),
                self.gpu_buffer.slot_ref((dst_frame.0 - self.gpu_first_frame.0) as nat) == dst_slot,
                before_core.well_formed(),
                core.well_formed(),
                cuda_backend_pre_park_wf(self, core, &before_core),
                dst_frame.0 < before_core.frames.metas.len(),
                before_core.frames.meta(dst_frame).state == STATE_EMPTY,
                before_core.frames.frame(dst_frame).slot == dst_slot,
                self.timeline.stream.source().source() == old(
                    self,
                ).timeline.stream.source().source(),
                core_transition(
                    &before_core,
                    core,
                    Some(tl),
                    Some(old(self).timeline.stream.source().source()),
                    FrameTouchMode::PreserveAllInit,
                ),
                instance.id() == core.instance_id(),
                before_core.node_id == core.node_id,
                before_core.instance_id() == core.instance_id(),
                before_core.tiers == core.tiers,
                before_core.spec == core.spec,
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.frames == old(core).frames,
                core.catalog == old(core).catalog,
                core.spec == old(core).spec,
                core.next_version_seq == old(core).next_version_seq,
                core.version_seq_limit == old(core).version_seq_limit,
                dst_frame.0 < core.frames.metas.len(),
                core.frames.meta(dst_frame).state == STATE_INITIALIZING,
                core.frames.frame(dst_frame).slot == dst_slot,
                dst_slot.is_full(),
                dst_slot.slot_len == core.frames.tier_frame_size(self.gpu_tier),
                dst_slot.loc.node == core.node(),
                core.timelines.count == before_core.timelines.count,
                core.timelines.next_seq[tl as int] >= before_core.timelines.next_seq[tl as int],
                cuda_timeline_events_preserved(
                    before_core.timelines.events@,
                    core.timelines.events@,
                    tl,
                    before_core.timelines.next_seq[tl as int],
                ),
                core.ledger@.knows == before_core.ledger@.knows,
                core.ledger@.succeeded == before_core.ledger@.succeeded,
                core.ledger@.revoked == before_core.ledger@.revoked,
                cuda_scatter_pieces_wf(
                    before_core.ledger@,
                    core.ledger@,
                    core.node(),
                    &core.timelines,
                    tl,
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
                    core.instance_id(),
                    core.node(),
                    dst_slot.loc,
                    srcs@,
                    src_certs,
                    offsets,
                    bytes,
                ),
                slot_is_owned(*dst_tok, core.instance_id(), dst_slot.loc),
                slot_frontier(*dst_tok).readers.subset_of(core.citations()),
                i == 0 ==> write_admitted(slot_frontier(*dst_tok), core.citations()) && last_ticket
                    == Ticket::NONE && core.ledger@ == before_core.ledger@,
                i > 0 ==> {
                    &&& last_ticket != Ticket::NONE
                    &&& last_ticket.timeline_id() == tl
                    &&& last_ticket.sequence() == piece_seqs[i - 1]
                    &&& core.timelines.issued(tl, last_ticket.sequence())
                    &&& core.timelines.events@[(tl, last_ticket.sequence())] == pieces[i - 1].writer
                    &&& core.timelines.ticket_covers(last_ticket, pieces[i - 1].writer)
                    &&& slot_frontier(*dst_tok) == (Frontier {
                        last_writer: Some(pieces[i - 1].writer),
                        readers: ISet::empty(),
                    })
                    &&& self.timeline.stream.tail() == ISet::empty().insert(pieces[i - 1].writer)
                },
            ensures
                err_opt.is_none() ==> i == srcs.len(),
                err_opt.is_some() ==> i < srcs.len(),
            decreases srcs.len() - i,
        {
            proof {
                lemma_scatter_offsets_bounded(
                    srcs@,
                    offsets,
                    core.frames.tier_frame_size(self.gpu_tier),
                    (i + 1) as int,
                );
                lemma_slab_frame_slot(
                    core,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                    dst_frame,
                );
            }
            let slice_len = srcs[i].len();
            let ghost pre_step_core = *core;
            match super::scatter_ops::scatter_enqueue_one(
                &self.gpu_buffer,
                self.gpu_first_frame,
                self.gpu_num_frames,
                &mut self.timeline,
                core,
                Tracked(instance),
                dst_frame,
                Ghost(dst_slot),
                srcs,
                i,
                offset,
                last_ticket,
                Tracked(src_perms),
                Tracked(dst_tok),
                Ghost(src_certs),
                Ghost(offsets),
                Ghost(bytes),
                Ghost(before_core),
                Ghost(pieces),
                Ghost(piece_seqs),
            ) {
                Err(e) => {
                    err_opt = Some(e);
                    break;
                },
                Ok((_ev, ticket, _new_piece)) => {
                    proof {
                        lemma_cuda_slabs_same_frames(self, &pre_step_core, core);
                        pieces = pieces.push(_new_piece@);
                        piece_seqs = piece_seqs.push(ticket.sequence());
                    }
                    last_ticket = ticket;
                    offset += slice_len;
                    i += 1;
                },
            }
        }
        (i, last_ticket, err_opt, Ghost(pieces), Ghost(piece_seqs))
    }
}

} // verus!
