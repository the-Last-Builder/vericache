use super::*;
#[cfg(verus_only)]
use foundation::tracked::{closure_from, Frontier};
use transfer::cuda::CudaSlice;

verus! {

/// Contiguous byte-offset predicate for a scatter/gather slice sequence.
pub open spec fn scatter_offsets_valid(
    srcs: Seq<CudaSlice<'_>>,
    offsets: Seq<usize>,
    total_len: usize,
) -> bool {
    &&& srcs.len() >= 1
    &&& offsets.len() == srcs.len() + 1
    &&& offsets[0] == 0
    &&& offsets[srcs.len() as int] == total_len
    &&& forall|i: int|
        0 <= i < srcs.len() ==> {
            &&& (#[trigger] srcs[i]).well_formed()
            &&& srcs[i].slot_ref().sub_len > 0
            &&& offsets[i] + srcs[i].slot_ref().sub_len == offsets[i + 1]
        }
}

impl<'a> CudaBackend<'a> {
    /// Asynchronously write one full frame from caller's `CudaSlice` (`src`)
    /// into `dst_tier` (`self.gpu_tier` or attached `pinned_host` tier) via
    /// `cuda::memcpy_async` + `event_record`, parking the write in `core` for
    /// non-blocking completion in `poll_completions`.
    pub fn put_gpu(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            src.well_formed(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().loc.node == old(core).node(),
            src.slot_ref().sub_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            old(src_perm).valid_local(old(core).instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(core).citations()),
            src.slot_ref().loc.space is GpuAlloc && src.slot_ref().loc.space->GpuAlloc_device
                != old(self).device ==> (old(src_perm).frontier().last_writer matches Some(w) ==> !(
            w.source is RdmaQueuePair)),
            replica_valid(old(core).ledger@, old(core).node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(core).frames.metas.len() && (#[trigger] old(core).frames.meta(f)).state
                    == STATE_EMPTY ==> src.slot_ref().loc != old(core).frames.frame(f).slot.loc,
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
            final(core).version_registry == old(core).version_registry,
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            match r {
                Ok((version, frame, ticket, _)) => {
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == old(self).timeline.timeline_id
                    &&& version.origin_node == old(core).node_id
                    &&& version.seq == old(core).next_version_seq
                    &&& final(core).next_version_seq == old(core).next_version_seq + 1
                    &&& ((dst_tier == old(self).gpu_tier && final(self).owns_gpu(frame))
                        || final(self).owns_pinned_host(frame))
                    &&& final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                        final(self).active_gpu_puts@.last(),
                    )
                    &&& final(self).active_gpu_puts@.last().action matches PendingGpuAction::Publish {
                        version: pv,
                        proof,
                        only_if_current: false,
                        ..
                    } && pv == version && proof@.certificate.bytes == src_cert.bytes
                },
                Err(_) => final(self).active_gpu_puts@ == old(self).active_gpu_puts@ || (
                final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                    final(self).active_gpu_puts@.last(),
                ) && (final(self).active_gpu_puts@.last().action is Abort)),
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(copy_effect);
        hide(effect_implies);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(CacheCore::disjoint_from_published);
        hide(TimelineTable::ticket_covers);
        hide(PinnedHostSlab::well_formed);
        let is_gpu = dst_tier.0 == self.gpu_tier.0;
        let is_pinned = match &self.pinned_host {
            Some(h) => dst_tier.0 == h.tier.0,
            None => false,
        };
        if !is_gpu && !is_pinned {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline.timeline_id),
                    Some(self.timeline.stream.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        let ghost before_core = *core;
        let (handle, Tracked(dst_tok)) = match core.begin_write(dst_tier, key) {
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
            assert(src.slot_ref().loc != handle.slot@.loc);
            if is_gpu {
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
            } else {
                reveal(PinnedHostSlab::well_formed);
                let h = self.pinned_host.unwrap();
                lemma_slab_owns_tier_frame(
                    core,
                    h.tier,
                    h.first_frame,
                    h.num_frames,
                    h.reg.space(),
                    handle.frame,
                );
                lemma_slab_frame_slot(
                    core,
                    h.tier,
                    h.first_frame,
                    h.num_frames,
                    h.reg.space(),
                    handle.frame,
                );
            }
            lemma_citations_admit_stream_closure(core, self.timeline.stream, src_perm.frontier());
            lemma_citations_admit_stream_closure(
                core,
                self.timeline.stream,
                slot_frontier(dst_tok),
            );
            lemma_write_disjoint_from_published(
                core,
                handle.frame,
                ISet::empty().insert(handle.slot@.loc),
            );
            reveal(put_gpu_enqueue_ready);
        }
        self.execute_put_gpu(
            core,
            Tracked(instance),
            handle,
            is_gpu,
            src,
            Tracked(src_perm),
            Tracked(dst_tok),
            Ghost(src_cert),
            Ghost(before_core),
        )
    }

    fn enqueue_put_memcpy(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_frame: FrameId,
        is_gpu: bool,
        Ghost(dst_slot): Ghost<SlotRef>,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(Ghost<TransferEvent>, Ticket)>)
        requires
            old(core).well_formed(),
            old(self).timeline.well_formed(
                &old(core).timelines,
                old(core).node(),
                &old(core).ledger@,
            ),
            old(self).gpu_buffer.well_formed(),
            old(self).gpu_buffer.instance_id() == old(core).instance_id(),
            old(self).gpu_buffer.node() == old(core).node(),
            old(self).gpu_buffer.space() is GpuAlloc && old(
                self,
            ).gpu_buffer.space()->GpuAlloc_device == old(self).device,
            old(self).gpu_buffer.num_slots() == old(self).gpu_num_frames as usize,
            old(self).gpu_buffer.slot_len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            old(self).pinned_host matches Some(h) ==> h.well_formed(old(core)) && old(
                core,
            ).frames.tier_frame_size(h.tier) == old(core).frames.tier_frame_size(
                old(self).gpu_tier,
            ),
            instance.id() == old(core).instance_id(),
            src.well_formed(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().loc.node == old(core).node(),
            dst_slot.loc.node == old(core).node(),
            src.slot_ref().sub_len == dst_slot.sub_len,
            src.slot_ref().loc != dst_slot.loc,
            old(src_perm).valid_local(old(core).instance_id(), src.slot_ref().loc),
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst_slot.loc),
            read_admitted(
                old(src_perm).frontier(),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).timeline.stream.tail().union(old(core).citations()),
                ),
            ),
            write_admitted(
                slot_frontier(*old(dst_tok)),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).timeline.stream.tail().union(old(core).citations()),
                ),
            ),
            old(core).disjoint_from_published(ISet::empty().insert(dst_slot.loc)),
            src.slot_ref().loc.space is GpuAlloc && src.slot_ref().loc.space->GpuAlloc_device
                != old(self).device ==> (old(src_perm).frontier().last_writer matches Some(w) ==> !(
            w.source is RdmaQueuePair)),
            is_gpu ==> old(self).owns_gpu(dst_frame) && old(self).gpu_buffer.slot_ref(
                (dst_frame.0 - old(self).gpu_first_frame.0) as nat,
            ) == dst_slot,
            !is_gpu ==> (old(self).pinned_host matches Some(h) && h.owns(dst_frame)
                && h.reg.slot_ref((dst_frame.0 - h.first_frame.0) as nat) == dst_slot),
        ensures
            final(core).well_formed(),
            final(self).timeline.well_formed(
                &final(core).timelines,
                final(core).node(),
                &final(core).ledger@,
            ),
            final(self).device == old(self).device,
            final(self).gpu_buffer == old(self).gpu_buffer,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).gpu_sync_points == old(self).gpu_sync_points,
            final(self).pinned_host == old(self).pinned_host,
            final(self).active_gpu_puts == old(self).active_gpu_puts,
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(self).timeline.device == old(self).timeline.device,
            final(self).timeline.stream.source().source() == old(
                self,
            ).timeline.stream.source().source(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).version_registry == old(core).version_registry,
            final(core).active_leases == old(core).active_leases,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.confirmed_seq == old(core).timelines.confirmed_seq,
            final(core).ledger@.knows == old(core).ledger@.knows,
            final(core).ledger@.succeeded == old(core).ledger@.succeeded,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            match r {
                Err(_) => *final(core) == *old(core) && *final(self) == *old(self)
                    && *final(src_perm) == *old(src_perm) && *final(dst_tok) == *old(dst_tok),
                Ok((ev, ticket)) => {
                    let s = old(core).timelines.next_seq[old(self).timeline.timeline_id as int];
                    &&& ev@ == old(self).timeline.stream.source().next_event()
                    &&& !old(core).ledger@.submitted.dom().contains(ev@)
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == old(self).timeline.timeline_id
                    &&& ticket.sequence() == s
                    &&& final(core).timelines.issued(old(self).timeline.timeline_id, s)
                    &&& final(core).timelines.events@ == old(core).timelines.events@.insert(
                        (old(self).timeline.timeline_id, s),
                        ev@,
                    )
                    &&& final(core).timelines.events@[(old(self).timeline.timeline_id, s)] == ev@
                    &&& final(core).timelines.next_seq@ == old(core).timelines.next_seq@.update(
                        old(self).timeline.timeline_id as int,
                        (s + 1) as u64,
                    )
                    &&& final(core).ledger@.submitted.instance_id() == old(
                        core,
                    ).ledger@.submitted.instance_id()
                    &&& final(core).ledger@.submitted.dom().contains(ev@)
                    &&& final(core).ledger@.submitted.map() == old(
                        core,
                    ).ledger@.submitted.map().insert(
                        ev@,
                        foundation::tracked::EventInfo {
                            access: copy_access(src.slot_ref(), dst_slot),
                            read_guards: old(src_perm).read_guards(),
                            effect: copy_effect(src.slot_ref(), dst_slot),
                            preds: closure_from(
                                old(core).ledger@.submitted.map(),
                                old(self).timeline.stream.tail().union(old(core).citations()),
                            ),
                        },
                    )
                    &&& final(core).ledger@.submitted.map()[ev@].preds.subset_of(
                        old(core).ledger@.submitted.dom(),
                    )
                    &&& old(core).ledger@.submitted.map().submap_of(
                        final(core).ledger@.submitted.map(),
                    )
                    &&& final(src_perm).advanced_read(*old(src_perm), ev@)
                    &&& slot_owned(
                        *final(dst_tok),
                        old(core).instance_id(),
                        dst_slot.loc,
                        Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                    )
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let dst_slice = if is_gpu {
            self.gpu_buffer.slot_slice((dst_frame.0 - self.gpu_first_frame.0) as usize)
        } else {
            match &self.pinned_host {
                Some(h) => h.reg.slot_slice((dst_frame.0 - h.first_frame.0) as usize),
                None => {
                    assert(false);
                    self.gpu_buffer.slot_slice(0)
                },
            }
        };
        self.timeline.enqueue_memcpy(
            core,
            Tracked(instance),
            &dst_slice,
            src,
            Tracked(src_perm),
            Tracked(dst_tok),
        )
    }

    fn execute_put_gpu(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        handle: WriteHandle,
        is_gpu: bool,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Ghost(src_cert): Ghost<ValueCertificate>,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            core_transition(
                &before_core,
                old(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            put_gpu_enqueue_ready(
                old(self),
                old(core),
                &before_core,
                instance,
                handle,
                is_gpu,
                src,
                *old(src_perm),
                dst_tok,
                src_cert,
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
            final(core).version_registry == before_core.version_registry,
            core_transition(
                &before_core,
                final(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            match r {
                Ok((version, frame, ticket, _)) => {
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == old(self).timeline.timeline_id
                    &&& version.origin_node == before_core.node_id
                    &&& version.seq == before_core.next_version_seq
                    &&& final(core).next_version_seq == before_core.next_version_seq + 1
                    &&& ((handle.tier == old(self).gpu_tier && final(self).owns_gpu(frame))
                        || final(self).owns_pinned_host(frame))
                    &&& final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                        final(self).active_gpu_puts@.last(),
                    )
                    &&& final(self).active_gpu_puts@.last().action matches PendingGpuAction::Publish {
                        version: pv,
                        proof,
                        only_if_current: false,
                        ..
                    } && pv == version && proof@.certificate.bytes == src_cert.bytes
                },
                Err(_) => final(self).active_gpu_puts@ == old(self).active_gpu_puts@ || (
                final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                    final(self).active_gpu_puts@.last(),
                ) && (final(self).active_gpu_puts@.last().action is Abort)),
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(CacheCore::disjoint_from_published);
        hide(TimelineTable::ticket_covers);
        hide(PinnedHostSlab::well_formed);
        let ghost after_begin_self = *self;
        let ghost after_begin_core = *core;
        let (ticket, handle, dst_tok, Ghost(put_proof)) = self.enqueue_or_abort_put_gpu(
            core,
            Tracked(instance),
            handle,
            is_gpu,
            src,
            Tracked(src_perm),
            Tracked(dst_tok),
            Ghost(src_cert),
            Ghost(before_core),
        )?;
        self.park_committed_put_gpu(
            core,
            handle,
            dst_tok,
            ticket,
            Ghost(put_proof),
            Ghost(before_core),
            Ghost(after_begin_self),
            Ghost(after_begin_core),
        )
    }

    pub(super) fn enqueue_put_memcpy_proved(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_frame: FrameId,
        is_gpu: bool,
        Ghost(dst_slot): Ghost<SlotRef>,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
        Ghost(src_cert): Ghost<ValueCertificate>,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(Ticket, Ghost<PendingPutProof>)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            old(self).timeline.well_formed(
                &old(core).timelines,
                old(core).node(),
                &old(core).ledger@,
            ),
            old(self).timeline.timeline_id < old(core).timelines.count,
            old(self).timeline.timeline_id < before_core.timelines.count,
            old(self).gpu_buffer.well_formed(),
            old(self).gpu_buffer.instance_id() == old(core).instance_id(),
            old(self).gpu_buffer.node() == old(core).node(),
            old(self).gpu_buffer.space() is GpuAlloc && old(
                self,
            ).gpu_buffer.space()->GpuAlloc_device == old(self).device,
            old(self).gpu_buffer.num_slots() == old(self).gpu_num_frames as usize,
            old(self).gpu_buffer.slot_len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            old(self).pinned_host matches Some(h) ==> h.well_formed(old(core)) && old(
                core,
            ).frames.tier_frame_size(h.tier) == old(core).frames.tier_frame_size(
                old(self).gpu_tier,
            ),
            instance.id() == old(core).instance_id(),
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
            dst_frame.0 < before_core.frames.metas.len(),
            before_core.frames.meta(dst_frame).state != STATE_INITIALIZING,
            dst_frame.0 < old(core).frames.metas.len(),
            old(core).frames.frame(dst_frame).slot == dst_slot,
            dst_slot.is_full(),
            dst_slot.well_formed(),
            dst_slot.slot_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            dst_slot.loc.node == old(core).node(),
            src.well_formed(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().loc.node == old(core).node(),
            src.slot_ref().sub_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            src.slot_ref().loc != dst_slot.loc,
            old(src_perm).valid_local(old(core).instance_id(), src.slot_ref().loc),
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst_slot.loc),
            read_admitted(
                old(src_perm).frontier(),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).timeline.stream.tail().union(old(core).citations()),
                ),
            ),
            write_admitted(
                slot_frontier(*old(dst_tok)),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).timeline.stream.tail().union(old(core).citations()),
                ),
            ),
            replica_valid(old(core).ledger@, old(core).node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            old(core).disjoint_from_published(ISet::empty().insert(dst_slot.loc)),
            src.slot_ref().loc.space is GpuAlloc && src.slot_ref().loc.space->GpuAlloc_device
                != old(self).device ==> (old(src_perm).frontier().last_writer matches Some(w) ==> !(
            w.source is RdmaQueuePair)),
            is_gpu ==> old(self).owns_gpu(dst_frame) && old(self).gpu_buffer.slot_ref(
                (dst_frame.0 - old(self).gpu_first_frame.0) as nat,
            ) == dst_slot,
            !is_gpu ==> (old(self).pinned_host matches Some(h) && h.owns(dst_frame)
                && h.reg.slot_ref((dst_frame.0 - h.first_frame.0) as nat) == dst_slot),
            core_transition(
                &before_core,
                old(core),
                Some(old(self).timeline.timeline_id),
                Some(old(self).timeline.stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
        ensures
            final(core).well_formed(),
            final(self).timeline.well_formed(
                &final(core).timelines,
                final(core).node(),
                &final(core).ledger@,
            ),
            final(self).device == old(self).device,
            final(self).gpu_buffer == old(self).gpu_buffer,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).gpu_sync_points == old(self).gpu_sync_points,
            final(self).pinned_host == old(self).pinned_host,
            final(self).active_gpu_puts == old(self).active_gpu_puts,
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(self).timeline.device == old(self).timeline.device,
            final(self).timeline.stream.source().source() == old(
                self,
            ).timeline.stream.source().source(),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).version_registry == old(core).version_registry,
            final(core).active_leases == old(core).active_leases,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.confirmed_seq == old(core).timelines.confirmed_seq,
            match r {
                Err(_) => *final(core) == *old(core) && *final(self) == *old(self)
                    && *final(src_perm) == *old(src_perm) && *final(dst_tok) == *old(dst_tok),
                Ok((ticket, put_proof)) => {
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == old(self).timeline.timeline_id
                    &&& ticket.sequence() == before_core.timelines.next_seq[old(
                        self,
                    ).timeline.timeline_id as int]
                    &&& final(core).timelines.next_seq@ == before_core.timelines.next_seq@.update(
                        old(self).timeline.timeline_id as int,
                        (ticket.sequence() + 1) as u64,
                    )
                    &&& final(core).timelines.issued(
                        old(self).timeline.timeline_id,
                        ticket.sequence(),
                    )
                    &&& final(core).timelines.events@.contains_key(
                        (old(self).timeline.timeline_id, ticket.sequence()),
                    )
                    &&& final(core).timelines.valid_ticket(ticket)
                    &&& final(core).timelines.ticket_covers(ticket, put_proof@.certificate.writer)
                    &&& cuda_timeline_events_preserved(
                        before_core.timelines.events@,
                        final(core).timelines.events@,
                        old(self).timeline.timeline_id,
                        before_core.timelines.next_seq[old(self).timeline.timeline_id as int],
                    )
                    &&& cuda_ledger_writes_only(
                        before_core.ledger@,
                        final(core).ledger@,
                        Some(dst_slot.loc),
                    )
                    &&& final(core).frames.preserves_initializing(
                        &before_core.frames,
                        Some(dst_frame),
                    )
                    &&& final(core).frames.versions@ == before_core.frames.versions@
                    &&& core_transition(
                        &before_core,
                        final(core),
                        Some(old(self).timeline.timeline_id),
                        Some(old(self).timeline.stream.source().source()),
                        FrameTouchMode::PreserveAllInit,
                    )
                    &&& slot_owned(
                        *final(dst_tok),
                        old(core).instance_id(),
                        dst_slot.loc,
                        Frontier {
                            last_writer: Some(put_proof@.certificate.writer),
                            readers: ISet::empty(),
                        },
                    )
                    &&& put_proof@.certificate.slot == dst_slot
                    &&& put_proof@.certificate.bytes == src_cert.bytes
                    &&& put_proof@.certificate.bytes.len() == final(core).frames.tier_frame_size(
                        old(self).gpu_tier,
                    )
                    &&& put_proof@.certificate.writer == final(core).timelines.events@[(
                        old(self).timeline.timeline_id,
                        ticket.sequence(),
                    )]
                    &&& put_proof@.pending(final(core).ledger@, final(core).node())
                    &&& cuda_put_events_covered(
                        put_proof@.events,
                        final(core).timelines.events@,
                        old(self).timeline.timeline_id,
                        ticket.sequence(),
                    )
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
        hide(TimelineTable::ticket_covers);
        hide(CudaBackend::well_formed);
        hide(PinnedHostSlab::well_formed);
        let ghost after_begin_core = *core;
        let ghost old_tail = self.timeline.stream.tail();
        let (Ghost(ev), ticket) = self.enqueue_put_memcpy(
            core,
            Tracked(instance),
            dst_frame,
            is_gpu,
            Ghost(dst_slot),
            src,
            Tracked(src_perm),
            Tracked(dst_tok),
        )?;
        let ghost put_proof;
        proof {
            put_proof =
            lemma_finish_put_gpu_core_step(
                &before_core,
                &after_begin_core,
                core,
                self.timeline.timeline_id,
                self.timeline.stream.source().source(),
                dst_frame,
                dst_slot,
                src.slot_ref(),
                ev,
                ticket,
                src_cert,
                old_tail,
            );
        }
        Ok((ticket, Ghost(put_proof)))
    }
}

} // verus!
