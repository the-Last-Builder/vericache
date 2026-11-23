use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::cuda::{self, CudaSlice, CudaStream};

verus! {

/// External caller stream invariant relative to `core` (for `get_into_gpu` and
/// `get_into_gpu_scatter`).
pub open spec fn caller_stream_valid(
    stream: CudaStream,
    core: &CacheCore,
    dst_timeline: u8,
    backend_stream: CudaStream,
) -> bool {
    &&& stream.well_formed()
    &&& stream.source().id() == core.instance_id()
    &&& stream.source().node() == core.node()
    &&& stream.source().source() is CudaStream
    &&& stream.source().source() != backend_stream.source().source()
    &&& source_fresh_for_ledger(stream.source(), core.ledger@)
    &&& stream.infos().map().submap_of(core.ledger@.submitted.map())
    &&& stream.tail().subset_of(stream.infos().dom())
    &&& dst_timeline < core.timelines.count <= MAX_TIMELINES
    &&& core.timelines.kinds[dst_timeline as int] == (TimelineKind::CudaStream {
        device: stream.device(),
    })
}

impl<'a> CudaBackend<'a> {
    pub(super) fn owned_src_slice(&self, frame: FrameId) -> (s: CudaSlice<'_>)
        requires
            self.gpu_buffer.well_formed(),
            self.gpu_buffer.num_slots() == self.gpu_num_frames as usize,
            self.pinned_host matches Some(h) ==> h.reg.well_formed() && h.reg.num_slots()
                == h.num_frames as usize,
            self.owns_gpu(frame) || self.owns_pinned_host(frame),
        ensures
            s.well_formed(),
            self.owns_gpu(frame) ==> s.slot_ref() == self.gpu_buffer.slot_ref(
                (frame.0 - self.gpu_first_frame.0) as nat,
            ) && s.instance_id() == self.gpu_buffer.instance_id(),
            !self.owns_gpu(frame) ==> {
                let h = self.pinned_host.unwrap();
                s.slot_ref() == h.reg.slot_ref((frame.0 - h.first_frame.0) as nat)
                    && s.instance_id() == h.reg.instance_id()
            },
    {
        if self.owns_gpu_frame(frame) {
            self.gpu_buffer.slot_slice((frame.0 - self.gpu_first_frame.0) as usize)
        } else {
            self.pinned_host_frame_slice(frame)
        }
    }

    pub(super) fn enqueue_on_caller_stream(
        dst_stream: &mut CudaStream,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_timeline: u8,
        dst: &CudaSlice,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
        Ghost(backend_stream): Ghost<CudaStream>,
    ) -> (r: BackendResult<(Ghost<TransferEvent>, Ticket)>)
        requires
            old(core).well_formed(),
            caller_stream_valid(*old(dst_stream), old(core), dst_timeline, backend_stream),
            instance.id() == old(core).instance_id(),
            dst.well_formed(),
            src.well_formed(),
            dst.instance_id() == old(core).instance_id(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().sub_len == dst.slot_ref().sub_len,
            src.slot_ref().loc.node == old(core).node(),
            dst.slot_ref().loc.node == old(core).node(),
            src.slot_ref().loc != dst.slot_ref().loc,
            old(src_perm).valid_local(old(core).instance_id(), src.slot_ref().loc),
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(core).citations()),
            write_admitted(slot_frontier(*old(dst_tok)), old(core).citations()),
            old(core).disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            src.slot_ref().loc.space is GpuAlloc && dst.slot_ref().loc.space is GpuAlloc
                && src.slot_ref().loc.space->GpuAlloc_device
                != dst.slot_ref().loc.space->GpuAlloc_device ==> (old(
                src_perm,
            ).frontier().last_writer matches Some(w) ==> !(w.source is RdmaQueuePair)),
        ensures
            final(core).well_formed(),
            caller_stream_valid(*final(dst_stream), final(core), dst_timeline, backend_stream),
            final(dst_stream).device() == old(dst_stream).device(),
            final(dst_stream).source().source() == old(dst_stream).source().source(),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).version_seq_start == old(core).version_seq_start,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).free_lease_slots == old(core).free_lease_slots,
            final(core).lease_generation == old(core).lease_generation,
            final(core).version_registry == old(core).version_registry,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.confirmed_seq == old(core).timelines.confirmed_seq,
            final(core).ledger@.knows == old(core).ledger@.knows,
            final(core).ledger@.succeeded == old(core).ledger@.succeeded,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            match r {
                Err(_) => *final(core) == *old(core) && *final(dst_stream) == *old(dst_stream)
                    && *final(src_perm) == *old(src_perm) && *final(dst_tok) == *old(dst_tok),
                Ok((ev, ticket)) => {
                    let s = old(core).timelines.next_seq[dst_timeline as int];
                    &&& ev@ == old(dst_stream).source().next_event()
                    &&& !old(core).ledger@.submitted.dom().contains(ev@)
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == dst_timeline
                    &&& ticket.sequence() == s
                    &&& final(core).timelines.issued(dst_timeline, s)
                    &&& final(core).timelines.events@ == old(core).timelines.events@.insert(
                        (dst_timeline, s),
                        ev@,
                    )
                    &&& final(core).timelines.events@[(dst_timeline, s)] == ev@
                    &&& final(core).timelines.next_seq@ == old(core).timelines.next_seq@.update(
                        dst_timeline as int,
                        (s + 1) as u64,
                    )
                    &&& final(core).ledger@.submitted.dom().contains(ev@)
                    &&& final(core).ledger@.submitted.map() == old(
                        core,
                    ).ledger@.submitted.map().insert(ev@, final(core).ledger@.submitted.map()[ev@])
                    &&& final(core).ledger@.submitted.map()[ev@].access.writes
                        == ISet::empty().insert(dst.slot_ref().loc)
                    &&& old(core).ledger@.submitted.map().submap_of(
                        final(core).ledger@.submitted.map(),
                    )
                    &&& final(src_perm).advanced_read(*old(src_perm), ev@)
                    &&& slot_owned(
                        *final(dst_tok),
                        old(core).instance_id(),
                        dst.slot_ref().loc,
                        Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                    )
                    &&& final(core).kernel_view().extends(old(core).kernel_view())
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_core_wf_facts(core);
        }
        if core.timelines.next_seq[dst_timeline as usize] > MAX_TICKET_SEQ {
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before_core = *core;
        let ghost before_stream = *dst_stream;
        let ghost cited = core.citations();
        proof {
            lemma_citations_admit_stream_closure(&before_core, before_stream, src_perm.frontier());
            lemma_citations_admit_stream_closure(
                &before_core,
                before_stream,
                slot_frontier(*dst_tok),
            );
            lemma_cuda_memcpy_admitted(before_stream, *src_perm, *dst_tok, &before_core);
        }
        let copy_res = {
            let tracked ledger_ref = core.ledger.borrow_mut();
            cuda::memcpy_async(
                dst_stream,
                Tracked(instance),
                dst,
                src,
                Tracked(src_perm),
                Tracked(dst_tok),
                Tracked(&mut ledger_ref.submitted),
                Tracked(&ledger_ref.knows),
                Ghost(cited),
            )
        };
        let (ev, receipt) = match copy_res {
            Ok(pair) => pair,
            Err(e) => {
                proof {
                    assert(*core == before_core);
                }
                return Err(cuda_error(e));
            },
        };
        proof {
            lemma_cuda_memcpy_receipt_wf(
                before_stream,
                *dst_stream,
                &before_core,
                dst.slot_ref(),
                src.slot_ref(),
                ev@,
                receipt@,
            );
            lemma_core_wf_same_submitted_map(&before_core, core);
        }
        let ticket = core.issue_ticket(dst_timeline, receipt).unwrap();
        proof {
            lemma_source_fresh_stepped(
                before_stream.source(),
                dst_stream.source(),
                before_core.ledger@,
                core.ledger@,
                ev@,
            );
            assert(dst_stream.infos().map().submap_of(core.ledger@.submitted.map())) by {
                broadcast use vstd::imap::group_imap_lemmas;

            }
            assert(core.ledger@.submitted.map()[ev@].access.writes =~= ISet::empty().insert(
                dst.slot_ref().loc,
            ));
        }
        Ok((ev, ticket))
    }

    /// 1-hop asynchronous GPU <-> Pinned-DRAM copy via `core.begin_copy_hop` ->
    /// `cuda::memcpy_async` -> `release_local_lease` (with ticket) ->
    /// `event_record` -> `park_write`.
    pub fn copy_hop_cuda(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
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
            r matches Ok((dst_frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == old(self).timeline.timeline_id
                &&& ((dst_tier == old(self).gpu_tier && final(self).owns_gpu(dst_frame))
                    || final(self).owns_pinned_host(dst_frame))
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
        let src_is_gpu = self.owns_gpu_frame(src_frame);
        let src_is_pinned = self.owns_pinned_host_frame(src_frame);
        let dst_is_gpu = dst_tier.0 == self.gpu_tier.0;
        let dst_is_pinned = match &self.pinned_host {
            Some(h) => dst_tier.0 == h.tier.0,
            None => false,
        };
        if !(src_is_gpu || src_is_pinned) || !(dst_is_gpu || dst_is_pinned) {
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
        if src_is_pinned && dst_is_pinned {
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
        let (hop, Tracked(src_cap), Tracked(mut dst_tok)) = match core.begin_copy_hop(
            src_frame,
            dst_tier,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_cuda_slabs_same_directory(self, &before_core, core);
                    lemma_cuda_timeline_preserved(&self.timeline, &before_core, core);
                    lemma_cuda_events_preserved_same(
                        core.timelines.events@,
                        self.timeline.timeline_id,
                        before_core.timelines.next_seq[self.timeline.timeline_id as int],
                    );
                    lemma_cuda_ledger_writes_only_same(core.ledger@, None);
                    lemma_cuda_puts_after_disjoint_step(
                        self.active_gpu_puts@,
                        &before_core,
                        core,
                        self.timeline.timeline_id,
                        None,
                        None,
                    );
                    lemma_core_transition_no_event(
                        &before_core,
                        core,
                        Some(self.timeline.timeline_id),
                        Some(self.timeline.stream.source().source()),
                    );
                }
                return Err(e);
            },
        };
        let tracked src_perm = ReadPerm::Shared(src_cap);
        proof {
            lemma_copy_hop_slots_ready(
                self,
                &before_core,
                core,
                hop.src_frame,
                hop.dst_frame,
                src_is_gpu,
                dst_is_gpu,
                src_perm.frontier(),
                slot_frontier(dst_tok),
            );
            lemma_core_transition_no_event(
                &before_core,
                core,
                Some(self.timeline.timeline_id),
                Some(self.timeline.stream.source().source()),
            );
        }
        self.execute_copy_hop_cuda(
            core,
            Tracked(instance),
            hop,
            src_is_gpu,
            dst_is_gpu,
            Tracked(src_perm),
            Tracked(dst_tok),
            Ghost(before_core),
        )
    }

    fn enqueue_hop_memcpy(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        src_frame: FrameId,
        dst_frame: FrameId,
        src_is_gpu: bool,
        dst_is_gpu: bool,
        Ghost(src_slot): Ghost<SlotRef>,
        Ghost(dst_slot): Ghost<SlotRef>,
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
            old(self).gpu_buffer.num_slots() == old(self).gpu_num_frames as usize,
            old(self).gpu_buffer.slot_len() == old(core).frames.tier_frame_size(old(self).gpu_tier),
            old(self).pinned_host matches Some(h) ==> h.well_formed(old(core)) && old(
                core,
            ).frames.tier_frame_size(h.tier) == old(core).frames.tier_frame_size(
                old(self).gpu_tier,
            ),
            instance.id() == old(core).instance_id(),
            src_slot.sub_len == dst_slot.sub_len,
            src_slot.loc.node == old(core).node(),
            dst_slot.loc.node == old(core).node(),
            src_slot.loc != dst_slot.loc,
            old(src_perm).valid_local(old(core).instance_id(), src_slot.loc),
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
            src_is_gpu ==> old(self).owns_gpu(src_frame) && old(self).gpu_buffer.slot_ref(
                (src_frame.0 - old(self).gpu_first_frame.0) as nat,
            ) == src_slot,
            old(self).gpu_buffer.space() is GpuAlloc && old(
                self,
            ).gpu_buffer.space()->GpuAlloc_device == old(self).device,
            !src_is_gpu ==> (old(self).pinned_host matches Some(h) && h.owns(src_frame)
                && h.reg.slot_ref((src_frame.0 - h.first_frame.0) as nat) == src_slot),
            dst_is_gpu ==> old(self).owns_gpu(dst_frame) && old(self).gpu_buffer.slot_ref(
                (dst_frame.0 - old(self).gpu_first_frame.0) as nat,
            ) == dst_slot,
            !dst_is_gpu ==> (old(self).pinned_host matches Some(h) && h.owns(dst_frame)
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
            final(core).active_leases == old(core).active_leases,
            final(core).free_lease_slots == old(core).free_lease_slots,
            final(core).lease_generation == old(core).lease_generation,
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
                            access: copy_access(src_slot, dst_slot),
                            read_guards: old(src_perm).read_guards(),
                            effect: copy_effect(src_slot, dst_slot),
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
                    &&& final(core).kernel_view().extends(old(core).kernel_view())
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let src_slice = if src_is_gpu {
            self.gpu_buffer.slot_slice((src_frame.0 - self.gpu_first_frame.0) as usize)
        } else {
            match &self.pinned_host {
                Some(h) => h.reg.slot_slice((src_frame.0 - h.first_frame.0) as usize),
                None => {
                    assert(false);
                    self.gpu_buffer.slot_slice(0)
                },
            }
        };
        let dst_slice = if dst_is_gpu {
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
        proof {
            if src_is_gpu && dst_is_gpu {
                assert(src_slice.slot_ref().loc.space == self.gpu_buffer.space());
                assert(dst_slice.slot_ref().loc.space == self.gpu_buffer.space());
            }
        }
        self.timeline.enqueue_memcpy(
            core,
            Tracked(instance),
            &dst_slice,
            &src_slice,
            Tracked(src_perm),
            Tracked(dst_tok),
        )
    }

    fn execute_copy_hop_cuda(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        hop: CopyHopHandle,
        src_is_gpu: bool,
        dst_is_gpu: bool,
        Tracked(src_perm): Tracked<ReadPerm>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            instance.id() == old(core).instance_id(),
            old(core).node_id == before_core.node_id,
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers == before_core.tiers,
            old(core).spec == before_core.spec,
            old(core).next_version_seq == before_core.next_version_seq,
            old(core).version_seq_limit == before_core.version_seq_limit,
            old(core).timelines.count == before_core.timelines.count,
            old(core).timelines.next_seq == before_core.timelines.next_seq,
            old(core).timelines.events == before_core.timelines.events,
            old(core).ledger == before_core.ledger,
            old(core).frames.same_directory(&before_core.frames),
            old(core).frames.preserves_initializing(&before_core.frames, None),
            hop.src_frame.0 < old(core).frames.metas.len(),
            hop.dst_frame.0 < old(core).frames.metas.len(),
            hop.dst_frame.0 < before_core.frames.metas.len(),
            before_core.frames.meta(hop.dst_frame).state != STATE_INITIALIZING,
            hop.src_slot@ == old(core).frames.frame(hop.src_frame).slot,
            hop.dst_slot@ == old(core).frames.frame(hop.dst_frame).slot,
            hop.src_slot@.loc != hop.dst_slot@.loc,
            hop.src_slot@.is_full(),
            hop.src_slot@.well_formed(),
            hop.src_slot@.slot_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            hop.dst_slot@.is_full(),
            hop.dst_slot@.well_formed(),
            hop.dst_slot@.slot_len == old(core).frames.tier_frame_size(old(self).gpu_tier),
            hop.dst_slot@.loc.node == old(core).node(),
            hop.src_slot@.loc.node == old(core).node(),
            old(core).frames.meta(hop.src_frame).state == STATE_PUBLISHED,
            old(core).frames.meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(hop.dst_frame).queued(),
            old(core).frames.frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames.frame(hop.dst_frame).bound_key == Some(hop.key),
            old(core).frames.frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames.frame(hop.dst_frame).token is None,
            hop.version.well_formed() && hop.version != VersionId::NONE,
            hop.version.origin_node != old(core).node_id || hop.version.seq < old(
                core,
            ).next_version_seq || hop.version.seq >= old(core).version_seq_limit,
            old(core).frames.versions@.contains_key(hop.version),
            old(core).frames.versions@[hop.version] == hop.src_cert@.bytes,
            hop.src_cert@.slot == hop.src_slot@,
            hop.src_cert@.bytes.len() == hop.src_slot@.sub_len,
            replica_valid(old(core).ledger@, old(core).node(), hop.src_cert@),
            old(core).citations().contains(hop.src_cert@.writer),
            hop.src_lease.frame == hop.src_frame,
            old(core).active_lease_matches(hop.src_lease),
            src_perm is Shared,
            old(core).active_leases[hop.src_lease.slot as int]->Some_0.share@
                == src_perm->Shared_0.share(),
            src_perm.valid_local(old(core).instance_id(), hop.src_slot@.loc),
            src_perm.frontier() == (Frontier {
                last_writer: Some(hop.src_cert@.writer),
                readers: ISet::empty(),
            }),
            read_admitted(
                src_perm.frontier(),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).timeline.stream.tail().union(old(core).citations()),
                ),
            ),
            slot_is_owned(dst_tok, old(core).instance_id(), hop.dst_slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(core).citations()),
            write_admitted(
                slot_frontier(dst_tok),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).timeline.stream.tail().union(old(core).citations()),
                ),
            ),
            old(core).frames.returned_reads(
                hop.src_frame,
                src_perm->Shared_0,
                Ticket::NONE,
                old(core).kernel_view(),
            ),
            old(core).disjoint_from_published(ISet::empty().insert(hop.dst_slot@.loc)),
            src_is_gpu ==> old(self).owns_gpu(hop.src_frame) && old(self).gpu_buffer.slot_ref(
                (hop.src_frame.0 - old(self).gpu_first_frame.0) as nat,
            ) == hop.src_slot@,
            !src_is_gpu ==> (old(self).pinned_host matches Some(h) && h.owns(hop.src_frame)
                && h.reg.slot_ref((hop.src_frame.0 - h.first_frame.0) as nat) == hop.src_slot@),
            dst_is_gpu ==> hop.dst_tier == old(self).gpu_tier && old(self).owns_gpu(hop.dst_frame)
                && old(self).gpu_buffer.slot_ref(
                (hop.dst_frame.0 - old(self).gpu_first_frame.0) as nat,
            ) == hop.dst_slot@,
            !dst_is_gpu ==> (old(self).pinned_host matches Some(h) && hop.dst_tier == h.tier
                && h.owns(hop.dst_frame) && h.reg.slot_ref(
                (hop.dst_frame.0 - h.first_frame.0) as nat,
            ) == hop.dst_slot@),
            (hop.src_lease.slot as usize) < before_core.active_leases.len(),
            before_core.active_leases[hop.src_lease.slot as int].is_none(),
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
            r matches Ok((dst_frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == old(self).timeline.timeline_id
                &&& ((hop.dst_tier == old(self).gpu_tier && final(self).owns_gpu(dst_frame))
                    || final(self).owns_pinned_host(dst_frame))
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
        let ghost after_hop_core = *core;
        let tracked mut src_perm = src_perm;
        let tracked mut dst_tok = dst_tok;
        let copy_res = self.enqueue_hop_memcpy(
            core,
            Tracked(instance),
            hop.src_frame,
            hop.dst_frame,
            src_is_gpu,
            dst_is_gpu,
            hop.src_slot,
            hop.dst_slot,
            Tracked(&mut src_perm),
            Tracked(&mut dst_tok),
        );
        let tracked src_cap = src_perm.into_shared();
        let (Ghost(ev), ticket) = match copy_res {
            Ok(pair) => pair,
            Err(e) => {
                self.abort_unsubmitted_copy_hop(
                    core,
                    hop,
                    Tracked(src_cap),
                    Tracked(dst_tok),
                    Tracked(instance),
                    Ghost(before_core),
                );
                return Err(e);
            },
        };
        proof {
            lemma_core_wf_facts(core);
            lemma_cuda_slabs_same_frames(self, &after_hop_core, core);
            assert(src_cap.frontier().readers =~= ISet::empty().insert(ev));
            lemma_non_init_frame_disjoint_from_init(&before_core, hop.dst_frame);
            assert(before_core.frames.frame(hop.dst_frame).slot == hop.dst_slot@) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.ledger@.submitted.dom() =~= after_hop_core.ledger@.submitted.dom().insert(
                ev,
            ));
            assert(core.ledger@.submitted.map()[ev].access.writes =~= ISet::empty().insert(
                hop.dst_slot@.loc,
            ));
            lemma_core_transition_step_single_event(
                &before_core,
                &after_hop_core,
                core,
                self.timeline.timeline_id,
                self.timeline.stream.source().source(),
                ev,
                hop.dst_slot@.loc,
            );
        }
        self.finish_copy_hop_cuda(
            core,
            Tracked(instance),
            hop,
            Tracked(src_cap),
            Tracked(dst_tok),
            Ghost(ev),
            ticket,
            Ghost(before_core),
            Ghost(after_hop_core),
            Ghost(old(self).timeline.stream.tail()),
        )
    }
}

} // verus!
