#[cfg(verus_only)]
use super::copy::caller_stream_valid;
use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::cuda::{CudaSlice, CudaStream};

verus! {

impl<'a> CudaBackend<'a> {
    /// Asynchronously copy a published GPU or pinned-DRAM frame into caller's
    /// destination `CudaSlice` on caller's `dst_stream` using Ordered Handoff
    /// (`stream_wait_event` if frame's `SyncPoint` is pending), returning the
    /// lease ticket on `dst_timeline`.
    pub fn get_into_gpu(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dst: &CudaSlice,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            caller_stream_valid(*old(dst_stream), old(core), dst_timeline, self.timeline.stream),
            dst_timeline != self.timeline.timeline_id,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            dst.well_formed(),
            dst.instance_id() == old(core).instance_id(),
            dst.slot_ref().loc.node == old(core).node(),
            dst.slot_ref().sub_len == old(core).frames.tier_frame_size(self.gpu_tier),
            dst.slot_ref().loc.space is GpuAlloc ==> dst.slot_ref().loc.space->GpuAlloc_device
                == self.device,
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(core).citations()),
            old(core).disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(core).frames.metas.len() ==> dst.slot_ref().loc != (#[trigger] old(
                    core,
                ).frames.frame(f)).slot.loc,
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            caller_stream_valid(
                *final(dst_stream),
                final(core),
                dst_timeline,
                self.timeline.stream,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).frames.versions@ == old(core).frames.versions@,
            core_transition(
                old(core),
                final(core),
                Some(dst_timeline),
                Some(old(dst_stream).source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((version, ticket, ev)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == dst_timeline
                &&& final(core).spec@.versions.contains_key(version)
                &&& slot_owned(
                    *final(dst_tok),
                    old(core).instance_id(),
                    dst.slot_ref().loc,
                    Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                )
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let is_gpu = self.owns_gpu_frame(frame);
        let is_pinned = self.owns_pinned_host_frame(frame);
        if !is_gpu && !is_pinned {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(dst_timeline),
                    Some(dst_stream.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        let ghost before_core = *core;
        let (lease, version, Ghost(_cert), Tracked(src_cap)) = match core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(dst_timeline),
                        Some(dst_stream.source().source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        let ghost after_acq_core = *core;
        proof {
            lemma_core_wf_facts(core);
            lemma_core_transition_no_event(
                &before_core,
                core,
                Some(dst_timeline),
                Some(dst_stream.source().source()),
            );
            lemma_slab_bound_same_directory(
                &before_core,
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
            );
            if let Some(ref h) = self.pinned_host {
                lemma_slab_bound_same_directory(
                    &before_core,
                    core,
                    h.tier,
                    h.first_frame,
                    h.num_frames,
                    h.reg.space(),
                );
            }
            if is_gpu {
                lemma_slab_frame_slot(
                    core,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                    frame,
                );
            } else {
                let h = self.pinned_host.unwrap();
                lemma_slab_frame_slot(
                    core,
                    h.tier,
                    h.first_frame,
                    h.num_frames,
                    h.reg.space(),
                    frame,
                );
            }
        }
        let sp = self.frame_sync_point(frame);
        if let Err(e) = self.timeline.wait_sync_point(dst_stream, core, sp) {
            core.release_local_lease(
                lease,
                Tracked(src_cap),
                Ticket::NONE,
                Tracked(instance),
            ).unwrap();
            proof {
                lemma_slab_bound_same_directory(
                    &before_core,
                    core,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                );
                if let Some(ref h) = self.pinned_host {
                    lemma_slab_bound_same_directory(
                        &before_core,
                        core,
                        h.tier,
                        h.first_frame,
                        h.num_frames,
                        h.reg.space(),
                    );
                }
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
                lemma_core_transition_step_no_event(
                    &before_core,
                    &after_acq_core,
                    core,
                    Some(dst_timeline),
                    Some(dst_stream.source().source()),
                    None,
                    Some(lease.slot),
                );
            }
            return Err(e);
        }
        let src_slice = self.owned_src_slice(frame);
        let tracked mut src_perm = ReadPerm::Shared(src_cap);
        let copy_res = Self::enqueue_on_caller_stream(
            dst_stream,
            core,
            Tracked(instance),
            dst_timeline,
            dst,
            &src_slice,
            Tracked(&mut src_perm),
            Tracked(dst_tok),
            Ghost(self.timeline.stream),
        );
        let tracked src_cap = src_perm.into_shared();
        match copy_res {
            Err(e) => {
                core.release_local_lease(
                    lease,
                    Tracked(src_cap),
                    Ticket::NONE,
                    Tracked(instance),
                ).unwrap();
                proof {
                    lemma_slab_bound_same_directory(
                        &before_core,
                        core,
                        self.gpu_tier,
                        self.gpu_first_frame,
                        self.gpu_num_frames,
                        self.gpu_buffer.space(),
                    );
                    if let Some(ref h) = self.pinned_host {
                        lemma_slab_bound_same_directory(
                            &before_core,
                            core,
                            h.tier,
                            h.first_frame,
                            h.num_frames,
                            h.reg.space(),
                        );
                    }
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
                    lemma_core_transition_step_no_event(
                        &before_core,
                        &after_acq_core,
                        core,
                        Some(dst_timeline),
                        Some(dst_stream.source().source()),
                        None,
                        Some(lease.slot),
                    );
                }
                Err(e)
            },
            Ok((ev, ticket)) => {
                let ghost after_copy_core = *core;
                proof {
                    assert(src_cap.frontier().readers =~= ISet::empty().insert(ev@));
                    lemma_returned_reads_by_ticket(core, frame, src_cap, ticket);
                    assert(core.ledger@.submitted.dom()
                        =~= after_acq_core.ledger@.submitted.dom().insert(ev@));
                    lemma_core_transition_step_single_event(
                        &before_core,
                        &after_acq_core,
                        core,
                        dst_timeline,
                        dst_stream.source().source(),
                        ev@,
                        dst.slot_ref().loc,
                    );
                }
                core.release_local_lease(
                    lease,
                    Tracked(src_cap),
                    ticket,
                    Tracked(instance),
                ).unwrap();
                proof {
                    lemma_core_wf_facts(core);
                    lemma_slab_bound_same_directory(
                        &before_core,
                        core,
                        self.gpu_tier,
                        self.gpu_first_frame,
                        self.gpu_num_frames,
                        self.gpu_buffer.space(),
                    );
                    if let Some(ref h) = self.pinned_host {
                        lemma_slab_bound_same_directory(
                            &before_core,
                            core,
                            h.tier,
                            h.first_frame,
                            h.num_frames,
                            h.reg.space(),
                        );
                    }
                    lemma_cuda_timeline_preserved(&self.timeline, &before_core, core);
                    lemma_cuda_events_preserved_insert(
                        before_core.timelines.events@,
                        core.timelines.events@,
                        self.timeline.timeline_id,
                        before_core.timelines.next_seq[self.timeline.timeline_id as int],
                        dst_timeline,
                        ticket.sequence(),
                        ev@,
                    );
                    lemma_cuda_ledger_writes_only_insert(
                        before_core.ledger@,
                        core.ledger@,
                        ev@,
                        core.ledger@.submitted.map()[ev@],
                        dst.slot_ref().loc,
                    );
                    lemma_cuda_puts_after_disjoint_step(
                        self.active_gpu_puts@,
                        &before_core,
                        core,
                        self.timeline.timeline_id,
                        None,
                        Some(dst.slot_ref().loc),
                    );
                    lemma_core_transition_step_no_event(
                        &before_core,
                        &after_copy_core,
                        core,
                        Some(dst_timeline),
                        Some(dst_stream.source().source()),
                        None,
                        Some(lease.slot),
                    );
                }
                Ok((version, ticket, ev))
            },
        }
    }
}

} // verus!
