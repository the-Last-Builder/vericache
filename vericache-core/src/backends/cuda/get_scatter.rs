#[cfg(verus_only)]
use super::copy::caller_stream_valid;
#[cfg(verus_only)]
use super::put::scatter_offsets_valid;
use super::*;
use transfer::cuda::{CudaSlice, CudaStream};

verus! {

proof fn lemma_get_scatter_step_ok<'a>(
    backend: &CudaBackend<'a>,
    before_core: &CacheCore,
    after_acq_core: &CacheCore,
    before_step_core: &CacheCore,
    core: &CacheCore,
    dst_timeline: u8,
    dst_source: EventSource,
    old_readers: ISet<TransferEvent>,
    new_readers: ISet<TransferEvent>,
    last_ticket: Ticket,
    ticket: Ticket,
    ev: TransferEvent,
    dst_loc: SlotLoc,
    is_first: bool,
)
    requires
        before_core.well_formed(),
        before_step_core.well_formed(),
        core.well_formed(),
        backend.well_formed(before_step_core),
        dst_timeline != backend.timeline.timeline_id,
        dst_source != backend.timeline.stream.source().source(),
        core_transition(
            before_core,
            before_step_core,
            Some(dst_timeline),
            Some(dst_source),
            FrameTouchMode::PreserveAllInit,
        ),
        before_step_core.frames == after_acq_core.frames,
        before_step_core.frames.same_directory(&before_core.frames),
        core.node_id == before_step_core.node_id,
        core.instance_id() == before_step_core.instance_id(),
        core.tiers == before_step_core.tiers,
        core.frames == before_step_core.frames,
        core.next_version_seq == before_step_core.next_version_seq,
        core.version_seq_limit == before_step_core.version_seq_limit,
        core.active_leases == before_step_core.active_leases,
        core.timelines.count == before_step_core.timelines.count,
        core.timelines.kinds == before_step_core.timelines.kinds,
        core.timelines.confirmed_seq == before_step_core.timelines.confirmed_seq,
        core.timelines.next_seq@ == before_step_core.timelines.next_seq@.update(
            dst_timeline as int,
            (ticket.sequence() + 1) as u64,
        ),
        core.timelines.events@ == before_step_core.timelines.events@.insert(
            (dst_timeline, ticket.sequence()),
            ev,
        ),
        before_step_core.ledger@.submitted.map().submap_of(core.ledger@.submitted.map()),
        core.ledger@.submitted.map() == before_step_core.ledger@.submitted.map().insert(
            ev,
            core.ledger@.submitted.map()[ev],
        ),
        core.ledger@.knows == before_step_core.ledger@.knows,
        core.ledger@.succeeded == before_step_core.ledger@.succeeded,
        core.ledger@.revoked == before_step_core.ledger@.revoked,
        ev.source == dst_source,
        core.ledger@.submitted.map()[ev].access.writes == ISet::empty().insert(dst_loc),
        ticket != Ticket::NONE,
        ticket.timeline_id() == dst_timeline,
        ticket.sequence() == before_step_core.timelines.next_seq[dst_timeline as int],
        core.timelines.issued(dst_timeline, ticket.sequence()),
        core.timelines.events@[(dst_timeline, ticket.sequence())] == ev,
        new_readers == old_readers.insert(ev),
        is_first ==> old_readers.is_empty(),
        !is_first ==> last_ticket != Ticket::NONE && last_ticket.timeline_id() == dst_timeline
            && before_step_core.timelines.valid_ticket(last_ticket) && cuda_readers_covered(
            old_readers,
            before_step_core,
            dst_timeline,
            last_ticket,
        ),
        forall|f: FrameId|
            f.0 < after_acq_core.frames.metas.len() ==> dst_loc != (
            #[trigger] after_acq_core.frames.frame(f)).slot.loc,
    ensures
        backend.well_formed(core),
        cuda_readers_covered(new_readers, core, dst_timeline, ticket),
        core_transition(
            before_core,
            core,
            Some(dst_timeline),
            Some(dst_source),
            FrameTouchMode::PreserveAllInit,
        ),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(FrameTable::same_directory);
    hide(FrameTable::preserves_initializing);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    lemma_slab_bound_same_frames(
        before_step_core,
        core,
        backend.gpu_tier,
        backend.gpu_first_frame,
        backend.gpu_num_frames,
        backend.gpu_buffer.space(),
    );
    if let Some(ref h) = backend.pinned_host {
        lemma_slab_bound_same_frames(
            before_step_core,
            core,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
    lemma_cuda_timeline_preserved(&backend.timeline, before_step_core, core);
    lemma_cuda_events_preserved_insert(
        before_step_core.timelines.events@,
        core.timelines.events@,
        backend.timeline.timeline_id,
        before_step_core.timelines.next_seq[backend.timeline.timeline_id as int],
        dst_timeline,
        ticket.sequence(),
        ev,
    );
    lemma_cuda_ledger_writes_only_insert(
        before_step_core.ledger@,
        core.ledger@,
        ev,
        core.ledger@.submitted.map()[ev],
        dst_loc,
    );
    assert(core.frames.same_directory(&before_step_core.frames)) by {
        reveal(FrameTable::same_directory);
    }
    assert(core.frames.preserves_initializing(&before_step_core.frames, None)) by {
        reveal(FrameTable::preserves_initializing);
    }
    assert forall|f: FrameId|
        f.0 < before_step_core.frames.metas.len() && (#[trigger] before_step_core.frames.meta(
            f,
        )).state == STATE_INITIALIZING implies dst_loc != before_step_core.frames.frame(
        f,
    ).slot.loc by {
        assert(before_step_core.frames.frame(f) == after_acq_core.frames.frame(f));
    }
    lemma_cuda_puts_after_disjoint_step(
        backend.active_gpu_puts@,
        before_step_core,
        core,
        backend.timeline.timeline_id,
        None,
        Some(dst_loc),
    );
    lemma_cuda_readers_covered_step(
        old_readers,
        new_readers,
        before_step_core,
        core,
        dst_timeline,
        last_ticket,
        ticket,
        ev,
        is_first,
    );
    assert(core.ledger@.submitted.dom() =~= before_step_core.ledger@.submitted.dom().insert(ev));
    assert forall|f: FrameId|
        f.0 < before_core.frames.metas.len() && (#[trigger] before_core.frames.meta(f)).state
            == STATE_INITIALIZING implies dst_loc != before_core.frames.frame(f).slot.loc by {
        reveal(FrameTable::same_directory);
        assert(after_acq_core.frames.frame(f).slot == before_core.frames.frame(f).slot);
    }
    lemma_core_transition_step_single_event(
        before_core,
        before_step_core,
        core,
        dst_timeline,
        dst_source,
        ev,
        dst_loc,
    );
}

proof fn lemma_get_scatter_dsts_preserved(
    before_core: &CacheCore,
    core: &CacheCore,
    dsts: Seq<CudaSlice<'_>>,
    dst_toks: Map<int, SlotToken>,
    device: DeviceId,
)
    requires
        core.node_id == before_core.node_id,
        core.instance_id() == before_core.instance_id(),
        core.frames.same_directory(&before_core.frames),
        before_core.citations().subset_of(core.citations()),
        forall|i: int| 0 <= i < dsts.len() ==> (#[trigger] dsts[i]).well_formed(),
        forall|i: int|
            0 <= i < dsts.len() ==> {
                &&& (#[trigger] dsts[i]).instance_id() == before_core.instance_id()
                &&& dsts[i].slot_ref().loc.node == before_core.node()
                &&& (dsts[i].slot_ref().loc.space is GpuAlloc
                    ==> dsts[i].slot_ref().loc.space->GpuAlloc_device == device)
                &&& dst_toks.dom().contains(i)
                &&& slot_is_owned(dst_toks[i], before_core.instance_id(), dsts[i].slot_ref().loc)
                &&& write_admitted(slot_frontier(dst_toks[i]), before_core.citations())
                &&& before_core.disjoint_from_published(
                    ISet::empty().insert(dsts[i].slot_ref().loc),
                )
                &&& forall|f: FrameId|
                    f.0 < before_core.frames.metas.len() ==> dsts[i].slot_ref().loc != (
                    #[trigger] before_core.frames.frame(f)).slot.loc
            },
    ensures
        forall|k: int|
            0 <= k < dsts.len() ==> {
                &&& (#[trigger] dsts[k]).instance_id() == core.instance_id()
                &&& dsts[k].slot_ref().loc.node == core.node()
                &&& (dsts[k].slot_ref().loc.space is GpuAlloc
                    ==> dsts[k].slot_ref().loc.space->GpuAlloc_device == device)
                &&& dst_toks.dom().contains(k)
                &&& slot_is_owned(dst_toks[k], core.instance_id(), dsts[k].slot_ref().loc)
                &&& write_admitted(slot_frontier(dst_toks[k]), core.citations())
                &&& core.disjoint_from_published(ISet::empty().insert(dsts[k].slot_ref().loc))
                &&& forall|f: FrameId|
                    f.0 < core.frames.metas.len() ==> dsts[k].slot_ref().loc != (
                    #[trigger] core.frames.frame(f)).slot.loc
            },
{
    reveal(CacheCore::disjoint_from_published);
    reveal(FrameTable::same_directory);
    assert forall|k: int| 0 <= k < dsts.len() implies {
        &&& (#[trigger] dsts[k]).instance_id() == core.instance_id()
        &&& dsts[k].slot_ref().loc.node == core.node()
        &&& (dsts[k].slot_ref().loc.space is GpuAlloc
            ==> dsts[k].slot_ref().loc.space->GpuAlloc_device == device)
        &&& dst_toks.dom().contains(k)
        &&& slot_is_owned(dst_toks[k], core.instance_id(), dsts[k].slot_ref().loc)
        &&& write_admitted(slot_frontier(dst_toks[k]), core.citations())
        &&& core.disjoint_from_published(ISet::empty().insert(dsts[k].slot_ref().loc))
        &&& forall|f: FrameId|
            f.0 < core.frames.metas.len() ==> dsts[k].slot_ref().loc != (
            #[trigger] core.frames.frame(f)).slot.loc
    } by {
        assert(dsts[k].well_formed());
        assert forall|f: FrameId| f.0 < core.frames.metas.len() implies dsts[k].slot_ref().loc != (
        #[trigger] core.frames.frame(f)).slot.loc by {
            assert(core.frames.frame(f).slot == before_core.frames.frame(f).slot);
        }
    }
}

impl<'a> CudaBackend<'a> {
    /// Scatter-copy a published GPU or pinned-DRAM frame into multiple caller
    /// destination `CudaSlice`s on caller's `dst_stream` with Ordered Handoff.
    pub fn get_into_gpu_scatter(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dsts: &[CudaSlice<'_>],
        Tracked(dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(offsets): Ghost<Seq<usize>>,
    ) -> (r: Result<(VersionId, Ticket), (CoreError, Option<Ticket>, usize)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            caller_stream_valid(*old(dst_stream), old(core), dst_timeline, self.timeline.stream),
            dst_timeline != self.timeline.timeline_id,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            scatter_offsets_valid(dsts@, offsets, old(core).frames.frame_size),
            forall|i: int|
                0 <= i < dsts@.len() ==> {
                    &&& (#[trigger] dsts@[i]).instance_id() == old(core).instance_id()
                    &&& dsts@[i].slot_ref().loc.node == old(core).node()
                    &&& (dsts@[i].slot_ref().loc.space is GpuAlloc
                        ==> dsts@[i].slot_ref().loc.space->GpuAlloc_device == self.device)
                    &&& old(dst_toks).dom().contains(i)
                    &&& slot_is_owned(
                        old(dst_toks)[i],
                        old(core).instance_id(),
                        dsts@[i].slot_ref().loc,
                    )
                    &&& write_admitted(slot_frontier(old(dst_toks)[i]), old(core).citations())
                    &&& old(core).disjoint_from_published(
                        ISet::empty().insert(dsts@[i].slot_ref().loc),
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(core).frames.metas.len() ==> dsts@[i].slot_ref().loc != (
                        #[trigger] old(core).frames.frame(f)).slot.loc
                },
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
            core_transition(
                old(core),
                final(core),
                Some(dst_timeline),
                Some(old(dst_stream).source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((version, ticket)) ==> ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline && final(core).spec@.versions.contains_key(version),
            r matches Err((_, Some(ticket), completed_slices)) ==> completed_slices > 0
                && completed_slices <= dsts.len() && ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline,
            r matches Err((_, None, completed_slices)) ==> completed_slices == 0,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CacheCore::disjoint_from_published);
        hide(copy_effect);
        hide(effect_implies);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(TimelineTable::ticket_covers);
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
            return Err((CoreError::UnsupportedTier, None, 0));
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
                return Err((e, None, 0));
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
            lemma_get_scatter_dsts_preserved(&before_core, core, dsts@, *dst_toks, self.device);
        }
        let sp = self.frame_sync_point(frame);
        if let Err(e) = self.timeline.wait_sync_point(dst_stream, core, sp) {
            let ghost before_rel = *core;
            core.release_local_lease(
                lease,
                Tracked(src_cap),
                Ticket::NONE,
                Tracked(instance),
            ).unwrap();
            proof {
                lemma_slab_bound_same_directory(
                    &before_rel,
                    core,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                );
                if let Some(ref h) = self.pinned_host {
                    lemma_slab_bound_same_directory(
                        &before_rel,
                        core,
                        h.tier,
                        h.first_frame,
                        h.num_frames,
                        h.reg.space(),
                    );
                }
                lemma_cuda_timeline_preserved(&self.timeline, &before_rel, core);
                lemma_cuda_events_preserved_same(
                    core.timelines.events@,
                    self.timeline.timeline_id,
                    before_rel.timelines.next_seq[self.timeline.timeline_id as int],
                );
                lemma_cuda_ledger_writes_only_same(core.ledger@, None);
                lemma_cuda_puts_after_disjoint_step(
                    self.active_gpu_puts@,
                    &before_rel,
                    core,
                    self.timeline.timeline_id,
                    None,
                    None,
                );
                lemma_core_transition_step_no_event(
                    &before_core,
                    &before_rel,
                    core,
                    Some(dst_timeline),
                    Some(dst_stream.source().source()),
                    None,
                    Some(lease.slot),
                );
            }
            return Err((e, None, 0));
        }
        let src_slice = self.owned_src_slice(frame);
        let tracked mut src_perm = ReadPerm::Shared(src_cap);
        let mut i: usize = 0;
        let mut offset: usize = 0;
        let mut last_ticket: Ticket = Ticket::NONE;
        while i < dsts.len()
            invariant
                i <= dsts.len(),
                offset == offsets[i as int],
                scatter_offsets_valid(dsts@, offsets, core.frames.frame_size),
                src_slice.well_formed(),
                src_slice.instance_id() == core.instance_id(),
                src_slice.slot_ref() == core.frames.frame(frame).slot,
                src_slice.slot_ref().sub_len == core.frames.frame_size,
                src_slice.slot_ref().loc.node == core.node(),
                before_core == *old(core),
                before_core.well_formed(),
                (lease.slot as usize) < before_core.active_leases.len(),
                before_core.active_leases[lease.slot as int].is_none(),
                core.well_formed(),
                self.well_formed(core),
                caller_stream_valid(*dst_stream, core, dst_timeline, self.timeline.stream),
                dst_timeline != self.timeline.timeline_id,
                core_transition(
                    &before_core,
                    core,
                    Some(dst_timeline),
                    Some(old(dst_stream).source().source()),
                    FrameTouchMode::PreserveAllInit,
                ),
                dst_stream.source().source() == old(dst_stream).source().source(),
                instance.id() == core.instance_id(),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.node() == old(shares).node(),
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.spec == old(core).spec,
                before_core.node_id == old(core).node_id,
                before_core.tiers == old(core).tiers,
                before_core.spec == old(core).spec,
                core.frames == after_acq_core.frames,
                core.frames.frame_size == before_core.frames.frame_size,
                core.frames.same_directory(&before_core.frames),
                frame.0 < core.frames.metas.len(),
                lease.frame == frame,
                core.active_lease_matches(lease),
                core.active_leases[lease.slot as int]->Some_0.share@ == src_perm->Shared_0.share(),
                src_perm is Shared,
                src_slice.slot_ref().loc.space is GpuAlloc
                    ==> src_slice.slot_ref().loc.space->GpuAlloc_device == self.device,
                src_perm.valid_local(core.instance_id(), src_slice.slot_ref().loc),
                read_admitted(src_perm.frontier(), core.citations()),
                core.frames.versions@.contains_key(version),
                i == 0 ==> last_ticket == Ticket::NONE && src_perm.frontier().readers.is_empty(),
                i > 0 ==> last_ticket != Ticket::NONE && last_ticket.timeline_id() == dst_timeline
                    && core.timelines.valid_ticket(last_ticket) && cuda_readers_covered(
                    src_perm.frontier().readers,
                    core,
                    dst_timeline,
                    last_ticket,
                ),
                forall|k: int|
                    i <= k < dsts.len() ==> {
                        &&& (#[trigger] dsts@[k]).instance_id() == core.instance_id()
                        &&& dsts@[k].slot_ref().loc.node == core.node()
                        &&& (dsts@[k].slot_ref().loc.space is GpuAlloc
                            ==> dsts@[k].slot_ref().loc.space->GpuAlloc_device == self.device)
                        &&& dst_toks.dom().contains(k)
                        &&& slot_is_owned(dst_toks[k], core.instance_id(), dsts@[k].slot_ref().loc)
                        &&& write_admitted(slot_frontier(dst_toks[k]), core.citations())
                        &&& after_acq_core.disjoint_from_published(
                            ISet::empty().insert(dsts@[k].slot_ref().loc),
                        )
                        &&& forall|f: FrameId|
                            f.0 < after_acq_core.frames.metas.len() ==> dsts@[k].slot_ref().loc != (
                            #[trigger] after_acq_core.frames.frame(f)).slot.loc
                    },
            decreases dsts.len() - i,
        {
            proof {
                lemma_scatter_offsets_bounded(
                    dsts@,
                    offsets,
                    core.frames.frame_size,
                    (i + 1) as int,
                );
                assert(dsts@[i as int].well_formed());
                assert(core.disjoint_from_published(
                    ISet::empty().insert(dsts@[i as int].slot_ref().loc),
                )) by {
                    reveal(CacheCore::disjoint_from_published);
                }
            }
            let slice_len = dsts[i].len();
            let src_sub = src_slice.sub_slice(offset, slice_len);
            let tracked mut dtok = dst_toks.tracked_remove(i as int);
            let ghost before_step_core = *core;
            let ghost old_readers = src_perm.frontier().readers;
            let step_res = Self::enqueue_on_caller_stream(
                dst_stream,
                core,
                Tracked(instance),
                dst_timeline,
                &dsts[i],
                &src_sub,
                Tracked(&mut src_perm),
                Tracked(&mut dtok),
                Ghost(self.timeline.stream),
            );
            proof {
                dst_toks.tracked_insert(i as int, dtok);
            }
            match step_res {
                Err(e) => {
                    let tracked cap = src_perm.into_shared();
                    proof {
                        if i > 0 {
                            lemma_cuda_readers_covered_to_returned(core, frame, cap, last_ticket);
                        }
                    }
                    core.release_local_lease(
                        lease,
                        Tracked(cap),
                        last_ticket,
                        Tracked(instance),
                    ).unwrap();
                    proof {
                        lemma_slab_bound_same_directory(
                            &before_step_core,
                            core,
                            self.gpu_tier,
                            self.gpu_first_frame,
                            self.gpu_num_frames,
                            self.gpu_buffer.space(),
                        );
                        if let Some(ref h) = self.pinned_host {
                            lemma_slab_bound_same_directory(
                                &before_step_core,
                                core,
                                h.tier,
                                h.first_frame,
                                h.num_frames,
                                h.reg.space(),
                            );
                        }
                        lemma_cuda_timeline_preserved(&self.timeline, &before_step_core, core);
                        lemma_cuda_events_preserved_same(
                            core.timelines.events@,
                            self.timeline.timeline_id,
                            before_step_core.timelines.next_seq[self.timeline.timeline_id as int],
                        );
                        lemma_cuda_ledger_writes_only_same(core.ledger@, None);
                        lemma_cuda_puts_after_disjoint_step(
                            self.active_gpu_puts@,
                            &before_step_core,
                            core,
                            self.timeline.timeline_id,
                            None,
                            None,
                        );
                        assert(core_transition(
                            &before_core,
                            core,
                            Some(dst_timeline),
                            Some(dst_stream.source().source()),
                            FrameTouchMode::PreserveAllInit,
                        )) by {
                            lemma_core_transition_step_no_event(
                                &before_core,
                                &before_step_core,
                                core,
                                Some(dst_timeline),
                                Some(dst_stream.source().source()),
                                None,
                                Some(lease.slot),
                            );
                        }
                    }
                    if i == 0 {
                        return Err((e, None, 0));
                    } else {
                        return Err((e, Some(last_ticket), i));
                    }
                },
                Ok((_ev, ticket)) => {
                    proof {
                        lemma_get_scatter_step_ok(
                            self,
                            &before_core,
                            &after_acq_core,
                            &before_step_core,
                            core,
                            dst_timeline,
                            dst_stream.source().source(),
                            old_readers,
                            src_perm.frontier().readers,
                            last_ticket,
                            ticket,
                            _ev@,
                            dsts@[i as int].slot_ref().loc,
                            i == 0,
                        );
                    }
                    last_ticket = ticket;
                    offset += slice_len;
                    i += 1;
                },
            }
        }
        let tracked cap = src_perm.into_shared();
        let ghost before_rel_core = *core;
        proof {
            lemma_cuda_readers_covered_to_returned(core, frame, cap, last_ticket);
        }
        core.release_local_lease(lease, Tracked(cap), last_ticket, Tracked(instance)).unwrap();
        proof {
            lemma_core_wf_facts(core);
            lemma_slab_bound_same_directory(
                &before_rel_core,
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
            );
            if let Some(ref h) = self.pinned_host {
                lemma_slab_bound_same_directory(
                    &before_rel_core,
                    core,
                    h.tier,
                    h.first_frame,
                    h.num_frames,
                    h.reg.space(),
                );
            }
            lemma_cuda_timeline_preserved(&self.timeline, &before_rel_core, core);
            lemma_cuda_events_preserved_same(
                core.timelines.events@,
                self.timeline.timeline_id,
                before_rel_core.timelines.next_seq[self.timeline.timeline_id as int],
            );
            lemma_cuda_ledger_writes_only_same(core.ledger@, None);
            lemma_cuda_puts_after_disjoint_step(
                self.active_gpu_puts@,
                &before_rel_core,
                core,
                self.timeline.timeline_id,
                None,
                None,
            );
            lemma_core_transition_step_no_event(
                &before_core,
                &before_rel_core,
                core,
                Some(dst_timeline),
                Some(dst_stream.source().source()),
                None,
                Some(lease.slot),
            );
        }
        Ok((version, last_ticket))
    }
}

} // verus!
