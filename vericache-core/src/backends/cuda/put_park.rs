use super::*;
#[cfg(verus_only)]
use foundation::tracked::{closure_from, Frontier};

verus! {

/// Structural invariant of `CudaBackend` after DMA submission and immediately
/// before `record_event` + `park_write` + pushing the new `PendingGpuWrite`.
pub(super) open spec fn cuda_backend_pre_park_wf<'a>(
    b: &CudaBackend<'a>,
    core: &CacheCore,
    before_core: &CacheCore,
) -> bool {
    let tl = b.timeline.timeline_id;
    &&& b.gpu_buffer.well_formed()
    &&& b.gpu_buffer.instance_id() == core.instance_id()
    &&& b.gpu_buffer.node() == core.node()
    &&& b.gpu_buffer.space().is_gpu()
    &&& b.gpu_buffer.space().gpu_device() == b.device
    &&& b.gpu_buffer.num_slots() == b.gpu_num_frames as usize
    &&& b.gpu_buffer.slot_len() == core.frames_spec().tier_frame_size(b.gpu_tier)
    &&& b.gpu_tier.valid(core.tiers_spec().len())
    &&& core.tiers_spec()[b.gpu_tier.index() as int].kind == TierKind::GpuVram
    &&& b.gpu_sync_points.len() == b.gpu_num_frames as usize
    &&& slab_bound_to_core(
        core,
        b.gpu_tier,
        b.gpu_first_frame,
        b.gpu_num_frames,
        b.gpu_buffer.space(),
    )
    &&& match b.pinned_host {
        None => true,
        Some(h) => h.well_formed(core) && h.tier != b.gpu_tier
            && core.frames_spec().tier_frame_size(h.tier) == core.frames_spec().tier_frame_size(
            b.gpu_tier,
        ),
    }
    &&& b.timeline.device == b.device
    &&& b.timeline.well_formed(&core.timelines_spec(), core.node(), &core.ledger_spec()@)
    &&& cuda_puts_valid(b.active_gpu_puts@, before_core, tl)
}

#[verifier::opaque]
pub(super) open spec fn put_gpu_committed_ready<'a>(
    b: &CudaBackend<'a>,
    after_begin_self: &CudaBackend<'a>,
    core: &CacheCore,
    after_begin_core: &CacheCore,
    before_core: &CacheCore,
    handle: WriteHandle,
    dst_tok: SlotToken,
    ticket: Ticket,
    put_proof: PendingPutProof,
) -> bool {
    &&& b.device == after_begin_self.device
    &&& b.gpu_buffer == after_begin_self.gpu_buffer
    &&& b.gpu_tier == after_begin_self.gpu_tier
    &&& b.gpu_first_frame == after_begin_self.gpu_first_frame
    &&& b.gpu_num_frames == after_begin_self.gpu_num_frames
    &&& b.gpu_sync_points == after_begin_self.gpu_sync_points
    &&& b.pinned_host == after_begin_self.pinned_host
    &&& b.active_gpu_puts == after_begin_self.active_gpu_puts
    &&& b.timeline.timeline_id == after_begin_self.timeline.timeline_id
    &&& b.timeline.device == after_begin_self.timeline.device
    &&& b.timeline.stream.source().source() == after_begin_self.timeline.stream.source().source()
    &&& core.node_id_spec() == after_begin_core.node_id_spec() && after_begin_core.node_id_spec()
        == before_core.node_id_spec()
    &&& core.instance_id() == after_begin_core.instance_id() && after_begin_core.instance_id()
        == before_core.instance_id()
    &&& core.tiers_spec() == after_begin_core.tiers_spec() && after_begin_core.tiers_spec()
        == before_core.tiers_spec()
    &&& core.spec_spec() == after_begin_core.spec_spec() && after_begin_core.spec_spec()
        == before_core.spec_spec()
    &&& core.next_version_seq_spec() == after_begin_core.next_version_seq_spec()
        && after_begin_core.next_version_seq_spec() == before_core.next_version_seq_spec() + 1
    &&& core.version_seq_limit_spec() == after_begin_core.version_seq_limit_spec()
        && after_begin_core.version_seq_limit_spec() == before_core.version_seq_limit_spec()
    &&& core.version_registry_spec() == after_begin_core.version_registry_spec()
        && after_begin_core.version_registry_spec() == before_core.version_registry_spec()
    &&& core.frames_spec() == after_begin_core.frames_spec()
    &&& after_begin_core.frames_spec().same_directory(&before_core.frames_spec())
    &&& core.frames_spec().same_directory(&before_core.frames_spec())
    &&& core.frames_spec().preserves_initializing(&before_core.frames_spec(), Some(handle.frame))
    &&& core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec()
    &&& b.timeline.timeline_id < before_core.timelines_spec().count_spec()
    &&& core.timelines_spec().next_seq_spec()[b.timeline.timeline_id as int]
        >= before_core.timelines_spec().next_seq_spec()[b.timeline.timeline_id as int]
    &&& cuda_timeline_events_preserved(
        before_core.timelines_spec().events_spec()@,
        core.timelines_spec().events_spec()@,
        b.timeline.timeline_id,
        before_core.timelines_spec().next_seq_spec()[b.timeline.timeline_id as int],
    )
    &&& cuda_ledger_writes_only(
        before_core.ledger_spec()@,
        core.ledger_spec()@,
        Some(handle.slot@.loc),
    )
    &&& before_core.frames_spec().meta(handle.frame).state != STATE_INITIALIZING
    &&& handle.frame.0 < core.frames_spec().metas_spec().len()
    &&& core.frames_spec().meta(handle.frame).state == STATE_INITIALIZING
    &&& !core.frames_spec().meta(handle.frame).queued()
    &&& core.frames_spec().frame(handle.frame).token is None
    &&& core.frames_spec().frame(handle.frame).tier == handle.tier
    &&& core.frames_spec().frame(handle.frame).slot == handle.slot@
    &&& core.frames_spec().frame(handle.frame).bound_key == Some(handle.key)
    &&& core.frames_spec().frame(handle.frame).bound_version == Some(handle.version)
    &&& handle.version.well_formed() && handle.version != VersionId::NONE
    &&& handle.version.origin_node == before_core.node_id_spec()
    &&& handle.version.seq == before_core.next_version_seq_spec()
    &&& !core.frames_spec().versions_spec()@.contains_key(handle.version)
    &&& handle.slot@.is_full()
    &&& handle.slot@.slot_len == core.frames_spec().tier_frame_size(b.gpu_tier)
    &&& slot_is_owned(dst_tok, core.instance_id(), handle.slot@.loc)
    &&& slot_frontier(dst_tok).readers.is_empty()
    &&& slot_frontier(dst_tok).readers.subset_of(core.citations())
    &&& (slot_frontier(dst_tok).last_writer matches Some(w) ==> (core.citations().contains(w)
        || core.timelines_spec().ticket_covers(ticket, w)))
    &&& ticket != Ticket::NONE
    &&& ticket.timeline_id() == b.timeline.timeline_id
    &&& core.timelines_spec().issued(b.timeline.timeline_id, ticket.sequence())
    &&& core.timelines_spec().events_spec()@.contains_key(
        (b.timeline.timeline_id, ticket.sequence()),
    )
    &&& core.timelines_spec().valid_ticket(ticket)
    &&& put_proof.certificate.slot == handle.slot@
    &&& put_proof.certificate.bytes.len() == core.frames_spec().tier_frame_size(b.gpu_tier)
    &&& put_proof.certificate.writer == core.timelines_spec().events_spec()@[(
        b.timeline.timeline_id,
        ticket.sequence(),
    )]
    &&& slot_frontier(dst_tok).last_writer == Some(put_proof.certificate.writer)
    &&& put_proof.pending(core.ledger_spec()@, core.node())
    &&& cuda_put_events_covered(
        put_proof.events,
        core.timelines_spec().events_spec()@,
        b.timeline.timeline_id,
        ticket.sequence(),
    )
    &&& ((handle.tier == b.gpu_tier && b.owns_gpu(handle.frame)) || b.owns_pinned_host(
        handle.frame,
    ))
}

impl<'a> CudaBackend<'a> {
    /// Record an event, park `handle`, and push the resulting `PendingGpuWrite`
    /// onto `self.active_gpu_puts` (falling back to `PendingGpuAction::Abort`
    /// if `record_event` fails).
    pub(super) fn commit_pending_gpu_put(
        &mut self,
        core: &mut CacheCore,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        ticket: Ticket,
        Ghost(put_proof): Ghost<PendingPutProof>,
        only_if_current: bool,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<SyncPoint>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            old(core).node_id_spec() == before_core.node_id_spec(),
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers_spec() == before_core.tiers_spec(),
            old(core).spec_spec() == before_core.spec_spec(),
            old(core).next_version_seq_spec() >= before_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            old(core).timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
            old(core).timelines_spec().next_seq_spec()[old(self).timeline.timeline_id as int]
                >= before_core.timelines_spec().next_seq_spec()[old(
                self,
            ).timeline.timeline_id as int],
            cuda_timeline_events_preserved(
                before_core.timelines_spec().events_spec()@,
                old(core).timelines_spec().events_spec()@,
                old(self).timeline.timeline_id,
                before_core.timelines_spec().next_seq_spec()[old(self).timeline.timeline_id as int],
            ),
            cuda_ledger_writes_only(
                before_core.ledger_spec()@,
                old(core).ledger_spec()@,
                Some(handle.slot@.loc),
            ),
            old(core).frames_spec().same_directory(&before_core.frames_spec()),
            old(core).frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(handle.frame),
            ),
            before_core.frames_spec().meta(handle.frame).state != STATE_INITIALIZING,
            handle.frame.0 < old(core).frames_spec().metas_spec().len(),
            old(core).frames_spec().meta(handle.frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(handle.frame).queued(),
            old(core).frames_spec().frame(handle.frame).token is None,
            old(core).frames_spec().frame(handle.frame).tier == handle.tier,
            old(core).frames_spec().frame(handle.frame).slot == handle.slot@,
            old(core).frames_spec().frame(handle.frame).bound_key == Some(handle.key),
            old(core).frames_spec().frame(handle.frame).bound_version == Some(handle.version),
            handle.version.well_formed() && handle.version != VersionId::NONE,
            handle.version.origin_node != old(core).node_id_spec() || handle.version.seq < old(
                core,
            ).next_version_seq_spec() || handle.version.seq >= old(core).version_seq_limit_spec(),
            handle.slot@.is_full(),
            handle.slot@.slot_len == old(core).frames_spec().tier_frame_size(old(self).gpu_tier),
            slot_is_owned(dst_tok, old(core).instance_id(), handle.slot@.loc),
            slot_frontier(dst_tok).readers.is_empty(),
            slot_frontier(dst_tok).readers.subset_of(old(core).citations()),
            slot_frontier(dst_tok).last_writer matches Some(w) ==> old(core).citations().contains(w)
                || old(core).timelines_spec().ticket_covers(ticket, w),
            ticket != Ticket::NONE,
            ticket.timeline_id() == old(self).timeline.timeline_id,
            old(core).timelines_spec().issued(old(self).timeline.timeline_id, ticket.sequence()),
            old(core).timelines_spec().valid_ticket(ticket),
            !only_if_current ==> !old(core).frames_spec().versions_spec()@.contains_key(
                handle.version,
            ) && handle.version.origin_node == before_core.node_id_spec() && handle.version.seq
                == before_core.next_version_seq_spec() && handle.version.seq
                < before_core.version_seq_limit_spec(),
            only_if_current ==> old(core).frames_spec().versions_spec()@.contains_key(
                handle.version,
            ) && old(core).frames_spec().versions_spec()@[handle.version]
                == put_proof.certificate.bytes,
            put_proof.certificate.slot == handle.slot@,
            put_proof.certificate.bytes.len() == old(core).frames_spec().tier_frame_size(
                old(self).gpu_tier,
            ),
            put_proof.certificate.writer == old(core).timelines_spec().events_spec()@[(
                old(self).timeline.timeline_id,
                ticket.sequence(),
            )],
            slot_frontier(dst_tok).last_writer == Some(put_proof.certificate.writer),
            put_proof.pending(old(core).ledger_spec()@, old(core).node()),
            cuda_put_events_covered(
                put_proof.events,
                old(core).timelines_spec().events_spec()@,
                old(self).timeline.timeline_id,
                ticket.sequence(),
            ),
            core_transition(
                &before_core,
                old(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).same_pinned_layout(old(self)),
            final(self).owns_pinned_host(handle.frame) == old(self).owns_pinned_host(handle.frame),
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            core_transition(
                &before_core,
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
            match r {
                Ok(sync_point) => final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                    final(self).active_gpu_puts@.last(),
                ) && final(self).active_gpu_puts@.last().action == (PendingGpuAction::Publish {
                    key: handle.key,
                    version: handle.version,
                    proof: Ghost(put_proof),
                    only_if_current,
                }),
                Err(_) => final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                    final(self).active_gpu_puts@.last(),
                ) && (final(self).active_gpu_puts@.last().action is Abort),
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(TimelineTable::ticket_covers);
        let ghost after_copy_core = *core;
        let h_frame = handle.frame;
        let h_tier = handle.tier;
        let h_key = handle.key;
        let h_version = handle.version;
        let ghost h_slot = handle.slot@;
        let event_res = self.timeline.record_event(core);
        core.park_write(handle, Tracked(dst_tok), ticket);
        let ghost before_puts = self.active_gpu_puts@;
        let (sync_out, frame_point, entry) = match event_res {
            Ok(sync_point) => (
                Ok(sync_point),
                sync_point,
                PendingGpuWrite {
                    frame: h_frame,
                    tier: h_tier,
                    slot: Ghost(h_slot),
                    ticket,
                    sync_point,
                    action: PendingGpuAction::Publish {
                        key: h_key,
                        version: h_version,
                        proof: Ghost(put_proof),
                        only_if_current,
                    },
                },
            ),
            Err(e) => (
                Err(e),
                SyncPoint::HostComplete,
                PendingGpuWrite {
                    frame: h_frame,
                    tier: h_tier,
                    slot: Ghost(h_slot),
                    ticket,
                    sync_point: SyncPoint::HostComplete,
                    action: PendingGpuAction::Abort,
                },
            ),
        };
        self.active_gpu_puts.push(entry);
        self.set_frame_sync_point(h_frame, frame_point);
        proof {
            lemma_core_wf_facts(&before_core);
            lemma_core_wf_facts(core);
            lemma_cuda_slabs_same_directory(self, &after_copy_core, core);
            assert(core.frames_spec().same_directory(&before_core.frames_spec())
                && core.frames_spec().versions_spec()@
                == after_copy_core.frames_spec().versions_spec()@
                && before_core.frames_spec().metas_spec().len()
                == core.frames_spec().metas_spec().len() && before_core.frames_spec().frame(
                h_frame,
            ).slot == h_slot) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(h_frame),
            )) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            lemma_cuda_puts_push(
                before_puts,
                self.active_gpu_puts@,
                self.active_gpu_puts@.last(),
                &before_core,
                core,
                self.timeline.timeline_id,
            );
            assert(core_transition(
                &before_core,
                core,
                CoreFootprint::preserve_all(
                    Some(self.timeline.timeline_id),
                    Some(self.timeline.stream.source().source()),
                ),
            )) by {
                lemma_core_transition_step_no_event(
                    &before_core,
                    &after_copy_core,
                    core,
                    Some(self.timeline.timeline_id),
                    Some(self.timeline.stream.source().source()),
                    Some(h_frame),
                    None,
                );
            }
        }
        sync_out
    }

    /// Park a partially-submitted write for abort once `last_ticket` completes.
    pub(super) fn park_aborted_gpu_put(
        &mut self,
        core: &mut CacheCore,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        last_ticket: Ticket,
        Ghost(before_core): Ghost<CacheCore>,
    )
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            old(core).node_id_spec() == before_core.node_id_spec(),
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers_spec() == before_core.tiers_spec(),
            old(core).spec_spec() == before_core.spec_spec(),
            old(core).next_version_seq_spec() >= before_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            old(core).timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
            old(core).timelines_spec().next_seq_spec()[old(self).timeline.timeline_id as int]
                >= before_core.timelines_spec().next_seq_spec()[old(
                self,
            ).timeline.timeline_id as int],
            cuda_timeline_events_preserved(
                before_core.timelines_spec().events_spec()@,
                old(core).timelines_spec().events_spec()@,
                old(self).timeline.timeline_id,
                before_core.timelines_spec().next_seq_spec()[old(self).timeline.timeline_id as int],
            ),
            cuda_ledger_writes_only(
                before_core.ledger_spec()@,
                old(core).ledger_spec()@,
                Some(handle.slot@.loc),
            ),
            old(core).frames_spec().same_directory(&before_core.frames_spec()),
            old(core).frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(handle.frame),
            ),
            before_core.frames_spec().meta(handle.frame).state != STATE_INITIALIZING,
            handle.frame.0 < old(core).frames_spec().metas_spec().len(),
            old(core).frames_spec().meta(handle.frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(handle.frame).queued(),
            old(core).frames_spec().frame(handle.frame).token is None,
            old(core).frames_spec().frame(handle.frame).tier == handle.tier,
            old(core).frames_spec().frame(handle.frame).slot == handle.slot@,
            old(core).frames_spec().frame(handle.frame).bound_key == Some(handle.key),
            old(core).frames_spec().frame(handle.frame).bound_version == Some(handle.version),
            slot_is_owned(dst_tok, old(core).instance_id(), handle.slot@.loc),
            slot_frontier(dst_tok).readers.subset_of(old(core).citations()),
            slot_frontier(dst_tok).last_writer matches Some(w) ==> old(core).citations().contains(w)
                || old(core).timelines_spec().ticket_covers(last_ticket, w),
            last_ticket != Ticket::NONE,
            last_ticket.timeline_id() == old(self).timeline.timeline_id,
            old(core).timelines_spec().issued(
                old(self).timeline.timeline_id,
                last_ticket.sequence(),
            ),
            old(core).timelines_spec().valid_ticket(last_ticket),
            core_transition(
                &before_core,
                old(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).pinned_host == old(self).pinned_host,
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            core_transition(
                &before_core,
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
            final(self).active_gpu_puts@ == old(self).active_gpu_puts@.push(
                final(self).active_gpu_puts@.last(),
            ) && (final(self).active_gpu_puts@.last().action is Abort),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(TimelineTable::ticket_covers);
        let _ = self.timeline.record_event(core);
        let ghost before_park_core = *core;
        let h_frame = handle.frame;
        let h_tier = handle.tier;
        let ghost h_slot = handle.slot@;
        core.park_write(handle, Tracked(dst_tok), last_ticket);
        let ghost before_puts = self.active_gpu_puts@;
        let entry = PendingGpuWrite {
            frame: h_frame,
            tier: h_tier,
            slot: Ghost(h_slot),
            ticket: last_ticket,
            sync_point: SyncPoint::HostComplete,
            action: PendingGpuAction::Abort,
        };
        self.active_gpu_puts.push(entry);
        proof {
            lemma_cuda_slabs_same_directory(self, &before_park_core, core);
            lemma_cuda_timeline_preserved(&self.timeline, &before_park_core, core);
            assert(core.frames_spec().same_directory(&before_core.frames_spec())
                && before_core.frames_spec().metas_spec().len()
                == core.frames_spec().metas_spec().len() && before_core.frames_spec().frame(
                h_frame,
            ).slot == h_slot) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(h_frame),
            )) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            lemma_cuda_puts_push(
                before_puts,
                self.active_gpu_puts@,
                self.active_gpu_puts@.last(),
                &before_core,
                core,
                self.timeline.timeline_id,
            );
            lemma_core_transition_step_no_event(
                &before_core,
                &before_park_core,
                core,
                Some(self.timeline.timeline_id),
                Some(self.timeline.stream.source().source()),
                Some(h_frame),
                None,
            );
        }
    }

    /// Cleanly abort a reserved GPU write before any DMA was submitted.
    pub(super) fn abort_unsubmitted_gpu_put(
        &mut self,
        core: &mut CacheCore,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        Ghost(before_core): Ghost<CacheCore>,
    )
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            old(core).node_id_spec() == before_core.node_id_spec(),
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers_spec() == before_core.tiers_spec(),
            old(core).spec_spec() == before_core.spec_spec(),
            old(core).next_version_seq_spec() >= before_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            old(core).timelines_spec() == before_core.timelines_spec(),
            old(core).ledger_spec() == before_core.ledger_spec(),
            old(core).frames_spec().same_directory(&before_core.frames_spec()),
            old(core).frames_spec().preserves_initializing(&before_core.frames_spec(), None),
            handle.frame.0 < before_core.frames_spec().metas_spec().len(),
            before_core.frames_spec().meta(handle.frame).state != STATE_INITIALIZING,
            handle.frame.0 < old(core).frames_spec().metas_spec().len(),
            old(core).frames_spec().meta(handle.frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(handle.frame).queued(),
            old(core).frames_spec().frame(handle.frame).token is None,
            old(core).frames_spec().frame(handle.frame).tier == handle.tier,
            old(core).frames_spec().frame(handle.frame).slot == handle.slot@,
            slot_is_owned(dst_tok, old(core).instance_id(), handle.slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(core).citations()),
            core_transition(
                &before_core,
                old(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).pinned_host == old(self).pinned_host,
            final(self).active_gpu_puts == old(self).active_gpu_puts,
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            core_transition(
                &before_core,
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(TimelineTable::ticket_covers);
        let ghost after_begin_core = *core;
        core.abort_write(handle, Tracked(dst_tok));
        proof {
            lemma_cuda_slabs_same_directory(self, &after_begin_core, core);
            assert(core.frames_spec().same_directory(&before_core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(handle.frame),
            )) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            lemma_cuda_events_preserved_same(
                core.timelines_spec().events_spec()@,
                self.timeline.timeline_id,
                before_core.timelines_spec().next_seq_spec()[self.timeline.timeline_id as int],
            );
            lemma_cuda_ledger_writes_only_same(core.ledger_spec()@, None);
            lemma_cuda_puts_after_disjoint_step(
                self.active_gpu_puts@,
                &before_core,
                core,
                self.timeline.timeline_id,
                Some(handle.frame),
                None,
            );
            lemma_core_transition_step_no_event(
                &before_core,
                &after_begin_core,
                core,
                Some(self.timeline.timeline_id),
                Some(self.timeline.stream.source().source()),
                Some(handle.frame),
                None,
            );
        }
    }

    /// Pick the index of the lowest-sequence ready parked GPU put (`ticket.seq() <= confirmed`),
    /// preserving FIFO publication order across `active_gpu_puts` (Minor 1).
    pub(super) fn find_min_ready_put(&self, confirmed: u64) -> (r: Option<usize>)
        ensures
            r matches Some(idx) ==> idx < self.active_gpu_puts@.len()
                && self.active_gpu_puts@[idx as int].ticket.sequence() <= confirmed,
    {
        let mut best: Option<usize> = None;
        let mut i: usize = 0;
        while i < self.active_gpu_puts.len()
            invariant
                i <= self.active_gpu_puts@.len(),
                best matches Some(idx) ==> idx < i
                    && self.active_gpu_puts@[idx as int].ticket.sequence() <= confirmed,
            decreases self.active_gpu_puts.len() - i,
        {
            let seq = self.active_gpu_puts[i].ticket.seq();
            if seq <= confirmed {
                match best {
                    None => best = Some(i),
                    Some(b) => {
                        if seq < self.active_gpu_puts[b].ticket.seq() {
                            best = Some(i);
                        }
                    },
                }
            }
            i += 1;
        }
        best
    }

    /// Re-establish `cuda_backend_pre_park_wf` after `enqueue_put_memcpy_proved`
    /// and park `handle` as a pending GPU write.
    pub(super) fn park_committed_put_gpu(
        &mut self,
        core: &mut CacheCore,
        handle: WriteHandle,
        Tracked(dst_tok): Tracked<SlotToken>,
        ticket: Ticket,
        Ghost(put_proof): Ghost<PendingPutProof>,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(after_begin_self): Ghost<CudaBackend<'a>>,
        Ghost(after_begin_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            after_begin_core.well_formed(),
            cuda_backend_pre_park_wf(&after_begin_self, &after_begin_core, &before_core),
            old(self).timeline.well_formed(
                &old(core).timelines_spec(),
                old(core).node(),
                &old(core).ledger_spec()@,
            ),
            core_transition(
                &before_core,
                old(core),
                CoreFootprint::preserve_all(
                    Some(after_begin_self.timeline.timeline_id),
                    Some(after_begin_self.timeline.stream.source().source()),
                ),
            ),
            put_gpu_committed_ready(
                old(self),
                &after_begin_self,
                old(core),
                &after_begin_core,
                &before_core,
                handle,
                dst_tok,
                ticket,
                put_proof,
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device == after_begin_self.device,
            final(self).gpu_tier == after_begin_self.gpu_tier,
            final(self).gpu_first_frame == after_begin_self.gpu_first_frame,
            final(self).gpu_num_frames == after_begin_self.gpu_num_frames,
            final(self).same_pinned_layout(&after_begin_self),
            final(self).timeline.timeline_id == after_begin_self.timeline.timeline_id,
            final(core).node_id_spec() == after_begin_core.node_id_spec(),
            final(core).instance_id() == after_begin_core.instance_id(),
            final(core).tiers_spec() == after_begin_core.tiers_spec(),
            final(core).spec_spec() == after_begin_core.spec_spec(),
            final(core).version_registry_spec() == before_core.version_registry_spec(),
            core_transition(
                &before_core,
                final(core),
                CoreFootprint::preserve_all(
                    Some(after_begin_self.timeline.timeline_id),
                    Some(after_begin_self.timeline.stream.source().source()),
                ),
            ),
            match r {
                Ok((version, frame, t, _)) => {
                    &&& t != Ticket::NONE
                    &&& t.timeline_id() == after_begin_self.timeline.timeline_id
                    &&& version.origin_node == before_core.node_id_spec()
                    &&& version.seq == before_core.next_version_seq_spec()
                    &&& final(core).next_version_seq_spec() == before_core.next_version_seq_spec()
                        + 1
                    &&& ((handle.tier == after_begin_self.gpu_tier && final(self).owns_gpu(frame))
                        || final(self).owns_pinned_host(frame))
                    &&& final(self).active_gpu_puts@ == after_begin_self.active_gpu_puts@.push(
                        final(self).active_gpu_puts@.last(),
                    )
                    &&& final(self).active_gpu_puts@.last().action == (PendingGpuAction::Publish {
                        key: handle.key,
                        version,
                        proof: Ghost(put_proof),
                        only_if_current: false,
                    })
                },
                Err(_) => final(self).active_gpu_puts@ == after_begin_self.active_gpu_puts@.push(
                    final(self).active_gpu_puts@.last(),
                ) && (final(self).active_gpu_puts@.last().action is Abort),
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(TimelineTable::ticket_covers);
        hide(PinnedHostSlab::well_formed);
        hide(cuda_backend_pre_park_wf);
        proof {
            reveal(put_gpu_committed_ready);
            lemma_core_wf_facts(core);
            assert(cuda_backend_pre_park_wf(self, core, &before_core)) by {
                reveal(cuda_backend_pre_park_wf);
                lemma_cuda_slabs_same_frames(self, &after_begin_core, core);
            }
        }
        let h_frame = handle.frame;
        let h_version = handle.version;
        let sync_point = self.commit_pending_gpu_put(
            core,
            handle,
            Tracked(dst_tok),
            ticket,
            Ghost(put_proof),
            false,
            Ghost(before_core),
        )?;
        Ok((h_version, h_frame, ticket, sync_point))
    }
}

} // verus!
