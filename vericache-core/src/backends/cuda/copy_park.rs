use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;

verus! {

impl<'a> CudaBackend<'a> {
    /// Cleanly abort a reserved CUDA copy hop when `enqueue_memcpy` fails.
    pub(super) fn abort_unsubmitted_copy_hop(
        &mut self,
        core: &mut CacheCore,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(before_core): Ghost<CacheCore>,
    )
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            instance.id() == old(core).instance_id(),
            old(core).node_id_spec() == before_core.node_id_spec(),
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers_spec() == before_core.tiers_spec(),
            old(core).spec_spec() == before_core.spec_spec(),
            old(core).next_version_seq_spec() == before_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            old(core).timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
            old(core).timelines_spec().next_seq_spec()
                == before_core.timelines_spec().next_seq_spec(),
            old(core).timelines_spec().events_spec() == before_core.timelines_spec().events_spec(),
            old(core).ledger_spec() == before_core.ledger_spec(),
            old(core).frames_spec().same_directory(&before_core.frames_spec()),
            old(core).frames_spec().preserves_initializing(&before_core.frames_spec(), None),
            hop.src_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < before_core.frames_spec().metas_spec().len(),
            before_core.frames_spec().meta(hop.dst_frame).state != STATE_INITIALIZING,
            hop.src_lease.frame_spec() == hop.src_frame,
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@
                == src_cap.share(),
            old(core).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(hop.dst_frame).queued(),
            old(core).frames_spec().frame(hop.dst_frame).token is None,
            old(core).frames_spec().frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            slot_is_owned(dst_tok, old(core).instance_id(), hop.dst_slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(core).citations()),
            old(core).frames_spec().returned_reads(
                hop.src_frame,
                src_cap,
                Ticket::NONE,
                old(core).kernel_view(),
            ),
            (hop.src_lease.slot_spec() as usize) < before_core.active_leases_spec().len(),
            before_core.active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
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
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
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
        let ghost after_hop_core = *core;
        core.abort_copy_hop(
            hop,
            Tracked(src_cap),
            Ticket::NONE,
            Tracked(dst_tok),
            Tracked(instance),
        );
        proof {
            lemma_cuda_slabs_same_directory(self, &after_hop_core, core);
            lemma_cuda_timeline_preserved(&self.timeline, &after_hop_core, core);
            assert(core.frames_spec().same_directory(&before_core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(hop.dst_frame),
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
                Some(hop.dst_frame),
                None,
            );
            lemma_core_transition_step_no_event(
                &before_core,
                &after_hop_core,
                core,
                Some(self.timeline.timeline_id),
                Some(self.timeline.stream.source().source()),
                Some(hop.dst_frame),
                Some(hop.src_lease.slot_spec()),
            );
        }
    }

    /// Certify the completed copy hop DMA, release the source lease, and park
    /// the destination write for async completion.
    pub(super) fn finish_copy_hop_cuda(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Ghost(ev): Ghost<TransferEvent>,
        ticket: Ticket,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(after_hop_core): Ghost<CacheCore>,
        Ghost(old_tail): Ghost<ISet<TransferEvent>>,
    ) -> (r: BackendResult<(FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            after_hop_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            instance.id() == old(core).instance_id(),
            old(core).node_id_spec() == before_core.node_id_spec() && before_core.node_id_spec()
                == after_hop_core.node_id_spec(),
            old(core).instance_id() == before_core.instance_id() && before_core.instance_id()
                == after_hop_core.instance_id(),
            old(core).tiers_spec() == before_core.tiers_spec() && before_core.tiers_spec()
                == after_hop_core.tiers_spec(),
            old(core).spec_spec() == before_core.spec_spec() && before_core.spec_spec()
                == after_hop_core.spec_spec(),
            old(core).next_version_seq_spec() == before_core.next_version_seq_spec()
                && before_core.next_version_seq_spec() == after_hop_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_core.version_seq_limit_spec()
                && before_core.version_seq_limit_spec() == after_hop_core.version_seq_limit_spec(),
            old(core).frames_spec() == after_hop_core.frames_spec(),
            after_hop_core.frames_spec().same_directory(&before_core.frames_spec()),
            after_hop_core.frames_spec().preserves_initializing(&before_core.frames_spec(), None),
            after_hop_core.timelines_spec().count_spec()
                == before_core.timelines_spec().count_spec(),
            after_hop_core.timelines_spec().next_seq_spec()
                == before_core.timelines_spec().next_seq_spec(),
            after_hop_core.timelines_spec().events_spec()
                == before_core.timelines_spec().events_spec(),
            after_hop_core.ledger_spec() == before_core.ledger_spec(),
            old(core).timelines_spec().count_spec() == after_hop_core.timelines_spec().count_spec(),
            old(core).timelines_spec().next_seq_spec()@
                == after_hop_core.timelines_spec().next_seq_spec()@.update(
                old(self).timeline.timeline_id as int,
                (ticket.sequence() + 1) as u64,
            ),
            old(core).timelines_spec().events_spec()@
                == after_hop_core.timelines_spec().events_spec()@.insert(
                (old(self).timeline.timeline_id, ticket.sequence()),
                ev,
            ),
            old(core).timelines_spec().events_spec()@[(
                old(self).timeline.timeline_id,
                ticket.sequence(),
            )] == ev,
            old(core).timelines_spec().issued(old(self).timeline.timeline_id, ticket.sequence()),
            ticket.sequence() <= MAX_TICKET_SEQ,
            ticket != Ticket::NONE,
            ticket.timeline_id() == old(self).timeline.timeline_id,
            ticket.sequence() == after_hop_core.timelines_spec().next_seq_spec()[old(
                self,
            ).timeline.timeline_id as int],
            !after_hop_core.ledger_spec()@.submitted.dom().contains(ev),
            ev.source.succeeds_on_observe(),
            old(core).ledger_spec()@.knows == after_hop_core.ledger_spec()@.knows,
            old(core).ledger_spec()@.succeeded == after_hop_core.ledger_spec()@.succeeded,
            old(core).ledger_spec()@.revoked == after_hop_core.ledger_spec()@.revoked,
            old(core).ledger_spec()@.submitted.instance_id()
                == after_hop_core.ledger_spec()@.submitted.instance_id(),
            old(core).ledger_spec()@.submitted.dom().contains(ev),
            old(core).ledger_spec()@.submitted.map()
                == after_hop_core.ledger_spec()@.submitted.map().insert(
                ev,
                foundation::tracked::EventInfo {
                    access: copy_access(hop.src_slot@, hop.dst_slot@),
                    read_guards: old(core).ledger_spec()@.submitted.map()[ev].read_guards,
                    effect: copy_effect(hop.src_slot@, hop.dst_slot@),
                    preds: closure_from(
                        after_hop_core.ledger_spec()@.submitted.map(),
                        old_tail.union(after_hop_core.citations()),
                    ),
                },
            ),
            old(core).ledger_spec()@.submitted.map()[ev].preds.subset_of(
                after_hop_core.ledger_spec()@.submitted.dom(),
            ),
            after_hop_core.ledger_spec()@.submitted.map().submap_of(
                old(core).ledger_spec()@.submitted.map(),
            ),
            hop.src_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < before_core.frames_spec().metas_spec().len(),
            before_core.frames_spec().meta(hop.dst_frame).state != STATE_INITIALIZING,
            hop.src_slot@ == old(core).frames_spec().frame(hop.src_frame).slot,
            hop.dst_slot@ == old(core).frames_spec().frame(hop.dst_frame).slot,
            hop.src_slot@.loc != hop.dst_slot@.loc,
            hop.src_slot@.is_full(),
            hop.src_slot@.slot_len == old(core).frames_spec().tier_frame_size(old(self).gpu_tier),
            hop.dst_slot@.is_full(),
            hop.dst_slot@.slot_len == old(core).frames_spec().tier_frame_size(old(self).gpu_tier),
            old(core).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(hop.dst_frame).queued(),
            old(core).frames_spec().frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames_spec().frame(hop.dst_frame).bound_key == Some(hop.key),
            old(core).frames_spec().frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames_spec().frame(hop.dst_frame).token is None,
            hop.version.well_formed() && hop.version != VersionId::NONE,
            hop.version.origin_node != old(core).node_id_spec() || hop.version.seq < old(
                core,
            ).next_version_seq_spec() || hop.version.seq >= old(core).version_seq_limit_spec(),
            old(core).frames_spec().versions_spec()@.contains_key(hop.version),
            old(core).frames_spec().versions_spec()@[hop.version] == hop.src_cert@.bytes,
            hop.src_cert@.slot == hop.src_slot@,
            hop.src_cert@.bytes.len() == hop.src_slot@.sub_len,
            replica_valid(after_hop_core.ledger_spec()@, old(core).node(), hop.src_cert@),
            hop.src_lease.frame_spec() == hop.src_frame,
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@
                == src_cap.share(),
            src_cap.well_formed(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == hop.src_slot@.loc,
            src_cap.frontier().readers == ISet::empty().insert(ev),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                hop.dst_slot@.loc,
                Frontier { last_writer: Some(ev), readers: ISet::empty() },
            ),
            (hop.dst_tier == old(self).gpu_tier && old(self).owns_gpu(hop.dst_frame)) || old(
                self,
            ).owns_pinned_host(hop.dst_frame),
            (hop.src_lease.slot_spec() as usize) < before_core.active_leases_spec().len(),
            before_core.active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
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
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            core_transition(
                &before_core,
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
            r matches Ok((dst_frame, t, _)) ==> {
                &&& t != Ticket::NONE
                &&& t.timeline_id() == old(self).timeline.timeline_id
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
        let ghost after_copy_core = *core;
        let ghost put_proof;
        proof {
            lemma_core_wf_facts(&after_hop_core);
            lemma_core_wf_facts(core);
            let val = core.ledger_spec()@.submitted.map()[ev];
            lemma_iset_subset_closure_union(
                after_hop_core.ledger_spec()@.submitted.map(),
                old_tail,
                after_hop_core.citations(),
                after_hop_core.citations(),
                ev,
            );
            put_proof =
            lemma_cuda_single_put_proof(
                after_hop_core.ledger_spec()@,
                core.ledger_spec()@,
                core.node(),
                hop.src_cert@,
                hop.src_slot@,
                hop.dst_slot@,
                ev,
                val,
            );
            assert(core.timelines_spec().events_spec()@[(ticket.timeline_id(), ticket.sequence())]
                == ev);
            assert forall|e: TransferEvent| #[trigger]
                src_cap.frontier().readers.contains(e) implies core.citations().contains(e)
                || exists|s: u64|
                1 <= s <= ticket.sequence() && #[trigger] core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == e by {
                assert(e == ev);
                let s = ticket.sequence();
                assert(1 <= s <= ticket.sequence() && core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == e);
            }
            lemma_returned_reads_by_ticket(core, hop.src_frame, src_cap, ticket);
        }
        core.release_local_lease(
            hop.src_lease,
            Tracked(src_cap),
            ticket,
            Tracked(instance),
        ).unwrap();
        proof {
            lemma_cuda_slabs_same_directory(self, &after_copy_core, core);
            lemma_cuda_timeline_preserved(&self.timeline, &after_copy_core, core);
            assert(core.frames_spec().same_directory(&before_core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().preserves_initializing(
                &before_core.frames_spec(),
                Some(hop.dst_frame),
            ) && core.frames_spec().meta(hop.dst_frame) == after_copy_core.frames_spec().meta(
                hop.dst_frame,
            ) && core.frames_spec().frame(hop.dst_frame) == after_copy_core.frames_spec().frame(
                hop.dst_frame,
            )) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            assert(ev.source.succeeds_on_observe());
            lemma_cuda_single_event_covered(
                put_proof.events,
                core.timelines_spec().events_spec()@,
                self.timeline.timeline_id,
                ticket.sequence(),
                ev,
            );
            assert(core.frames_spec().versions_spec()@
                == after_hop_core.frames_spec().versions_spec()@) by {
                reveal(FrameTable::same_directory);
            }
            lemma_cuda_ticket_covers_exact(&core.timelines_spec(), ticket, ev);
            lemma_cuda_events_preserved_insert(
                before_core.timelines_spec().events_spec()@,
                core.timelines_spec().events_spec()@,
                self.timeline.timeline_id,
                before_core.timelines_spec().next_seq_spec()[self.timeline.timeline_id as int],
                self.timeline.timeline_id,
                ticket.sequence(),
                ev,
            );
            lemma_cuda_ledger_writes_only_insert(
                before_core.ledger_spec()@,
                core.ledger_spec()@,
                ev,
                core.ledger_spec()@.submitted.map()[ev],
                hop.dst_slot@.loc,
            );
            lemma_core_transition_step_no_event(
                &before_core,
                &after_copy_core,
                core,
                Some(self.timeline.timeline_id),
                Some(self.timeline.stream.source().source()),
                None,
                Some(hop.src_lease.slot_spec()),
            );
        }
        let dst_frame = hop.dst_frame;
        let dst_handle = WriteHandle {
            frame: hop.dst_frame,
            tier: hop.dst_tier,
            key: hop.key,
            version: hop.version,
            slot: hop.dst_slot,
        };
        let sync_point = self.commit_pending_gpu_put(
            core,
            dst_handle,
            Tracked(dst_tok),
            ticket,
            Ghost(put_proof),
            true,
            Ghost(before_core),
        )?;
        Ok((dst_frame, ticket, sync_point))
    }

    /// Synchronize the backend CUDA stream and publish/retire all completed
    /// events and parked GPU writes in one deterministic call.
    pub fn synchronize(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<CudaPollReport>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device_spec() == old(self).device_spec(),
            final(self).gpu_tier_spec() == old(self).gpu_tier_spec(),
            final(self).gpu_first_frame_spec() == old(self).gpu_first_frame_spec(),
            final(self).gpu_num_frames_spec() == old(self).gpu_num_frames_spec(),
            final(self).same_pinned_layout(old(self)),
            final(self).timeline_spec().timeline_id_spec() == old(
                self,
            ).timeline_spec().timeline_id_spec(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is Some,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_sync_core = *core;
        let ghost fp = self.poll_footprint();
        proof {
            lemma_cuda_poll_footprint_frames(self, core);
        }
        let sync_res = self.timeline.synchronize(core, Tracked(instance));
        proof {
            lemma_cuda_backend_after_poll_events(
                self.active_gpu_puts@,
                &before_sync_core,
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                &self.pinned_host,
                self.timeline.timeline_id,
            );
            assert(core.frames_spec().same_layout(&before_sync_core.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            assert(core.frames_spec().preserves_initializing(&before_sync_core.frames_spec(), None))
                by {
                reveal(FrameTable::preserves_initializing);
            }
            assert forall|u: TimelineId|
                u < before_sync_core.timelines_spec().count_spec() && u
                    != self.timeline.timeline_id implies (
            #[trigger] core.timelines_spec().confirmed_seq_spec()[u as int])
                == before_sync_core.timelines_spec().confirmed_seq_spec()[u as int] by {
                assert(core.timelines_spec().confirmed_seq_spec()[u as int]
                    == before_sync_core.timelines_spec().confirmed_seq_spec()[u as int]);
            }
            lemma_core_transition_after_retirement(&before_sync_core, core, fp, None, None);
        }
        if let Err(e) = sync_res {
            return Err(e);
        }
        let ghost after_sync_core = *core;
        let res = match self.poll_completions(core, Tracked(instance)) {
            Ok(cr) => Ok(cr),
            Err((e, _)) => Err(e),
        };
        proof {
            lemma_versions_monotonic_trans(
                before_sync_core.spec_spec()@,
                after_sync_core.spec_spec()@,
                core.spec_spec()@,
            );
            lemma_core_transition_trans_same(&before_sync_core, &after_sync_core, core, fp);
        }
        res
    }
}

} // verus!
