use super::*;
use transfer::nvme;

verus! {

impl NvmeBackend {
    fn finish_one_completion(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        completed: nvme::Completed,
        Ghost(before_ring): Ghost<nvme::IoUring>,
    ) -> (succeeded: bool)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            old(self).file.well_formed(),
            old(self).file.instance_id() == old(core).instance_id(),
            old(self).file.node() == old(core).node(),
            old(self).file.num_slots() == old(self).num_frames as usize,
            old(self).file.slot_len() == old(core).frames.tier_frame_size(old(self).tier) as u64,
            0 < old(self).file.slot_len() <= MAX_IO_LEN,
            old(self).file.slot_len() % DIRECT_IO_ALIGNMENT == 0,
            before_ring.well_formed(),
            before_ring.instance_id() == old(core).instance_id(),
            before_ring.node() == old(core).node(),
            before_ring.source().source() is IoUring,
            old(self).ring.well_formed(),
            old(self).ring.same_ring(before_ring),
            old(self).ring.source() == before_ring.source(),
            completed.describes(before_ring, old(self).ring),
            old(self).tier.valid(old(core).tiers.len()),
            old(core).tiers[old(self).tier.index() as int].kind == TierKind::LocalNvme,
            old(self).timeline < old(core).timelines.count,
            old(core).timelines.count <= MAX_TIMELINES,
            old(core).timelines.kinds[old(self).timeline as int] == TimelineKind::IoUring,
            slab_bound_to_core(
                old(core),
                old(self).tier,
                old(self).first_frame,
                old(self).num_frames,
                old(self).file.space(),
            ),
            old(self).pending_transfers.len() == BITMAP_CAPACITY,
            nvme_window_wf(
                old(self).bitmap,
                old(self).in_flight_ops,
                before_ring,
                old(core),
                old(self).timeline,
            ),
            nvme_pending_valid(
                old(self).pending_transfers@,
                old(self).in_flight_ops,
                before_ring,
                old(core),
                old(self).timeline,
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline),
                None,
                FrameTouchMode::PreserveParkedInit,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_ops = self.in_flight_ops;
        let ghost before_transfers = self.pending_transfers@;
        let ghost before_core = *core;
        let (seq, ok, Ghost(ev), knows, success) = match completed {
            nvme::Completed::Data { user_data, event, knows, succeeded: _succeeded, .. } => (
                user_data,
                true,
                event,
                knows,
                Tracked(Some(_succeeded.get())),
            ),
            nvme::Completed::DataFailed { user_data, event, knows, .. } => (
                user_data,
                false,
                event,
                knows,
                Tracked(None),
            ),
        };
        let ghost knows_val = knows@;
        let ghost io_preds = before_ring.pending()[seq].preds;
        let ghost node = core.node();
        let idx = (seq % BITMAP_CAPACITY) as usize;
        proof {
            lemma_nvme_poll_unfold(
                self.bitmap,
                self.in_flight_ops,
                self.pending_transfers@,
                before_ring,
                self.ring,
                &before_core,
                self.timeline,
                seq,
                ev,
                knows_val,
            );
        }
        let _pending_op = complete_data(
            &mut self.in_flight_ops,
            &mut self.bitmap,
            &mut core.timelines,
            self.timeline,
            seq,
            Ghost(ring_event_map(before_ring)),
            Ghost(node),
            knows,
            success,
            Tracked(core.ledger.borrow_mut()),
        );
        proof {
            lemma_nvme_preserve_core_after_complete(&before_core, core, self.timeline);
            lemma_slab_bound_same_frames(
                &before_core,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
            );
        }
        let ghost after_complete_core = *core;
        let mut taken: Option<InFlightNvmeOp> = None;
        self.pending_transfers.set_and_swap(idx, &mut taken);
        let mut op = taken.unwrap();
        proof {
            assert(before_transfers[idx as int] == Some(op));
            assert(self.pending_transfers@ =~= before_transfers.update(idx as int, None));
            lemma_nvme_completion_observed(
                &before_core,
                core,
                self.timeline,
                seq,
                ev,
                knows_val,
                io_preds,
                &op,
            );
            lemma_retain_copy_after_knows_growth(before_core.ledger@, core.ledger@, node, op.copy@);
        }
        proof {
            lemma_core_wf_facts(core);
            lemma_core_catalog_indexed(core);
        }
        let current = match core.catalog.lookup(op.hop.key) {
            Some(entry) => entry.version.origin_node == op.hop.version.origin_node
                && entry.version.seq == op.hop.version.seq,
            None => false,
        };
        let ghost removed_op = op;
        if ok && current {
            let ghost dst_cert;
            proof {
                assert(core.ledger@.succeeded.contains(ev));
                assert(event_succeeded(core.ledger@, node, ev));
                lemma_nvme_dst_latest(
                    op.copy@.writer,
                    op.copy@.destination.loc,
                    &core.ledger.borrow().submitted,
                    op.dst_tok.borrow_mut(),
                    instance,
                );
                dst_cert = lemma_nvme_certify_completed_copy(op.copy@, core.ledger@, node);
            }
            core.finish_copy_hop(
                op.hop,
                op.src_cap,
                op.ticket,
                op.dst_tok,
                op.ticket,
                Ghost(dst_cert),
                Tracked(instance),
            );
            proof {
                lemma_core_wf_facts(&after_complete_core);
                lemma_core_wf_facts(core);
                assert(core.frames.versions@ =~= after_complete_core.frames.versions@);
            }
        } else {
            core.abort_copy_hop(op.hop, op.src_cap, op.ticket, op.dst_tok, Tracked(instance));
            proof {
                lemma_core_wf_facts(&after_complete_core);
                lemma_core_wf_facts(core);
            }
        }
        proof {
            lemma_slab_bound_same_layout(
                &after_complete_core,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
            );
            assert(before_core.frames.versions@.submap_of(core.frames.versions@)) by {
                reveal(CacheCore::well_formed);
            }
            lemma_nvme_pending_preserved_after_reap(
                before_transfers,
                self.pending_transfers@,
                before_ops,
                self.in_flight_ops,
                before_ring,
                self.ring,
                &before_core,
                core,
                self.timeline,
                idx as int,
                seq,
                removed_op,
            );
            lemma_nvme_pack_window(self.bitmap, self.in_flight_ops, self.ring, core, self.timeline);
            lemma_core_wf_facts(&before_core);
            lemma_core_wf_facts(&after_complete_core);
            lemma_core_wf_facts(core);
            lemma_core_transition_nvme_poll_step(
                &before_core,
                core,
                self.timeline,
                removed_op.hop.dst_frame,
            );
        }
        ok
    }

    /// Drain up to `budget` (capped at `BITMAP_CAPACITY`) currently ready
    /// `io_uring` completions, reporting non-dropping `NvmeProgress` (M8).
    pub fn poll_completions_up_to(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        budget: usize,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline),
                None,
                FrameTouchMode::PreserveParkedInit,
            ),
            r matches Ok(p) ==> p.completed + p.failed <= budget,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let limit = if budget < BITMAP_CAPACITY as usize {
            budget
        } else {
            BITMAP_CAPACITY as usize
        };
        let mut progress = NvmeProgress::empty();
        let mut steps: usize = 0;
        proof {
            lemma_core_transition_refl(
                core,
                Some(self.timeline),
                None,
                FrameTouchMode::PreserveParkedInit,
            );
        }
        while steps < limit
            invariant
                steps <= limit <= BITMAP_CAPACITY as usize,
                limit <= budget,
                core.well_formed(),
                self.well_formed(core),
                self.tier == old(self).tier,
                self.timeline == old(self).timeline,
                self.first_frame == old(self).first_frame,
                self.num_frames == old(self).num_frames,
                core.node_id == old(core).node_id,
                core.tiers == old(core).tiers,
                core.timelines.count == old(core).timelines.count,
                core.timelines.next_seq == old(core).timelines.next_seq,
                core.version_registry == old(core).version_registry,
                instance.id() == core.instance_id(),
                old(core).spec@.versions_monotonic(core.spec@),
                core_transition(
                    old(core),
                    core,
                    Some(self.timeline),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                ),
                progress.completed + progress.failed == steps,
            decreases limit - steps,
        {
            let ghost before_ring = self.ring;
            let Some(completed) = self.ring.poll_one(Tracked(instance)) else {
                break;
            };
            let ghost before_step_core = *core;
            let ok = self.finish_one_completion(
                core,
                Tracked(instance),
                completed,
                Ghost(before_ring),
            );
            proof {
                lemma_versions_monotonic_trans(old(core).spec@, before_step_core.spec@, core.spec@);
                lemma_core_transition_trans(
                    old(core),
                    &before_step_core,
                    core,
                    Some(self.timeline),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                );
            }
            if ok {
                progress.completed += 1;
            } else {
                progress.failed += 1;
            }
            steps += 1;
        }
        Ok(progress)
    }

    /// Drain all currently ready `io_uring` completions, reporting non-dropping
    /// `NvmeProgress { completed, failed }` (M8).
    pub fn poll_completions(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline),
                None,
                FrameTouchMode::PreserveParkedInit,
            ),
    {
        self.poll_completions_up_to(core, Tracked(instance), BITMAP_CAPACITY as usize)
    }

    /// Block until one in-flight `io_uring` operation completes, returning
    /// non-dropping `Err((err, prior))` if waiting fails (M8).
    pub fn wait_one_completion(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        mut progress: NvmeProgress,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            progress.completed < usize::MAX,
            progress.failed < usize::MAX,
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline),
                None,
                FrameTouchMode::PreserveParkedInit,
            ),
            match r {
                Err((_, p)) => p == progress && *final(core) == *old(core) && *final(self) == *old(
                    self,
                ),
                Ok(p) => p.completed + p.failed == progress.completed + progress.failed + 1,
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if self.ring.in_flight_len() == 0 {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                );
            }
            return Err((CoreError::WouldConflict, progress));
        }
        let ghost before_ring = self.ring;
        let completed = match self.ring.wait_one(Tracked(instance)) {
            Ok(c) => c,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(self.timeline),
                        None,
                        FrameTouchMode::PreserveParkedInit,
                    );
                }
                return Err((nvme_error(e), progress));
            },
        };
        let ok = self.finish_one_completion(core, Tracked(instance), completed, Ghost(before_ring));
        if ok {
            progress.completed += 1;
        } else {
            progress.failed += 1;
        }
        Ok(progress)
    }
}

} // verus!
