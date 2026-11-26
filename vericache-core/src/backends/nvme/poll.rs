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
        Ghost(scope): Ghost<Seq<Option<InFlightNvmeOp>>>,
    ) -> (succeeded: bool)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            old(self).file.well_formed(),
            old(self).file.instance_id() == old(core).instance_id(),
            old(self).file.node() == old(core).node(),
            old(self).file.num_slots() == old(self).num_frames as usize,
            old(self).file.slot_len() == old(core).frames_spec().tier_frame_size(
                old(self).tier,
            ) as u64,
            0 < old(self).file.slot_len() <= MAX_IO_LEN,
            old(self).file.slot_len() % DIRECT_IO_ALIGNMENT == 0,
            before_ring.well_formed(),
            before_ring.instance_id() == old(core).instance_id(),
            before_ring.node() == old(core).node(),
            before_ring.source().source().is_io_uring(),
            old(self).ring.well_formed(),
            old(self).ring.same_ring(before_ring),
            old(self).ring.source() == before_ring.source(),
            completed.describes(before_ring, old(self).ring),
            old(self).tier.valid(old(core).tiers_spec().len()),
            old(core).tiers_spec()[old(self).tier.index() as int].kind == TierKind::LocalNvme,
            old(self).timeline < old(core).timelines_spec().count_spec(),
            old(core).timelines_spec().count_spec() <= MAX_TIMELINES,
            old(core).timelines_spec().kinds_spec()[old(self).timeline as int]
                == TimelineKind::IoUring,
            slab_bound_to_core(
                old(core),
                old(self).tier,
                old(self).first_frame,
                old(self).num_frames,
                old(self).file.space(),
            ),
            old(self).pending_transfers.len() == BITMAP_CAPACITY,
            scope.len() == old(self).pending_transfers@.len(),
            forall|i: int|
                0 <= i < scope.len() && (#[trigger] old(self).pending_transfers@[i]).is_some()
                    ==> scope[i] == old(self).pending_transfers@[i],
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
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), nvme_poll_footprint(old(self).timeline, scope)),
            forall|i: int|
                0 <= i < old(self).pending_transfers@.len() && (
                #[trigger] final(self).pending_transfers@[i]).is_some()
                    ==> final(self).pending_transfers@[i] == old(self).pending_transfers@[i],
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
        let _pending_op = core.complete_data(
            &mut self.in_flight_ops,
            &mut self.bitmap,
            self.timeline,
            seq,
            Ghost(ring_event_map(before_ring)),
            Ghost(node),
            knows,
            success,
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
            lemma_retain_copy_after_knows_growth(
                before_core.ledger_spec()@,
                core.ledger_spec()@,
                node,
                op.copy@,
            );
        }
        proof {
            lemma_core_wf_facts(core);
            lemma_core_catalog_indexed(core);
        }
        let current = match core.catalog().lookup(op.hop.key) {
            Some(entry) => entry.version.origin_node == op.hop.version.origin_node
                && entry.version.seq == op.hop.version.seq,
            None => false,
        };
        let ghost removed_op = op;
        if ok && current {
            let ghost dst_cert;
            proof {
                assert(core.ledger_spec()@.succeeded.contains(ev));
                assert(event_succeeded(core.ledger_spec()@, node, ev));
                lemma_nvme_dst_latest(
                    op.copy@.writer,
                    op.copy@.destination.loc,
                    &core.borrow_ledger().submitted,
                    op.dst_tok.borrow_mut(),
                    instance,
                );
                dst_cert = lemma_nvme_certify_completed_copy(op.copy@, core.ledger_spec()@, node);
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
                assert(core.frames_spec().versions_spec()@
                    =~= after_complete_core.frames_spec().versions_spec()@);
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
            assert(before_core.frames_spec().versions_spec()@.submap_of(
                core.frames_spec().versions_spec()@,
            )) by {
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
            assert(scope[idx as int] == Some(removed_op));
            lemma_nvme_poll_contains(scope, self.timeline, idx as int, removed_op);
            let fp = nvme_poll_footprint(self.timeline, scope);
            lemma_core_transition_after_retirement(
                &before_core,
                core,
                fp,
                Some(removed_op.hop.dst_frame),
                Some(removed_op.hop.src_lease.slot_spec()),
            );
        }
        ok
    }

    /// Drain up to `budget` (capped at `BITMAP_CAPACITY`) currently ready
    /// `io_uring` completions, reporting non-dropping `NvmeProgress` (M8).
    pub(crate) fn poll_completions_up_to(
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
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is None,
            final(self).pending_from(old(self)),
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
        let ghost fp = self.poll_footprint();
        proof {
            lemma_nvme_poll_footprint_frames(self, core);
        }
        let mut progress = NvmeProgress::empty();
        let mut steps: usize = 0;
        proof {
            lemma_core_transition_refl(core, fp);
        }
        while steps < limit
            invariant
                fp == old(self).poll_footprint(),
                self.pending_transfers@.len() == old(self).pending_transfers@.len(),
                forall|i: int|
                    0 <= i < self.pending_transfers@.len() && (
                    #[trigger] self.pending_transfers@[i]).is_some() ==> self.pending_transfers@[i]
                        == old(self).pending_transfers@[i],
                steps <= limit <= BITMAP_CAPACITY as usize,
                limit <= budget,
                core.well_formed(),
                self.well_formed(core),
                self.tier == old(self).tier,
                self.timeline == old(self).timeline,
                self.first_frame == old(self).first_frame,
                self.num_frames == old(self).num_frames,
                core.node_id_spec() == old(core).node_id_spec(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                instance.id() == core.instance_id(),
                old(core).spec_spec()@.versions_monotonic(core.spec_spec()@),
                core_transition(old(core), core, old(self).poll_footprint()),
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
                Ghost(old(self).pending_transfers@),
            );
            proof {
                lemma_versions_monotonic_trans(
                    old(core).spec_spec()@,
                    before_step_core.spec_spec()@,
                    core.spec_spec()@,
                );
                lemma_core_transition_trans_same(old(core), &before_step_core, core, fp);
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
            final(self).tier_spec() == old(self).tier_spec(),
            final(self).timeline_spec() == old(self).timeline_spec(),
            final(self).first_frame_spec() == old(self).first_frame_spec(),
            final(self).num_frames_spec() == old(self).num_frames_spec(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is None,
            final(self).pending_from(old(self)),
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
            final(self).tier_spec() == old(self).tier_spec(),
            final(self).timeline_spec() == old(self).timeline_spec(),
            final(self).first_frame_spec() == old(self).first_frame_spec(),
            final(self).num_frames_spec() == old(self).num_frames_spec(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is None,
            final(self).pending_from(old(self)),
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
        let ghost fp = self.poll_footprint();
        proof {
            lemma_nvme_poll_footprint_frames(self, core);
        }
        if self.ring.in_flight_len() == 0 {
            proof {
                lemma_core_transition_refl(core, fp);
            }
            return Err((CoreError::WouldConflict, progress));
        }
        let ghost before_ring = self.ring;
        let completed = match self.ring.wait_one(Tracked(instance)) {
            Ok(c) => c,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(core, fp);
                }
                return Err((nvme_error(e), progress));
            },
        };
        let ok = self.finish_one_completion(
            core,
            Tracked(instance),
            completed,
            Ghost(before_ring),
            Ghost(old(self).pending_transfers@),
        );
        if ok {
            progress.completed += 1;
        } else {
            progress.failed += 1;
        }
        Ok(progress)
    }
}

} // verus!
