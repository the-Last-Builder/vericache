use super::*;

verus! {

impl<'a> RdmaBackend<'a> {
    /// Step 1 of unilateral fenced revocation via local MW invalidation: post a
    /// `LocalInvalidate` WR on a `Bound` lease without unpinning `core`.
    pub fn begin_fenced_invalidate(&mut self, core: &CacheCore, handle: RemoteLeaseHandle) -> (r:
        BackendResult<u64>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).qp_spec().ready(),
            old(self).qp_spec().state() is ReadyToSend,
        ensures
            final(self).well_formed(core),
            peer_foreign_shares_preserved(
                core,
                core,
                old(self).active_leases_spec()@,
                final(self).active_leases_spec()@,
            ),
            final(self).timeline_id_spec() == old(self).timeline_id_spec(),
            final(self).peer_node_spec() == old(self).peer_node_spec(),
            final(self).qp_spec() == old(self).qp_spec(),
            final(self).host_mr_spec() == old(self).host_mr_spec(),
            final(self).gpu_mr_spec() == old(self).gpu_mr_spec(),
            final(self).shared_host_mr_spec() == old(self).shared_host_mr_spec(),
            final(self).shared_gpu_mr_spec() == old(self).shared_gpu_mr_spec(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls_spec() == old(self).completed_pulls_spec(),
            match r {
                Err(_) => final(self).window_pool_spec().slots_spec()@ == old(
                    self,
                ).window_pool_spec().slots_spec()@ && final(self).window_pool_spec().free_spec()@
                    == old(self).window_pool_spec().free_spec()@ && final(self).cq_spec() == old(
                    self,
                ).cq_spec() && final(self).active_leases_spec()@ == old(self).active_leases_spec()@,
                Ok(wr_id) => {
                    &&& old(self).lease_index_spec(handle) matches Some(idx)
                    &&& old(self).active_leases_spec()@[idx].state == RemoteLeaseState::Bound
                    &&& final(self).lease_index_spec(handle) == Some(idx)
                    &&& final(self).active_leases_spec()@.len() == old(
                        self,
                    ).active_leases_spec()@.len()
                    &&& final(self).active_leases_spec()@[idx].state == RemoteLeaseState::Revoking
                    &&& final(self).active_leases_spec()@[idx].wr_id == wr_id
                    &&& final(self).cq_spec().in_flight().contains_key(wr_id)
                    &&& final(self).cq_spec().in_flight()[wr_id].kind is LocalInvalidate
                    &&& (final(self).lease_matches_window_state(
                        handle,
                        RemoteLeaseState::Revoking,
                    ) matches Some(w)
                        ==> final(self).cq_spec().in_flight()[wr_id].kind->LocalInvalidate_window_id
                        == w.id())
                    &&& forall|j: int|
                        0 <= j < old(self).active_leases_spec()@.len() && j != idx ==> (
                        #[trigger] final(self).active_leases_spec()@[j]) == old(
                            self,
                        ).active_leases_spec()@[j]
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            reveal(rdma_leases_valid);
        }
        let lease_idx = match self.find_lease_index(handle) {
            Some(i) => i,
            None => {
                proof {
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Bound) {
            proof {
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        if self.cq.in_flight_count() >= self.cq.capacity_exec() {
            proof {
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::CapacityExhausted);
        }
        if self.next_control_wr_id >= CONTROL_WR_ID_LIMIT {
            proof {
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::SequenceExhausted);
        }
        let _ = core;
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let wr_id = self.next_control_wr_id;
        proof {
            lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, lease_idx as int);
            lemma_rdma_control_wr_id_fresh(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                self.timeline_id,
                &core.ledger_spec()@,
                before_cq,
                core.instance_id(),
                wr_id,
            );
        }
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        if let Err(e) = self.window_pool.post_invalidate(
            mw_slot,
            generation,
            &self.qp,
            &mut self.cq,
            wr_id,
        ) {
            proof {
                lemma_rdma_leases_same_pool_slots(
                    before_leases,
                    &before_pool,
                    &self.window_pool,
                    &self.qp,
                    core,
                );
                lemma_rdma_opaque_same_pool_slots(
                    before_leases,
                    &before_pool,
                    &self.window_pool,
                    before_cq,
                    self.next_control_wr_id,
                );
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(e);
        }
        self.next_control_wr_id += 1;
        self.active_leases[lease_idx].wr_id = wr_id;
        self.active_leases[lease_idx].state = RemoteLeaseState::Revoking;
        proof {
            let req = rdma::CqSnapshot::of(self.cq).pending[wr_id];
            lemma_rdma_window_after_control_post(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                self.timeline_id,
                &core.ledger_spec()@,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                wr_id,
                req,
            );
            assert(core.frames_spec().same_directory(&core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().same_other_pins(
                &core.frames_spec(),
                Some(before_leases[lease_idx as int].frame),
            )) by {
                reveal(FrameTable::same_other_pins);
            }
            lemma_rdma_leases_update_in_place(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                core,
                core,
                None,
            );
            assert((before_leases[lease_idx as int].mw_slot as int) < before_pool.slots.len()) by {
                reveal(rdma_leases_valid);
                reveal(ActiveRemoteLease::valid);
            }
            lemma_control_leases_match_cq_after_update_invalidate(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                wr_id,
                req,
            );
            lemma_pool_slots_owned_after_update_in_place(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
            );
            reveal(rdma_leases_valid);
            reveal(rdma_leases_disjoint);
            assert forall|j: int|
                0 <= j < self.active_leases@.len() && !(
                #[trigger] self.active_leases@[j]).is_unpinned() implies exists|i: int|
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == self.active_leases@[j].frame
                    && before_leases[i].share@ == self.active_leases@[j].share@ by {
                assert(before_leases[j].frame == self.active_leases@[j].frame);
                assert(before_leases[j].share@ == self.active_leases@[j].share@);
                assert(!before_leases[j].is_unpinned());
            }
            lemma_foreign_shares_same_pins(core, core, before_leases, self.active_leases@);
        }
        Ok(wr_id)
    }

    /// Step 2 of unilateral fenced revocation via local MW invalidation:
    /// `core.begin_fenced_revocation` -> `revoke_read_share_with_window` ->
    /// `core.finish_fenced_revocation`. Consumes/deallocates the MW and transitions
    /// the slot to `MwSlotState::Quarantined(_)`. If deallocation fails, cancels
    /// via `core.cancel_fenced_revocation` and leaves all state untouched.
    pub fn finish_fenced_revoke_with_invalidate(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        handle: RemoteLeaseHandle,
        Tracked(witness): Tracked<rdma::InvalidateWitness>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<u32>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
            old(self).lease_matches_window_state(handle, RemoteLeaseState::Revoking) matches Some(w)
                ==> witness.matches(w),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases_spec()@,
                final(self).active_leases_spec()@,
            ),
            final(self).timeline_id_spec() == old(self).timeline_id_spec(),
            final(self).peer_node_spec() == old(self).peer_node_spec(),
            final(self).qp_spec() == old(self).qp_spec(),
            final(self).cq_spec() == old(self).cq_spec(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).host_mr_spec() == old(self).host_mr_spec(),
            final(self).gpu_mr_spec() == old(self).gpu_mr_spec(),
            final(self).shared_host_mr_spec() == old(self).shared_host_mr_spec(),
            final(self).shared_gpu_mr_spec() == old(self).shared_gpu_mr_spec(),
            r.is_ok() ==> final(core).frames_spec().same_directory(&old(core).frames_spec()),
            match r {
                Err(_) => *final(core) == *old(core) && final(self).window_pool_spec().slots_spec()@
                    == old(self).window_pool_spec().slots_spec()@
                    && final(self).window_pool_spec().free_spec()@ == old(
                    self,
                ).window_pool_spec().free_spec()@ && final(self).active_leases_spec()@ == old(
                    self,
                ).active_leases_spec()@ && *final(host) == *old(host),
                Ok(slot_idx) => {
                    &&& final(host).stepped(*old(host))
                    &&& slot_idx as usize == handle.mw_slot
                    &&& old(self).lease_index_spec(handle) is Some
                    &&& final(self).window_pool_spec().slots_spec()[slot_idx as int].state_spec() is Quarantined
                    &&& final(self).active_leases_spec()@.len() + 1 == old(
                        self,
                    ).active_leases_spec()@.len()
                    &&& rdma_leases_preserved_except(
                        old(self).active_leases_spec()@,
                        final(self).active_leases_spec()@,
                        handle,
                    )
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        proof {
            reveal(rdma_leases_valid);
        }
        let lease_idx = match self.find_lease_index(handle) {
            Some(i) => i,
            None => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Revoking) {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        let ghost before_core = *core;
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        proof {
            lemma_rdma_lease_extract(
                before_leases,
                &before_pool,
                &self.qp,
                &before_core,
                lease_idx as int,
            );
        }
        let frame = self.active_leases[lease_idx].frame;
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        let ghost share = self.active_leases[lease_idx as int].share@;
        let mut pending = match core.begin_fenced_revocation(frame, Ghost(share)) {
            Ok(p) => p,
            Err(_) => {
                assert(false);
                return Err(CoreError::StaleReplica);
            },
        };
        let tracked mut lent_opt: Option<SlotToken> = None;
        proof {
            vstd::modes::tracked_swap(pending.token.borrow_mut(), &mut lent_opt);
            reveal(window_pool_slots_wf);
        }
        let tracked lent = lent_opt.tracked_unwrap();
        let (Tracked(completion), Tracked(revoked_tok), Tracked(slot_tok)) =
            match self.window_pool.revoke_with_invalidate(
            mw_slot,
            generation,
            host,
            Tracked(instance),
            Tracked(witness),
            Tracked(lent),
        ) {
            Ok(res) => res,
            Err((e, _ret_w, lent_back_trk)) => {
                let Tracked(lent_back) = lent_back_trk;
                pending.token = Tracked(Some(lent_back));
                core.cancel_fenced_revocation(pending);
                proof {
                    core.lemma_same_state(&before_core);
                    lemma_rdma_leases_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                    );
                    lemma_rdma_opaque_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        rdma::CqSnapshot::of(self.cq),
                        self.next_control_wr_id,
                    );
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        let _removed = self.active_leases.swap_remove(lease_idx);
        core.finish_fenced_revocation(
            pending,
            Tracked(completion.submitted),
            Tracked(completion.knows),
            Tracked(slot_tok),
            Tracked(&revoked_tok),
        );
        proof {
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            assert(before_core.ledger_spec()@.knows.set().subset_of(
                core.ledger_spec()@.knows.set(),
            )) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            lemma_rdma_window_after_fence_ledger(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                self.timeline_id,
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines_spec(),
                &core.timelines_spec(),
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
            lemma_rdma_leases_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &self.qp,
                &before_core,
                core,
                Some(share),
            );
            lemma_control_leases_match_cq_after_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
                None,
            );
            lemma_pool_slots_owned_after_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                mw_slot,
            );
            reveal(rdma_leases_valid);
            lemma_rdma_leases_swap_remove_preserves(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                handle,
            );
            reveal(peer_core_step_except);
            reveal(core_transition);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            lemma_swap_remove_foreign_shares_after_unpin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        }
        Ok(mw_slot)
    }

    /// Finish step 2 of unilateral fenced revocation once `pump_cq` has already reaped
    /// `WindowInvalidated` and staged `inv_witness` into `RemoteLeaseState::RevokeInvReady`.
    pub(crate) fn finish_ready_fenced_revoke(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        handle: RemoteLeaseHandle,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<u32>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            r.is_ok() ==> final(core).frames_spec().same_directory(&old(core).frames_spec()),
            match r {
                Err(_) => *final(core) == *old(core) && final(self).window_pool.slots@ == old(
                    self,
                ).window_pool.slots@ && final(self).window_pool.free@ == old(self).window_pool.free@
                    && final(self).active_leases@ == old(self).active_leases@ && *final(host)
                    == *old(host),
                Ok(slot_idx) => {
                    &&& final(host).stepped(*old(host))
                    &&& slot_idx as usize == handle.mw_slot
                    &&& old(self).lease_index_spec(handle) is Some
                    &&& final(self).window_pool.slots[slot_idx as int].state is Quarantined
                    &&& final(self).active_leases@.len() + 1 == old(self).active_leases@.len()
                    &&& rdma_leases_preserved_except(
                        old(self).active_leases@,
                        final(self).active_leases@,
                        handle,
                    )
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        proof {
            reveal(rdma_leases_valid);
        }
        let lease_idx = match self.find_lease_index(handle) {
            Some(i) => i,
            None => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::RevokeInvReady) {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        let ghost before_core = *core;
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        proof {
            lemma_rdma_lease_extract(
                before_leases,
                &before_pool,
                &self.qp,
                &before_core,
                lease_idx as int,
            );
        }
        let frame = self.active_leases[lease_idx].frame;
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        let ghost share = self.active_leases[lease_idx as int].share@;
        let mut pending = match core.begin_fenced_revocation(frame, Ghost(share)) {
            Ok(p) => p,
            Err(_) => {
                assert(false);
                return Err(CoreError::StaleReplica);
            },
        };
        let tracked mut lent_opt: Option<SlotToken> = None;
        proof {
            vstd::modes::tracked_swap(pending.token.borrow_mut(), &mut lent_opt);
            reveal(window_pool_slots_wf);
        }
        let tracked lent = lent_opt.tracked_unwrap();
        let Tracked(witness) = self.active_leases[lease_idx].inv_witness.take().unwrap();
        let (Tracked(completion), Tracked(revoked_tok), Tracked(slot_tok)) =
            match self.window_pool.revoke_with_invalidate(
            mw_slot,
            generation,
            host,
            Tracked(instance),
            Tracked(witness),
            Tracked(lent),
        ) {
            Ok(res) => res,
            Err((e, ret_w, lent_back_trk)) => {
                let Tracked(lent_back) = lent_back_trk;
                self.active_leases[lease_idx].inv_witness = Some(ret_w);
                pending.token = Tracked(Some(lent_back));
                core.cancel_fenced_revocation(pending);
                proof {
                    core.lemma_same_state(&before_core);
                    assert(self.active_leases@ =~= before_leases);
                    lemma_rdma_leases_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                    );
                    lemma_rdma_opaque_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        rdma::CqSnapshot::of(self.cq),
                        self.next_control_wr_id,
                    );
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        let _removed = self.active_leases.swap_remove(lease_idx);
        core.finish_fenced_revocation(
            pending,
            Tracked(completion.submitted),
            Tracked(completion.knows),
            Tracked(slot_tok),
            Tracked(&revoked_tok),
        );
        proof {
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            assert(before_core.ledger_spec()@.knows.set().subset_of(
                core.ledger_spec()@.knows.set(),
            )) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            lemma_rdma_window_after_fence_ledger(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                self.timeline_id,
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines_spec(),
                &core.timelines_spec(),
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
            lemma_rdma_leases_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &self.qp,
                &before_core,
                core,
                Some(share),
            );
            lemma_control_leases_match_cq_after_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
                None,
            );
            lemma_pool_slots_owned_after_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                mw_slot,
            );
            reveal(rdma_leases_valid);
            lemma_rdma_leases_swap_remove_preserves(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                handle,
            );
            reveal(peer_core_step_except);
            reveal(core_transition);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            lemma_swap_remove_foreign_shares_after_unpin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        }
        Ok(mw_slot)
    }

    /// Transition the QP into `IBV_QPS_ERR` so in-flight WRs flush to the CQ.
    pub fn enter_qp_error(&mut self, core: &CacheCore) -> (r: BackendResult<
        Tracked<rdma::QpErrorWitness>,
    >)
        requires
            core.well_formed(),
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id_spec() == old(self).timeline_id_spec(),
            final(self).peer_node_spec() == old(self).peer_node_spec(),
            final(self).cq_spec() == old(self).cq_spec(),
            final(self).window_pool_spec() == old(self).window_pool_spec(),
            final(self).host_mr_spec() == old(self).host_mr_spec(),
            final(self).gpu_mr_spec() == old(self).gpu_mr_spec(),
            final(self).shared_host_mr_spec() == old(self).shared_host_mr_spec(),
            final(self).shared_gpu_mr_spec() == old(self).shared_gpu_mr_spec(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls_spec() == old(self).completed_pulls_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            r.is_err() ==> *final(self) == *old(self),
            r matches Ok(w) ==> final(self).qp_spec().state() == rdma::QueuePairState::Error
                && final(self).qp_spec().id() == old(self).qp_spec().id()
                && final(self).qp_spec().source() == old(self).qp_spec().source()
                && final(self).qp_spec().peer() == old(self).qp_spec().peer()
                && final(self).qp_spec().pd_id() == old(self).qp_spec().pd_id()
                && final(self).qp_spec().configuration() == old(self).qp_spec().configuration()
                && w@.matches_qp(old(self).qp_spec().id(), core.node()),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        let ghost before_qp = self.qp;
        let w = match rdma::modify_queue_pair_error(&mut self.qp) {
            Ok(w) => w,
            Err(e) => return Err(rdma_error(e)),
        };
        proof {
            lemma_rdma_leases_after_qp_state_change(
                self.active_leases@,
                &self.window_pool,
                &self.window_pool,
                &before_qp,
                &self.qp,
                core,
            );
        }
        Ok(w)
    }

    /// Reset the QP to `IBV_QPS_RESET` once all pending CQ completions have been
    /// drained (`cq.in_flight_count() == 0`), unquarantining all `Quarantined`
    /// window slots to `Vacant`.
    pub(crate) fn reset_qp(&mut self, core: &CacheCore) -> (r: BackendResult<
        Tracked<rdma::QpResetWitness>,
    >)
        requires
            core.well_formed(),
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).active_leases == old(self).active_leases,
            r.is_err() ==> *final(self) == *old(self),
            r matches Ok(w) ==> {
                &&& final(self).qp.state() == rdma::QueuePairState::Reset
                &&& final(self).qp.id() == old(self).qp.id()
                &&& final(self).qp.source() == old(self).qp.source()
                &&& final(self).qp.peer() == old(self).qp.peer()
                &&& final(self).qp.pd_id() == old(self).qp.pd_id()
                &&& w@.matches_qp(old(self).qp.id(), core.node())
                &&& forall|j: int|
                    0 <= j < final(self).window_pool.slots.len() ==> !(
                    #[trigger] final(self).window_pool.slots[j].state is Quarantined)
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        if self.cq.in_flight_count() != 0 {
            return Err(CoreError::WouldConflict);
        }
        let ghost before_qp = self.qp;
        let ghost before_pool = self.window_pool;
        proof {
            assert(self.cq.in_flight().is_empty()) by {
                broadcast use vstd::map::group_map_lemmas;

            }
        }
        let w = match rdma::reset_queue_pair(&mut self.qp, &self.cq) {
            Ok(w) => w,
            Err(e) => return Err(rdma_error(e)),
        };
        self.window_pool.unquarantine_all();
        self.completed_pulls = [None;BITMAP_SLOTS];
        proof {
            lemma_rdma_leases_after_qp_state_change(
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                &before_qp,
                &self.qp,
                core,
            );
            assert(rdma_control_leases_match_cq(
                self.active_leases@,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            )) by {
                reveal(rdma_control_leases_match_cq);
                assert forall|i: int|
                    0 <= i < self.active_leases@.len() && (
                    #[trigger] self.active_leases@[i]).has_active_wr() implies {
                    let l = self.active_leases@[i];
                    let w = self.window_pool.slots[l.mw_slot as int].window;
                    let cq_snap = rdma::CqSnapshot::of(self.cq);
                    &&& CONTROL_WR_ID_BASE <= l.wr_id < self.next_control_wr_id
                    &&& (l.state == RemoteLeaseState::Binding && cq_snap.pending.contains_key(
                        l.wr_id,
                    )) ==> {
                        &&& cq_snap.pending[l.wr_id].kind is BindWindow
                        &&& (w matches Some(win)
                            ==> cq_snap.pending[l.wr_id].kind->BindWindow_window_id == win.id())
                    }
                    &&& ((l.state == RemoteLeaseState::Unbinding || l.state
                        == RemoteLeaseState::Revoking) && cq_snap.pending.contains_key(l.wr_id))
                        ==> {
                        &&& cq_snap.pending[l.wr_id].kind is LocalInvalidate
                        &&& (w matches Some(win)
                            ==> cq_snap.pending[l.wr_id].kind->LocalInvalidate_window_id
                            == win.id())
                    }
                } by {
                    assert(!(
                    before_pool.slots[self.active_leases@[i].mw_slot as int].state is Quarantined))
                        by {
                        reveal(rdma_leases_valid);
                        assert(self.active_leases@[i].valid(&before_pool, &before_qp, core));
                    }
                }
            }
            assert(rdma_pool_slots_owned_by_leases(self.active_leases@, &self.window_pool)) by {
                reveal(rdma_pool_slots_owned_by_leases);
                assert forall|s: int|
                    0 <= s < self.window_pool.slots.len() && ((
                    #[trigger] self.window_pool.slots[s]).state is Binding
                        || self.window_pool.slots[s].state is Bound
                        || self.window_pool.slots[s].state is Invalidating) implies exists|i: int|
                    0 <= i < self.active_leases@.len() && (
                    #[trigger] self.active_leases@[i]).mw_slot as int == s by {
                    assert(!(before_pool.slots[s].state is Quarantined));
                    assert(self.window_pool.slots[s] == before_pool.slots[s]);
                }
            }
        }
        Ok(w)
    }

    /// Unilateral fenced revocation on a reset QP (`IBV_QPS_RESET`): deallocates
    /// the MW via `rdma::revoke_read_share_on_reset_qp`, leaving the slot `Vacant`.
    /// Works on any `Binding`, `Bound`, or `Revoking` lease.
    pub(crate) fn fenced_revoke_on_qp_reset(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        handle: RemoteLeaseHandle,
        Tracked(qp_reset): Tracked<&rdma::QpResetWitness>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<u32>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
            qp_reset.matches_qp(old(self).qp.id(), old(core).node()),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            r.is_ok() ==> final(core).frames_spec().same_directory(&old(core).frames_spec()),
            match r {
                Err(_) => *final(core) == *old(core) && final(self).window_pool.slots@ == old(
                    self,
                ).window_pool.slots@ && final(self).window_pool.free@ == old(self).window_pool.free@
                    && final(self).active_leases@ == old(self).active_leases@ && *final(host)
                    == *old(host),
                Ok(slot_idx) => {
                    &&& final(host).stepped(*old(host))
                    &&& slot_idx as usize == handle.mw_slot
                    &&& old(self).lease_index_spec(handle) is Some
                    &&& final(self).window_pool.slots[slot_idx as int].state == MwSlotState::Vacant
                    &&& final(self).active_leases@.len() + 1 == old(self).active_leases@.len()
                    &&& rdma_leases_preserved_except(
                        old(self).active_leases@,
                        final(self).active_leases@,
                        handle,
                    )
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        proof {
            reveal(rdma_leases_valid);
        }
        let lease_idx = match self.find_lease_index(handle) {
            Some(i) => i,
            None => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        let is_unpinned = matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Unbinding)
            || (matches!(self.active_leases[lease_idx].state, RemoteLeaseState::ControlFailedState)
            && self.active_leases[lease_idx].returned_share.is_some());
        if is_unpinned {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        let ghost before_core = *core;
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        proof {
            lemma_rdma_lease_extract(
                before_leases,
                &before_pool,
                &self.qp,
                &before_core,
                lease_idx as int,
            );
        }
        let frame = self.active_leases[lease_idx].frame;
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        let ghost share = self.active_leases[lease_idx as int].share@;
        let mut pending = match core.begin_fenced_revocation(frame, Ghost(share)) {
            Ok(p) => p,
            Err(_) => {
                assert(false);
                return Err(CoreError::StaleReplica);
            },
        };
        let tracked mut lent_opt: Option<SlotToken> = None;
        proof {
            vstd::modes::tracked_swap(pending.token.borrow_mut(), &mut lent_opt);
            reveal(window_pool_slots_wf);
        }
        let tracked lent = lent_opt.tracked_unwrap();
        let (Tracked(completion), Tracked(revoked_tok), Tracked(slot_tok)) =
            match self.window_pool.revoke_on_qp_reset(
            mw_slot,
            generation,
            host,
            Tracked(instance),
            Tracked(qp_reset),
            Tracked(lent),
        ) {
            Ok(res) => res,
            Err((e, lent_back_trk)) => {
                let Tracked(lent_back) = lent_back_trk;
                pending.token = Tracked(Some(lent_back));
                core.cancel_fenced_revocation(pending);
                proof {
                    core.lemma_same_state(&before_core);
                    lemma_rdma_leases_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                    );
                    lemma_rdma_opaque_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        rdma::CqSnapshot::of(self.cq),
                        self.next_control_wr_id,
                    );
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        let _removed = self.active_leases.swap_remove(lease_idx);
        core.finish_fenced_revocation(
            pending,
            Tracked(completion.submitted),
            Tracked(completion.knows),
            Tracked(slot_tok),
            Tracked(&revoked_tok),
        );
        proof {
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            assert(before_core.ledger_spec()@.knows.set().subset_of(
                core.ledger_spec()@.knows.set(),
            )) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            lemma_rdma_window_after_fence_ledger(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                self.timeline_id,
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines_spec(),
                &core.timelines_spec(),
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
            lemma_rdma_leases_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &self.qp,
                &before_core,
                core,
                Some(share),
            );
            lemma_control_leases_match_cq_after_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
                None,
            );
            lemma_pool_slots_owned_after_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                mw_slot,
            );
            reveal(rdma_leases_valid);
            lemma_rdma_leases_swap_remove_preserves(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                handle,
            );
            reveal(peer_core_step_except);
            reveal(core_transition);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            lemma_swap_remove_foreign_shares_after_unpin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        }
        Ok(mw_slot)
    }

    /// Reconnect a `Reset` QP back to `ReadyToSend` via `rdma::reconnect_rc`.
    pub(crate) fn reconnect_qp(
        &mut self,
        core: &CacheCore,
        psn: u32,
        peer: rdma::QueuePairEndpoint,
    ) -> (r: BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).qp.state() == rdma::QueuePairState::Reset,
            old(self).qp.peer() == old(self).peer_node,
            old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().well_formed(),
            old(self).qp.source().source().is_rdma_qp(),
            old(self).qp.source().node() != old(self).peer_node,
            psn <= 0xffffff,
            peer.node() == old(self).peer_node,
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).cq == old(self).cq,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            final(self).active_leases == old(self).active_leases,
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> final(self).qp.ready() && final(self).qp.state() is ReadyToSend
                && final(self).qp.id() == old(self).qp.id() && final(self).qp.peer() == old(
                self,
            ).peer_node && final(self).qp.pd_id() == old(self).window_pool.pd()
                && final(self).qp.source() == old(self).qp.source(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        let ghost before_qp = self.qp;
        if let Err(e) = rdma::reconnect_rc(&mut self.qp, psn, peer) {
            return Err(rdma_error(e));
        }
        proof {
            lemma_rdma_leases_after_qp_state_change(
                self.active_leases@,
                &self.window_pool,
                &self.window_pool,
                &before_qp,
                &self.qp,
                core,
            );
        }
        Ok(())
    }

    /// Acknowledge a peer `RevokeAck` for a `Quarantined` MW slot (`Quarantined -> Vacant`).
    pub fn acknowledge_quarantined_window(&mut self, core: &CacheCore, mw_slot: u32) -> (r:
        BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id_spec() == old(self).timeline_id_spec(),
            final(self).peer_node_spec() == old(self).peer_node_spec(),
            final(self).qp_spec() == old(self).qp_spec(),
            final(self).cq_spec() == old(self).cq_spec(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> (mw_slot as int) < final(self).window_pool_spec().slots_spec().len()
                && final(self).window_pool_spec().slots_spec()[mw_slot as int].state_spec()
                == MwSlotState::Vacant,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        let ghost before_pool = self.window_pool;
        self.window_pool.acknowledge_quarantined(mw_slot)?;
        proof {
            lemma_rdma_leases_after_pool_retire_replenish(
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                &self.qp,
                core,
                mw_slot,
            );
            reveal(rdma_control_leases_match_cq);
            reveal(rdma_pool_slots_owned_by_leases);
            reveal(rdma_leases_valid);
            assert forall|i: int| 0 <= i < self.active_leases@.len() implies (
            #[trigger] self.active_leases@[i]).mw_slot != mw_slot by {
                assert(self.active_leases@[i].valid(&before_pool, &self.qp, core));
            }
        }
        Ok(())
    }

    /// Allocate a fresh Type-2 MW into a `Vacant` pool slot (`Vacant -> Free`).
    pub fn replenish_window(
        &mut self,
        core: &CacheCore,
        mw_slot: u32,
        pd: &'a rdma::ProtectionDomain<'a>,
        Tracked(name): Tracked<EventSourceToken>,
    ) -> (r: BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            (mw_slot as int) < old(self).window_pool_spec().slots_spec().len(),
            old(self).window_pool_spec().slots_spec()[mw_slot as int].state_spec()
                == MwSlotState::Vacant,
            pd.id() == old(self).window_pool_spec().pd(),
            pd.node() == core.node(),
            name.well_formed(),
            name.source().is_rdma_qp(),
            name.node() == pd.node(),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id_spec() == old(self).timeline_id_spec(),
            final(self).peer_node_spec() == old(self).peer_node_spec(),
            final(self).qp_spec() == old(self).qp_spec(),
            final(self).cq_spec() == old(self).cq_spec(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            r.is_ok() ==> final(self).window_pool_spec().slots_spec()[mw_slot as int].state_spec()
                == MwSlotState::Free
                && final(self).window_pool_spec().slots_spec()[mw_slot as int].generation_spec()
                == old(self).window_pool_spec().slots_spec()[mw_slot as int].generation_spec() + 1,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        let ghost before_pool = self.window_pool;
        let res = self.window_pool.replenish(mw_slot, pd, Tracked(name));
        proof {
            reveal(rdma_leases_valid);
            reveal(ActiveRemoteLease::valid);
            assert forall|i: int| 0 <= i < self.active_leases@.len() implies (
            #[trigger] self.active_leases@[i]).mw_slot != mw_slot by {
                let l = self.active_leases@[i];
                assert(l.valid(&before_pool, &self.qp, core));
                assert(before_pool.slots[mw_slot as int].state == MwSlotState::Vacant);
            }
            lemma_rdma_leases_after_pool_retire_replenish(
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                &self.qp,
                core,
                mw_slot,
            );
            assert(rdma_control_leases_match_cq(
                self.active_leases@,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            )) by {
                reveal(rdma_control_leases_match_cq);
            }
            assert(rdma_pool_slots_owned_by_leases(self.active_leases@, &self.window_pool)) by {
                reveal(rdma_pool_slots_owned_by_leases);
            }
        }
        res
    }

    pub(super) fn commit_bound_lease(
        &mut self,
        core: &CacheCore,
        frame: FrameId,
        mw_slot: u32,
        generation: u32,
        wr_id: u64,
        Tracked(share_cap): Tracked<ShareCap>,
        Ghost(before_cq): Ghost<rdma::CqSnapshot>,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(before_pool): Ghost<WindowPool<'a>>,
    ) -> (r: (RemoteLeaseHandle, u32, u32, u64))
        requires
            core.well_formed(),
            before_core.well_formed(),
            before_pool.well_formed(),
            old(self).timeline_id < core.timelines_spec().count_spec(),
            (old(self).timeline_id as int) < core.timelines_spec().kinds_spec()@.len(),
            core.timelines_spec().kinds_spec()[old(self).timeline_id as int] == (
            TimelineKind::RdmaQueuePair { peer_node: old(self).peer_node }),
            old(self).qp.ready(),
            old(self).qp.peer() == old(self).peer_node,
            old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.source().id() == core.instance_id(),
            old(self).qp.source().node() == core.node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == core.node(),
            old(self).cq.infos().instance_id() == core.instance_id(),
            old(self).next_control_wr_id == wr_id,
            CONTROL_WR_ID_BASE <= wr_id < CONTROL_WR_ID_LIMIT,
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &core.timelines_spec(),
                old(self).timeline_id,
                &core.ledger_spec()@,
                before_cq,
                core.instance_id(),
                wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &core.timelines_spec(),
                old(self).timeline_id,
                old(self).cq.observer(),
                &core.ledger_spec()@,
            ),
            rdma::CqSnapshot::of(old(self).cq).observer == before_cq.observer,
            rdma::CqSnapshot::of(old(self).cq).pending.contains_key(wr_id),
            !(rdma::CqSnapshot::of(old(self).cq).pending[wr_id].kind is DataTransfer),
            rdma::CqSnapshot::of(old(self).cq).pending[wr_id].kind is BindWindow,
            rdma::CqSnapshot::of(old(self).cq).pending == before_cq.pending.insert(
                wr_id,
                rdma::CqSnapshot::of(old(self).cq).pending[wr_id],
            ),
            old(self).window_pool.well_formed(),
            old(self).window_pool.pd() == before_pool.pd(),
            old(self).window_pool.node() == core.node(),
            old(self).window_pool.slots.len() == before_pool.slots.len(),
            (mw_slot as int) < old(self).window_pool.slots.len(),
            forall|j: int|
                0 <= j < old(self).window_pool.slots.len() && j != mw_slot as int ==> (
                #[trigger] old(self).window_pool.slots[j]) == before_pool.slots[j],
            old(self).window_pool.slots[mw_slot as int].generation == generation,
            old(self).window_pool.slots[mw_slot as int].state == MwSlotState::Binding(
                frame.0 as u64,
            ),
            old(self).window_pool.slots[mw_slot as int].window matches Some(w) && w.consistent()
                && w.pd_id() == old(self).window_pool.pd() && w.node() == core.node()
                && w.state() is Binding && w.state()->Binding_qp_id == old(self).qp.id()
                && w.state()->Binding_share == Some(share_cap.share())
                && w.state()->Binding_instance_id == core.instance_id() && w.state()->Binding_wr_id
                == wr_id && w.state()->Binding_target_slot == core.frames_spec().frame(frame).slot
                && w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                && rdma::CqSnapshot::of(old(self).cq).pending[wr_id].kind->BindWindow_window_id
                == w.id(),
            forall|i: int|
                0 <= i < old(self).active_leases@.len() ==> (#[trigger] old(
                    self,
                ).active_leases@[i]).mw_slot != mw_slot,
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).effective_host_mr_spec() matches Some(s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).effective_gpu_mr_spec() matches Some(s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            rdma_leases_valid(old(self).active_leases@, &before_pool, &old(self).qp, &before_core),
            rdma_control_leases_match_cq(old(self).active_leases@, &before_pool, before_cq, wr_id),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &before_pool),
            core.instance_id() == before_core.instance_id(),
            core.frames_spec().same_directory(&before_core.frames_spec()),
            core.frames_spec().same_other_pins(&before_core.frames_spec(), Some(frame)),
            frame.0 < core.frames_spec().metas_spec().len(),
            core.frames_spec().meta(frame).state == STATE_PUBLISHED,
            !before_core.frames_spec().frame(frame).has_fenced_share(share_cap.share()),
            core.frames_spec().frame(frame).fenced_shares == before_core.frames_spec().frame(
                frame,
            ).fenced_shares.insert(share_cap.share()),
            core.frames_spec().frame(frame).has_fenced_share(share_cap.share()),
            share_cap.is_fenced(),
            share_cap.instance_id() == core.instance_id(),
            share_cap.loc() == core.frames_spec().frame(frame).slot.loc,
            share_cap.frontier().readers.is_empty(),
            core.frames_spec().published(frame) matches Some((_, _, c, _))
                && share_cap.frontier().last_writer == Some(c.writer),
        ensures
            final(self).well_formed(core),
            peer_foreign_shares_preserved(
                &before_core,
                core,
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            r.0.mw_slot == mw_slot as usize,
            r.0.generation == generation,
            r.1 == mw_slot,
            r.2 == generation,
            r.3 == wr_id,
            final(self).active_leases@.len() == old(self).active_leases@.len() + 1,
            final(self).active_leases@.take(old(self).active_leases@.len() as int) == old(
                self,
            ).active_leases@,
            final(self).lease_index_spec(r.0) == Some(
                (final(self).active_leases@.len() - 1) as int,
            ),
            final(self).active_leases@[(final(self).active_leases@.len() - 1) as int].frame
                == frame,
            final(self).active_leases@[(final(self).active_leases@.len() - 1) as int].mw_slot
                == mw_slot,
            final(self).active_leases@[(final(self).active_leases@.len() - 1) as int].generation
                == generation,
            final(self).active_leases@[(final(self).active_leases@.len() - 1) as int].wr_id
                == wr_id,
            final(self).active_leases@[(final(self).active_leases@.len() - 1) as int].state
                == RemoteLeaseState::Binding,
            final(self).active_leases@[(final(self).active_leases@.len()
                - 1) as int].share_cap matches Some(sc) && sc@ == share_cap,
            final(self).cq == old(self).cq,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            final(self).next_control_wr_id == wr_id + 1,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(RegisteredSlab::well_formed);
        hide(WindowPool::well_formed);
        hide(ActiveRemoteLease::valid);
        let _ = core;
        let ghost before_leases = self.active_leases@;
        let ghost share = share_cap.share();
        self.next_control_wr_id += 1;
        let handle = RemoteLeaseHandle { mw_slot: mw_slot as usize, generation };
        self.active_leases.push(
            ActiveRemoteLease {
                frame,
                mw_slot,
                generation,
                wr_id,
                state: RemoteLeaseState::Binding,
                share: Ghost(share),
                share_cap: Some(Tracked(share_cap)),
                returned_share: None,
                bind_witness: None,
                bind_failed_witness: None,
                inv_witness: None,
                staged_error: None,
            },
        );
        proof {
            let req = rdma::CqSnapshot::of(self.cq).pending[wr_id];
            lemma_rdma_window_after_control_post(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                self.timeline_id,
                &core.ledger_spec()@,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                wr_id,
                req,
            );
            lemma_frame_fenced_share_implies_nonzero(core, frame, share);
            assert(self.active_leases@.last().valid(&self.window_pool, &self.qp, core)) by {
                reveal(ActiveRemoteLease::valid);
            }
            lemma_rdma_leases_push(
                before_leases,
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
                self.active_leases@.last(),
            );
            assert((self.active_leases@.last().mw_slot as int) < self.window_pool.slots.len());
            lemma_control_leases_match_cq_after_push_bind(
                before_leases,
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                wr_id,
                self.active_leases@.last(),
                req,
            );
            lemma_pool_slots_owned_after_push(
                before_leases,
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                mw_slot,
            );
            reveal(rdma_leases_valid);
            assert(self.active_leases@.take(before_leases.len() as int) =~= before_leases);
            lemma_rdma_leases_push_index(self.active_leases@, handle);
            reveal(FrameTable::same_directory);
            broadcast use vstd::map::group_map_lemmas;

            assert forall|j: int|
                0 <= j < self.active_leases@.len() && !(
                #[trigger] self.active_leases@[j]).is_unpinned() implies (
            self.active_leases@[j].frame == frame && self.active_leases@[j].share@ == share) || (
            exists|i: int|
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == self.active_leases@[j].frame
                    && before_leases[i].share@ == self.active_leases@[j].share@) by {
                if j < before_leases.len() {
                    assert(self.active_leases@[j] == before_leases[j]);
                    assert(!before_leases[j].is_unpinned());
                }
            }
            lemma_foreign_shares_after_pin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                frame,
                share,
            );
        }
        (handle, mw_slot, generation, wr_id)
    }

    /// Recover/revoke all active remote leases across all 8 `RemoteLeaseState`
    /// variants once the QP has been reset (`IBV_QPS_RESET`).
    pub(crate) fn recover_all_leases_on_qp_reset(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        Tracked(qp_reset): Tracked<&rdma::QpResetWitness>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
            qp_reset.matches_qp(old(self).qp.id(), old(core).node()),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            r.is_ok() ==> final(self).active_leases@.len() == 0,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut guard: usize = self.active_leases.len();
        proof {
            lemma_peer_core_step_refl(core, self.timeline_id);
            lemma_foreign_shares_refl(core, self.active_leases@);
        }
        while self.active_leases.len() > 0 && guard > 0
            invariant
                guard == self.active_leases@.len(),
                core.well_formed(),
                self.well_formed(core),
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                qp_reset.matches_qp(self.qp.id(), core.node()),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).active_leases@,
                    self.active_leases@,
                ),
                self.timeline_id == old(self).timeline_id,
                self.peer_node == old(self).peer_node,
                self.qp == old(self).qp,
                self.cq == old(self).cq,
                self.window_pool.pd() == old(self).window_pool.pd(),
                self.host_mr == old(self).host_mr,
                self.gpu_mr == old(self).gpu_mr,
                self.shared_host_mr == old(self).shared_host_mr,
                self.shared_gpu_mr == old(self).shared_gpu_mr,
            decreases guard,
        {
            let lease_h = RemoteLeaseHandle {
                mw_slot: self.active_leases[0].mw_slot as usize,
                generation: self.active_leases[0].generation,
            };
            let is_binding_like =
                matches!(
                self.active_leases[0].state,
                RemoteLeaseState::Binding | RemoteLeaseState::BindReady
                    | RemoteLeaseState::BindFailedReady
            )
                || (matches!(self.active_leases[0].state, RemoteLeaseState::ControlFailedState)
                && self.active_leases[0].share_cap.is_some());
            let is_unbinding_like =
                matches!(self.active_leases[0].state, RemoteLeaseState::Unbinding) || (
            matches!(self.active_leases[0].state, RemoteLeaseState::ControlFailedState)
                && self.active_leases[0].returned_share.is_some());
            let ghost before_step = *core;
            let ghost before_leases = self.active_leases@;
            let step_res = if is_binding_like {
                self.recover_binding_on_qp_reset(
                    core,
                    lease_h,
                    Tracked(qp_reset),
                    Tracked(instance),
                )
            } else if is_unbinding_like {
                let r = self.recover_unbinding_on_qp_reset(core, lease_h, Tracked(qp_reset));
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                }
                r
            } else {
                self.fenced_revoke_on_qp_reset(
                    core,
                    host,
                    lease_h,
                    Tracked(qp_reset),
                    Tracked(instance),
                )
            };
            proof {
                lemma_peer_core_step_trans(old(core), &before_step, core, self.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &before_step,
                    core,
                    old(self).active_leases@,
                    before_leases,
                    self.active_leases@,
                );
            }
            step_res?;
            guard -= 1;
        }
        Ok(())
    }

    /// Full RDMA teardown: transition QP to Error, drain CQ completions, reset QP,
    /// recover all active remote leases across all 8 states, deallocate all Type-2
    /// MWs in `window_pool`, and release owned/shared MR handles.
    #[verifier::rlimit(250)]
    pub fn shutdown(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases_spec()@,
                final(self).active_leases_spec()@,
            ),
            final(self).timeline_id_spec() == old(self).timeline_id_spec(),
            final(self).peer_node_spec() == old(self).peer_node_spec(),
            final(self).window_pool_spec().pd() == old(self).window_pool_spec().pd(),
            r.is_ok() ==> {
                &&& final(self).active_leases_spec()@.len() == 0
                &&& final(self).qp_spec().state() == rdma::QueuePairState::Reset
                &&& final(self).cq_spec().in_flight().is_empty()
                &&& final(self).window_pool_spec().free_spec()@.len() == 0
                &&& final(self).host_mr_spec() is None
                &&& final(self).gpu_mr_spec() is None
                &&& final(self).shared_host_mr_spec() is None
                &&& final(self).shared_gpu_mr_spec() is None
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.timeline_id);
            lemma_foreign_shares_refl(core, self.active_leases@);
        }
        let _err_w = self.enter_qp_error(core)?;
        let mut spins: u32 = 0;
        while self.cq.in_flight_count() > 0 && spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).active_leases@,
                    self.active_leases@,
                ),
                self.timeline_id == old(self).timeline_id,
                self.peer_node == old(self).peer_node,
                self.window_pool.pd() == old(self).window_pool.pd(),
            decreases 1_000_000 - spins,
        {
            let ghost step_core = *core;
            let ghost step_leases = self.active_leases@;
            let cq_res = self.pump_cq_all(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(old(core), &step_core, core, self.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &step_core,
                    core,
                    old(self).active_leases@,
                    step_leases,
                    self.active_leases@,
                );
            }
            let _ = cq_res?;
            spins += 1;
        }
        if self.cq.in_flight_count() > 0 {
            return Err(CoreError::WouldConflict);
        }
        let Tracked(qp_reset) = self.reset_qp(core)?;
        let ghost before_rec = *core;
        let ghost before_rec_leases = self.active_leases@;
        let rec_res = self.recover_all_leases_on_qp_reset(
            core,
            host,
            Tracked(&qp_reset),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans(old(core), &before_rec, core, self.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_rec,
                core,
                old(self).active_leases@,
                before_rec_leases,
                self.active_leases@,
            );
        }
        rec_res?;
        proof {
            assert forall|j: int| 0 <= j < self.window_pool.slots.len() implies {
                let s = #[trigger] self.window_pool.slots[j];
                &&& (s.state == MwSlotState::Free || s.state is Quarantined || s.state
                    == MwSlotState::Vacant)
                &&& match s.window {
                    None => true,
                    Some(w) => w.state() == rdma::WindowState::Unbound,
                }
            } by {
                reveal(window_pool_slots_wf);
                reveal(rdma_pool_slots_owned_by_leases);
                let s = self.window_pool.slots[j];
                assert(!(s.state is Binding || s.state is Bound || s.state is Invalidating));
            }
        }
        let dealloc_res = self.window_pool.dealloc_all();
        proof {
            lemma_rdma_leases_empty(self.active_leases@, &self.window_pool, &self.qp, core);
            assert(rdma_leases_disjoint(self.active_leases@)) by {
                reveal(rdma_leases_disjoint);
            }
            assert(rdma_pool_slots_owned_by_leases(self.active_leases@, &self.window_pool)) by {
                reveal(rdma_pool_slots_owned_by_leases);
            }
            assert(rdma_control_leases_match_cq(
                self.active_leases@,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            )) by {
                reveal(rdma_control_leases_match_cq);
            }
        }
        dealloc_res?;
        self.host_mr = None;
        self.gpu_mr = None;
        self.shared_host_mr = None;
        self.shared_gpu_mr = None;
        proof {
            assert(rdma_leases_valid(self.active_leases@, &self.window_pool, &self.qp, core)) by {
                reveal(rdma_leases_valid);
                reveal(rdma_leases_disjoint);
            }
            assert(rdma_pool_slots_owned_by_leases(self.active_leases@, &self.window_pool)) by {
                reveal(rdma_pool_slots_owned_by_leases);
            }
            assert(rdma_control_leases_match_cq(
                self.active_leases@,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            )) by {
                reveal(rdma_control_leases_match_cq);
            }
            assert(self.cq.in_flight().is_empty()) by {
                broadcast use vstd::map::group_map_lemmas;

            }
        }
        Ok(())
    }
}

} // verus!
