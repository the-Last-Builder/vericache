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
            old(self).qp.ready(),
            old(self).qp.state() is ReadyToSend,
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            match r {
                Err(_) => final(self).window_pool.slots@ == old(self).window_pool.slots@
                    && final(self).window_pool.free@ == old(self).window_pool.free@
                    && final(self).cq == old(self).cq && final(self).active_leases@ == old(
                    self,
                ).active_leases@,
                Ok(wr_id) => {
                    &&& old(self).lease_index_spec(handle) matches Some(idx)
                    &&& old(self).active_leases@[idx].state == RemoteLeaseState::Bound
                    &&& final(self).lease_index_spec(handle) == Some(idx)
                    &&& final(self).active_leases@.len() == old(self).active_leases@.len()
                    &&& final(self).active_leases@[idx].state == RemoteLeaseState::Revoking
                    &&& final(self).active_leases@[idx].wr_id == wr_id
                    &&& final(self).cq.in_flight().contains_key(wr_id)
                    &&& final(self).cq.in_flight()[wr_id].kind is LocalInvalidate
                    &&& (final(self).lease_matches_window_state(
                        handle,
                        RemoteLeaseState::Revoking,
                    ) matches Some(w)
                        ==> final(self).cq.in_flight()[wr_id].kind->LocalInvalidate_window_id
                        == w.id())
                    &&& forall|j: int|
                        0 <= j < old(self).active_leases@.len() && j != idx ==> (
                        #[trigger] final(self).active_leases@[j]) == old(self).active_leases@[j]
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
            None => return Err(CoreError::InvalidArgument),
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Bound) {
            return Err(CoreError::WouldConflict);
        }
        if self.cq.in_flight_count() >= self.cq.capacity_exec() {
            return Err(CoreError::CapacityExhausted);
        }
        if self.next_control_wr_id == u64::MAX {
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
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
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
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                wr_id,
                req,
            );
            assert(core.frames.same_directory(&core.frames)) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames.same_other_pins(
                &core.frames,
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
            reveal(rdma_leases_valid);
            reveal(rdma_leases_disjoint);
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
        host: &mut HostContext,
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).timeline_id),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            r.is_ok() ==> final(core).frames.same_directory(&old(core).frames),
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
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Revoking) {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
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
        let mut pending = match core.begin_fenced_revocation(
            frame,
            mw_slot,
            generation,
            Ghost(share),
        ) {
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
                    lemma_rdma_leases_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                    );
                    lemma_peer_core_step_refl(core, self.timeline_id);
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
            assert(before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set())) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            lemma_rdma_window_after_fence_ledger(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines,
                self.timeline_id,
                &before_core.ledger@,
                &core.ledger@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_rdma_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
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
            reveal(rdma_leases_valid);
            lemma_rdma_leases_swap_remove_preserves(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                handle,
            );
            reveal(peer_core_step_except);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
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
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).cq == old(self).cq,
            final(self).window_pool == old(self).window_pool,
            final(self).active_leases == old(self).active_leases,
            r.is_err() ==> *final(self) == *old(self),
            r matches Ok(w) ==> final(self).qp.state() == rdma::QueuePairState::Error
                && final(self).qp.id() == old(self).qp.id() && final(self).qp.source() == old(
                self,
            ).qp.source() && final(self).qp.peer() == old(self).qp.peer() && final(self).qp.pd_id()
                == old(self).qp.pd_id() && final(self).qp.configuration() == old(
                self,
            ).qp.configuration() && w@.matches_qp(old(self).qp.id(), core.node()),
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
    pub fn reset_qp(&mut self, core: &CacheCore) -> (r: BackendResult<
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
        proof {
            lemma_rdma_leases_after_qp_state_change(
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                &before_qp,
                &self.qp,
                core,
            );
        }
        Ok(w)
    }

    /// Unilateral fenced revocation on a reset QP (`IBV_QPS_RESET`): deallocates
    /// the MW via `rdma::revoke_read_share_on_reset_qp`, leaving the slot `Vacant`.
    /// Works on any `Binding`, `Bound`, or `Revoking` lease.
    pub fn fenced_revoke_on_qp_reset(
        &mut self,
        core: &mut CacheCore,
        host: &mut HostContext,
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).timeline_id),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            r.is_ok() ==> final(core).frames.same_directory(&old(core).frames),
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
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Unbinding) {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
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
        let mut pending = match core.begin_fenced_revocation(
            frame,
            mw_slot,
            generation,
            Ghost(share),
        ) {
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
                    lemma_rdma_leases_same_pool_slots(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                    );
                    lemma_peer_core_step_refl(core, self.timeline_id);
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
            assert(before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set())) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            lemma_rdma_window_after_fence_ledger(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines,
                self.timeline_id,
                &before_core.ledger@,
                &core.ledger@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_rdma_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
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
            reveal(rdma_leases_valid);
            lemma_rdma_leases_swap_remove_preserves(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                handle,
            );
            reveal(peer_core_step_except);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        }
        Ok(mw_slot)
    }

    /// Reconnect a `Reset` QP back to `ReadyToSend` via `rdma::reconnect_rc`.
    pub fn reconnect_qp(&mut self, core: &CacheCore, psn: u32, peer: rdma::QueuePairEndpoint) -> (r:
        BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).qp.state() == rdma::QueuePairState::Reset,
            old(self).qp.peer() == old(self).peer_node,
            old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().well_formed(),
            old(self).qp.source().source() is RdmaQueuePair,
            old(self).qp.source().node() != old(self).peer_node,
            psn <= 0xffffff,
            peer.node() == old(self).peer_node,
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).cq == old(self).cq,
            final(self).window_pool == old(self).window_pool,
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
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).active_leases == old(self).active_leases,
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> (mw_slot as int) < final(self).window_pool.slots.len()
                && final(self).window_pool.slots[mw_slot as int].state == MwSlotState::Vacant,
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
            (mw_slot as int) < old(self).window_pool.slots.len(),
            old(self).window_pool.slots[mw_slot as int].state == MwSlotState::Vacant,
            pd.id() == old(self).window_pool.pd(),
            pd.node() == core.node(),
            name.well_formed(),
            name.source() is RdmaQueuePair,
            name.node() == pd.node(),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).active_leases == old(self).active_leases,
            r.is_ok() ==> final(self).window_pool.slots[mw_slot as int].state == MwSlotState::Free
                && final(self).window_pool.slots[mw_slot as int].generation == old(
                self,
            ).window_pool.slots[mw_slot as int].generation + 1,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        let ghost before_pool = self.window_pool;
        let res = self.window_pool.replenish(mw_slot, pd, Tracked(name));
        proof {
            lemma_rdma_leases_after_pool_retire_replenish(
                self.active_leases@,
                &before_pool,
                &self.window_pool,
                &self.qp,
                core,
                mw_slot,
            );
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
            old(self).timeline_id < core.timelines.count,
            (old(self).timeline_id as int) < core.timelines.kinds@.len(),
            core.timelines.kinds[old(self).timeline_id as int] == (TimelineKind::RdmaQueuePair {
                peer_node: old(self).peer_node,
            }),
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
            wr_id < u64::MAX,
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &core.timelines,
                old(self).timeline_id,
                &core.ledger@,
                before_cq,
                core.instance_id(),
                wr_id,
            ),
            rdma::CqSnapshot::of(old(self).cq).observer == before_cq.observer,
            rdma::CqSnapshot::of(old(self).cq).pending.contains_key(wr_id),
            !(rdma::CqSnapshot::of(old(self).cq).pending[wr_id].kind is DataTransfer),
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
                == wr_id && w.state()->Binding_target_slot == core.frames.frame(frame).slot
                && w.state()->Binding_access == rdma::RemoteAccess::ReadOnly,
            forall|i: int|
                0 <= i < old(self).active_leases@.len() ==> (#[trigger] old(
                    self,
                ).active_leases@[i]).mw_slot != mw_slot,
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            rdma_leases_valid(old(self).active_leases@, &before_pool, &old(self).qp, &before_core),
            core.instance_id() == before_core.instance_id(),
            core.frames.same_directory(&before_core.frames),
            core.frames.same_other_pins(&before_core.frames, Some(frame)),
            frame.0 < core.frames.metas.len(),
            core.frames.meta(frame).state == STATE_PUBLISHED,
            !before_core.frames.frame(frame).remote_shares.contains_key(share_cap.share()),
            core.frames.frame(frame).remote_shares == before_core.frames.frame(
                frame,
            ).remote_shares.insert(
                share_cap.share(),
                crate::frame::RemoteShareEntry { mw_slot, mw_gen: generation },
            ),
            core.frames.frame(frame).matches_lease(share_cap.share(), mw_slot, generation),
            share_cap.is_rdma_bound(),
            share_cap.instance_id() == core.instance_id(),
            share_cap.loc() == core.frames.frame(frame).slot.loc,
            share_cap.frontier().readers.is_empty(),
            core.frames.published(frame) matches Some((_, _, c, _))
                && share_cap.frontier().last_writer == Some(c.writer),
        ensures
            final(self).well_formed(core),
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
            },
        );
        proof {
            let req = rdma::CqSnapshot::of(self.cq).pending[wr_id];
            lemma_rdma_window_after_control_post(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                wr_id,
                req,
            );
            lemma_frame_remote_share_implies_nonzero(core, frame, share);
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
            reveal(rdma_leases_valid);
            assert(self.active_leases@.take(before_leases.len() as int) =~= before_leases);
            lemma_rdma_leases_push_index(self.active_leases@, handle);
        }
        (handle, mw_slot, generation, wr_id)
    }
}

} // verus!
