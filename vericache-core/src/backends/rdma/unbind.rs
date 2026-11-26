use super::*;

verus! {

impl<'a> RdmaBackend<'a> {
    /// Cooperatively release a `Bound` remote read lease: posts local MW
    /// invalidation first (atomic failure safety) and then calls
    /// `core.unpin_fenced_lease`, retaining the `ReturnedShare` until
    /// `finish_unbind_lease` observes `Completed::WindowInvalidated`.
    pub(crate) fn unpin_and_unbind_lease(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(returned_cap): Tracked<ShareCap>,
        read_ticket: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, (CoreError, Tracked<ShareCap>)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).qp.ready(),
            old(self).qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            match old(self).lease_index_spec(handle) {
                Some(idx) => old(self).active_leases@[idx].state == RemoteLeaseState::Bound ==> {
                    &&& returned_cap.share() == old(self).active_leases@[idx].share@
                    &&& old(core).timelines_spec().valid_ticket(read_ticket)
                    &&& old(core).frames_spec().returned_reads(
                        old(self).active_leases@[idx].frame,
                        returned_cap,
                        read_ticket,
                        old(core).kernel_view(),
                    )
                },
                None => true,
            },
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec() == old(core).timelines_spec(),
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
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            r.is_ok() ==> final(core).frames_spec().same_directory(&old(core).frames_spec()),
            match r {
                Err((_, cap_back)) => cap_back@ == returned_cap && *final(core) == *old(core)
                    && final(self).window_pool.slots@ == old(self).window_pool.slots@
                    && final(self).window_pool.free@ == old(self).window_pool.free@
                    && final(self).cq == old(self).cq && final(self).active_leases@ == old(
                    self,
                ).active_leases@,
                Ok(wr_id) => {
                    &&& old(self).lease_index_spec(handle) matches Some(idx)
                    &&& old(self).active_leases@[idx].state == RemoteLeaseState::Bound
                    &&& final(self).lease_index_spec(handle) == Some(idx)
                    &&& final(self).active_leases@.len() == old(self).active_leases@.len()
                    &&& final(self).active_leases@[idx].state == RemoteLeaseState::Unbinding
                    &&& final(self).active_leases@[idx].wr_id == wr_id
                    &&& final(self).cq.in_flight().contains_key(wr_id)
                    &&& final(self).cq.in_flight()[wr_id].kind is LocalInvalidate
                    &&& (final(self).lease_matches_window_state(
                        handle,
                        RemoteLeaseState::Unbinding,
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
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        proof {
            reveal(rdma_leases_valid);
            reveal(window_pool_slots_wf);
        }
        let lease_idx = match self.find_lease_index(handle) {
            Some(i) => i,
            None => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err((CoreError::InvalidArgument, Tracked(returned_cap)));
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Bound) {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err((CoreError::WouldConflict, Tracked(returned_cap)));
        }
        if self.cq.in_flight_count() >= self.cq.capacity_exec() {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err((CoreError::CapacityExhausted, Tracked(returned_cap)));
        }
        if self.next_control_wr_id >= CONTROL_WR_ID_LIMIT {
            proof {
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err((CoreError::SequenceExhausted, Tracked(returned_cap)));
        }
        let ghost before_core = *core;
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let wr_id = self.next_control_wr_id;
        proof {
            lemma_rdma_lease_extract(
                before_leases,
                &before_pool,
                &self.qp,
                &before_core,
                lease_idx as int,
            );
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
        let frame = self.active_leases[lease_idx].frame;
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
                lemma_peer_core_step_refl(core, self.timeline_id);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err((e, Tracked(returned_cap)));
        }
        self.next_control_wr_id += 1;
        let Tracked(returned_share) = match core.unpin_fenced_lease(
            frame,
            Tracked(returned_cap),
            read_ticket,
            Tracked(instance),
        ) {
            Ok(rs) => rs,
            Err((e, cap_back)) => {
                assert(false);
                return Err((e, cap_back));
            },
        };
        self.active_leases[lease_idx].wr_id = wr_id;
        self.active_leases[lease_idx].state = RemoteLeaseState::Unbinding;
        self.active_leases[lease_idx].returned_share = Some(Tracked(returned_share));
        proof {
            let req = rdma::CqSnapshot::of(self.cq).pending[wr_id];
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            lemma_rdma_window_after_control_post(
                self.in_flight_ops,
                self.bitmap,
                &before_core.timelines_spec(),
                self.timeline_id,
                &before_core.ledger_spec()@,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                wr_id,
                req,
            );
            lemma_rdma_window_after_watermarks(
                self.in_flight_ops,
                self.bitmap,
                &before_core.timelines_spec(),
                &core.timelines_spec(),
                self.timeline_id,
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
            assert(core.frames_spec().metas_spec().len()
                == before_core.frames_spec().metas_spec().len() && core.frames_spec().frame(
                frame,
            ).slot == before_core.frames_spec().frame(frame).slot) by {
                reveal(FrameTable::same_directory);
            }
            lemma_rdma_leases_update_in_place(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
                Some(before_leases[lease_idx as int].share@),
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
            reveal(peer_core_step_except);
            reveal(core_transition);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

            assert(rdma_data_events(rdma::CqSnapshot::of(self.cq)) =~= rdma_data_events(before_cq));
            assert forall|j: int|
                0 <= j < self.active_leases@.len() && !(
                #[trigger] self.active_leases@[j]).is_unpinned() implies exists|i: int|
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == self.active_leases@[j].frame
                    && before_leases[i].share@ == self.active_leases@[j].share@ by {
                assert(j != lease_idx as int);
                assert(self.active_leases@[j] == before_leases[j]);
                assert(!before_leases[j].is_unpinned());
            }
            lemma_foreign_shares_after_unpin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
        }
        Ok(wr_id)
    }

    /// Complete a cooperative unbind after observing `Completed::WindowInvalidated`,
    /// returning the unbound MW slot to `self.window_pool`.
    pub(crate) fn finish_unbind_lease(
        &mut self,
        core: &CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(witness): Tracked<rdma::InvalidateWitness>,
    ) -> (r: BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).lease_matches_window_state(handle, RemoteLeaseState::Unbinding) matches Some(
                w,
            ) ==> witness.matches(w),
        ensures
            final(self).well_formed(core),
            peer_foreign_shares_preserved(
                core,
                core,
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            match r {
                Err(_) => *final(self) == *old(self),
                Ok(()) => {
                    &&& old(self).lease_index_spec(handle) is Some
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
        let _ = core;
        proof {
            reveal(rdma_leases_valid);
            reveal(window_pool_slots_wf);
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
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Unbinding) {
            proof {
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        proof {
            lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, lease_idx as int);
        }
        let lease = self.active_leases.swap_remove(lease_idx);
        let Tracked(returned) = lease.returned_share.unwrap();
        self.window_pool.finish_invalidate(
            lease.mw_slot,
            lease.generation,
            Tracked(witness),
            Tracked(Some(returned)),
        );
        proof {
            assert(core.frames_spec().same_directory(&core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().same_other_pins(&core.frames_spec(), Some(lease.frame))) by {
                reveal(FrameTable::same_other_pins);
            }
            lemma_rdma_leases_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &self.qp,
                core,
                core,
                None,
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
                lease.mw_slot,
            );
            reveal(rdma_leases_valid);
            lemma_rdma_leases_swap_remove_preserves(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                handle,
            );
            lemma_swap_remove_foreign_shares_same_pins(
                core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
        }
        Ok(())
    }

    /// Recover a lease stuck in `RemoteLeaseState::Unbinding` once the QP has been
    /// reset (`IBV_QPS_RESET`): deallocates the MW via `self.window_pool.dealloc_on_qp_reset`
    /// using the already-retained `lease.returned_share` without unpinning `core` a second time,
    /// leaving the slot `Vacant`.
    pub(crate) fn recover_unbinding_on_qp_reset(
        &mut self,
        core: &CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(qp_reset): Tracked<&rdma::QpResetWitness>,
    ) -> (r: BackendResult<u32>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            qp_reset.matches_qp(old(self).qp.id(), core.node()),
        ensures
            final(self).well_formed(core),
            peer_foreign_shares_preserved(
                core,
                core,
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).cq == old(self).cq,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            match r {
                Err(_) => final(self).window_pool.slots@ == old(self).window_pool.slots@
                    && final(self).window_pool.free@ == old(self).window_pool.free@
                    && final(self).active_leases@ == old(self).active_leases@,
                Ok(slot_idx) => {
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
        let _ = core;
        proof {
            reveal(rdma_leases_valid);
            reveal(window_pool_slots_wf);
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
        let is_unbinding_like =
            matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Unbinding) || (
        matches!(self.active_leases[lease_idx].state, RemoteLeaseState::ControlFailedState)
            && self.active_leases[lease_idx].returned_share.is_some());
        if !is_unbinding_like {
            proof {
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        proof {
            lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, lease_idx as int);
        }
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        let frame = self.active_leases[lease_idx].frame;
        let Tracked(returned) = self.active_leases[lease_idx].returned_share.take().unwrap();
        proof {
            let w = self.window_pool.slots[mw_slot as int].window->Some_0;
            assert(w.state().binding().qp_id == self.qp.id());
            assert(w.node() == core.node());
        }
        if let Err((e, ret_sh)) = self.window_pool.dealloc_on_qp_reset(
            mw_slot,
            generation,
            Tracked(qp_reset),
            Tracked(Some(returned)),
        ) {
            let Tracked(sh_opt) = ret_sh;
            self.active_leases[lease_idx].returned_share = Some(Tracked(sh_opt.tracked_unwrap()));
            proof {
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
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(e);
        }
        let _removed = self.active_leases.swap_remove(lease_idx);
        proof {
            assert(core.frames_spec().same_directory(&core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().same_other_pins(&core.frames_spec(), Some(frame))) by {
                reveal(FrameTable::same_other_pins);
            }
            lemma_rdma_leases_swap_remove(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &self.qp,
                core,
                core,
                None,
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
            assert forall|j: int|
                0 <= j < self.active_leases@.len() && !(
                #[trigger] self.active_leases@[j]).is_unpinned() implies exists|i: int|
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == self.active_leases@[j].frame
                    && before_leases[i].share@ == self.active_leases@[j].share@ by {
                let oi = if j == lease_idx as int {
                    before_leases.len() - 1
                } else {
                    j
                };
                assert(self.active_leases@[j] == before_leases[oi]);
                assert(!before_leases[oi].is_unpinned());
            }
            lemma_foreign_shares_same_pins(core, core, before_leases, self.active_leases@);
        }
        Ok(mw_slot)
    }

    pub(super) fn rollback_bind_post(
        &mut self,
        core: &mut CacheCore,
        frame: FrameId,
        mw_slot: u32,
        generation: u32,
        Tracked(share_cap): Tracked<ShareCap>,
        Tracked(returned_grant): Tracked<rdma::WindowBindGrant>,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(after_pin_core): Ghost<CacheCore>,
        Ghost(before_pool): Ghost<WindowPool<'a>>,
        Ghost(before_leases): Ghost<Seq<ActiveRemoteLease>>,
        Ghost(share): Ghost<TransferShare>,
    )
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            after_pin_core.well_formed(),
            *old(core) == after_pin_core,
            instance.id() == old(core).instance_id(),
            old(self).timeline_id < old(core).timelines_spec().count_spec(),
            old(core).timelines_spec().kinds_spec()[old(self).timeline_id as int] == (
            TimelineKind::RdmaQueuePair { peer_node: old(self).peer_node }),
            old(self).qp.ready(),
            old(self).qp.state() is ReadyToSend,
            old(self).qp.peer() == old(self).peer_node,
            old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.source().id() == old(core).instance_id(),
            old(self).qp.source().node() == old(core).node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == old(core).node(),
            old(self).cq.infos().instance_id() == old(core).instance_id(),
            old(self).window_pool.well_formed(),
            old(self).window_pool.pd() == before_pool.pd(),
            old(self).window_pool.node() == old(core).node(),
            old(self).window_pool.slots.len() == before_pool.slots.len(),
            (mw_slot as int) < old(self).window_pool.slots.len(),
            old(self).window_pool.slots[mw_slot as int].generation == generation,
            old(self).window_pool.slots[mw_slot as int].state == MwSlotState::Binding(
                frame.0 as u64,
            ),
            old(self).window_pool.slots[mw_slot as int].window matches Some(w) && w.state()
                == rdma::WindowState::Unbound,
            before_pool.well_formed(),
            before_pool.slots.len() == old(self).window_pool.slots.len(),
            forall|j: int|
                0 <= j < old(self).window_pool.slots.len() && j != mw_slot as int ==> (
                #[trigger] old(self).window_pool.slots[j]) == before_pool.slots[j],
            old(self).active_leases@ == before_leases,
            forall|i: int|
                0 <= i < before_leases.len() ==> (#[trigger] before_leases[i]).mw_slot != mw_slot,
            rdma_leases_valid(before_leases, &before_pool, &old(self).qp, &before_core),
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).effective_host_mr_spec() matches Some(s) ==> s.well_formed(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).effective_gpu_mr_spec() matches Some(s) ==> s.well_formed(
                old(core),
                old(self).window_pool.pd(),
            ),
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &old(core).timelines_spec(),
                old(self).timeline_id,
                &old(core).ledger_spec()@,
                rdma::CqSnapshot::of(old(self).cq),
                old(core).instance_id(),
                old(self).next_control_wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &old(core).timelines_spec(),
                old(self).timeline_id,
                old(self).cq.observer(),
                &old(core).ledger_spec()@,
            ),
            rdma_control_leases_match_cq(
                before_leases,
                &before_pool,
                rdma::CqSnapshot::of(old(self).cq),
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(before_leases, &before_pool),
            old(core).node_id_spec() == before_core.node_id_spec(),
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers_spec() == before_core.tiers_spec(),
            old(core).spec_spec() == before_core.spec_spec(),
            old(core).version_registry_spec() == before_core.version_registry_spec(),
            old(core).next_version_seq_spec() == before_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            old(core).active_leases_spec() == before_core.active_leases_spec(),
            old(core).timelines_spec() == before_core.timelines_spec(),
            old(core).ledger_spec() == before_core.ledger_spec(),
            old(core).frames_spec().same_directory(&before_core.frames_spec()),
            old(core).frames_spec().preserves_initializing(&before_core.frames_spec(), None),
            old(core).frames_spec().same_other_pins(&before_core.frames_spec(), Some(frame)),
            frame.0 < old(core).frames_spec().metas_spec().len(),
            old(core).frames_spec().meta(frame).state == STATE_PUBLISHED,
            !before_core.frames_spec().frame(frame).has_fenced_share(share),
            old(core).frames_spec().frame(frame).fenced_shares == before_core.frames_spec().frame(
                frame,
            ).fenced_shares.insert(share),
            old(core).frames_spec().frame(frame).has_fenced_share(share),
            share == share_cap.share(),
            share_cap.is_fenced(),
            share_cap.instance_id() == old(core).instance_id(),
            share_cap.loc() == old(core).frames_spec().frame(frame).slot.loc,
            share_cap.frontier().readers.is_empty(),
            old(core).frames_spec().returned_reads(
                frame,
                share_cap,
                Ticket::NONE,
                old(core).kernel_view(),
            ),
            returned_grant.access() == rdma::RemoteAccess::ReadOnly,
            returned_grant matches rdma::WindowBindGrant::ReadOnly(r) && r.share() == share
                && r.instance_id() == old(core).instance_id() && r.loc() == old(
                core,
            ).frames_spec().frame(frame).slot.loc,
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == before_core.node_id_spec(),
            final(core).instance_id() == before_core.instance_id(),
            final(core).tiers_spec() == before_core.tiers_spec(),
            final(core).spec_spec() == before_core.spec_spec(),
            final(core).version_registry_spec() == before_core.version_registry_spec(),
            final(core).next_version_seq_spec() == before_core.next_version_seq_spec(),
            final(core).version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            final(core).active_leases_spec() == before_core.active_leases_spec(),
            final(core).frames_spec().same_directory(&before_core.frames_spec()),
            final(core).frames_spec().preserves_initializing(&before_core.frames_spec(), None),
            final(core).frames_spec().preserves_fenced_leases(&before_core.frames_spec()),
            peer_foreign_shares_preserved(
                &before_core,
                final(core),
                before_leases,
                final(self).active_leases@,
            ),
            final(core).timelines_spec().only_watermarks(&before_core.timelines_spec()),
            final(core).ledger_spec() == before_core.ledger_spec(),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).completed_pulls == old(self).completed_pulls,
            final(self).active_leases == old(self).active_leases,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).pending() == old(self).pending(),
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool.slots.len() == old(self).window_pool.slots.len(),
            forall|j: int|
                0 <= j < old(self).window_pool.slots.len() && j != mw_slot as int ==> (
                #[trigger] final(self).window_pool.slots[j]) == old(self).window_pool.slots[j],
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        let tracked mut share_cap = share_cap;
        proof {
            lemma_frame_fenced_share_implies_nonzero(&after_pin_core, frame, share);
            let tracked restored_right = match returned_grant {
                rdma::WindowBindGrant::ReadOnly(r) => r,
                rdma::WindowBindGrant::ReadWrite => {
                    assert(false);
                    proof_from_false()
                },
            };
            share_cap.restore_fence_right(restored_right);
            assert(core.frames_spec().returned_reads(
                frame,
                share_cap,
                Ticket::NONE,
                core.kernel_view(),
            ));
        }
        let _ = core.unpin_fenced_lease(frame, Tracked(share_cap), Ticket::NONE, Tracked(instance));
        let ret = self.window_pool.return_unbound(mw_slot, generation);
        proof {
            assert(ret.is_ok());
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            lemma_rdma_window_after_watermarks(
                self.in_flight_ops,
                self.bitmap,
                &after_pin_core.timelines_spec(),
                &core.timelines_spec(),
                self.timeline_id,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &after_pin_core.timelines_spec(),
                &core.timelines_spec(),
                self.timeline_id,
                self.cq.observer(),
                &after_pin_core.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &after_pin_core,
                core,
                self.window_pool.pd(),
            );
            assert(rdma_pool_slots_owned_by_leases(before_leases, &self.window_pool)) by {
                reveal(rdma_pool_slots_owned_by_leases);
                assert forall|s: int|
                    0 <= s < self.window_pool.slots.len() && ((
                    #[trigger] self.window_pool.slots[s]).state is Binding
                        || self.window_pool.slots[s].state is Bound
                        || self.window_pool.slots[s].state is Invalidating) implies exists|i: int|
                    0 <= i < before_leases.len() && (#[trigger] before_leases[i]).mw_slot as int
                        == s by {
                    assert(s != mw_slot as int);
                    assert(before_pool.slots[s] == self.window_pool.slots[s]);
                }
            }
            assert(rdma_control_leases_match_cq(
                before_leases,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            )) by {
                reveal(rdma_control_leases_match_cq);
                assert forall|i: int|
                    0 <= i < before_leases.len() && (
                    #[trigger] before_leases[i]).has_active_wr() implies {
                    let l = before_leases[i];
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
                    assert(before_leases[i].mw_slot != mw_slot);
                    assert(self.window_pool.slots[before_leases[i].mw_slot as int]
                        == before_pool.slots[before_leases[i].mw_slot as int]);
                }
            }
            reveal(rdma_leases_valid);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::preserves_initializing);
            reveal(FrameTable::same_other_pins);
            broadcast use vstd::set::group_set_lemmas;

            assert(core.frames_spec().frame(frame).fenced_shares
                =~= before_core.frames_spec().frame(frame).fenced_shares);
            assert forall|f: FrameId| f.0 < before_core.frames_spec().metas_spec().len() implies {
                &&& (#[trigger] core.frames_spec().frame(f)).fenced_shares
                    == before_core.frames_spec().frame(f).fenced_shares
                &&& core.frames_spec().meta(f).active_fenced_leases
                    == before_core.frames_spec().meta(f).active_fenced_leases
            } by {
                if f != frame {
                    assert(after_pin_core.frames_spec().frame(f).fenced_shares
                        == before_core.frames_spec().frame(f).fenced_shares);
                    assert(after_pin_core.frames_spec().meta(f).active_fenced_leases
                        == before_core.frames_spec().meta(f).active_fenced_leases);
                } else {
                    assert(before_core.frames_spec().valid_frame(frame, before_core.kernel_view()))
                        by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(core.frames_spec().valid_frame(frame, core.kernel_view())) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(before_core.frames_spec().frame(frame).authority(
                        before_core.frames_spec().meta(frame),
                        before_core.instance_id(),
                    ));
                    assert(core.frames_spec().frame(frame).authority(
                        core.frames_spec().meta(frame),
                        core.instance_id(),
                    ));
                }
            }
            lemma_rdma_leases_after_bind_rollback(
                before_leases,
                &before_pool,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
                mw_slot,
            );
            lemma_foreign_shares_identical_pins(&before_core, core, before_leases);
            assert forall|f: FrameId|
                f.0 < before_core.frames_spec().metas_spec().len() && (
                #[trigger] before_core.frames_spec().meta(f)).state
                    == STATE_INITIALIZING implies core.frames_spec().meta(f)
                == before_core.frames_spec().meta(f) && core.frames_spec().frame(f)
                == before_core.frames_spec().frame(f) by {
                assert(after_pin_core.frames_spec().meta(f) == before_core.frames_spec().meta(f));
                assert(after_pin_core.frames_spec().frame(f) == before_core.frames_spec().frame(f));
            }
        }
    }
}

/// Helper preserving `rdma_leases_valid` on rollback of `post_bind` failure
/// (`pin_fenced_lease` followed immediately by `unpin_fenced_lease` of the same share).
proof fn lemma_rdma_leases_after_bind_rollback<'a>(
    leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    mw_slot: u32,
)
    requires
        rdma_leases_valid(leases, before_pool, qp, before_core),
        after_core.instance_id() == before_core.instance_id(),
        forall|i: int| 0 <= i < leases.len() ==> (#[trigger] leases[i]).mw_slot != mw_slot,
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != mw_slot as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().preserves_fenced_leases(&before_core.frames_spec()),
    ensures
        rdma_leases_valid(leases, after_pool, qp, after_core),
{
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        after_pool,
        qp,
        after_core,
    ) by {
        assert(leases[i].valid(before_pool, qp, before_core));
        assert(after_pool.slots[leases[i].mw_slot as int]
            == before_pool.slots[leases[i].mw_slot as int]);
        assert(after_core.frames_spec().metas_spec().len()
            == before_core.frames_spec().metas_spec().len());
        assert(after_core.frames_spec().frame(leases[i].frame).slot
            == before_core.frames_spec().frame(leases[i].frame).slot);
        if !leases[i].is_unpinned() {
            let f = leases[i].frame;
            assert(after_core.frames_spec().frame(f).fenced_shares
                == before_core.frames_spec().frame(f).fenced_shares);
            assert(after_core.frames_spec().meta(f).active_fenced_leases
                == before_core.frames_spec().meta(f).active_fenced_leases);
        }
    }
}

} // verus!
