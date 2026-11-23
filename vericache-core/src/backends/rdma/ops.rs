use super::*;

verus! {

impl<'a> RdmaBackend<'a> {
    /// Bind a fresh Type-2 MemoryWindow to a published frame in `host_mr` or
    /// `gpu_mr` via `core.pin_remote_lease` + `WindowPool::checkout` (M5) +
    /// `post_bind_window`. On any failure, restores `WindowBindRight`, unpins
    /// the frame, and returns the unbound MW slot without leaking.
    #[verifier::rlimit(250)]
    pub fn bind_remote_read_lease(
        &mut self,
        core: &mut CacheCore,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<(RemoteLeaseHandle, u32, u32, u64)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).qp.ready(),
            old(self).qp.state() is ReadyToSend,
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).frames.same_directory(&old(core).frames),
            final(core).frames.preserves_initializing(&old(core).frames, None),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(core).timelines.only_watermarks(&old(core).timelines) || final(core).timelines
                == old(core).timelines,
            final(core).ledger == old(core).ledger,
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            r.is_err() ==> final(self).active_leases == old(self).active_leases,
            r matches Ok((handle, mw_slot, generation, wr_id)) ==> {
                let idx = (final(self).active_leases@.len() - 1) as int;
                &&& handle.mw_slot == mw_slot as usize
                &&& handle.generation == generation
                &&& final(self).active_leases@.len() == old(self).active_leases@.len() + 1
                &&& final(self).active_leases@.take(old(self).active_leases@.len() as int) == old(
                    self,
                ).active_leases@
                &&& final(self).lease_index_spec(handle) == Some(idx)
                &&& final(self).active_leases@[idx].frame == frame
                &&& final(self).active_leases@[idx].mw_slot == mw_slot
                &&& final(self).active_leases@[idx].generation == generation
                &&& final(self).active_leases@[idx].wr_id == wr_id
                &&& final(self).active_leases@[idx].state == RemoteLeaseState::Binding
                &&& (final(self).active_leases@[idx].share_cap matches Some(sc)
                    && final(core).frames.published(frame) matches Some((_, _, c, _))
                    && sc@.frontier().last_writer == Some(c.writer))
                &&& final(self).cq.in_flight().contains_key(wr_id)
                &&& final(self).cq.in_flight()[wr_id].kind is BindWindow
                &&& (final(self).lease_matches_window_state(
                    handle,
                    RemoteLeaseState::Binding,
                ) matches Some(w) ==> final(self).cq.in_flight()[wr_id].kind->BindWindow_window_id
                    == w.id())
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        let in_host = match self.effective_host_mr() {
            Some(s) => s.owns_frame(frame),
            None => false,
        };
        let in_gpu = if in_host {
            false
        } else {
            match self.effective_gpu_mr() {
                Some(s) => s.owns_frame(frame),
                None => false,
            }
        };
        if !in_host && !in_gpu {
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::UnsupportedTier);
        }
        if self.cq.in_flight_count() >= self.cq.capacity_exec() {
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::CapacityExhausted);
        }
        if self.next_control_wr_id >= CONTROL_WR_ID_LIMIT {
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before_core = *core;
        let ghost before_pool = self.window_pool;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let ghost before_leases = self.active_leases@;
        let (mw_slot, generation) = match self.window_pool.checkout(frame.0 as u64) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::preserves_initializing);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        proof {
            lemma_rdma_free_slot_not_in_leases(
                before_leases,
                &before_pool,
                &self.qp,
                &before_core,
                mw_slot,
            );
        }
        let (_version, Ghost(_cert), Tracked(share_cap)) = match core.pin_remote_lease(
            frame,
            mw_slot,
            generation,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                let ret = self.window_pool.return_unbound(mw_slot, generation);
                proof {
                    assert(ret.is_ok());
                    lemma_rdma_leases_after_pool_retire_replenish_back(
                        before_leases,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                        mw_slot,
                    );
                    assert(rdma_pool_slots_owned_by_leases(before_leases, &self.window_pool)) by {
                        reveal(rdma_pool_slots_owned_by_leases);
                        assert forall|s: int|
                            0 <= s < self.window_pool.slots.len() && ((
                            #[trigger] self.window_pool.slots[s]).state is Binding
                                || self.window_pool.slots[s].state is Bound
                                || self.window_pool.slots[s].state is Invalidating) implies exists|
                            i: int,
                        |
                            0 <= i < before_leases.len() && (
                            #[trigger] before_leases[i]).mw_slot as int == s by {
                            assert(s != mw_slot as int);
                            assert(before_pool.slots[s] == self.window_pool.slots[s]);
                        }
                    }
                    assert(rdma_control_leases_match_cq(
                        before_leases,
                        &self.window_pool,
                        before_cq,
                        self.next_control_wr_id,
                    )) by {
                        reveal(rdma_control_leases_match_cq);
                        assert forall|i: int|
                            0 <= i < before_leases.len() && (
                            #[trigger] before_leases[i]).has_active_wr() implies {
                            let l = before_leases[i];
                            let w = self.window_pool.slots[l.mw_slot as int].window;
                            &&& CONTROL_WR_ID_BASE <= l.wr_id < self.next_control_wr_id
                            &&& (l.state == RemoteLeaseState::Binding
                                && before_cq.pending.contains_key(l.wr_id)) ==> {
                                &&& before_cq.pending[l.wr_id].kind is BindWindow
                                &&& (w matches Some(win)
                                    ==> before_cq.pending[l.wr_id].kind->BindWindow_window_id
                                    == win.id())
                            }
                            &&& ((l.state == RemoteLeaseState::Unbinding || l.state
                                == RemoteLeaseState::Revoking) && before_cq.pending.contains_key(
                                l.wr_id,
                            )) ==> {
                                &&& before_cq.pending[l.wr_id].kind is LocalInvalidate
                                &&& (w matches Some(win)
                                    ==> before_cq.pending[l.wr_id].kind->LocalInvalidate_window_id
                                    == win.id())
                            }
                        } by {
                            assert(before_leases[i].mw_slot != mw_slot);
                            assert(self.window_pool.slots[before_leases[i].mw_slot as int]
                                == before_pool.slots[before_leases[i].mw_slot as int]);
                        }
                    }
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        proof {
            lemma_core_wf_facts(core);
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
            );
            if let Some(sh) = self.shared_host_mr {
                sh.lemma_shared_wf_iff(core, self.window_pool.pd());
            }
            if let Some(sg) = self.shared_gpu_mr {
                sg.lemma_shared_wf_iff(core, self.window_pool.pd());
            }
            if in_host {
                let s = self.effective_host_mr_spec().unwrap();
                lemma_slab_frame_slot(
                    core,
                    s.tier,
                    s.first_frame,
                    s.num_frames,
                    s.mr.space(),
                    frame,
                );
            } else {
                let s = self.effective_gpu_mr_spec().unwrap();
                lemma_slab_frame_slot(
                    core,
                    s.tier,
                    s.first_frame,
                    s.num_frames,
                    s.mr.space(),
                    frame,
                );
            }
        }
        let res = self.post_bind_pinned_lease(
            core,
            frame,
            in_host,
            mw_slot,
            generation,
            Tracked(share_cap),
            Tracked(instance),
            Ghost(before_core),
            Ghost(before_pool),
        );
        proof {
            reveal(FrameTable::same_directory);
        }
        res
    }

    fn post_bind_pinned_slice(
        &mut self,
        core: &CacheCore,
        frame: FrameId,
        in_host: bool,
        mw_slot: u32,
        generation: u32,
        Tracked(right): Tracked<WindowBindRight>,
        wr_id: u64,
    ) -> (r: Result<u32, (CoreError, Tracked<rdma::WindowBindGrant>)>)
        requires
            core.well_formed(),
            old(self).qp.ready(),
            old(self).qp.state() is ReadyToSend,
            old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.source().id() == core.instance_id(),
            old(self).qp.source().node() == core.node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == core.node(),
            old(self).cq.infos().instance_id() == core.instance_id(),
            old(self).cq.in_flight().len() < old(self).cq.capacity(),
            !old(self).cq.in_flight().contains_key(wr_id),
            old(self).window_pool.well_formed(),
            old(self).window_pool.node() == core.node(),
            (mw_slot as int) < old(self).window_pool.slots.len(),
            old(self).window_pool.slots[mw_slot as int].generation == generation,
            old(self).window_pool.slots[mw_slot as int].state is Binding,
            old(self).window_pool.slots[mw_slot as int].window matches Some(w) && w.state()
                == rdma::WindowState::Unbound,
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).effective_host_mr_spec() matches Some(s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).effective_gpu_mr_spec() matches Some(s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            in_host ==> (old(self).effective_host_mr_spec() matches Some(s) && s.owns(frame)
                && s.mr.slot_ref((frame.0 - s.first_frame.0) as nat) == core.frames.frame(
                frame,
            ).slot),
            !in_host ==> (old(self).effective_gpu_mr_spec() matches Some(s) && s.owns(frame)
                && s.mr.slot_ref((frame.0 - s.first_frame.0) as nat) == core.frames.frame(
                frame,
            ).slot),
            frame.0 < core.frames.metas.len(),
            core.frames.frame(frame).slot.is_full(),
            core.frames.frame(frame).slot.well_formed(),
            core.frames.frame(frame).slot.slot_len == core.frames.tier_frame_size(
                core.frames.frame(frame).tier,
            ),
            core.frames.tier_frame_size(core.frames.frame(frame).tier) > 0,
            right.instance_id() == core.instance_id(),
            right.loc() == core.frames.frame(frame).slot.loc,
        ensures
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).bitmap == old(self).bitmap,
            final(self).in_flight_ops == old(self).in_flight_ops,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).completed_pulls == old(self).completed_pulls,
            final(self).window_pool.well_formed(),
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).window_pool.node() == old(self).window_pool.node(),
            final(self).window_pool.len() == old(self).window_pool.len(),
            forall|j: int|
                0 <= j < final(self).window_pool.slots.len() && j != mw_slot as int
                    ==> #[trigger] final(self).window_pool.slots[j] == old(
                    self,
                ).window_pool.slots[j],
            match r {
                Err((_, returned)) => {
                    &&& returned@ == rdma::WindowBindGrant::ReadOnly(right)
                    &&& final(self).window_pool.slots@ == old(self).window_pool.slots@
                    &&& final(self).window_pool.free@ == old(self).window_pool.free@
                    &&& final(self).cq == old(self).cq
                },
                Ok(next_rkey) => {
                    let old_w = old(self).window_pool.slots[mw_slot as int].window->Some_0;
                    let slot = core.frames.frame(frame).slot;
                    &&& final(self).cq.same_queue(old(self).cq)
                    &&& final(self).cq.well_formed()
                    &&& final(self).window_pool.slots[mw_slot as int].state == old(
                        self,
                    ).window_pool.slots[mw_slot as int].state
                    &&& final(self).window_pool.slots[mw_slot as int].generation == generation
                    &&& final(self).window_pool.slots[mw_slot as int].window matches Some(w)
                    &&& w.consistent()
                    &&& w.id() == old_w.id()
                    &&& w.pd_id() == old(self).window_pool.pd()
                    &&& w.node() == old(self).window_pool.node()
                    &&& w.state() is Binding
                    &&& w.state()->Binding_qp_id == old(self).qp.id()
                    &&& w.state()->Binding_share == Some(right.share())
                    &&& w.state()->Binding_instance_id == core.instance_id()
                    &&& w.state()->Binding_wr_id == wr_id
                    &&& w.state()->Binding_target_slot == slot
                    &&& w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                    &&& w.state()->Binding_next_rkey == next_rkey
                    &&& final(self).cq.in_flight() == old(self).cq.in_flight().insert(
                        wr_id,
                        rdma::PendingWorkRequest {
                            kind: rdma::PendingKind::BindWindow {
                                qp_id: old(self).qp.id(),
                                share: Some(right.share()),
                                window_id: old_w.id(),
                                target_slot: slot,
                                addr: w.state()->Binding_addr,
                                next_rkey,
                                access: rdma::RemoteAccess::ReadOnly,
                            },
                        },
                    )
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        if in_host {
            let slab = match (&self.host_mr, self.shared_host_mr) {
                (Some(s), _) => s,
                (None, Some(s)) => s,
                (None, None) => {
                    assert(false);
                    return Err(
                        (
                            CoreError::UnsupportedTier,
                            Tracked(rdma::WindowBindGrant::ReadOnly(right)),
                        ),
                    );
                },
            };
            let slice = slab.mr.slot_slice((frame.0 - slab.first_frame.0) as usize);
            self.window_pool.post_bind(
                mw_slot,
                generation,
                &self.qp,
                &mut self.cq,
                &slice,
                rdma::RemoteAccess::ReadOnly,
                Tracked(rdma::WindowBindGrant::ReadOnly(right)),
                wr_id,
            )
        } else {
            let slab = match (&self.gpu_mr, self.shared_gpu_mr) {
                (Some(s), _) => s,
                (None, Some(s)) => s,
                (None, None) => {
                    assert(false);
                    return Err(
                        (
                            CoreError::UnsupportedTier,
                            Tracked(rdma::WindowBindGrant::ReadOnly(right)),
                        ),
                    );
                },
            };
            let slice = slab.mr.slot_slice((frame.0 - slab.first_frame.0) as usize);
            self.window_pool.post_bind(
                mw_slot,
                generation,
                &self.qp,
                &mut self.cq,
                &slice,
                rdma::RemoteAccess::ReadOnly,
                Tracked(rdma::WindowBindGrant::ReadOnly(right)),
                wr_id,
            )
        }
    }

    #[verifier::rlimit(30)]
    fn post_bind_pinned_lease(
        &mut self,
        core: &mut CacheCore,
        frame: FrameId,
        in_host: bool,
        mw_slot: u32,
        generation: u32,
        Tracked(share_cap): Tracked<ShareCap>,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(before_core): Ghost<CacheCore>,
        Ghost(before_pool): Ghost<WindowPool<'a>>,
    ) -> (r: BackendResult<(RemoteLeaseHandle, u32, u32, u64)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            before_pool.well_formed(),
            old(self).timeline_id < old(core).timelines.count,
            old(core).timelines.kinds[old(self).timeline_id as int] == (
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
            old(self).cq.in_flight().len() < old(self).cq.capacity(),
            old(self).next_control_wr_id < CONTROL_WR_ID_LIMIT,
            old(self).window_pool.well_formed(),
            old(self).window_pool.pd() == before_pool.pd(),
            old(self).window_pool.node() == old(core).node(),
            old(self).window_pool.slots.len() == before_pool.slots.len(),
            (mw_slot as int) < old(self).window_pool.slots.len(),
            forall|j: int|
                0 <= j < old(self).window_pool.slots.len() && j != mw_slot as int ==> (
                #[trigger] old(self).window_pool.slots[j]) == before_pool.slots[j],
            old(self).window_pool.slots[mw_slot as int].generation == generation,
            old(self).window_pool.slots[mw_slot as int].state == MwSlotState::Binding(
                frame.0 as u64,
            ),
            old(self).window_pool.slots[mw_slot as int].window matches Some(w) && w.state()
                == rdma::WindowState::Unbound,
            forall|i: int|
                0 <= i < old(self).active_leases@.len() ==> (#[trigger] old(
                    self,
                ).active_leases@[i]).mw_slot != mw_slot,
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
            in_host ==> (old(self).effective_host_mr_spec() matches Some(s) && s.owns(frame)
                && s.mr.slot_ref((frame.0 - s.first_frame.0) as nat) == old(core).frames.frame(
                frame,
            ).slot),
            !in_host ==> (old(self).effective_gpu_mr_spec() matches Some(s) && s.owns(frame)
                && s.mr.slot_ref((frame.0 - s.first_frame.0) as nat) == old(core).frames.frame(
                frame,
            ).slot),
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &old(core).timelines,
                old(self).timeline_id,
                &old(core).ledger@,
                rdma::CqSnapshot::of(old(self).cq),
                old(core).instance_id(),
                old(self).next_control_wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &old(core).timelines,
                old(self).timeline_id,
                old(self).cq.observer(),
                &old(core).ledger@,
            ),
            rdma_leases_valid(old(self).active_leases@, &before_pool, &old(self).qp, &before_core),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &before_pool,
                rdma::CqSnapshot::of(old(self).cq),
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &before_pool),
            instance.id() == old(core).instance_id(),
            old(core).node_id == before_core.node_id,
            old(core).instance_id() == before_core.instance_id(),
            old(core).tiers == before_core.tiers,
            old(core).spec == before_core.spec,
            old(core).version_registry == before_core.version_registry,
            old(core).next_version_seq == before_core.next_version_seq,
            old(core).version_seq_limit == before_core.version_seq_limit,
            old(core).active_leases == before_core.active_leases,
            old(core).timelines == before_core.timelines,
            old(core).ledger == before_core.ledger,
            old(core).frames.same_directory(&before_core.frames),
            old(core).frames.preserves_initializing(&before_core.frames, None),
            old(core).frames.same_other_pins(&before_core.frames, Some(frame)),
            frame.0 < old(core).frames.metas.len(),
            old(core).frames.meta(frame).state == STATE_PUBLISHED,
            !before_core.frames.frame(frame).remote_shares.contains_key(share_cap.share()),
            old(core).frames.frame(frame).remote_shares == before_core.frames.frame(
                frame,
            ).remote_shares.insert(
                share_cap.share(),
                crate::frame::RemoteShareEntry { mw_slot, mw_gen: generation },
            ),
            old(core).frames.frame(frame).matches_lease(share_cap.share(), mw_slot, generation),
            share_cap.is_unbound(),
            share_cap.instance_id() == old(core).instance_id(),
            share_cap.loc() == old(core).frames.frame(frame).slot.loc,
            share_cap.frontier().readers.is_empty(),
            old(core).frames.published(frame) matches Some((_, _, c, _))
                && share_cap.frontier().last_writer == Some(c.writer),
            old(core).frames.returned_reads(
                frame,
                share_cap,
                Ticket::NONE,
                old(core).kernel_view(),
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).frames.same_directory(&old(core).frames),
            final(core).frames.preserves_initializing(&before_core.frames, None),
            peer_foreign_shares_preserved(
                &before_core,
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(core).timelines.only_watermarks(&old(core).timelines) || final(core).timelines
                == old(core).timelines,
            final(core).ledger == old(core).ledger,
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            r.is_err() ==> final(self).active_leases == old(self).active_leases,
            r matches Ok((handle, slot_out, gen_out, wr_id)) ==> {
                let idx = (final(self).active_leases@.len() - 1) as int;
                &&& handle.mw_slot == mw_slot as usize
                &&& handle.generation == generation
                &&& slot_out == mw_slot
                &&& gen_out == generation
                &&& final(self).active_leases@.len() == old(self).active_leases@.len() + 1
                &&& final(self).active_leases@.take(old(self).active_leases@.len() as int) == old(
                    self,
                ).active_leases@
                &&& final(self).lease_index_spec(handle) == Some(idx)
                &&& final(self).active_leases@[idx].frame == frame
                &&& final(self).active_leases@[idx].mw_slot == mw_slot
                &&& final(self).active_leases@[idx].generation == generation
                &&& final(self).active_leases@[idx].wr_id == wr_id
                &&& final(self).active_leases@[idx].state == RemoteLeaseState::Binding
                &&& (final(self).active_leases@[idx].share_cap matches Some(sc)
                    && final(core).frames.published(frame) matches Some((_, _, c, _))
                    && sc@.frontier().last_writer == Some(c.writer))
                &&& final(self).cq.in_flight().contains_key(wr_id)
                &&& final(self).cq.in_flight()[wr_id].kind is BindWindow
                &&& (final(self).lease_matches_window_state(
                    handle,
                    RemoteLeaseState::Binding,
                ) matches Some(w) ==> final(self).cq.in_flight()[wr_id].kind->BindWindow_window_id
                    == w.id())
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        let ghost after_pin_core = *core;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let ghost before_leases = self.active_leases@;
        let tracked mut share_cap = share_cap;
        let ghost share = share_cap.share();
        let tracked right = share_cap.take_window_bind_right();
        let wr_id = self.next_control_wr_id;
        proof {
            lemma_core_wf_facts(core);
            lemma_published_frame_slot_valid(core, frame);
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
        let bind_res = self.post_bind_pinned_slice(
            core,
            frame,
            in_host,
            mw_slot,
            generation,
            Tracked(right),
            wr_id,
        );
        if let Err((e, returned_grant_trk)) = bind_res {
            self.rollback_bind_post(
                core,
                frame,
                mw_slot,
                generation,
                Tracked(share_cap),
                returned_grant_trk,
                Tracked(instance),
                Ghost(before_core),
                Ghost(after_pin_core),
                Ghost(before_pool),
                Ghost(before_leases),
                Ghost(share),
            );
            proof {
                reveal(FrameTable::same_directory);
            }
            return Err(e);
        }
        let res = self.commit_bound_lease(
            core,
            frame,
            mw_slot,
            generation,
            wr_id,
            Tracked(share_cap),
            Ghost(before_cq),
            Ghost(before_core),
            Ghost(before_pool),
        );
        proof {
            reveal(FrameTable::same_directory);
            assert(rdma_data_events(rdma::CqSnapshot::of(self.cq)) =~= rdma_data_events(before_cq));
        }
        Ok(res)
    }

    /// Complete a pending MW bind after observing `Completed::WindowBound`,
    /// returning the `RemoteCapability` and `ShareCap` for remote transfer.
    pub fn complete_bind(
        &mut self,
        core: &CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(witness): Tracked<rdma::BindWitness>,
    ) -> (r: BackendResult<(rdma::RemoteCapability, Tracked<ShareCap>)>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).lease_matches_window_state(handle, RemoteLeaseState::Binding) matches Some(w)
                ==> witness.matches(w),
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
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((cap, share_cap)) => old(self).lease_index_spec(handle) matches Some(idx) && {
                    let lease = old(self).active_leases@[idx];
                    &&& lease.state == RemoteLeaseState::Binding
                    &&& cap.well_formed()
                    &&& cap.slot_ref() == core.frames.frame(lease.frame).slot
                    &&& cap.access() == rdma::RemoteAccess::ReadOnly
                    &&& cap.share() == Some(lease.share@)
                    &&& share_cap@.is_rdma_bound()
                    &&& share_cap@.instance_id() == core.instance_id()
                    &&& share_cap@.loc() == core.frames.frame(lease.frame).slot.loc
                    &&& share_cap@.share() == lease.share@
                    &&& share_cap@.frontier().readers.is_empty()
                    &&& (lease.share_cap matches Some(sc) && share_cap@ == sc@)
                    &&& final(self).lease_index_spec(handle) == Some(idx)
                    &&& final(self).active_leases@.len() == old(self).active_leases@.len()
                    &&& final(self).active_leases@[idx].frame == lease.frame
                    &&& final(self).active_leases@[idx].share == lease.share
                    &&& final(self).active_leases@[idx].state == RemoteLeaseState::Bound
                    &&& forall|j: int|
                        0 <= j < old(self).active_leases@.len() && j != idx ==> (
                        #[trigger] final(self).active_leases@[j]) == old(self).active_leases@[j]
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        proof {
            reveal(rdma_leases_valid);
        }
        let lease_idx = match self.find_lease_index(handle) {
            Some(i) => i,
            None => {
                proof {
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::preserves_initializing);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Binding) {
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
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
        let cap = match self.window_pool.complete_bind(mw_slot, generation, Tracked(witness)) {
            Ok(c) => c,
            Err(e) => {
                proof {
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        let share_cap = self.active_leases[lease_idx].share_cap.take().unwrap();
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        self.active_leases[lease_idx].state = RemoteLeaseState::Bound;
        proof {
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
            lemma_control_leases_match_cq_after_cq_remove_update(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                before_cq,
                before_cq,
                self.next_control_wr_id,
                None,
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
        Ok((cap, share_cap))
    }

    /// Complete a pending MW bind whose `WindowBound` CQE was already reaped and
    /// staged into `BindReady` by `pump_cq`.
    pub fn complete_ready_bind(&mut self, core: &CacheCore, handle: RemoteLeaseHandle) -> (r:
        BackendResult<(rdma::RemoteCapability, Tracked<ShareCap>)>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
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
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((cap, share_cap)) => old(self).lease_index_spec(handle) matches Some(idx) && {
                    let lease = old(self).active_leases@[idx];
                    &&& lease.state == RemoteLeaseState::BindReady
                    &&& cap.well_formed()
                    &&& cap.slot_ref() == core.frames.frame(lease.frame).slot
                    &&& cap.access() == rdma::RemoteAccess::ReadOnly
                    &&& cap.share() == Some(lease.share@)
                    &&& share_cap@.is_rdma_bound()
                    &&& share_cap@.instance_id() == core.instance_id()
                    &&& share_cap@.loc() == core.frames.frame(lease.frame).slot.loc
                    &&& share_cap@.share() == lease.share@
                    &&& share_cap@.frontier().readers.is_empty()
                    &&& (lease.share_cap matches Some(sc) && share_cap@ == sc@)
                    &&& final(self).lease_index_spec(handle) == Some(idx)
                    &&& final(self).active_leases@.len() == old(self).active_leases@.len()
                    &&& final(self).active_leases@[idx].frame == lease.frame
                    &&& final(self).active_leases@[idx].share == lease.share
                    &&& final(self).active_leases@[idx].state == RemoteLeaseState::Bound
                    &&& forall|j: int|
                        0 <= j < old(self).active_leases@.len() && j != idx ==> (
                        #[trigger] final(self).active_leases@[j]) == old(self).active_leases@[j]
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
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
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::BindReady) {
            proof {
                lemma_foreign_shares_refl(core, self.active_leases@);
            }
            return Err(CoreError::WouldConflict);
        }
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        proof {
            lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, lease_idx as int);
        }
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        let Tracked(witness) = self.active_leases[lease_idx].bind_witness.take().unwrap();
        let cap = match self.window_pool.complete_bind(mw_slot, generation, Tracked(witness)) {
            Ok(c) => c,
            Err(_) => {
                assert(false);
                return Err(CoreError::WouldConflict);
            },
        };
        let share_cap = self.active_leases[lease_idx].share_cap.take().unwrap();
        self.active_leases[lease_idx].state = RemoteLeaseState::Bound;
        proof {
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
            lemma_control_leases_match_cq_after_cq_remove_update(
                before_leases,
                self.active_leases@,
                lease_idx as int,
                &before_pool,
                &self.window_pool,
                before_cq,
                before_cq,
                self.next_control_wr_id,
                None,
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
        Ok((cap, share_cap))
    }

    /// Recover from `Completed::BindFailed` (M3!): extracts `WindowBindGrant`
    /// from the failed MW, restores `WindowBindRight` into the retained
    /// `ShareCap`, unpins the remote lease in `core`, and frees the MW slot.
    pub fn fail_bind(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(witness): Tracked<rdma::BindFailedWitness>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).lease_matches_window_state(handle, RemoteLeaseState::Binding) matches Some(w)
                ==> witness.matches(w),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).frames.same_directory(&old(core).frames),
            final(core).frames.preserves_initializing(&old(core).frames, None),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(core).timelines.only_watermarks(&old(core).timelines) || final(core).timelines
                == old(core).timelines,
            final(core).ledger == old(core).ledger,
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).cq == old(self).cq,
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            match r {
                Err(_) => *final(core) == *old(core) && *final(self) == *old(self),
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
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::preserves_initializing);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::Binding) {
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
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
        let lease = self.active_leases.swap_remove(lease_idx);
        let Tracked(grant) = self.window_pool.fail_bind(
            lease.mw_slot,
            lease.generation,
            Tracked(witness),
        );
        let Tracked(mut share_cap) = lease.share_cap.unwrap();
        proof {
            let tracked right = match grant {
                rdma::WindowBindGrant::ReadOnly(r) => r,
                rdma::WindowBindGrant::ReadWrite => {
                    assert(false);
                    proof_from_false()
                },
            };
            share_cap.restore_window_bind_right(right);
            assert(core.frames.returned_reads(
                lease.frame,
                share_cap,
                Ticket::NONE,
                core.kernel_view(),
            ));
        }
        let _unpin_res = core.unpin_remote_lease(
            lease.frame,
            lease.mw_slot,
            lease.generation,
            Tracked(share_cap),
            Ticket::NONE,
            Tracked(instance),
        );
        proof {
            assert(_unpin_res.is_ok());
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            lemma_rdma_window_after_watermarks(
                self.in_flight_ops,
                self.bitmap,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
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
                Some(lease.share@),
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
            reveal(FrameTable::same_directory);
            lemma_swap_remove_foreign_shares_after_unpin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
        }
        Ok(())
    }

    /// Clean up a failed MW bind whose `BindFailed` CQE was already reaped and
    /// staged into `BindFailedReady` by `pump_cq`, returning the CQE error status.
    pub fn complete_failed_bind(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<i32>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).frames.same_directory(&old(core).frames),
            final(core).frames.preserves_initializing(&old(core).frames, None),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(core).timelines.only_watermarks(&old(core).timelines) || final(core).timelines
                == old(core).timelines,
            final(core).ledger == old(core).ledger,
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).cq == old(self).cq,
            final(self).pending() == old(self).pending(),
            final(self).completed_pulls == old(self).completed_pulls,
            match r {
                Err(_) => *final(core) == *old(core) && *final(self) == *old(self),
                Ok(_) => {
                    &&& old(self).lease_index_spec(handle) is Some
                    &&& old(self).active_leases@[old(self).lease_index_spec(handle)->Some_0].state
                        == RemoteLeaseState::BindFailedReady
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
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::preserves_initializing);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(CoreError::InvalidArgument);
            },
        };
        if !matches!(self.active_leases[lease_idx].state, RemoteLeaseState::BindFailedReady) {
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
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
        let err_code = self.active_leases[lease_idx].staged_error.unwrap();
        let lease = self.active_leases.swap_remove(lease_idx);
        let Tracked(witness) = lease.bind_failed_witness.unwrap();
        let Tracked(grant) = self.window_pool.fail_bind(
            lease.mw_slot,
            lease.generation,
            Tracked(witness),
        );
        let Tracked(mut share_cap) = lease.share_cap.unwrap();
        proof {
            let tracked right = match grant {
                rdma::WindowBindGrant::ReadOnly(r) => r,
                rdma::WindowBindGrant::ReadWrite => {
                    assert(false);
                    proof_from_false()
                },
            };
            share_cap.restore_window_bind_right(right);
            assert(core.frames.returned_reads(
                lease.frame,
                share_cap,
                Ticket::NONE,
                core.kernel_view(),
            ));
        }
        let _unpin_res = core.unpin_remote_lease(
            lease.frame,
            lease.mw_slot,
            lease.generation,
            Tracked(share_cap),
            Ticket::NONE,
            Tracked(instance),
        );
        proof {
            assert(_unpin_res.is_ok());
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            lemma_rdma_window_after_watermarks(
                self.in_flight_ops,
                self.bitmap,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
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
                Some(lease.share@),
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
            reveal(FrameTable::same_directory);
            lemma_swap_remove_foreign_shares_after_unpin(
                &before_core,
                core,
                before_leases,
                self.active_leases@,
                lease_idx as int,
            );
        }
        Ok(err_code)
    }

    /// Recover any uncompleted bind lease (`Binding`, `BindReady`, `BindFailedReady`,
    /// or `ControlFailedState` with `share_cap.is_some()`) once the QP has been reset
    /// (`IBV_QPS_RESET`), restoring `WindowBindRight` and unpinning `core` without
    /// advancing host revocation counters.
    pub fn recover_binding_on_qp_reset(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        Tracked(qp_reset): Tracked<&rdma::QpResetWitness>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<u32>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            qp_reset.matches_qp(old(self).qp.id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
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
            r.is_ok() ==> final(core).frames.same_directory(&old(core).frames),
            match r {
                Err(_) => *final(core) == *old(core) && final(self).window_pool.slots@ == old(
                    self,
                ).window_pool.slots@ && final(self).window_pool.free@ == old(self).window_pool.free@
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
                return Err(CoreError::InvalidArgument);
            },
        };
        let is_binding_like =
            matches!(
            self.active_leases[lease_idx].state,
            RemoteLeaseState::Binding | RemoteLeaseState::BindReady | RemoteLeaseState::BindFailedReady
        )
            || (matches!(self.active_leases[lease_idx].state, RemoteLeaseState::ControlFailedState)
            && self.active_leases[lease_idx].share_cap.is_some());
        if !is_binding_like {
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
        let mw_slot = self.active_leases[lease_idx].mw_slot;
        let generation = self.active_leases[lease_idx].generation;
        let frame = self.active_leases[lease_idx].frame;
        proof {
            let w = self.window_pool.slots[mw_slot as int].window->Some_0;
            assert(w.state() is Binding);
            assert(w.state().binding().qp_id == self.qp.id());
            assert(w.node() == core.node());
        }
        let grant_opt = match self.window_pool.dealloc_on_qp_reset(
            mw_slot,
            generation,
            Tracked(qp_reset),
            Tracked(None),
        ) {
            Ok(g) => g,
            Err((e, _)) => {
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
                        rdma::CqSnapshot::of(self.cq),
                        self.next_control_wr_id,
                    );
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(e);
            },
        };
        let mut lease = self.active_leases.swap_remove(lease_idx);
        let Tracked(mut share_cap) = lease.share_cap.take().unwrap();
        proof {
            let tracked grant = grant_opt.get().tracked_unwrap();
            let tracked right = match grant {
                rdma::WindowBindGrant::ReadOnly(r) => r,
                rdma::WindowBindGrant::ReadWrite => {
                    assert(false);
                    proof_from_false()
                },
            };
            share_cap.restore_window_bind_right(right);
            assert(core.frames.returned_reads(frame, share_cap, Ticket::NONE, core.kernel_view()));
        }
        let _unpin_res = core.unpin_remote_lease(
            frame,
            mw_slot,
            generation,
            Tracked(share_cap),
            Ticket::NONE,
            Tracked(instance),
        );
        proof {
            assert(_unpin_res.is_ok());
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            lemma_rdma_window_after_watermarks(
                self.in_flight_ops,
                self.bitmap,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
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
                Some(lease.share@),
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
}

/// Helper preserving `rdma_leases_valid` when a freshly checked-out MW slot is
/// immediately returned unbound on `pin_remote_lease` failure.
proof fn lemma_rdma_leases_after_pool_retire_replenish_back<'a>(
    leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
    mw_slot: u32,
)
    requires
        rdma_leases_valid(leases, before_pool, qp, core),
        forall|i: int| 0 <= i < leases.len() ==> (#[trigger] leases[i]).mw_slot != mw_slot,
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != mw_slot as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
    ensures
        rdma_leases_valid(leases, after_pool, qp, core),
{
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        after_pool,
        qp,
        core,
    ) by {
        assert(leases[i].valid(before_pool, qp, core));
        assert(after_pool.slots[leases[i].mw_slot as int]
            == before_pool.slots[leases[i].mw_slot as int]);
    }
}

} // verus!
