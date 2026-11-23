use super::*;
use transfer::context::{EventSourceToken, ReturnedShare, SlotToken, TransferInstance};
use transfer::host::HostContext;
use transfer::rdma;

verus! {

impl<'a> WindowPool<'a> {
    /// Post a local invalidate WR on a `Bound` MW slot.
    pub fn post_invalidate(
        &mut self,
        i: u32,
        generation: u32,
        qp: &rdma::QueuePair<'a>,
        cq: &mut rdma::CompletionQueue<'a>,
        wr_id: u64,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Bound,
            old(self).slots[i as int].window matches Some(w) && w.state() is Bound
                && w.state()->Bound_qp_id == qp.id(),
            qp.can_post(*old(cq), wr_id),
            qp.pd_id() == old(self).pd(),
            qp.source().node() == old(self).node(),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            r.is_err() ==> final(self).slots@ == old(self).slots@ && final(self).free@ == old(
                self,
            ).free@ && *final(cq) == *old(cq),
            r.is_ok() ==> {
                let old_w = old(self).slots[i as int].window->Some_0;
                &&& final(cq).well_formed()
                &&& final(cq).same_queue(*old(cq))
                &&& final(self).slots[i as int].state == MwSlotState::Invalidating(
                    old(self).slots[i as int].state->Bound_0,
                )
                &&& final(self).slots[i as int].generation == generation
                &&& final(self).slots[i as int].window matches Some(w)
                &&& w.id() == old_w.id()
                &&& w.state() == (rdma::WindowState::Invalidating {
                    wr_id,
                    instance_id: old_w.state()->Bound_instance_id,
                    qp_id: qp.id(),
                    share: old_w.state()->Bound_share,
                    slot: old_w.state()->Bound_slot,
                    access: old_w.state()->Bound_access,
                    prev_rkey: old_w.state()->Bound_rkey,
                })
                &&& final(cq).in_flight() == old(cq).in_flight().insert(
                    wr_id,
                    rdma::PendingWorkRequest {
                        kind: rdma::PendingKind::LocalInvalidate {
                            qp_id: qp.id(),
                            window_id: old_w.id(),
                        },
                    },
                )
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let lease_tag = match self.slots[idx].state {
            MwSlotState::Bound(tag) => tag,
            _ => 0,
        };
        let mut mw = self.slots[idx].window.take().unwrap();
        match rdma::post_local_invalidate(qp, cq, &mut mw, wr_id) {
            Ok(()) => {
                self.slots[idx].window = Some(mw);
                self.slots[idx].state = MwSlotState::Invalidating(lease_tag);
                Ok(())
            },
            Err(e) => {
                self.slots[idx].window = Some(mw);
                proof {
                    assert(self.slots@ =~= old(self).slots@);
                }
                Err(rdma_error(e))
            },
        }
    }

    /// Complete a cooperative local invalidation and return the unbound MW slot to free.
    pub fn finish_invalidate(
        &mut self,
        i: u32,
        generation: u32,
        Tracked(witness): Tracked<rdma::InvalidateWitness>,
        Tracked(returned): Tracked<Option<ReturnedShare>>,
    )
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Invalidating,
            old(self).slots[i as int].window matches Some(w) && w.state() is Invalidating
                && witness.matches(w) && (w.state()->Invalidating_share matches Some(s)
                ==> returned matches Some(r) && r.share() == s && r.instance_id()
                == w.state()->Invalidating_instance_id && r.loc()
                == w.state()->Invalidating_slot.loc),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            final(self).slots[i as int].state == MwSlotState::Free,
            final(self).slots[i as int].generation == generation,
            final(self).slots[i as int].window matches Some(w) && w.state()
                == rdma::WindowState::Unbound,
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let mut mw = self.slots[idx].window.take().unwrap();
        mw.finish_invalidate(Tracked(witness), Tracked(returned));
        self.slots[idx].window = Some(mw);
        self.slots[idx].state = MwSlotState::Free;
        self.free.push(i);
        proof {
            broadcast use vstd::seq_lib::lemma_seq_contains_after_push;

        }
    }

    /// Unilaterally fence and deallocate an `Invalidating` MW slot via
    /// `InvalidateWitness` (Option-A IBTA-strict revocation -> `Quarantined`).
    pub fn revoke_with_invalidate(
        &mut self,
        i: u32,
        generation: u32,
        host: &mut HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(witness): Tracked<rdma::InvalidateWitness>,
        Tracked(lent): Tracked<SlotToken>,
    ) -> (r: Result<
        rdma::Revocation,
        (CoreError, Tracked<rdma::InvalidateWitness>, Tracked<SlotToken>),
    >)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Invalidating,
            old(self).slots[i as int].window matches Some(w) && w.state() is Invalidating
                && w.can_revoke(*old(host), instance.id(), lent) && witness.matches(w),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            match r {
                Err((_, ret_w, ret_lent)) => {
                    &&& ret_w@ == witness
                    &&& ret_lent@ == lent
                    &&& final(self).slots@ == old(self).slots@
                    &&& final(self).free@ == old(self).free@
                    &&& *final(host) == *old(host)
                },
                Ok(result) => {
                    &&& final(host).stepped(*old(host))
                    &&& final(self).slots[i as int].state == MwSlotState::Quarantined(
                        old(self).slots[i as int].state->Invalidating_0,
                    )
                    &&& final(self).slots[i as int].generation == generation
                    &&& final(self).slots[i as int].window is None
                    &&& rdma::revocation_result(
                        result,
                        instance.id(),
                        lent,
                        old(self).slots[i as int].window->Some_0.state().binding().share->Some_0,
                        old(host).source@.next_event(),
                    )
                },
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let lease_tag = match self.slots[idx].state {
            MwSlotState::Invalidating(tag) => tag,
            _ => 0,
        };
        let mw = self.slots[idx].window.take().unwrap();
        match rdma::revoke_read_share_with_window(
            host,
            Tracked(instance),
            mw,
            Tracked(witness),
            Tracked(lent),
        ) {
            Ok(res) => {
                self.slots[idx].state = MwSlotState::Quarantined(lease_tag);
                Ok(res)
            },
            Err((e, ret_mw, ret_w, ret_lent)) => {
                self.slots[idx].window = Some(ret_mw);
                proof {
                    assert(self.slots@ =~= old(self).slots@);
                }
                Err((rdma_error(e), ret_w, ret_lent))
            },
        }
    }

    /// Unilaterally fence and deallocate a `Binding`/`Bound`/`Invalidating` MW
    /// slot once its bound QP has been reset (`IBV_QPS_RESET`), leaving the slot `Vacant`.
    pub fn revoke_on_qp_reset(
        &mut self,
        i: u32,
        generation: u32,
        host: &mut HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(qp_reset): Tracked<&rdma::QpResetWitness>,
        Tracked(lent): Tracked<SlotToken>,
    ) -> (r: Result<rdma::Revocation, (CoreError, Tracked<SlotToken>)>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            !(old(self).slots[i as int].state is Free) && !(old(
                self,
            ).slots[i as int].state is Vacant) && !(old(self).slots[i as int].state is Quarantined),
            old(self).slots[i as int].window matches Some(w) && w.can_revoke(
                *old(host),
                instance.id(),
                lent,
            ) && qp_reset.matches_qp(w.state().binding().qp_id, w.node()),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            match r {
                Err((_, tok)) => {
                    &&& tok@ == lent
                    &&& final(self).slots@ == old(self).slots@
                    &&& final(self).free@ == old(self).free@
                    &&& *final(host) == *old(host)
                },
                Ok(result) => {
                    &&& final(host).stepped(*old(host))
                    &&& final(self).slots[i as int].state == MwSlotState::Vacant
                    &&& final(self).slots[i as int].generation == generation
                    &&& final(self).slots[i as int].window is None
                    &&& rdma::revocation_result(
                        result,
                        instance.id(),
                        lent,
                        old(self).slots[i as int].window->Some_0.state().binding().share->Some_0,
                        old(host).source@.next_event(),
                    )
                },
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let mw = self.slots[idx].window.take().unwrap();
        match rdma::revoke_read_share_on_reset_qp(
            host,
            Tracked(instance),
            mw,
            Tracked(qp_reset),
            Tracked(lent),
        ) {
            Ok(result) => {
                self.slots[idx].state = MwSlotState::Vacant;
                Ok(result)
            },
            Err((e, ret_mw, tok)) => {
                self.slots[idx].window = Some(ret_mw);
                proof {
                    assert(self.slots@ =~= old(self).slots@);
                }
                Err((rdma_error(e), tok))
            },
        }
    }

    /// Deallocate a `Binding` or cooperative `Invalidating` MW slot once its
    /// bound QP has been reset (`IBV_QPS_RESET`), leaving the slot `Vacant`.
    pub fn dealloc_on_qp_reset(
        &mut self,
        i: u32,
        generation: u32,
        Tracked(qp_reset): Tracked<&rdma::QpResetWitness>,
        Tracked(returned): Tracked<Option<ReturnedShare>>,
    ) -> (r: Result<
        Tracked<Option<rdma::WindowBindGrant>>,
        (CoreError, Tracked<Option<ReturnedShare>>),
    >)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Binding || old(
                self,
            ).slots[i as int].state is Invalidating,
            old(self).slots[i as int].window matches Some(w) && w.consistent() && (
            w.state() is Binding || w.state() is Invalidating) && qp_reset.matches_qp(
                w.state().binding().qp_id,
                w.node(),
            ) && (w.state() is Invalidating ==> (w.state()->Invalidating_share matches Some(s)
                ==> returned matches Some(r) && r.share() == s && r.instance_id()
                == w.state()->Invalidating_instance_id && r.loc()
                == w.state()->Invalidating_slot.loc)),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            match r {
                Err((_, ret_sh)) => {
                    &&& ret_sh@ == returned
                    &&& final(self).slots@ == old(self).slots@
                    &&& final(self).free@ == old(self).free@
                },
                Ok(grant_opt) => {
                    &&& final(self).slots[i as int].state == MwSlotState::Vacant
                    &&& final(self).slots[i as int].generation == generation
                    &&& final(self).slots[i as int].window is None
                    &&& match old(self).slots[i as int].window->Some_0.state() {
                        rdma::WindowState::Binding {
                            instance_id,
                            share,
                            target_slot,
                            access,
                            ..
                        } => {
                            &&& grant_opt@ matches Some(g)
                            &&& g.access() == access
                            &&& g.share() == share
                            &&& (g matches rdma::WindowBindGrant::ReadOnly(right)
                                ==> right.instance_id() == instance_id && right.loc()
                                == target_slot.loc)
                        },
                        _ => grant_opt@ is None,
                    }
                },
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let mw = self.slots[idx].window.take().unwrap();
        match rdma::dealloc_window_on_reset_qp(mw, Tracked(qp_reset), Tracked(returned)) {
            Ok(grant_opt) => {
                self.slots[idx].state = MwSlotState::Vacant;
                Ok(grant_opt)
            },
            Err((e, ret_mw, ret_sh)) => {
                self.slots[idx].window = Some(ret_mw);
                proof {
                    assert(self.slots@ =~= old(self).slots@);
                }
                Err((rdma_error(e), ret_sh))
            },
        }
    }

    /// Replenish a `Vacant` slot with a freshly allocated Type-2 `MemoryWindow`.
    pub fn replenish(
        &mut self,
        i: u32,
        pd: &'a rdma::ProtectionDomain<'a>,
        Tracked(name): Tracked<EventSourceToken>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].state == MwSlotState::Vacant,
            pd.id() == old(self).pd(),
            pd.node() == old(self).node(),
            name.well_formed(),
            name.source() is RdmaQueuePair,
            name.node() == pd.node(),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> final(self).slots[i as int].state == MwSlotState::Free
                && final(self).slots[i as int].generation == old(self).slots[i as int].generation
                + 1,
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let idx = i as usize;
        if self.slots[idx].generation == u32::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let w = match rdma::alloc_window(pd, Tracked(name)) {
            Ok(w) => w,
            Err(e) => return Err(rdma_error(e)),
        };
        let slot = &mut self.slots[idx];
        slot.window = Some(w);
        slot.generation += 1;
        slot.state = MwSlotState::Free;
        self.free.push(i);
        proof {
            broadcast use vstd::seq_lib::lemma_seq_contains_after_push;

        }
        Ok(())
    }

    /// Deallocate all `Free` windows and clear `Quarantined` slots during teardown
    /// (preserves any window whose `dealloc_window` call fails, leaving the pool well-formed).
    pub fn dealloc_all(&mut self) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            forall|j: int|
                0 <= j < old(self).slots.len() ==> {
                    let s = #[trigger] old(self).slots[j];
                    &&& (s.state == MwSlotState::Free || s.state is Quarantined || s.state
                        == MwSlotState::Vacant)
                    &&& match s.window {
                        None => true,
                        Some(w) => w.state() == rdma::WindowState::Unbound,
                    }
                },
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            r.is_ok() ==> final(self).free@.len() == 0 && forall|j: int|
                0 <= j < final(self).slots.len() ==> (#[trigger] final(self).slots[j]).state
                    == MwSlotState::Vacant,
            forall|j: int|
                0 <= j < final(self).slots.len() ==> !((
                #[trigger] final(self).slots[j]).state is Binding
                    || final(self).slots[j].state is Bound
                    || final(self).slots[j].state is Invalidating),
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let mut first_err: Option<CoreError> = None;
        let mut idx: usize = 0;
        self.free.clear();
        while idx < self.slots.len()
            invariant
                idx <= self.slots.len(),
                self.slots.len() == old(self).slots.len(),
                self.slots.len() <= u32::MAX,
                self.pd_id == old(self).pd_id,
                self.node == old(self).node,
                self.free@.no_duplicates(),
                first_err.is_none() ==> self.free@.len() == 0,
                forall|j: int|
                    0 <= j < self.free.len() ==> {
                        let k = #[trigger] self.free[j];
                        (k as int) < idx && self.slots[k as int].state == MwSlotState::Free
                    },
                forall|j: int|
                    0 <= j < idx ==> {
                        let s = #[trigger] self.slots[j];
                        &&& (s.state == MwSlotState::Vacant || s.state is Quarantined)
                            == s.window.is_none()
                        &&& s.state == MwSlotState::Free ==> (s.window matches Some(w) && w.state()
                            == rdma::WindowState::Unbound && self.free@.contains(j as u32))
                        &&& s.window matches Some(w) ==> (w.pd_id() == self.pd_id@ && w.node()
                            == self.node@) && (w.state() is Binding || w.state() is Bound
                            || w.state() is Invalidating ==> w.consistent())
                        &&& (s.state == MwSlotState::Free || s.state is Quarantined || s.state
                            == MwSlotState::Vacant)
                        &&& first_err.is_none() ==> s.state == MwSlotState::Vacant
                    },
                forall|j: int|
                    idx <= j < self.slots.len() ==> {
                        let s = #[trigger] self.slots[j];
                        &&& s == old(self).slots[j]
                        &&& (s.state == MwSlotState::Free || s.state is Quarantined || s.state
                            == MwSlotState::Vacant)
                        &&& (s.window matches Some(w) ==> w.state() == rdma::WindowState::Unbound
                            && w.pd_id() == self.pd_id@ && w.node() == self.node@)
                        &&& (s.state == MwSlotState::Vacant || s.state is Quarantined)
                            == s.window.is_none()
                        &&& (s.state == MwSlotState::Free ==> (s.window matches Some(w) && w.state()
                            == rdma::WindowState::Unbound))
                    },
            decreases self.slots.len() - idx,
        {
            if self.slots[idx].window.is_some() {
                let w = self.slots[idx].window.take().unwrap();
                match rdma::dealloc_window(w) {
                    Ok(()) => {
                        self.slots[idx].state = MwSlotState::Vacant;
                    },
                    Err((e, w)) => {
                        self.slots[idx].window = Some(w);
                        if first_err.is_none() {
                            first_err = Some(rdma_error(e));
                        }
                        if matches!(self.slots[idx].state, MwSlotState::Free) {
                            self.free.push(idx as u32);
                            proof {
                                broadcast use vstd::seq_lib::lemma_seq_contains_after_push;

                            }
                        }
                    },
                }
            } else {
                self.slots[idx].state = MwSlotState::Vacant;
            }
            idx += 1;
        }
        match first_err {
            None => Ok(()),
            Some(e) => Err(e),
        }
    }
}

} // verus!
