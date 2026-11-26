use super::*;
use transfer::context::{
    EventSourceToken, ReturnedShare, SlotToken, SyncContext, TransferInstance,
};
use transfer::rdma;

verus! {

#[derive(Clone, Copy)]
pub struct WindowInvalidation {
    pub slot: u32,
    pub generation: u32,
    pub wr_id: u64,
}

impl<'a> WindowPool<'a> {
    /// Post local invalidations for distinct bound pool slots. Only the
    /// accepted prefix changes state. Every window stays owned by the pool.
    pub fn post_invalidate_batch(
        &mut self,
        qp: &rdma::QueuePair<'a>,
        cq: &mut rdma::CompletionQueue<'a>,
        entries: &[WindowInvalidation],
    ) -> (r: (usize, Option<CoreError>))
        requires
            old(self).well_formed(),
            old(cq).well_formed(),
            qp.ready(),
            qp.cq_id() == old(cq).id(),
            qp.source().node() == old(cq).observer(),
            qp.source().id() == old(cq).infos().instance_id(),
            qp.pd_id() == old(self).pd(),
            qp.source().node() == old(self).node(),
            old(cq).in_flight().len() + entries.len() <= old(cq).capacity(),
            forall|i: int|
                0 <= i < entries.len() ==> {
                    let e = #[trigger] entries@[i];
                    &&& e.slot < old(self).slots_spec().len()
                    &&& !old(cq).in_flight().contains_key(e.wr_id)
                    &&& old(self).slots_spec()[e.slot as int].generation_spec() == e.generation
                    &&& old(self).slots_spec()[e.slot as int].state_spec() is Bound
                    &&& old(self).slot_window(e.slot as int) matches Some(w) && w.state() is Bound
                        && w.state()->Bound_qp_id == qp.id()
                },
            forall|i: int, j: int|
                0 <= i < j < entries.len() ==> {
                    let a = #[trigger] entries@[i];
                    let b = #[trigger] entries@[j];
                    &&& a.slot != b.slot && a.wr_id != b.wr_id
                },
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).free_spec() == old(self).free_spec(),
            final(self).len() == old(self).len(),
            final(cq).well_formed(),
            final(cq).same_queue(*old(cq)),
            final(cq).in_flight().len() == old(cq).in_flight().len() + r.0,
            old(cq).in_flight().submap_of(final(cq).in_flight()),
            r.0 <= entries.len(),
            r.1.is_none() ==> r.0 == entries.len(),
            forall|i: int|
                0 <= i < r.0 ==> {
                    let e = #[trigger] entries@[i];
                    let s = final(self).slots_spec()[e.slot as int];
                    let prev = old(self).slots_spec()[e.slot as int];
                    &&& s.state_spec() == MwSlotState::Invalidating(prev.state_spec()->Bound_0)
                    &&& s.generation_spec() == e.generation
                    &&& s.window_spec() matches Some(w) && w.id() == prev.window_spec()->Some_0.id()
                        && w.state() is Invalidating && w.state()->Invalidating_wr_id == e.wr_id
                    &&& final(cq).in_flight().contains_key(e.wr_id)
                    &&& final(cq).in_flight()[e.wr_id].kind == (rdma::PendingKind::LocalInvalidate {
                        qp_id: qp.id(),
                        window_id: prev.window_spec()->Some_0.id(),
                    })
                },
            forall|slot: int|
                0 <= slot < final(self).slots_spec().len() && !(exists|i: int|
                    0 <= i < r.0 && (#[trigger] entries@[i]).slot == slot) ==> (
                #[trigger] final(self).slots_spec()[slot]) == old(self).slots_spec()[slot],
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let mut requests = Vec::<rdma::InvalidateRequest<'a>>::with_capacity(entries.len());
        let ghost mut cleared = Map::<u32, int>::empty();
        while requests.len() < entries.len()
            invariant
                requests.len() <= entries.len(),
                self.pd_id == old(self).pd_id,
                self.node == old(self).node,
                qp.pd_id() == old(self).pd(),
                qp.source().node() == old(self).node(),
                self.free == old(self).free,
                self.slots.len() == old(self).slots.len(),
                old(self).well_formed(),
                old(self).slots.len() <= u32::MAX,
                forall|i: int|
                    0 <= i < entries.len() ==> {
                        let e = #[trigger] entries@[i];
                        &&& e.slot < old(self).slots.len()
                        &&& !old(cq).in_flight().contains_key(e.wr_id)
                        &&& old(self).slots[e.slot as int].generation == e.generation
                        &&& old(self).slots[e.slot as int].state is Bound
                        &&& old(self).slot_window(e.slot as int) matches Some(w)
                            && w.state() is Bound && w.state()->Bound_qp_id == qp.id()
                    },
                forall|i: int, j: int|
                    0 <= i < j < entries.len() ==> (#[trigger] entries@[i]).slot != (
                    #[trigger] entries@[j]).slot && entries@[i].wr_id != entries@[j].wr_id,
                forall|i: int|
                    0 <= i < requests.len() ==> {
                        let e = #[trigger] entries@[i];
                        &&& cleared.contains_key(e.slot) && cleared[e.slot] == i
                        &&& requests[i].wr_id == e.wr_id
                        &&& old(self).slot_window(e.slot as int) == Some(requests[i].window)
                        &&& requests[i].admissible(qp)
                    },
                forall|s: int|
                    0 <= s < self.slots.len() ==> {
                        let slot = #[trigger] self.slots[s];
                        &&& if cleared.contains_key(s as u32) {
                            let i = cleared[s as u32];
                            &&& 0 <= i < requests.len() && entries@[i].slot == s
                            &&& slot == (MwPoolSlot { window: None, ..old(self).slots[s] })
                        } else {
                            slot == old(self).slots[s]
                        }
                    },
            decreases entries.len() - requests.len(),
        {
            let i = requests.len();
            let e = entries[i];
            let idx = e.slot as usize;
            proof {
                reveal(window_pool_slots_wf);
                assert(idx < self.slots.len());
                assert(self.slots@[idx as int] == self.slots@[e.slot as int]);
                if cleared.contains_key(e.slot) {
                    let j = cleared[e.slot];
                    assert(0 <= j < i);
                    assert(entries@[j].slot == e.slot);
                    assert(entries@[j].slot != entries@[i as int].slot);
                    assert(false);
                }
                let s = self.slots[idx as int];
                assert(s == old(self).slots[e.slot as int]);
                let w = s.window.unwrap();
                assert(w.state() is Bound && w.state()->Bound_qp_id == qp.id());
                assert(w.consistent() && w.pd_id() == qp.pd_id() && w.node() == qp.source().node());
            }
            let ghost prior = requests@;
            let ghost before = cleared;
            let window = self.slots[idx].window.take().unwrap();
            requests.push(rdma::InvalidateRequest { window, wr_id: e.wr_id });
            proof {
                cleared = cleared.insert(e.slot, i as int);
                assert(requests[i as int].admissible(qp));
                assert forall|j: int| 0 <= j < i implies {
                    let entry = #[trigger] entries@[j];
                    &&& requests[j] == prior[j]
                    &&& cleared.contains_key(entry.slot) && cleared[entry.slot] == j
                } by {
                    let entry = entries@[j];
                    assert(entry.slot != e.slot);
                    assert(before.contains_key(entry.slot));
                }
            }
        }
        let ghost planned = requests@;
        assert forall|i: int| 0 <= i < requests.len() implies {
            let r = #[trigger] requests[i];
            &&& r.admissible(qp) && !old(cq).in_flight().contains_key(r.wr_id)
        } by {
            assert(requests[i].wr_id == entries@[i].wr_id);
        }
        assert forall|i: int, j: int| 0 <= i < j < requests.len() implies (
        #[trigger] requests[i]).wr_id != (#[trigger] requests[j]).wr_id by {
            assert(requests[i].wr_id == entries@[i].wr_id);
            assert(requests[j].wr_id == entries@[j].wr_id);
        }
        let (posted, error) = rdma::post_invalidate_batch(qp, cq, &mut requests);
        let ghost submitted = requests@;
        while !requests.is_empty()
            invariant
                requests.len() <= entries.len() == submitted.len() == planned.len(),
                posted <= entries.len(),
                self.pd_id == old(self).pd_id,
                self.node == old(self).node,
                self.free == old(self).free,
                self.slots.len() == old(self).slots.len(),
                old(self).well_formed(),
                old(self).slots.len() <= u32::MAX,
                forall|i: int|
                    0 <= i < entries.len() ==> {
                        let e = #[trigger] entries@[i];
                        &&& e.slot < old(self).slots.len()
                        &&& !old(cq).in_flight().contains_key(e.wr_id)
                        &&& old(self).slots[e.slot as int].generation == e.generation
                        &&& old(self).slots[e.slot as int].state is Bound
                        &&& old(self).slot_window(e.slot as int) matches Some(w)
                            && w.state() is Bound && w.state()->Bound_qp_id == qp.id()
                    },
                forall|i: int, j: int|
                    0 <= i < j < entries.len() ==> (#[trigger] entries@[i]).slot != (
                    #[trigger] entries@[j]).slot && entries@[i].wr_id != entries@[j].wr_id,
                requests@ == submitted.take(requests.len() as int),
                forall|i: int|
                    0 <= i < entries.len() ==> {
                        let e = #[trigger] entries@[i];
                        &&& planned[i].wr_id == e.wr_id
                        &&& old(self).slot_window(e.slot as int) == Some(planned[i].window)
                        &&& if i < posted {
                            submitted[i].invalidates(planned[i], qp)
                        } else {
                            submitted[i] == planned[i]
                        }
                    },
                forall|i: int|
                    0 <= i < entries.len() ==> {
                        let e = #[trigger] entries@[i];
                        let slot = self.slots[e.slot as int];
                        let prev = old(self).slots[e.slot as int];
                        &&& cleared.contains_key(e.slot) && cleared[e.slot] == i
                        &&& if i < requests.len() {
                            slot == (MwPoolSlot { window: None, ..prev })
                        } else if i < posted {
                            slot == (MwPoolSlot {
                                state: MwSlotState::Invalidating(prev.state->Bound_0),
                                window: Some(submitted[i].window),
                                ..prev
                            })
                        } else {
                            slot == prev
                        }
                    },
                forall|s: int|
                    0 <= s < self.slots.len() ==> {
                        let slot = #[trigger] self.slots[s];
                        &&& if cleared.contains_key(s as u32) {
                            let i = cleared[s as u32];
                            0 <= i < entries.len() && entries@[i].slot == s
                        } else {
                            slot == old(self).slots[s]
                        }
                    },
            decreases requests.len(),
        {
            let idx = requests.len() - 1;
            let e = entries[idx];
            let request = requests.pop().unwrap();
            proof {
                assert(self.slots[e.slot as int] == (MwPoolSlot {
                    window: None,
                    ..old(self).slots[e.slot as int]
                }));
            }
            let s = &mut self.slots[e.slot as usize];
            s.window = Some(request.window);
            if idx < posted {
                let tag = match s.state {
                    MwSlotState::Bound(tag) => tag,
                    _ => {
                        assert(false);
                        unreachable!()
                    },
                };
                s.state = MwSlotState::Invalidating(tag);
            }
        }
        proof {
            reveal(window_pool_slots_wf);
            assert forall|j: int| 0 <= j < self.slots.len() implies {
                let s = #[trigger] self.slots[j];
                &&& (s.state == MwSlotState::Vacant || s.state is Quarantined) == s.window.is_none()
                &&& s.state == MwSlotState::Free ==> (s.window matches Some(w)
                    && w.state() is Unbound && self.free@.contains(j as u32))
                &&& s.window matches Some(w) ==> w.pd_id() == self.pd() && w.node() == self.node()
                    && ((w.state() is Binding || w.state() is Bound || w.state() is Invalidating)
                    ==> w.consistent())
            } by {
                if cleared.contains_key(j as u32) {
                    let i = cleared[j as u32];
                    assert(0 <= i < entries.len());
                    assert(entries@[i].slot == j);
                }
            }
        }
        let error = match error {
            Some(e) => Some(rdma_error(e)),
            None => None,
        };
        (posted, error)
    }
}

impl<'a> WindowPool<'a> {
    /// Post a local invalidate WR on a `Bound` MW slot.
    pub(crate) fn post_invalidate(
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
            (i as int) < old(self).slots_spec().len(),
            old(self).slots_spec()[i as int].generation_spec() == generation,
            old(self).slots_spec()[i as int].state_spec() is Invalidating,
            old(self).slots_spec()[i as int].window_spec() matches Some(w)
                && w.state() is Invalidating && witness.matches(w) && (
            w.state()->Invalidating_share matches Some(s) ==> returned matches Some(r) && r.share()
                == s && r.instance_id() == w.state()->Invalidating_instance_id && r.loc()
                == w.state()->Invalidating_slot.loc),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots_spec().len() && j != i as int
                    ==> #[trigger] final(self).slots_spec()[j] == old(self).slots_spec()[j],
            final(self).slots_spec()[i as int].state_spec() == MwSlotState::Free,
            final(self).slots_spec()[i as int].generation_spec() == generation,
            final(self).slots_spec()[i as int].window_spec() matches Some(w) && w.state()
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
    pub(crate) fn revoke_with_invalidate(
        &mut self,
        i: u32,
        generation: u32,
        host: &mut SyncContext,
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
    pub(crate) fn revoke_on_qp_reset(
        &mut self,
        i: u32,
        generation: u32,
        host: &mut SyncContext,
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
    pub(crate) fn dealloc_on_qp_reset(
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
            (i as int) < old(self).slots_spec().len(),
            old(self).slots_spec()[i as int].state_spec() == MwSlotState::Vacant,
            pd.id() == old(self).pd(),
            pd.node() == old(self).node(),
            name.well_formed(),
            name.source().is_rdma_qp(),
            name.node() == pd.node(),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots_spec().len() && j != i as int
                    ==> #[trigger] final(self).slots_spec()[j] == old(self).slots_spec()[j],
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> final(self).slots_spec()[i as int].state_spec() == MwSlotState::Free
                && final(self).slots_spec()[i as int].generation_spec() == old(
                self,
            ).slots_spec()[i as int].generation_spec() + 1,
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
    pub(crate) fn dealloc_all(&mut self) -> (r: Result<(), CoreError>)
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
