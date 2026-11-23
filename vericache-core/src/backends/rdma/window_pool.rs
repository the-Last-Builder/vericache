use crate::backends::rdma_error;
use crate::spec::CoreError;
#[cfg(verus_only)]
use transfer::context::{EventSourceToken, ReturnedShare, SlotToken, TransferInstance};
use transfer::domain::NodeId;
#[cfg(verus_only)]
use transfer::host::HostContext;
use transfer::rdma;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

verus! {

/// Lifecycle state of one pre-allocated Type-2 RDMA MemoryWindow slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MwSlotState {
    Free,
    Binding(u64),
    Bound(u64),
    Invalidating(u64),
    Quarantined(u64),
    Vacant,
}

/// One slot in a [`WindowPool`], pairing a Type-2 [`rdma::MemoryWindow`] with a
/// monotonically increasing generation counter (M5) to reject stale lease tokens.
pub struct MwPoolSlot<'a> {
    pub window: Option<rdma::MemoryWindow<'a>>,
    pub generation: u32,
    pub state: MwSlotState,
}

/// Opaque structural invariant for a [`WindowPool`]'s slot array and free list.
#[verifier::opaque]
pub open spec fn window_pool_slots_wf<'a>(
    slots: Seq<MwPoolSlot<'a>>,
    free: Seq<u32>,
    pd_id: nat,
    node: NodeId,
) -> bool {
    &&& slots.len() <= u32::MAX
    &&& free.no_duplicates()
    &&& forall|i: int|
        0 <= i < free.len() ==> {
            let k = #[trigger] free[i];
            (k as int) < slots.len() && slots[k as int].state == MwSlotState::Free
        }
    &&& forall|i: int|
        0 <= i < slots.len() ==> {
            let slot = #[trigger] slots[i];
            &&& (slot.state == MwSlotState::Vacant || slot.state is Quarantined)
                == slot.window.is_none()
            &&& slot.state == MwSlotState::Free ==> (slot.window matches Some(w) && w.state()
                == rdma::WindowState::Unbound && free.contains(i as u32))
            &&& slot.window matches Some(w) ==> (w.pd_id() == pd_id && w.node() == node) && (
            w.state() is Binding || w.state() is Bound || w.state() is Invalidating
                ==> w.consistent())
        }
}

/// Pool of pre-allocated Type-2 RDMA MemoryWindows bound to one ProtectionDomain.
///
/// Guarantees:
/// - **M5**: `checkout` monotonically increments `slot.generation`; stale
///   `(slot, generation)` pairs are rejected with `CoreError::StaleReplica`.
/// - **M3**: `fail_bind` extracts and returns `Tracked<WindowBindGrant>` so the
///   caller can restore `WindowBindRight` into its `ShareCap` without leaking.
/// - **M4**: `retire_and_replenish` and `dealloc_all` preserve any live
///   `MemoryWindow` handle on `dealloc_window` failure instead of dropping it.
pub struct WindowPool<'a> {
    pub slots: Vec<MwPoolSlot<'a>>,
    pub free: Vec<u32>,
    pub pd_id: Ghost<nat>,
    pub node: Ghost<NodeId>,
}

impl<'a> WindowPool<'a> {
    pub open spec fn well_formed(&self) -> bool {
        window_pool_slots_wf(self.slots@, self.free@, self.pd_id@, self.node@)
    }

    pub open spec fn pd(&self) -> nat {
        self.pd_id@
    }

    pub open spec fn node(&self) -> NodeId {
        self.node@
    }

    pub open spec fn len(&self) -> nat {
        self.slots@.len()
    }

    pub open spec fn slot_window(&self, idx: int) -> Option<rdma::MemoryWindow<'a>> {
        self.slots@[idx].window
    }

    pub fn capacity(&self) -> (n: usize)
        ensures
            n as nat == self.len(),
    {
        self.slots.len()
    }

    pub fn free_count(&self) -> (n: usize)
        ensures
            n == self.free@.len(),
    {
        self.free.len()
    }

    /// Build a pool from pre-allocated, unbound Type-2 `MemoryWindow`s on `pd`.
    pub fn new(
        mut windows: Vec<rdma::MemoryWindow<'a>>,
        Ghost(pd_id): Ghost<nat>,
        Ghost(node): Ghost<NodeId>,
    ) -> (pool: Self)
        requires
            windows.len() <= u32::MAX,
            forall|j: int|
                0 <= j < windows.len() ==> (#[trigger] windows[j]).state()
                    == rdma::WindowState::Unbound && windows[j].pd_id() == pd_id
                    && windows[j].node() == node,
        ensures
            pool.well_formed(),
            pool.pd() == pd_id,
            pool.node() == node,
            pool.len() == windows.len(),
            pool.free@.len() == windows.len(),
            forall|j: int|
                0 <= j < pool.slots@.len() ==> (#[trigger] pool.slots@[j]).state
                    == MwSlotState::Free,
    {
        let len = windows.len();
        let mut slots = Vec::<MwPoolSlot<'a>>::with_capacity(len);
        let mut free = Vec::<u32>::with_capacity(len);
        while !windows.is_empty()
            invariant
                windows.len() + slots.len() == len,
                free.len() == slots.len(),
                len <= u32::MAX,
                forall|j: int|
                    0 <= j < windows.len() ==> (#[trigger] windows[j]).state()
                        == rdma::WindowState::Unbound && windows[j].pd_id() == pd_id
                        && windows[j].node() == node,
                forall|j: int|
                    0 <= j < slots.len() ==> (#[trigger] slots[j]).state == MwSlotState::Free && (
                    slots[j].window matches Some(w) && w.state() == rdma::WindowState::Unbound
                        && w.pd_id() == pd_id && w.node() == node),
                forall|j: int| 0 <= j < free.len() ==> #[trigger] free[j] == j,
            decreases windows.len(),
        {
            let w = windows.pop().unwrap();
            free.push(slots.len() as u32);
            slots.push(MwPoolSlot { window: Some(w), generation: 0, state: MwSlotState::Free });
        }
        let pool = Self { slots, free, pd_id: Ghost(pd_id), node: Ghost(node) };
        proof {
            reveal(window_pool_slots_wf);
            assert forall|i: int| 0 <= i < pool.slots.len() implies #[trigger] pool.free@.contains(
                i as u32,
            ) by {
                assert(pool.free@[i] == i);
            }
        }
        pool
    }

    /// Check out a free MW slot, monotonically incrementing its generation (M5).
    pub fn checkout(&mut self, lease_tag: u64) -> (r: Result<(u32, u32), CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() ==> (#[trigger] final(self).slots[j]).window
                    == old(self).slots[j].window,
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((i, g)) => {
                    &&& (i as int) < final(self).slots.len()
                    &&& old(self).slots[i as int].state == MwSlotState::Free
                    &&& final(self).slots[i as int].generation == g
                    &&& g == old(self).slots[i as int].generation + 1
                    &&& final(self).slots[i as int].state == MwSlotState::Binding(lease_tag)
                    &&& final(self).slots[i as int].window matches Some(w) && w.state()
                        == rdma::WindowState::Unbound && w.pd_id() == old(self).pd() && w.node()
                        == old(self).node()
                    &&& forall|j: int|
                        0 <= j < final(self).slots.len() && j != i as int
                            ==> #[trigger] final(self).slots[j] == old(self).slots[j]
                },
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        if self.free.is_empty() {
            return Err(CoreError::CapacityExhausted);
        }
        let top = self.free.len() - 1;
        let i = self.free[top];
        if self.slots[i as usize].generation == u32::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let _popped = self.free.pop().unwrap();
        let slot = &mut self.slots[i as usize];
        slot.generation += 1;
        slot.state = MwSlotState::Binding(lease_tag);
        let generation = slot.generation;
        proof {
            assert(old(self).free@ =~= self.free@.push(i));
            assert forall|j: int|
                0 <= j < self.slots.len() && self.slots[j].state
                    == MwSlotState::Free implies #[trigger] self.free@.contains(j as u32) by {
                assert(old(self).free@.contains(j as u32));
                vstd::seq_lib::lemma_seq_contains_after_push(self.free@, i, j as u32);
            }
        }
        Ok((i, generation))
    }

    /// Return an unbound MW slot back to the free list, rejecting stale generations (M5).
    pub fn return_unbound(&mut self, i: u32, generation: u32) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].window matches Some(w) && w.state()
                == rdma::WindowState::Unbound,
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() ==> (#[trigger] final(self).slots[j]).window
                    == old(self).slots[j].window,
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() <==> (old(self).slots[i as int].generation == generation && !(old(
                self,
            ).slots[i as int].state is Free || old(self).slots[i as int].state is Vacant || old(
                self,
            ).slots[i as int].state is Quarantined)),
            r.is_ok() ==> {
                &&& final(self).slots[i as int].state == MwSlotState::Free
                &&& final(self).slots[i as int].generation == old(self).slots[i as int].generation
                &&& forall|j: int|
                    0 <= j < final(self).slots.len() && j != i as int
                        ==> #[trigger] final(self).slots[j] == old(self).slots[j]
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let idx = i as usize;
        if self.slots[idx].generation != generation
            || matches!(
                self.slots[idx].state,
                MwSlotState::Free | MwSlotState::Vacant | MwSlotState::Quarantined(_)
            ) {
            return Err(CoreError::StaleReplica);
        }
        self.slots[idx].state = MwSlotState::Free;
        self.free.push(i);
        proof {
            broadcast use vstd::seq_lib::lemma_seq_contains_after_push;

        }
        Ok(())
    }

    /// Post a Type-2 MW bind WR on a checked-out unbound slot.
    pub fn post_bind<'b>(
        &mut self,
        i: u32,
        generation: u32,
        qp: &rdma::QueuePair<'a>,
        cq: &mut rdma::CompletionQueue<'a>,
        slice: &rdma::MemoryRegionSlice<'b>,
        access: rdma::RemoteAccess,
        Tracked(grant): Tracked<rdma::WindowBindGrant>,
        wr_id: u64,
    ) -> (r: Result<u32, (CoreError, Tracked<rdma::WindowBindGrant>)>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Binding,
            old(self).slots[i as int].window matches Some(w) && w.state()
                == rdma::WindowState::Unbound,
            qp.can_post(*old(cq), wr_id),
            qp.pd_id() == old(self).pd(),
            qp.source().node() == old(self).node(),
            slice.well_formed(),
            slice.slot_ref().sub_len > 0,
            slice.registration().access().mw_bind,
            access == rdma::RemoteAccess::ReadWrite ==> slice.registration().access().local_write,
            slice.registration().pd_id() == qp.pd_id(),
            slice.slot_ref().loc.node == qp.source().node(),
            grant.access() == access,
            grant matches rdma::WindowBindGrant::ReadOnly(right) ==> right.instance_id()
                == slice.registration().instance_id() && right.loc() == slice.slot_ref().loc,
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            match r {
                Err((_, returned)) => {
                    &&& returned@ == grant
                    &&& final(self).slots@ == old(self).slots@
                    &&& final(self).free@ == old(self).free@
                    &&& *final(cq) == *old(cq)
                },
                Ok(next_rkey) => {
                    &&& final(cq).same_queue(*old(cq))
                    &&& final(cq).well_formed()
                    &&& final(self).slots[i as int].state == old(self).slots[i as int].state
                    &&& final(self).slots[i as int].generation == generation
                    &&& final(self).slots[i as int].window matches Some(w)
                    &&& w.consistent()
                    &&& w.id() == old(self).slots[i as int].window->Some_0.id()
                    &&& w.pd_id() == old(self).pd()
                    &&& w.node() == old(self).node()
                    &&& w.state() is Binding
                    &&& w.state()->Binding_qp_id == qp.id()
                    &&& w.state()->Binding_share == grant.share()
                    &&& w.state()->Binding_instance_id == slice.registration().instance_id()
                    &&& w.state()->Binding_wr_id == wr_id
                    &&& w.state()->Binding_target_slot == slice.slot_ref()
                    &&& w.state()->Binding_access == access
                    &&& w.state()->Binding_next_rkey == next_rkey
                    &&& final(cq).in_flight() == old(cq).in_flight().insert(
                        wr_id,
                        rdma::PendingWorkRequest {
                            kind: rdma::PendingKind::BindWindow {
                                qp_id: qp.id(),
                                share: grant.share(),
                                window_id: old(self).slots[i as int].window->Some_0.id(),
                                target_slot: slice.slot_ref(),
                                addr: w.state()->Binding_addr,
                                next_rkey,
                                access,
                            },
                        },
                    )
                },
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let mut mw = self.slots[idx].window.take().unwrap();
        let next_rkey = mw.next_rkey();
        match rdma::post_bind_window(
            qp,
            cq,
            slice,
            &mut mw,
            next_rkey,
            access,
            Tracked(grant),
            wr_id,
        ) {
            Ok(()) => {
                self.slots[idx].window = Some(mw);
                Ok(next_rkey)
            },
            Err((e, returned)) => {
                self.slots[idx].window = Some(mw);
                proof {
                    assert(self.slots@ =~= old(self).slots@);
                }
                Err((rdma_error(e), returned))
            },
        }
    }

    /// Complete a pending MW bind after observing `Completed::WindowBound`.
    pub fn complete_bind(
        &mut self,
        i: u32,
        generation: u32,
        Tracked(witness): Tracked<rdma::BindWitness>,
    ) -> (r: Result<rdma::RemoteCapability, CoreError>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Binding,
            old(self).slots[i as int].window matches Some(w) && w.state() is Binding
                && witness.matches(w),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            forall|j: int|
                0 <= j < final(self).slots.len() && j != i as int
                    ==> #[trigger] final(self).slots[j] == old(self).slots[j],
            r matches Ok(cap) && {
                let old_w = old(self).slots[i as int].window->Some_0;
                &&& final(self).slots[i as int].state == MwSlotState::Bound(
                    old(self).slots[i as int].state->Binding_0,
                )
                &&& final(self).slots[i as int].generation == generation
                &&& final(self).slots[i as int].window matches Some(w)
                &&& w.consistent()
                &&& w.id() == old_w.id()
                &&& w.state() == (rdma::WindowState::Bound {
                    instance_id: old_w.state()->Binding_instance_id,
                    qp_id: old_w.state()->Binding_qp_id,
                    share: old_w.state()->Binding_share,
                    slot: old_w.state()->Binding_target_slot,
                    addr: old_w.state()->Binding_addr,
                    rkey: old_w.state()->Binding_next_rkey,
                    access: old_w.state()->Binding_access,
                })
                &&& cap.well_formed()
                &&& cap.slot_ref() == old_w.state()->Binding_target_slot
                &&& cap.access() == old_w.state()->Binding_access
                &&& cap.share() == old_w.state()->Binding_share
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let lease_tag = match self.slots[idx].state {
            MwSlotState::Binding(tag) => tag,
            _ => 0,
        };
        let mut mw = self.slots[idx].window.take().unwrap();
        mw.finish_bind(Tracked(witness));
        let cap = mw.remote_capability();
        self.slots[idx].window = Some(mw);
        self.slots[idx].state = MwSlotState::Bound(lease_tag);
        Ok(cap)
    }

    /// Recover the `WindowBindGrant` on `Completed::BindFailed` (M3) and return
    /// the unbound MW slot to the pool's free list.
    pub fn fail_bind(
        &mut self,
        i: u32,
        generation: u32,
        Tracked(witness): Tracked<rdma::BindFailedWitness>,
    ) -> (grant: Tracked<rdma::WindowBindGrant>)
        requires
            old(self).well_formed(),
            (i as int) < old(self).slots.len(),
            old(self).slots[i as int].generation == generation,
            old(self).slots[i as int].state is Binding,
            old(self).slots[i as int].window matches Some(w) && w.state() is Binding
                && witness.matches(w),
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
            ({
                let old_w = old(self).slots[i as int].window->Some_0;
                &&& grant@.access() == old_w.state()->Binding_access
                &&& grant@.share() == old_w.state()->Binding_share
                &&& grant@ matches rdma::WindowBindGrant::ReadOnly(right) ==> right.instance_id()
                    == old_w.state()->Binding_instance_id && right.loc()
                    == old_w.state()->Binding_target_slot.loc
            }),
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let _ = generation;
        let idx = i as usize;
        let mut mw = self.slots[idx].window.take().unwrap();
        let grant = mw.fail_bind(Tracked(witness));
        self.slots[idx].window = Some(mw);
        self.slots[idx].state = MwSlotState::Free;
        self.free.push(i);
        proof {
            broadcast use vstd::seq_lib::lemma_seq_contains_after_push;

        }
        grant
    }

    /// Acknowledge a peer `RevokeAck` for a `Quarantined` slot, transitioning it
    /// to `Vacant` so it may be replenished with a fresh Type-2 MW.
    pub fn acknowledge_quarantined(&mut self, i: u32) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> {
                &&& (i as int) < old(self).slots.len()
                &&& old(self).slots[i as int].state is Quarantined
                &&& final(self).slots[i as int].state == MwSlotState::Vacant
                &&& final(self).slots[i as int].window is None
                &&& final(self).slots[i as int].generation == old(self).slots[i as int].generation
                &&& forall|j: int|
                    0 <= j < final(self).slots.len() && j != i as int
                        ==> #[trigger] final(self).slots[j] == old(self).slots[j]
            },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let idx = i as usize;
        if idx >= self.slots.len() {
            return Err(CoreError::InvalidArgument);
        }
        if !matches!(self.slots[idx].state, MwSlotState::Quarantined(_)) {
            return Err(CoreError::WouldConflict);
        }
        self.slots[idx].state = MwSlotState::Vacant;
        Ok(())
    }

    /// Transition all `Quarantined(_)` slots to `Vacant` (called on QP reset).
    pub fn unquarantine_all(&mut self)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).pd() == old(self).pd(),
            final(self).node() == old(self).node(),
            final(self).len() == old(self).len(),
            final(self).free@ == old(self).free@,
            forall|j: int|
                0 <= j < final(self).slots.len() ==> {
                    if old(self).slots[j].state is Quarantined {
                        (#[trigger] final(self).slots[j]).state == MwSlotState::Vacant
                            && final(self).slots[j].window is None
                            && final(self).slots[j].generation == old(self).slots[j].generation
                    } else {
                        (#[trigger] final(self).slots[j]) == old(self).slots[j]
                    }
                },
    {
        proof {
            reveal(window_pool_slots_wf);
        }
        let mut idx: usize = 0;
        while idx < self.slots.len()
            invariant
                idx <= self.slots.len(),
                self.slots.len() == old(self).slots.len(),
                self.slots.len() <= u32::MAX,
                self.pd_id == old(self).pd_id,
                self.node == old(self).node,
                self.free@ == old(self).free@,
                self.free@.no_duplicates(),
                forall|k: int|
                    0 <= k < self.free.len() ==> {
                        let s = #[trigger] self.free[k];
                        (s as int) < self.slots.len() && old(self).slots[s as int].state
                            == MwSlotState::Free
                    },
                forall|j: int|
                    0 <= j < self.slots.len() && (#[trigger] old(
                        self,
                    ).slots[j]).state is Quarantined ==> old(self).slots[j].window is None,
                forall|j: int|
                    0 <= j < idx ==> {
                        if old(self).slots[j].state is Quarantined {
                            (#[trigger] self.slots[j]).state == MwSlotState::Vacant
                                && self.slots[j].window is None && self.slots[j].generation == old(
                                self,
                            ).slots[j].generation
                        } else {
                            (#[trigger] self.slots[j]) == old(self).slots[j]
                        }
                    },
                forall|j: int|
                    idx <= j < self.slots.len() ==> (#[trigger] self.slots[j]) == old(
                        self,
                    ).slots[j],
            decreases self.slots.len() - idx,
        {
            if matches!(self.slots[idx].state, MwSlotState::Quarantined(_)) {
                proof {
                    assert(old(self).slots[idx as int] == self.slots[idx as int]);
                    assert(old(self).slots[idx as int].window is None);
                }
                self.slots[idx].state = MwSlotState::Vacant;
            }
            idx += 1;
        }
    }
}

} // verus!
