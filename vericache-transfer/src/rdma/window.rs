use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub tracked enum WindowBindGrant {
    ReadOnly(FenceRight),
    ReadWrite,
}

impl WindowBindGrant {
    pub open spec fn access(self) -> RemoteAccess {
        match self {
            Self::ReadOnly(_) => RemoteAccess::ReadOnly,
            Self::ReadWrite => RemoteAccess::ReadWrite,
        }
    }

    pub open spec fn share(self) -> Option<TransferShare> {
        match self {
            Self::ReadOnly(right) => Some(right.share()),
            Self::ReadWrite => None,
        }
    }
}

pub enum WindowState {
    Unbound,
    Binding {
        instance_id: InstanceId,
        qp_id: nat,
        share: Option<TransferShare>,
        wr_id: u64,
        target_slot: SlotRef,
        addr: u64,
        next_rkey: u32,
        access: RemoteAccess,
    },
    Bound {
        instance_id: InstanceId,
        qp_id: nat,
        share: Option<TransferShare>,
        slot: SlotRef,
        addr: u64,
        rkey: u32,
        access: RemoteAccess,
    },
    Invalidating {
        instance_id: InstanceId,
        qp_id: nat,
        share: Option<TransferShare>,
        wr_id: u64,
        slot: SlotRef,
        access: RemoteAccess,
        prev_rkey: u32,
    },
    Retired,
}

pub struct MemoryWindow<'a> {
    pub(super) raw: ffi::RawMwGuard,
    pub(super) pd: &'a ProtectionDomain<'a>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    window_id: Ghost<nat>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    state: Ghost<WindowState>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pending_grant: Tracked<Option<WindowBindGrant>>,
    pub(super) addr: u64,
    pub(super) len: u64,
    pub(super) rkey: u32,
}

pub fn alloc_window<'a>(
    pd: &'a ProtectionDomain<'a>,
    Tracked(name): Tracked<EventSourceToken>,
) -> (r: Result<MemoryWindow<'a>, RdmaError>)
    requires
        name.well_formed(),
        name.source().is_rdma_qp(),
        name.node() == pd.node(),
    ensures
        r matches Ok(w) ==> w.state() == WindowState::Unbound && w.pd_id() == pd.id() && w.node()
            == pd.node() && w.id() == name.source().id(),
{
    if !pd.context.supports_type_2()? {
        return Err(RdmaError(95));
    }
    let (raw, rkey) = ffi::alloc_window(pd.raw.as_ptr())?;
    Ok(
        MemoryWindow {
            raw,
            pd,
            window_id: Ghost(name.source().id()),
            state: Ghost(WindowState::Unbound),
            pending_grant: Tracked(None),
            addr: 0,
            len: 0,
            rkey,
        },
    )
}

pub fn dealloc_window<'a>(w: MemoryWindow<'a>) -> (r: Result<(), (RdmaError, MemoryWindow<'a>)>)
    requires
        w.state() == WindowState::Unbound || w.state() == WindowState::Retired,
    ensures
        r matches Err((_, returned)) ==> returned == w,
{
    match ffi::dealloc_window(w.raw.as_ptr()) {
        Ok(()) => {
            let _ = w.raw.into_raw();
            Ok(())
        },
        Err(e) => Err((e, w)),
    }
}

impl<'a> MemoryWindow<'a> {
    pub fn protection_domain(&self) -> &'a ProtectionDomain<'a> {
        self.pd
    }

    pub closed spec fn state(&self) -> WindowState {
        self.state@
    }

    pub closed spec fn pd_id(&self) -> nat {
        self.pd.id()
    }

    pub closed spec fn node(&self) -> NodeId {
        self.pd.node()
    }

    pub closed spec fn id(&self) -> nat {
        self.window_id@
    }

    pub closed spec fn consistent(&self) -> bool {
        match self.state@ {
            WindowState::Binding {
                instance_id,
                share,
                target_slot,
                addr,
                next_rkey,
                access,
                ..
            } => target_slot.well_formed() && target_slot.sub_len == self.len && self.addr == addr
                && next_rkey == self.rkey && self.addr + self.len <= u64::MAX && self.len > 0 && (
            self.pending_grant@ matches Some(g) && g.access() == access && g.share() == share && (
            g matches WindowBindGrant::ReadOnly(right) ==> right.instance_id() == instance_id
                && right.loc() == target_slot.loc)),
            WindowState::Bound { slot, addr, rkey, .. } => slot.well_formed() && slot.sub_len
                == self.len && self.addr == addr && rkey == self.rkey && self.addr + self.len
                <= u64::MAX && self.len > 0,
            _ => true,
        }
    }

    pub fn next_rkey(&self) -> u32 {
        (self.rkey & 0xffffff00) | (self.rkey.wrapping_add(1) & 255)
    }

    pub fn finish_bind(&mut self, Tracked(w): Tracked<BindWitness>)
        requires
            old(self).consistent(),
            old(self).state() is Binding,
            w.matches(*old(self)),
        ensures
            final(self).consistent(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).pd_id() == old(self).pd_id(),
            final(self).state() == (WindowState::Bound {
                instance_id: old(self).state()->Binding_instance_id,
                qp_id: old(self).state()->Binding_qp_id,
                share: old(self).state()->Binding_share,
                slot: old(self).state()->Binding_target_slot,
                addr: old(self).state()->Binding_addr,
                rkey: old(self).state()->Binding_next_rkey,
                access: old(self).state()->Binding_access,
            }),
    {
        proof {
            let s = self.state@;
            self.pending_grant = Tracked(None);
            self.state = Ghost(
                WindowState::Bound {
                    instance_id: s->Binding_instance_id,
                    qp_id: s->Binding_qp_id,
                    share: s->Binding_share,
                    slot: s->Binding_target_slot,
                    addr: s->Binding_addr,
                    rkey: s->Binding_next_rkey,
                    access: s->Binding_access,
                },
            );
        }
    }

    pub fn fail_bind(&mut self, Tracked(w): Tracked<BindFailedWitness>) -> (grant: Tracked<
        WindowBindGrant,
    >)
        requires
            old(self).consistent(),
            old(self).state() is Binding,
            w.matches(*old(self)),
        ensures
            final(self).consistent(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).pd_id() == old(self).pd_id(),
            final(self).state() == WindowState::Unbound,
            grant@.access() == old(self).state()->Binding_access,
            grant@.share() == old(self).state()->Binding_share,
            grant@ matches WindowBindGrant::ReadOnly(right) ==> right.instance_id() == old(
                self,
            ).state()->Binding_instance_id && right.loc() == old(
                self,
            ).state()->Binding_target_slot.loc,
    {
        let tracked mut taken = None;
        proof {
            vstd::modes::tracked_swap(self.pending_grant.borrow_mut(), &mut taken);
            self.state = Ghost(WindowState::Unbound);
        }
        Tracked(taken.tracked_unwrap())
    }

    pub fn remote_capability(&self) -> (cap: RemoteCapability)
        requires
            self.state() is Bound,
            self.consistent(),
        ensures
            cap.well_formed(),
            cap.slot_ref() == self.state()->Bound_slot,
            cap.access() == self.state()->Bound_access,
            cap.share() == self.state()->Bound_share,
            cap.descriptor() == (
                self.state()->Bound_addr,
                cap.slot_ref().sub_len,
                self.state()->Bound_rkey,
            ),
    {
        ffi::bound_capability(
            self.addr,
            self.len,
            self.rkey,
            Ghost(self.state@->Bound_slot),
            Ghost(self.state@->Bound_access),
            Ghost(self.state@->Bound_share),
        )
    }

    pub fn finish_invalidate(
        &mut self,
        Tracked(w): Tracked<InvalidateWitness>,
        Tracked(returned): Tracked<Option<ReturnedShare>>,
    )
        requires
            old(self).state() is Invalidating,
            old(self).state()->Invalidating_share matches Some(s) ==> returned matches Some(r)
                && r.share() == s && r.instance_id() == old(self).state()->Invalidating_instance_id
                && r.loc() == old(self).state()->Invalidating_slot.loc,
            w.matches(*old(self)),
        ensures
            final(self).state() == WindowState::Unbound,
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).pd_id() == old(self).pd_id(),
    {
        proof {
            self.state = Ghost(WindowState::Unbound);
        }
    }

    pub open spec fn can_revoke(
        &self,
        host: SyncContext,
        instance: InstanceId,
        lent: SlotToken,
    ) -> bool {
        let binding = self.state().binding();
        &&& self.state() is Binding || self.state() is Bound || self.state() is Invalidating
        &&& binding.instance == instance && host.ready(instance, self.node())
        &&& binding.share is Some
        &&& slot_is_lent_with(lent, instance, binding.slot.loc, binding.share->Some_0)
    }
}

pub ghost struct WindowBinding {
    pub qp_id: nat,
    pub instance: InstanceId,
    pub slot: SlotRef,
    pub share: Option<TransferShare>,
}

impl WindowState {
    pub open spec fn binding(self) -> WindowBinding
        recommends
            self is Binding || self is Bound || self is Invalidating,
    {
        match self {
            Self::Binding { qp_id, instance_id, target_slot, share, .. } => WindowBinding {
                qp_id,
                instance: instance_id,
                slot: target_slot,
                share,
            },
            Self::Bound { qp_id, instance_id, slot, share, .. }
            | Self::Invalidating { qp_id, instance_id, slot, share, .. } => WindowBinding {
                qp_id,
                instance: instance_id,
                slot,
                share,
            },
            _ => arbitrary(),
        }
    }
}

pub type Revocation = (Tracked<Completion>, Tracked<RevokedToken>, Tracked<SlotToken>);

pub open spec fn revocation_result(
    result: Revocation,
    instance: InstanceId,
    lent: SlotToken,
    share: TransferShare,
    event: TransferEvent,
) -> bool {
    &&& result.0@.records(instance, event, no_access(), |_, _| true, ISet::empty(), IMap::empty())
    &&& result.1@.instance_id() == instance && result.1@.key() == (lent.key(), share)
        && result.1@.value() == event
    &&& result.2@.well_formed() && result.2@.instance_id() == instance && result.2@.key()
        == lent.key() && result.2@.value() == fold_share(
        lent.value(),
        share,
        Frontier { last_writer: None, readers: ISet::empty().insert(event) },
    )
}

fn perform_revocation(
    host: &mut SyncContext,
    Tracked(instance): Tracked<&TransferInstance>,
    Ghost(share): Ghost<TransferShare>,
    Tracked(lent): Tracked<SlotToken>,
    Tracked(completed): Tracked<CompletedFence>,
) -> (result: Revocation)
    requires
        old(host).ready(instance.id(), old(host).source@.node()),
        slot_is_lent_with(lent, instance.id(), lent.key(), share),
        completed.matches(instance.id(), lent.key(), share),
    ensures
        final(host).stepped(*old(host)),
        revocation_result(result, instance.id(), lent, share, old(host).source@.next_event()),
{
    let tracked name = host.source.borrow_mut().take_next();
    let tracked (Tracked(submitted), Tracked(revoked), Tracked(slot), Tracked(knows)) =
        instance.revoke_slot(share, lent, name, completed);
    (Tracked(Completion { submitted, knows }), Tracked(revoked), Tracked(slot))
}

/// IBTA-strict unilateral MW revocation: deallocates the invalidated MW (`ibv_dealloc_mw` == 0,
/// IBTA C10-67..70) before minting the `Revocation` fence witness.
pub fn revoke_read_share_with_window<'a>(
    host: &mut SyncContext,
    Tracked(instance): Tracked<&TransferInstance>,
    mw: MemoryWindow<'a>,
    Tracked(witness): Tracked<InvalidateWitness>,
    Tracked(lent): Tracked<SlotToken>,
) -> (r: Result<
    Revocation,
    (RdmaError, MemoryWindow<'a>, Tracked<InvalidateWitness>, Tracked<SlotToken>),
>)
    requires
        mw.can_revoke(*old(host), instance.id(), lent),
        mw.state() is Invalidating,
        witness.matches(mw),
    ensures
        r matches Err((_, ret_mw, ret_w, ret_lent)) ==> ret_mw == mw && ret_w@ == witness
            && ret_lent@ == lent && *final(host) == *old(host),
        r matches Ok(result) ==> final(host).stepped(*old(host)) && revocation_result(
            result,
            instance.id(),
            lent,
            mw.state().binding().share->Some_0,
            old(host).source@.next_event(),
        ),
{
    let evidence = match ffi::dealloc_fenced_window(&mw) {
        Err(e) => return Err((e, mw, Tracked(witness), Tracked(lent))),
        Ok(evidence) => evidence,
    };
    let _ = mw.raw.into_raw();
    let ghost share = mw.state().binding().share->Some_0;
    Ok(perform_revocation(host, Tracked(instance), Ghost(share), Tracked(lent), evidence))
}

/// Unilateral MW revocation after resetting the bound QP (`IBV_QPS_RESET`) and deallocating
/// the window (`ibv_dealloc_mw` == 0).
pub fn revoke_read_share_on_reset_qp<'a>(
    host: &mut SyncContext,
    Tracked(instance): Tracked<&TransferInstance>,
    mw: MemoryWindow<'a>,
    Tracked(qp_reset): Tracked<&QpResetWitness>,
    Tracked(lent): Tracked<SlotToken>,
) -> (r: Result<Revocation, (RdmaError, MemoryWindow<'a>, Tracked<SlotToken>)>)
    requires
        mw.can_revoke(*old(host), instance.id(), lent),
        qp_reset.matches_qp(mw.state().binding().qp_id, mw.node()),
    ensures
        r matches Err((_, ret_mw, ret_lent)) ==> ret_mw == mw && ret_lent@ == lent && *final(host)
            == *old(host),
        r matches Ok(result) ==> final(host).stepped(*old(host)) && revocation_result(
            result,
            instance.id(),
            lent,
            mw.state().binding().share->Some_0,
            old(host).source@.next_event(),
        ),
{
    let evidence = match ffi::dealloc_fenced_window(&mw) {
        Err(e) => return Err((e, mw, Tracked(lent))),
        Ok(evidence) => evidence,
    };
    let _ = mw.raw.into_raw();
    let ghost share = mw.state().binding().share->Some_0;
    Ok(perform_revocation(host, Tracked(instance), Ghost(share), Tracked(lent), evidence))
}

/// Deallocate a `Binding` or cooperative `Invalidating` window once its QP has been reset.
pub fn dealloc_window_on_reset_qp<'a>(
    mw: MemoryWindow<'a>,
    Tracked(qp_reset): Tracked<&QpResetWitness>,
    Tracked(returned): Tracked<Option<ReturnedShare>>,
) -> (r: Result<
    Tracked<Option<WindowBindGrant>>,
    (RdmaError, MemoryWindow<'a>, Tracked<Option<ReturnedShare>>),
>)
    requires
        mw.consistent(),
        mw.state() is Binding || mw.state() is Invalidating,
        qp_reset.matches_qp(mw.state().binding().qp_id, mw.node()),
        mw.state() is Invalidating ==> ({
            let s = mw.state();
            s->Invalidating_share matches Some(sh) ==> returned matches Some(r) && r.share() == sh
                && r.instance_id() == s->Invalidating_instance_id && r.loc()
                == s->Invalidating_slot.loc
        }),
    ensures
        r matches Err((_, ret_mw, ret_sh)) ==> ret_mw == mw && ret_sh@ == returned,
        r matches Ok(grant_opt) ==> match mw.state() {
            WindowState::Binding { instance_id, share, target_slot, access, .. } => {
                &&& grant_opt@ matches Some(g)
                &&& g.access() == access
                &&& g.share() == share
                &&& (g matches WindowBindGrant::ReadOnly(right) ==> right.instance_id()
                    == instance_id && right.loc() == target_slot.loc)
            },
            _ => grant_opt@ is None,
        },
{
    if let Err(e) = ffi::dealloc_window(mw.raw.as_ptr()) {
        return Err((e, mw, Tracked(returned)));
    }
    let tracked out = if mw.state@ is Binding {
        mw.pending_grant.get()
    } else {
        None
    };
    let _ = mw.raw.into_raw();
    Ok(Tracked(out))
}

pub fn post_bind_window(
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    slice: &MemoryRegionSlice,
    mw: &mut MemoryWindow,
    next_rkey: u32,
    access: RemoteAccess,
    Tracked(grant): Tracked<WindowBindGrant>,
    wr_id: u64,
) -> (r: Result<(), (RdmaError, Tracked<WindowBindGrant>)>)
    requires
        qp.can_post(*old(cq), wr_id),
        old(mw).state() == WindowState::Unbound,
        slice.well_formed(),
        slice.slot_ref().sub_len > 0,
        grant.access() == access,
        grant matches WindowBindGrant::ReadOnly(right) ==> right.instance_id()
            == slice.registration().instance_id() && right.loc() == slice.slot_ref().loc,
        slice.registration().access().mw_bind,
        access == RemoteAccess::ReadWrite ==> slice.registration().access().local_write,
        slice.registration().pd_id() == qp.pd_id(),
        old(mw).pd_id() == qp.pd_id(),
        old(mw).node() == qp.source().node(),
        slice.slot_ref().loc.node == qp.source().node(),
    ensures
        r matches Err((_, returned)) ==> returned@ == grant && *final(cq) == *old(cq) && *final(mw)
            == *old(mw),
        r.is_ok() ==> final(cq).same_queue(*old(cq)) && final(cq).well_formed()
            && final(mw).consistent() && final(mw).id() == old(mw).id() && final(mw).node() == old(
            mw,
        ).node() && final(mw).pd_id() == old(mw).pd_id() && final(mw).state() is Binding
            && final(mw).state()->Binding_qp_id == qp.id() && final(mw).state()->Binding_share
            == grant.share() && final(mw).state()->Binding_instance_id
            == slice.registration().instance_id() && final(mw).state()->Binding_wr_id == wr_id
            && final(mw).state()->Binding_target_slot == slice.slot_ref()
            && final(mw).state()->Binding_access == access && final(mw).state()->Binding_next_rkey
            == next_rkey && final(cq).pending() == old(cq).pending().insert(
            wr_id,
            PendingWorkRequest {
                kind: PendingKind::BindWindow {
                    qp_id: qp.id(),
                    share: grant.share(),
                    window_id: old(mw).id(),
                    target_slot: slice.slot_ref(),
                    addr: final(mw).state()->Binding_addr,
                    next_rkey,
                    access,
                },
            },
        ),
{
    match post_or_prepare_bind_window(
        qp,
        cq,
        slice,
        mw,
        next_rkey,
        access,
        Tracked(grant),
        wr_id,
        false,
    ) {
        Ok(_) => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn post_or_prepare_bind_window(
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    slice: &MemoryRegionSlice,
    mw: &mut MemoryWindow,
    next_rkey: u32,
    access: RemoteAccess,
    Tracked(grant): Tracked<WindowBindGrant>,
    wr_id: u64,
    prepare: bool,
) -> (r: Result<Option<PreparedBind>, (RdmaError, Tracked<WindowBindGrant>)>)
    requires
        qp.can_post(*old(cq), wr_id),
        old(mw).state() == WindowState::Unbound,
        slice.well_formed(),
        slice.slot_ref().sub_len > 0,
        grant.access() == access,
        grant matches WindowBindGrant::ReadOnly(right) ==> right.instance_id()
            == slice.registration().instance_id() && right.loc() == slice.slot_ref().loc,
        slice.registration().access().mw_bind,
        access == RemoteAccess::ReadWrite ==> slice.registration().access().local_write,
        slice.registration().pd_id() == qp.pd_id(),
        old(mw).pd_id() == qp.pd_id(),
        old(mw).node() == qp.source().node(),
        slice.slot_ref().loc.node == qp.source().node(),
    ensures
        r matches Err((_, returned)) ==> returned@ == grant && *final(cq) == *old(cq) && *final(mw)
            == *old(mw),
        r.is_ok() ==> final(cq).same_queue(*old(cq)) && final(cq).well_formed()
            && final(mw).consistent() && final(mw).id() == old(mw).id() && final(mw).node() == old(
            mw,
        ).node() && final(mw).pd_id() == old(mw).pd_id() && final(mw).state() is Binding
            && final(mw).state()->Binding_qp_id == qp.id() && final(mw).state()->Binding_share
            == grant.share() && final(mw).state()->Binding_instance_id
            == slice.registration().instance_id() && final(mw).state()->Binding_wr_id == wr_id
            && final(mw).state()->Binding_target_slot == slice.slot_ref()
            && final(mw).state()->Binding_access == access && final(mw).state()->Binding_next_rkey
            == next_rkey && final(cq).pending() == old(cq).pending().insert(
            wr_id,
            PendingWorkRequest {
                kind: PendingKind::BindWindow {
                    qp_id: qp.id(),
                    share: grant.share(),
                    window_id: old(mw).id(),
                    target_slot: slice.slot_ref(),
                    addr: final(mw).state()->Binding_addr,
                    next_rkey,
                    access,
                },
            },
        ),
        r matches Ok(reserved) ==> {
            &&& reserved.is_some() == prepare
            &&& reserved matches Some(p) ==> p.matches(qp, final(cq), final(mw), *slice)
                && p.matches_window(qp, final(cq), final(mw)) && p.id() == wr_id
        },
{
    let mr = slice.mr;
    let addr = slice.addr;
    let len = slice.len;
    if len == 0 || next_rkey != mw.next_rkey() || !mr.flags.mw_bind || (access
        == RemoteAccess::ReadWrite && !mr.flags.local_write) || mr.pd.raw.as_ptr()
        != qp.connection.pd.raw.as_ptr() || mw.pd.raw.as_ptr() != mr.pd.raw.as_ptr() {
        return Err((RdmaError(22), Tracked(grant)));
    }
    if !prepare {
        if let Err(e) = ffi::bind(
            qp.connection.raw.as_ptr(),
            mw.raw.as_ptr(),
            mr.raw.as_ptr(),
            addr,
            len,
            next_rkey,
            access,
            wr_id,
        ) {
            return Err((e, Tracked(grant)));
        }
    }
    mw.addr = addr;
    mw.len = len;
    mw.rkey = next_rkey;
    cq.in_flight_count += 1;
    proof {
        let target_slot = slice.slot_ref();
        let ghost share = grant.share();
        mw.pending_grant = Tracked(Some(grant));
        mw.state = Ghost(
            WindowState::Binding {
                instance_id: mr.instance_id(),
                qp_id: qp.id(),
                share,
                wr_id,
                target_slot,
                addr,
                next_rkey,
                access,
            },
        );
        cq.in_flight = Ghost(
            cq.pending().insert(
                wr_id,
                PendingWorkRequest {
                    kind: PendingKind::BindWindow {
                        qp_id: qp.id(),
                        share,
                        window_id: mw.id(),
                        target_slot,
                        addr,
                        next_rkey,
                        access,
                    },
                },
            ),
        );
    }
    let reservation = if prepare {
        Some(
            PreparedBind {
                wr_id,
                access,
                queue: Ghost(cq.id()),
                window: Ghost(mw.id()),
                qp: Ghost(qp.id()),
                slot: Ghost(slice.slot_ref()),
            },
        )
    } else {
        None
    };
    Ok(reservation)
}

/// A reservation owns the right to submit one pending bind. The window remains
/// in Binding until its actual completion or an unposted-suffix failure.
pub struct PreparedBind {
    pub(super) wr_id: u64,
    pub(super) access: RemoteAccess,
    queue: Ghost<nat>,
    window: Ghost<nat>,
    qp: Ghost<nat>,
    slot: Ghost<SlotRef>,
}

impl PreparedBind {
    pub closed spec fn matches(
        &self,
        qp: &QueuePair,
        cq: &CompletionQueue,
        mw: &MemoryWindow,
        slice: MemoryRegionSlice,
    ) -> bool {
        &&& self.matches_window(qp, cq, mw)
        &&& slice.well_formed() && slice.slot_ref() == self.slot@
        &&& slice.registration().pd_id() == qp.pd_id()
    }

    pub closed spec fn matches_window(
        &self,
        qp: &QueuePair,
        cq: &CompletionQueue,
        mw: &MemoryWindow,
    ) -> bool {
        &&& qp.ready() && qp.cq_id() == cq.id() && self.queue@ == cq.id() && self.qp@ == qp.id()
        &&& self.window@ == mw.id()
        &&& mw.pd_id() == qp.pd_id()
        &&& mw.state() is Binding && mw.consistent()
        &&& mw.state()->Binding_target_slot == self.slot@
        &&& mw.state()->Binding_access == self.access
        &&& mw.state()->Binding_wr_id == self.wr_id
        &&& self.is_pending(*cq)
    }

    pub proof fn lemma_matches_slice(
        &self,
        qp: &QueuePair,
        cq: &CompletionQueue,
        mw: &MemoryWindow,
        slice: MemoryRegionSlice,
    )
        requires
            self.matches_window(qp, cq, mw),
            slice.well_formed(),
            slice.slot_ref() == mw.state()->Binding_target_slot,
            slice.registration().pd_id() == qp.pd_id(),
        ensures
            self.matches(qp, cq, mw, slice),
            self.id() == mw.state()->Binding_wr_id,
    {
    }

    pub closed spec fn is_pending(&self, cq: CompletionQueue) -> bool {
        &&& cq.id() == self.queue@
        &&& cq.pending().contains_key(self.wr_id)
        &&& cq.pending()[self.wr_id].kind is BindWindow
        &&& cq.pending()[self.wr_id].kind->BindWindow_window_id == self.window@
    }

    pub closed spec fn id(&self) -> u64 {
        self.wr_id
    }

    pub proof fn lemma_matches_preserved(
        &self,
        qp: &QueuePair,
        before: &CompletionQueue,
        after: &CompletionQueue,
        mw: &MemoryWindow,
    )
        requires
            self.matches_window(qp, before, mw),
            after.same_queue(*before),
            before.in_flight().submap_of(after.in_flight()),
        ensures
            self.matches_window(qp, after, mw),
    {
    }
}

pub struct BindBatchRequest<'w, 'a> {
    pub prepared: PreparedBind,
    pub window: &'w MemoryWindow<'a>,
    pub slice: MemoryRegionSlice<'w>,
}

/// One doorbell for the reserved WRs, in list order. The returned count is
/// the exact native prefix. Polling also reports an error for each unposted
/// reservation, allowing callers to return its grant and unpin its frame.
pub fn post_bind_batch(
    qp: &QueuePair,
    cq: &CompletionQueue,
    requests: Vec<BindBatchRequest>,
) -> (out: (usize, Option<RdmaError>))
    requires
        forall|i: int|
            0 <= i < requests.len() ==> {
                let r = #[trigger] requests[i];
                r.prepared.matches(qp, cq, r.window, r.slice)
            },
        forall|i: int, j: int|
            0 <= i < j < requests.len() ==> (#[trigger] requests[i]).prepared.id() != (
            #[trigger] requests[j]).prepared.id(),
    ensures
        out.0 <= requests.len(),
        out.1.is_none() ==> out.0 == requests.len(),
{
    ffi::bind_batch(qp, cq, requests.as_slice())
}

pub fn post_local_invalidate(
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    mw: &mut MemoryWindow,
    wr_id: u64,
) -> (r: Result<(), RdmaError>)
    requires
        qp.can_post(*old(cq), wr_id),
        old(mw).state() is Bound,
        old(mw).state()->Bound_qp_id == qp.id(),
        old(mw).consistent(),
        old(mw).pd_id() == qp.pd_id(),
        old(mw).node() == qp.source().node(),
    ensures
        r.is_err() ==> *final(cq) == *old(cq) && *final(mw) == *old(mw),
        r.is_ok() ==> final(cq).well_formed() && final(cq).same_queue(*old(cq))
            && final(mw).consistent() && final(mw).id() == old(mw).id() && final(mw).node() == old(
            mw,
        ).node() && final(mw).pd_id() == old(mw).pd_id() && final(mw).state() == (
        WindowState::Invalidating {
            wr_id,
            instance_id: old(mw).state()->Bound_instance_id,
            qp_id: qp.id(),
            share: old(mw).state()->Bound_share,
            slot: old(mw).state()->Bound_slot,
            access: old(mw).state()->Bound_access,
            prev_rkey: old(mw).state()->Bound_rkey,
        }) && final(cq).pending() == old(cq).pending().insert(
            wr_id,
            PendingWorkRequest {
                kind: PendingKind::LocalInvalidate { qp_id: qp.id(), window_id: old(mw).id() },
            },
        ),
{
    ffi::invalidate(qp.connection.raw.as_ptr(), mw.rkey, wr_id)?;
    record_local_invalidate(qp, cq, mw, wr_id);
    Ok(())
}

fn record_local_invalidate(
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    mw: &mut MemoryWindow,
    wr_id: u64,
)
    requires
        qp.can_post(*old(cq), wr_id),
        old(mw).state() is Bound,
        old(mw).state()->Bound_qp_id == qp.id(),
        old(mw).consistent(),
        old(mw).pd_id() == qp.pd_id(),
        old(mw).node() == qp.source().node(),
    ensures
        final(cq).well_formed() && final(cq).same_queue(*old(cq)) && final(mw).consistent()
            && final(mw).id() == old(mw).id() && final(mw).node() == old(mw).node()
            && final(mw).pd_id() == old(mw).pd_id() && final(mw).state() == (
        WindowState::Invalidating {
            wr_id,
            instance_id: old(mw).state()->Bound_instance_id,
            qp_id: qp.id(),
            share: old(mw).state()->Bound_share,
            slot: old(mw).state()->Bound_slot,
            access: old(mw).state()->Bound_access,
            prev_rkey: old(mw).state()->Bound_rkey,
        }) && final(cq).pending() == old(cq).pending().insert(
            wr_id,
            PendingWorkRequest {
                kind: PendingKind::LocalInvalidate { qp_id: qp.id(), window_id: old(mw).id() },
            },
        ),
{
    cq.in_flight_count += 1;
    proof {
        mw.state = Ghost(
            WindowState::Invalidating {
                wr_id,
                prev_rkey: mw.rkey,
                instance_id: mw.state@->Bound_instance_id,
                qp_id: qp.id(),
                share: mw.state@->Bound_share,
                slot: mw.state@->Bound_slot,
                access: mw.state@->Bound_access,
            },
        );
        cq.in_flight = Ghost(
            cq.pending().insert(
                wr_id,
                PendingWorkRequest {
                    kind: PendingKind::LocalInvalidate { qp_id: qp.id(), window_id: mw.id() },
                },
            ),
        );
    }
}

pub struct InvalidateRequest<'a> {
    pub window: MemoryWindow<'a>,
    pub wr_id: u64,
}

impl InvalidateRequest<'_> {
    pub open spec fn entry(self, qp: &QueuePair) -> PendingWorkRequest {
        PendingWorkRequest {
            kind: PendingKind::LocalInvalidate { qp_id: qp.id(), window_id: self.window.id() },
        }
    }

    pub open spec fn admissible(self, qp: &QueuePair) -> bool {
        &&& self.window.state() is Bound
        &&& self.window.state()->Bound_qp_id == qp.id()
        &&& self.window.consistent()
        &&& self.window.pd_id() == qp.pd_id()
        &&& self.window.node() == qp.source().node()
    }

    pub open spec fn invalidates(self, prior: Self, qp: &QueuePair) -> bool {
        let w = prior.window.state();
        &&& self.wr_id == prior.wr_id
        &&& self.window.id() == prior.window.id()
        &&& self.window.pd_id() == prior.window.pd_id()
        &&& self.window.node() == prior.window.node()
        &&& self.window.consistent()
        &&& self.window.state() == (WindowState::Invalidating {
            wr_id: prior.wr_id,
            instance_id: w->Bound_instance_id,
            qp_id: qp.id(),
            share: w->Bound_share,
            slot: w->Bound_slot,
            access: w->Bound_access,
            prev_rkey: w->Bound_rkey,
        })
    }
}

/// Post one signaled LOCAL_INV per window in a single doorbell. Only the
/// accepted prefix enters Invalidating. The caller retains every window.
pub fn post_invalidate_batch(
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    requests: &mut Vec<InvalidateRequest>,
) -> (out: (usize, Option<RdmaError>))
    requires
        qp.ready(),
        old(cq).well_formed(),
        old(cq).id() == qp.cq_id(),
        old(cq).observer() == qp.source().node(),
        old(cq).infos().instance_id() == qp.source().id(),
        old(cq).pending().len() + old(requests).len() <= old(cq).capacity(),
        forall|i: int|
            0 <= i < old(requests).len() ==> {
                let r = #[trigger] old(requests)[i];
                r.admissible(qp) && !old(cq).pending().contains_key(r.wr_id)
            },
        forall|i: int, j: int|
            0 <= i < j < old(requests).len() ==> (#[trigger] old(requests)[i]).wr_id != (
            #[trigger] old(requests)[j]).wr_id,
    ensures
        out.0 <= old(requests).len(),
        out.1.is_none() ==> out.0 == old(requests).len(),
        final(requests).len() == old(requests).len(),
        final(cq).well_formed(),
        final(cq).same_queue(*old(cq)),
        final(cq).pending().len() == old(cq).pending().len() + out.0,
        old(cq).pending().submap_of(final(cq).pending()),
        forall|i: int|
            0 <= i < out.0 ==> {
                &&& (#[trigger] final(requests)[i]).invalidates(old(requests)[i], qp)
                &&& final(cq).pending().contains_key(old(requests)[i].wr_id)
                &&& final(cq).pending()[old(requests)[i].wr_id] == old(requests)[i].entry(qp)
            },
        forall|i: int|
            out.0 <= i < old(requests).len() ==> (#[trigger] final(requests)[i]) == old(requests)[i]
                && !final(cq).pending().contains_key(old(requests)[i].wr_id),
{
    let (count, error) = ffi::invalidate_batch(qp, requests.as_slice());
    let mut i = 0;
    while i < count
        invariant
            qp.ready(),
            i <= count <= old(requests).len() == requests.len(),
            old(cq).pending().len() + old(requests).len() <= old(cq).capacity(),
            cq.well_formed(),
            cq.same_queue(*old(cq)),
            old(cq).id() == qp.cq_id(),
            old(cq).observer() == qp.source().node(),
            old(cq).infos().instance_id() == qp.source().id(),
            cq.pending().len() == old(cq).pending().len() + i,
            old(cq).pending().submap_of(cq.pending()),
            forall|j: int|
                0 <= j < i ==> {
                    &&& (#[trigger] requests[j]).invalidates(old(requests)[j], qp)
                    &&& cq.pending().contains_key(old(requests)[j].wr_id)
                    &&& cq.pending()[old(requests)[j].wr_id] == old(requests)[j].entry(qp)
                },
            forall|j: int|
                i <= j < requests.len() ==> (#[trigger] requests[j]) == old(requests)[j]
                    && !cq.pending().contains_key(old(requests)[j].wr_id),
            forall|j: int|
                0 <= j < old(requests).len() ==> (#[trigger] old(requests)[j]).admissible(qp),
            forall|j: int, k: int|
                0 <= j < k < old(requests).len() ==> (#[trigger] old(requests)[j]).wr_id != (
                #[trigger] old(requests)[k]).wr_id,
        decreases count - i,
    {
        let wr_id = requests[i].wr_id;
        let ghost before = cq.pending();
        let ghost prior = requests@;
        let ghost entry = old(requests)[i as int];
        assert(requests@[i as int] == entry && wr_id == entry.wr_id);
        record_local_invalidate(qp, cq, &mut requests[i].window, wr_id);
        proof {
            assert(cq.pending() == before.insert(wr_id, entry.entry(qp)));
            assert(requests@[i as int].invalidates(entry, qp));
            assert forall|j: int| 0 <= j < i + 1 implies {
                &&& (#[trigger] requests@[j]).invalidates(old(requests)[j], qp)
                &&& cq.pending().contains_key(old(requests)[j].wr_id)
                &&& cq.pending()[old(requests)[j].wr_id] == old(requests)[j].entry(qp)
            } by {
                if j < i {
                    assert(requests@[j] == prior[j]);
                    assert(old(requests)[j].wr_id != wr_id);
                }
            }
            assert forall|j: int| i < j < requests.len() implies (#[trigger] requests@[j]) == old(
                requests,
            )[j] && !cq.pending().contains_key(old(requests)[j].wr_id) by {
                assert(requests@[j] == prior[j]);
                assert(!before.contains_key(prior[j].wr_id));
                assert(old(requests)[i as int].wr_id != old(requests)[j].wr_id);
            }
        }
        i += 1;
    }
    assert forall|j: int| 0 <= j < count implies {
        &&& (#[trigger] requests@[j]).invalidates(old(requests)[j], qp)
        &&& cq.pending().contains_key(old(requests)[j].wr_id)
        &&& cq.pending()[old(requests)[j].wr_id] == old(requests)[j].entry(qp)
    } by {
        assert(j < i);
    }
    (count, error)
}

} // verus!
