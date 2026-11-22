use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub tracked enum WindowBindGrant {
    ReadOnly(WindowBindRight),
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

pub struct RemoteCapability {
    pub(super) share: Ghost<Option<TransferShare>>,
    pub(super) addr: u64,
    pub(super) len: u64,
    pub(super) rkey: u32,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) slot: Ghost<SlotRef>,
    pub(super) access: Ghost<RemoteAccess>,
}

impl RemoteCapability {
    pub closed spec fn share(&self) -> Option<TransferShare> {
        self.share@
    }

    pub closed spec fn slot_ref(&self) -> SlotRef {
        self.slot@
    }

    pub closed spec fn access(&self) -> RemoteAccess {
        self.access@
    }

    pub closed spec fn descriptor(&self) -> (u64, u64, u32) {
        (self.addr, self.len, self.rkey)
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.len == self.slot@.sub_len && self.slot@.well_formed() && self.addr + self.len
            <= u64::MAX && self.len > 0
    }

    pub fn len(&self) -> (n: u64)
        requires
            self.well_formed(),
        ensures
            n == self.slot_ref().sub_len,
    {
        self.len
    }

    pub(crate) fn from_wire(
        addr: u64,
        len: u64,
        rkey: u32,
        Ghost(sent): Ghost<RemoteCapability>,
    ) -> (r: Self)
        requires
            sent.descriptor() == (addr, len, rkey),
        ensures
            r == sent,
    {
        Self {
            addr,
            len,
            rkey,
            share: Ghost(sent.share()),
            slot: Ghost(sent.slot_ref()),
            access: Ghost(sent.access()),
        }
    }

    /// A narrower descriptor keeps the original window key, slot identity, and access rights.
    pub fn sub_slice(&self, offset: u64, len: u64) -> (s: Self)
        requires
            self.well_formed(),
            0 < len,
            offset + len <= self.slot_ref().sub_len,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref().sub_slice(offset, len),
            s.share() == self.share(),
            s.access() == self.access(),
            s.descriptor() == ((self.descriptor().0 + offset) as u64, len, self.descriptor().2),
    {
        assert!(len > 0 && offset <= self.len && len <= self.len - offset);
        Self {
            addr: self.addr + offset,
            len,
            rkey: self.rkey,
            share: self.share,
            access: self.access,
            slot: Ghost(self.slot_ref().sub_slice(offset, len)),
        }
    }

    pub fn wire_fields(&self) -> (r: (u64, u64, u32))
        ensures
            r == self.descriptor(),
    {
        (self.addr, self.len, self.rkey)
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
    raw: *mut u8,
    pd: &'a ProtectionDomain<'a>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    window_id: Ghost<nat>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    state: Ghost<WindowState>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pending_grant: Tracked<Option<WindowBindGrant>>,
    addr: u64,
    len: u64,
    rkey: u32,
}

pub fn alloc_window<'a>(
    pd: &'a ProtectionDomain<'a>,
    Tracked(name): Tracked<EventSourceToken>,
) -> (r: Result<MemoryWindow<'a>, RdmaError>)
    requires
        name.well_formed(),
        name.source() is RdmaQueuePair,
        name.node() == pd.node(),
    ensures
        r matches Ok(w) ==> w.state() == WindowState::Unbound && w.pd_id() == pd.id() && w.node()
            == pd.node() && w.id() == name.source()->RdmaQueuePair_id,
{
    if !pd.context.supports_type_2()? {
        return Err(RdmaError(95));
    }
    let (raw, rkey) = ffi::alloc_window(pd.raw)?;
    Ok(
        MemoryWindow {
            raw,
            pd,
            window_id: Ghost(name.source()->RdmaQueuePair_id),
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
    match ffi::dealloc_window(w.raw) {
        Ok(()) => Ok(()),
        Err(e) => Err((e, w)),
    }
}

impl<'a> MemoryWindow<'a> {
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
        RemoteCapability {
            addr: self.addr,
            len: self.len,
            rkey: self.rkey,
            share: Ghost(self.state@->Bound_share),
            slot: Ghost(self.state@->Bound_slot),
            access: Ghost(self.state@->Bound_access),
        }
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
        host: HostContext,
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
    host: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    Ghost(share): Ghost<TransferShare>,
    Tracked(lent): Tracked<SlotToken>,
) -> (result: Revocation)
    requires
        old(host).ready(instance.id(), old(host).source@.node()),
        slot_is_lent_with(lent, instance.id(), lent.key(), share),
    ensures
        final(host).stepped(*old(host)),
        revocation_result(result, instance.id(), lent, share, old(host).source@.next_event()),
{
    let ghost event = host.source@.next_event();
    let tracked name = host.source.borrow_mut().take_next();
    let tracked (Tracked(submitted), Tracked(revoked), Tracked(slot)) = instance.revoke_slot(
        share,
        lent,
        name,
    );
    let tracked mut infos = SubmittedMap::empty(instance.id());
    proof {
        infos.insert(submitted);
    }
    let ghost one = ISet::empty().insert(event);
    let tracked knows = instance.observe(event.observer, one, &infos);
    proof {
        assert(closure_from(infos.map(), one) =~= one);
    }
    (Tracked(Completion { submitted, knows }), Tracked(revoked), Tracked(slot))
}

/// IBTA-strict unilateral MW revocation: deallocates the invalidated MW (`ibv_dealloc_mw` == 0,
/// IBTA C10-67..70) before minting the `Revocation` fence witness.
pub fn revoke_read_share_with_window<'a>(
    host: &mut HostContext,
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
    if let Err(e) = ffi::dealloc_window(mw.raw) {
        return Err((e, mw, Tracked(witness), Tracked(lent)));
    }
    let ghost share = mw.state().binding().share->Some_0;
    Ok(perform_revocation(host, Tracked(instance), Ghost(share), Tracked(lent)))
}

/// Unilateral MW revocation after resetting the bound QP (`IBV_QPS_RESET`) and deallocating
/// the window (`ibv_dealloc_mw` == 0).
pub fn revoke_read_share_on_reset_qp<'a>(
    host: &mut HostContext,
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
    if let Err(e) = ffi::dealloc_window(mw.raw) {
        return Err((e, mw, Tracked(lent)));
    }
    let ghost share = mw.state().binding().share->Some_0;
    Ok(perform_revocation(host, Tracked(instance), Ghost(share), Tracked(lent)))
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
    if let Err(e) = ffi::dealloc_window(mw.raw) {
        return Err((e, mw, Tracked(returned)));
    }
    let tracked out = if mw.state@ is Binding {
        mw.pending_grant.get()
    } else {
        None
    };
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
    let mr = slice.mr;
    let addr = slice.addr;
    let len = slice.len;
    if len == 0 || next_rkey != mw.next_rkey() || !mr.flags.mw_bind || (access
        == RemoteAccess::ReadWrite && !mr.flags.local_write) || mr.pd.raw != qp.connection.pd.raw
        || mw.pd.raw != mr.pd.raw {
        return Err((RdmaError(22), Tracked(grant)));
    }
    if let Err(e) = ffi::bind(
        qp.connection.raw,
        mw.raw,
        mr.raw,
        addr,
        len,
        next_rkey,
        access,
        wr_id,
    ) {
        return Err((e, Tracked(grant)));
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
    Ok(())
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
    ffi::invalidate(qp.connection.raw, mw.rkey, wr_id)?;
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
    Ok(())
}

} // verus!
