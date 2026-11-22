use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::set::group_set_lemmas,
};

/// Sealed remote destination slot token held after an RDMA WRITE until a
/// same-QP post-write `RDMA_READ` succeeds (IBTA §9.5 responder visibility).
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct PendingRemoteWrite {
    pub(super) ghost write_event: TransferEvent,
    pub(super) tracked token: SlotToken,
}

impl PendingRemoteWrite {
    pub closed spec fn write_event(self) -> TransferEvent {
        self.write_event
    }

    pub closed spec fn instance_id(self) -> InstanceId {
        self.token.instance_id()
    }

    pub closed spec fn remote_loc(self) -> SlotLoc {
        self.token.key()
    }

    pub closed spec fn frontier(self) -> Frontier<TransferDomain> {
        slot_frontier(self.token)
    }

    pub closed spec fn well_formed(self) -> bool {
        slot_is_owned(self.token, self.token.instance_id(), self.token.key())
    }
}

/// Unwrap a `PendingRemoteWrite` once a same-QP post-write `RDMA_READ` on
/// `pending.remote_loc()` has succeeded, flushing responder writes to coherence.
pub proof fn complete_remote_write_visibility(
    tracked pending: PendingRemoteWrite,
    flush_read_ev: Ghost<TransferEvent>,
    tracked infos: &SubmittedMap,
    tracked succeeded: &Succeeded,
) -> (tracked out: SlotToken)
    requires
        pending.well_formed(),
        infos.instance_id() == pending.instance_id(),
        succeeded.instance_id() == pending.instance_id(),
        succeeded.event() == flush_read_ev@,
        infos.dom().contains(flush_read_ev@),
        flush_read_ev@.source == pending.write_event().source,
        infos.map()[flush_read_ev@].preds.contains(pending.write_event()),
        infos.map()[flush_read_ev@].access.reads.contains(pending.remote_loc()),
    ensures
        slot_owned(out, pending.instance_id(), pending.remote_loc(), pending.frontier()),
{
    pending.token
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub(super) tracked struct PostedTransfer {
    pub(super) ghost access: Access<TransferDomain>,
    pub(super) ghost effect: Effect<TransferDomain>,
    pub(super) ghost after: ISet<TransferEvent>,
}

/// Submit a 2-slot RDMA transfer (1 source `ReadPerm` + 1 distinct destination `SlotToken`).
#[inline(always)]
#[cfg_attr(not(verus_only), allow(unused_variables))]
fn submit_transfer_2(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(work): Tracked<PostedTransfer>,
    wr_id: u64,
    fence: bool,
    read: bool,
    Tracked(src_perm): Tracked<&mut ReadPerm>,
    Tracked(dst_tok): Tracked<&mut SlotToken>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: (Ghost<TransferEvent>, Tracked<Submitted>))
    requires
        old(qp).can_post(*old(cq), wr_id),
        old(qp).source().id() == instance.id(),
        work.after == old(qp).fence_after(fence),
        old(infos).instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(old(infos).dom()),
        knowledge_of(old(qp).source().node(), cited).subset_of(knows.set()),
        old(src_perm).instance_id() == instance.id(),
        old(src_perm).is_owned() || old(src_perm).is_shared_wf(),
        slot_is_owned(*old(dst_tok), instance.id(), old(dst_tok).key()),
        old(src_perm).loc() != old(dst_tok).key(),
        work.access.writes == ISet::empty().insert(old(dst_tok).key()),
        work.access.reads.contains(old(src_perm).loc()),
        work.access.reads.subset_of(
            ISet::empty().insert(old(src_perm).loc()).insert(old(dst_tok).key()),
        ),
        read_admitted(
            old(src_perm).frontier(),
            old(qp).transfer_ordered(fence, *old(infos), cited),
        ),
        write_admitted(
            slot_frontier(*old(dst_tok)),
            old(qp).transfer_ordered(fence, *old(infos), cited),
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == instance.id(),
        final(src_perm).advanced_read(*old(src_perm), old(qp).source().next_event()),
        slot_owned(
            *final(dst_tok),
            instance.id(),
            old(dst_tok).key(),
            Frontier { last_writer: Some(old(qp).source().next_event()), readers: ISet::empty() },
        ),
        final(qp).qp_cq_stepped(
            *old(qp),
            *final(cq),
            *old(cq),
            work.access,
            work.effect,
            old(src_perm).read_guards(),
            wr_id,
            fence,
            read,
            *old(infos),
            cited,
            r.0@,
            r.1@,
        ),
{
    let ghost ev = qp.source().next_event();
    let tracked receipt;
    cq.in_flight_count += 1;
    proof {
        use_type_invariant(&qp.connection);
        let tracked name = qp.connection.source.borrow_mut().take_next();
        let tracked s = if fence {
            let tracked mut citations = SubmittedMap::empty(instance.id());
            vstd::modes::tracked_swap(infos, &mut citations);
            let tracked mut reads = SubmittedMap::empty(instance.id());
            vstd::modes::tracked_swap(qp.pending_read_infos.borrow_mut(), &mut reads);
            let ghost extra = citations.dom().difference(reads.dom());
            let tracked mut citations = citations.into_map();
            let tracked mut entries = reads.into_map();
            entries.tracked_union_prefer_right(citations.tracked_remove_keys(extra));
            let tracked merged = SubmittedMap::from_map(instance.id(), entries);
            assert(merged.map() =~= old(infos).map().union_prefer_right(
                old(qp).read_infos().map(),
            ));
            let tracked res = instance.submit_2(
                ev.observer,
                work.access,
                work.effect,
                work.after,
                cited,
                &merged,
                knows,
                name,
                src_perm,
                dst_tok,
            );
            let tracked mut entries = merged.into_map();
            citations.tracked_union_prefer_right(entries.tracked_remove_keys(extra));
            *infos = SubmittedMap::from_map(instance.id(), citations);
            qp.pending_read_infos = Tracked(SubmittedMap::from_map(instance.id(), entries));
            assert(infos.map() =~= old(infos).map());
            assert(qp.read_infos().map() =~= old(qp).read_infos().map());
            res
        } else {
            instance.submit_2(
                ev.observer,
                work.access,
                work.effect,
                work.after,
                cited,
                infos,
                knows,
                name,
                src_perm,
                dst_tok,
            )
        };
        receipt = s;
        cq.in_flight_infos.borrow_mut().insert(s);
        cq.in_flight = Ghost(
            cq.pending().insert(
                wr_id,
                PendingWorkRequest { kind: PendingKind::DataTransfer { event: ev } },
            ),
        );
        if read {
            qp.pending_reads = Ghost(qp.reads().insert(ev));
            qp.pending_read_infos.borrow_mut().insert(s);
        }
    }
    (Ghost(ev), Tracked(receipt))
}

/// Submit a multi-slot RDMA transfer (finite owned slot map + optional single `ShareCap`).
#[inline(always)]
#[cfg_attr(not(verus_only), allow(unused_variables))]
pub(super) fn submit_transfer_multi(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(work): Tracked<PostedTransfer>,
    wr_id: u64,
    fence: bool,
    read: bool,
    Tracked(owned): Tracked<&mut Map<SlotLoc, SlotToken>>,
    Tracked(shared): Tracked<Option<ShareCap>>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: (Ghost<TransferEvent>, Tracked<Submitted>, Tracked<Option<ShareCap>>))
    requires
        old(qp).can_post(*old(cq), wr_id),
        old(qp).source().id() == instance.id(),
        work.after == old(qp).fence_after(fence),
        old(infos).instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(old(infos).dom()),
        knowledge_of(old(qp).source().node(), cited).subset_of(knows.set()),
        owned_map_valid(*old(owned), instance.id()),
        forall|l: SlotLoc| #[trigger]
            old(owned).contains_key(l) ==> admitted(
                slot_frontier(old(owned)[l]),
                l,
                work.access,
                old(qp).transfer_ordered(fence, *old(infos), cited),
            ),
        forall|l: SlotLoc| #[trigger] work.access.writes.contains(l) ==> old(owned).contains_key(l),
        match shared {
            None => forall|l: SlotLoc| #[trigger]
                work.access.reads.contains(l) ==> old(owned).contains_key(l),
            Some(cap) => {
                &&& cap.well_formed()
                &&& cap.instance_id() == instance.id()
                &&& !old(owned).contains_key(cap.loc())
                &&& work.access.reads.contains(cap.loc())
                &&& read_admitted(
                    cap.frontier(),
                    old(qp).transfer_ordered(fence, *old(infos), cited),
                )
                &&& forall|l: SlotLoc| #[trigger]
                    work.access.reads.contains(l) ==> old(owned).contains_key(l) || l == cap.loc()
            },
        },
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == instance.id(),
        owned_map_advanced(
            *final(owned),
            *old(owned),
            instance.id(),
            old(qp).source().next_event(),
            work.access,
        ),
        match (shared, r.2@) {
            (None, None) => true,
            (Some(old_cap), Some(new_cap)) => {
                &&& new_cap.well_formed()
                &&& new_cap.instance_id() == old_cap.instance_id()
                &&& new_cap.loc() == old_cap.loc()
                &&& new_cap.share() == old_cap.share()
                &&& new_cap.is_local() == old_cap.is_local()
                &&& new_cap.is_rdma_bound() == old_cap.is_rdma_bound()
                &&& new_cap.is_unbound() == old_cap.is_unbound()
                &&& new_cap.frontier() == (Frontier {
                    last_writer: old_cap.frontier().last_writer,
                    readers: old_cap.frontier().readers.insert(old(qp).source().next_event()),
                })
            },
            _ => false,
        },
        final(qp).qp_cq_stepped(
            *old(qp),
            *final(cq),
            *old(cq),
            work.access,
            work.effect,
            match shared {
                None => IMap::empty(),
                Some(cap) => IMap::empty().insert(cap.loc(), cap.share()),
            },
            wr_id,
            fence,
            read,
            *old(infos),
            cited,
            r.0@,
            r.1@,
        ),
{
    let ghost ev = qp.source().next_event();
    let tracked receipt;
    let tracked out_shared;
    cq.in_flight_count += 1;
    proof {
        use_type_invariant(&qp.connection);
        let tracked name = qp.connection.source.borrow_mut().take_next();
        let tracked mut prev_owned = Map::tracked_empty();
        vstd::modes::tracked_swap(owned, &mut prev_owned);
        let tracked (Tracked(s), Tracked(new_owned), Tracked(new_shared)) = if fence {
            let tracked mut citations = SubmittedMap::empty(instance.id());
            vstd::modes::tracked_swap(infos, &mut citations);
            let tracked mut reads = SubmittedMap::empty(instance.id());
            vstd::modes::tracked_swap(qp.pending_read_infos.borrow_mut(), &mut reads);
            let ghost extra = citations.dom().difference(reads.dom());
            let tracked mut citations = citations.into_map();
            let tracked mut entries = reads.into_map();
            entries.tracked_union_prefer_right(citations.tracked_remove_keys(extra));
            let tracked merged = SubmittedMap::from_map(instance.id(), entries);
            assert(merged.map() =~= old(infos).map().union_prefer_right(
                old(qp).read_infos().map(),
            ));
            let tracked res = instance.submit_multi(
                ev.observer,
                work.access,
                work.effect,
                work.after,
                cited,
                &merged,
                knows,
                name,
                prev_owned,
                shared,
            );
            let tracked mut entries = merged.into_map();
            citations.tracked_union_prefer_right(entries.tracked_remove_keys(extra));
            *infos = SubmittedMap::from_map(instance.id(), citations);
            qp.pending_read_infos = Tracked(SubmittedMap::from_map(instance.id(), entries));
            assert(infos.map() =~= old(infos).map());
            assert(qp.read_infos().map() =~= old(qp).read_infos().map());
            res
        } else {
            instance.submit_multi(
                ev.observer,
                work.access,
                work.effect,
                work.after,
                cited,
                infos,
                knows,
                name,
                prev_owned,
                shared,
            )
        };
        *owned = new_owned;
        out_shared = new_shared;
        receipt = s;
        cq.in_flight_infos.borrow_mut().insert(s);
        cq.in_flight = Ghost(
            cq.pending().insert(
                wr_id,
                PendingWorkRequest { kind: PendingKind::DataTransfer { event: ev } },
            ),
        );
        if read {
            qp.pending_reads = Ghost(qp.reads().insert(ev));
            qp.pending_read_infos.borrow_mut().insert(s);
        }
    }
    (Ghost(ev), Tracked(receipt), Tracked(out_shared))
}

pub fn post_rdma_read(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    local: &MemoryRegionSlice,
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(remote_perm): Tracked<&mut ReadPerm>,
    Tracked(local_tok): Tracked<&mut SlotToken>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<(Ghost<TransferEvent>, Tracked<Submitted>), RdmaError>)
    requires
        local.registration().gpu_device_spec().is_none(),
        old(qp).read_transfer_admitted(
            *old(cq),
            instance.id(),
            *local,
            *remote,
            wr_id,
            fence,
            *old(remote_perm),
            *old(local_tok),
            *old(infos),
            *knows,
            cited,
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r.is_err() ==> *final(qp) == *old(qp) && *final(cq) == *old(cq) && *final(remote_perm)
            == *old(remote_perm) && *final(local_tok) == *old(local_tok) && *final(infos) == *old(
            infos,
        ),
        r matches Ok((e, s)) ==> {
            &&& final(remote_perm).advanced_read(*old(remote_perm), old(qp).source().next_event())
            &&& slot_owned(
                *final(local_tok),
                instance.id(),
                local.slot_ref().loc,
                Frontier {
                    last_writer: Some(old(qp).source().next_event()),
                    readers: ISet::empty(),
                },
            )
            &&& final(qp).state() is ReadyToSend
            &&& final(qp).source().well_formed()
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                copy_access(remote.slot_ref(), local.slot_ref()),
                copy_effect(remote.slot_ref(), local.slot_ref()),
                old(remote_perm).read_guards(),
                wr_id,
                fence,
                true,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    if local.len != remote.len || local.len > MAX_RDMA_MSG_LEN {
        return Err(RdmaError(22));
    }
    let Tracked(work) = ffi::transfer(qp, local, remote, wr_id, true, fence)?;
    Ok(
        submit_transfer_2(
            qp,
            cq,
            Tracked(instance),
            Tracked(work),
            wr_id,
            fence,
            true,
            Tracked(remote_perm),
            Tracked(local_tok),
            Tracked(infos),
            Tracked(knows),
            Ghost(cited),
        ),
    )
}

/// GPU-target RDMA READ: seals the destination `SlotToken` inside
/// `PendingGdrSlot` until `MemoryRegion::complete_gdr_slot` flushes BAR1 writes.
pub fn post_rdma_read_gdr(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    local: &MemoryRegionSlice,
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(remote_perm): Tracked<&mut ReadPerm>,
    Tracked(local_tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Ghost<TransferEvent>, Tracked<Submitted>, Tracked<PendingGdrSlot>),
    (RdmaError, Tracked<SlotToken>),
>)
    requires
        old(qp).read_transfer_admitted(
            *old(cq),
            instance.id(),
            *local,
            *remote,
            wr_id,
            fence,
            *old(remote_perm),
            local_tok,
            *old(infos),
            *knows,
            cited,
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r matches Err((_, ret_tok)) ==> ret_tok@ == local_tok && *final(qp) == *old(qp)
            && *final(cq) == *old(cq) && *final(remote_perm) == *old(remote_perm) && *final(infos)
            == *old(infos),
        r matches Ok((e, s, pending)) ==> {
            &&& final(remote_perm).advanced_read(*old(remote_perm), old(qp).source().next_event())
            &&& pending@.well_formed()
            &&& pending@.writer() == e@
            &&& pending@.instance_id() == instance.id()
            &&& pending@.loc() == local.slot_ref().loc
            &&& pending@.frontier() == (Frontier { last_writer: Some(e@), readers: ISet::empty() })
            &&& final(qp).state() is ReadyToSend
            &&& final(qp).source().well_formed()
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                copy_access(remote.slot_ref(), local.slot_ref()),
                copy_effect(remote.slot_ref(), local.slot_ref()),
                old(remote_perm).read_guards(),
                wr_id,
                fence,
                true,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    if local.len != remote.len || local.len > MAX_RDMA_MSG_LEN {
        return Err((RdmaError(22), Tracked(local_tok)));
    }
    let Tracked(work) = match ffi::transfer(qp, local, remote, wr_id, true, fence) {
        Ok(w) => w,
        Err(e) => return Err((e, Tracked(local_tok))),
    };
    let tracked mut local_tok = local_tok;
    let (ev, sub) = submit_transfer_2(
        qp,
        cq,
        Tracked(instance),
        Tracked(work),
        wr_id,
        fence,
        true,
        Tracked(remote_perm),
        Tracked(&mut local_tok),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    Ok((ev, sub, Tracked(PendingGdrSlot { writer: ev@, token: local_tok })))
}

/// Issue a post-write `RDMA_READ` against a `PendingRemoteWrite` (without
/// unsealing it) so its completion can prove IBTA §9.5 responder visibility.
pub fn post_rdma_write_flush_read(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    local: &MemoryRegionSlice,
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(pending): Tracked<&mut PendingRemoteWrite>,
    Tracked(local_tok): Tracked<&mut SlotToken>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<(Ghost<TransferEvent>, Tracked<Submitted>), RdmaError>)
    requires
        local.registration().gpu_device_spec().is_none(),
        old(qp).can_post(*old(cq), wr_id) && old(qp).source().id() == instance.id(),
        old(qp).effective_max_rd_atomic() > 0,
        local.well_formed() && remote.well_formed(),
        local.registration().pd_id() == old(qp).pd_id(),
        local.registration().instance_id() == instance.id(),
        local.registration().access().local_write,
        local.slot_ref().loc.node == old(qp).source().node(),
        remote.slot_ref().loc.node == old(qp).peer(),
        remote.slot_ref().loc.node != local.slot_ref().loc.node,
        local.slot_ref().sub_len == remote.slot_ref().sub_len,
        local.slot_ref().sub_len <= MAX_RDMA_MSG_LEN,
        remote.share() is None,
        old(pending).well_formed(),
        old(pending).instance_id() == instance.id(),
        old(pending).remote_loc() == remote.slot_ref().loc,
        slot_is_owned(*old(local_tok), instance.id(), local.slot_ref().loc),
        old(infos).instance_id() == instance.id() && knows.instance_id() == instance.id(),
        cited.subset_of(old(infos).dom()),
        knowledge_of(old(qp).source().node(), cited).subset_of(knows.set()),
        read_admitted(old(pending).frontier(), old(qp).transfer_ordered(fence, *old(infos), cited)),
        write_admitted(
            slot_frontier(*old(local_tok)),
            old(qp).transfer_ordered(fence, *old(infos), cited),
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r.is_err() ==> *final(qp) == *old(qp) && *final(cq) == *old(cq) && *final(pending) == *old(
            pending,
        ) && *final(local_tok) == *old(local_tok) && *final(infos) == *old(infos),
        r matches Ok((e, s)) ==> {
            &&& final(pending).well_formed()
            &&& final(pending).write_event() == old(pending).write_event()
            &&& final(pending).instance_id() == old(pending).instance_id()
            &&& final(pending).remote_loc() == old(pending).remote_loc()
            &&& final(pending).frontier() == (Frontier {
                last_writer: old(pending).frontier().last_writer,
                readers: old(pending).frontier().readers.insert(e@),
            })
            &&& slot_owned(
                *final(local_tok),
                instance.id(),
                local.slot_ref().loc,
                Frontier {
                    last_writer: Some(old(qp).source().next_event()),
                    readers: ISet::empty(),
                },
            )
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                copy_access(remote.slot_ref(), local.slot_ref()),
                copy_effect(remote.slot_ref(), local.slot_ref()),
                IMap::empty(),
                wr_id,
                fence,
                true,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    if local.len != remote.len || local.len > MAX_RDMA_MSG_LEN {
        return Err(RdmaError(22));
    }
    let Tracked(work) = ffi::transfer(qp, local, remote, wr_id, true, fence)?;
    let tracked mut remote_perm = ReadPerm::Owned(
        SlotToken::placeholder(instance.id(), remote.slot_ref().loc),
    );
    proof {
        let tracked mut tok = SlotToken::placeholder(instance.id(), remote.slot_ref().loc);
        vstd::modes::tracked_swap(&mut pending.token, &mut tok);
        remote_perm = ReadPerm::Owned(tok);
    }
    let (ev, sub) = submit_transfer_2(
        qp,
        cq,
        Tracked(instance),
        Tracked(work),
        wr_id,
        fence,
        true,
        Tracked(&mut remote_perm),
        Tracked(local_tok),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    proof {
        pending.token = remote_perm.into_owned();
    }
    Ok((ev, sub))
}

pub fn post_rdma_write(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    local: &MemoryRegionSlice,
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(local_perm): Tracked<&mut ReadPerm>,
    Tracked(remote_tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Ghost<TransferEvent>, Tracked<Submitted>, Tracked<PendingRemoteWrite>),
    (RdmaError, Tracked<SlotToken>),
>)
    requires
        old(qp).write_transfer_admitted(
            *old(cq),
            instance.id(),
            *local,
            *remote,
            wr_id,
            fence,
            *old(local_perm),
            remote_tok,
            *old(infos),
            *knows,
            cited,
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r matches Err((_, ret_tok)) ==> ret_tok@ == remote_tok && *final(qp) == *old(qp)
            && *final(cq) == *old(cq) && *final(local_perm) == *old(local_perm) && *final(infos)
            == *old(infos),
        r matches Ok((e, s, pending)) ==> {
            &&& final(local_perm).advanced_read(*old(local_perm), old(qp).source().next_event())
            &&& pending@.well_formed()
            &&& pending@.write_event() == e@
            &&& pending@.instance_id() == instance.id()
            &&& pending@.remote_loc() == remote.slot_ref().loc
            &&& pending@.frontier() == (Frontier { last_writer: Some(e@), readers: ISet::empty() })
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                copy_access(local.slot_ref(), remote.slot_ref()),
                copy_effect(local.slot_ref(), remote.slot_ref()),
                old(local_perm).read_guards(),
                wr_id,
                fence,
                false,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    if local.len != remote.len || local.len > MAX_RDMA_MSG_LEN {
        return Err((RdmaError(22), Tracked(remote_tok)));
    }
    let Tracked(work) = match ffi::transfer(qp, local, remote, wr_id, false, fence) {
        Ok(w) => w,
        Err(e) => return Err((e, Tracked(remote_tok))),
    };
    let tracked mut remote_tok = remote_tok;
    let (ev, sub) = submit_transfer_2(
        qp,
        cq,
        Tracked(instance),
        Tracked(work),
        wr_id,
        fence,
        false,
        Tracked(local_perm),
        Tracked(&mut remote_tok),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    Ok((ev, sub, Tracked(PendingRemoteWrite { write_event: ev@, token: remote_tok })))
}

pub fn post_rdma_read_sg(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    locals: &[MemoryRegionSlice],
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(remote_perm): Tracked<&mut ReadPerm>,
    Tracked(local_tokens): Tracked<&mut Map<SlotLoc, SlotToken>>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<(Ghost<TransferEvent>, Tracked<Submitted>), RdmaError>)
    requires
        forall|i: int|
            0 <= i < locals@.len() ==> (
            #[trigger] locals@[i]).registration().gpu_device_spec().is_none(),
        old(qp).read_sg_admitted(
            *old(cq),
            instance.id(),
            locals@,
            *remote,
            wr_id,
            fence,
            *old(remote_perm),
            *old(local_tokens),
            *old(infos),
            *knows,
            cited,
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r.is_err() ==> *final(qp) == *old(qp) && *final(cq) == *old(cq) && *final(remote_perm)
            == *old(remote_perm) && *final(local_tokens) == *old(local_tokens) && *final(infos)
            == *old(infos),
        r matches Ok((e, s)) ==> {
            &&& final(remote_perm).advanced_read(*old(remote_perm), old(qp).source().next_event())
            &&& owned_map_advanced(
                *final(local_tokens),
                *old(local_tokens),
                instance.id(),
                old(qp).source().next_event(),
                sge_transfer_access(true, locals@, remote.slot_ref()),
            )
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                sge_transfer_access(true, locals@, remote.slot_ref()),
                sge_copy_effect(true, locals@, remote.slot_ref()),
                old(remote_perm).read_guards(),
                wr_id,
                fence,
                true,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    let Tracked(work) = ffi::transfer_sg(qp, locals, remote, wr_id, true, fence)?;
    let ghost rloc = remote.slot_ref().loc;
    let ghost is_shared = *remote_perm is Shared;
    let tracked mut shared_opt: Option<ShareCap> = None;
    proof {
        let tracked mut prev_perm = ReadPerm::Owned(SlotToken::placeholder(instance.id(), rloc));
        vstd::modes::tracked_swap(remote_perm, &mut prev_perm);
        match prev_perm {
            ReadPerm::Shared(cap) => {
                shared_opt = Some(cap);
            },
            ReadPerm::Owned(rtok) => {
                local_tokens.tracked_insert(rloc, rtok);
            },
        }
        assert forall|l: SlotLoc| #[trigger]
            work.access.writes.contains(l) implies local_tokens.contains_key(l) by {
            let i = choose|i: int|
                0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == l;
            assert(old(local_tokens).contains_key(locals@[i].slot_ref().loc));
        }
        assert forall|l: SlotLoc| #[trigger]
            work.access.reads.contains(l) implies local_tokens.contains_key(l) || (is_shared && l
            == rloc) by {
            if l != rloc {
                let i = choose|i: int|
                    0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == l
                        && !locals@[i].slot_ref().is_full();
                assert(old(local_tokens).contains_key(locals@[i].slot_ref().loc));
            }
        }
        if !is_shared {
            assert(!work.access.writes.contains(rloc)) by {
                if work.access.writes.contains(rloc) {
                    let i = choose|i: int|
                        0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == rloc;
                    assert(locals@[i].slot_ref().loc.node == qp.source().node());
                }
            }
        }
    }
    let ghost before_submit = *local_tokens;
    let (ev, sub, Tracked(out_shared)) = submit_transfer_multi(
        qp,
        cq,
        Tracked(instance),
        Tracked(work),
        wr_id,
        fence,
        true,
        Tracked(local_tokens),
        Tracked(shared_opt),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    proof {
        if is_shared {
            *remote_perm = ReadPerm::Shared(out_shared.tracked_unwrap());
        } else {
            let tracked rtok = local_tokens.tracked_remove(rloc);
            *remote_perm = ReadPerm::Owned(rtok);
            assert(local_tokens.dom() =~= old(local_tokens).dom());
            assert(work.access.reads.contains(rloc) && !work.access.writes.contains(rloc));
            assert(remote_perm.advanced_read(*old(remote_perm), ev@));
            assert forall|l: SlotLoc| #[trigger]
                old(local_tokens).contains_key(l) implies slot_owned(
                local_tokens[l],
                instance.id(),
                l,
                advance(slot_frontier(old(local_tokens)[l]), l, ev@, work.access),
            ) by {
                assert(l != rloc);
                assert(before_submit.contains_key(l));
                assert(before_submit[l] == old(local_tokens)[l]);
            }
        }
    }
    Ok((ev, sub))
}

/// GPU-target scatter/gather RDMA READ: seals the written GPU `SlotToken`s inside
/// `PendingGdrSlots` until `MemoryRegion::complete_gdr_slots` flushes BAR1 writes.
pub fn post_rdma_read_sg_gdr(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    locals: &[MemoryRegionSlice],
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(remote_perm): Tracked<&mut ReadPerm>,
    Tracked(local_tokens): Tracked<Map<SlotLoc, SlotToken>>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Ghost<TransferEvent>, Tracked<Submitted>, Tracked<PendingGdrSlots>),
    (RdmaError, Tracked<Map<SlotLoc, SlotToken>>),
>)
    requires
        old(qp).read_sg_admitted(
            *old(cq),
            instance.id(),
            locals@,
            *remote,
            wr_id,
            fence,
            *old(remote_perm),
            local_tokens,
            *old(infos),
            *knows,
            cited,
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r matches Err((_, ret_toks)) ==> ret_toks@ == local_tokens && *final(qp) == *old(qp)
            && *final(cq) == *old(cq) && *final(remote_perm) == *old(remote_perm) && *final(infos)
            == *old(infos),
        r matches Ok((e, s, pending)) ==> {
            &&& final(remote_perm).advanced_read(*old(remote_perm), old(qp).source().next_event())
            &&& pending@.well_formed()
            &&& pending@.instance_id() == instance.id()
            &&& pending@.dom() == local_tokens.dom()
            &&& forall|l: SlotLoc| #[trigger]
                pending@.dom().contains(l) ==> pending@.writers()[l] == e@
                    && pending@.frontiers()[l] == advance(
                    slot_frontier(local_tokens[l]),
                    l,
                    e@,
                    sge_transfer_access(true, locals@, remote.slot_ref()),
                )
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                sge_transfer_access(true, locals@, remote.slot_ref()),
                sge_copy_effect(true, locals@, remote.slot_ref()),
                old(remote_perm).read_guards(),
                wr_id,
                fence,
                true,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    let Tracked(work) = match ffi::transfer_sg(qp, locals, remote, wr_id, true, fence) {
        Ok(w) => w,
        Err(e) => return Err((e, Tracked(local_tokens))),
    };
    let tracked mut local_tokens = local_tokens;
    let ghost orig_tokens = local_tokens;
    let ghost rloc = remote.slot_ref().loc;
    let ghost is_shared = *remote_perm is Shared;
    let tracked mut shared_opt: Option<ShareCap> = None;
    proof {
        let tracked mut prev_perm = ReadPerm::Owned(SlotToken::placeholder(instance.id(), rloc));
        vstd::modes::tracked_swap(remote_perm, &mut prev_perm);
        match prev_perm {
            ReadPerm::Shared(cap) => {
                shared_opt = Some(cap);
            },
            ReadPerm::Owned(rtok) => {
                local_tokens.tracked_insert(rloc, rtok);
            },
        }
        assert forall|l: SlotLoc| #[trigger]
            work.access.writes.contains(l) implies local_tokens.contains_key(l) by {
            let i = choose|i: int|
                0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == l;
            assert(orig_tokens.contains_key(locals@[i].slot_ref().loc));
        }
        assert forall|l: SlotLoc| #[trigger]
            work.access.reads.contains(l) implies local_tokens.contains_key(l) || (is_shared && l
            == rloc) by {
            if l != rloc {
                let i = choose|i: int|
                    0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == l
                        && !locals@[i].slot_ref().is_full();
                assert(orig_tokens.contains_key(locals@[i].slot_ref().loc));
            }
        }
        if !is_shared {
            assert(!work.access.writes.contains(rloc)) by {
                if work.access.writes.contains(rloc) {
                    let i = choose|i: int|
                        0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == rloc;
                    assert(locals@[i].slot_ref().loc.node == qp.source().node());
                }
            }
        }
    }
    let ghost before_submit = local_tokens;
    let (ev, sub, Tracked(out_shared)) = submit_transfer_multi(
        qp,
        cq,
        Tracked(instance),
        Tracked(work),
        wr_id,
        fence,
        true,
        Tracked(&mut local_tokens),
        Tracked(shared_opt),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    proof {
        if is_shared {
            *remote_perm = ReadPerm::Shared(out_shared.tracked_unwrap());
        } else {
            let tracked rtok = local_tokens.tracked_remove(rloc);
            *remote_perm = ReadPerm::Owned(rtok);
            assert(local_tokens.dom() =~= orig_tokens.dom());
            assert(work.access.reads.contains(rloc) && !work.access.writes.contains(rloc));
            assert(remote_perm.advanced_read(*old(remote_perm), ev@));
            assert forall|l: SlotLoc| #[trigger] orig_tokens.contains_key(l) implies slot_owned(
                local_tokens[l],
                instance.id(),
                l,
                advance(slot_frontier(orig_tokens[l]), l, ev@, work.access),
            ) by {
                assert(l != rloc);
                assert(before_submit.contains_key(l));
                assert(before_submit[l] == orig_tokens[l]);
            }
        }
    }
    let ghost writers = Map::new(local_tokens.dom(), |_l: SlotLoc| ev@);
    Ok(
        (
            ev,
            sub,
            Tracked(PendingGdrSlots { instance: instance.id(), writers, tokens: local_tokens }),
        ),
    )
}

pub fn post_rdma_write_sg(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    locals: &[MemoryRegionSlice],
    remote: &RemoteCapability,
    wr_id: u64,
    fence: bool,
    Tracked(local_tokens): Tracked<&mut Map<SlotLoc, SlotToken>>,
    Tracked(remote_tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Ghost<TransferEvent>, Tracked<Submitted>, Tracked<PendingRemoteWrite>),
    (RdmaError, Tracked<SlotToken>),
>)
    requires
        old(qp).write_sg_admitted(
            *old(cq),
            instance.id(),
            locals@,
            *remote,
            wr_id,
            fence,
            *old(local_tokens),
            remote_tok,
            *old(infos),
            *knows,
            cited,
        ),
    ensures
        final(infos).map() == old(infos).map(),
        final(infos).instance_id() == old(infos).instance_id(),
        r matches Err((_, ret_tok)) ==> ret_tok@ == remote_tok && *final(qp) == *old(qp)
            && *final(cq) == *old(cq) && *final(local_tokens) == *old(local_tokens) && *final(infos)
            == *old(infos),
        r matches Ok((e, s, pending)) ==> {
            &&& owned_map_advanced(
                *final(local_tokens),
                *old(local_tokens),
                instance.id(),
                old(qp).source().next_event(),
                sge_transfer_access(false, locals@, remote.slot_ref()),
            )
            &&& pending@.well_formed()
            &&& pending@.write_event() == e@
            &&& pending@.instance_id() == instance.id()
            &&& pending@.remote_loc() == remote.slot_ref().loc
            &&& pending@.frontier() == (Frontier { last_writer: Some(e@), readers: ISet::empty() })
            &&& final(qp).qp_cq_stepped(
                *old(qp),
                *final(cq),
                *old(cq),
                sge_transfer_access(false, locals@, remote.slot_ref()),
                sge_copy_effect(false, locals@, remote.slot_ref()),
                IMap::empty(),
                wr_id,
                fence,
                false,
                *old(infos),
                cited,
                e@,
                s@,
            )
        },
{
    hide(QueuePair::valid_sges);
    hide(sge_copy_effect);
    hide(closure_from);
    let Tracked(work) = match ffi::transfer_sg(qp, locals, remote, wr_id, false, fence) {
        Ok(w) => w,
        Err(e) => return Err((e, Tracked(remote_tok))),
    };
    let ghost rloc = remote.slot_ref().loc;
    proof {
        local_tokens.tracked_insert(rloc, remote_tok);
        assert forall|l: SlotLoc| #[trigger]
            work.access.reads.contains(l) implies local_tokens.contains_key(l) by {
            if l != rloc {
                let i = choose|i: int|
                    0 <= i < locals@.len() && (#[trigger] locals@[i].slot_ref()).loc == l;
                assert(old(local_tokens).contains_key(locals@[i].slot_ref().loc));
            }
        }
    }
    let ghost before_submit = *local_tokens;
    let (ev, sub, _) = submit_transfer_multi(
        qp,
        cq,
        Tracked(instance),
        Tracked(work),
        wr_id,
        fence,
        false,
        Tracked(local_tokens),
        Tracked(None),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    let tracked out_remote_tok;
    proof {
        let tracked rtok = local_tokens.tracked_remove(rloc);
        out_remote_tok = rtok;
        assert(local_tokens.dom() =~= old(local_tokens).dom());
        assert(before_submit.contains_key(rloc));
        assert(work.access.writes.contains(rloc));
        assert(slot_owned(
            out_remote_tok,
            instance.id(),
            rloc,
            Frontier { last_writer: Some(ev@), readers: ISet::empty() },
        ));
        assert forall|l: SlotLoc| #[trigger] old(local_tokens).contains_key(l) implies slot_owned(
            local_tokens[l],
            instance.id(),
            l,
            advance(slot_frontier(old(local_tokens)[l]), l, ev@, work.access),
        ) by {
            assert(l != rloc);
            assert(before_submit.contains_key(l));
            assert(before_submit[l] == old(local_tokens)[l]);
        }
    }
    Ok((ev, sub, Tracked(PendingRemoteWrite { write_event: ev@, token: out_remote_tok })))
}

} // verus!
