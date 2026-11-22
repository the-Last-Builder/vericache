use super::proofs::receive_read;
use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};
// The example endpoints use a dedicated CQ, so the next completion belongs
// to the one outstanding operation. Bound the poll loop for the executable test.

pub(super) fn wait_one(
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
) -> (r: Result<Completed, RdmaError>)
    requires
        old(cq).well_formed(),
        old(cq).infos().instance_id() == instance.id(),
    ensures
        final(cq).well_formed(),
        final(cq).same_queue(*old(cq)),
        r matches Ok(c) ==> c.describes(*old(cq), *final(cq), instance.id()),
{
    let mut left = 10_000_000u32;
    while left > 0
        invariant
            *cq == *old(cq),
            cq.well_formed(),
            cq.infos().instance_id() == instance.id(),
        decreases left,
    {
        if let PollStatus::Completed(c) = poll_completion_queue(cq, Tracked(instance))? {
            if let Completed::ControlFailed { status, .. }
            | Completed::BindFailed { status, .. } = c {
                return Err(RdmaError(status));
            }
            return Ok(c);
        }
        left -= 1;
    }
    Err(RdmaError(110))
}

type WriterKnowledge = foundation::tracked::HistoryTokens::knows<TransferDomain>;

pub open spec fn can_grant_read_window<T: ByteTransport>(
    channel: Channel<T>,
    qp: QueuePair,
    cq: CompletionQueue,
    mr: MemoryRegion,
    mw: MemoryWindow,
    instance: InstanceId,
    owned: SlotToken,
    writer: Completion,
    names: ShareSource,
    tag: u64,
) -> bool {
    &&& names.well_formed() && names.id() == instance && names.node() == mr.node()
    &&& qp.can_post(cq, 1)
    &&& cq.pending().is_empty()
    &&& qp.source().id() == instance
    &&& channel.local() == qp.source().node()
    &&& channel.peer() == qp.peer()
    &&& channel.id() == instance
    &&& !channel.sent().contains_key(tag)
    &&& mr.well_formed()
    &&& mr.num_slots() == 1
    &&& mr.pd_id() == qp.pd_id()
    &&& mr.access().mw_bind
    &&& mr.node() == qp.source().node()
    &&& mr.instance_id() == instance
    &&& mr.slot_len() > 0
    &&& mw.state() == WindowState::Unbound
    &&& mw.pd_id() == qp.pd_id()
    &&& mw.node() == qp.source().node()
    &&& slot_owned(
        owned,
        instance,
        mr.slot_loc(0),
        Frontier { last_writer: Some(writer.submitted.key()), readers: ISet::empty() },
    )
    &&& writer.submitted.instance_id() == instance
    &&& writer.knows.instance_id() == instance
    &&& writer.knows.set().contains((mr.node(), writer.submitted.key()))
}

proof fn prepare_grant_statement(
    id: InstanceId,
    slot: SlotRef,
    share: TransferShare,
    tracked writer: Completion,
    tracked share_cap: &ShareCap,
) -> (tracked result: (Tracked<Statement>, Tracked<WriterKnowledge>))
    requires
        writer.submitted.instance_id() == id,
        writer.knows.instance_id() == id,
        writer.knows.set().contains((slot.loc.node, writer.submitted.key())),
        share_cap.is_rdma_bound(),
        share_cap.instance_id() == id,
        share_cap.loc() == slot.loc,
        share_cap.share() == share,
        share_cap.frontier() == (Frontier {
            last_writer: Some(writer.submitted.key()),
            readers: ISet::empty(),
        }),
    ensures
        result.0@.valid(id, slot.loc.node),
        share_matches(*share_cap, id, slot, share, result.0@.closure()),
        result.1@.instance_id() == id,
        result.1@.element() == (slot.loc.node, writer.submitted.key()),
{
    let event = writer.submitted.key();
    let observer = slot.loc.node;
    let tracked infos = writer.infos();
    let tracked mut known = writer.knows;
    let tracked prior = known.remove((observer, event));
    known.insert(prior);
    let tracked s = Statement {
        succeeded: SucceededMap::empty(id),
        submitted: infos,
        knows: known,
        facts: ISet::empty().insert(event),
    };
    assert(s.facts.contains(event));
    assert(s.closure().contains(event));
    assert(knowledge_of(observer, s.facts).subset_of(s.knows.set()));
    assert(s.valid(id, observer));
    assert(share_matches(*share_cap, id, slot, share, s.closure()));
    (Tracked(s), Tracked(prior))
}

fn grant_read_window<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    mr: &MemoryRegion,
    mw: &mut MemoryWindow,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(owned): Tracked<SlotToken>,
    Tracked(writer): Tracked<Completion>,
    Tracked(names): Tracked<&mut ShareSource>,
    tag: u64,
    user: &[u8],
) -> (r: Result<(Tracked<SlotToken>, Tracked<WriterKnowledge>), RdmaError>)
    requires
        can_grant_read_window(
            *old(channel),
            *qp,
            *old(cq),
            *mr,
            *old(mw),
            instance.id(),
            owned,
            writer,
            *old(names),
            tag,
        ),
    ensures
        final(names).well_formed(),
        final(names).id() == old(names).id(),
        final(names).node() == old(names).node(),
        final(channel).same_session(*old(channel)),
        r matches Ok((lent, prior)) ==> {
            let share = old(names).next_share();
            &&& final(channel).sent() == old(channel).sent().insert(
                tag,
                crate::transport::GrantKind::ReadShare { slot: mr.slot_ref(0), share_id: share },
            )
            &&& final(cq).same_queue(*old(cq)) && final(cq).well_formed()
                && final(cq).pending().is_empty()
            &&& final(mw).consistent() && final(mw).state() is Bound
                && final(mw).state()->Bound_slot == mr.slot_ref(0)
            &&& final(mw).state()->Bound_share == Some(share) && final(mw).state()->Bound_qp_id
                == qp.id()
            &&& final(mw).state()->Bound_instance_id == instance.id() && final(mw).pd_id()
                == qp.pd_id() && final(mw).node() == mr.node()
            &&& slot_lent(
                lent@,
                instance.id(),
                mr.slot_loc(0),
                Frontier { last_writer: Some(writer.submitted.key()), readers: ISet::empty() },
                ISet::empty().insert(share),
            )
            &&& prior@.instance_id() == instance.id() && prior@.element() == (
                mr.node(),
                writer.submitted.key(),
            )
        },
{
    let ghost slot = mr.slot_ref(0);
    let ghost event = writer.submitted.key();
    let ghost observer = mr.node();
    let ghost share = names.next_share();
    let tracked fresh = names.allocate();
    let tracked (Tracked(lent), Tracked(mut share_cap)) = instance.lend_slot(owned, fresh);
    let tracked right = share_cap.take_window_bind_right();
    let key = mw.next_rkey();
    let slice = mr.as_slice();
    if let Err((e, _)) = post_bind_window(
        qp,
        cq,
        &slice,
        mw,
        key,
        RemoteAccess::ReadOnly,
        Tracked(WindowBindGrant::ReadOnly(right)),
        1,
    ) {
        return Err(e);
    }
    let Completed::WindowBound { witness: bind_witness, .. } = wait_one(
        cq,
        Tracked(instance),
    )? else {
        return Err(RdmaError(71));
    };
    mw.finish_bind(bind_witness);
    let cap = mw.remote_capability();
    let tracked (Tracked(s), Tracked(prior)) = prepare_grant_statement(
        instance.id(),
        slot,
        share,
        writer,
        &share_cap,
    );
    proof {
        assert(cap.slot_ref() == slot);
    }
    if channel.send_remote_read_grant(tag, &cap, user, Tracked(s), Tracked(share_cap)).is_err() {
        return Err(RdmaError(71));
    }
    assert(cq.pending() =~= Map::empty());
    Ok((Tracked(lent), Tracked(prior)))
}

/// Citing this completion admits the next access to the returned slot.
pub open spec fn completed_slot(
    instance: InstanceId,
    slot: SlotRef,
    tok: SlotToken,
    done: Completion,
) -> bool {
    let event = done.submitted.key();
    &&& done.submitted.instance_id() == instance && done.knows.instance_id() == instance
    &&& done.knows.set().contains((slot.loc.node, event))
    &&& slot_matches(tok, instance, slot, done.submitted.value().preds.insert(event))
}

/// Donor half of §7.4, starting immediately after a synchronous host write.
/// Return the deregistration receipt and the slot's ownership token.
pub fn read_lifecycle_donor<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    mr: MemoryRegion,
    mw: &mut MemoryWindow,
    host: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(owned): Tracked<SlotToken>,
    Tracked(writer): Tracked<Completion>,
    Tracked(names): Tracked<&mut ShareSource>,
    tag: u64,
    user: &[u8],
) -> (r: Result<(Tracked<Completion>, Tracked<SlotToken>), RdmaError>)
    requires
        can_grant_read_window(
            *old(channel),
            *qp,
            *old(cq),
            mr,
            *old(mw),
            instance.id(),
            owned,
            writer,
            *old(names),
            tag,
        ),
        old(host).ready(instance.id(), mr.node()),
    ensures
        final(names).well_formed(),
        final(names).id() == old(names).id(),
        final(names).node() == old(names).node(),
        r.is_ok() ==> final(mw).state() == WindowState::Unbound && final(cq).pending().is_empty()
            && final(channel).sent() == old(channel).sent() && final(cq).same_queue(*old(cq))
            && final(cq).well_formed() && final(mw).pd_id() == old(mw).pd_id() && final(mw).node()
            == old(mw).node() && final(channel).same_session(*old(channel)) && final(host).ready(
            instance.id(),
            mr.node(),
        ),
        r matches Ok((done, slot)) ==> completed_slot(instance.id(), mr.slot_ref(0), slot@, done@),
{
    let ghost slot_ref = mr.slot_ref(0);
    let ghost event = writer.submitted.key();
    let ghost share = names.next_share();
    let ghost ids = ISet::empty().insert(share);
    let tracked original_write = writer.submitted;
    let (Tracked(lent), Tracked(prior)) = grant_read_window(
        channel,
        qp,
        cq,
        &mr,
        mw,
        Tracked(instance),
        Tracked(owned),
        Tracked(writer),
        Tracked(names),
        tag,
        user,
    )?;
    let returned = match channel.recv_remote_read_release(Tracked(instance)) {
        Ok(v) => v,
        Err(_) => return Err(RdmaError(71)),
    };
    if returned.tag != tag {
        return Err(RdmaError(71));
    }
    let Tracked(share_opt) = returned.share;
    let tracked share_cap = share_opt.tracked_unwrap();
    let Tracked(mut statement) = returned.statement;
    let ghost release = statement.closure();
    let ghost received_infos = statement.submitted.map();
    let ghost cited = statement.facts.insert(event);
    let tracked (Tracked(slot_tok), Tracked(returned_share)) = instance.give_back_slot(
        lent,
        share_cap,
    );
    proof {
        if !statement.submitted.dom().contains(event) {
            statement.submitted.insert(original_write);
        }
        statement.knows.insert(prior);
        assert forall|x: TransferEvent| #[trigger] release.contains(x) implies closure_from(
            statement.submitted.map(),
            cited,
        ).contains(x) by {
            assert forall|member: TransferEvent|
                statement.facts.contains(member)
                    && #[trigger] received_infos[member].preds.contains(x) implies closure_from(
                statement.submitted.map(),
                cited,
            ).contains(x) by {
                assert(statement.submitted.map()[member].preds.contains(x));
            }
        }
        assert(ids.remove(share) =~= ISet::<TransferShare>::empty());
        assert(covers(slot_frontier(slot_tok), closure_from(statement.submitted.map(), cited)));
    }
    post_local_invalidate(qp, cq, mw, 2)?;
    let Completed::WindowInvalidated { witness: inv_witness, .. } = wait_one(
        cq,
        Tracked(instance),
    )? else {
        return Err(RdmaError(71));
    };
    mw.finish_invalidate(inv_witness, Tracked(Some(returned_share)));
    proof {
        assert(cq.pending() =~= Map::empty());
        assert(channel.sent() =~= old(channel).sent());
    }
    match deregister_memory(
        host,
        Tracked(instance),
        mr,
        Tracked(slot_tok),
        Tracked(&statement.submitted),
        Tracked(&statement.knows),
        Ghost(cited),
    ) {
        Ok((done, out_tok)) => {
            assert(completed_slot(instance.id(), slot_ref, out_tok@, done@));
            Ok((done, out_tok))
        },
        Err((e, _, _)) => Err(e),
    }
}

/// Reclaim and overwrite the donor buffer without receiving a read release.
pub fn fenced_revocation_donor<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    mr: &MemoryRegion,
    mut mw: MemoryWindow,
    cpu: &crate::host::HostSlice,
    host: &mut HostContext,
    replacement: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(owned): Tracked<SlotToken>,
    Tracked(writer): Tracked<Completion>,
    Tracked(names): Tracked<&mut ShareSource>,
    tag: u64,
) -> (r: Result<(Tracked<Completion>, Tracked<SlotToken>), RdmaError>)
    requires
        can_grant_read_window(
            *old(channel),
            *qp,
            *old(cq),
            *mr,
            mw,
            instance.id(),
            owned,
            writer,
            *old(names),
            tag,
        ),
        old(host).ready(instance.id(), mr.node()),
        cpu.well_formed(),
        cpu.slot_ref() == mr.slot_ref(0),
        cpu.instance_id() == instance.id(),
        replacement@.len() == mr.slot_len(),
    ensures
        final(names).well_formed(),
        final(names).id() == old(names).id(),
        final(names).node() == old(names).node(),
        r matches Ok((write, slot)) ==> {
            let event = write@.submitted.key();
            &&& write@.submitted.instance_id() == instance.id() && write@.knows.instance_id()
                == instance.id()
            &&& write@.knows.set().contains((mr.node(), event))
            &&& write@.submitted.value().effect == splice_bytes(mr.slot_ref(0), replacement@)
            &&& slot_owned(
                slot@,
                instance.id(),
                mr.slot_loc(0),
                Frontier { last_writer: Some(event), readers: ISet::empty() },
            )
            &&& final(cq).well_formed() && final(cq).pending().is_empty()
                && final(channel).same_session(*old(channel))
            &&& final(host).ready(instance.id(), mr.node())
        },
{
    let ghost share = names.next_share();
    let tracked original_write = writer.submitted;
    let (Tracked(lent), Tracked(prior)) = grant_read_window(
        channel,
        qp,
        cq,
        mr,
        &mut mw,
        Tracked(instance),
        Tracked(owned),
        Tracked(writer),
        Tracked(names),
        tag,
        &[],
    )?;
    post_local_invalidate(qp, cq, &mut mw, 2)?;
    let Completed::WindowInvalidated { witness: inv_witness, .. } = wait_one(
        cq,
        Tracked(instance),
    )? else {
        return Err(RdmaError(71));
    };
    let ghost fence_event = host.source@.next_event();
    let (Tracked(fence), _, Tracked(mut slot)) = match revoke_read_share_with_window(
        host,
        Tracked(instance),
        mw,
        inv_witness,
        Tracked(lent),
    ) {
        Ok(rev) => rev,
        Err((e, _, _, _)) => return Err(e),
    };
    let tracked mut infos = fence.infos();
    let tracked mut known = fence.knows;
    let ghost cited = ISet::empty().insert(writer.submitted.key()).insert(fence_event);
    proof {
        infos.insert(original_write);
        known.insert(prior);
        assert(ISet::empty().insert(share).remove(share) =~= ISet::<TransferShare>::empty());
    }
    let Tracked(write) = crate::host::write_slice(
        host,
        Tracked(instance),
        cpu,
        replacement,
        Tracked(&mut slot),
        Tracked(&infos),
        Tracked(&known),
        Ghost(cited),
    );
    let tracked written_infos = write.infos();
    let ghost event = write.submitted.key();
    let tracked mut known = write.knows;
    let tracked fact = known.remove((mr.node(), event));
    let tracked mut notice = Knows::empty(instance.id());
    proof {
        known.insert(fact);
        notice.insert(fact);
    }
    let tracked statement = Statement {
        succeeded: SucceededMap::empty(instance.id()),
        submitted: written_infos,
        facts: ISet::empty().insert(event),
        knows: notice,
    };
    if channel.send_completion_notice(&[], Tracked(statement)).is_err() {
        return Err(RdmaError(71));
    }
    assert(cq.pending() =~= Map::empty());
    Ok((Tracked(Completion { submitted: write.submitted, knows: known }), Tracked(slot)))
}

pub open spec fn can_receive_read<T: ByteTransport>(
    channel: Channel<T>,
    qp: QueuePair,
    cq: CompletionQueue,
    local: MemoryRegionSlice,
    cpu: crate::host::HostSlice,
    host: HostContext,
    len: nat,
    instance: InstanceId,
    owned: SlotToken,
) -> bool {
    &&& qp.can_post(cq, 1)
    &&& qp.effective_max_rd_atomic() > 0
    &&& cq.pending().is_empty()
    &&& qp.source().id() == instance
    &&& channel.local() == qp.source().node()
    &&& channel.peer() == qp.peer()
    &&& channel.local() != channel.peer()
    &&& channel.id() == instance
    &&& local.well_formed()
    &&& local.slot_ref().sub_len <= MAX_RDMA_MSG_LEN
    &&& local.registration().gpu_device_spec().is_none()
    &&& local.slot_ref().loc.node == qp.source().node()
    &&& local.registration().pd_id() == qp.pd_id()
    &&& local.registration().access().local_write
    &&& local.registration().instance_id() == instance
    &&& cpu.well_formed()
    &&& cpu.instance_id() == instance
    &&& cpu.slot_ref() == local.slot_ref()
    &&& host.ready(instance, local.slot_ref().loc.node)
    &&& len == local.slot_ref().sub_len
    &&& slot_owned(owned, instance, local.slot_ref().loc, untouched())
}

/// Initiator half. Polling proves the data receipt, which the release carries
/// back to the donor. The application can inspect the copied bytes before release.
pub fn read_lifecycle_initiator<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    local: &MemoryRegionSlice,
    cpu: &crate::host::HostSlice,
    host: &mut HostContext,
    out: &mut [u8],
    expected: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(owned): Tracked<SlotToken>,
) -> (r: Result<(Tracked<SlotToken>, Tracked<Completion>), RdmaError>)
    requires
        can_receive_read(
            *old(channel),
            *old(qp),
            *old(cq),
            *local,
            *cpu,
            *old(host),
            old(out)@.len(),
            instance.id(),
            owned,
        ),
        expected@.len() == old(out)@.len(),
    ensures
        r matches Ok((slot, done)) ==> completed_slot(
            instance.id(),
            local.slot_ref(),
            slot@,
            done@,
        ),
        r.is_ok() ==> final(cq).same_queue(*old(cq)) && final(cq).well_formed()
            && final(cq).pending().is_empty() && final(qp).cq_id() == old(qp).cq_id()
            && final(qp).source().node() == old(qp).source().node() && final(channel).same_session(
            *old(channel),
        ) && final(host).ready(instance.id(), local.slot_ref().loc.node),
        r.is_ok() ==> final(out)@ == expected@,
{
    let (_, slot, done) = receive_read(
        channel,
        qp,
        cq,
        local,
        cpu,
        host,
        out,
        expected,
        Tracked(instance),
        Tracked(owned),
        false,
    )?;
    Ok((slot, done))
}

/// Wait for retirement, observe the rejected read, overwrite the destination and check its bytes.
pub fn fenced_revocation_initiator<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    local: &MemoryRegionSlice,
    cpu: &crate::host::HostSlice,
    host: &mut HostContext,
    out: &mut [u8],
    sentinel: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(owned): Tracked<SlotToken>,
) -> (r: Result<(i32, Tracked<SlotToken>, Tracked<Completion>), RdmaError>)
    requires
        can_receive_read(
            *old(channel),
            *old(qp),
            *old(cq),
            *local,
            *cpu,
            *old(host),
            old(out)@.len(),
            instance.id(),
            owned,
        ),
        sentinel@.len() == old(out)@.len(),
    ensures
        r.is_ok() ==> final(cq).same_queue(*old(cq)) && final(cq).well_formed()
            && final(cq).pending().is_empty() && final(qp).cq_id() == old(qp).cq_id()
            && final(qp).source().node() == old(qp).source().node() && final(channel).same_session(
            *old(channel),
        ) && final(host).ready(instance.id(), local.slot_ref().loc.node),
        r.is_ok() ==> final(out)@ == sentinel@,
        r matches Ok((_, slot, done)) ==> completed_slot(
            instance.id(),
            local.slot_ref(),
            slot@,
            done@,
        ),
        r matches Ok((status, _, _)) ==> status != 0,
{
    receive_read(
        channel,
        qp,
        cq,
        local,
        cpu,
        host,
        out,
        sentinel,
        Tracked(instance),
        Tracked(owned),
        true,
    )
}

} // verus!
