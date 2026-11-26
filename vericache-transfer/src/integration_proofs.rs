//! Executable examples that compose the data-plane drivers and their proofs.
use crate::context::*;
#[cfg(verus_only)]
use crate::domain::*;
use crate::host;
use crate::memory::HostSlice;
use crate::rdma::*;
#[cfg(verus_only)]
use crate::transport::{covers, share_matches, slot_matches, Statement};
use crate::transport::{ByteTransport, Channel};
#[cfg(verus_only)]
use foundation::tracked::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// Reclaim and overwrite the donor buffer without receiving a read release.
pub fn fenced_revocation_donor<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    mr: &MemoryRegion,
    mut mw: MemoryWindow,
    cpu: &HostSlice,
    host: &mut SyncContext,
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
    let Tracked(write) = host::write_slice(
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
    cpu: HostSlice,
    host: SyncContext,
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
    cpu: &HostSlice,
    host: &mut SyncContext,
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
    cpu: &HostSlice,
    host: &mut SyncContext,
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

fn receive_read<T: ByteTransport>(
    channel: &mut Channel<T>,
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    local: &MemoryRegionSlice,
    cpu: &HostSlice,
    host: &mut SyncContext,
    out: &mut [u8],
    expected: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(owned): Tracked<SlotToken>,
    recover: bool,
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
        expected@.len() == old(out)@.len(),
    ensures
        r.is_ok() ==> final(cq).same_queue(*old(cq)) && final(cq).well_formed()
            && final(cq).pending().is_empty() && final(qp).cq_id() == old(qp).cq_id()
            && final(qp).source().node() == old(qp).source().node() && final(channel).same_session(
            *old(channel),
        ) && final(host).ready(instance.id(), local.slot_ref().loc.node),
        r.is_ok() ==> final(out)@ == expected@,
        r matches Ok((_, slot, done)) ==> completed_slot(
            instance.id(),
            local.slot_ref(),
            slot@,
            done@,
        ),
        r matches Ok((status, _, _)) ==> (status != 0) == recover,
{
    let grant = match channel.recv_remote_read_grant(Tracked(instance)) {
        Ok(v) => v,
        Err(_) => return Err(RdmaError(71)),
    };
    if recover && channel.recv_completion_notice(Tracked(instance)).is_err() {
        return Err(RdmaError(71));
    }
    let Some(cap) = grant.capability else {
        return Err(RdmaError(71));
    };
    if local.len() != cap.len() {
        return Err(RdmaError(22));
    }
    let Tracked(mut s) = grant.statement;
    let Tracked(share_opt) = grant.share;
    let tracked share_cap = share_opt.tracked_unwrap();
    let ghost incoming_fr = share_cap.frontier();
    let ghost prior_facts = s.closure();
    let ghost share_id = grant.kind@->Some_0->ReadShare_share_id;
    let tracked mut remote_perm = ReadPerm::Shared(share_cap);
    let tracked mut local_tok = owned;
    proof {
        assert(ISet::<TransferEvent>::empty().union(s.facts) =~= s.facts);
    }
    let (Ghost(ev), Tracked(receipt)) = post_rdma_read(
        qp,
        cq,
        Tracked(instance),
        local,
        &cap,
        1,
        false,
        Tracked(&mut remote_perm),
        Tracked(&mut local_tok),
        Tracked(&mut s.submitted),
        Tracked(&s.knows),
        Ghost(s.facts),
    )?;
    let tracked mut successes = SucceededMap::empty(instance.id());
    let (status, submitted, knows) = match wait_one(cq, Tracked(instance))? {
        Completed::Data { submitted, knows, succeeded: _success, .. } if !recover => {
            proof {
                successes.insert(_success.get());
            }
            (0, submitted, knows)
        },
        Completed::DataFailed { status, submitted, knows, .. } if recover => (
            status,
            submitted,
            knows,
        ),
        Completed::DataFailed { status, .. } => return Err(RdmaError(status)),
        _ => return Err(RdmaError(71)),
    };
    let Tracked(observed) = submitted;
    let Tracked(known) = knows;
    let ghost one = ISet::empty().insert(ev);
    let tracked mut infos = SubmittedMap::empty(instance.id());
    proof {
        infos.insert(observed);
    }
    assert(cq.pending() =~= Map::empty());
    let result = if recover {
        let Tracked(written) = host::write_slice(
            host,
            Tracked(instance),
            cpu,
            expected,
            Tracked(&mut local_tok),
            Tracked(&infos),
            Tracked(&known),
            Ghost(one),
        );
        let tracked writes = written.infos();
        let ghost cited = ISet::empty().insert(written.submitted.key());
        let tracked mut read_perm = ReadPerm::Owned(local_tok);
        let done = host::read_slice(
            host,
            Tracked(instance),
            cpu,
            out,
            Tracked(&mut read_perm),
            Tracked(&writes),
            Tracked(&written.knows),
            Ghost(cited),
        );
        proof {
            local_tok = read_perm.into_owned();
        }
        done
    } else {
        let tracked mut read_perm = ReadPerm::Owned(local_tok);
        let done = host::read_slice(
            host,
            Tracked(instance),
            cpu,
            out,
            Tracked(&mut read_perm),
            Tracked(&infos),
            Tracked(&known),
            Ghost(one),
        );
        proof {
            local_tok = read_perm.into_owned();
        }
        done
    };
    let mut i = 0usize;
    while i < out.len()
        invariant
            i <= out.len(),
            out.len() == expected.len(),
            forall|j: int| 0 <= j < i ==> #[trigger] out@[j] == expected@[j],
        decreases out.len() - i,
    {
        if out[i] != expected[i] {
            return Err(RdmaError(74));
        }
        i += 1;
    }
    assert(out@ =~= expected@);
    if recover {
        return Ok((status, Tracked(local_tok), result));
    }
    let tracked share_cap = remote_perm.into_shared();
    let tracked s = Statement { submitted: infos, knows: known, facts: one, succeeded: successes };
    proof {
        assert(receipt.value() == observed.value());
        assert forall|x: TransferEvent| #[trigger]
            receipt.value().preds.contains(x) implies s.closure().contains(x) by {
            assert(s.submitted.map()[ev].preds.contains(x));
        }
        assert(prior_facts == receipt.value().preds);
        assert(s.closure().contains(ev));
        assert(covers(incoming_fr, prior_facts));
        assert(share_cap.frontier().readers == incoming_fr.readers.insert(ev));
        assert(share_cap.frontier().last_writer == incoming_fr.last_writer);
        assert(share_cap.frontier().readers.subset_of(s.closure()));
        if let Some(w) = incoming_fr.last_writer {
            assert(prior_facts.contains(w));
            assert(s.closure().contains(w));
        }
        assert(share_matches(share_cap, instance.id(), cap.slot_ref(), share_id, s.closure()));
    }
    if channel.send_remote_read_release(
        grant.tag,
        &grant.user_bytes,
        Tracked(s),
        Tracked(share_cap),
    ).is_err() {
        return Err(RdmaError(71));
    }
    Ok((0, Tracked(local_tok), result))
}

#[cfg(verus_only)]
/// Recover a failed read's destination using the receipt returned by poll or wait.
fn nvme_failed_io_recovery(
    Ghost(before): Ghost<crate::nvme::IoUring>,
    ring: &crate::nvme::IoUring,
    completion: crate::nvme::Completed,
    cpu: &mut SyncContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: host::HostBuffer,
    replacement: &[u8],
    Tracked(tok): Tracked<SlotToken>,
) -> (reclaimed: Tracked<Completion>)
    requires
        completion is DataFailed,
        completion.describes(before, *ring),
        before.instance_id() == instance.id(),
        old(cpu).ready(instance.id(), before.node()),
        buf.well_formed(),
        buf.num_slots() == 1,
        buf.instance_id() == instance.id(),
        buf.node() == before.node(),
        buf.slot_len() == replacement.len(),
        before.pending()[completion->DataFailed_user_data].access.writes == ISet::empty().insert(
            buf.slot_loc(0),
        ),
        slot_owned(
            tok,
            instance.id(),
            buf.slot_loc(0),
            Frontier { last_writer: Some(completion->DataFailed_event@), readers: ISet::empty() },
        ),
    ensures
        final(cpu).source@.next_seq() == old(cpu).source@.next_seq() + 2,
        reclaimed@.submitted.value().access == reclaim_access(buf.slot_loc(0)),
        reclaimed@.knows.set().contains((buf.node(), reclaimed@.submitted.key())),
{
    let crate::nvme::Completed::DataFailed { event, submitted, knows, .. } = completion else {
        unreachable!()
    };
    let tracked mut tok = tok;
    let tracked mut infos = SubmittedMap::empty(instance.id());
    proof {
        infos.insert(submitted.get());
    }
    let Tracked(written) = host::write_slice(
        cpu,
        Tracked(instance),
        &buf.as_slice(),
        replacement,
        Tracked(&mut tok),
        Tracked(&infos),
        Tracked(knows.borrow()),
        Ghost(ISet::empty().insert(event@)),
    );
    let tracked infos = written.infos();
    host::free(
        cpu,
        Tracked(instance),
        buf,
        Tracked(tok),
        Tracked(&infos),
        Tracked(&written.knows),
        Ghost(ISet::empty().insert(written.submitted.key())),
    )
}

} // verus!
