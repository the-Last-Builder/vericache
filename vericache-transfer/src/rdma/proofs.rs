use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

proof fn sge_prefix_bound(slices: Seq<MemoryRegionSlice>, i: int, n: int)
    requires
        0 <= i < n <= slices.len(),
    ensures
        sge_total_len(slices, i) + slices[i].slot_ref().sub_len <= sge_total_len(slices, n),
    decreases n - i,
{
    if i + 1 < n {
        sge_prefix_bound(slices, i, n - 1);
    }
}

/// Every destination SGE slot receives the matching byte subrange from the last
/// remote write on `remote.loc` (zero byte-level quantifiers).
#[cfg(verus_only)]
pub proof fn scatter_read_returns_last_write(
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    event: TransferEvent,
    slices: Seq<MemoryRegionSlice>,
    remote: SlotRef,
    i: int,
)
    requires
        history::reachable(h),
        valid_run(h, run),
        h.events.contains(event),
        (run.succeeded)(event),
        (run.end)(event) is Some,
        slices.len() >= 2,
        sges_distinct_locs(slices),
        0 <= i < slices.len(),
        slices[i].well_formed(),
        remote.well_formed(),
        sge_total_len(slices, slices.len() as int) == remote.sub_len,
        (h.access)(event) == sge_transfer_access(true, slices, remote),
        (h.effect)(event) == sge_copy_effect(true, slices, remote),
    ensures
        ({
            let dst = slices[i].slot_ref();
            let off = sge_total_len(slices, i) as u64;
            let src_val = value_at_start(run, event, remote.loc);
            let dst_val = (run.memory)((run.end)(event).unwrap())(dst.loc);
            &&& reads_last_write(h, run, event, remote.loc)
            &&& dst_val.len() == dst.slot_len
            &&& src_val.len() >= remote.sub_offset + off + dst.sub_len
            &&& dst_val.subrange(dst.sub_offset as int, (dst.sub_offset + dst.sub_len) as int)
                == src_val.subrange(
                (remote.sub_offset + off) as int,
                (remote.sub_offset + off + dst.sub_len) as int,
            )
        }),
{
    sge_prefix_bound(slices, i, slices.len() as int);
    let dst = slices[i].slot_ref();
    let off = sge_total_len(slices, i) as u64;
    let access = sge_transfer_access(true, slices, remote);
    assert(access.reads.contains(remote.loc));
    assert(access.writes.contains(dst.loc)) by {
        assert(slices[i].slot_ref().loc == dst.loc);
    }
    read_returns_last_write(h, run, event, remote.loc);
    every_event_isolated(h, run, event);
    let input = restrict::<TransferDomain>((run.memory)((run.start)(event)), access.reads);
    let output = restrict::<TransferDomain>((run.memory)((run.end)(event).unwrap()), access.writes);
    assert(sge_copy_effect(true, slices, remote)(input, output));
    assert((copy_effect(remote.sub_slice(off, dst.sub_len), dst))(input, output));
    assert(input[remote.loc] == value_at_start(run, event, remote.loc));
    assert(output[dst.loc] == (run.memory)((run.end)(event).unwrap())(dst.loc));
}

/// A completed, successful operation in the concrete history corresponding
/// to a submitted receipt. Its output snapshot obeys that operation's effect.
#[cfg(verus_only)]
spec fn ran(
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    e: TransferEvent,
    a: Access<TransferDomain>,
    fx: Effect<TransferDomain>,
) -> bool {
    h.events.contains(e) && (run.succeeded)(e) && (run.end)(e) is Some && (h.access)(e) == a && (
    h.effect)(e) == fx
}

#[cfg(verus_only)]
proof fn copied_slot_full(
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    e: TransferEvent,
    src: SlotRef,
    dst: SlotRef,
)
    requires
        history::reachable(h),
        valid_run(h, run),
        src.is_full(),
        dst.is_full(),
        src.slot_len == dst.slot_len,
        value_at_start(run, e, src.loc).len() == src.slot_len,
        ran(h, run, e, copy_access(src, dst), copy_effect(src, dst)),
    ensures
        (run.memory)((run.end)(e).unwrap())(dst.loc) == value_at_start(run, e, src.loc),
{
    every_event_isolated(h, run, e);
    let input = restrict::<TransferDomain>(
        (run.memory)((run.start)(e)),
        copy_access(src, dst).reads,
    );
    let output = restrict::<TransferDomain>(
        (run.memory)((run.end)(e).unwrap()),
        copy_access(src, dst).writes,
    );
    assert(input[src.loc] == value_at_start(run, e, src.loc));
    copy_effect_full(src, dst, input, output);
}

#[cfg(verus_only)]
proof fn read_from_sole_writer(
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    reader: TransferEvent,
    writer: TransferEvent,
    l: SlotLoc,
)
    requires
        history::reachable(h),
        valid_run(h, run),
        h.events.contains(reader),
        (run.succeeded)(reader),
        (h.access)(reader).reads.contains(l),
        writers_before(h, reader, l) == ISet::empty().insert(writer),
    ensures
        value_at_start(run, reader, l) == (run.memory)((run.end)(writer).unwrap())(l),
{
    read_returns_last_write(h, run, reader, l);
    assert(writers_before(h, reader, l).contains(writer));
}

/// The four fresh allocations and five successful operations in the
/// cross-process GDR round trip, proved at slot granularity with zero quantifiers.
#[cfg(verus_only)]
proof fn gdr_read_round_trip_bytes(
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    host0: SlotRef,
    donor: SlotRef,
    initiator: SlotRef,
    host1: SlotRef,
    write: TransferEvent,
    h2d: TransferEvent,
    pull: TransferEvent,
    d2h: TransferEvent,
    read: TransferEvent,
    bytes: Seq<u8>,
    observed: Seq<u8>,
)
    requires
        history::reachable(h),
        valid_run(h, run),
        host0.is_full(),
        donor.is_full(),
        initiator.is_full(),
        host1.is_full(),
        host0.loc.space.is_host(),
        donor.loc.space.is_gpu(),
        initiator.loc.space.is_gpu(),
        host1.loc.space.is_host(),
        host0.loc.node == donor.loc.node,
        donor.loc.node != initiator.loc.node,
        initiator.loc.node == host1.loc.node,
        bytes.len() == host0.slot_len == donor.slot_len == initiator.slot_len == host1.slot_len
            == observed.len(),
        ran(h, run, write, write_access(host0), splice_bytes(host0, bytes)),
        ran(h, run, h2d, copy_access(host0, donor), copy_effect(host0, donor)),
        ran(h, run, pull, transfer_access(true, initiator, donor), copy_effect(donor, initiator)),
        ran(h, run, d2h, copy_access(initiator, host1), copy_effect(initiator, host1)),
        ran(h, run, read, read_access(host1), read_bytes_effect(host1, observed)),
        writers_before(h, h2d, host0.loc) == ISet::empty().insert(write),
        writers_before(h, pull, donor.loc) == ISet::empty().insert(h2d),
        writers_before(h, d2h, initiator.loc) == ISet::empty().insert(pull),
        writers_before(h, read, host1.loc) == ISet::empty().insert(d2h),
    ensures
        observed == bytes,
{
    every_event_isolated(h, run, write);
    every_event_isolated(h, run, read);
    let w_in = restrict::<TransferDomain>(
        (run.memory)((run.start)(write)),
        write_access(host0).reads,
    );
    let w_out = restrict::<TransferDomain>(
        (run.memory)((run.end)(write).unwrap()),
        write_access(host0).writes,
    );
    splice_bytes_full(host0, bytes, w_in, w_out);
    assert((run.memory)((run.end)(write).unwrap())(host0.loc) == bytes);

    read_from_sole_writer(h, run, h2d, write, host0.loc);
    copied_slot_full(h, run, h2d, host0, donor);

    read_from_sole_writer(h, run, pull, h2d, donor.loc);
    copied_slot_full(h, run, pull, donor, initiator);

    read_from_sole_writer(h, run, d2h, pull, initiator.loc);
    copied_slot_full(h, run, d2h, initiator, host1);

    read_from_sole_writer(h, run, read, d2h, host1.loc);
    let r_in = restrict::<TransferDomain>(
        (run.memory)((run.start)(read)),
        read_access(host1).reads,
    );
    let r_out = restrict::<TransferDomain>(
        (run.memory)((run.end)(read).unwrap()),
        read_access(host1).writes,
    );
    assert(r_in[host1.loc] == bytes);
    read_bytes_full(host1, observed, r_in, r_out);
}

proof fn prefix_pending_grows(
    initial: BatchState,
    rs: Seq<TransferRequest>,
    i: int,
    j: int,
    b: BatchInputs,
)
    requires
        0 <= i <= j <= rs.len(),
    ensures
        batch_prefix(initial, rs, i, b).pending.dom().subset_of(
            batch_prefix(initial, rs, j, b).pending.dom(),
        ),
    decreases j - i,
{
    if i < j {
        prefix_pending_grows(initial, rs, i, j - 1, b);
    }
}

/// Locate the producer of a pending cookie in a batch on an initially empty CQ.
proof fn batch_producer(
    initial: BatchState,
    rs: Seq<TransferRequest>,
    n: int,
    b: BatchInputs,
    id: u64,
) -> (index: int)
    requires
        0 <= n <= rs.len(),
        initial.pending.is_empty(),
        batch_prefix(initial, rs, n, b).pending.contains_key(id),
    ensures
        0 <= index < n,
        rs[index].wr_id == id,
        batch_prefix(initial, rs, n, b).pending[id].kind == (PendingKind::DataTransfer {
            event: batch_prefix(initial, rs, index, b).next,
        }),
    decreases n,
{
    if rs[n - 1].wr_id == id {
        n - 1
    } else {
        batch_producer(initial, rs, n - 1, b, id)
    }
}

proof fn reaping_preserves_pending(
    cs: Seq<Completed>,
    n: int,
    before: CqSnapshot,
    instance: InstanceId,
    id: u64,
)
    requires
        0 <= n <= cs.len(),
        completion_prefix(cs, n, before, instance).0.pending.contains_key(id),
    ensures
        before.pending.contains_key(id),
        before.pending[id] == completion_prefix(cs, n, before, instance).0.pending[id],
    decreases n,
{
    if n > 0 {
        reaping_preserves_pending(cs, n - 1, before, instance, id);
    }
}

/// Exercise a multi-WR doorbell followed by a single batched observation. Even
/// on partial posting, every later request is still absent when polling begins.
#[allow(dead_code)]
#[verifier::spinoff_prover]
fn batch_submission_and_observation(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    requests: &[TransferRequest],
    Tracked(owned): Tracked<&mut Map<SlotLoc, SlotToken>>,
    Tracked(shares): Tracked<&mut Map<SlotLoc, ShareCap>>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> Result<(BatchPostOutcome, Vec<Completed>), RdmaError>
    requires
        requests.len() >= 2,
        old(cq).pending().is_empty(),
        forall|i: int, j: int|
            0 <= i < requests@.len() && 0 <= j < (#[trigger] requests@[i]).locals@.len() ==> (
            #[trigger] requests@[i].locals@[j]).registration().gpu_device_spec().is_none(),
        batch_admitted(
            *old(qp),
            *old(cq),
            requests@,
            *old(owned),
            *old(shares),
            BatchInputs { infos: *old(infos), cited },
            instance.id(),
            *knows,
        ),
{
    hide(QueuePair::valid_sges);
    hide(sge_transfer_access);
    hide(sge_copy_effect);
    hide(closure_from);
    let ghost b = BatchInputs { infos: *infos, cited };
    let ghost initial = BatchState::of(*qp, *cq, *owned, *shares);
    let result = post_send_batch(
        qp,
        cq,
        Tracked(instance),
        requests,
        Tracked(owned),
        Tracked(shares),
        Tracked(infos),
        Tracked(knows),
        Ghost(cited),
    );
    let ghost count = result.posted@.len() as int;
    proof {
        assert forall|i: int| count <= i < requests.len() implies !cq.pending().contains_key(
            (#[trigger] requests@[i]).wr_id,
        ) by {
            prefix_pending_grows(initial, requests@, count, i, b);
            assert(batch_prefix(initial, requests@, i, b).accepts(
                requests@[i],
                b,
                *old(qp),
                instance.id(),
                *knows,
            ));
        }
    }
    let ghost before_poll = *cq;
    let completions = poll_completion_queue_batch(cq, result.posted.len(), Tracked(instance))?;
    proof {
        assert forall|i: int| 0 <= i < completions.len() implies ({
            let c = #[trigger] completions@[i];
            let before = completion_prefix(
                completions@,
                i,
                CqSnapshot::of(before_poll),
                instance.id(),
            ).0;
            let after = completion_prefix(
                completions@,
                i + 1,
                CqSnapshot::of(before_poll),
                instance.id(),
            ).0;
            &&& c.describes_snapshot(before, after, instance.id())
            &&& match c {
                Completed::Data { event, .. } | Completed::DataFailed { event, .. } => exists|
                    j: int,
                |
                    0 <= j < count && (#[trigger] result.posted@[j]).wr_id == c.id()
                        && result.posted@[j].event@ == event@,
                _ => false,
            }
        }) by {
            completion_at(
                completions@,
                i,
                completions.len() as int,
                CqSnapshot::of(before_poll),
                instance.id(),
            );
            let c = completions@[i];
            let before = completion_prefix(
                completions@,
                i,
                CqSnapshot::of(before_poll),
                instance.id(),
            ).0;
            assert(c.describes_snapshot(
                before,
                completion_prefix(
                    completions@,
                    i + 1,
                    CqSnapshot::of(before_poll),
                    instance.id(),
                ).0,
                instance.id(),
            ));
            assert(before.pending.contains_key(c.id()));
            reaping_preserves_pending(
                completions@,
                i,
                CqSnapshot::of(before_poll),
                instance.id(),
                c.id(),
            );
            let j = batch_producer(initial, requests@, count, b, c.id());
            assert(result.posted@[j].event@ == before.pending[c.id()].kind->DataTransfer_event);
        }
    }
    Ok((result, completions))
}

} // verus!
