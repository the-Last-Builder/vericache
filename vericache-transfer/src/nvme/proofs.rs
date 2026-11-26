use super::*;

verus! {

broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

/// Correspondence between a full-length CQE and its event in a concrete run.
spec fn completed_copy(
    c: Completed,
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    src: SlotRef,
    dst: SlotRef,
) -> bool {
    c matches Completed::Data { event, submitted, knows, succeeded, bytes, .. } && {
        let e = event@;
        &&& src.well_formed() && dst.well_formed() && src.is_full() && dst.is_full()
        &&& src.slot_len == bytes && dst.slot_len == bytes
        &&& submitted@.key() == e && succeeded@.event() == e
        &&& succeeded@.instance_id() == submitted@.instance_id() && knows@.instance_id()
            == submitted@.instance_id()
        &&& knows@.set().contains((e.observer, e))
        &&& h.events.contains(e) && (run.succeeded)(e) && (run.end)(e) is Some
        &&& (h.access)(e) == submitted@.value().access && submitted@.value().access == copy_access(
            src,
            dst,
        )
        &&& (h.effect)(e) == submitted@.value().effect && submitted@.value().effect == copy_effect(
            src,
            dst,
        )
        &&& forall|p| #[trigger] (h.happens_before)(p, e) == submitted@.value().preds.contains(p)
    }
}

/// The read cites the completed write. Any earlier file writers in its
/// predecessor closure therefore precede that write, even in a larger history.
/// With slot-level `Seq<u8>` values, the round-trip proof needs zero byte quantifiers.
proof fn nvme_write_read_round_trip(
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
    host: SlotRef,
    file: SlotRef,
    written: Completed,
    read: Completed,
)
    requires
        history::reachable(h),
        valid_run(h, run),
        host.loc.space.is_host(),
        file.loc.space.is_nvme(),
        completed_copy(written, h, run, host, file),
        completed_copy(read, h, run, file, host),
        value_at_start(run, written->Data_event@, host.loc).len() == host.slot_len,
        read->Data_submitted@.value().preds == written->Data_submitted@.value().preds.insert(
            written->Data_event@,
        ),
    ensures
        (run.memory)((run.end)(read->Data_event@).unwrap())(host.loc) == value_at_start(
            run,
            written->Data_event@,
            host.loc,
        ),
{
    let w = written->Data_event@;
    let r = read->Data_event@;
    every_event_isolated(h, run, w);
    every_event_isolated(h, run, r);
    history::reachable_is_well_formed(h);
    let l = file.loc;
    read_returns_last_write(h, run, r, l);
    let writers = writers_before(h, r, l);
    assert(writers.contains(w));
    assert forall|last: TransferEvent| #[trigger]
        foundation::order::is_last(h.happens_before, writers, last) implies last == w by {
        if last != w {
            assert((h.happens_before)(last, w) && (h.happens_before)(w, last));
            assert((h.happens_before)(w, w));
        }
    }
    assert(value_at_start(run, r, l) == (run.memory)((run.end)(w).unwrap())(l));
    let w_in = restrict::<TransferDomain>((run.memory)((run.start)(w)), (h.access)(w).reads);
    let w_out = restrict::<TransferDomain>(
        (run.memory)((run.end)(w).unwrap()),
        (h.access)(w).writes,
    );
    let r_in = restrict::<TransferDomain>((run.memory)((run.start)(r)), (h.access)(r).reads);
    let r_out = restrict::<TransferDomain>(
        (run.memory)((run.end)(r).unwrap()),
        (h.access)(r).writes,
    );
    copy_effect_full(host, file, w_in, w_out);
    assert(r_in[l] == w_out[l]);
    copy_effect_full(file, host, r_in, r_out);
}

} // verus!
