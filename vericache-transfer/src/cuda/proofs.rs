use super::*;

verus! {

broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};
// Submitted receipts are persistent. The vstd token-map container is linear.
// Keep the finite receipts alongside it so a shared borrow can copy a snapshot.

pub(super) open spec fn receipt_values(receipts: Seq<Submitted>, n: nat) -> IMap<
    TransferEvent,
    EventInfo<TransferDomain>,
>
    recommends
        n <= receipts.len(),
    decreases n,
{
    if n == 0 {
        IMap::empty()
    } else {
        let receipt = receipts[n - 1];
        receipt_values(receipts, (n - 1) as nat).insert(receipt.key(), receipt.value())
    }
}

pub(super) open spec fn valid_receipts(receipts: Seq<Submitted>, infos: SubmittedMap) -> bool {
    &&& receipt_values(receipts, receipts.len()) == infos.map()
    &&& forall|i: int|
        0 <= i < receipts.len() ==> #[trigger] receipts[i].instance_id() == infos.instance_id()
}

pub(super) proof fn append_receipt_values(left: Seq<Submitted>, right: Seq<Submitted>, n: nat)
    requires
        n <= right.len(),
    ensures
        receipt_values(left + right.take(n as int), left.len() + n) == receipt_values(
            left,
            left.len(),
        ).union_prefer_right(receipt_values(right, n)),
    decreases n,
{
    broadcast use vstd::seq::group_seq_axioms;

    if n == 0 {
        assert(left + right.take(0) =~= left);
        assert(receipt_values(left, left.len()).union_prefer_right(IMap::empty())
            =~= receipt_values(left, left.len()));
    } else {
        append_receipt_values(left, right, (n - 1) as nat);
        let shorter = left + right.take(n - 1);
        let longer = left + right.take(n as int);
        same_prefix(shorter, longer, left.len() + (n - 1) as nat);
        assert(longer[left.len() + n - 1] == right[n - 1]);
        assert(receipt_values(longer, left.len() + n) =~= receipt_values(
            left,
            left.len(),
        ).union_prefer_right(receipt_values(right, n)));
    }
}

proof fn same_prefix(a: Seq<Submitted>, b: Seq<Submitted>, n: nat)
    requires
        n <= a.len(),
        n <= b.len(),
        forall|i: int| 0 <= i < n ==> #[trigger] a[i] == b[i],
    ensures
        receipt_values(a, n) == receipt_values(b, n),
    decreases n,
{
    if n > 0 {
        same_prefix(a, b, (n - 1) as nat);
    }
}

pub(super) proof fn append_receipts(
    tracked src: &Seq<Submitted>,
    n: nat,
    tracked dst: &mut Seq<Submitted>,
    tracked infos: &mut SubmittedMap,
)
    requires
        n <= src.len(),
        forall|i: int| 0 <= i < n ==> #[trigger] src[i].instance_id() == old(infos).instance_id(),
    ensures
        *final(dst) == *old(dst) + src.take(n as int),
        final(infos).instance_id() == old(infos).instance_id(),
        final(infos).map() == old(infos).map().union_prefer_right(receipt_values(*src, n)),
    decreases n,
{
    broadcast use vstd::seq::group_seq_axioms;

    if n > 0 {
        append_receipts(src, (n - 1) as nat, dst, infos);
        let tracked receipt = *src.tracked_borrow(n - 1);
        dst.tracked_push(receipt);
        infos.insert(receipt);
        assert(*dst =~= *old(dst) + src.take(n as int));
    } else {
        assert(*dst =~= *old(dst) + src.take(0));
    }
    assert(infos.map() =~= old(infos).map().union_prefer_right(receipt_values(*src, n)));
}

/// Event re-recording race scenario: A and B both read the same slot. A separate
/// FIFO overwrites it after waiting for B. No host completion is needed to enqueue W.
#[allow(dead_code)]
fn ordered_handoff_and_rerecord(
    producer: &mut CudaStream,
    consumer: &mut CudaStream,
    snapshot: &mut CudaEvent,
    handoff: &mut CudaEvent,
    region: &CudaSlice,
    first: &CudaSlice,
    second: &CudaSlice,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(region_tok): Tracked<SlotToken>,
    Tracked(first_tok): Tracked<SlotToken>,
    Tracked(second_tok): Tracked<SlotToken>,
) -> Result<(), CudaError>
    requires
        old(producer).well_formed(),
        old(consumer).well_formed(),
        old(producer).source().id() == instance.id(),
        old(consumer).source().id() == instance.id(),
        old(producer).source().node() == region.slot_ref().loc.node,
        old(consumer).source().node() == region.slot_ref().loc.node,
        old(producer).device() == old(consumer).device(),
        old(producer).tail().is_empty(),
        old(consumer).tail().is_empty(),
        old(snapshot).well_formed(),
        old(handoff).well_formed(),
        old(snapshot).device() == old(producer).device(),
        old(handoff).device() == old(producer).device(),
        old(snapshot).observer() == region.slot_ref().loc.node,
        old(handoff).observer() == region.slot_ref().loc.node,
        old(snapshot).infos().instance_id() == instance.id(),
        old(handoff).infos().instance_id() == instance.id(),
        region.well_formed(),
        first.well_formed(),
        second.well_formed(),
        region.kind() == CudaPointerKind::PinnedHost,
        first.kind() == CudaPointerKind::Device(old(producer).device()),
        second.kind() == CudaPointerKind::Device(old(producer).device()),
        region.instance_id() == instance.id(),
        first.instance_id() == instance.id(),
        second.instance_id() == instance.id(),
        region.slot_ref().sub_len == first.slot_ref().sub_len,
        region.slot_ref().sub_len == second.slot_ref().sub_len,
        region.slot_ref().loc.node == first.slot_ref().loc.node,
        region.slot_ref().loc.node == second.slot_ref().loc.node,
        region.slot_ref().loc != first.slot_ref().loc,
        region.slot_ref().loc != second.slot_ref().loc,
        slot_owned(region_tok, instance.id(), region.slot_ref().loc, untouched()),
        slot_owned(first_tok, instance.id(), first.slot_ref().loc, untouched()),
        slot_owned(second_tok, instance.id(), second.slot_ref().loc, untouched()),
{
    let ghost empty = ISet::<TransferEvent>::empty();
    let ghost observer = region.slot_ref().loc.node;
    let tracked none = Knows::empty(instance.id());
    let tracked mut citations = SubmittedMap::empty(instance.id());
    let tracked mut region_perm = ReadPerm::Owned(region_tok);
    let tracked mut first_tok = first_tok;
    let tracked mut second_tok = second_tok;
    proof {
        assert(knowledge_of(observer, empty) =~= ISet::empty());
    }
    let (Ghost(a), Tracked(receipt_a)) = memcpy_async(
        producer,
        Tracked(instance),
        first,
        region,
        Tracked(&mut region_perm),
        Tracked(&mut first_tok),
        Tracked(&mut citations),
        Tracked(&none),
        Ghost(empty),
    )?;
    event_record(snapshot, producer)?;
    let ghost only_a = empty.insert(a);
    proof {
        assert(receipt_a.value().preds =~= empty);
    }
    let (Ghost(b), Tracked(receipt_b)) = memcpy_async(
        producer,
        Tracked(instance),
        second,
        region,
        Tracked(&mut region_perm),
        Tracked(&mut second_tok),
        Tracked(&mut citations),
        Tracked(&none),
        Ghost(empty),
    )?;
    let tracked mut region_tok = region_perm.into_owned();
    let tracked mut second_perm = ReadPerm::Owned(second_tok);
    proof {
        assert(receipt_b.value().preds =~= only_a);
        assert(a != b);
        assert(slot_frontier(region_tok).readers.contains(b));
        assert(!write_admitted(slot_frontier(region_tok), only_a));
    }
    event_record(handoff, producer)?;
    stream_wait_event(consumer, handoff)?;
    proof {
        assert(consumer.tail() =~= empty.insert(b));
        assert(consumer.infos().map()[b].preds.contains(a));
        assert(closure_from(consumer.infos().map(), consumer.tail()) =~= only_a.insert(b));
        let merged = consumer.infos().map().union_prefer_right(IMap::empty());
        assert(merged =~= consumer.infos().map());
        assert(consumer.tail().union(empty) =~= consumer.tail());
    }
    let (Ghost(w), Tracked(receipt_w)) = memcpy_async(
        consumer,
        Tracked(instance),
        region,
        second,
        Tracked(&mut second_perm),
        Tracked(&mut region_tok),
        Tracked(&mut citations),
        Tracked(&none),
        Ghost(empty),
    )?;
    proof {
        assert(receipt_w.value().preds.contains(b));
    }
    if let EventStatus::Ready(knowledge) = event_query(snapshot, Tracked(instance))? {
        let Tracked(known) = knowledge;
        proof {
            assert(closure_from(snapshot.infos().map(), snapshot.recorded()) =~= only_a);
            assert(known.set() == knowledge_of(observer, only_a));
            assert(!known.set().contains((observer, b)));
        }
        event_record(snapshot, consumer)?;
        proof {
            assert(snapshot.recorded() == empty.insert(w));
            assert(known.set() == knowledge_of(observer, only_a));
            assert(!known.set().contains((observer, b)));
        }
    }
    Ok(())
}

} // verus!
