//! End-to-end ghost proof examples exercising `HistoryTokens`.
#[cfg(verus_only)]
use super::*;
#[cfg(verus_only)]
use crate::domain::Domain;
#[cfg(verus_only)]
use crate::space::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::{IMapToken, KeyValueToken};

verus! {

broadcast use vstd::iset_lib::group_iset_properties;

/// Splits the tokens with keys in `keys` off `tokens`. A ghost operation, so
/// its cost does not depend on how many keys there are.
proof fn take_keys<Key, Value, Token: KeyValueToken<Key, Value>>(
    tracked tokens: &mut IMapToken<Key, Value, Token>,
    keys: ISet<Key>,
) -> (tracked taken: IMapToken<Key, Value, Token>)
    requires
        keys.subset_of(old(tokens).dom()),
    ensures
        taken.instance_id() == old(tokens).instance_id(),
        final(tokens).instance_id() == old(tokens).instance_id(),
        taken.map() == old(tokens).map().restrict(keys),
        final(tokens).map() == old(tokens).map().remove_keys(keys),
{
    broadcast use vstd::imap::group_imap_lemmas;

    let id = tokens.instance_id();
    let tracked mut all = IMapToken::empty(id);
    vstd::modes::tracked_swap(tokens, &mut all);
    let old_map = all.map();
    let tracked mut map = all.into_map();
    let tracked taken = map.tracked_remove_keys(keys);
    let tracked rest = IMapToken::from_map(id, map);
    let tracked taken = IMapToken::from_map(id, taken);
    assert(taken.map() =~= old_map.restrict(keys));
    assert(rest.map() =~= old_map.remove_keys(keys));
    *tokens = rest;
    taken
}

/// One location `loc`: `writer` writes it, then the location is lent as two
/// read shares and `reader` reads it through one share, waiting for `writer`.
/// The two events conflict, and the tokens show `writer` happens before
/// `reader`.
proof fn example_write_then_shared_read<D: Domain>(
    observer: D::Observer,
    loc: D::Loc,
    writer: D::Event,
    reader: D::Event,
    first: D::Share,
    second: D::Share,
)
    requires
        writer != reader,
        first != second,
{
    broadcast use vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas;

    let tracked (
        Tracked(instance),
        Tracked(submitted),
        Tracked(knows),
        _,
        Tracked(mut unused),
        Tracked(mut unused_shares),
        Tracked(mut region),
        Tracked(read_share),
    ) = HistoryTokens::Instance::<D>::initialize();
    let here = ISet::empty().insert(loc);
    let none = IMap::<D::Event, EventInfo<D>>::empty();
    let no_events = ISet::<D::Event>::empty();
    let no_shares = IMap::<(D::Loc, D::Share), Frontier<D>>::empty();

    // `writer` writes `loc`, which is owned and untouched.
    let write = Access::<D> { reads: ISet::empty(), writes: here };
    let any_output = |input: Snapshot<D>, output: Snapshot<D>| true;
    let untouched_here = IMap::<D::Loc, Frontier<D>>::empty().insert(loc, untouched());
    let tracked writer_name = unused.remove(writer);
    let tracked loc_slot = take_keys(&mut region, here);
    assert(loc_slot.map() =~= owned_slots(untouched_here));
    let tracked mut writer_infos = HistoryTokens::submitted_map::empty(instance.id());
    let tracked (Tracked(writer_info), Tracked(loc_slot), Tracked(empty_shares)) = instance.submit(
        observer,
        writer,
        write,
        any_output,
        IMap::empty(),
        no_events,
        no_events,
        none,
        untouched_here,
        no_shares,
        &writer_infos,
        &HistoryTokens::knows_set::empty(instance.id()),
        writer_name,
        loc_slot,
        HistoryTokens::read_share_map::empty(instance.id()),
    );
    writer_infos.insert(writer_info);
    assert(writer_infos.map()[writer].preds =~= no_events);
    let written_here = IMap::<D::Loc, Frontier<D>>::empty().insert(
        loc,
        Frontier { last_writer: Some(writer), readers: ISet::empty() },
    );
    assert(loc_slot.map() =~= owned_slots(written_here));

    // Each new share starts with `writer` as its last writer.
    let ids = ISet::empty().insert(first).insert(second);
    let tracked mut fresh = HistoryTokens::unused_shares_set::empty(instance.id());
    fresh.insert(unused_shares.remove(first));
    fresh.insert(unused_shares.remove(second));
    assert(fresh.set() =~= ids);
    let tracked (Tracked(loc_slot), Tracked(mut loc_shares)) = instance.lend(
        written_here,
        ids,
        fresh,
        loc_slot,
    );

    // `reader` reads `loc` through one share, after `writer` in queue order.
    let read = Access::<D> { reads: here, writes: ISet::empty() };
    let first_keys = ISet::empty().insert((loc, first));
    let tracked share = take_keys(&mut loc_shares, first_keys);
    let shares = share.map();
    assert(shares.dom() =~= first_keys);
    let after = ISet::empty().insert(writer);
    assert(after.union(no_events) =~= after);
    assert(shares.dom().contains((loc, first)));
    assert(writer_infos.dom().contains(writer));
    let tracked reader_name = unused.remove(reader);
    let tracked (Tracked(reader_info), _, Tracked(share)) = instance.submit(
        observer,
        reader,
        read,
        any_output,
        IMap::empty().insert(loc, first),
        after,
        no_events,
        writer_infos.map(),
        IMap::empty(),
        shares,
        &writer_infos,
        &HistoryTokens::knows_set::empty(instance.id()),
        reader_name,
        HistoryTokens::region_map::empty(instance.id()),
        share,
    );

    // The tokens alone show the write happens before the read.
    let tracked mut both = writer_infos;
    both.insert(reader_info);
    instance.conflicting_events_are_ordered(writer, reader, loc, both.map(), &both);
    assert(both.map()[reader].preds.contains(writer));

    // Returning the first share leaves the second one out.
    assert(loc_slot.dom() =~= here);
    assert(is_lent_with(loc_slot.map()[loc], first));
    assert(share_keys::<D>(loc_slot.dom(), first) =~= first_keys);
    assert(share.dom() =~= first_keys);
    assert(forall|l: D::Loc| #[trigger]
        loc_slot.dom().contains(l) ==> is_lent_with(loc_slot.map()[l], first));
    instance.give_back(loc_slot.map(), first, share.map(), loc_slot, share);
}

/// The newer lease can be returned or fenced while the original lease remains live.
proof fn example_lend_more<D: Domain>(
    loc: D::Loc,
    first: D::Share,
    second: D::Share,
    fence: D::Event,
    revoke: bool,
)
    requires
        first != second,
{
    broadcast use vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas;

    let tracked (
        Tracked(instance),
        _,
        _,
        _,
        Tracked(mut events),
        Tracked(mut names),
        Tracked(mut region),
        _,
    ) = HistoryTokens::Instance::<D>::initialize();
    let here = ISet::empty().insert(loc);
    let frontiers = here.mk_map(|_| untouched::<D>());
    let tracked slot = take_keys(&mut region, here);
    assert(slot.map() =~= owned_slots(frontiers));
    let tracked mut fresh = HistoryTokens::unused_shares_set::empty(instance.id());
    fresh.insert(names.remove(first));
    let tracked (Tracked(slot), Tracked(original)) = instance.lend(
        frontiers,
        fresh.set(),
        fresh,
        slot,
    );
    let tracked mut fresh = HistoryTokens::unused_shares_set::empty(instance.id());
    fresh.insert(names.remove(second));
    assert(fresh.set().contains(second));
    let tracked (Tracked(slot), Tracked(added)) = instance.lend_more(
        slot.map(),
        fresh.set(),
        fresh,
        slot,
    );
    assert(added.dom() =~= share_keys::<D>(slot.dom(), second));
    assert(slot.map()[loc]->Lent_shares.remove(second).contains(first));
    let tracked slot = if revoke {
        let tracked (_, _, Tracked(slot)) = instance.revoke_read_share(
            slot.map(),
            second,
            fence,
            events.remove(fence),
            slot,
        );
        slot
    } else {
        instance.give_back(slot.map(), second, added.map(), slot, added)
    };
    assert(is_lent_with(slot.map()[loc], first));
    assert(!is_lent_with(slot.map()[loc], second));
    assert(original.dom() =~= share_keys::<D>(slot.dom(), first));
    assert(slot.map()[loc]->Lent_shares.remove(first) =~= ISet::<D::Share>::empty());
    let tracked owned = instance.give_back(slot.map(), first, original.map(), slot, original);
    assert(owned.map()[loc] is Owned);
    assert(revoke ==> owned.map()[loc]->Owned_0.readers.contains(fence));
}

/// The reader keeps its share across revocation. The later writer cites
/// the observed fence and proceeds without waiting for the late read.
proof fn example_fenced_read_revocation<D: Domain>(
    observer: D::Observer,
    loc: D::Loc,
    share: D::Share,
    writer: D::Event,
    fence_event: D::Event,
    reader: D::Event,
    later_writer: D::Event,
)
    requires
        writer != fence_event,
        writer != reader,
        writer != later_writer,
        fence_event != reader,
        fence_event != later_writer,
        reader != later_writer,
{
    broadcast use vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas;

    let tracked (
        Tracked(instance),
        Tracked(mut infos),
        Tracked(knows),
        _,
        Tracked(mut unused),
        Tracked(mut unused_shares),
        Tracked(mut region),
        Tracked(empty_shares),
    ) = HistoryTokens::Instance::<D>::initialize();
    let here = ISet::empty().insert(loc);
    let no_events = ISet::<D::Event>::empty();
    let after_write = no_events.insert(writer);
    let fence_only = no_events.insert(fence_event);
    let write = Access::<D> { reads: ISet::empty(), writes: here };
    let read = Access::<D> { reads: here, writes: ISet::empty() };
    let any_output = |_: Snapshot<D>, _: Snapshot<D>| true;
    let frontiers = IMap::<D::Loc, Frontier<D>>::empty().insert(loc, untouched());
    let tracked slot = take_keys(&mut region, here);
    assert(slot.map() =~= owned_slots(frontiers));
    let tracked (Tracked(writer_info), Tracked(slot), Tracked(empty_shares)) = instance.submit(
        observer,
        writer,
        write,
        any_output,
        IMap::empty(),
        no_events,
        no_events,
        infos.map(),
        frontiers,
        empty_shares.map(),
        &infos,
        &knows,
        unused.remove(writer),
        slot,
        empty_shares,
    );
    infos.insert(writer_info);
    let written = advance_owned(frontiers, writer, write);

    let tracked mut fresh = HistoryTokens::unused_shares_set::empty(instance.id());
    fresh.insert(unused_shares.remove(share));
    assert(fresh.set().remove(share) =~= ISet::<D::Share>::empty());
    let tracked (Tracked(slot), Tracked(read_share)) = instance.lend(
        written,
        fresh.set(),
        fresh,
        slot,
    );
    assert(slot.dom() =~= here);
    assert forall|location: D::Loc| #[trigger] slot.dom().contains(location) implies is_lent_with(
        slot.map()[location],
        share,
    ) by {}
    let tracked (Tracked(fence_info), Tracked(fences), Tracked(slot)) = instance.revoke_read_share(
        slot.map(),
        share,
        fence_event,
        unused.remove(fence_event),
        slot,
    );
    let tracked mut fence_infos = HistoryTokens::submitted_map::empty(instance.id());
    fence_infos.insert(fence_info);
    let tracked observed = instance.observe(observer, fence_only, fence_infos.map(), &fence_infos);

    // The partitioned reader still waits only for the original writer.
    let tracked (Tracked(reader_info), _, _) = instance.submit(
        observer,
        reader,
        read,
        any_output,
        IMap::empty().insert(loc, share),
        after_write,
        no_events,
        infos.map(),
        IMap::empty(),
        read_share.map(),
        &infos,
        &knows,
        unused.remove(reader),
        HistoryTokens::region_map::empty(instance.id()),
        read_share,
    );
    infos.insert(fence_infos.remove(fence_event));

    let reclaimed = IMap::<D::Loc, Frontier<D>>::empty().insert(
        loc,
        Frontier { last_writer: Some(writer), readers: fence_only },
    );
    assert(no_events.union(fence_only) =~= fence_only);
    assert(slot.map() =~= owned_slots(reclaimed));
    let tracked (Tracked(later_info), _, _) = instance.submit(
        observer,
        later_writer,
        write,
        any_output,
        IMap::empty(),
        after_write,
        fence_only,
        infos.map(),
        reclaimed,
        empty_shares.map(),
        &infos,
        &observed,
        unused.remove(later_writer),
        slot,
        empty_shares,
    );
    infos.insert(reader_info);
    infos.insert(later_info);
    assert(!infos.map()[reader].preds.contains(later_writer));
    assert(!infos.map()[later_writer].preds.contains(reader));
    instance.revoked_read_conflict(
        reader,
        later_writer,
        loc,
        share,
        fence_event,
        infos.map(),
        fences.map(),
        &infos,
        &fences,
    );
    assert(infos.map()[later_writer].preds.contains(fence_event));
}

} // verus!
