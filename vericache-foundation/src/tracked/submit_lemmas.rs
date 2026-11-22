//! Inductive preservation lemmas for `HistoryTokens::submit`.
//!
//! Splitting the `submit` inductive step by invariant group keeps each SMT
//! query small and avoids boolean case-split blowups across unrelated
//! invariants under both Z3 and CVC5.
#[cfg(verus_only)]
use super::lemmas::{
    adding_submitted_is_submit, closure_from_is_with_predecessors, waiting_orders_predecessors,
};
#[cfg(verus_only)]
use super::*;
#[cfg(verus_only)]
use crate::domain::Domain;
#[cfg(verus_only)]
use crate::history::*;
#[cfg(verus_only)]
use crate::order::*;
#[cfg(verus_only)]
use crate::space::*;
use vstd::prelude::*;

verus! {

broadcast use vstd::iset_lib::group_iset_properties;

/// At a single location `loc`, the caller's owned or share token frontier admits
/// any conflict with an earlier event `old`.
proof fn submit_conflict_admitted<D: Domain>(
    h: History<D>,
    old: D::Event,
    loc: D::Loc,
    access: Access<D>,
    read_guards: IMap<D::Loc, D::Share>,
    ordered: ISet<D::Event>,
    region: IMap<D::Loc, Slot<D>>,
    read_share: IMap<(D::Loc, D::Share), Frontier<D>>,
    owned: IMap<D::Loc, Frontier<D>>,
    shares: IMap<(D::Loc, D::Share), Frontier<D>>,
)
    requires
        well_formed(h),
        includes_predecessors(h.happens_before, ordered),
        h.events.contains(old),
        conflict_at((h.access)(old), access, loc),
        access.writes.subset_of(owned.dom()),
        access.reads.subset_of(owned.dom().union(read_guards.dom())),
        owned.dom().contains(loc) ==> {
            &&& covers(h, loc, owned[loc])
            &&& admitted(owned[loc], loc, access, ordered)
        },
        !owned.dom().contains(loc) ==> {
            let id = read_guards[loc];
            &&& shares.dom().contains((loc, id))
            &&& read_share.dom().contains((loc, id))
            &&& read_share[(loc, id)] == shares[(loc, id)]
            &&& admitted(shares[(loc, id)], loc, access, ordered)
            &&& !h.revoked.dom().contains((loc, id)) ==> {
                &&& region.dom().contains(loc)
                &&& is_lent_with(region[loc], id)
                &&& covers(h, loc, frontier_of(region[loc], read_share, loc))
                &&& read_share[(loc, id)].last_writer == region[loc]->Lent_base.last_writer
            }
            &&& h.revoked.dom().contains((loc, id)) && (h.access)(old).writes.contains(loc) ==> {
                ||| at_or_before(h, old, read_share[(loc, id)].last_writer)
                ||| (h.happens_before)(h.revoked[(loc, id)], old)
            }
        },
    ensures
        conflict_admitted_at(h, old, access, read_guards, loc, ordered),
{
    if owned.dom().contains(loc) {
        let frontier = owned[loc];
        let guards = (h.read_guards)(old);
        if at_or_before(h, old, frontier.last_writer) {
            waiting_orders_predecessors(h, old, frontier.last_writer, ordered);
            assert(ordered.contains(old));
        } else if frontier.readers.contains(old) {
            assert(ordered.contains(old));
        } else {
            let fence = h.revoked[(loc, guards[loc])];
            if at_or_before(h, fence, frontier.last_writer) {
                waiting_orders_predecessors(h, fence, frontier.last_writer, ordered);
            }
            assert(ordered.contains(fence));
        }
    } else {
        assert(!access.writes.contains(loc));
        assert(access.reads.contains(loc));
        assert((h.access)(old).writes.contains(loc));
        assert(read_guards.dom().contains(loc));
        let id = read_guards[loc];
        assert(shares.dom().contains((loc, id)));
        if !h.revoked.dom().contains((loc, id)) {
            assert(frontier_of(region[loc], read_share, loc).last_writer == shares[(
                loc,
                id,
            )].last_writer);
            assert(at_or_before(h, old, shares[(loc, id)].last_writer));
        }
        if at_or_before(h, old, shares[(loc, id)].last_writer) {
            waiting_orders_predecessors(h, old, shares[(loc, id)].last_writer, ordered);
            assert(ordered.contains(old));
        } else {
            assert((h.happens_before)(h.revoked[(loc, id)], old));
        }
    }
}

/// One `submit` step preserves `covers` and revoked-fence coverage at a single location `loc`.
proof fn submit_preserves_loc_cover<D: Domain>(
    h: History<D>,
    next: History<D>,
    loc: D::Loc,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
    ordered: ISet<D::Event>,
    before: Frontier<D>,
    after_submit: Frontier<D>,
    pre_revoked: IMap<(D::Loc, D::Share), D::Event>,
    owned_loc: bool,
)
    requires
        well_formed(h),
        next == submit(h, new_event, access, effect, read_guards, after, cited),
        ordered == ordered_before(h, after, cited),
        includes_predecessors(h.happens_before, ordered),
        h.revoked == pre_revoked,
        !h.events.contains(new_event),
        covers(h, loc, before),
        forall|id: D::Share| #[trigger]
            pre_revoked.dom().contains((loc, id)) ==> frontier_contains(
                h,
                before,
                pre_revoked[(loc, id)],
            ),
        forall|old: D::Event|
            h.events.contains(old) && #[trigger] conflict_at((h.access)(old), access, loc)
                ==> conflict_admitted_at(h, old, access, read_guards, loc, ordered),
        owned_loc ==> {
            &&& admitted(before, loc, access, ordered)
            &&& after_submit == advance(before, loc, new_event, access)
        },
        !owned_loc ==> {
            &&& !access.writes.contains(loc)
            &&& after_submit.last_writer == before.last_writer
            &&& before.readers.subset_of(after_submit.readers)
            &&& access.reads.contains(loc) ==> {
                &&& read_guards.dom().contains(loc)
                &&& !pre_revoked.dom().contains((loc, read_guards[loc]))
                    ==> after_submit.readers.contains(new_event)
            }
        },
    ensures
        covers(next, loc, after_submit),
        forall|id: D::Share| #[trigger]
            pre_revoked.dom().contains((loc, id)) ==> frontier_contains(
                next,
                after_submit,
                pre_revoked[(loc, id)],
            ),
{
    assert forall|event: D::Event|
        next.events.contains(event) && #[trigger] (next.access)(event).reads.contains(
            loc,
        ) implies reader_is_covered(next, loc, after_submit, event) by {
        if event != new_event {
            assert((h.access)(event).reads.contains(loc));
            if access.writes.contains(loc) {
                assert(conflict_at((h.access)(event), access, loc));
            }
        }
    }
    assert forall|event: D::Event|
        next.events.contains(event) && #[trigger] (next.access)(event).writes.contains(
            loc,
        ) implies at_or_before(next, event, after_submit.last_writer) by {
        if event != new_event {
            assert((h.access)(event).writes.contains(loc));
            if access.writes.contains(loc) {
                assert(at_or_before(h, event, before.last_writer));
                waiting_orders_predecessors(h, event, before.last_writer, ordered);
            }
        }
    }
    assert forall|id: D::Share| #[trigger]
        pre_revoked.dom().contains((loc, id)) implies frontier_contains(
        next,
        after_submit,
        pre_revoked[(loc, id)],
    ) by {
        let fence = pre_revoked[(loc, id)];
        if access.writes.contains(loc) && at_or_before(h, fence, before.last_writer) {
            waiting_orders_predecessors(h, fence, before.last_writer, ordered);
        }
    }
}

/// A reader recorded in one share `id` of `ids` is in `share_readers`.
proof fn share_readers_contains<D: Domain>(
    read_share: IMap<(D::Loc, D::Share), Frontier<D>>,
    loc: D::Loc,
    ids: ISet<D::Share>,
    id: D::Share,
    event: D::Event,
)
    requires
        ids.contains(id),
        read_share[(loc, id)].readers.contains(event),
    ensures
        share_readers(read_share, loc, ids).contains(event),
{
}

/// `submit` preserves the non-history token invariants (`unused_events`,
/// `unused_shares`, `shares_match_slots`, `shares_agree_on_last_writer`,
/// `share_readers_are_guarded`) and yields pointwise facts for `post.region`
/// and `post.read_share`.
pub(super) proof fn submit_preserves_share_invariants<D: Domain>(
    pre: HistoryTokens::State<D>,
    post: HistoryTokens::State<D>,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    ordered: ISet<D::Event>,
    owned: IMap<D::Loc, Frontier<D>>,
    shares: IMap<(D::Loc, D::Share), Frontier<D>>,
)
    requires
        pre.unused_events_are_not_submitted(),
        pre.unused_shares_are_fresh(),
        pre.shares_match_slots(),
        pre.shares_agree_on_last_writer(),
        pre.share_readers_are_guarded(),
        pre.unused.contains(new_event),
        owned_slots(owned).submap_of(pre.region),
        shares.submap_of(pre.read_share),
        forall|loc: D::Loc| #[trigger]
            read_guards.dom().contains(loc) ==> !owned.dom().contains(loc) && shares.dom().contains(
                (loc, read_guards[loc]),
            ),
        post.unused == pre.unused.remove(new_event),
        post.unused_shares == pre.unused_shares,
        post.revoked == pre.revoked,
        post.submitted == pre.submitted.insert(
            new_event,
            EventInfo { access, effect, read_guards, preds: ordered },
        ),
        post.region == pre.region.remove_keys(owned_slots(owned).dom()).union_prefer_right(
            owned_slots(advance_owned(owned, new_event, access)),
        ),
        post.read_share == pre.read_share.remove_keys(shares.dom()).union_prefer_right(
            advance_shares(shares, new_event, read_guards),
        ),
    ensures
        post.unused_events_are_not_submitted(),
        post.unused_shares_are_fresh(),
        post.shares_match_slots(),
        post.shares_agree_on_last_writer(),
        post.share_readers_are_guarded(),
        post.region.dom() == pre.region.dom(),
        post.read_share.dom() == pre.read_share.dom(),
        forall|loc: D::Loc| #[trigger]
            owned.dom().contains(loc) ==> {
                &&& pre.region.dom().contains(loc)
                &&& pre.region[loc] == Slot::Owned(owned[loc])
                &&& post.region[loc] == Slot::Owned(advance(owned[loc], loc, new_event, access))
            },
        forall|loc: D::Loc| #[trigger]
            post.region.dom().contains(loc) && !owned.dom().contains(loc) ==> post.region[loc]
                == pre.region[loc],
        forall|key: (D::Loc, D::Share)| #[trigger]
            post.read_share.dom().contains(key) ==> {
                &&& post.read_share[key].last_writer == pre.read_share[key].last_writer
                &&& pre.read_share[key].readers.subset_of(post.read_share[key].readers)
            },
        forall|loc: D::Loc| #[trigger]
            read_guards.dom().contains(loc) ==> post.read_share[(
                loc,
                read_guards[loc],
            )].readers.contains(new_event),
{
    broadcast use vstd::imap::group_imap_lemmas;

    assert(!pre.submitted.dom().contains(new_event));
    assert(post.region.dom() =~= pre.region.dom());
    assert(post.read_share.dom() =~= pre.read_share.dom());
    assert forall|loc: D::Loc| #[trigger] owned.dom().contains(loc) implies {
        &&& pre.region.dom().contains(loc)
        &&& pre.region[loc] == Slot::Owned(owned[loc])
        &&& post.region[loc] == Slot::Owned(advance(owned[loc], loc, new_event, access))
    } by {
        assert(owned_slots(owned).dom().contains(loc));
        assert(pre.region.dom().contains(loc));
        assert(owned_slots(owned)[loc] == Slot::Owned(owned[loc]));
        assert(pre.region[loc] == owned_slots(owned)[loc]);
        assert(owned_slots(advance_owned(owned, new_event, access)).dom().contains(loc));
    }
    assert forall|loc: D::Loc| #[trigger]
        post.region.dom().contains(loc) && !owned.dom().contains(loc) implies post.region[loc]
        == pre.region[loc] by {}
    assert forall|loc: D::Loc, id: D::Share|
        post.region.dom().contains(loc) implies #[trigger] is_lent_with(post.region[loc], id)
        == is_lent_with(pre.region[loc], id) by {
        if owned.dom().contains(loc) {
            assert(post.region[loc] == Slot::Owned(advance(owned[loc], loc, new_event, access)));
        }
    }
    assert forall|loc: D::Loc, id: D::Share| #[trigger]
        post.read_share.dom().contains((loc, id)) && !post.revoked.dom().contains(
            (loc, id),
        ) implies post.read_share[(loc, id)].last_writer
        == post.region[loc]->Lent_base.last_writer by {
        assert(pre.read_share.dom().contains((loc, id)));
        assert(is_lent_with(pre.region[loc], id));
        assert(!owned.dom().contains(loc));
    }
    assert forall|loc: D::Loc, id: D::Share, event: D::Event|
        post.read_share.dom().contains((loc, id)) && #[trigger] post.read_share[(
            loc,
            id,
        )].readers.contains(event) implies post.submitted.dom().contains(event)
        && post.submitted[event].read_guards.dom().contains(loc)
        && post.submitted[event].read_guards[loc] == id by {
        assert(pre.read_share.dom().contains((loc, id)));
        if !pre.read_share[(loc, id)].readers.contains(event) {
            assert(shares.dom().contains((loc, id)));
            assert(event == new_event);
        }
    }
}

/// `submit` satisfies `can_submit` and advances the tokens' history by one
/// `Action::Submit` step, preserving `history_is_reachable`.
pub(super) proof fn submit_preserves_history<D: Domain>(
    pre: HistoryTokens::State<D>,
    post: HistoryTokens::State<D>,
    observer: D::Observer,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
    infos: IMap<D::Event, EventInfo<D>>,
    owned: IMap<D::Loc, Frontier<D>>,
    shares: IMap<(D::Loc, D::Share), Frontier<D>>,
)
    requires
        pre.history_is_reachable(),
        pre.unused_events_are_not_submitted(),
        pre.frontiers_cover(),
        pre.shares_match_slots(),
        pre.shares_agree_on_last_writer(),
        pre.revoked_share_bound(),
        pre.unused.contains(new_event),
        after.union(cited).subset_of(infos.dom()),
        infos.submap_of(pre.submitted),
        knowledge_of(observer, cited).subset_of(pre.knows),
        access.writes.subset_of(owned.dom()),
        read_guards.dom().subset_of(access.reads),
        forall|loc: D::Loc| #[trigger]
            read_guards.dom().contains(loc) ==> !owned.dom().contains(loc) && shares.dom().contains(
                (loc, read_guards[loc]),
            ),
        access.reads.subset_of(owned.dom().union(read_guards.dom())),
        forall|loc: D::Loc| #[trigger]
            owned.dom().contains(loc) ==> admitted(
                owned[loc],
                loc,
                access,
                closure_from(infos, after.union(cited)),
            ),
        forall|key: (D::Loc, D::Share)| #[trigger]
            shares.dom().contains(key) ==> admitted(
                shares[key],
                key.0,
                access,
                closure_from(infos, after.union(cited)),
            ),
        forall|loc: D::Loc| #[trigger]
            owned.dom().contains(loc) ==> pre.region.dom().contains(loc) && pre.region[loc]
                == Slot::Owned(owned[loc]),
        shares.submap_of(pre.read_share),
        post.knows == pre.knows,
        post.revoked == pre.revoked,
        post.submitted == pre.submitted.insert(
            new_event,
            EventInfo {
                access,
                effect,
                read_guards,
                preds: closure_from(infos, after.union(cited)),
            },
        ),
    ensures
        well_formed(history_of(pre.submitted, pre.knows, pre.revoked)),
        closure_from(infos, after.union(cited)) == ordered_before(
            history_of(pre.submitted, pre.knows, pre.revoked),
            after,
            cited,
        ),
        includes_predecessors(
            history_of(pre.submitted, pre.knows, pre.revoked).happens_before,
            ordered_before(history_of(pre.submitted, pre.knows, pre.revoked), after, cited),
        ),
        can_submit(
            history_of(pre.submitted, pre.knows, pre.revoked),
            observer,
            new_event,
            access,
            read_guards,
            after,
            cited,
        ),
        history_of(post.submitted, post.knows, post.revoked) == submit(
            history_of(pre.submitted, pre.knows, pre.revoked),
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
        ),
        post.history_is_reachable(),
{
    let h = history_of(pre.submitted, pre.knows, pre.revoked);
    reachable_is_well_formed(h);
    let ordered = ordered_before(h, after, cited);
    closure_from_is_with_predecessors(
        pre.submitted,
        pre.knows,
        pre.revoked,
        infos,
        after.union(cited),
    );
    with_predecessors_is_closed(h.happens_before, after.union(cited));
    assert(!h.events.contains(new_event));

    assert(cited.subset_of((h.known_complete)(observer))) by {
        assert forall|event: D::Event| cited.contains(event) implies #[trigger] (h.known_complete)(
            observer,
        ).contains(event) by {
            assert(knowledge_of(observer, cited).contains((observer, event)));
        }
    }
    assert(after.subset_of(h.events)) by {
        assert forall|event: D::Event| after.contains(event) implies #[trigger] h.events.contains(
            event,
        ) by {
            assert(after.union(cited).contains(event));
            assert(infos.dom().contains(event));
        }
    }
    assert forall|old: D::Event, loc: D::Loc|
        h.events.contains(old) && #[trigger] conflict_at(
            (h.access)(old),
            access,
            loc,
        ) implies conflict_admitted_at(h, old, access, read_guards, loc, ordered) by {
        if !owned.dom().contains(loc) {
            assert(!access.writes.contains(loc));
            assert(access.reads.contains(loc));
            assert(read_guards.dom().contains(loc));
            let id = read_guards[loc];
            assert(shares.dom().contains((loc, id)));
            assert(pre.read_share.dom().contains((loc, id)));
        }
        submit_conflict_admitted(
            h,
            old,
            loc,
            access,
            read_guards,
            ordered,
            pre.region,
            pre.read_share,
            owned,
            shares,
        );
    }
    assert(can_submit(h, observer, new_event, access, read_guards, after, cited));

    let next = submit(h, new_event, access, effect, read_guards, after, cited);
    adding_submitted_is_submit(
        pre.submitted,
        pre.knows,
        pre.revoked,
        new_event,
        access,
        effect,
        read_guards,
        after,
        cited,
    );
    assert(history_of(post.submitted, post.knows, post.revoked) == next);
    step_keeps_reachable(
        h,
        Action::Submit { observer, new_event, access, effect, read_guards, after, cited },
        next,
    );
}

/// `submit` preserves `frontiers_cover` and `fences_are_covered`.
pub(super) proof fn submit_preserves_frontiers_and_fences<D: Domain>(
    pre: HistoryTokens::State<D>,
    post: HistoryTokens::State<D>,
    observer: D::Observer,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
    owned: IMap<D::Loc, Frontier<D>>,
    shares: IMap<(D::Loc, D::Share), Frontier<D>>,
)
    requires
        pre.frontiers_cover(),
        pre.fences_are_covered(),
        pre.shares_match_slots(),
        well_formed(history_of(pre.submitted, pre.knows, pre.revoked)),
        can_submit(
            history_of(pre.submitted, pre.knows, pre.revoked),
            observer,
            new_event,
            access,
            read_guards,
            after,
            cited,
        ),
        includes_predecessors(
            history_of(pre.submitted, pre.knows, pre.revoked).happens_before,
            ordered_before(history_of(pre.submitted, pre.knows, pre.revoked), after, cited),
        ),
        history_of(post.submitted, post.knows, post.revoked) == submit(
            history_of(pre.submitted, pre.knows, pre.revoked),
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
        ),
        access.writes.subset_of(owned.dom()),
        access.reads.subset_of(owned.dom().union(read_guards.dom())),
        forall|loc: D::Loc| #[trigger]
            read_guards.dom().contains(loc) ==> !owned.dom().contains(loc) && shares.dom().contains(
                (loc, read_guards[loc]),
            ),
        forall|loc: D::Loc| #[trigger]
            owned.dom().contains(loc) ==> admitted(
                owned[loc],
                loc,
                access,
                ordered_before(history_of(pre.submitted, pre.knows, pre.revoked), after, cited),
            ),
        shares.submap_of(pre.read_share),
        post.revoked == pre.revoked,
        post.submitted == pre.submitted.insert(
            new_event,
            EventInfo {
                access,
                effect,
                read_guards,
                preds: ordered_before(
                    history_of(pre.submitted, pre.knows, pre.revoked),
                    after,
                    cited,
                ),
            },
        ),
        post.region.dom() == pre.region.dom(),
        post.read_share.dom() == pre.read_share.dom(),
        forall|loc: D::Loc| #[trigger]
            owned.dom().contains(loc) ==> {
                &&& pre.region.dom().contains(loc)
                &&& pre.region[loc] == Slot::Owned(owned[loc])
                &&& post.region[loc] == Slot::Owned(advance(owned[loc], loc, new_event, access))
            },
        forall|loc: D::Loc| #[trigger]
            post.region.dom().contains(loc) && !owned.dom().contains(loc) ==> post.region[loc]
                == pre.region[loc],
        forall|key: (D::Loc, D::Share)| #[trigger]
            post.read_share.dom().contains(key) ==> {
                &&& post.read_share[key].last_writer == pre.read_share[key].last_writer
                &&& pre.read_share[key].readers.subset_of(post.read_share[key].readers)
            },
        forall|loc: D::Loc| #[trigger]
            read_guards.dom().contains(loc) ==> post.read_share[(
                loc,
                read_guards[loc],
            )].readers.contains(new_event),
    ensures
        post.frontiers_cover(),
        post.fences_are_covered(),
{
    broadcast use vstd::imap::group_imap_lemmas;

    let h = history_of(pre.submitted, pre.knows, pre.revoked);
    let next = history_of(post.submitted, post.knows, post.revoked);
    let ordered = ordered_before(h, after, cited);

    assert forall|loc: D::Loc| #[trigger] post.region.dom().contains(loc) implies {
        let frontier = frontier_of(post.region[loc], post.read_share, loc);
        &&& covers(next, loc, frontier)
        &&& forall|id: D::Share| #[trigger]
            post.revoked.dom().contains((loc, id)) ==> frontier_contains(
                next,
                frontier,
                post.revoked[(loc, id)],
            )
    } by {
        let before = frontier_of(pre.region[loc], pre.read_share, loc);
        let after_submit = frontier_of(post.region[loc], post.read_share, loc);
        if owned.dom().contains(loc) {
            assert(after_submit == advance(owned[loc], loc, new_event, access));
        } else {
            assert(post.region[loc] == pre.region[loc]);
            assert(before.readers.subset_of(after_submit.readers)) by {
                if let Slot::Lent { base, shares: ids } = pre.region[loc] {
                    assert forall|event: D::Event| #[trigger]
                        share_readers(pre.read_share, loc, ids).contains(
                            event,
                        ) implies share_readers(post.read_share, loc, ids).contains(event) by {
                        let id = choose|id: D::Share|
                            ids.contains(id) && #[trigger] pre.read_share[(
                                loc,
                                id,
                            )].readers.contains(event);
                        assert(pre.read_share.dom().contains((loc, id)));
                        assert(post.read_share[(loc, id)].readers.contains(event));
                    }
                }
            }
            if access.reads.contains(loc) {
                assert(read_guards.dom().contains(loc));
                let id = read_guards[loc];
                assert(shares.dom().contains((loc, id)));
                assert(pre.read_share.dom().contains((loc, id)));
                if !h.revoked.dom().contains((loc, id)) {
                    assert(!pre.revoked.dom().contains((loc, id)));
                    assert(is_lent_with(pre.region[loc], id));
                    assert(pre.region[loc]->Lent_shares.contains(id));
                    assert(post.read_share[(loc, id)].readers.contains(new_event));
                    share_readers_contains(
                        post.read_share,
                        loc,
                        pre.region[loc]->Lent_shares,
                        id,
                        new_event,
                    );
                    assert(after_submit.readers.contains(new_event));
                }
            }
        }
        assert(covers(h, loc, before));
        submit_preserves_loc_cover(
            h,
            next,
            loc,
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
            ordered,
            before,
            after_submit,
            pre.revoked,
            owned.dom().contains(loc),
        );
    }
    assert forall|loc: D::Loc, id: D::Share| #[trigger]
        post.revoked.dom().contains((loc, id)) implies {
        let fence = post.revoked[(loc, id)];
        &&& post.submitted[fence].read_guards == IMap::<D::Loc, D::Share>::empty()
        &&& frontier_contains(next, frontier_of(post.region[loc], post.read_share, loc), fence)
    } by {
        let fence = pre.revoked[(loc, id)];
        assert(pre.region.dom().contains(loc));
        assert(post.region.dom().contains(loc));
        assert(pre.submitted.dom().contains(fence));
        assert(fence != new_event);
        assert(post.submitted.dom().contains(fence));
        assert(post.submitted[fence] == pre.submitted[fence]);
    }
}

/// `submit` preserves `revoked_share_bound`.
pub(super) proof fn submit_preserves_revoked_share_bound<D: Domain>(
    pre: HistoryTokens::State<D>,
    post: HistoryTokens::State<D>,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
    owned: IMap<D::Loc, Frontier<D>>,
)
    requires
        pre.revoked_share_bound(),
        pre.fences_are_covered(),
        includes_predecessors(
            history_of(pre.submitted, pre.knows, pre.revoked).happens_before,
            ordered_before(history_of(pre.submitted, pre.knows, pre.revoked), after, cited),
        ),
        history_of(post.submitted, post.knows, post.revoked) == submit(
            history_of(pre.submitted, pre.knows, pre.revoked),
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
        ),
        access.writes.subset_of(owned.dom()),
        forall|loc: D::Loc| #[trigger]
            owned.dom().contains(loc) ==> {
                &&& pre.region.dom().contains(loc)
                &&& pre.region[loc] == Slot::Owned(owned[loc])
                &&& admitted(
                    owned[loc],
                    loc,
                    access,
                    ordered_before(history_of(pre.submitted, pre.knows, pre.revoked), after, cited),
                )
            },
        post.revoked == pre.revoked,
        post.read_share.dom() == pre.read_share.dom(),
        forall|key: (D::Loc, D::Share)| #[trigger]
            post.read_share.dom().contains(key) ==> post.read_share[key].last_writer
                == pre.read_share[key].last_writer,
        pre.shares_match_slots(),
    ensures
        post.revoked_share_bound(),
{
    let h = history_of(pre.submitted, pre.knows, pre.revoked);
    let next = history_of(post.submitted, post.knows, post.revoked);
    let ordered = ordered_before(h, after, cited);

    assert forall|loc: D::Loc, id: D::Share, writer: D::Event| #[trigger]
        post.revoked.dom().contains((loc, id)) && next.events.contains(writer) && #[trigger] (
        next.access)(writer).writes.contains(loc) implies at_or_before(
        next,
        writer,
        post.read_share[(loc, id)].last_writer,
    ) || (next.happens_before)(post.revoked[(loc, id)], writer) by {
        assert(pre.read_share.dom().contains((loc, id)));
        assert(post.read_share.dom().contains((loc, id)));
        if writer == new_event {
            let frontier = frontier_of(pre.region[loc], pre.read_share, loc);
            let fence = pre.revoked[(loc, id)];
            assert(owned.dom().contains(loc));
            if at_or_before(h, fence, frontier.last_writer) {
                waiting_orders_predecessors(h, fence, frontier.last_writer, ordered);
            }
            assert(ordered.contains(fence));
        } else {
            assert((h.access)(writer).writes.contains(loc));
        }
    }
}

} // verus!
