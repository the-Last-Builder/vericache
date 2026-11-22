//! The `HistoryTokens` VerusSync tokenized state machine.
// The state machine macro names its generated module after the machine.
#![allow(non_snake_case)]

#[cfg(verus_only)]
use super::lemmas::*;
#[cfg(verus_only)]
use super::submit_lemmas::*;
#[cfg(verus_only)]
use super::*;
use super::{EventInfo, Frontier, Slot};
use crate::domain::Domain;
#[cfg(verus_only)]
use crate::history::*;
#[cfg(verus_only)]
use crate::order::*;
use crate::space::*;
use verus_state_machines_macros::tokenized_state_machine;
use vstd::prelude::*;

verus! {

broadcast use vstd::iset_lib::group_iset_properties;
// ---------------------------------------------------------------------------
// The token system
// ---------------------------------------------------------------------------

tokenized_state_machine!{ HistoryTokens<D: Domain> {
    fields {
        /// Persistent: each token says one event is submitted, with its info.
        #[sharding(persistent_imap)]
        pub submitted: IMap<D::Event, EventInfo<D>>,

        /// Persistent: each token says one observer knows one event completed.
        #[sharding(persistent_iset)]
        pub knows: ISet<(D::Observer, D::Event)>,

        /// The fence that revoked a share on one location.
        #[sharding(persistent_imap)]
        pub revoked: IMap<(D::Loc, D::Share), D::Event>,

        /// Exclusive: each token is the right to submit one event name.
        #[sharding(iset)]
        pub unused: ISet<D::Event>,

        /// Exclusive: each token is the right to lend one fresh share name.
        #[sharding(iset)]
        pub unused_shares: ISet<D::Share>,

        /// Exclusive: each token is the right to access one location, or,
        /// while the location is lent, to take back its read shares.
        #[sharding(imap)]
        pub region: IMap<D::Loc, Slot<D>>,

        /// Exclusive: each token is one read share of one location.
        #[sharding(imap)]
        pub read_share: IMap<(D::Loc, D::Share), Frontier<D>>,
    }

    #[invariant]
    pub fn history_is_reachable(&self) -> bool {
        reachable(history_of(self.submitted, self.knows, self.revoked))
    }

    #[invariant]
    pub fn unused_events_are_not_submitted(&self) -> bool {
        forall|event: D::Event| #[trigger] self.unused.contains(event)
            ==> !self.submitted.dom().contains(event)
    }

    /// Every location's frontier accounts for all events touching it.
    #[invariant]
    pub fn frontiers_cover(&self) -> bool {
        forall|loc: D::Loc| #[trigger] self.region.dom().contains(loc)
            ==> covers(
                history_of(self.submitted, self.knows, self.revoked),
                loc,
                frontier_of(self.region[loc], self.read_share, loc),
            )
    }

    #[invariant]
    pub fn unused_shares_are_fresh(&self) -> bool {
        forall|loc: D::Loc, id: D::Share| self.unused_shares.contains(id) ==>
            !#[trigger] self.read_share.dom().contains((loc, id)) && !#[trigger] self.revoked.dom().contains((loc, id))
    }

    #[invariant]
    pub fn shares_match_slots(&self) -> bool {
        forall|loc: D::Loc, id: D::Share|
            #![trigger self.revoked.dom().contains((loc, id))]
            #![trigger self.read_share.dom().contains((loc, id))]
            #![trigger is_lent_with(self.region[loc], id)]
            if self.revoked.dom().contains((loc, id)) {
                &&& self.region.dom().contains(loc)
                &&& !is_lent_with(self.region[loc], id)
                &&& self.read_share.dom().contains((loc, id))
            } else {
                self.read_share.dom().contains((loc, id))
                    == (self.region.dom().contains(loc) && is_lent_with(self.region[loc], id))
            }
    }

    #[invariant]
    pub fn shares_agree_on_last_writer(&self) -> bool {
        forall|loc: D::Loc, id: D::Share| #[trigger] self.read_share.dom().contains((loc, id))
            && !self.revoked.dom().contains((loc, id))
            ==> self.read_share[(loc, id)].last_writer == self.region[loc]->Lent_base.last_writer
    }

    #[invariant]
    pub fn share_readers_are_guarded(&self) -> bool {
        forall|loc: D::Loc, id: D::Share, event: D::Event| self.read_share.dom().contains((loc, id))
            && #[trigger] self.read_share[(loc, id)].readers.contains(event)
            ==> self.submitted.dom().contains(event) && self.submitted[event].read_guards.dom().contains(loc)
                && self.submitted[event].read_guards[loc] == id
    }

    #[invariant]
    pub fn fences_are_covered(&self) -> bool {
        let h = history_of(self.submitted, self.knows, self.revoked);
        forall|loc: D::Loc, id: D::Share| #[trigger] self.revoked.dom().contains((loc, id)) ==> {
            let fence = self.revoked[(loc, id)];
            // A fence cannot be a reader recorded only on another share.
            &&& self.submitted[fence].read_guards == IMap::<D::Loc, D::Share>::empty()
            &&& frontier_contains(h, frontier_of(self.region[loc], self.read_share, loc), fence)
        }
    }

    #[invariant]
    pub fn revoked_share_bound(&self) -> bool {
        let h = history_of(self.submitted, self.knows, self.revoked);
        forall|loc: D::Loc, id: D::Share, writer: D::Event| #[trigger] self.revoked.dom().contains((loc, id))
            && h.events.contains(writer) && #[trigger] (h.access)(writer).writes.contains(loc)
            ==> at_or_before(h, writer, self.read_share[(loc, id)].last_writer)
                || (h.happens_before)(self.revoked[(loc, id)], writer)
    }

    init!{
        initialize() {
            init submitted = IMap::empty();
            init knows = ISet::empty();
            init revoked = IMap::empty();
            init unused = ISet::full();
            init unused_shares = ISet::full();
            init region = IMap::new(|loc: D::Loc| true, |loc: D::Loc| Slot::Owned(untouched()));
            init read_share = IMap::empty();
        }
    }

    /// Submit `new_event`. `owned` are the frontiers of the locations the
    /// caller owns, `shares` the read shares it holds; together they must
    /// cover the access. `infos` are `submitted` facts for `after` and `cited`.
    transition!{
        submit(
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
        ) {
            let waits_for = after.union(cited);
            let ordered = closure_from(infos, waits_for);
            require waits_for.subset_of(infos.dom());
            require access.writes.subset_of(owned.dom());
            require read_guards.dom().subset_of(access.reads);
            require forall|loc: D::Loc| #[trigger] read_guards.dom().contains(loc) ==> !owned.dom().contains(loc)
                && shares.dom().contains((loc, read_guards[loc]));
            require access.reads.subset_of(owned.dom().union(read_guards.dom()));
            require forall|loc: D::Loc| #[trigger] owned.dom().contains(loc)
                ==> admitted(owned[loc], loc, access, ordered);
            require forall|key: (D::Loc, D::Share)| #[trigger] shares.dom().contains(key)
                ==> admitted(shares[key], key.0, access, ordered);

            remove unused -= set { new_event };
            have submitted >= (infos);
            have knows >= (knowledge_of(observer, cited));
            remove region -= (owned_slots(owned));
            remove read_share -= (shares);

            add submitted (union)= [new_event => EventInfo { access, effect, read_guards, preds: ordered }];
            add region += (owned_slots(advance_owned(owned, new_event, access)));
            add read_share += (advance_shares(shares, new_event, read_guards));
        }
    }

    /// `observer` sees `completed` finish.
    transition!{
        observe(
            observer: D::Observer,
            completed: ISet<D::Event>,
            infos: IMap<D::Event, EventInfo<D>>,
        ) {
            require completed.subset_of(infos.dom());
            have submitted >= (infos);
            add knows (union)= (knowledge_of(observer, closure_from(infos, completed)));
        }
    }

    /// `receiver` is told by `sender` that `facts` completed.
    transition!{
        learn(
            receiver: D::Observer,
            sender: D::Observer,
            facts: ISet<D::Event>,
            infos: IMap<D::Event, EventInfo<D>>,
        ) {
            require facts.subset_of(infos.dom());
            have submitted >= (infos);
            have knows >= (knowledge_of(sender, facts));
            add knows (union)= (knowledge_of(receiver, closure_from(infos, facts)));
        }
    }

    /// Lend owned locations out as the read shares `ids`.
    transition!{
        lend(frontiers: IMap<D::Loc, Frontier<D>>, ids: ISet<D::Share>) {
            require ids != ISet::<D::Share>::empty();
            remove unused_shares -= (ids);
            remove region -= (owned_slots(frontiers));
            add region += (lent_slots(frontiers, ids));
            add read_share += (new_shares(frontiers, ids));
        }
    }

    /// Add fresh readers while earlier shares are still outstanding.
    transition!{
        lend_more(lent: IMap<D::Loc, Slot<D>>, ids: ISet<D::Share>) {
            require ids != ISet::<D::Share>::empty();
            require forall|loc: D::Loc| #[trigger] lent.dom().contains(loc) ==> lent[loc] is Lent;
            remove unused_shares -= (ids);
            remove region -= (lent);
            add region += (add_lent_shares(lent, ids));
            add read_share += (new_shares(lent_bases(lent), ids));
        }
    }

    /// Return share `id` of every location in `lent`.
    transition!{
        give_back(
            lent: IMap<D::Loc, Slot<D>>,
            id: D::Share,
            returned: IMap<(D::Loc, D::Share), Frontier<D>>,
        ) {
            require forall|loc: D::Loc| #[trigger] lent.dom().contains(loc)
                ==> is_lent_with(lent[loc], id);
            require returned.dom() == share_keys::<D>(lent.dom(), id);
            remove region -= (lent);
            remove read_share -= (returned);
            add region += (fold_shares(lent, id, returned));
        }
    }

    /// Reclaim the share using only the lent slots and a fresh fence event.
    transition!{
        revoke_read_share(lent: IMap<D::Loc, Slot<D>>, id: D::Share, fence_event: D::Event) {
            require forall|loc: D::Loc| #[trigger] lent.dom().contains(loc) ==> is_lent_with(lent[loc], id);
            remove unused -= set { fence_event };
            remove region -= (lent);
            add submitted (union)= [fence_event => EventInfo {
                access: no_access(), effect: |_, _| true,
                read_guards: IMap::empty(), preds: ISet::empty(),
            }];
            add revoked (union)= (revoked_entries::<D>(lent.dom(), id, fence_event)) by {
                assert forall|key: (D::Loc, D::Share)| #[trigger] revoked_entries::<D>(lent.dom(), id, fence_event).dom().contains(key)
                    implies !pre.revoked.dom().contains(key) by {
                    assert(lent.dom().contains(key.0));
                    assert(is_lent_with(lent[key.0], id));
                    assert(pre.region.dom().contains(key.0));
                    assert(pre.region[key.0] == lent[key.0]);
                    assert(!pre.revoked.dom().contains((key.0, id)));
                }
            };
            add region += (revoke_slots(lent, id, fence_event));
        }
    }

    property!{
        conflicting_events_are_ordered(
            a: D::Event, b: D::Event, loc: D::Loc, infos: IMap<D::Event, EventInfo<D>>,
        ) {
            have submitted >= (infos);
            require infos.dom().contains(a) && infos.dom().contains(b) && a != b;
            require conflict_at(infos[a].access, infos[b].access, loc);
            birds_eye let fences = pre.revoked;
            assert(infos[a].preds.contains(b) || infos[b].preds.contains(a)
                || writer_waits_for_fence(infos[a], infos[b], loc, fences)
                || writer_waits_for_fence(infos[b], infos[a], loc, fences)) by {
                let h = history_of(pre.submitted, pre.knows, pre.revoked);
                race_free(h);
                assert(conflict_at((h.access)(a), (h.access)(b), loc));
                assert(conflict_resolved_at(h, a, b, loc));
            };
        }
    }

    property!{
        revoked_read_conflict(
            reader: D::Event, writer: D::Event, loc: D::Loc, share: D::Share, fence: D::Event,
            infos: IMap<D::Event, EventInfo<D>>, fences: IMap<(D::Loc, D::Share), D::Event>,
        ) {
            have submitted >= (infos);
            have revoked >= (fences);
            require infos.dom().contains(reader) && infos.dom().contains(writer);
            require infos[reader].read_guards.dom().contains(loc) && infos[reader].read_guards[loc] == share;
            require infos[writer].access.writes.contains(loc);
            require fences.dom().contains((loc, share)) && fences[(loc, share)] == fence;
            assert(infos[reader].preds.contains(writer) || infos[writer].preds.contains(reader)
                || infos[writer].preds.contains(fence)) by {
                let h = history_of(pre.submitted, pre.knows, pre.revoked);
                reachable_is_well_formed(h);
                race_free(h);
                assert((h.read_guards)(reader).dom().contains(loc));
                assert(!(h.read_guards)(writer).dom().contains(loc));
                assert(conflict_at((h.access)(reader), (h.access)(writer), loc));
            };
        }
    }

    property!{
        unused_not_submitted(names: ISet<D::Event>, infos: IMap<D::Event, EventInfo<D>>) {
            have unused >= (names);
            have submitted >= (infos);
            assert(forall|e: D::Event| #[trigger] names.contains(e) ==> !infos.dom().contains(e)) by {
                assert forall|e: D::Event| #[trigger] names.contains(e) implies !infos.dom().contains(e) by {
                    assert(pre.unused.contains(e));
                };
            };
        }
    }

    property!{
        slot_writer_is_latest(
            loc: D::Loc, slot: Slot<D>, w: D::Event, e: D::Event,
            infos: IMap<D::Event, EventInfo<D>>,
        ) {
            have region >= [loc => slot];
            have submitted >= (infos);
            require slot_base_frontier(slot).last_writer == Some(w);
            require infos.dom().contains(w) && infos.dom().contains(e) && e != w;
            require infos[e].access.writes.contains(loc);
            assert(infos[w].preds.contains(e)) by {
                let h = history_of(pre.submitted, pre.knows, pre.revoked);
                assert(pre.region.dom().contains(loc));
                assert(covers(h, loc, frontier_of(pre.region[loc], pre.read_share, loc)));
                assert(frontier_of(pre.region[loc], pre.read_share, loc).last_writer == Some(w));
                assert((h.access)(e).writes.contains(loc));
                assert(at_or_before(h, e, Some(w)));
                assert((h.happens_before)(e, w));
            };
        }
    }

    transition!{
        observe_share_bound(
            loc: D::Loc, id: D::Share, fr: Frontier<D>, w: D::Event,
            infos: IMap<D::Event, EventInfo<D>>,
        ) {
            have read_share >= [(loc, id) => fr];
            have submitted >= (infos);
            require fr.last_writer == Some(w) && infos.dom().contains(w);
            birds_eye let maybe_revoked: IMap<(D::Loc, D::Share), D::Event> =
                if pre.revoked.dom().contains((loc, id)) {
                    IMap::empty().insert((loc, id), pre.revoked[(loc, id)])
                } else {
                    IMap::empty()
                };
            add revoked (union)= (maybe_revoked) by {
                assert forall|k: (D::Loc, D::Share)| #[trigger] maybe_revoked.dom().contains(k)
                    implies !pre.revoked.dom().contains(k) || maybe_revoked[k] == pre.revoked[k] by {};
            };
            assert(forall|e: D::Event| #[trigger] infos.dom().contains(e) && infos[e].access.writes.contains(loc) && e != w
                ==> (infos[w].preds.contains(e) || (maybe_revoked.dom().contains((loc, id)) && infos[e].preds.contains(maybe_revoked[(loc, id)])))) by {
                let h = history_of(pre.submitted, pre.knows, pre.revoked);
                assert(pre.read_share.dom().contains((loc, id)));
                assert forall|e: D::Event| #[trigger] infos.dom().contains(e) && infos[e].access.writes.contains(loc) && e != w
                    implies (infos[w].preds.contains(e) || (maybe_revoked.dom().contains((loc, id)) && infos[e].preds.contains(maybe_revoked[(loc, id)]))) by {
                    assert(h.events.contains(e));
                    assert((h.access)(e).writes.contains(loc));
                    if pre.revoked.dom().contains((loc, id)) {
                        assert(maybe_revoked.dom().contains((loc, id)));
                        assert(maybe_revoked[(loc, id)] == pre.revoked[(loc, id)]);
                        assert(at_or_before(h, e, pre.read_share[(loc, id)].last_writer)
                            || (h.happens_before)(pre.revoked[(loc, id)], e));
                    } else {
                        assert(pre.region.dom().contains(loc));
                        assert(is_lent_with(pre.region[loc], id));
                        assert(pre.read_share[(loc, id)].last_writer == pre.region[loc]->Lent_base.last_writer);
                        assert(covers(h, loc, frontier_of(pre.region[loc], pre.read_share, loc)));
                        assert(at_or_before(h, e, Some(w)));
                    }
                };
            };
        }
    }

    #[inductive(observe_share_bound)]
    fn observe_share_bound_inductive(
        pre: Self,
        post: Self,
        loc: D::Loc,
        id: D::Share,
        fr: Frontier<D>,
        w: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
    ) {
        assert(post.revoked =~= pre.revoked);
    }

    transition!{
        duplicate_submitted(infos: IMap<D::Event, EventInfo<D>>) {
            have submitted >= (infos);
            add submitted (union)= (infos) by {};
        }
    }

    #[inductive(duplicate_submitted)]
    fn duplicate_submitted_inductive(
        pre: Self,
        post: Self,
        infos: IMap<D::Event, EventInfo<D>>,
    ) {
        assert(post.submitted =~= pre.submitted);
    }

    transition!{
        duplicate_knows(known: ISet<(D::Observer, D::Event)>) {
            have knows >= (known);
            add knows (union)= (known);
        }
    }

    #[inductive(duplicate_knows)]
    fn duplicate_knows_inductive(
        pre: Self,
        post: Self,
        known: ISet<(D::Observer, D::Event)>,
    ) {
        assert(post.knows =~= pre.knows);
    }

    transition!{
        duplicate_revoked(fences: IMap<(D::Loc, D::Share), D::Event>) {
            have revoked >= (fences);
            add revoked (union)= (fences) by {};
        }
    }

    #[inductive(duplicate_revoked)]
    fn duplicate_revoked_inductive(
        pre: Self,
        post: Self,
        fences: IMap<(D::Loc, D::Share), D::Event>,
    ) {
        assert(post.revoked =~= pre.revoked);
    }

    #[inductive(initialize)]
    fn initialize_inductive(post: Self) {
        no_tokens_is_initial::<D>();
        initial_is_reachable::<D>();
        // No event is submitted yet, and every location is owned and untouched.
        assert forall|loc: D::Loc| #[trigger] post.region.dom().contains(loc) implies covers(
            history_of(post.submitted, post.knows, post.revoked),
            loc,
            frontier_of(post.region[loc], post.read_share, loc),
        ) by {}
    }

    #[inductive(submit)]
    fn submit_inductive(
        pre: Self,
        post: Self,
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
    ) {
        submit_preserves_share_invariants(
            pre,
            post,
            new_event,
            access,
            effect,
            read_guards,
            closure_from(infos, after.union(cited)),
            owned,
            shares,
        );
        submit_preserves_history(
            pre,
            post,
            observer,
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
            infos,
            owned,
            shares,
        );
        submit_preserves_frontiers_and_fences(
            pre,
            post,
            observer,
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
            owned,
            shares,
        );
        submit_preserves_revoked_share_bound(
            pre,
            post,
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
            owned,
        );
    }

    #[inductive(revoke_read_share)]
    fn revoke_read_share_inductive(
        pre: Self, post: Self, lent: IMap<D::Loc, Slot<D>>, id: D::Share, fence_event: D::Event,
    ) {
        broadcast use vstd::imap::group_imap_lemmas, vstd::imap_lib::group_imap_properties;

        let h = history_of(pre.submitted, pre.knows, pre.revoked);
        let next = history_of(post.submitted, post.knows, post.revoked);
        let expected = revoke_read(h, fence_event, id, lent.dom());
        assert(pre.unused.contains(fence_event));
        assert forall|loc: D::Loc| lent.dom().contains(loc) implies !#[trigger] h.revoked.dom().contains((loc, id)) by {
            assert(pre.region.dom().contains(loc));
            assert(is_lent_with(pre.region[loc], id));
        }
        assert(post.shares_match_slots()) by {
            assert forall|loc: D::Loc, share: D::Share|
                if #[trigger] post.revoked.dom().contains((loc, share)) {
                    &&& post.region.dom().contains(loc)
                    &&& !is_lent_with(post.region[loc], share)
                    &&& post.read_share.dom().contains((loc, share))
                } else {
                    post.read_share.dom().contains((loc, share))
                        == (post.region.dom().contains(loc) && is_lent_with(post.region[loc], share))
                } by {
                if lent.dom().contains(loc) && share == id {
                    assert(revoked_entries::<D>(lent.dom(), id, fence_event).dom().contains((loc, id)));
                    assert(pre.region.dom().contains(loc));
                    assert(is_lent_with(pre.region[loc], id));
                    assert(pre.read_share.dom().contains((loc, id)));
                } else {
                    assert(!revoked_entries::<D>(lent.dom(), id, fence_event).dom().contains((loc, share)));
                    assert(post.revoked.dom().contains((loc, share)) == pre.revoked.dom().contains((loc, share)));
                    if lent.dom().contains(loc) {
                        assert(pre.region.dom().contains(loc));
                        assert(post.region.dom().contains(loc));
                        assert(is_lent_with(post.region[loc], share) == is_lent_with(pre.region[loc], share));
                    }
                }
            }
        }
        assert(next.events =~= expected.events);
        assert(next.access =~= expected.access);
        assert(next.effect =~= expected.effect);
        assert(next.read_guards =~= expected.read_guards);
        assert(next.happens_before =~= expected.happens_before);
        step_keeps_reachable(h, Action::RevokeRead { fence_event, share: id, locs: lent.dom() }, next);

        assert forall|loc: D::Loc| #[trigger] post.region.dom().contains(loc) implies {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after = frontier_of(post.region[loc], post.read_share, loc);
            &&& before.last_writer == after.last_writer
            &&& covers(next, loc, after)
            &&& forall|share: D::Share| #[trigger] post.revoked.dom().contains((loc, share)) ==>
                frontier_contains(next, after, post.revoked[(loc, share)])
        } by {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after = frontier_of(post.region[loc], post.read_share, loc);
            assert(pre.region.dom().contains(loc));
            assert(covers(h, loc, before));
            if lent.dom().contains(loc) {
                assert(pre.region[loc] == lent[loc]);
                assert(is_lent_with(lent[loc], id));
                assert(after.readers.contains(fence_event));
                // Only readers of the revoked share can drop out of the frontier.
                assert forall|event: D::Event| #[trigger] before.readers.contains(event) && !after.readers.contains(event)
                    implies pre.read_share[(loc, id)].readers.contains(event) by {
                    let ids = lent[loc]->Lent_shares;
                    let share = choose|share: D::Share| ids.contains(share)
                        && #[trigger] pre.read_share[(loc, share)].readers.contains(event);
                    if share != id {
                        assert(ids.remove(id).contains(share));
                        assert(share_readers(post.read_share, loc, ids.remove(id)).contains(event));
                    }
                }
            }
            assert forall|event: D::Event| next.events.contains(event) && #[trigger] (next.access)(event).reads.contains(loc)
                implies reader_is_covered(next, loc, after, event) by {
                assert((h.access)(event).reads.contains(loc));
                assert(reader_is_covered(h, loc, before, event));
            }
        }
    }

    #[inductive(observe)]
    fn observe_inductive(
        pre: Self,
        post: Self,
        observer: D::Observer,
        completed: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
    ) {
        let h = history_of(pre.submitted, pre.knows, pre.revoked);
        closure_from_is_with_predecessors(pre.submitted, pre.knows, pre.revoked, infos, completed);
        adding_knows_is_add_knowledge(
            pre.submitted,
            pre.knows,
            pre.revoked,
            observer,
            closure_from(infos, completed),
        );
        step_keeps_reachable(h, Action::Observe { observer, completed }, observe(h, observer, completed));
    }

    #[inductive(learn)]
    fn learn_inductive(
        pre: Self,
        post: Self,
        receiver: D::Observer,
        sender: D::Observer,
        facts: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
    ) {
        let h = history_of(pre.submitted, pre.knows, pre.revoked);
        // The sender's tokens say it knows `facts`.
        assert(facts.subset_of((h.known_complete)(sender))) by {
            assert forall|event: D::Event| facts.contains(event) implies #[trigger] (h.known_complete)(sender).contains(event) by {
                assert(knowledge_of(sender, facts).contains((sender, event)));
            }
        }
        closure_from_is_with_predecessors(pre.submitted, pre.knows, pre.revoked, infos, facts);
        adding_knows_is_add_knowledge(pre.submitted, pre.knows, pre.revoked, receiver, closure_from(infos, facts));
        step_keeps_reachable(h, Action::Learn { receiver, sender, facts }, learn(h, receiver, facts));
    }

    #[inductive(lend)]
    fn lend_inductive(pre: Self, post: Self, frontiers: IMap<D::Loc, Frontier<D>>, ids: ISet<D::Share>) {
        broadcast use vstd::imap::group_imap_lemmas, vstd::imap_lib::group_imap_properties;

        let h = history_of(post.submitted, post.knows, post.revoked);
        // Lending moves readers nowhere: a newly lent slot starts with empty
        // shares, and other slots keep their shares.
        assert forall|loc: D::Loc| #[trigger] post.region.dom().contains(loc) implies {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after = frontier_of(post.region[loc], post.read_share, loc);
            &&& before.last_writer == after.last_writer
            &&& before.readers.subset_of(after.readers)
            &&& covers(h, loc, after)
        } by {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after_lend = frontier_of(post.region[loc], post.read_share, loc);
            if frontiers.dom().contains(loc) {
                assert(owned_slots(frontiers).dom().contains(loc));
                assert(pre.region.dom().contains(loc));
                assert(pre.region[loc] == Slot::Owned(frontiers[loc]));
            } else {
                assert(pre.region.dom().contains(loc));
                assert(post.region[loc] == pre.region[loc]);
                if let Slot::Lent { base, shares: lent_ids } = pre.region[loc] {
                    assert forall|event: D::Event| #[trigger] share_readers(pre.read_share, loc, lent_ids).contains(event)
                        implies share_readers(post.read_share, loc, lent_ids).contains(event) by {
                        let i = choose|i: D::Share| lent_ids.contains(i) && #[trigger] pre.read_share[(loc, i)].readers.contains(event);
                        assert(pre.read_share.dom().contains((loc, i)));
                        assert(post.read_share[(loc, i)] == pre.read_share[(loc, i)]);
                    }
                }
            }
            assert(before.readers.subset_of(after_lend.readers));
            covers_with_more_readers(h, loc, before, after_lend);
        }
    }

    #[inductive(lend_more)]
    fn lend_more_inductive(pre: Self, post: Self, lent: IMap<D::Loc, Slot<D>>, ids: ISet<D::Share>) {
        broadcast use vstd::imap::group_imap_lemmas, vstd::imap_lib::group_imap_properties;

        let h = history_of(post.submitted, post.knows, post.revoked);
        assert forall|loc: D::Loc| #[trigger] post.region.dom().contains(loc) implies {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after = frontier_of(post.region[loc], post.read_share, loc);
            &&& before.last_writer == after.last_writer
            &&& before.readers.subset_of(after.readers)
            &&& covers(h, loc, after)
        } by {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after = frontier_of(post.region[loc], post.read_share, loc);
            assert(pre.region.dom().contains(loc));
            if let Slot::Lent { base, shares } = pre.region[loc] {
                assert forall|event: D::Event| #[trigger] share_readers(pre.read_share, loc, shares).contains(event)
                    implies after.readers.contains(event) by {
                    let id = choose|id: D::Share| shares.contains(id) && #[trigger] pre.read_share[(loc, id)].readers.contains(event);
                    assert(pre.read_share.dom().contains((loc, id)));
                    assert(post.read_share[(loc, id)] == pre.read_share[(loc, id)]);
                    assert(share_readers(post.read_share, loc, post.region[loc]->Lent_shares).contains(event));
                }
            }
            assert(before.readers.subset_of(after.readers));
            covers_with_more_readers(h, loc, before, after);
        }
    }

    #[inductive(give_back)]
    fn give_back_inductive(
        pre: Self,
        post: Self,
        lent: IMap<D::Loc, Slot<D>>,
        id: D::Share,
        returned: IMap<(D::Loc, D::Share), Frontier<D>>,
    ) {
        broadcast use vstd::imap::group_imap_lemmas, vstd::imap_lib::group_imap_properties;

        let h = history_of(post.submitted, post.knows, post.revoked);
        // Facts about a returned location: it was lent with `id`, and the
        // returned share is the one that was out.
        assert forall|loc: D::Loc| #[trigger] lent.dom().contains(loc) implies {
            &&& pre.region.dom().contains(loc)
            &&& pre.region[loc] == lent[loc]
            &&& lent[loc] is Lent
            &&& lent[loc]->Lent_shares.contains(id)
            &&& returned.dom().contains((loc, id))
            &&& pre.read_share.dom().contains((loc, id))
            &&& pre.read_share[(loc, id)] == returned[(loc, id)]
        } by {
            assert(pre.region.dom().contains(loc));
            assert(pre.region[loc] == lent[loc]);
            assert(is_lent_with(lent[loc], id));
            assert(share_keys::<D>(lent.dom(), id).contains((loc, id)));
            assert(returned.dom().contains((loc, id)));
            assert(pre.read_share.dom().contains((loc, id)));
        }
        // A share other than the returned ones is unchanged.
        assert forall|key: (D::Loc, D::Share)| #[trigger] pre.read_share.dom().contains(key)
            && !(lent.dom().contains(key.0) && key.1 == id) implies post.read_share.dom().contains(key)
            && post.read_share[key] == pre.read_share[key] by {
            assert(!share_keys::<D>(lent.dom(), id).contains(key));
        }
        assert forall|loc: D::Loc, i: D::Share| #[trigger] post.read_share.dom().contains((loc, i))
            implies pre.read_share.dom().contains((loc, i)) && !(lent.dom().contains(loc) && i == id) by {
            if lent.dom().contains(loc) && i == id {
                assert(share_keys::<D>(lent.dom(), id).contains((loc, i)));
            }
        }

        assert forall|loc: D::Loc| #[trigger] post.region.dom().contains(loc) implies {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after = frontier_of(post.region[loc], post.read_share, loc);
            &&& before.last_writer == after.last_writer
            &&& before.readers.subset_of(after.readers)
            &&& covers(h, loc, after)
        } by {
            let before = frontier_of(pre.region[loc], pre.read_share, loc);
            let after_return = frontier_of(post.region[loc], post.read_share, loc);
            if let Slot::Lent { base, shares: lent_ids } = pre.region[loc] {
                assert forall|event: D::Event| #[trigger] share_readers(pre.read_share, loc, lent_ids).contains(event)
                    implies after_return.readers.contains(event) by {
                    let i = choose|i: D::Share| lent_ids.contains(i) && #[trigger] pre.read_share[(loc, i)].readers.contains(event);
                    assert(pre.read_share.dom().contains((loc, i)));
                    if lent.dom().contains(loc) && i == id {
                        // Folded into the slot's own readers.
                    } else {
                        assert(post.read_share[(loc, i)] == pre.read_share[(loc, i)]);
                        if lent.dom().contains(loc) {
                            assert(lent_ids.remove(id).contains(i));
                        }
                        assert(share_readers(post.read_share, loc, post.region[loc]->Lent_shares).contains(event));
                    }
                }
            }
            assert(before.readers.subset_of(after_return.readers));
            covers_with_more_readers(h, loc, before, after_return);
        }
    }
}}

} // verus!
