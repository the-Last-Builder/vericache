//! Checked trait wrapper exposing `HistoryTokens::Instance` transitions to
//! downstream crates during verification.
use super::HistoryTokens;
#[cfg(verus_only)]
use super::*;
use crate::domain::Domain;
#[cfg(verus_only)]
use crate::history::*;
#[cfg(verus_only)]
use crate::space::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::{ElementToken, KeyValueToken};

verus! {

// The pinned VerusSync macro exports the exchange methods only while checking
// this crate. This trait also makes their checked signatures available to
// downstream crates during verification.
pub trait HistoryTokenTransitions<D: Domain>: Sized {
    spec fn history_instance_id(&self) -> vstd::tokens::InstanceId;

    proof fn lend(
        tracked &self,
        frontiers: IMap<D::Loc, Frontier<D>>,
        ids: ISet<D::Share>,
        tracked fresh: HistoryTokens::unused_shares_set<D>,
        tracked owned: HistoryTokens::region_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::region_map<D>>,
        Tracked<HistoryTokens::read_share_map<D>>,
    ))
        requires
            ids != ISet::<D::Share>::empty(),
            fresh.instance_id() == self.history_instance_id(),
            fresh.set() == ids,
            owned.instance_id() == self.history_instance_id(),
            owned.map() == owned_slots(frontiers),
        ensures
            result.0@.instance_id() == self.history_instance_id(),
            result.1@.instance_id() == self.history_instance_id(),
            result.0@.map() == lent_slots(frontiers, ids),
            result.1@.map() == new_shares(frontiers, ids),
    ;

    proof fn lend_more(
        tracked &self,
        lent: IMap<D::Loc, Slot<D>>,
        ids: ISet<D::Share>,
        tracked fresh: HistoryTokens::unused_shares_set<D>,
        tracked slots: HistoryTokens::region_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::region_map<D>>,
        Tracked<HistoryTokens::read_share_map<D>>,
    ))
        requires
            ids != ISet::<D::Share>::empty(),
            fresh.instance_id() == self.history_instance_id(),
            fresh.set() == ids,
            slots.instance_id() == self.history_instance_id(),
            slots.map() == lent,
            forall|loc: D::Loc| #[trigger] lent.dom().contains(loc) ==> lent[loc] is Lent,
        ensures
            result.0@.instance_id() == self.history_instance_id(),
            result.1@.instance_id() == self.history_instance_id(),
            result.0@.map() == add_lent_shares(lent, ids),
            result.1@.map() == new_shares(lent_bases(lent), ids),
    ;

    proof fn give_back(
        tracked &self,
        lent: IMap<D::Loc, Slot<D>>,
        id: D::Share,
        returned: IMap<(D::Loc, D::Share), Frontier<D>>,
        tracked slots: HistoryTokens::region_map<D>,
        tracked shares: HistoryTokens::read_share_map<D>,
    ) -> (tracked result: HistoryTokens::region_map<D>)
        requires
            slots.instance_id() == self.history_instance_id(),
            shares.instance_id() == self.history_instance_id(),
            slots.map() == lent,
            shares.map() == returned,
            forall|l: D::Loc| #[trigger] lent.dom().contains(l) ==> is_lent_with(lent[l], id),
            returned.dom() == share_keys::<D>(lent.dom(), id),
        ensures
            result.instance_id() == self.history_instance_id(),
            result.map() == fold_shares(lent, id, returned),
    ;

    proof fn revoke_read_share(
        tracked &self,
        lent: IMap<D::Loc, Slot<D>>,
        id: D::Share,
        fence_event: D::Event,
        tracked name: HistoryTokens::unused<D>,
        tracked slots: HistoryTokens::region_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::submitted<D>>,
        Tracked<HistoryTokens::revoked_map<D>>,
        Tracked<HistoryTokens::region_map<D>>,
    ))
        requires
            name.instance_id() == self.history_instance_id(),
            name.element() == fence_event,
            slots.instance_id() == self.history_instance_id(),
            slots.map() == lent,
            forall|loc: D::Loc| #[trigger] lent.dom().contains(loc) ==> is_lent_with(lent[loc], id),
        ensures
            result.0@.instance_id() == self.history_instance_id(),
            result.0@.key() == fence_event,
            result.0@.value() == (EventInfo {
                access: no_access(),
                effect: |_, _| true,
                read_guards: IMap::empty(),
                preds: ISet::empty(),
            }),
            result.1@.instance_id() == self.history_instance_id(),
            result.1@.map() == revoked_entries::<D>(lent.dom(), id, fence_event),
            result.2@.instance_id() == self.history_instance_id(),
            result.2@.map() == revoke_slots(lent, id, fence_event),
    ;

    proof fn revoked_read_conflict(
        tracked &self,
        reader: D::Event,
        writer: D::Event,
        loc: D::Loc,
        share: D::Share,
        fence: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
        fences: IMap<(D::Loc, D::Share), D::Event>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
        tracked revoked: &HistoryTokens::revoked_map<D>,
    )
        requires
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
            revoked.instance_id() == self.history_instance_id(),
            revoked.map() == fences,
            infos.dom().contains(reader),
            infos.dom().contains(writer),
            infos[reader].read_guards.dom().contains(loc),
            infos[reader].read_guards[loc] == share,
            infos[writer].access.writes.contains(loc),
            fences.dom().contains((loc, share)),
            fences[(loc, share)] == fence,
        ensures
            infos[reader].preds.contains(writer) || infos[writer].preds.contains(reader)
                || infos[writer].preds.contains(fence),
    ;

    proof fn conflicting_events_are_ordered(
        tracked &self,
        a: D::Event,
        b: D::Event,
        loc: D::Loc,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (fences: IMap<(D::Loc, D::Share), D::Event>)
        requires
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
            infos.dom().contains(a),
            infos.dom().contains(b),
            a != b,
            conflict_at(infos[a].access, infos[b].access, loc),
        ensures
            infos[a].preds.contains(b) || infos[b].preds.contains(a) || writer_waits_for_fence(
                infos[a],
                infos[b],
                loc,
                fences,
            ) || writer_waits_for_fence(infos[b], infos[a], loc, fences),
    ;

    proof fn slot_writer_is_latest(
        tracked &self,
        loc: D::Loc,
        slot: Slot<D>,
        w: D::Event,
        e: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked slot_tok: &HistoryTokens::region<D>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    )
        requires
            slot_tok.instance_id() == self.history_instance_id(),
            slot_tok.key() == loc,
            slot_tok.value() == slot,
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
            slot_base_frontier(slot).last_writer == Some(w),
            infos.dom().contains(w),
            infos.dom().contains(e),
            e != w,
            infos[e].access.writes.contains(loc),
        ensures
            infos[w].preds.contains(e),
    ;

    proof fn unused_not_submitted(
        tracked &self,
        names: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked unused: &HistoryTokens::unused_set<D>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    )
        requires
            unused.instance_id() == self.history_instance_id(),
            unused.set() == names,
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
        ensures
            forall|e: D::Event| #[trigger] names.contains(e) ==> !infos.dom().contains(e),
    ;

    proof fn observe_share_bound(
        tracked &self,
        loc: D::Loc,
        id: D::Share,
        fr: Frontier<D>,
        w: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked share: &HistoryTokens::read_share<D>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (tracked result: HistoryTokens::revoked_map<D>)
        requires
            share.instance_id() == self.history_instance_id(),
            share.key() == (loc, id),
            share.value() == fr,
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
            fr.last_writer == Some(w),
            infos.dom().contains(w),
        ensures
            result.instance_id() == self.history_instance_id(),
            forall|e: D::Event| #[trigger]
                infos.dom().contains(e) && infos[e].access.writes.contains(loc) && e != w ==> (
                infos[w].preds.contains(e) || (result.map().dom().contains((loc, id))
                    && infos[e].preds.contains(result.map()[(loc, id)]))),
    ;

    proof fn submit(
        tracked &self,
        observer: D::Observer,
        event: D::Event,
        access: Access<D>,
        effect: Effect<D>,
        read_guards: IMap<D::Loc, D::Share>,
        after: ISet<D::Event>,
        cited: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        owned: IMap<D::Loc, Frontier<D>>,
        shares: IMap<(D::Loc, D::Share), Frontier<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
        tracked knows: &HistoryTokens::knows_set<D>,
        tracked name: HistoryTokens::unused<D>,
        tracked slots: HistoryTokens::region_map<D>,
        tracked read_share: HistoryTokens::read_share_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::submitted<D>>,
        Tracked<HistoryTokens::region_map<D>>,
        Tracked<HistoryTokens::read_share_map<D>>,
    ))
        requires
            submitted.instance_id() == self.history_instance_id(),
            knows.instance_id() == self.history_instance_id(),
            name.instance_id() == self.history_instance_id(),
            slots.instance_id() == self.history_instance_id(),
            read_share.instance_id() == self.history_instance_id(),
            name.element() == event,
            submitted.map() == infos,
            knowledge_of(observer, cited).subset_of(knows.set()),
            slots.map() == owned_slots(owned),
            read_share.map() == shares,
            after.union(cited).subset_of(infos.dom()),
            access.writes.subset_of(owned.dom()),
            read_guards.dom().subset_of(access.reads),
            forall|loc: D::Loc| #[trigger]
                read_guards.dom().contains(loc) ==> !owned.dom().contains(loc)
                    && shares.dom().contains((loc, read_guards[loc])),
            access.reads.subset_of(owned.dom().union(read_guards.dom())),
            forall|l: D::Loc| #[trigger]
                owned.dom().contains(l) ==> admitted(
                    owned[l],
                    l,
                    access,
                    closure_from(infos, after.union(cited)),
                ),
            forall|k: (D::Loc, D::Share)| #[trigger]
                shares.dom().contains(k) ==> admitted(
                    shares[k],
                    k.0,
                    access,
                    closure_from(infos, after.union(cited)),
                ),
        ensures
            result.0@.instance_id() == self.history_instance_id(),
            result.0@.key() == event,
            result.0@.value() == (EventInfo {
                access,
                effect,
                read_guards,
                preds: closure_from(infos, after.union(cited)),
            }),
            result.1@.instance_id() == self.history_instance_id(),
            result.1@.map() == owned_slots(advance_owned(owned, event, access)),
            result.2@.instance_id() == self.history_instance_id(),
            result.2@.map() == advance_shares(shares, event, read_guards),
    ;

    proof fn observe(
        tracked &self,
        observer: D::Observer,
        events: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (tracked result: HistoryTokens::knows_set<D>)
        requires
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
            events.subset_of(infos.dom()),
        ensures
            result.instance_id() == self.history_instance_id(),
            result.set() == knowledge_of(observer, closure_from(infos, events)),
    ;

    proof fn learn(
        tracked &self,
        receiver: D::Observer,
        sender: D::Observer,
        events: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
        tracked knows: &HistoryTokens::knows_set<D>,
    ) -> (tracked result: HistoryTokens::knows_set<D>)
        requires
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
            events.subset_of(infos.dom()),
            knows.instance_id() == self.history_instance_id(),
            knowledge_of(sender, events).subset_of(knows.set()),
        ensures
            result.instance_id() == self.history_instance_id(),
            result.set() == knowledge_of(receiver, closure_from(infos, events)),
    ;

    proof fn duplicate_submitted(
        tracked &self,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (tracked result: HistoryTokens::submitted_map<D>)
        requires
            submitted.instance_id() == self.history_instance_id(),
            submitted.map() == infos,
        ensures
            result.instance_id() == self.history_instance_id(),
            result.map() == infos,
    ;

    proof fn duplicate_knows(
        tracked &self,
        known: ISet<(D::Observer, D::Event)>,
        tracked knows: &HistoryTokens::knows_set<D>,
    ) -> (tracked result: HistoryTokens::knows_set<D>)
        requires
            knows.instance_id() == self.history_instance_id(),
            knows.set() == known,
        ensures
            result.instance_id() == self.history_instance_id(),
            result.set() == known,
    ;

    proof fn duplicate_revoked(
        tracked &self,
        fences: IMap<(D::Loc, D::Share), D::Event>,
        tracked revoked: &HistoryTokens::revoked_map<D>,
    ) -> (tracked result: HistoryTokens::revoked_map<D>)
        requires
            revoked.instance_id() == self.history_instance_id(),
            revoked.map() == fences,
        ensures
            result.instance_id() == self.history_instance_id(),
            result.map() == fences,
    ;
}

impl<D: Domain> HistoryTokenTransitions<D> for HistoryTokens::Instance<D> {
    open spec fn history_instance_id(&self) -> vstd::tokens::InstanceId {
        self.id()
    }

    proof fn lend(
        tracked &self,
        frontiers: IMap<D::Loc, Frontier<D>>,
        ids: ISet<D::Share>,
        tracked fresh: HistoryTokens::unused_shares_set<D>,
        tracked owned: HistoryTokens::region_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::region_map<D>>,
        Tracked<HistoryTokens::read_share_map<D>>,
    )) {
        HistoryTokens::Instance::lend(self, frontiers, ids, fresh, owned)
    }

    proof fn lend_more(
        tracked &self,
        lent: IMap<D::Loc, Slot<D>>,
        ids: ISet<D::Share>,
        tracked fresh: HistoryTokens::unused_shares_set<D>,
        tracked slots: HistoryTokens::region_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::region_map<D>>,
        Tracked<HistoryTokens::read_share_map<D>>,
    )) {
        HistoryTokens::Instance::lend_more(self, lent, ids, fresh, slots)
    }

    proof fn give_back(
        tracked &self,
        lent: IMap<D::Loc, Slot<D>>,
        id: D::Share,
        returned: IMap<(D::Loc, D::Share), Frontier<D>>,
        tracked slots: HistoryTokens::region_map<D>,
        tracked shares: HistoryTokens::read_share_map<D>,
    ) -> (tracked result: HistoryTokens::region_map<D>) {
        HistoryTokens::Instance::give_back(self, lent, id, returned, slots, shares)
    }

    proof fn revoke_read_share(
        tracked &self,
        lent: IMap<D::Loc, Slot<D>>,
        id: D::Share,
        fence_event: D::Event,
        tracked name: HistoryTokens::unused<D>,
        tracked slots: HistoryTokens::region_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::submitted<D>>,
        Tracked<HistoryTokens::revoked_map<D>>,
        Tracked<HistoryTokens::region_map<D>>,
    )) {
        HistoryTokens::Instance::revoke_read_share(self, lent, id, fence_event, name, slots)
    }

    proof fn revoked_read_conflict(
        tracked &self,
        reader: D::Event,
        writer: D::Event,
        loc: D::Loc,
        share: D::Share,
        fence: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
        fences: IMap<(D::Loc, D::Share), D::Event>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
        tracked revoked: &HistoryTokens::revoked_map<D>,
    ) {
        HistoryTokens::Instance::revoked_read_conflict(
            self,
            reader,
            writer,
            loc,
            share,
            fence,
            infos,
            fences,
            submitted,
            revoked,
        );
    }

    proof fn conflicting_events_are_ordered(
        tracked &self,
        a: D::Event,
        b: D::Event,
        loc: D::Loc,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (fences: IMap<(D::Loc, D::Share), D::Event>) {
        HistoryTokens::Instance::conflicting_events_are_ordered(self, a, b, loc, infos, submitted)
    }

    proof fn slot_writer_is_latest(
        tracked &self,
        loc: D::Loc,
        slot: Slot<D>,
        w: D::Event,
        e: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked slot_tok: &HistoryTokens::region<D>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) {
        HistoryTokens::Instance::slot_writer_is_latest(
            self,
            loc,
            slot,
            w,
            e,
            infos,
            submitted,
            slot_tok,
        );
    }

    proof fn unused_not_submitted(
        tracked &self,
        names: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked unused: &HistoryTokens::unused_set<D>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) {
        HistoryTokens::Instance::unused_not_submitted(self, names, infos, submitted, unused);
    }

    proof fn observe_share_bound(
        tracked &self,
        loc: D::Loc,
        id: D::Share,
        fr: Frontier<D>,
        w: D::Event,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked share: &HistoryTokens::read_share<D>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (tracked result: HistoryTokens::revoked_map<D>) {
        let tracked (_, Tracked(revoked)) = HistoryTokens::Instance::observe_share_bound(
            self,
            loc,
            id,
            fr,
            w,
            infos,
            submitted,
            share,
        );
        revoked
    }

    proof fn submit(
        tracked &self,
        observer: D::Observer,
        event: D::Event,
        access: Access<D>,
        effect: Effect<D>,
        read_guards: IMap<D::Loc, D::Share>,
        after: ISet<D::Event>,
        cited: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        owned: IMap<D::Loc, Frontier<D>>,
        shares: IMap<(D::Loc, D::Share), Frontier<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
        tracked knows: &HistoryTokens::knows_set<D>,
        tracked name: HistoryTokens::unused<D>,
        tracked slots: HistoryTokens::region_map<D>,
        tracked read_share: HistoryTokens::read_share_map<D>,
    ) -> (tracked result: (
        Tracked<HistoryTokens::submitted<D>>,
        Tracked<HistoryTokens::region_map<D>>,
        Tracked<HistoryTokens::read_share_map<D>>,
    )) {
        HistoryTokens::Instance::submit(
            self,
            observer,
            event,
            access,
            effect,
            read_guards,
            after,
            cited,
            infos,
            owned,
            shares,
            submitted,
            knows,
            name,
            slots,
            read_share,
        )
    }

    proof fn observe(
        tracked &self,
        observer: D::Observer,
        events: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (tracked result: HistoryTokens::knows_set<D>) {
        HistoryTokens::Instance::observe(self, observer, events, infos, submitted)
    }

    proof fn learn(
        tracked &self,
        receiver: D::Observer,
        sender: D::Observer,
        events: ISet<D::Event>,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
        tracked knows: &HistoryTokens::knows_set<D>,
    ) -> (tracked result: HistoryTokens::knows_set<D>) {
        HistoryTokens::Instance::learn(self, receiver, sender, events, infos, submitted, knows)
    }

    proof fn duplicate_submitted(
        tracked &self,
        infos: IMap<D::Event, EventInfo<D>>,
        tracked submitted: &HistoryTokens::submitted_map<D>,
    ) -> (tracked result: HistoryTokens::submitted_map<D>) {
        HistoryTokens::Instance::duplicate_submitted(self, infos, submitted)
    }

    proof fn duplicate_knows(
        tracked &self,
        known: ISet<(D::Observer, D::Event)>,
        tracked knows: &HistoryTokens::knows_set<D>,
    ) -> (tracked result: HistoryTokens::knows_set<D>) {
        HistoryTokens::Instance::duplicate_knows(self, known, knows)
    }

    proof fn duplicate_revoked(
        tracked &self,
        fences: IMap<(D::Loc, D::Share), D::Event>,
        tracked revoked: &HistoryTokens::revoked_map<D>,
    ) -> (tracked result: HistoryTokens::revoked_map<D>) {
        HistoryTokens::Instance::duplicate_revoked(self, fences, revoked)
    }
}

} // verus!
