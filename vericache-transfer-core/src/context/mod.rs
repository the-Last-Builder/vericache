//! The cluster history and the exclusive single-key tokens used at backend boundaries.
#[cfg(verus_only)]
mod batch;
mod ffi;
mod sources;
mod sync;
mod tokens;

pub use crate::domain::EventSourceKind;
use crate::domain::*;
#[cfg(verus_only)]
use foundation::history::{no_access, revoked_entries};
#[cfg(verus_only)]
use foundation::space::{conflict_at, Access, Effect};
use foundation::tracked::*;
use vstd::prelude::*;
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::{ElementToken, ISetToken};
#[cfg(verus_only)]
use vstd::tokens::{IMapToken, KeyValueToken};

pub use sources::*;
#[cfg(verus_only)]
pub use sync::{can_reclaim_slab, can_reclaim_slot};
#[cfg(verus_only)]
pub use sync::{finish_copy, finish_owned, finish_read, finish_slab_owned};
pub use sync::{CompletedAccess, CompletedFence, ObservedEvents};
pub use sync::{Completion, HostContext, SyncContext};
#[cfg(verus_only)]
use tokens::take_keys;
pub use tokens::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::set::group_set_lemmas,
};

pub type RegionMap = HistoryTokens::region_map<TransferDomain>;

pub type ShareToken = HistoryTokens::read_share<TransferDomain>;

pub type RevokedToken = HistoryTokens::revoked<TransferDomain>;

pub type UnusedSet = HistoryTokens::unused_set<TransferDomain>;

pub type UnusedShare = HistoryTokens::unused_shares<TransferDomain>;

pub type UnusedShareSet = HistoryTokens::unused_shares_set<TransferDomain>;

pub type ReadGuards = IMap<SlotLoc, TransferShare>;

pub type Submitted = HistoryTokens::submitted<TransferDomain>;

pub type SubmittedMap = HistoryTokens::submitted_map<TransferDomain>;

pub type Knows = HistoryTokens::knows_set<TransferDomain>;

pub(crate) type ReadShareMap = HistoryTokens::read_share_map<TransferDomain>;

pub type RevokedMap = HistoryTokens::revoked_map<TransferDomain>;

pub tracked struct TransferInstance {
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(crate) tracked inner: HistoryTokens::Instance<TransferDomain>,
}

impl TransferInstance {
    pub closed spec fn id(&self) -> InstanceId {
        self.inner.id()
    }

    pub proof fn duplicate(tracked &self) -> (tracked out: Self)
        ensures
            out == *self,
    {
        Self { inner: self.inner }
    }

    /// Lend one owned slot out as a single read share (`ShareCap` in `Unbound` state).
    pub proof fn lend_slot(
        tracked &self,
        tracked owned: SlotToken,
        tracked fresh: UnusedShare,
    ) -> (tracked result: (Tracked<SlotToken>, Tracked<ShareCap>))
        requires
            owned.well_formed(),
            owned.instance_id() == self.id(),
            owned.value() is Owned,
            fresh.instance_id() == self.id(),
        ensures
            slot_lent(
                result.0@,
                self.id(),
                owned.key(),
                owned.value()->Owned_0,
                ISet::empty().insert(fresh.element()),
            ),
            result.1@.instance_id() == self.id(),
            result.1@.loc() == owned.key(),
            result.1@.share() == fresh.element(),
            result.1@.frontier() == (Frontier {
                last_writer: owned.value()->Owned_0.last_writer,
                readers: ISet::empty(),
            }),
            result.1@.is_unbound(),
    {
        let loc = owned.key();
        let fr = owned.value()->Owned_0;
        let share = fresh.element();
        let ids = ISet::empty().insert(share);
        let frontiers = IMap::empty().insert(loc, fr);
        let tracked mut fresh_set = UnusedShareSet::empty(self.id());
        fresh_set.insert(fresh);
        let tracked owned_map = owned.into_singleton_map();
        assert(ids.contains(share));
        assert(owned_map.map() =~= owned_slots(frontiers));
        let tracked (Tracked(lent_map), Tracked(share_map)) = self.inner.lend(
            frontiers,
            ids,
            fresh_set,
            owned_map,
        );
        assert(lent_map.dom() =~= ISet::empty().insert(loc));
        assert(share_map.dom() =~= ISet::empty().insert((loc, share)));
        let tracked lent_tok = SlotToken::from_singleton_map(loc, lent_map);
        let tracked cap = ShareCap {
            loc,
            share,
            map: share_map,
            usage: ShareUsage::Unbound(FenceRight { instance: self.id(), loc, share }),
        };
        (Tracked(lent_tok), Tracked(cap))
    }

    /// Add one more fresh read share to an already-lent slot.
    pub proof fn lend_more_slot(
        tracked &self,
        tracked lent: SlotToken,
        tracked fresh: UnusedShare,
    ) -> (tracked result: (Tracked<SlotToken>, Tracked<ShareCap>))
        requires
            lent.well_formed(),
            lent.instance_id() == self.id(),
            lent.value() is Lent,
            fresh.instance_id() == self.id(),
        ensures
            slot_lent(
                result.0@,
                self.id(),
                lent.key(),
                lent.value()->Lent_base,
                lent.value()->Lent_shares.insert(fresh.element()),
            ),
            result.1@.instance_id() == self.id(),
            result.1@.loc() == lent.key(),
            result.1@.share() == fresh.element(),
            result.1@.frontier() == (Frontier {
                last_writer: lent.value()->Lent_base.last_writer,
                readers: ISet::empty(),
            }),
            result.1@.is_unbound(),
    {
        let loc = lent.key();
        let shares = lent.value()->Lent_shares;
        let share = fresh.element();
        let ids = ISet::empty().insert(share);
        let lent_map_spec = IMap::empty().insert(loc, lent.value());
        let tracked mut fresh_set = UnusedShareSet::empty(self.id());
        fresh_set.insert(fresh);
        let tracked lent_map = lent.into_singleton_map();
        assert(ids.contains(share));
        assert(lent_map.map() =~= lent_map_spec);
        let tracked (Tracked(out_lent), Tracked(out_shares)) = self.inner.lend_more(
            lent_map_spec,
            ids,
            fresh_set,
            lent_map,
        );
        assert(out_lent.dom() =~= ISet::empty().insert(loc));
        assert(out_shares.dom() =~= ISet::empty().insert((loc, share)));
        assert(shares.union(ids) =~= shares.insert(share));
        let tracked new_lent = SlotToken::from_singleton_map(loc, out_lent);
        let tracked cap = ShareCap {
            loc,
            share,
            map: out_shares,
            usage: ShareUsage::Unbound(FenceRight { instance: self.id(), loc, share }),
        };
        (Tracked(new_lent), Tracked(cap))
    }

    /// Return a read share (`ShareCap`) into its lent slot. When it was the last
    /// outstanding share, the slot becomes `Owned` again.
    pub proof fn give_back_slot(
        tracked &self,
        tracked lent: SlotToken,
        tracked cap: ShareCap,
    ) -> (tracked result: (Tracked<SlotToken>, Tracked<ReturnedShare>))
        requires
            lent.well_formed(),
            cap.well_formed(),
            lent.instance_id() == self.id(),
            cap.instance_id() == self.id(),
            lent.key() == cap.loc(),
            is_lent_with(lent.value(), cap.share()),
        ensures
            result.0@.well_formed(),
            result.0@.instance_id() == self.id(),
            result.0@.key() == lent.key(),
            result.0@.value() == fold_share(lent.value(), cap.share(), cap.frontier()),
            result.1@.instance_id() == self.id(),
            result.1@.loc() == lent.key(),
            result.1@.share() == cap.share(),
    {
        let loc = lent.key();
        let share = cap.share();
        let lent_val = lent.value();
        let returned_fr = cap.frontier();
        let tracked lent_map = lent.into_singleton_map();
        let tracked share_map = cap.into_singleton_map();
        assert(share_map.dom() =~= share_keys::<TransferDomain>(lent_map.dom(), share));
        let tracked out = self.inner.give_back(
            lent_map.map(),
            share,
            share_map.map(),
            lent_map,
            share_map,
        );
        assert(out.dom() =~= ISet::empty().insert(loc));
        let tracked slot = SlotToken::from_singleton_map(loc, out);
        assert(slot.value() == fold_share(lent_val, share, returned_fr));
        (Tracked(slot), Tracked(ReturnedShare { instance: self.id(), loc, share }))
    }

    /// Fence and reclaim a lent read share on one slot using a fresh event name.
    pub proof fn revoke_slot(
        tracked &self,
        share: TransferShare,
        tracked lent: SlotToken,
        tracked name: HistoryTokens::unused<TransferDomain>,
        tracked completed: CompletedFence,
    ) -> (tracked result: (
        Tracked<Submitted>,
        Tracked<RevokedToken>,
        Tracked<SlotToken>,
        Tracked<Knows>,
    ))
        requires
            name.instance_id() == self.id(),
            lent.well_formed(),
            lent.instance_id() == self.id(),
            is_lent_with(lent.value(), share),
            completed.matches(self.id(), lent.key(), share),
        ensures
            result.0@.instance_id() == self.id(),
            result.0@.key() == name.element(),
            result.0@.value() == (EventInfo {
                access: no_access(),
                effect: |_, _| true,
                read_guards: IMap::empty(),
                preds: ISet::empty(),
            }),
            result.1@.instance_id() == self.id(),
            result.1@.key() == (lent.key(), share),
            result.1@.value() == name.element(),
            result.3@.instance_id() == self.id(),
            result.3@.set() == knowledge_of(
                name.element().observer,
                ISet::empty().insert(name.element()),
            ),
            result.2@.well_formed(),
            result.2@.instance_id() == self.id(),
            result.2@.key() == lent.key(),
            result.2@.value() == fold_share(
                lent.value(),
                share,
                Frontier { last_writer: None, readers: ISet::empty().insert(name.element()) },
            ),
    {
        let loc = lent.key();
        let fence = name.element();
        let tracked lent_map = lent.into_singleton_map();
        let tracked (Tracked(sub), Tracked(mut rev_map), Tracked(out_map)) =
            self.inner.revoke_read_share(lent_map.map(), share, fence, name, lent_map);
        assert(rev_map.dom().contains((loc, share)));
        assert(out_map.dom() =~= ISet::empty().insert(loc));
        let tracked rev_tok = rev_map.remove((loc, share));
        let tracked out_tok = SlotToken::from_singleton_map(loc, out_map);
        let tracked mut infos = SubmittedMap::empty(self.id());
        infos.insert(sub);
        let completed = ISet::empty().insert(fence);
        let tracked knows = self.observe(fence.observer, completed, &infos);
        assert(closure_from(infos.map(), completed) =~= completed);
        (Tracked(sub), Tracked(rev_tok), Tracked(out_tok), Tracked(knows))
    }

    pub proof fn conflicting_events_are_ordered(
        tracked &self,
        a: TransferEvent,
        b: TransferEvent,
        loc: SlotLoc,
        tracked infos: &SubmittedMap,
    ) -> (fences: IMap<(SlotLoc, TransferShare), TransferEvent>)
        requires
            infos.instance_id() == self.id(),
            infos.dom().contains(a),
            infos.dom().contains(b),
            a != b,
            conflict_at(infos.map()[a].access, infos.map()[b].access, loc),
        ensures
            infos.map()[a].preds.contains(b) || infos.map()[b].preds.contains(a)
                || writer_waits_for_fence(infos.map()[a], infos.map()[b], loc, fences)
                || writer_waits_for_fence(infos.map()[b], infos.map()[a], loc, fences),
    {
        self.inner.conflicting_events_are_ordered(a, b, loc, infos.map(), infos)
    }

    pub proof fn revoked_read_conflict(
        tracked &self,
        reader: TransferEvent,
        writer: TransferEvent,
        loc: SlotLoc,
        share: TransferShare,
        fence: TransferEvent,
        tracked infos: &SubmittedMap,
        tracked revoked: &RevokedToken,
    )
        requires
            infos.instance_id() == self.id(),
            revoked.instance_id() == self.id(),
            revoked.key() == (loc, share),
            revoked.value() == fence,
            infos.dom().contains(reader),
            infos.dom().contains(writer),
            infos.map()[reader].read_guards.dom().contains(loc),
            infos.map()[reader].read_guards[loc] == share,
            infos.map()[writer].access.writes.contains(loc),
        ensures
            infos.map()[reader].preds.contains(writer) || infos.map()[writer].preds.contains(reader)
                || infos.map()[writer].preds.contains(fence),
    {
        let tracked mut rev_map = RevokedMap::empty(self.id());
        rev_map.insert(*revoked);
        self.inner.revoked_read_conflict(
            reader,
            writer,
            loc,
            share,
            fence,
            infos.map(),
            rev_map.map(),
            infos,
            &rev_map,
        );
    }

    /// Submit a single-slot access using an owned `&mut SlotToken` (write,
    /// partial splice, deregistration, or reclamation).
    pub proof fn submit_1_owned(
        tracked &self,
        observer: NodeId,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        after: ISet<TransferEvent>,
        cited: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked knows: &Knows,
        tracked name: HistoryTokens::unused<TransferDomain>,
        tracked slot: &mut SlotToken,
    ) -> (tracked sub: Submitted)
        requires
            name.instance_id() == self.id(),
            name.element().observer == observer,
            infos.instance_id() == self.id(),
            knows.instance_id() == self.id(),
            old(slot).well_formed(),
            old(slot).instance_id() == self.id(),
            old(slot).value() is Owned,
            after.union(cited).subset_of(infos.dom()),
            knowledge_of(observer, cited).subset_of(knows.set()),
            access.writes.subset_of(ISet::empty().insert(old(slot).key())),
            access.reads.subset_of(ISet::empty().insert(old(slot).key())),
            admitted(
                old(slot).value()->Owned_0,
                old(slot).key(),
                access,
                closure_from(infos.map(), after.union(cited)),
            ),
        ensures
            sub.instance_id() == self.id(),
            sub.key() == name.element(),
            sub.value() == (EventInfo {
                access,
                effect,
                read_guards: IMap::empty(),
                preds: closure_from(infos.map(), after.union(cited)),
            }),
            final(slot).well_formed(),
            final(slot).instance_id() == self.id(),
            final(slot).key() == old(slot).key(),
            final(slot).value() == Slot::Owned(
                advance(old(slot).value()->Owned_0, old(slot).key(), name.element(), access),
            ),
    {
        let loc = slot.loc;
        let fr = slot.map.map()[loc]->Owned_0;
        let frontiers = IMap::empty().insert(loc, fr);
        let tracked mut owned_map = RegionMap::empty(self.id());
        vstd::modes::tracked_swap(&mut slot.map, &mut owned_map);
        let tracked shares = ReadShareMap::empty(self.id());
        assert(owned_map.map() =~= owned_slots(frontiers));
        assert(access.reads.subset_of(
            frontiers.dom().union(IMap::<SlotLoc, TransferShare>::empty().dom()),
        ));
        let tracked (Tracked(sub), Tracked(new_owned), _) = self.inner.submit(
            observer,
            name.element(),
            access,
            effect,
            IMap::empty(),
            after,
            cited,
            infos.map(),
            frontiers,
            IMap::empty(),
            infos,
            knows,
            name,
            owned_map,
            shares,
        );
        assert(new_owned.dom() =~= ISet::empty().insert(loc));
        slot.map = new_owned;
        sub
    }

    /// Submit a single-slot pure read using either an owned `SlotToken` or a `ShareCap`.
    pub proof fn submit_1_read(
        tracked &self,
        observer: NodeId,
        effect: Effect<TransferDomain>,
        after: ISet<TransferEvent>,
        cited: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked knows: &Knows,
        tracked name: HistoryTokens::unused<TransferDomain>,
        tracked src: &mut ReadPerm,
    ) -> (tracked sub: Submitted)
        requires
            name.instance_id() == self.id(),
            name.element().observer == observer,
            infos.instance_id() == self.id(),
            knows.instance_id() == self.id(),
            old(src).instance_id() == self.id(),
            old(src).is_owned() || old(src).is_shared_wf(),
            after.union(cited).subset_of(infos.dom()),
            knowledge_of(observer, cited).subset_of(knows.set()),
            read_admitted(old(src).frontier(), closure_from(infos.map(), after.union(cited))),
        ensures
            sub.instance_id() == self.id(),
            sub.key() == name.element(),
            sub.value() == (EventInfo {
                access: Access {
                    reads: ISet::empty().insert(old(src).loc()),
                    writes: ISet::empty(),
                },
                effect,
                read_guards: old(src).read_guards(),
                preds: closure_from(infos.map(), after.union(cited)),
            }),
            final(src).advanced_read(*old(src), name.element()),
    {
        let loc = src.loc();
        let ev = name.element();
        let access = Access { reads: ISet::empty().insert(loc), writes: ISet::empty() };
        match src {
            ReadPerm::Owned(tok) => {
                self.submit_1_owned(observer, access, effect, after, cited, infos, knows, name, tok)
            },
            ReadPerm::Shared(cap) => {
                let share = cap.share;
                let guards = IMap::empty().insert(loc, share);
                let tracked owned_map = RegionMap::empty(self.id());
                let tracked mut shares_map = ReadShareMap::empty(self.id());
                vstd::modes::tracked_swap(&mut cap.map, &mut shares_map);
                assert(owned_map.map() =~= owned_slots(IMap::empty()));
                assert(access.reads.subset_of(
                    IMap::<SlotLoc, Frontier<TransferDomain>>::empty().dom().union(guards.dom()),
                ));
                let tracked (Tracked(sub), _, Tracked(new_shares)) = self.inner.submit(
                    observer,
                    ev,
                    access,
                    effect,
                    guards,
                    after,
                    cited,
                    infos.map(),
                    IMap::empty(),
                    shares_map.map(),
                    infos,
                    knows,
                    name,
                    owned_map,
                    shares_map,
                );
                assert(new_shares.dom() =~= ISet::empty().insert((loc, share)));
                cap.map = new_shares;
                sub
            },
        }
    }

    /// Submit a two-slot copy (`src` read + distinct owned `dst` write).
    pub proof fn submit_2(
        tracked &self,
        observer: NodeId,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        after: ISet<TransferEvent>,
        cited: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked knows: &Knows,
        tracked name: HistoryTokens::unused<TransferDomain>,
        tracked src: &mut ReadPerm,
        tracked dst: &mut SlotToken,
    ) -> (tracked sub: Submitted)
        requires
            name.instance_id() == self.id(),
            name.element().observer == observer,
            infos.instance_id() == self.id(),
            knows.instance_id() == self.id(),
            old(src).instance_id() == self.id(),
            old(dst).well_formed(),
            old(dst).instance_id() == self.id(),
            old(src).is_owned() || old(src).is_shared_wf(),
            old(dst).value() is Owned,
            old(src).loc() != old(dst).key(),
            after.union(cited).subset_of(infos.dom()),
            knowledge_of(observer, cited).subset_of(knows.set()),
            access.writes == ISet::empty().insert(old(dst).key()),
            access.reads.contains(old(src).loc()),
            access.reads.subset_of(ISet::empty().insert(old(src).loc()).insert(old(dst).key())),
            read_admitted(old(src).frontier(), closure_from(infos.map(), after.union(cited))),
            write_admitted(
                old(dst).value()->Owned_0,
                closure_from(infos.map(), after.union(cited)),
            ),
        ensures
            sub.instance_id() == self.id(),
            sub.key() == name.element(),
            sub.value() == (EventInfo {
                access,
                effect,
                read_guards: old(src).read_guards(),
                preds: closure_from(infos.map(), after.union(cited)),
            }),
            final(src).advanced_read(*old(src), name.element()),
            slot_owned(
                *final(dst),
                self.id(),
                old(dst).key(),
                Frontier { last_writer: Some(name.element()), readers: ISet::empty() },
            ),
    {
        let src_loc = src.loc();
        let dst_loc = dst.loc;
        let dst_fr = dst.map.map()[dst_loc]->Owned_0;
        let ev = name.element();
        let tracked mut dst_map = RegionMap::empty(self.id());
        vstd::modes::tracked_swap(&mut dst.map, &mut dst_map);
        match src {
            ReadPerm::Owned(src_tok) => {
                let src_fr = src_tok.map.map()[src_loc]->Owned_0;
                let frontiers = IMap::empty().insert(src_loc, src_fr).insert(dst_loc, dst_fr);
                let tracked mut src_map = RegionMap::empty(self.id());
                vstd::modes::tracked_swap(&mut src_tok.map, &mut src_map);
                let tracked src_kv = src_map.remove(src_loc);
                let tracked mut owned_map = dst_map;
                owned_map.insert(src_kv);
                let tracked shares = ReadShareMap::empty(self.id());
                assert(owned_map.map() =~= owned_slots(frontiers));
                assert(access.reads.subset_of(
                    frontiers.dom().union(IMap::<SlotLoc, TransferShare>::empty().dom()),
                ));
                let tracked (Tracked(sub), Tracked(mut new_owned), _) = self.inner.submit(
                    observer,
                    ev,
                    access,
                    effect,
                    IMap::empty(),
                    after,
                    cited,
                    infos.map(),
                    frontiers,
                    IMap::empty(),
                    infos,
                    knows,
                    name,
                    owned_map,
                    shares,
                );
                assert(new_owned.dom().contains(src_loc));
                let tracked new_src_kv = new_owned.remove(src_loc);
                let tracked mut new_src_map = RegionMap::empty(self.id());
                new_src_map.insert(new_src_kv);
                assert(new_src_map.dom() =~= ISet::empty().insert(src_loc));
                assert(new_owned.dom() =~= ISet::empty().insert(dst_loc));
                src_tok.map = new_src_map;
                dst.map = new_owned;
                sub
            },
            ReadPerm::Shared(cap) => {
                let share = cap.share;
                let guards = IMap::empty().insert(src_loc, share);
                let frontiers = IMap::empty().insert(dst_loc, dst_fr);
                let tracked mut shares_map = ReadShareMap::empty(self.id());
                vstd::modes::tracked_swap(&mut cap.map, &mut shares_map);
                assert(dst_map.map() =~= owned_slots(frontiers));
                assert(access.reads.subset_of(frontiers.dom().union(guards.dom())));
                let tracked (Tracked(sub), Tracked(new_owned), Tracked(new_shares)) =
                    self.inner.submit(
                    observer,
                    ev,
                    access,
                    effect,
                    guards,
                    after,
                    cited,
                    infos.map(),
                    frontiers,
                    shares_map.map(),
                    infos,
                    knows,
                    name,
                    dst_map,
                    shares_map,
                );
                assert(new_owned.dom() =~= ISet::empty().insert(dst_loc));
                assert(new_shares.dom() =~= ISet::empty().insert((src_loc, share)));
                dst.map = new_owned;
                cap.map = new_shares;
                sub
            },
        }
    }

    /// Submit a whole-slab write/reclaim/deregistration event across `0..count` owned slots.
    pub proof fn submit_slab_owned(
        tracked &self,
        observer: NodeId,
        node: NodeId,
        space: MemorySpace,
        count: usize,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        after: ISet<TransferEvent>,
        cited: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked knows: &Knows,
        tracked name: HistoryTokens::unused<TransferDomain>,
        tracked tokens: Map<usize, SlotToken>,
    ) -> (tracked result: (Tracked<Submitted>, Tracked<Map<usize, SlotToken>>))
        requires
            name.instance_id() == self.id(),
            name.element().observer == observer,
            infos.instance_id() == self.id(),
            knows.instance_id() == self.id(),
            after.union(cited).subset_of(infos.dom()),
            knowledge_of(observer, cited).subset_of(knows.set()),
            access.writes == slab_locs(node, space, count as nat),
            access.reads.subset_of(slab_locs(node, space, count as nat)),
            slab_tokens_valid(tokens, self.id(), node, space, count),
            forall|i: usize|
                i < count ==> write_admitted(
                    slot_frontier(#[trigger] tokens[i]),
                    closure_from(infos.map(), after.union(cited)),
                ),
        ensures
            result.0@.instance_id() == self.id(),
            result.0@.key() == name.element(),
            result.0@.value() == (EventInfo {
                access,
                effect,
                read_guards: IMap::empty(),
                preds: closure_from(infos.map(), after.union(cited)),
            }),
            slab_tokens_owned(
                result.1@,
                self.id(),
                node,
                space,
                count,
                Frontier { last_writer: Some(name.element()), readers: ISet::empty() },
            ),
    {
        batch::run_submit_slab_owned(
            self,
            observer,
            node,
            space,
            count,
            access,
            effect,
            after,
            cited,
            infos,
            knows,
            name,
            tokens,
        )
    }

    /// General multi-slot submission for scatter-gather transfers (finite owned
    /// slots plus an optional read-share capability).
    pub proof fn submit_multi(
        tracked &self,
        observer: NodeId,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        after: ISet<TransferEvent>,
        cited: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked knows: &Knows,
        tracked name: HistoryTokens::unused<TransferDomain>,
        tracked owned: Map<SlotLoc, SlotToken>,
        tracked shared: Option<ShareCap>,
    ) -> (tracked result: (
        Tracked<Submitted>,
        Tracked<Map<SlotLoc, SlotToken>>,
        Tracked<Option<ShareCap>>,
    ))
        requires
            name.instance_id() == self.id(),
            name.element().observer == observer,
            infos.instance_id() == self.id(),
            knows.instance_id() == self.id(),
            after.union(cited).subset_of(infos.dom()),
            knowledge_of(observer, cited).subset_of(knows.set()),
            forall|l: SlotLoc| #[trigger]
                owned.contains_key(l) ==> slot_is_owned(owned[l], self.id(), l),
            forall|l: SlotLoc| #[trigger]
                owned.contains_key(l) ==> admitted(
                    slot_frontier(owned[l]),
                    l,
                    access,
                    closure_from(infos.map(), after.union(cited)),
                ),
            forall|l: SlotLoc| #[trigger] access.writes.contains(l) ==> owned.contains_key(l),
            match shared {
                None => forall|l: SlotLoc| #[trigger]
                    access.reads.contains(l) ==> owned.contains_key(l),
                Some(cap) => {
                    &&& cap.well_formed()
                    &&& cap.instance_id() == self.id()
                    &&& !owned.contains_key(cap.loc())
                    &&& access.reads.contains(cap.loc())
                    &&& read_admitted(cap.frontier(), closure_from(infos.map(), after.union(cited)))
                    &&& forall|l: SlotLoc| #[trigger]
                        access.reads.contains(l) ==> owned.contains_key(l) || l == cap.loc()
                },
            },
        ensures
            result.0@.instance_id() == self.id(),
            result.0@.key() == name.element(),
            result.0@.value() == (EventInfo {
                access,
                effect,
                read_guards: match shared {
                    None => IMap::empty(),
                    Some(cap) => IMap::empty().insert(cap.loc(), cap.share()),
                },
                preds: closure_from(infos.map(), after.union(cited)),
            }),
            result.1@.dom() == owned.dom(),
            forall|l: SlotLoc| #[trigger]
                owned.contains_key(l) ==> slot_owned(
                    result.1@[l],
                    self.id(),
                    l,
                    advance(slot_frontier(owned[l]), l, name.element(), access),
                ),
            match (shared, result.2@) {
                (None, None) => true,
                (Some(old_cap), Some(new_cap)) => {
                    &&& new_cap.well_formed()
                    &&& new_cap.instance_id() == old_cap.instance_id()
                    &&& new_cap.loc() == old_cap.loc()
                    &&& new_cap.share() == old_cap.share()
                    &&& new_cap.is_cooperative() == old_cap.is_cooperative()
                    &&& new_cap.is_fenced() == old_cap.is_fenced()
                    &&& new_cap.is_unbound() == old_cap.is_unbound()
                    &&& new_cap.frontier() == (Frontier {
                        last_writer: old_cap.frontier().last_writer,
                        readers: old_cap.frontier().readers.insert(name.element()),
                    })
                },
                _ => false,
            },
    {
        batch::run_submit_multi(
            self,
            observer,
            access,
            effect,
            after,
            cited,
            infos,
            knows,
            name,
            owned,
            shared,
        )
    }

    pub(crate) proof fn observe(
        tracked &self,
        observer: NodeId,
        completed: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
    ) -> (tracked knowledge: Knows)
        requires
            infos.instance_id() == self.id(),
            completed.subset_of(infos.dom()),
        ensures
            knowledge.instance_id() == self.id(),
            knowledge.set() == knowledge_of(observer, closure_from(infos.map(), completed)),
    {
        self.inner.observe(observer, completed, infos.map(), infos)
    }

    pub proof fn observe_completed(
        tracked &self,
        observer: NodeId,
        completed: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked evidence: &ObservedEvents,
    ) -> (tracked knowledge: Knows)
        requires
            infos.instance_id() == self.id(),
            completed.subset_of(infos.dom()),
            evidence.observed(self.id(), observer, completed),
        ensures
            knowledge.instance_id() == self.id(),
            knowledge.set() == knowledge_of(observer, closure_from(infos.map(), completed)),
    {
        self.observe(observer, completed, infos)
    }

    #[allow(dead_code)]
    pub(crate) proof fn learn(
        tracked &self,
        receiver: NodeId,
        sender: NodeId,
        facts: ISet<TransferEvent>,
        tracked infos: &SubmittedMap,
        tracked knows: &Knows,
    ) -> (tracked knowledge: Knows)
        requires
            infos.instance_id() == self.id(),
            knows.instance_id() == self.id(),
            facts.subset_of(infos.dom()),
            knowledge_of(sender, facts).subset_of(knows.set()),
        ensures
            knowledge.instance_id() == self.id(),
            knowledge.set() == knowledge_of(receiver, closure_from(infos.map(), facts)),
    {
        self.inner.learn(receiver, sender, facts, infos.map(), infos, knows)
    }

    pub proof fn duplicate_submitted(
        tracked &self,
        tracked submitted: &SubmittedMap,
    ) -> (tracked out: SubmittedMap)
        requires
            submitted.instance_id() == self.id(),
        ensures
            out.instance_id() == self.id(),
            out.map() == submitted.map(),
    {
        self.inner.duplicate_submitted(submitted.map(), submitted)
    }

    pub proof fn duplicate_knows(tracked &self, tracked knows: &Knows) -> (tracked out: Knows)
        requires
            knows.instance_id() == self.id(),
        ensures
            out.instance_id() == self.id(),
            out.set() == knows.set(),
    {
        self.inner.duplicate_knows(knows.set(), knows)
    }

    pub proof fn duplicate_revoked(tracked &self, tracked revoked: &RevokedMap) -> (tracked out:
        RevokedMap)
        requires
            revoked.instance_id() == self.id(),
        ensures
            out.instance_id() == self.id(),
            out.map() == revoked.map(),
    {
        self.inner.duplicate_revoked(revoked.map(), revoked)
    }

    pub proof fn slot_writer_is_latest(
        tracked &self,
        w: TransferEvent,
        e: TransferEvent,
        tracked slot_tok: &mut SlotToken,
        tracked submitted: &SubmittedMap,
    )
        requires
            old(slot_tok).well_formed(),
            old(slot_tok).instance_id() == self.id(),
            submitted.instance_id() == self.id(),
            foundation::tracked::slot_base_frontier(old(slot_tok).value()).last_writer == Some(w),
            submitted.dom().contains(w),
            submitted.dom().contains(e),
            e != w,
            submitted.map()[e].access.writes.contains(old(slot_tok).loc()),
        ensures
            final(slot_tok).well_formed(),
            final(slot_tok).instance_id() == old(slot_tok).instance_id(),
            final(slot_tok).loc() == old(slot_tok).loc(),
            final(slot_tok).value() == old(slot_tok).value(),
            submitted.map()[w].preds.contains(e),
    {
        let loc = slot_tok.loc;
        let slot = slot_tok.value();
        assert(slot_tok.map.dom().contains(loc));
        let tracked kv = slot_tok.map.remove(loc);
        self.inner.slot_writer_is_latest(loc, slot, w, e, submitted.map(), &kv, submitted);
        slot_tok.map.insert(kv);
        assert(slot_tok.map.dom() =~= ISet::empty().insert(loc));
    }

    pub proof fn observe_share_bound(
        tracked &self,
        w: TransferEvent,
        tracked share: &mut ShareCap,
        tracked submitted: &SubmittedMap,
    ) -> (tracked result: RevokedMap)
        requires
            old(share).well_formed(),
            old(share).instance_id() == self.id(),
            submitted.instance_id() == self.id(),
            old(share).frontier().last_writer == Some(w),
            submitted.dom().contains(w),
        ensures
            final(share).well_formed(),
            final(share).instance_id() == old(share).instance_id(),
            final(share).loc() == old(share).loc(),
            final(share).share() == old(share).share(),
            final(share).frontier() == old(share).frontier(),
            final(share).is_cooperative() == old(share).is_cooperative(),
            final(share).is_fenced() == old(share).is_fenced(),
            final(share).is_unbound() == old(share).is_unbound(),
            result.instance_id() == self.id(),
            forall|e: TransferEvent| #[trigger]
                submitted.dom().contains(e) && submitted.map()[e].access.writes.contains(
                    old(share).loc(),
                ) && e != w ==> (submitted.map()[w].preds.contains(e) || (
                result.map().dom().contains((old(share).loc(), old(share).share()))
                    && submitted.map()[e].preds.contains(
                    result.map()[(old(share).loc(), old(share).share())],
                ))),
    {
        let loc = share.loc;
        let id = share.share;
        let fr = share.frontier();
        assert(share.map.dom().contains((loc, id)));
        let tracked kv = share.map.remove((loc, id));
        let tracked result = self.inner.observe_share_bound(
            loc,
            id,
            fr,
            w,
            submitted.map(),
            &kv,
            submitted,
        );
        share.map.insert(kv);
        assert(share.map.dom() =~= ISet::empty().insert((loc, id)));
        result
    }
}

pub fn clone_transfer_instance(Tracked(i): Tracked<&TransferInstance>) -> (out: Tracked<
    TransferInstance,
>)
    ensures
        out@ == *i,
{
    Tracked(i.duplicate())
}

} // verus!
