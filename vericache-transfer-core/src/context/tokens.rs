use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::set::group_set_lemmas,
};

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct Succeeded {
    ghost instance: InstanceId,
    ghost event: TransferEvent,
}

impl Succeeded {
    pub closed spec fn instance_id(self) -> InstanceId {
        self.instance
    }

    pub closed spec fn event(self) -> TransferEvent {
        self.event
    }

    pub(crate) proof fn mint(instance: InstanceId, event: TransferEvent) -> (tracked result: Self)
        ensures
            result.instance_id() == instance,
            result.event() == event,
    {
        Self { instance, event }
    }

    pub proof fn duplicate(tracked &self) -> (tracked copy: Self)
        ensures
            copy == *self,
    {
        Self { instance: self.instance, event: self.event }
    }
}

pub tracked struct SucceededMap {
    pub ghost instance_id: InstanceId,
    pub tracked witnesses: Map<TransferEvent, Succeeded>,
}

impl SucceededMap {
    pub open spec fn instance_id(self) -> InstanceId {
        self.instance_id
    }

    pub open spec fn dom(self) -> Set<TransferEvent> {
        self.witnesses.dom()
    }

    pub open spec fn contains(self, event: TransferEvent) -> bool {
        self.witnesses.contains_key(event)
    }

    pub open spec fn well_formed(self) -> bool {
        forall|e|
            self.contains(e) ==> #[trigger] self.witnesses[e].event() == e
                && self.witnesses[e].instance_id() == self.instance_id
    }

    pub proof fn empty(id: InstanceId) -> (tracked r: Self)
        ensures
            r.instance_id == id,
            r.well_formed(),
            r.witnesses == Map::empty(),
    {
        Self { instance_id: id, witnesses: Map::tracked_empty() }
    }

    pub proof fn insert(tracked &mut self, tracked witness: Succeeded)
        requires
            old(self).well_formed(),
            witness.instance_id() == old(self).instance_id,
        ensures
            final(self).well_formed(),
            final(self).instance_id == old(self).instance_id,
            final(self).witnesses == old(self).witnesses.insert(witness.event(), witness),
    {
        self.witnesses.tracked_insert(witness.event(), witness);
    }

    pub proof fn get(tracked &self, event: TransferEvent) -> (tracked r: &Succeeded)
        requires
            self.contains(event),
        ensures
            *r == self.witnesses[event],
    {
        self.witnesses.tracked_borrow(event)
    }

    pub proof fn duplicate_for(tracked &self, event: TransferEvent) -> (tracked r: Succeeded)
        requires
            self.contains(event),
        ensures
            r == self.witnesses[event],
    {
        self.get(event).duplicate()
    }

    pub proof fn duplicate(tracked &self) -> (tracked out: Self)
        requires
            self.well_formed(),
        ensures
            out == *self,
    {
        let tracked witnesses = duplicate_witnesses(&self.witnesses, self.witnesses.dom());
        assert(witnesses =~= self.witnesses);
        Self { instance_id: self.instance_id, witnesses }
    }
}

proof fn duplicate_witnesses(
    tracked src: &Map<TransferEvent, Succeeded>,
    dom: Set<TransferEvent>,
) -> (tracked out: Map<TransferEvent, Succeeded>)
    requires
        dom.subset_of(src.dom()),
    ensures
        out.dom() == dom,
        forall|e: TransferEvent| dom.contains(e) ==> #[trigger] out[e] == src[e],
    decreases dom.len(),
{
    if dom.is_empty() {
        let tracked m = Map::tracked_empty();
        assert(m.dom() =~= dom);
        m
    } else {
        let e = dom.choose();
        let rest = dom.remove(e);
        assert(rest.len() < dom.len());
        let tracked mut out = duplicate_witnesses(src, rest);
        let tracked dup = src.tracked_borrow(e).duplicate();
        out.tracked_insert(e, dup);
        assert(out.dom() =~= dom);
        out
    }
}

// ---------------------------------------------------------------------------
// Single-slot token (singleton RegionMap wrapper supporting `&mut SlotToken`)
// ---------------------------------------------------------------------------
/// Exclusive single-slot capability for one `SlotLoc`.
///
/// Internally holds a singleton `RegionMap` so callers can pass `&mut SlotToken`
/// directly to backend operations without moving tokens out of maps or error tuples.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct SlotToken {
    pub(super) ghost loc: SlotLoc,
    pub(super) tracked map: RegionMap,
}

impl SlotToken {
    pub open(super) spec fn well_formed(self) -> bool {
        self.map.dom() == ISet::empty().insert(self.loc)
    }

    pub open(super) spec fn instance_id(self) -> InstanceId {
        self.map.instance_id()
    }

    pub open(super) spec fn loc(self) -> SlotLoc {
        self.loc
    }

    pub open spec fn key(self) -> SlotLoc {
        self.loc()
    }

    pub open(super) spec fn value(self) -> Slot<TransferDomain> {
        self.map.map()[self.loc]
    }

    pub proof fn placeholder(id: InstanceId, loc: SlotLoc) -> (tracked tok: Self)
        ensures
            tok.instance_id() == id,
            tok.loc() == loc,
    {
        Self { loc, map: RegionMap::empty(id) }
    }

    pub(super) proof fn from_raw(tracked kv: HistoryTokens::region<TransferDomain>) -> (tracked tok:
        Self)
        ensures
            tok.well_formed(),
            tok.instance_id() == kv.instance_id(),
            tok.loc() == kv.key(),
            tok.value() == kv.value(),
    {
        let loc = kv.key();
        let tracked mut m = RegionMap::empty(kv.instance_id());
        m.insert(kv);
        assert(m.dom() =~= ISet::empty().insert(loc));
        Self { loc, map: m }
    }

    pub(super) proof fn into_raw(tracked self) -> (tracked kv: HistoryTokens::region<
        TransferDomain,
    >)
        requires
            self.well_formed(),
        ensures
            kv.instance_id() == self.instance_id(),
            kv.key() == self.loc(),
            kv.value() == self.value(),
    {
        let tracked mut m = self.map;
        let loc = self.loc;
        assert(m.dom().contains(loc));
        m.remove(loc)
    }

    pub(super) proof fn from_singleton_map(loc: SlotLoc, tracked map: RegionMap) -> (tracked tok:
        Self)
        requires
            map.dom() == ISet::empty().insert(loc),
        ensures
            tok.well_formed(),
            tok.instance_id() == map.instance_id(),
            tok.loc() == loc,
            tok.value() == map.map()[loc],
    {
        Self { loc, map }
    }

    pub(super) proof fn into_singleton_map(tracked self) -> (tracked map: RegionMap)
        requires
            self.well_formed(),
        ensures
            map.instance_id() == self.instance_id(),
            map.dom() == ISet::empty().insert(self.loc()),
            map.map()[self.loc()] == self.value(),
    {
        self.map
    }
}

// ---------------------------------------------------------------------------
// Single-slot predicates and admission helpers
// ---------------------------------------------------------------------------
pub open spec fn slot_is_owned(tok: SlotToken, id: InstanceId, loc: SlotLoc) -> bool {
    tok.well_formed() && tok.instance_id() == id && tok.key() == loc && tok.value() is Owned
}

pub open spec fn slot_owned(
    tok: SlotToken,
    id: InstanceId,
    loc: SlotLoc,
    fr: Frontier<TransferDomain>,
) -> bool {
    tok.well_formed() && tok.instance_id() == id && tok.key() == loc && tok.value() == Slot::Owned(
        fr,
    )
}

pub open spec fn slot_frontier(tok: SlotToken) -> Frontier<TransferDomain>
    recommends
        tok.value() is Owned,
{
    tok.value()->Owned_0
}

pub open spec fn slot_lent(
    tok: SlotToken,
    id: InstanceId,
    loc: SlotLoc,
    base: Frontier<TransferDomain>,
    shares: ISet<TransferShare>,
) -> bool {
    tok.well_formed() && tok.instance_id() == id && tok.key() == loc && tok.value() == (Slot::Lent {
        base,
        shares,
    })
}

pub open spec fn slot_is_lent_with(
    tok: SlotToken,
    id: InstanceId,
    loc: SlotLoc,
    share: TransferShare,
) -> bool {
    tok.well_formed() && tok.instance_id() == id && tok.key() == loc && is_lent_with(
        tok.value(),
        share,
    )
}

/// A read on a slot is admitted when its `last_writer` is in `ordered` (or absent).
pub open spec fn read_admitted(fr: Frontier<TransferDomain>, ordered: ISet<TransferEvent>) -> bool {
    ordered_or_absent(fr.last_writer, ordered)
}

/// A write on a slot is admitted when its `last_writer` and all `readers` are in `ordered`.
pub open spec fn write_admitted(
    fr: Frontier<TransferDomain>,
    ordered: ISet<TransferEvent>,
) -> bool {
    ordered_or_absent(fr.last_writer, ordered) && fr.readers.subset_of(ordered)
}

/// All `0..count` slots of a slab are owned with the uniform frontier `fr`.
pub open spec fn slab_tokens_owned(
    tokens: Map<usize, SlotToken>,
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    fr: Frontier<TransferDomain>,
) -> bool {
    &&& forall|i: usize| i < count <==> #[trigger] tokens.contains_key(i)
    &&& forall|i: usize|
        i < count ==> slot_owned(
            #[trigger] tokens[i],
            id,
            SlotLoc { node, space, slot_index: i as nat },
            fr,
        )
}

/// All `0..count` slots of a slab are owned (possibly with distinct frontiers).
pub open spec fn slab_tokens_valid(
    tokens: Map<usize, SlotToken>,
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    count: usize,
) -> bool {
    &&& forall|i: usize| i < count <==> #[trigger] tokens.contains_key(i)
    &&& forall|i: usize|
        i < count ==> slot_is_owned(
            #[trigger] tokens[i],
            id,
            SlotLoc { node, space, slot_index: i as nat },
        )
}

// ---------------------------------------------------------------------------
// Converged read-share capability: ShareCap + FenceRight + ReadPerm
// ---------------------------------------------------------------------------
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct FenceRight {
    pub(super) ghost instance: InstanceId,
    pub(super) ghost loc: SlotLoc,
    pub(super) ghost share: TransferShare,
}

impl FenceRight {
    pub open(super) spec fn instance_id(self) -> InstanceId {
        self.instance
    }

    pub open(super) spec fn loc(self) -> SlotLoc {
        self.loc
    }

    pub open(super) spec fn share(self) -> TransferShare {
        self.share
    }
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked enum ShareUsage {
    Unbound(FenceRight),
    Cooperative,
    Fenced,
}

/// Single-key read-share token with a choice of cooperative or fenced access.
///
/// Consuming an unbound share's right reserves it for cooperative reads.
/// Extracting the right makes the share usable through a fenceable grant.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct ShareCap {
    pub(super) ghost loc: SlotLoc,
    pub(super) ghost share: TransferShare,
    pub(super) tracked map: ReadShareMap,
    pub(super) tracked usage: ShareUsage,
}

impl ShareCap {
    pub open(super) spec fn well_formed(self) -> bool {
        self.map.dom() == ISet::empty().insert((self.loc, self.share))
    }

    pub open(super) spec fn instance_id(self) -> InstanceId {
        self.map.instance_id()
    }

    pub open(super) spec fn loc(self) -> SlotLoc {
        self.loc
    }

    pub open(super) spec fn share(self) -> TransferShare {
        self.share
    }

    pub open(super) spec fn frontier(self) -> Frontier<TransferDomain> {
        self.map.map()[(self.loc, self.share)]
    }

    pub open(super) spec fn usage_is_unbound(self) -> bool {
        self.usage matches ShareUsage::Unbound(r) && r.instance_id() == self.map.instance_id()
            && r.loc() == self.loc && r.share() == self.share
    }

    pub open(super) spec fn usage_is_cooperative(self) -> bool {
        self.usage is Cooperative
    }

    pub open(super) spec fn usage_is_fenced(self) -> bool {
        self.usage is Fenced
    }

    pub open spec fn is_unbound(self) -> bool {
        self.well_formed() && self.usage_is_unbound()
    }

    pub open spec fn is_cooperative(self) -> bool {
        self.well_formed() && self.usage_is_cooperative()
    }

    pub open spec fn is_fenced(self) -> bool {
        self.well_formed() && self.usage_is_fenced()
    }

    /// Consume the fence right to reserve the share for cooperative reads.
    pub proof fn into_cooperative(tracked self) -> (tracked out: Self)
        requires
            self.is_unbound(),
        ensures
            out.well_formed(),
            out.instance_id() == self.instance_id(),
            out.loc() == self.loc(),
            out.share() == self.share(),
            out.frontier() == self.frontier(),
            out.is_cooperative(),
    {
        Self { loc: self.loc, share: self.share, map: self.map, usage: ShareUsage::Cooperative }
    }

    /// Extract the right to establish a fenceable grant for the share.
    pub proof fn take_fence_right(tracked &mut self) -> (tracked right: FenceRight)
        requires
            old(self).is_unbound(),
        ensures
            final(self).well_formed(),
            final(self).instance_id() == old(self).instance_id(),
            final(self).loc() == old(self).loc(),
            final(self).share() == old(self).share(),
            final(self).frontier() == old(self).frontier(),
            final(self).is_fenced(),
            right.instance_id() == old(self).instance_id(),
            right.loc() == old(self).loc(),
            right.share() == old(self).share(),
    {
        let tracked mut taken = ShareUsage::Fenced;
        vstd::modes::tracked_swap(&mut self.usage, &mut taken);
        match taken {
            ShareUsage::Unbound(r) => r,
            _ => {
                assert(false);
                proof_from_false()
            },
        }
    }

    /// Restore the unbound share when the grant was not established.
    pub proof fn restore_fence_right(tracked &mut self, tracked right: FenceRight)
        requires
            old(self).is_fenced(),
            right.instance_id() == old(self).instance_id(),
            right.loc() == old(self).loc(),
            right.share() == old(self).share(),
        ensures
            final(self).well_formed(),
            final(self).instance_id() == old(self).instance_id(),
            final(self).loc() == old(self).loc(),
            final(self).share() == old(self).share(),
            final(self).frontier() == old(self).frontier(),
            final(self).is_unbound(),
    {
        self.usage = ShareUsage::Unbound(right);
    }

    /// Wrap a transferred read-share token as a fenced capability.
    pub(crate) proof fn from_fenced_token(tracked token: ShareToken) -> (tracked out: Self)
        ensures
            out.well_formed(),
            out.instance_id() == token.instance_id(),
            out.loc() == token.key().0,
            out.share() == token.key().1,
            out.frontier() == token.value(),
            out.is_fenced(),
    {
        let (loc, share) = token.key();
        let tracked mut map = ReadShareMap::empty(token.instance_id());
        map.insert(token);
        assert(map.dom() =~= ISet::empty().insert((loc, share)));
        Self { loc, share, map, usage: ShareUsage::Fenced }
    }

    /// Recover the underlying token when transferring a fenced share.
    pub(crate) proof fn into_fenced_token(tracked self) -> (tracked token: ShareToken)
        requires
            self.is_fenced(),
        ensures
            token.instance_id() == self.instance_id(),
            token.key() == (self.loc(), self.share()),
            token.value() == self.frontier(),
    {
        let tracked mut m = self.map;
        let key = (self.loc, self.share);
        assert(m.dom().contains(key));
        m.remove(key)
    }

    pub(super) proof fn into_singleton_map(tracked self) -> (tracked map: ReadShareMap)
        requires
            self.well_formed(),
        ensures
            map.instance_id() == self.instance_id(),
            map.dom() == ISet::empty().insert((self.loc(), self.share())),
            map.map()[(self.loc(), self.share())] == self.frontier(),
    {
        self.map
    }
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct ReturnedShare {
    pub(super) ghost instance: InstanceId,
    pub(super) ghost loc: SlotLoc,
    pub(super) ghost share: TransferShare,
}

impl ReturnedShare {
    pub open(super) spec fn instance_id(self) -> InstanceId {
        self.instance
    }

    pub open(super) spec fn loc(self) -> SlotLoc {
        self.loc
    }

    pub open(super) spec fn share(self) -> TransferShare {
        self.share
    }
}

/// Read permission for a single slot: either exclusive ownership or a read share.
pub tracked enum ReadPerm {
    Owned(SlotToken),
    Shared(ShareCap),
}

impl ReadPerm {
    pub open spec fn instance_id(self) -> InstanceId {
        match self {
            Self::Owned(t) => t.instance_id(),
            Self::Shared(c) => c.instance_id(),
        }
    }

    pub open spec fn loc(self) -> SlotLoc {
        match self {
            Self::Owned(t) => t.key(),
            Self::Shared(c) => c.loc(),
        }
    }

    pub open spec fn is_owned(self) -> bool {
        self matches Self::Owned(t) && t.well_formed() && t.value() is Owned
    }

    pub open spec fn is_shared_wf(self) -> bool {
        self matches Self::Shared(c) && c.well_formed()
    }

    pub open spec fn is_cooperative_shared(self) -> bool {
        self matches Self::Shared(c) && c.is_cooperative()
    }

    pub open spec fn is_fenced_shared(self) -> bool {
        self matches Self::Shared(c) && c.is_fenced()
    }

    /// Valid read permission for cooperative operations.
    pub open spec fn valid_cooperative(self, id: InstanceId, loc: SlotLoc) -> bool {
        self.instance_id() == id && self.loc() == loc && (self.is_owned()
            || self.is_cooperative_shared())
    }

    /// Valid read permission for operations that can use fenced shares.
    pub open spec fn valid_fenced(self, id: InstanceId, loc: SlotLoc) -> bool {
        self.instance_id() == id && self.loc() == loc && (self.is_owned()
            || self.is_fenced_shared())
    }

    pub open spec fn frontier(self) -> Frontier<TransferDomain> {
        match self {
            Self::Owned(t) => t.value()->Owned_0,
            Self::Shared(c) => c.frontier(),
        }
    }

    pub open spec fn read_guards(self) -> ReadGuards {
        match self {
            Self::Owned(_) => IMap::empty(),
            Self::Shared(c) => IMap::empty().insert(c.loc(), c.share()),
        }
    }

    pub open spec fn share_opt(self) -> Option<TransferShare> {
        match self {
            Self::Owned(_) => None,
            Self::Shared(c) => Some(c.share()),
        }
    }

    /// Post-state after submitting a read event `ev` on `self.loc()`.
    pub open spec fn advanced_read(self, old: Self, ev: TransferEvent) -> bool {
        &&& self.instance_id() == old.instance_id()
        &&& self.loc() == old.loc()
        &&& match (old, self) {
            (Self::Owned(o), Self::Owned(n)) => {
                &&& o.well_formed() && n.well_formed()
                &&& o.value() is Owned
                &&& n.value() == Slot::Owned(
                    Frontier {
                        last_writer: o.value()->Owned_0.last_writer,
                        readers: o.value()->Owned_0.readers.insert(ev),
                    },
                )
            },
            (Self::Shared(o), Self::Shared(n)) => {
                &&& o.well_formed() && n.well_formed()
                &&& n.share() == o.share()
                &&& n.is_cooperative() == o.is_cooperative()
                &&& n.is_fenced() == o.is_fenced()
                &&& n.is_unbound() == o.is_unbound()
                &&& n.frontier() == (Frontier {
                    last_writer: o.frontier().last_writer,
                    readers: o.frontier().readers.insert(ev),
                })
            },
            _ => false,
        }
    }

    pub proof fn into_owned(tracked self) -> (tracked tok: SlotToken)
        requires
            self is Owned,
        ensures
            tok == self->Owned_0,
    {
        match self {
            Self::Owned(t) => t,
            Self::Shared(_) => {
                assert(false);
                proof_from_false()
            },
        }
    }

    pub proof fn into_shared(tracked self) -> (tracked cap: ShareCap)
        requires
            self is Shared,
        ensures
            cap == self->Shared_0,
    {
        match self {
            Self::Shared(c) => c,
            Self::Owned(_) => {
                assert(false);
                proof_from_false()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Internal IMapToken <-> single-key token conversions (confined to `context`)
// ---------------------------------------------------------------------------
pub(super) proof fn take_keys<Key, Value, Token: KeyValueToken<Key, Value>>(
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

} // verus!
