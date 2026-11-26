use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// An infinite sequence of distinct identifiers. Even and odd indices give
/// independent supplies; advancing skips the identifier just consumed.
pub ghost struct NameSupply {
    pub name: spec_fn(nat) -> nat,
}

impl NameSupply {
    pub open spec fn distinct(self) -> bool {
        forall|i: nat, j: nat| #[trigger] (self.name)(i) == #[trigger] (self.name)(j) ==> i == j
    }

    pub open spec fn contains(self, id: nat) -> bool {
        exists|i: nat| #[trigger] (self.name)(i) == id
    }

    pub open spec fn tail(self) -> Self {
        Self { name: |i: nat| (self.name)(i + 1) }
    }

    pub open spec fn half(self, odd: bool) -> Self {
        Self {
            name: |i: nat|
                (self.name)(
                    2 * i + if odd {
                        1nat
                    } else {
                        0nat
                    },
                ),
        }
    }

    pub proof fn partition(self)
        requires
            self.distinct(),
        ensures
            self.half(true).distinct(),
            self.half(false).distinct(),
            self.tail().distinct(),
            forall|id: nat| #[trigger]
                self.half(true).contains(id) ==> self.contains(id) && !self.half(false).contains(
                    id,
                ),
            forall|id: nat| #[trigger]
                self.half(false).contains(id) ==> self.contains(id) && !self.half(true).contains(
                    id,
                ),
            forall|id: nat| #[trigger]
                self.tail().contains(id) ==> self.contains(id) && id != (self.name)(0),
    {
        assert forall|id: nat| #[trigger] self.half(true).contains(id) implies self.contains(id)
            && !self.half(false).contains(id) by {
            let i = choose|i: nat| #[trigger] (self.half(true).name)(i) == id;
            assert((self.name)(2 * i + 1) == id);
            if self.half(false).contains(id) {
                let j = choose|j: nat| #[trigger] (self.half(false).name)(j) == id;
                assert((self.name)(2 * j) == id);
            }
        }
        assert forall|id: nat| #[trigger] self.half(false).contains(id) implies self.contains(
            id,
        ) by {
            let j = choose|j: nat| #[trigger] (self.half(false).name)(j) == id;
            assert((self.name)(2 * j) == id);
        }
        assert forall|id: nat| #[trigger] self.tail().contains(id) implies self.contains(id) && id
            != (self.name)(0) by {
            let j = choose|j: nat| #[trigger] (self.tail().name)(j) == id;
            assert((self.name)(j + 1) == id);
        }
    }
}

proof fn take_slab_slots(
    tracked local_regions: &mut RegionMap,
    node: NodeId,
    space: MemorySpace,
    count: usize,
) -> (tracked slots: Map<usize, SlotToken>)
    requires
        forall|i: usize|
            i < count ==> #[trigger] old(local_regions).dom().contains(
                SlotLoc { node, space, slot_index: i as nat },
            ) && old(local_regions).map().index(SlotLoc { node, space, slot_index: i as nat })
                == Slot::Owned(untouched()),
    ensures
        final(local_regions).instance_id() == old(local_regions).instance_id(),
        slab_tokens_owned(slots, old(local_regions).instance_id(), node, space, count, untouched()),
        forall|l: SlotLoc|
            l.space != space || l.node != node || l.slot_index >= count ==> (
            #[trigger] final(local_regions).dom().contains(l) == old(local_regions).dom().contains(
                l,
            ) && (final(local_regions).dom().contains(l) ==> final(local_regions).map()[l] == old(
                local_regions,
            ).map()[l])),
    decreases count,
{
    if count == 0 {
        Map::tracked_empty()
    } else {
        let last = (count - 1) as usize;
        let loc = SlotLoc { node, space, slot_index: last as nat };
        assert(local_regions.dom().contains(loc));
        let tracked kv = local_regions.remove(loc);
        let tracked mut rest = take_slab_slots(local_regions, node, space, last);
        rest.tracked_insert(last, SlotToken::from_raw(kv));
        rest
    }
}

/// Keeps the unused slot locations private so a caller cannot reset the supply
/// after taking an allocation, including a zero-length allocation.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct AllocSource {
    pub(super) ghost node: NodeId,
    pub(super) ghost names: NameSupply,
    pub(super) tracked local_regions: RegionMap,
}

impl AllocSource {
    pub open(super) spec fn node(self) -> NodeId {
        self.node
    }

    pub open(super) spec fn names(self) -> NameSupply {
        self.names
    }

    pub open(super) spec fn locations(node: NodeId, names: NameSupply) -> ISet<SlotLoc> {
        ISet::new(|l: SlotLoc| l.node == node && names.contains(l.space.alloc_id()))
    }

    pub open(super) spec fn well_formed(self) -> bool {
        &&& self.names.distinct()
        &&& forall|l: SlotLoc| #[trigger]
            Self::locations(self.node, self.names).contains(l)
                ==> self.local_regions.dom().contains(l) && self.local_regions.map()[l]
                == Slot::Owned(untouched())
    }

    pub open(super) spec fn id(self) -> InstanceId {
        self.local_regions.instance_id()
    }

    pub open spec fn next_alloc_id(self) -> AllocId {
        (self.names().name)(0)
    }

    pub open spec fn space_for(self, kind: u16, device: DeviceId) -> MemorySpace {
        MemorySpace::new(kind, device, self.next_alloc_id())
    }

    pub open spec fn slot_loc_for(self, kind: u16, device: DeviceId, slot_index: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space_for(kind, device), slot_index }
    }

    pub open spec fn slot_ref_for(self, kind: u16, device: DeviceId, len: u64) -> SlotRef {
        SlotRef::full(self.slot_loc_for(kind, device, 0), len)
    }

    /// One single-slot allocation consumes the head name and returns one untouched owned slot.
    pub open spec fn allocated_slot(self, previous: Self, loc: SlotLoc, slot: SlotToken) -> bool {
        &&& self.well_formed() && self.id() == previous.id() && self.node() == previous.node()
        &&& self.names() == previous.names().tail() && !self.names().contains(
            previous.next_alloc_id(),
        )
        &&& slot_owned(slot, previous.id(), loc, untouched())
    }

    /// One slab allocation consumes the head name and returns `count` untouched owned slots.
    pub open spec fn allocated_slab(
        self,
        previous: Self,
        space: MemorySpace,
        count: usize,
        slots: Map<usize, SlotToken>,
    ) -> bool {
        &&& self.well_formed() && self.id() == previous.id() && self.node() == previous.node()
        &&& self.names() == previous.names().tail() && !self.names().contains(
            previous.next_alloc_id(),
        )
        &&& slab_tokens_owned(slots, previous.id(), previous.node(), space, count, untouched())
    }

    pub proof fn allocate_slot_in(tracked &mut self, space: MemorySpace) -> (tracked slot:
        SlotToken)
        requires
            old(self).well_formed(),
            space.alloc_id() == old(self).next_alloc_id(),
        ensures
            final(self).allocated_slot(
                *old(self),
                SlotLoc { node: old(self).node(), space, slot_index: 0 },
                slot,
            ),
    {
        let names = self.names;
        names.partition();
        assert(names.contains((names.name)(0)));
        let loc = SlotLoc { node: self.node, space, slot_index: 0 };
        assert(Self::locations(self.node, names).contains(loc));
        let tracked kv = self.local_regions.remove(loc);
        let tracked slot = SlotToken::from_raw(kv);
        self.names = names.tail();
        assert forall|l: SlotLoc| #[trigger]
            Self::locations(self.node, self.names).contains(
                l,
            ) implies self.local_regions.dom().contains(l) && self.local_regions.map()[l]
            == Slot::Owned(untouched()) by {
            assert(Self::locations(self.node, names).contains(l));
            assert(l != loc);
        }
        slot
    }

    pub proof fn allocate_slab_in(
        tracked &mut self,
        space: MemorySpace,
        count: usize,
    ) -> (tracked slots: Map<usize, SlotToken>)
        requires
            old(self).well_formed(),
            space.alloc_id() == old(self).next_alloc_id(),
        ensures
            final(self).allocated_slab(*old(self), space, count, slots),
    {
        let names = self.names;
        names.partition();
        assert(names.contains((names.name)(0)));
        assert forall|i: usize| i < count implies #[trigger] self.local_regions.dom().contains(
            SlotLoc { node: self.node, space, slot_index: i as nat },
        ) && self.local_regions.map().index(
            SlotLoc { node: self.node, space, slot_index: i as nat },
        ) == Slot::Owned(untouched()) by {
            let loc = SlotLoc { node: self.node, space, slot_index: i as nat };
            assert(Self::locations(self.node, names).contains(loc));
        }
        let tracked slots = take_slab_slots(&mut self.local_regions, self.node, space, count);
        self.names = names.tail();
        assert forall|l: SlotLoc| #[trigger]
            Self::locations(self.node, self.names).contains(
                l,
            ) implies self.local_regions.dom().contains(l) && self.local_regions.map()[l]
            == Slot::Owned(untouched()) by {
            assert(Self::locations(self.node, names).contains(l));
            assert(l.space.alloc_id() != space.alloc_id());
            assert(l.space != space);
        }
        slots
    }

    pub proof fn lemma_names_distinct(tracked &self)
        requires
            self.well_formed(),
        ensures
            self.names().distinct(),
    {
    }

    pub proof fn split(tracked &mut self) -> (tracked other: Self)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            other.well_formed(),
            final(self).id() == old(self).id(),
            other.id() == old(self).id(),
            final(self).node() == old(self).node(),
            other.node() == old(self).node(),
            final(self).names() == old(self).names().half(false),
            other.names() == old(self).names().half(true),
            forall|id: nat| #[trigger]
                other.names().contains(id) ==> old(self).names().contains(id)
                    && !final(self).names().contains(id),
            forall|id: nat| #[trigger]
                final(self).names().contains(id) ==> old(self).names().contains(id)
                    && !other.names().contains(id),
    {
        let names = self.names;
        names.partition();
        let take = Self::locations(self.node, names.half(true));
        assert forall|l: SlotLoc|
            take.contains(l) implies #[trigger] self.local_regions.dom().contains(l) by {
            assert(Self::locations(self.node, names).contains(l));
        }
        let tracked part = take_keys(&mut self.local_regions, take);
        self.names = names.half(false);
        assert forall|l: SlotLoc| #[trigger]
            Self::locations(self.node, self.names).contains(l) || take.contains(l) implies old(
            self,
        ).local_regions.map()[l] == Slot::Owned(untouched()) && old(
            self,
        ).local_regions.dom().contains(l) by {
            assert(Self::locations(self.node, names).contains(l));
        }
        assert forall|l: SlotLoc| #[trigger] take.contains(l) implies part.dom().contains(l)
            && part.map()[l] == Slot::Owned(untouched()) by {
            assert(Self::locations(self.node, names).contains(l));
        }
        Self { node: self.node, names: names.half(true), local_regions: part }
    }
}

proof fn take_names<E, T: ElementToken<E>>(
    tracked events: &mut ISetToken<E, T>,
    keys: ISet<E>,
) -> (tracked part: ISetToken<E, T>)
    requires
        keys.subset_of(old(events).set()),
    ensures
        part.instance_id() == old(events).instance_id(),
        final(events).instance_id() == old(events).instance_id(),
        part.set() == keys,
        final(events).set() == old(events).set().difference(keys),
{
    let id = events.instance_id();
    let tracked mut all = ISetToken::empty(id);
    vstd::modes::tracked_swap(events, &mut all);
    let tracked mut map = all.into_map();
    let tracked selected = map.tracked_remove_keys(keys);
    let tracked part = ISetToken::from_map(id, selected);
    *events = ISetToken::from_map(id, map);
    assert(events.set() =~= old(events).set().difference(keys));
    part
}

/// Exclusive sequence of event names assigned to one hardware source.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct EventSourceToken {
    ghost node: NodeId,
    ghost source: EventSource,
    ghost next_seq: nat,
    tracked unused: UnusedSet,
}

impl EventSourceToken {
    pub closed spec fn node(self) -> NodeId {
        self.node
    }

    pub closed spec fn source(self) -> EventSource {
        self.source
    }

    pub closed spec fn next_seq(self) -> nat {
        self.next_seq
    }

    pub closed spec fn id(self) -> InstanceId {
        self.unused.instance_id()
    }

    pub open spec fn next_event(self) -> TransferEvent {
        TransferEvent { observer: self.node(), source: self.source(), seq: self.next_seq() }
    }

    pub closed spec fn well_formed(self) -> bool {
        self.unused.set() == ISet::new(
            |e: TransferEvent|
                e.observer == self.node && e.source == self.source && e.seq >= self.next_seq,
        )
    }

    pub proof fn take_next(tracked &mut self) -> (tracked name: HistoryTokens::unused<
        TransferDomain,
    >)
        requires
            old(self).well_formed(),
        ensures
            name.element() == old(self).next_event(),
            name.instance_id() == old(self).id(),
            final(self).well_formed(),
            final(self).node() == old(self).node(),
            final(self).source() == old(self).source(),
            final(self).next_seq() == old(self).next_seq() + 1,
            final(self).id() == old(self).id(),
    {
        let event = self.next_event();
        let tracked name = self.unused.remove(event);
        self.next_seq = self.next_seq + 1;
        assert(self.unused.set() =~= ISet::new(
            |e: TransferEvent|
                e.observer == self.node && e.source == self.source && e.seq >= self.next_seq,
        ));
        name
    }

    /// The next event name of a well-formed source is not yet submitted anywhere.
    pub proof fn lemma_next_event_unsubmitted(
        tracked &self,
        tracked instance: &TransferInstance,
        tracked submitted: &SubmittedMap,
    )
        requires
            self.well_formed(),
            self.id() == instance.id(),
            submitted.instance_id() == instance.id(),
        ensures
            !submitted.dom().contains(self.next_event()),
    {
        let ev = self.next_event();
        assert(self.unused.set().contains(ev));
        instance.inner.unused_not_submitted(
            self.unused.set(),
            submitted.map(),
            &self.unused,
            submitted,
        );
    }

    /// Every event in `submitted` from this source has sequence strictly below `self.next_seq()`.
    pub proof fn lemma_all_future_unsubmitted(
        tracked &self,
        tracked instance: &TransferInstance,
        tracked submitted: &SubmittedMap,
    )
        requires
            self.well_formed(),
            self.id() == instance.id(),
            submitted.instance_id() == instance.id(),
        ensures
            forall|e: TransferEvent| #[trigger]
                submitted.dom().contains(e) && e.source == self.source() && e.observer
                    == self.node() ==> e.seq < self.next_seq(),
    {
        instance.inner.unused_not_submitted(
            self.unused.set(),
            submitted.map(),
            &self.unused,
            submitted,
        );
        assert forall|e: TransferEvent| #[trigger]
            submitted.dom().contains(e) && e.source == self.source() && e.observer
                == self.node() implies e.seq < self.next_seq() by {
            if e.seq >= self.next_seq() {
                assert(self.unused.set().contains(e));
            }
        }
    }
}

/// Splittable supply of source identifiers and their unused event names.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct EventSourceAllocator {
    pub(super) ghost node: NodeId,
    pub(super) ghost names: NameSupply,
    pub(super) tracked unused: UnusedSet,
}

impl EventSourceAllocator {
    pub open(super) spec fn node(self) -> NodeId {
        self.node
    }

    pub open(super) spec fn names(self) -> NameSupply {
        self.names
    }

    pub open(super) spec fn id(self) -> InstanceId {
        self.unused.instance_id()
    }

    pub open(super) spec fn events(node: NodeId, names: NameSupply) -> ISet<TransferEvent> {
        ISet::new(|e: TransferEvent| e.observer == node && names.contains(e.source.id()))
    }

    pub open(super) spec fn well_formed(self) -> bool {
        self.names.distinct() && Self::events(self.node, self.names).subset_of(self.unused.set())
    }

    pub proof fn lemma_names_distinct(tracked &self)
        requires
            self.well_formed(),
        ensures
            self.names().distinct(),
    {
    }

    pub proof fn mint(tracked &mut self, kind: EventSourceKind) -> (tracked token: EventSourceToken)
        requires
            old(self).well_formed(),
        ensures
            token.well_formed(),
            token.node() == old(self).node(),
            token.source() == kind.source((old(self).names().name)(0)),
            old(self).names().contains(token.source().id()),
            !final(self).names().contains(token.source().id()),
            token.next_seq() == 0,
            token.id() == old(self).id(),
            final(self).well_formed(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).names() == old(self).names().tail(),
            forall|id: nat| #[trigger]
                final(self).names().contains(id) ==> old(self).names().contains(id),
    {
        let names = self.names;
        names.partition();
        assert(names.contains((names.name)(0)));
        let source = kind.source((names.name)(0));
        let keys = ISet::new(|e: TransferEvent| e.observer == self.node && e.source == source);
        assert(keys.subset_of(Self::events(self.node, names)));
        let tracked unused = take_names(&mut self.unused, keys);
        self.names = names.tail();
        assert(Self::events(self.node, self.names).subset_of(Self::events(self.node, names)));
        assert(unused.set() =~= ISet::new(
            |e: TransferEvent| e.observer == self.node && e.source == source && e.seq >= 0,
        ));
        EventSourceToken { node: self.node, source, next_seq: 0, unused }
    }

    pub proof fn split(tracked &mut self) -> (tracked other: Self)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            other.well_formed(),
            final(self).node() == old(self).node(),
            other.node() == old(self).node(),
            final(self).id() == old(self).id(),
            other.id() == old(self).id(),
            final(self).names() == old(self).names().half(false),
            other.names() == old(self).names().half(true),
            forall|id: nat| #[trigger]
                other.names().contains(id) ==> old(self).names().contains(id)
                    && !final(self).names().contains(id),
            forall|id: nat| #[trigger]
                final(self).names().contains(id) ==> old(self).names().contains(id)
                    && !other.names().contains(id),
    {
        let names = self.names;
        names.partition();
        let keys = Self::events(self.node, names.half(true));
        assert(keys.subset_of(Self::events(self.node, names)));
        let tracked unused = take_names(&mut self.unused, keys);
        self.names = names.half(false);
        assert(Self::events(self.node, self.names).subset_of(Self::events(self.node, names)));
        Self { node: self.node, names: names.half(true), unused }
    }
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct ShareSource {
    pub(super) ghost node: NodeId,
    pub(super) ghost names: NameSupply,
    pub(super) tracked unused: UnusedShareSet,
}

impl ShareSource {
    pub open(super) spec fn node(self) -> NodeId {
        self.node
    }

    pub open(super) spec fn names(self) -> NameSupply {
        self.names
    }

    pub open(super) spec fn id(self) -> InstanceId {
        self.unused.instance_id()
    }

    pub open(super) spec fn shares(node: NodeId, names: NameSupply) -> ISet<TransferShare> {
        ISet::new(|s: TransferShare| s.owner == node && names.contains(s.seq))
    }

    pub open(super) spec fn well_formed(self) -> bool {
        self.names.distinct() && Self::shares(self.node, self.names).subset_of(self.unused.set())
    }

    pub open spec fn next_share(self) -> TransferShare {
        TransferShare { owner: self.node(), seq: (self.names().name)(0) }
    }

    pub proof fn lemma_names_distinct(tracked &self)
        requires
            self.well_formed(),
        ensures
            self.names().distinct(),
    {
    }

    pub proof fn allocate(tracked &mut self) -> (tracked token: UnusedShare)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).names() == old(self).names().tail(),
            !final(self).names().contains(old(self).next_share().seq),
            token.element() == old(self).next_share(),
            token.instance_id() == old(self).id(),
    {
        let names = self.names;
        names.partition();
        assert(names.contains((names.name)(0)));
        let share = self.next_share();
        assert(Self::shares(self.node, names).contains(share));
        let tracked token = self.unused.remove(share);
        self.names = names.tail();
        assert(Self::shares(self.node, self.names).subset_of(Self::shares(self.node, names)));
        token
    }

    pub proof fn split(tracked &mut self) -> (tracked other: Self)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            other.well_formed(),
            final(self).node() == old(self).node(),
            other.node() == old(self).node(),
            final(self).id() == old(self).id(),
            other.id() == old(self).id(),
            final(self).names() == old(self).names().half(false),
            other.names() == old(self).names().half(true),
            forall|id: nat| #[trigger]
                other.names().contains(id) ==> old(self).names().contains(id)
                    && !final(self).names().contains(id),
            forall|id: nat| #[trigger]
                final(self).names().contains(id) ==> old(self).names().contains(id)
                    && !other.names().contains(id),
    {
        let names = self.names;
        names.partition();
        let keys = Self::shares(self.node, names.half(true));
        assert(keys.subset_of(Self::shares(self.node, names)));
        let tracked unused = take_names(&mut self.unused, keys);
        self.names = names.half(false);
        assert(Self::shares(self.node, self.names).subset_of(Self::shares(self.node, names)));
        Self { node: self.node, names: names.half(true), unused }
    }
}

/// Cluster-wide write-once version registry types: `(NodeId, u64) -> Seq<u8>`.
pub type VersionKey = (NodeId, u64);

pub type VersionUnused = foundation::write_once::WriteOnceTokens::unused<VersionKey, Seq<u8>>;

pub type VersionUnusedSet = foundation::write_once::WriteOnceTokens::unused_set<
    VersionKey,
    Seq<u8>,
>;

pub type VersionBinding = foundation::write_once::WriteOnceTokens::bindings<VersionKey, Seq<u8>>;

/// Identity provisioned for the shared write-once version registry of this cluster.
pub uninterp spec fn cluster_version_instance_id() -> InstanceId;

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct VersionInstance {
    pub(crate) tracked inner: foundation::write_once::WriteOnceTokens::Instance<
        VersionKey,
        Seq<u8>,
    >,
}

impl VersionInstance {
    pub closed spec fn id(&self) -> InstanceId {
        self.inner.id()
    }

    pub proof fn duplicate(tracked &self) -> (tracked out: Self)
        ensures
            out == *self,
    {
        Self { inner: self.inner }
    }

    pub proof fn bind(
        tracked &self,
        key: VersionKey,
        bytes: Seq<u8>,
        tracked unused: VersionUnused,
    ) -> (tracked b: VersionBinding)
        requires
            unused.instance_id() == self.id(),
            unused.element() == key,
        ensures
            b.instance_id() == self.id(),
            b.key() == key,
            b.value() == bytes,
    {
        use foundation::write_once::WriteOnceTransitions;
        self.inner.bind(key, bytes, unused)
    }

    pub proof fn unused_key_differs_from_binding(
        tracked &self,
        tracked unused: &VersionUnused,
        tracked binding: &VersionBinding,
    )
        requires
            unused.instance_id() == self.id(),
            binding.instance_id() == self.id(),
        ensures
            unused.element() != binding.key(),
    {
        foundation::write_once::unused_key_differs_from_binding(&self.inner, unused, binding);
    }
}

pub fn clone_version_instance(Tracked(i): Tracked<&VersionInstance>) -> (out: Tracked<
    VersionInstance,
>)
    ensures
        out@ == *i,
{
    Tracked(i.duplicate())
}

pub proof fn version_binding_agree(tracked a: &VersionBinding, tracked b: &VersionBinding)
    requires
        a.instance_id() == b.instance_id(),
        a.key() == b.key(),
    ensures
        a.value() == b.value(),
{
    foundation::write_once::binding_agree(a, b);
}

pub proof fn version_binding_duplicate(tracked a: &VersionBinding) -> (tracked b: VersionBinding)
    ensures
        b == *a,
{
    foundation::write_once::binding_duplicate(a)
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct VersionSource {
    ghost node: NodeId,
    ghost min_seq: nat,
    ghost max_seq: Option<nat>,
    tracked unused: VersionUnusedSet,
}

impl VersionSource {
    pub closed spec fn node(self) -> NodeId {
        self.node
    }

    pub closed spec fn min_seq(self) -> nat {
        self.min_seq
    }

    pub closed spec fn max_seq(self) -> Option<nat> {
        self.max_seq
    }

    pub closed spec fn id(self) -> InstanceId {
        self.unused.instance_id()
    }

    spec fn range_keys(node: NodeId, min_seq: nat, max_seq: Option<nat>) -> ISet<VersionKey> {
        ISet::new(
            |k: VersionKey|
                k.0 == node && k.1 as nat >= min_seq && (match max_seq {
                    Some(lim) => (k.1 as nat) < lim,
                    None => true,
                }),
        )
    }

    pub closed spec fn well_formed(self) -> bool {
        Self::range_keys(self.node, self.min_seq, self.max_seq).subset_of(self.unused.set())
    }

    pub proof fn take_seq(tracked &mut self, seq: u64) -> (tracked tok: VersionUnused)
        requires
            old(self).well_formed(),
            seq as nat >= old(self).min_seq(),
            old(self).max_seq() matches Some(lim) ==> (seq as nat) < lim,
        ensures
            tok.instance_id() == old(self).id(),
            tok.element() == (old(self).node(), seq),
            final(self).well_formed(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).min_seq() == (seq as nat) + 1,
            final(self).max_seq() == old(self).max_seq(),
    {
        let key: VersionKey = (self.node, seq);
        assert(Self::range_keys(self.node, self.min_seq, self.max_seq).contains(key));
        let tracked tok = self.unused.remove(key);
        let old_min = self.min_seq;
        self.min_seq = (seq as nat) + 1;
        assert forall|k: VersionKey| #[trigger]
            Self::range_keys(self.node, self.min_seq, self.max_seq).contains(
                k,
            ) implies self.unused.set().contains(k) by {
            assert(Self::range_keys(self.node, old_min, self.max_seq).contains(k));
            assert(k != key);
        }
        tok
    }

    pub proof fn split_range(tracked &mut self, quota: u64) -> (tracked sub: Self)
        requires
            old(self).well_formed(),
            quota > 0,
            old(self).max_seq() matches Some(lim) ==> old(self).min_seq() + quota <= lim,
        ensures
            sub.well_formed(),
            sub.id() == old(self).id(),
            sub.node() == old(self).node(),
            sub.min_seq() == old(self).min_seq(),
            sub.max_seq() == Some(old(self).min_seq() + quota as nat),
            final(self).well_formed(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).min_seq() == old(self).min_seq() + quota as nat,
            final(self).max_seq() == old(self).max_seq(),
    {
        let old_min = self.min_seq;
        let new_min = old_min + quota as nat;
        let sub_max = Some(new_min);
        let keys = Self::range_keys(self.node, old_min, sub_max);
        assert(keys.subset_of(Self::range_keys(self.node, old_min, self.max_seq)));
        let tracked sub_unused = take_names(&mut self.unused, keys);
        self.min_seq = new_min;
        assert forall|k: VersionKey| #[trigger]
            Self::range_keys(self.node, self.min_seq, self.max_seq).contains(
                k,
            ) implies self.unused.set().contains(k) by {
            assert(Self::range_keys(self.node, old_min, self.max_seq).contains(k));
            assert(!keys.contains(k));
        }
        Self { node: self.node, min_seq: old_min, max_seq: sub_max, unused: sub_unused }
    }
}

/// The local share of the cluster history, before it is split among workers.
pub struct NodeContext {
    pub node: NodeId,
    pub instance: Tracked<TransferInstance>,
    pub unused: Tracked<UnusedSet>,
    pub unused_shares: Tracked<UnusedShareSet>,
    pub local_regions: Tracked<RegionMap>,
    pub next_alloc_id: Ghost<AllocId>,
    pub next_source_id: Ghost<nat>,
    pub version_instance: Tracked<VersionInstance>,
    pub unused_versions: Tracked<VersionUnusedSet>,
}

/// Identity provisioned for the shared history of this cluster.
pub uninterp spec fn cluster_instance_id() -> InstanceId;

impl NodeContext {
    pub open spec fn initial_partition(&self) -> bool {
        &&& self.instance@.id() == cluster_instance_id()
        &&& self.next_alloc_id@ == 0 && self.next_source_id@ == 0
        &&& self.unused@.instance_id() == self.instance@.id()
        &&& self.local_regions@.instance_id() == self.instance@.id()
        &&& self.unused_shares@.instance_id() == self.instance@.id()
        &&& self.unused_shares@.set() == ISet::new(|s: TransferShare| s.owner == self.node)
        &&& self.unused@.set() == ISet::new(|e: TransferEvent| e.observer == self.node)
        &&& self.local_regions@.map() == IMap::new(
            |l: SlotLoc| l.node == self.node,
            |l| Slot::Owned(untouched()),
        )
        &&& self.version_instance@.id() == cluster_version_instance_id()
        &&& self.unused_versions@.instance_id() == self.version_instance@.id()
        &&& self.unused_versions@.set() == ISet::new(|k: VersionKey| k.0 == self.node)
    }

    pub fn into_sources(self) -> (result: (
        Tracked<TransferInstance>,
        Tracked<AllocSource>,
        Tracked<EventSourceAllocator>,
        Tracked<ShareSource>,
    ))
        requires
            self.initial_partition(),
        ensures
            result.0@ == self.instance@,
            result.1@.well_formed(),
            result.2@.well_formed(),
            result.3@.well_formed(),
            result.1@.node() == self.node,
            result.2@.node() == self.node,
            result.3@.node() == self.node,
            result.1@.id() == self.instance@.id(),
            result.2@.id() == self.instance@.id(),
            result.3@.id() == self.instance@.id(),
            result.1@.names().name == (|i: nat| i),
            result.2@.names().name == (|i: nat| i),
            result.3@.names().name == (|i: nat| i),
    {
        let (instance, alloc, events, shares, _, _) = self.into_sources_with_versions();
        (instance, alloc, events, shares)
    }

    pub fn into_sources_with_versions(self) -> (result: (
        Tracked<TransferInstance>,
        Tracked<AllocSource>,
        Tracked<EventSourceAllocator>,
        Tracked<ShareSource>,
        Tracked<VersionInstance>,
        Tracked<VersionSource>,
    ))
        requires
            self.initial_partition(),
        ensures
            result.0@ == self.instance@,
            result.1@.well_formed(),
            result.2@.well_formed(),
            result.3@.well_formed(),
            result.1@.node() == self.node,
            result.2@.node() == self.node,
            result.3@.node() == self.node,
            result.1@.id() == self.instance@.id(),
            result.2@.id() == self.instance@.id(),
            result.3@.id() == self.instance@.id(),
            result.1@.names().name == (|i: nat| i),
            result.2@.names().name == (|i: nat| i),
            result.3@.names().name == (|i: nat| i),
            result.4@ == self.version_instance@,
            result.5@.well_formed(),
            result.5@.node() == self.node,
            result.5@.id() == self.version_instance@.id(),
            result.5@.min_seq() == 0,
            result.5@.max_seq() is None,
    {
        let ghost names = NameSupply { name: |i: nat| i };
        let tracked alloc = AllocSource {
            node: self.node,
            names,
            local_regions: self.local_regions.get(),
        };
        let tracked events = EventSourceAllocator {
            node: self.node,
            names,
            unused: self.unused.get(),
        };
        let tracked shares = ShareSource {
            node: self.node,
            names,
            unused: self.unused_shares.get(),
        };
        let tracked versions = VersionSource {
            node: self.node,
            min_seq: 0,
            max_seq: None,
            unused: self.unused_versions.get(),
        };
        (
            self.instance,
            Tracked(alloc),
            Tracked(events),
            Tracked(shares),
            self.version_instance,
            Tracked(versions),
        )
    }
}

/// Exclusive upper bound (`MAX_VERSION_SEQ + 1`) for shard version sequence ranges.
pub const VERSION_SEQ_LIMIT: u64 = 0x0001_0000_0000_0000;

/// Disjoint per-shard capability bundle carved from a node's cluster partition.
///
/// Guarantees pairwise-disjoint allocation/event-source/share `NameSupply`s and
/// strictly positive, non-overlapping version ranges `[version_start, version_limit)`
/// starting at `>= 1` so `VersionId::NONE` (`seq == 0`) is never minted.
pub struct ShardSources {
    pub node: NodeId,
    pub instance: Tracked<TransferInstance>,
    pub alloc: Tracked<AllocSource>,
    pub events: Tracked<EventSourceAllocator>,
    pub shares: Tracked<ShareSource>,
    pub version_instance: Tracked<VersionInstance>,
    pub versions: Tracked<VersionSource>,
    pub version_start: u64,
    pub version_limit: u64,
}

impl ShardSources {
    pub open spec fn well_formed(&self) -> bool {
        &&& self.instance@.id() == cluster_instance_id()
        &&& self.alloc@.well_formed()
        &&& self.events@.well_formed()
        &&& self.shares@.well_formed()
        &&& self.alloc@.id() == self.instance@.id()
        &&& self.events@.id() == self.instance@.id()
        &&& self.shares@.id() == self.instance@.id()
        &&& self.alloc@.node() == self.node
        &&& self.events@.node() == self.node
        &&& self.shares@.node() == self.node
        &&& self.alloc@.names().distinct()
        &&& self.events@.names().distinct()
        &&& self.shares@.names().distinct()
        &&& self.version_instance@.id() == cluster_version_instance_id()
        &&& self.versions@.well_formed()
        &&& self.versions@.id() == self.version_instance@.id()
        &&& self.versions@.node() == self.node
        &&& 1 <= self.version_start < self.version_limit <= VERSION_SEQ_LIMIT
        &&& self.versions@.min_seq() == self.version_start as nat
        &&& self.versions@.max_seq() == Some(self.version_limit as nat)
    }

    pub open spec fn disjoint_from(&self, other: &Self) -> bool {
        &&& self.node == other.node
        &&& self.instance@.id() == other.instance@.id()
        &&& self.version_instance@.id() == other.version_instance@.id()
        &&& (forall|id: nat| #[trigger]
            self.alloc@.names().contains(id) ==> !other.alloc@.names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.events@.names().contains(id) ==> !other.events@.names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.shares@.names().contains(id) ==> !other.shares@.names().contains(id))
        &&& (self.version_limit <= other.version_start || other.version_limit <= self.version_start)
    }

    pub open spec fn disjoint_from_allocator(&self, alloc: &ShardAllocator) -> bool {
        &&& self.node == alloc.node()
        &&& self.instance@.id() == alloc.instance_id()
        &&& self.version_instance@.id() == alloc.version_instance_id()
        &&& (forall|id: nat| #[trigger]
            self.alloc@.names().contains(id) ==> !alloc.alloc_names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.events@.names().contains(id) ==> !alloc.event_names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.shares@.names().contains(id) ==> !alloc.share_names().contains(id))
        &&& self.version_limit <= alloc.next_version_seq()
    }

    pub open spec fn contained_in_allocator(&self, alloc: &ShardAllocator) -> bool {
        &&& self.node == alloc.node()
        &&& self.instance@.id() == alloc.instance_id()
        &&& self.version_instance@.id() == alloc.version_instance_id()
        &&& (forall|id: nat| #[trigger]
            self.alloc@.names().contains(id) ==> alloc.alloc_names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.events@.names().contains(id) ==> alloc.event_names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.shares@.names().contains(id) ==> alloc.share_names().contains(id))
        &&& alloc.next_version_seq() <= self.version_start
        &&& self.version_limit <= alloc.version_limit()
    }
}

/// Progressive multi-shard partitioner over a node's `TransferInstance`,
/// `AllocSource`, `EventSourceAllocator`, `ShareSource`, and bounded `VersionSource`.
pub struct ShardAllocator {
    node: NodeId,
    instance: Tracked<TransferInstance>,
    alloc: Tracked<AllocSource>,
    events: Tracked<EventSourceAllocator>,
    shares: Tracked<ShareSource>,
    version_instance: Tracked<VersionInstance>,
    versions: Tracked<VersionSource>,
    next_version_seq: u64,
    version_limit: u64,
}

impl ShardAllocator {
    pub closed spec fn node(&self) -> NodeId {
        self.node
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.instance@.id()
    }

    pub closed spec fn version_instance_id(&self) -> InstanceId {
        self.version_instance@.id()
    }

    pub closed spec fn next_version_seq(&self) -> u64 {
        self.next_version_seq
    }

    pub closed spec fn version_limit(&self) -> u64 {
        self.version_limit
    }

    pub open spec fn remaining_versions(&self) -> nat {
        (self.version_limit() - self.next_version_seq()) as nat
    }

    pub closed spec fn alloc_names(&self) -> NameSupply {
        self.alloc@.names()
    }

    pub closed spec fn event_names(&self) -> NameSupply {
        self.events@.names()
    }

    pub closed spec fn share_names(&self) -> NameSupply {
        self.shares@.names()
    }

    pub closed spec fn well_formed(&self) -> bool {
        &&& self.instance@.id() == cluster_instance_id()
        &&& self.alloc@.well_formed()
        &&& self.events@.well_formed()
        &&& self.shares@.well_formed()
        &&& self.alloc@.id() == self.instance@.id()
        &&& self.events@.id() == self.instance@.id()
        &&& self.shares@.id() == self.instance@.id()
        &&& self.alloc@.node() == self.node
        &&& self.events@.node() == self.node
        &&& self.shares@.node() == self.node
        &&& self.version_instance@.id() == cluster_version_instance_id()
        &&& self.versions@.well_formed()
        &&& self.versions@.id() == self.version_instance@.id()
        &&& self.versions@.node() == self.node
        &&& 1 <= self.next_version_seq <= self.version_limit <= VERSION_SEQ_LIMIT
        &&& self.versions@.min_seq() == self.next_version_seq as nat
        &&& self.versions@.max_seq() == Some(self.version_limit as nat)
    }

    pub open spec fn contained_in_allocator(&self, older: &Self) -> bool {
        &&& self.node() == older.node()
        &&& self.instance_id() == older.instance_id()
        &&& self.version_instance_id() == older.version_instance_id()
        &&& (forall|id: nat| #[trigger]
            self.alloc_names().contains(id) ==> older.alloc_names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.event_names().contains(id) ==> older.event_names().contains(id))
        &&& (forall|id: nat| #[trigger]
            self.share_names().contains(id) ==> older.share_names().contains(id))
        &&& older.next_version_seq() <= self.next_version_seq()
        &&& self.version_limit() <= older.version_limit()
    }

    pub fn node_id(&self) -> (n: NodeId)
        ensures
            n == self.node(),
    {
        self.node
    }

    pub fn next_version_seq_val(&self) -> (n: u64)
        ensures
            n == self.next_version_seq(),
    {
        self.next_version_seq
    }

    pub fn version_limit_val(&self) -> (n: u64)
        ensures
            n == self.version_limit(),
    {
        self.version_limit
    }

    pub fn remaining_version_budget(&self) -> (n: u64)
        requires
            self.well_formed(),
        ensures
            n as nat == self.remaining_versions(),
    {
        self.version_limit - self.next_version_seq
    }

    /// Construct a node-wide `ShardAllocator` reserving `[1, VERSION_SEQ_LIMIT)` so
    /// `VersionId::NONE` (`seq == 0`) can never be minted by any shard.
    pub fn from_node_context(ctx: NodeContext) -> (a: Self)
        requires
            ctx.initial_partition(),
        ensures
            a.well_formed(),
            a.node() == ctx.node,
            a.instance_id() == ctx.instance@.id(),
            a.version_instance_id() == ctx.version_instance@.id(),
            a.next_version_seq() == 1,
            a.version_limit() == VERSION_SEQ_LIMIT,
    {
        let node = ctx.node;
        let (instance, alloc, events, shares, version_instance, Tracked(mut full_versions)) =
            ctx.into_sources_with_versions();
        let tracked versions;
        proof {
            let tracked _reserved_zero = full_versions.take_seq(0);
            versions = full_versions.split_range((VERSION_SEQ_LIMIT - 1) as u64);
        }
        Self {
            node,
            instance,
            alloc,
            events,
            shares,
            version_instance,
            versions: Tracked(versions),
            next_version_seq: 1,
            version_limit: VERSION_SEQ_LIMIT,
        }
    }

    /// Sub-partition an existing `ShardSources` into smaller sub-shards.
    pub fn from_shard(shard: ShardSources) -> (a: Self)
        requires
            shard.well_formed(),
        ensures
            a.well_formed(),
            a.node() == shard.node,
            a.instance_id() == shard.instance@.id(),
            a.version_instance_id() == shard.version_instance@.id(),
            a.next_version_seq() == shard.version_start,
            a.version_limit() == shard.version_limit,
            a.alloc_names() == shard.alloc@.names(),
            a.event_names() == shard.events@.names(),
            a.share_names() == shard.shares@.names(),
    {
        Self {
            node: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            version_instance: shard.version_instance,
            versions: shard.versions,
            next_version_seq: shard.version_start,
            version_limit: shard.version_limit,
        }
    }

    /// Carve one disjoint `ShardSources` with `version_quota` sequences `[start, start + quota)`.
    pub fn split_shard(&mut self, version_quota: u64) -> (r: Option<ShardSources>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).node() == old(self).node(),
            final(self).instance_id() == old(self).instance_id(),
            final(self).version_instance_id() == old(self).version_instance_id(),
            final(self).version_limit() == old(self).version_limit(),
            match r {
                None => {
                    &&& version_quota == 0 || version_quota > old(self).remaining_versions()
                    &&& *final(self) == *old(self)
                },
                Some(shard) => {
                    &&& 0 < version_quota <= old(self).remaining_versions()
                    &&& shard.well_formed()
                    &&& shard.node == old(self).node()
                    &&& shard.instance@.id() == old(self).instance_id()
                    &&& shard.version_instance@.id() == old(self).version_instance_id()
                    &&& shard.version_start == old(self).next_version_seq()
                    &&& shard.version_limit == old(self).next_version_seq() + version_quota
                    &&& final(self).next_version_seq() == old(self).next_version_seq()
                        + version_quota
                    &&& shard.disjoint_from_allocator(&*final(self))
                    &&& shard.contained_in_allocator(&*old(self))
                    &&& final(self).contained_in_allocator(&*old(self))
                },
            },
    {
        if version_quota == 0 || version_quota > self.version_limit - self.next_version_seq {
            return None;
        }
        let version_start = self.next_version_seq;
        let version_limit = self.next_version_seq + version_quota;
        let tracked shard_alloc;
        let tracked shard_events;
        let tracked shard_shares;
        let tracked shard_versions;
        let tracked shard_instance;
        let tracked shard_ver_instance;
        proof {
            shard_alloc = self.alloc.borrow_mut().split();
            shard_events = self.events.borrow_mut().split();
            shard_shares = self.shares.borrow_mut().split();
            shard_versions = self.versions.borrow_mut().split_range(version_quota);
            shard_instance = self.instance.borrow().duplicate();
            shard_ver_instance = self.version_instance.borrow().duplicate();
        }
        self.next_version_seq = version_limit;
        Some(
            ShardSources {
                node: self.node,
                instance: Tracked(shard_instance),
                alloc: Tracked(shard_alloc),
                events: Tracked(shard_events),
                shares: Tracked(shard_shares),
                version_instance: Tracked(shard_ver_instance),
                versions: Tracked(shard_versions),
                version_start,
                version_limit,
            },
        )
    }

    /// Consume the remaining allocator budget as a final `ShardSources` if non-empty.
    pub fn into_shard(self) -> (r: Option<ShardSources>)
        requires
            self.well_formed(),
        ensures
            self.next_version_seq() < self.version_limit() <==> r is Some,
            r matches Some(shard) ==> {
                &&& shard.well_formed()
                &&& shard.node == self.node()
                &&& shard.instance@.id() == self.instance_id()
                &&& shard.version_instance@.id() == self.version_instance_id()
                &&& shard.version_start == self.next_version_seq()
                &&& shard.version_limit == self.version_limit()
                &&& shard.contained_in_allocator(&self)
            },
    {
        if self.next_version_seq >= self.version_limit {
            return None;
        }
        proof {
            self.alloc.borrow().lemma_names_distinct();
            self.events.borrow().lemma_names_distinct();
            self.shares.borrow().lemma_names_distinct();
        }
        Some(
            ShardSources {
                node: self.node,
                instance: self.instance,
                alloc: self.alloc,
                events: self.events,
                shares: self.shares,
                version_instance: self.version_instance,
                versions: self.versions,
                version_start: self.next_version_seq,
                version_limit: self.version_limit,
            },
        )
    }
}

pub proof fn lemma_split_disjoint(
    prev_shard: &ShardSources,
    old_alloc: &ShardAllocator,
    next_shard: &ShardSources,
)
    requires
        prev_shard.disjoint_from_allocator(old_alloc),
        next_shard.contained_in_allocator(old_alloc),
    ensures
        prev_shard.disjoint_from(next_shard),
{
}

pub proof fn lemma_remainder_disjoint(
    prev_shard: &ShardSources,
    old_alloc: &ShardAllocator,
    new_alloc: &ShardAllocator,
)
    requires
        prev_shard.disjoint_from_allocator(old_alloc),
        new_alloc.contained_in_allocator(old_alloc),
    ensures
        prev_shard.disjoint_from_allocator(new_alloc),
{
}

#[cfg(verus_only)]
fn verify_four_way_shard_split(ctx: NodeContext) -> (r: (
    ShardSources,
    ShardSources,
    ShardSources,
    ShardSources,
))
    requires
        ctx.initial_partition(),
    ensures
        r.0.well_formed() && r.1.well_formed() && r.2.well_formed() && r.3.well_formed(),
        r.0.disjoint_from(&r.1) && r.0.disjoint_from(&r.2) && r.0.disjoint_from(&r.3),
        r.1.disjoint_from(&r.2) && r.1.disjoint_from(&r.3) && r.2.disjoint_from(&r.3),
{
    let mut alloc = ShardAllocator::from_node_context(ctx);
    let ghost a0 = alloc;
    let s0 = alloc.split_shard(100).unwrap();
    let ghost a1 = alloc;
    let s1 = alloc.split_shard(200).unwrap();
    let ghost a2 = alloc;
    let s2 = alloc.split_shard(300).unwrap();
    let ghost a3 = alloc;
    let s3 = alloc.into_shard().unwrap();
    proof {
        lemma_split_disjoint(&s0, &a1, &s1);
        lemma_remainder_disjoint(&s0, &a1, &a2);
        lemma_split_disjoint(&s0, &a2, &s2);
        lemma_remainder_disjoint(&s0, &a2, &a3);
        lemma_split_disjoint(&s0, &a3, &s3);

        lemma_split_disjoint(&s1, &a2, &s2);
        lemma_remainder_disjoint(&s1, &a2, &a3);
        lemma_split_disjoint(&s1, &a3, &s3);

        lemma_split_disjoint(&s2, &a3, &s3);
    }
    (s0, s1, s2, s3)
}

} // verus!
