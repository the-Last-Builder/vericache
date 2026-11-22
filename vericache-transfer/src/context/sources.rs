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

    proof fn partition(self)
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
            forall|id: nat| #[trigger] self.half(false).contains(id) ==> self.contains(id),
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

    pub open spec fn space(self, device: Option<DeviceId>) -> MemorySpace {
        let alloc = self.next_alloc_id();
        match device {
            None => MemorySpace::HostAlloc { alloc },
            Some(device) => MemorySpace::GpuAlloc { device, alloc },
        }
    }

    pub open spec fn nvme_space(self) -> MemorySpace {
        MemorySpace::NvmeFile { alloc: self.next_alloc_id() }
    }

    pub open spec fn slot_loc(self, device: Option<DeviceId>, slot_index: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space(device), slot_index }
    }

    pub open spec fn nvme_slot_loc(self, slot_index: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.nvme_space(), slot_index }
    }

    pub open spec fn slot_ref(self, device: Option<DeviceId>, len: u64) -> SlotRef {
        SlotRef::full(self.slot_loc(device, 0), len)
    }

    pub open spec fn nvme_slot_ref(self, len: u64) -> SlotRef {
        SlotRef::full(self.nvme_slot_loc(0), len)
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

    pub proof fn allocate_slot(tracked &mut self, device: Option<DeviceId>) -> (tracked slot:
        SlotToken)
        requires
            old(self).well_formed(),
        ensures
            final(self).allocated_slot(*old(self), old(self).slot_loc(device, 0), slot),
    {
        self.allocate_one_slot(self.space(device))
    }

    pub proof fn allocate_slab_slots(
        tracked &mut self,
        device: Option<DeviceId>,
        count: usize,
    ) -> (tracked slots: Map<usize, SlotToken>)
        requires
            old(self).well_formed(),
        ensures
            final(self).allocated_slab(*old(self), old(self).space(device), count, slots),
    {
        self.allocate_many_slots(self.space(device), count)
    }

    pub proof fn allocate_nvme_slot(tracked &mut self) -> (tracked slot: SlotToken)
        requires
            old(self).well_formed(),
        ensures
            final(self).allocated_slot(*old(self), old(self).nvme_slot_loc(0), slot),
    {
        self.allocate_one_slot(self.nvme_space())
    }

    pub proof fn allocate_nvme_slab_slots(tracked &mut self, count: usize) -> (tracked slots: Map<
        usize,
        SlotToken,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).allocated_slab(*old(self), old(self).nvme_space(), count, slots),
    {
        self.allocate_many_slots(self.nvme_space(), count)
    }

    proof fn allocate_one_slot(tracked &mut self, space: MemorySpace) -> (tracked slot: SlotToken)
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

    proof fn allocate_many_slots(
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
                other.names().contains(id) ==> !final(self).names().contains(id),
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

pub ghost enum EventSourceKind {
    Host,
    CudaStream,
    IoUring,
    RdmaQueuePair,
}

impl EventSourceKind {
    pub open spec fn source(self, id: nat) -> EventSource {
        match self {
            Self::Host => EventSource::Host { id },
            Self::CudaStream => EventSource::CudaStream { id },
            Self::IoUring => EventSource::IoUring { id },
            Self::RdmaQueuePair => EventSource::RdmaQueuePair { id },
        }
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

    pub(crate) proof fn take_next(tracked &mut self) -> (tracked name: HistoryTokens::unused<
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

    pub proof fn mint(tracked &mut self, kind: EventSourceKind) -> (tracked token: EventSourceToken)
        requires
            old(self).well_formed(),
        ensures
            token.well_formed(),
            token.node() == old(self).node(),
            token.source() == kind.source((old(self).names().name)(0)),
            !final(self).names().contains(token.source().id()),
            token.next_seq() == 0,
            token.id() == old(self).id(),
            final(self).well_formed(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).names() == old(self).names().tail(),
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
                other.names().contains(id) ==> !final(self).names().contains(id),
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
                other.names().contains(id) ==> !final(self).names().contains(id),
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
    tracked unused: VersionUnusedSet,
}

impl VersionSource {
    pub closed spec fn node(self) -> NodeId {
        self.node
    }

    pub closed spec fn min_seq(self) -> nat {
        self.min_seq
    }

    pub closed spec fn id(self) -> InstanceId {
        self.unused.instance_id()
    }

    pub closed spec fn well_formed(self) -> bool {
        ISet::new(|k: VersionKey| k.0 == self.node && k.1 as nat >= self.min_seq).subset_of(
            self.unused.set(),
        )
    }

    pub proof fn take_seq(tracked &mut self, seq: u64) -> (tracked tok: VersionUnused)
        requires
            old(self).well_formed(),
            seq as nat >= old(self).min_seq(),
        ensures
            tok.instance_id() == old(self).id(),
            tok.element() == (old(self).node(), seq),
            final(self).well_formed(),
            final(self).id() == old(self).id(),
            final(self).node() == old(self).node(),
            final(self).min_seq() == (seq as nat) + 1,
    {
        let key: VersionKey = (self.node, seq);
        assert(ISet::new(|k: VersionKey| k.0 == self.node && k.1 as nat >= self.min_seq).contains(
            key,
        ));
        let tracked tok = self.unused.remove(key);
        let old_min = self.min_seq;
        self.min_seq = (seq as nat) + 1;
        assert forall|k: VersionKey|
            ISet::new(|k: VersionKey| k.0 == self.node && k.1 as nat >= self.min_seq).contains(
                k,
            ) implies self.unused.set().contains(k) by {
            assert(ISet::new(|k: VersionKey| k.0 == self.node && k.1 as nat >= old_min).contains(
                k,
            ));
            assert(k != key);
        }
        tok
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

} // verus!
