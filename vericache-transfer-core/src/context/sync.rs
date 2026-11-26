//! Record operations completed synchronously by a backend.
use super::*;
use foundation::space::{Access, Effect};

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// The CPU event source used to record synchronous operations across backends.
pub struct SyncContext {
    pub source: Tracked<EventSourceToken>,
}

impl SyncContext {
    pub fn new(Tracked(source): Tracked<EventSourceToken>) -> (result: Self)
        requires
            source.well_formed(),
            source.source().is_host(),
        ensures
            result.source@ == source,
    {
        Self { source: Tracked(source) }
    }

    pub open spec fn ready(&self, instance: InstanceId, node: NodeId) -> bool {
        self.source@.well_formed() && self.source@.node() == node && self.source@.source().is_host()
            && self.source@.id() == instance
    }

    pub open spec fn stepped(&self, previous: Self) -> bool {
        &&& self.ready(previous.source@.id(), previous.source@.node())
        &&& self.source@.source() == previous.source@.source()
        &&& self.source@.next_seq() == previous.source@.next_seq() + 1
    }
}

/// Compatibility name for the CPU event source.
pub type HostContext = SyncContext;

/// The submitted event and the observer's knowledge of its completion and predecessors.
pub tracked struct Completion {
    pub tracked submitted: Submitted,
    pub tracked knows: Knows,
}

impl Completion {
    pub open spec fn records(
        self,
        id: InstanceId,
        event: TransferEvent,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        preds: ISet<TransferEvent>,
        read_guards: ReadGuards,
    ) -> bool {
        &&& self.submitted.key() == event && self.submitted.instance_id() == id
        &&& self.submitted.value() == (EventInfo { access, effect, preds, read_guards })
        &&& self.knows.instance_id() == id
        &&& self.knows.set() == knowledge_of(event.observer, preds.insert(event))
    }

    pub proof fn infos(tracked &self) -> (tracked infos: SubmittedMap)
        ensures
            infos.instance_id() == self.submitted.instance_id(),
            infos.map() == IMap::empty().insert(self.submitted.key(), self.submitted.value()),
    {
        let tracked mut infos = SubmittedMap::empty(self.submitted.instance_id());
        infos.insert(self.submitted);
        infos
    }
}

/// The access that a synchronous shim completed, with its actual arguments.
/// Recording it consumes the witness. Ordinary read/write permissions remain
/// separate so a hardware operation can never create a foundation token.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct CompletedAccess {
    ghost access: Access<TransferDomain>,
    ghost effect: Effect<TransferDomain>,
}

impl CompletedAccess {
    pub(super) fn new(
        _access: Ghost<Access<TransferDomain>>,
        _effect: Ghost<Effect<TransferDomain>>,
    ) -> (r: Tracked<Self>)
        ensures
            r@.access() == _access@,
            r@.effect() == _effect@,
    {
        Tracked(Self { access: _access@, effect: _effect@ })
    }

    pub closed spec fn access(self) -> Access<TransferDomain> {
        self.access
    }

    pub closed spec fn effect(self) -> Effect<TransferDomain> {
        self.effect
    }
}

/// Events whose hardware completion has been observed by a driver.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct ObservedEvents {
    ghost instance: InstanceId,
    ghost observer: NodeId,
    ghost completed: ISet<TransferEvent>,
    ghost successful: ISet<TransferEvent>,
}

impl ObservedEvents {
    pub(super) fn new(
        _id: Ghost<InstanceId>,
        _observer: Ghost<NodeId>,
        _completed: Ghost<ISet<TransferEvent>>,
        _successful: Ghost<ISet<TransferEvent>>,
    ) -> (r: Tracked<Self>)
        ensures
            r@.observed(_id@, _observer@, _completed@),
            forall|e: TransferEvent| #[trigger] r@.succeeded(_id@, e) == _successful@.contains(e),
    {
        Tracked(
            Self {
                instance: _id@,
                observer: _observer@,
                completed: _completed@,
                successful: _successful@,
            },
        )
    }

    pub closed spec fn observed(
        self,
        id: InstanceId,
        observer: NodeId,
        completed: ISet<TransferEvent>,
    ) -> bool {
        self.instance == id && self.observer == observer && self.completed == completed
    }

    pub closed spec fn succeeded(self, id: InstanceId, event: TransferEvent) -> bool {
        self.instance == id && self.successful.contains(event)
    }

    pub proof fn success(tracked &self, id: InstanceId, event: TransferEvent) -> (tracked r:
        Succeeded)
        requires
            self.succeeded(id, event),
        ensures
            r.instance_id() == id,
            r.event() == event,
    {
        Succeeded::mint(id, event)
    }
}

/// A read share whose hardware window has been fenced and deallocated.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct CompletedFence {
    ghost instance: InstanceId,
    ghost loc: SlotLoc,
    ghost share: TransferShare,
}

impl CompletedFence {
    pub(super) fn new(
        _id: Ghost<InstanceId>,
        _loc: Ghost<SlotLoc>,
        _share: Ghost<TransferShare>,
    ) -> (r: Tracked<Self>)
        ensures
            r@.matches(_id@, _loc@, _share@),
    {
        Tracked(Self { instance: _id@, loc: _loc@, share: _share@ })
    }

    pub closed spec fn matches(self, id: InstanceId, loc: SlotLoc, share: TransferShare) -> bool {
        self.instance == id && self.loc == loc && self.share == share
    }
}

proof fn observe_single(
    tracked instance: &TransferInstance,
    observer: NodeId,
    tracked submitted: Submitted,
) -> (tracked knowledge: Knows)
    requires
        submitted.instance_id() == instance.id(),
    ensures
        knowledge.instance_id() == instance.id(),
        knowledge.set() == knowledge_of(observer, submitted.value().preds.insert(submitted.key())),
{
    let event = submitted.key();
    let tracked mut completed = SubmittedMap::empty(instance.id());
    completed.insert(submitted);
    let singleton = ISet::empty().insert(event);
    let tracked knowledge = instance.observe(observer, singleton, &completed);
    assert forall|e: TransferEvent| #[trigger]
        submitted.value().preds.contains(e) implies closure_from(
        completed.map(),
        singleton,
    ).contains(e) by {
        assert(completed.map()[event].preds.contains(e));
    }
    assert(closure_from(completed.map(), singleton) =~= submitted.value().preds.insert(event));
    knowledge
}

pub proof fn finish_owned(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    tracked completed_access: CompletedAccess,
    tracked slot: &mut SlotToken,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: Completion)
    requires
        old(source).well_formed(),
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        old(slot).well_formed(),
        old(slot).instance_id() == instance.id(),
        old(slot).value() is Owned,
        cited.subset_of(infos.dom()),
        knowledge_of(old(source).node(), cited).subset_of(knows.set()),
        completed_access.access().writes.subset_of(ISet::empty().insert(old(slot).key())),
        completed_access.access().reads.subset_of(ISet::empty().insert(old(slot).key())),
        admitted(
            slot_frontier(*old(slot)),
            old(slot).key(),
            completed_access.access(),
            closure_from(infos.map(), cited),
        ),
    ensures
        final(source).well_formed(),
        final(source).node() == old(source).node(),
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        final(slot).well_formed(),
        final(slot).instance_id() == instance.id(),
        final(slot).key() == old(slot).key(),
        final(slot).value() == Slot::Owned(
            advance(
                slot_frontier(*old(slot)),
                old(slot).key(),
                old(source).next_event(),
                completed_access.access(),
            ),
        ),
        result.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access(),
            completed_access.effect(),
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked submitted = instance.submit_1_owned(
        source.node(),
        completed_access.access(),
        completed_access.effect(),
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        slot,
    );
    let tracked knowledge = observe_single(instance, source.node(), submitted);
    Completion { submitted, knows: knowledge }
}

pub proof fn finish_read(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    tracked completed_access: CompletedAccess,
    tracked src_perm: &mut ReadPerm,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: Completion)
    requires
        old(source).well_formed(),
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        old(src_perm).instance_id() == instance.id(),
        old(src_perm).is_owned() || old(src_perm).is_shared_wf(),
        cited.subset_of(infos.dom()),
        knowledge_of(old(source).node(), cited).subset_of(knows.set()),
        completed_access.access() == (Access {
            reads: ISet::empty().insert(old(src_perm).loc()),
            writes: ISet::empty(),
        }),
        read_admitted(old(src_perm).frontier(), closure_from(infos.map(), cited)),
    ensures
        final(source).well_formed(),
        final(source).node() == old(source).node(),
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        final(src_perm).advanced_read(*old(src_perm), old(source).next_event()),
        result.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access(),
            completed_access.effect(),
            closure_from(infos.map(), cited),
            old(src_perm).read_guards(),
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked submitted = instance.submit_1_read(
        source.node(),
        completed_access.effect(),
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        src_perm,
    );
    let tracked knowledge = observe_single(instance, source.node(), submitted);
    Completion { submitted, knows: knowledge }
}

pub proof fn finish_copy(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    tracked completed_access: CompletedAccess,
    tracked src_perm: &mut ReadPerm,
    tracked dst_tok: &mut SlotToken,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: Completion)
    requires
        old(source).well_formed(),
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        old(src_perm).instance_id() == instance.id(),
        old(dst_tok).well_formed(),
        old(dst_tok).instance_id() == instance.id(),
        old(src_perm).is_owned() || old(src_perm).is_shared_wf(),
        old(dst_tok).value() is Owned,
        old(src_perm).loc() != old(dst_tok).key(),
        cited.subset_of(infos.dom()),
        knowledge_of(old(source).node(), cited).subset_of(knows.set()),
        completed_access.access().writes == ISet::empty().insert(old(dst_tok).key()),
        completed_access.access().reads.contains(old(src_perm).loc()),
        completed_access.access().reads.subset_of(
            ISet::empty().insert(old(src_perm).loc()).insert(old(dst_tok).key()),
        ),
        read_admitted(old(src_perm).frontier(), closure_from(infos.map(), cited)),
        write_admitted(slot_frontier(*old(dst_tok)), closure_from(infos.map(), cited)),
    ensures
        final(source).well_formed(),
        final(source).node() == old(source).node(),
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        final(src_perm).advanced_read(*old(src_perm), old(source).next_event()),
        slot_owned(
            *final(dst_tok),
            instance.id(),
            old(dst_tok).key(),
            Frontier { last_writer: Some(old(source).next_event()), readers: ISet::empty() },
        ),
        result.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access(),
            completed_access.effect(),
            closure_from(infos.map(), cited),
            old(src_perm).read_guards(),
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked submitted = instance.submit_2(
        source.node(),
        completed_access.access(),
        completed_access.effect(),
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        src_perm,
        dst_tok,
    );
    let tracked knowledge = observe_single(instance, source.node(), submitted);
    Completion { submitted, knows: knowledge }
}

pub proof fn finish_slab_owned(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    tracked completed_access: CompletedAccess,
    tracked tokens: Map<usize, SlotToken>,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: (Completion, Map<usize, SlotToken>))
    requires
        old(source).well_formed(),
        old(source).node() == node,
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(infos.dom()),
        knowledge_of(node, cited).subset_of(knows.set()),
        completed_access.access().writes == slab_locs(node, space, count as nat),
        completed_access.access().reads.subset_of(slab_locs(node, space, count as nat)),
        slab_tokens_valid(tokens, instance.id(), node, space, count),
        forall|i: usize|
            i < count ==> write_admitted(
                slot_frontier(#[trigger] tokens[i]),
                closure_from(infos.map(), cited),
            ),
    ensures
        final(source).well_formed(),
        final(source).node() == node,
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        result.0.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access(),
            completed_access.effect(),
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
        slab_tokens_owned(
            result.1,
            instance.id(),
            node,
            space,
            count,
            Frontier { last_writer: Some(old(source).next_event()), readers: ISet::empty() },
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked (Tracked(submitted), Tracked(out_tokens)) = instance.submit_slab_owned(
        node,
        node,
        space,
        count,
        completed_access.access(),
        completed_access.effect(),
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        tokens,
    );
    let tracked knowledge = observe_single(instance, node, submitted);
    (Completion { submitted, knows: knowledge }, out_tokens)
}

pub open spec fn can_reclaim_slot(
    id: InstanceId,
    loc: SlotLoc,
    tok: SlotToken,
    infos: SubmittedMap,
    knows: Knows,
    cited: ISet<TransferEvent>,
) -> bool {
    &&& slot_is_owned(tok, id, loc)
    &&& infos.instance_id() == id
    &&& knows.instance_id() == id
    &&& cited.subset_of(infos.dom())
    &&& knowledge_of(loc.node, cited).subset_of(knows.set())
    &&& write_admitted(slot_frontier(tok), closure_from(infos.map(), cited))
}

pub open spec fn can_reclaim_slab(
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    tokens: Map<usize, SlotToken>,
    infos: SubmittedMap,
    knows: Knows,
    cited: ISet<TransferEvent>,
) -> bool {
    &&& infos.instance_id() == id
    &&& knows.instance_id() == id
    &&& cited.subset_of(infos.dom())
    &&& knowledge_of(node, cited).subset_of(knows.set())
    &&& slab_tokens_valid(tokens, id, node, space, count)
    &&& forall|i: usize|
        i < count ==> write_admitted(
            slot_frontier(#[trigger] tokens[i]),
            closure_from(infos.map(), cited),
        )
}

} // verus!
