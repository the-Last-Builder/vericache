//! Multi-slot (slab and scatter-gather) submission helpers confined to `context`.
use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::set::group_set_lemmas,
};

proof fn pack_slab_tokens(
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    tracked tokens: Map<usize, SlotToken>,
) -> (tracked result: RegionMap)
    requires
        slab_tokens_valid(tokens, id, node, space, count),
    ensures
        result.instance_id() == id,
        result.dom() == slab_locs(node, space, count as nat),
        forall|i: usize|
            i < count ==> #[trigger] result.map().index(
                SlotLoc { node, space, slot_index: i as nat },
            ) == tokens[i].value(),
    decreases count,
{
    if count == 0 {
        let tracked m = RegionMap::empty(id);
        assert(m.dom() =~= slab_locs(node, space, 0nat));
        m
    } else {
        let last = (count - 1) as usize;
        let tracked mut rest_map = tokens;
        let tracked tok = rest_map.tracked_remove(last);
        assert(slab_tokens_valid(rest_map, id, node, space, last));
        let tracked mut m = pack_slab_tokens(id, node, space, last, rest_map);
        let loc = SlotLoc { node, space, slot_index: last as nat };
        assert(!m.dom().contains(loc));
        m.insert(tok.into_raw());
        assert(m.dom() =~= slab_locs(node, space, count as nat));
        m
    }
}

proof fn unpack_slab_tokens(
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    tracked m: RegionMap,
) -> (tracked tokens: Map<usize, SlotToken>)
    requires
        m.instance_id() == id,
        m.dom() == slab_locs(node, space, count as nat),
    ensures
        forall|i: usize| i < count <==> #[trigger] tokens.contains_key(i),
        forall|i: usize|
            i < count ==> (#[trigger] tokens[i]).well_formed() && tokens[i].instance_id() == id
                && tokens[i].key() == (SlotLoc { node, space, slot_index: i as nat })
                && tokens[i].value() == m.map().index(
                SlotLoc { node, space, slot_index: i as nat },
            ),
    decreases count,
{
    if count == 0 {
        Map::tracked_empty()
    } else {
        let last = (count - 1) as usize;
        let loc = SlotLoc { node, space, slot_index: last as nat };
        let tracked mut m_mut = m;
        assert(m_mut.dom().contains(loc));
        let tracked kv = m_mut.remove(loc);
        assert(m_mut.dom() =~= slab_locs(node, space, last as nat));
        let tracked mut res = unpack_slab_tokens(id, node, space, last, m_mut);
        res.tracked_insert(last, SlotToken::from_raw(kv));
        res
    }
}

proof fn pack_multi_owned(
    id: InstanceId,
    tracked tokens: Map<SlotLoc, SlotToken>,
) -> (tracked result: RegionMap)
    requires
        forall|l: SlotLoc| #[trigger] tokens.contains_key(l) ==> slot_is_owned(tokens[l], id, l),
    ensures
        result.instance_id() == id,
        forall|l: SlotLoc| #[trigger] result.dom().contains(l) <==> tokens.contains_key(l),
        forall|l: SlotLoc| #[trigger]
            tokens.contains_key(l) ==> result.map()[l] == tokens[l].value(),
    decreases tokens.dom().len(),
{
    if tokens.dom().is_empty() {
        let tracked m = RegionMap::empty(id);
        assert forall|l: SlotLoc| #[trigger] m.dom().contains(l) <==> tokens.contains_key(l) by {}
        m
    } else {
        let loc = tokens.dom().choose();
        let tracked mut rest = tokens;
        let tracked tok = rest.tracked_remove(loc);
        let tracked mut m = pack_multi_owned(id, rest);
        assert(!m.dom().contains(loc));
        m.insert(tok.into_raw());
        m
    }
}

proof fn unpack_multi_owned(
    id: InstanceId,
    locs: Set<SlotLoc>,
    tracked m: RegionMap,
) -> (tracked tokens: Map<SlotLoc, SlotToken>)
    requires
        m.instance_id() == id,
        forall|l: SlotLoc| #[trigger] m.dom().contains(l) <==> locs.contains(l),
    ensures
        tokens.dom() == locs,
        forall|l: SlotLoc| #[trigger]
            locs.contains(l) ==> tokens[l].well_formed() && tokens[l].instance_id() == id
                && tokens[l].key() == l && tokens[l].value() == m.map()[l],
    decreases locs.len(),
{
    if locs.is_empty() {
        Map::tracked_empty()
    } else {
        let loc = locs.choose();
        let tracked mut m_mut = m;
        assert(m_mut.dom().contains(loc));
        let tracked kv = m_mut.remove(loc);
        let rest_locs = locs.remove(loc);
        assert forall|l: SlotLoc| #[trigger]
            m_mut.dom().contains(l) <==> rest_locs.contains(l) by {}
        let tracked mut res = unpack_multi_owned(id, rest_locs, m_mut);
        res.tracked_insert(loc, SlotToken::from_raw(kv));
        assert(res.dom() =~= locs);
        res
    }
}

pub(super) proof fn run_submit_slab_owned(
    tracked inst: &TransferInstance,
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
        name.instance_id() == inst.id(),
        name.element().observer == observer,
        infos.instance_id() == inst.id(),
        knows.instance_id() == inst.id(),
        after.union(cited).subset_of(infos.dom()),
        knowledge_of(observer, cited).subset_of(knows.set()),
        access.writes == slab_locs(node, space, count as nat),
        access.reads.subset_of(slab_locs(node, space, count as nat)),
        slab_tokens_valid(tokens, inst.id(), node, space, count),
        forall|i: usize|
            i < count ==> write_admitted(
                slot_frontier(#[trigger] tokens[i]),
                closure_from(infos.map(), after.union(cited)),
            ),
    ensures
        result.0@.instance_id() == inst.id(),
        result.0@.key() == name.element(),
        result.0@.value() == (EventInfo {
            access,
            effect,
            read_guards: IMap::empty(),
            preds: closure_from(infos.map(), after.union(cited)),
        }),
        slab_tokens_owned(
            result.1@,
            inst.id(),
            node,
            space,
            count,
            Frontier { last_writer: Some(name.element()), readers: ISet::empty() },
        ),
{
    let ev = name.element();
    let locs = slab_locs(node, space, count as nat);
    let tracked owned_map = pack_slab_tokens(inst.id(), node, space, count, tokens);
    let frontiers = IMap::new(
        |l: SlotLoc| locs.contains(l),
        |l: SlotLoc| tokens[l.slot_index as usize].value()->Owned_0,
    );
    assert(owned_map.map() =~= owned_slots(frontiers)) by {
        assert forall|l: SlotLoc| #[trigger] locs.contains(l) implies owned_map.map()[l]
            == Slot::Owned(frontiers[l]) by {
            let idx = l.slot_index as usize;
            assert(idx < count);
            assert(l == SlotLoc { node, space, slot_index: idx as nat });
        }
    }
    assert forall|l: SlotLoc| #[trigger] frontiers.dom().contains(l) implies admitted(
        frontiers[l],
        l,
        access,
        closure_from(infos.map(), after.union(cited)),
    ) by {
        let idx = l.slot_index as usize;
        assert(idx < count);
        assert(write_admitted(
            slot_frontier(tokens[idx]),
            closure_from(infos.map(), after.union(cited)),
        ));
    }
    assert(access.reads.subset_of(
        frontiers.dom().union(IMap::<SlotLoc, TransferShare>::empty().dom()),
    ));
    let tracked shares = ReadShareMap::empty(inst.id());
    let tracked (Tracked(sub), Tracked(new_owned), _) = inst.inner.submit(
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
    assert(new_owned.dom() =~= locs);
    let tracked out_tokens = unpack_slab_tokens(inst.id(), node, space, count, new_owned);
    assert forall|i: usize| i < count implies slot_owned(
        #[trigger] out_tokens[i],
        inst.id(),
        SlotLoc { node, space, slot_index: i as nat },
        Frontier { last_writer: Some(ev), readers: ISet::empty() },
    ) by {
        let l = SlotLoc { node, space, slot_index: i as nat };
        assert(locs.contains(l));
        assert(advance(frontiers[l], l, ev, access) == Frontier {
            last_writer: Some(ev),
            readers: ISet::empty(),
        });
    }
    (Tracked(sub), Tracked(out_tokens))
}

pub(super) proof fn run_submit_multi(
    tracked inst: &TransferInstance,
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
        name.instance_id() == inst.id(),
        name.element().observer == observer,
        infos.instance_id() == inst.id(),
        knows.instance_id() == inst.id(),
        after.union(cited).subset_of(infos.dom()),
        knowledge_of(observer, cited).subset_of(knows.set()),
        forall|l: SlotLoc| #[trigger]
            owned.contains_key(l) ==> slot_is_owned(owned[l], inst.id(), l),
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
                &&& cap.instance_id() == inst.id()
                &&& !owned.contains_key(cap.loc())
                &&& access.reads.contains(cap.loc())
                &&& read_admitted(cap.frontier(), closure_from(infos.map(), after.union(cited)))
                &&& forall|l: SlotLoc| #[trigger]
                    access.reads.contains(l) ==> owned.contains_key(l) || l == cap.loc()
            },
        },
    ensures
        result.0@.instance_id() == inst.id(),
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
                inst.id(),
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
    let ev = name.element();
    let owned_locs = owned.dom();
    let tracked owned_map = pack_multi_owned(inst.id(), owned);
    let frontiers = IMap::new(
        |l: SlotLoc| owned.contains_key(l),
        |l: SlotLoc| owned[l].value()->Owned_0,
    );
    assert(owned_map.map() =~= owned_slots(frontiers));
    match shared {
        None => {
            let tracked shares = ReadShareMap::empty(inst.id());
            assert(access.reads.subset_of(
                frontiers.dom().union(IMap::<SlotLoc, TransferShare>::empty().dom()),
            ));
            let tracked (Tracked(sub), Tracked(new_owned), _) = inst.inner.submit(
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
            assert forall|l: SlotLoc| #[trigger]
                new_owned.dom().contains(l) <==> owned_locs.contains(l) by {}
            let tracked out_owned = unpack_multi_owned(inst.id(), owned_locs, new_owned);
            (Tracked(sub), Tracked(out_owned), Tracked(None))
        },
        Some(cap) => {
            let share = cap.share;
            let sloc = cap.loc;
            let guards = IMap::empty().insert(sloc, share);
            let tracked shares_map = cap.map;
            assert(access.reads.subset_of(frontiers.dom().union(guards.dom()))) by {
                assert forall|l: SlotLoc| #[trigger]
                    access.reads.contains(l) implies frontiers.dom().contains(l)
                    || guards.dom().contains(l) by {}
            }
            let tracked (Tracked(sub), Tracked(new_owned), Tracked(new_shares)) = inst.inner.submit(
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
                owned_map,
                shares_map,
            );
            assert forall|l: SlotLoc| #[trigger]
                new_owned.dom().contains(l) <==> owned_locs.contains(l) by {}
            let tracked out_owned = unpack_multi_owned(inst.id(), owned_locs, new_owned);
            assert(new_shares.dom() =~= ISet::empty().insert((sloc, share)));
            let tracked new_cap = ShareCap { loc: sloc, share, map: new_shares, usage: cap.usage };
            (Tracked(sub), Tracked(out_owned), Tracked(Some(new_cap)))
        },
    }
}

} // verus!
