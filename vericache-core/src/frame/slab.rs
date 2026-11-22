use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

/// Convert a `vericache-transfer` slab allocation (`slab_tokens_owned`) into the
/// per-slot `Seq<SlotRef>` + disjointness obligations of `register_slab`.
pub proof fn lemma_slab_slots_valid(
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    num_slots: usize,
    frame_size: usize,
    tokens: Map<usize, SlotToken>,
) -> (slots: Seq<SlotRef>)
    requires
        slab_tokens_owned(tokens, id, node, space, num_slots, untouched()),
    ensures
        slots.len() == num_slots,
        forall|i: int|
            0 <= i < num_slots ==> (#[trigger] slots[i]) == SlotRef::full(
                SlotLoc { node, space, slot_index: i as nat },
                frame_size as u64,
            ),
        forall|i: usize|
            i < num_slots ==> tokens.contains_key(i) && (#[trigger] slots[i as int]).is_full()
                && slots[i as int].slot_len == frame_size && slots[i as int].loc.node == node
                && slot_owned(tokens[i], id, slots[i as int].loc, untouched()),
        forall|i: int, j: int|
            0 <= i < j < num_slots ==> (#[trigger] slots[i]).loc != (#[trigger] slots[j]).loc,
{
    let slots = Seq::new(
        num_slots as nat,
        |i: int| SlotRef::full(SlotLoc { node, space, slot_index: i as nat }, frame_size as u64),
    );
    assert forall|i: usize| i < num_slots implies #[trigger] tokens.contains_key(i)
        && slots[i as int].is_full() && slots[i as int].slot_len == frame_size
        && slots[i as int].loc.node == node && slot_owned(
        tokens[i],
        id,
        slots[i as int].loc,
        untouched(),
    ) by {
        assert(tokens.contains_key(i));
        assert(slot_owned(
            tokens[i],
            id,
            SlotLoc { node, space, slot_index: i as nat },
            untouched(),
        ));
    }
    slots
}

impl FrameTable {
    fn append_empty(
        &mut self,
        tier: TierId,
        Ghost(slot): Ghost<SlotRef>,
        Tracked(tok): Tracked<SlotToken>,
        _table: &TimelineTable,
        Tracked(ledger): Tracked<&KnowledgeLedger>,
    )
        requires
            old(self).well_formed(tok.instance_id(), slot.loc.node, _table, ledger),
            old(self).metas.len() < u32::MAX,
            tier.valid(old(self).num_tiers),
            slot.is_full(),
            slot.slot_len == old(self).frame_size,
            slot_owned(tok, tok.instance_id(), slot.loc, untouched()),
            forall|i: nat|
                i < old(self).metas.len() ==> (#[trigger] old(self).ghosts@[i].slot).loc
                    != slot.loc,
        ensures
            final(self).well_formed(tok.instance_id(), slot.loc.node, _table, ledger),
            final(self).num_tiers == old(self).num_tiers,
            final(self).frame_size == old(self).frame_size,
            final(self).versions == old(self).versions,
            final(self).metas.len() == old(self).metas.len() + 1,
            final(self).ghosts@[old(self).metas.len() as nat].slot == slot,
            final(self).ghosts@[old(self).metas.len() as nat].tier == tier,
            final(self).metas[old(self).metas.len() as int].state == STATE_EMPTY,
            final(self).metas[old(self).metas.len() as int].queued(),
            forall|i: nat|
                i < old(self).metas.len() ==> #[trigger] final(self).ghosts@[i] == old(
                    self,
                ).ghosts@[i] && final(self).metas[i as int] == old(self).metas[i as int],
            final(self).ready_queue[tier.index() as int]@ == old(
                self,
            ).ready_queue[tier.index() as int]@.push(FrameId(old(self).metas.len() as u32)),
    {
        let idx = self.metas.len();
        let t = tier.as_index();
        let m = FrameSyncMeta::empty();
        let tracked g = FrameGhost {
            slot,
            tier,
            token: Some(tok),
            local_shares: Map::empty(),
            remote_shares: Map::empty(),
            cert: None,
            bound_key: None,
            bound_version: None,
        };
        assert(m.valid_tickets(_table));
        assert(g.authority(m, tok.instance_id()));
        assert(g.frontier() == untouched());
        assert(g.covered(m, _table, slot.loc.node, ledger));
        assert(g.binding(m, _table));
        assert(2u8 & 2u8 != 0u8 && 2u8 & 1u8 == 0u8) by (bit_vector);
        assert(m.reusable(_table));
        assert(g.valid(
            m,
            self.num_tiers,
            self.frame_size,
            tok.instance_id(),
            slot.loc.node,
            _table,
            ledger,
        ));
        self.bindings.push((ObjectKey([0;4]), tier));
        self.metas.push(m);
        proof {
            self.ghosts.borrow_mut().tracked_insert(idx as nat, g);
        }
        self.ready_queue[t].push(FrameId(idx as u32));
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            _table,
        ) by {
            assert(old(self).tier_queues(u, _table));
            let ready = self.ready_queue[u]@;
            let draining = self.draining_queue[u]@;
            assert forall|i: int| 0 <= i < ready.len() implies {
                let f = #[trigger] ready[i];
                f.0 < self.metas.len() && self.frame(f).tier.index() == u && self.meta(f).state
                    == STATE_EMPTY && self.meta(f).queued() && !self.meta(f).overflow()
                    && self.meta(f).reusable(_table)
            } by {
                if u != t || i < ready.len() - 1 {
                    assert(old(self).ready_queue[u][i].0 < idx);
                }
            }
            assert forall|i: int| 0 <= i < draining.len() implies {
                let f = #[trigger] draining[i];
                f.0 < self.metas.len() && self.frame(f).tier.index() == u && self.meta(f).state
                    == STATE_DRAINING && !self.meta(f).queued()
            } by {
                assert(old(self).draining_queue[u][i].0 < idx);
            }
        }
        assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts@[i].valid(
            self.metas[i as int],
            self.num_tiers,
            self.frame_size,
            tok.instance_id(),
            slot.loc.node,
            _table,
            ledger,
        ) by {
            if i < idx {
                assert(old(self).ghosts@[i].valid(
                    old(self).metas[i as int],
                    self.num_tiers,
                    self.frame_size,
                    tok.instance_id(),
                    slot.loc.node,
                    _table,
                    ledger,
                ));
            }
        }
    }

    pub fn register_slab(
        &mut self,
        tier: TierId,
        Ghost(slots): Ghost<Seq<SlotRef>>,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
        num_frames: usize,
        table: &TimelineTable,
        Tracked(ledger): Tracked<&KnowledgeLedger>,
        Ghost(node): Ghost<NodeId>,
    )
        requires
            old(self).well_formed(ledger.submitted.instance_id(), node, table, ledger),
            slots.len() == num_frames,
            old(self).metas.len() + num_frames <= u32::MAX,
            tier.valid(old(self).num_tiers),
            forall|i: usize|
                i < num_frames ==> tokens.contains_key(i) && (#[trigger] slots[i as int]).is_full()
                    && slots[i as int].slot_len == old(self).frame_size && slots[i as int].loc.node
                    == node && slot_owned(
                    tokens[i],
                    ledger.submitted.instance_id(),
                    slots[i as int].loc,
                    untouched(),
                ),
            forall|i: int, j: int|
                0 <= i < j < num_frames ==> (#[trigger] slots[i]).loc != (#[trigger] slots[j]).loc,
            forall|i: nat, j: int|
                i < old(self).metas.len() && 0 <= j < num_frames ==> (#[trigger] old(
                    self,
                ).ghosts@[i].slot).loc != (#[trigger] slots[j]).loc,
        ensures
            final(self).well_formed(ledger.submitted.instance_id(), node, table, ledger),
            final(self).num_tiers == old(self).num_tiers,
            final(self).frame_size == old(self).frame_size,
            final(self).versions == old(self).versions,
            final(self).metas.len() == old(self).metas.len() + num_frames,
            forall|j: nat|
                j < old(self).metas.len() ==> #[trigger] final(self).ghosts@[j] == old(
                    self,
                ).ghosts@[j] && final(self).metas[j as int] == old(self).metas[j as int],
            forall|j: nat|
                old(self).metas.len() <= j < final(self).metas.len() ==> (
                #[trigger] final(self).ghosts@[j]).tier == tier && final(self).metas[j as int].state
                    == STATE_EMPTY && final(self).metas[j as int].queued(),
            forall|j: int|
                0 <= j < num_frames ==> (#[trigger] final(self).ghosts@[(old(self).metas.len()
                    + j) as nat]).slot == slots[j],
            final(self).ready_queue[tier.index() as int].len() == old(
                self,
            ).ready_queue[tier.index() as int].len() + num_frames,
    {
        let t = tier.as_index();
        let tracked mut tokens = tokens;
        let total = self.metas.len() + num_frames;
        let extra = total.saturating_sub(self.ready_queue[t].len());
        self.ready_queue[t].reserve(extra);
        let extra = total.saturating_sub(self.draining_queue[t].len());
        self.draining_queue[t].reserve(extra);
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            table,
        ) by {
            assert(old(self).tier_queues(u, table));
        }
        let ghost size = self.frame_size;
        let ghost id = ledger.submitted.instance_id();
        let ghost first = self.metas.len();
        let ghost ready = old(self).ready_queue[t as int].len();
        let mut i = 0usize;
        while i < num_frames
            invariant
                i <= num_frames,
                self.num_tiers == old(self).num_tiers,
                self.frame_size == size,
                self.versions == old(self).versions,
                self.metas.len() == first + i,
                self.well_formed(id, node, table, ledger),
                self.ready_queue[t as int].len() == ready + i,
                forall|j: nat|
                    j < first ==> #[trigger] self.ghosts@[j] == old(self).ghosts@[j]
                        && self.metas[j as int] == old(self).metas[j as int],
                forall|j: nat|
                    first <= j < first + i ==> (#[trigger] self.ghosts@[j]).tier == tier
                        && self.metas[j as int].state == STATE_EMPTY
                        && self.metas[j as int].queued(),
                forall|j: nat|
                    first <= j < first + i ==> (#[trigger] self.ghosts@[j]).slot == slots[(j
                        - first) as int],
                forall|k: usize|
                    i <= k < num_frames ==> tokens.contains_key(k) && (
                    #[trigger] slots[k as int]).is_full() && slots[k as int].slot_len == size
                        && slots[k as int].loc.node == node && slot_owned(
                        tokens[k],
                        id,
                        slots[k as int].loc,
                        untouched(),
                    ),
            decreases num_frames - i,
        {
            let ghost slot = slots[i as int];
            let tracked tok;
            proof {
                tok = tokens.tracked_remove(i);
            }
            assert forall|j: nat| j < self.metas.len() implies (#[trigger] self.ghosts@[j].slot).loc
                != slot.loc by {
                if j < first {
                    assert(old(self).ghosts@[j].slot.loc != slots[i as int].loc);
                } else {
                    let prev = (j - first) as int;
                    assert(0 <= prev < i);
                    assert(slots[prev].loc != slots[i as int].loc);
                }
            }
            self.append_empty(tier, Ghost(slot), Tracked(tok), table, Tracked(ledger));
            i += 1;
        }
    }

    /// Reserve an already retired frame. A capacity error leaves the pool intact.
    pub fn reserve_ready(
        &mut self,
        tier: TierId,
        key: ObjectKey,
        version: VersionId,
        table: &TimelineTable,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<FrameId, CoreError>)
        requires
            old(self).consistent(v),
            v.timelines == *table,
            tier.valid(old(self).num_tiers),
            version.well_formed(),
            version != VersionId::NONE,
        ensures
            final(self).consistent(v),
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            r matches Err(_) ==> *final(self) == *old(self),
            r matches Ok(f) ==> {
                let g = final(self).frame(f);
                let m = final(self).meta(f);
                &&& old(self).meta(f).state == STATE_EMPTY
                &&& f.0 < final(self).metas.len() && m.state == STATE_INITIALIZING && g.tier == tier
                &&& g.bound_key == Some(key) && g.bound_version == Some(version) && !m.queued()
                &&& g.writable(v.id(), v.node, &v.ledger, ISet::empty())
                &&& !m.overflow() && forall|i: int|
                    0 <= i < 5 ==> table.retired(#[trigger] m.ticket(i))
            },
    {
        let Some(f) = self.alloc_ready_frame(tier, Ghost(v)) else {
            return Err(CoreError::CapacityExhausted);
        };
        let ghost allocated = *self;
        proof {
            self.lemma_observed_reusable_admits_any_write(f, table, v.node, &v.ledger);
        }
        self.begin_initializing(f, key, version, table, Ghost(v)).unwrap();
        assert forall|i: int| 0 <= i < 5 implies table.retired(
            #[trigger] self.meta(f).ticket(i),
        ) by {
            if i < 4 {
                assert(table.retired(allocated.meta(f).inline_read_tickets[i]));
            } else {
                assert(table.retired(allocated.meta(f).last_write_ticket));
            }
        }
        assert(self.frame(f).tier == tier);
        assert(!self.meta(f).queued() && !self.meta(f).overflow());
        assert(self.frame(f).writable(v.id(), v.node, &v.ledger, ISet::empty()));
        Ok(f)
    }

    /// Reserve an already retired frame and check out its exclusive `SlotToken`
    /// while preserving `FrameTable::consistent(v)`.
    pub fn reserve_ready_with_token(
        &mut self,
        tier: TierId,
        key: ObjectKey,
        version: VersionId,
        table: &TimelineTable,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<(FrameId, Tracked<SlotToken>), CoreError>)
        requires
            old(self).consistent(v),
            v.timelines == *table,
            tier.valid(old(self).num_tiers),
            version.well_formed(),
            version != VersionId::NONE,
        ensures
            final(self).consistent(v),
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            r matches Err(_) ==> *final(self) == *old(self),
            r matches Ok((f, tok)) ==> {
                let g = final(self).frame(f);
                let m = final(self).meta(f);
                &&& old(self).meta(f).state == STATE_EMPTY
                &&& f.0 < final(self).metas.len() && m.state == STATE_INITIALIZING && g.tier == tier
                &&& g.bound_key == Some(key) && g.bound_version == Some(version) && !m.queued()
                &&& g.token is None
                &&& slot_is_owned(tok@, v.id(), g.slot.loc)
                &&& write_admitted(slot_frontier(tok@), v.ledger.citations(v.node))
                &&& !m.overflow() && forall|i: int|
                    0 <= i < 5 ==> table.retired(#[trigger] m.ticket(i))
            },
    {
        let f = self.reserve_ready(tier, key, version, table, Ghost(v))?;
        let ghost reserved = *self;
        let Tracked(tok) = self.checkout_initializing_token(f, Ghost(v));
        proof {
            assert(v.ledger.citations(v.node).union(ISet::empty()) =~= v.ledger.citations(v.node));
            assert(write_admitted(slot_frontier(tok), v.ledger.citations(v.node)));
            assert forall|g: FrameId|
                g.0 < old(self).metas.len() && (#[trigger] old(self).meta(g)).state
                    == STATE_INITIALIZING implies self.meta(g) == old(self).meta(g) && self.frame(g)
                == old(self).frame(g) by {
                assert(g != f);
                assert(reserved.meta(g) == old(self).meta(g));
            }
            assert forall|g: FrameId| g.0 < self.metas.len() implies {
                &&& (#[trigger] self.frame(g)).slot == old(self).frame(g).slot
                &&& self.frame(g).tier == old(self).frame(g).tier
                &&& self.published(g) == old(self).published(g)
            } by {
                assert(reserved.frame(g).slot == old(self).frame(g).slot);
            }
        }
        Ok((f, Tracked(tok)))
    }
}

} // verus!
