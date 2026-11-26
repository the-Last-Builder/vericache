use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

/// The slot descriptions match the owned, untouched tokens for this range.
pub open spec fn slab_slots_ready(
    tokens: Map<usize, SlotToken>,
    slots: Seq<SlotRef>,
    from: usize,
    count: usize,
    id: InstanceId,
    node: NodeId,
    size: Option<usize>,
) -> bool {
    forall|i: usize|
        from <= i < count ==> #[trigger] slab_slot_ready(tokens, slots, i, id, node, size)
}

pub open spec fn slab_slot_ready(
    tokens: Map<usize, SlotToken>,
    slots: Seq<SlotRef>,
    i: usize,
    id: InstanceId,
    node: NodeId,
    size: Option<usize>,
) -> bool {
    let slot = slots[i as int];
    &&& tokens.contains_key(i)
    &&& slot.is_full()
    &&& (size is None || slot.slot_len == size.unwrap())
    &&& slot.loc.node == node
    &&& slot_owned(tokens[i], id, slot.loc, untouched())
}

proof fn lemma_slab_take_slot(
    tokens: Map<usize, SlotToken>,
    slots: Seq<SlotRef>,
    i: usize,
    count: usize,
    id: InstanceId,
    node: NodeId,
    size: usize,
)
    requires
        i < count <= slots.len(),
        slab_slots_ready(tokens, slots, i, count, id, node, Some(size)),
    ensures
        tokens.contains_key(i),
        slots[i as int].is_full(),
        slots[i as int].slot_len == size,
        slots[i as int].loc.node == node,
        slot_owned(tokens[i], id, slots[i as int].loc, untouched()),
        slab_slots_ready(tokens.remove(i), slots, (i + 1) as usize, count, id, node, Some(size)),
{
    assert(slab_slot_ready(tokens, slots, i, id, node, Some(size)));
    let remaining = tokens.remove(i);
    let next = (i + 1) as usize;
    assert forall|k: usize| next <= k < count implies #[trigger] slab_slot_ready(
        remaining,
        slots,
        k,
        id,
        node,
        Some(size),
    ) by {
        assert(slab_slot_ready(tokens, slots, k, id, node, Some(size)));
        assert(remaining[k] == tokens[k]);
    }
    assert(slab_slots_ready(remaining, slots, next, count, id, node, Some(size)));
}

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
        slab_slots_ready(tokens, slots, 0, num_slots, id, node, Some(frame_size)),
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
    assert forall|i: usize| i < num_slots implies tokens.contains_key(i)
        && slots[i as int].is_full() && slots[i as int].slot_len == frame_size
        && slots[i as int].loc.node == node && slot_owned(
        #[trigger] tokens[i],
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
    assert forall|i: usize| i < num_slots implies #[trigger] slab_slot_ready(
        tokens,
        slots,
        i,
        id,
        node,
        Some(frame_size),
    ) by {
        assert(slot_owned(tokens[i], id, slots[i as int].loc, untouched()));
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
            slot.slot_len == old(self).tier_frame_size(tier),
            slot_owned(tok, tok.instance_id(), slot.loc, untouched()),
            forall|i: nat|
                i < old(self).metas.len() ==> (#[trigger] old(self).ghosts_spec()@[i].slot).loc
                    != slot.loc,
        ensures
            final(self).well_formed(tok.instance_id(), slot.loc.node, _table, ledger),
            final(self).num_tiers_spec() == old(self).num_tiers_spec(),
            final(self).frame_size_spec() == old(self).frame_size_spec(),
            final(self).tier_frame_sizes_spec() == old(self).tier_frame_sizes_spec(),
            final(self).versions == old(self).versions,
            final(self).metas_spec().len() == old(self).metas_spec().len() + 1,
            final(self).ghosts_spec()@[old(self).metas_spec().len() as nat].slot == slot,
            final(self).ghosts_spec()@[old(self).metas_spec().len() as nat].tier == tier,
            final(self).metas_spec()[old(self).metas_spec().len() as int].state == STATE_EMPTY,
            final(self).metas_spec()[old(self).metas_spec().len() as int].queued(),
            final(self).overflow_seqs_spec()[old(self).metas_spec().len() as int]@ == Seq::<
                u64,
            >::empty(),
            forall|i: nat|
                i < old(self).metas_spec().len() ==> #[trigger] final(self).ghosts_spec()@[i]
                    == old(self).ghosts_spec()@[i] && final(self).metas_spec()[i as int] == old(
                    self,
                ).metas_spec()[i as int] && final(self).overflow_seqs_spec()[i as int] == old(
                    self,
                ).overflow_seqs_spec()[i as int],
            final(self).ready_queue_spec()[tier.index() as int]@ == old(
                self,
            ).ready_queue_spec()[tier.index() as int]@.push(
                FrameId(old(self).metas_spec().len() as u32),
            ),
    {
        let idx = self.metas.len();
        let t = tier.as_index();
        let m = FrameSyncMeta::empty();
        let ovf = Vec::new();
        let tracked g = FrameGhost {
            slot,
            tier,
            token: Some(tok),
            local_shares: Map::empty(),
            fenced_shares: Set::empty(),
            cert: None,
            bound_key: None,
            bound_version: None,
        };
        assert(m.valid_tickets(_table));
        assert(g.authority(m, tok.instance_id()));
        assert(g.frontier() == untouched());
        assert(g.covered(m, ovf@, _table, slot.loc.node, ledger));
        assert(g.binding(m, _table));
        assert(2u8 & 2u8 != 0u8 && 2u8 & 1u8 == 0u8) by (bit_vector);
        assert(m.reusable(ovf@, _table));
        assert(g.valid(
            m,
            ovf@,
            self.num_tiers,
            self.tier_frame_size(tier),
            tok.instance_id(),
            slot.loc.node,
            _table,
            ledger,
        ));
        self.bindings.push((ObjectKey([0;4]), tier));
        self.metas.push(m);
        self.overflow_seqs.push(ovf);
        proof {
            self.ghosts.borrow_mut().tracked_insert(idx as nat, g);
        }
        self.ready_queue[t].push(FrameId(idx as u32));
        assert forall|u: int| 0 <= u < self.num_tiers_spec() implies #[trigger] self.tier_queues(
            u,
            _table,
        ) by {
            assert(old(self).tier_queues(u, _table));
            let ready = self.ready_queue_spec()[u]@;
            let draining = self.draining_queue_spec()[u]@;
            assert forall|i: int| 0 <= i < ready.len() implies {
                let f = #[trigger] ready[i];
                f.0 < self.metas_spec().len() && self.frame(f).tier.index() == u && self.meta(
                    f,
                ).state == STATE_EMPTY && self.meta(f).queued() && !self.meta(f).overflow()
                    && self.meta(f).reusable(self.ovf(f), _table)
            } by {
                if u != t || i < ready.len() - 1 {
                    assert(old(self).ready_queue_spec()[u][i].0 < idx);
                }
            }
            assert forall|i: int| 0 <= i < draining.len() implies {
                let f = #[trigger] draining[i];
                f.0 < self.metas_spec().len() && self.frame(f).tier.index() == u && self.meta(
                    f,
                ).state == STATE_DRAINING && !self.meta(f).queued()
            } by {
                assert(old(self).draining_queue_spec()[u][i].0 < idx);
            }
        }
        assert forall|i: nat|
            i < self.metas_spec().len() implies #[trigger] self.ghosts_spec()@[i].valid(
            self.metas_spec()[i as int],
            self.overflow_seqs_spec()[i as int]@,
            self.num_tiers_spec(),
            self.tier_frame_size(self.ghosts_spec()@[i].tier),
            tok.instance_id(),
            slot.loc.node,
            _table,
            ledger,
        ) by {
            if i < idx {
                assert(old(self).ghosts_spec()@[i].valid(
                    old(self).metas_spec()[i as int],
                    old(self).overflow_seqs_spec()[i as int]@,
                    old(self).num_tiers_spec(),
                    old(self).tier_frame_size(old(self).ghosts_spec()@[i].tier),
                    tok.instance_id(),
                    slot.loc.node,
                    _table,
                    ledger,
                ));
            }
        }
        assert(self.shape());
        assert(self.separated());
        assert forall|i: nat|
            i < self.metas_spec().len() implies #[trigger] self.ghosts_spec()@[i].version_agrees(
            self.versions_spec()@,
        ) by {
            if i < idx {
                assert(old(self).ghosts_spec()@[i].version_agrees(old(self).versions_spec()@));
            }
        }
        assert(self.values_agree());
        assert(self.queues(_table));
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
            old(self).metas_spec().len() + num_frames <= u32::MAX,
            tier.valid(old(self).num_tiers_spec()),
            slab_slots_ready(
                tokens,
                slots,
                0,
                num_frames,
                ledger.submitted.instance_id(),
                node,
                Some(old(self).tier_frame_size(tier)),
            ),
            forall|i: int, j: int|
                0 <= i < j < num_frames ==> (#[trigger] slots[i]).loc != (#[trigger] slots[j]).loc,
            forall|i: nat, j: int|
                i < old(self).metas_spec().len() && 0 <= j < num_frames ==> (#[trigger] old(
                    self,
                ).ghosts_spec()@[i].slot).loc != (#[trigger] slots[j]).loc,
        ensures
            final(self).well_formed(ledger.submitted.instance_id(), node, table, ledger),
            final(self).num_tiers_spec() == old(self).num_tiers_spec(),
            final(self).frame_size_spec() == old(self).frame_size_spec(),
            final(self).tier_frame_sizes_spec() == old(self).tier_frame_sizes_spec(),
            final(self).versions_spec() == old(self).versions_spec(),
            final(self).metas_spec().len() == old(self).metas_spec().len() + num_frames,
            forall|j: nat|
                j < old(self).metas_spec().len() ==> #[trigger] final(self).ghosts_spec()@[j]
                    == old(self).ghosts_spec()@[j] && final(self).metas_spec()[j as int] == old(
                    self,
                ).metas_spec()[j as int] && final(self).overflow_seqs_spec()[j as int] == old(
                    self,
                ).overflow_seqs_spec()[j as int],
            forall|j: nat|
                old(self).metas_spec().len() <= j < final(self).metas_spec().len() ==> (
                #[trigger] final(self).ghosts_spec()@[j]).tier == tier
                    && final(self).metas_spec()[j as int].state == STATE_EMPTY
                    && final(self).metas_spec()[j as int].queued(),
            forall|j: int|
                0 <= j < num_frames ==> (#[trigger] final(self).ghosts_spec()@[(old(
                    self,
                ).metas_spec().len() + j) as nat]).slot == slots[j],
            final(self).ready_queue_spec()[tier.index() as int].len() == old(
                self,
            ).ready_queue_spec()[tier.index() as int].len() + num_frames,
    {
        let t = tier.as_index();
        let tracked mut tokens = tokens;
        let total = self.metas.len() + num_frames;
        let extra = total.saturating_sub(self.ready_queue[t].len());
        self.ready_queue[t].reserve(extra);
        let extra = total.saturating_sub(self.draining_queue[t].len());
        self.draining_queue[t].reserve(extra);
        let ghost id = ledger.submitted.instance_id();
        proof {
            self.prove_valid_frames_unchanged(*old(self), None, id, node, table, ledger);
        }
        assert forall|u: int| 0 <= u < self.num_tiers_spec() implies #[trigger] self.tier_queues(
            u,
            table,
        ) by {
            assert(old(self).tier_queues(u, table));
        }
        assert(self.shape());
        assert forall|i: nat|
            i < self.metas_spec().len() implies #[trigger] self.ghosts_spec()@[i].version_agrees(
            self.versions_spec()@,
        ) by {
            assert(old(self).ghosts_spec()@[i].version_agrees(old(self).versions_spec()@));
        }
        assert(self.values_agree());
        assert(self.separated());
        assert(self.well_formed(id, node, table, ledger));
        let ghost size = self.tier_frame_size(tier);
        let ghost first = self.metas_spec().len();
        let ghost ready = old(self).ready_queue_spec()[t as int].len();
        assert(size == old(self).tier_frame_size(tier));
        let mut i = 0usize;
        while i < num_frames
            invariant
                i <= num_frames,
                self.num_tiers_spec() == old(self).num_tiers_spec(),
                self.frame_size_spec() == old(self).frame_size_spec(),
                self.tier_frame_sizes_spec() == old(self).tier_frame_sizes_spec(),
                self.tier_frame_size(tier) == size,
                self.versions == old(self).versions,
                self.metas_spec().len() == first + i,
                self.well_formed(id, node, table, ledger),
                self.ready_queue_spec()[t as int].len() == ready + i,
                forall|j: nat|
                    j < first ==> #[trigger] self.ghosts_spec()@[j] == old(self).ghosts_spec()@[j]
                        && self.metas_spec()[j as int] == old(self).metas_spec()[j as int]
                        && self.overflow_seqs_spec()[j as int] == old(
                        self,
                    ).overflow_seqs_spec()[j as int],
                forall|j: nat|
                    first <= j < first + i ==> (#[trigger] self.ghosts_spec()@[j]).tier == tier
                        && self.metas_spec()[j as int].state == STATE_EMPTY
                        && self.metas_spec()[j as int].queued(),
                forall|j: nat|
                    first <= j < first + i ==> (#[trigger] self.ghosts_spec()@[j]).slot == slots[(j
                        - first) as int],
                slab_slots_ready(tokens, slots, i, num_frames, id, node, Some(size)),
            decreases num_frames - i,
        {
            let ghost slot = slots[i as int];
            let tracked tok;
            proof {
                lemma_slab_take_slot(tokens, slots, i, num_frames, id, node, size);
                tok = tokens.tracked_remove(i);
            }
            assert forall|j: nat| j < self.metas_spec().len() implies (
            #[trigger] self.ghosts_spec()@[j].slot).loc != slot.loc by {
                if j < first {
                    assert(old(self).ghosts_spec()@[j].slot.loc != slots[i as int].loc);
                } else {
                    let prev = (j - first) as int;
                    assert(0 <= prev < i);
                    assert(slots[prev].loc != slots[i as int].loc);
                }
            }
            let ghost before = *self;
            self.append_empty(tier, Ghost(slot), Tracked(tok), table, Tracked(ledger));
            assert forall|j: nat| j < first implies #[trigger] self.ghosts_spec()@[j] == old(
                self,
            ).ghosts_spec()@[j] && self.metas_spec()[j as int] == old(self).metas_spec()[j as int]
                && self.overflow_seqs_spec()[j as int] == old(
                self,
            ).overflow_seqs_spec()[j as int] by {
                assert(j < before.metas_spec().len());
                assert(before.ghosts_spec()@[j] == old(self).ghosts_spec()@[j]);
                assert(self.ghosts_spec()@[j] == before.ghosts_spec()@[j]);
            }
            i += 1;
        }
    }

    /// Reserve an already retired frame. A capacity error leaves the pool intact.
    pub(crate) fn reserve_ready(
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
            tier.valid(old(self).num_tiers_spec()),
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
                &&& f.0 < final(self).metas_spec().len() && m.state == STATE_INITIALIZING && g.tier
                    == tier
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
