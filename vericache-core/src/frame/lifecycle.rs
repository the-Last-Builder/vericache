use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl FrameTable {
    fn bind_initializing(
        &mut self,
        f: FrameId,
        _key: ObjectKey,
        version: VersionId,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).meta(f).available(),
            !old(self).meta(f).queued(),
            version.well_formed(),
            version != VersionId::NONE,
            old(self).meta(f).state != STATE_DRAINING || !old(self).draining_queue[old(self).frame(
                f,
            ).tier.index() as int]@.contains(f),
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).frame(f).token == old(self).frame(f).token,
            final(self).meta(f).flags == old(self).meta(f).flags,
            final(self).meta(f).last_write_ticket == old(self).meta(f).last_write_ticket,
            final(self).meta(f).inline_read_tickets == old(self).meta(f).inline_read_tickets,
            final(self).consistent(v),
            final(self).meta(f).state == STATE_INITIALIZING,
            final(self).frame(f).bound_key == Some(_key),
            final(self).frame(f).bound_version == Some(version),
            final(self).frame(f).slot == old(self).frame(f).slot,
    {
        let index = f.0 as usize;
        assert(self.valid_frame(f, v));
        self.metas[index].state = STATE_INITIALIZING;
        self.metas[index].version_packed = version.pack();
        self.bindings[index].0 = _key;
        proof {
            let tracked g = self.ghosts.borrow_mut().tracked_borrow_mut(f.0 as nat);
            g.cert = None;
            g.bound_key = Some(_key);
            g.bound_version = Some(version);
        }
        proof {
            assert(self.ghosts@.dom() =~= old(self).ghosts@.dom());
            self.preserve_coverage(*old(self), f, v);
            self.queues_unchanged_off_frame(*old(self), f, v);
            self.restore_frame(*old(self), f, v);
        }
    }

    fn remove_draining(&mut self, t: usize, index: usize, Ghost(v): Ghost<KernelView>) -> (f:
        FrameId)
        requires
            old(self).consistent(v),
            t < old(self).num_tiers,
            index < old(self).draining_queue[t as int].len(),
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).consistent(v),
            final(self).metas == old(self).metas,
            final(self).ghosts == old(self).ghosts,
            final(self).ready_queue == old(self).ready_queue,
            f == old(self).draining_queue[t as int][index as int],
            !final(self).draining_queue[t as int]@.contains(f),
            final(self).draining_queue[t as int].len() + 1 == old(
                self,
            ).draining_queue[t as int].len(),
    {
        let ghost prior = self.draining_queue[t as int]@;
        assert(self.tier_queues(t as int, &v.timelines));
        let f = self.draining_queue[t].swap_remove(index);
        let ghost after = self.draining_queue[t as int]@;
        assert forall|i: int| 0 <= i < after.len() implies #[trigger] after[i] == prior[if i
            == index {
            prior.len() - 1
        } else {
            i
        }] by {}
        assert forall|i: int, j: int|
            0 <= i < after.len() && 0 <= j < after.len() && i != j implies (#[trigger] after[i])
            != (#[trigger] after[j]) by {
            let a = if i == index {
                prior.len() - 1
            } else {
                i
            };
            let b = if j == index {
                prior.len() - 1
            } else {
                j
            };
            assert(prior[a] != prior[b]);
        }
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &v.timelines,
        ) by {
            assert(old(self).tier_queues(u, &v.timelines));
        }
        f
    }

    pub(super) fn return_to_ready(&mut self, f: FrameId, t: usize, Ghost(v): Ghost<KernelView>)
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).frame(f).tier.index() == t,
            t < old(self).num_tiers,
            old(self).meta(f).state == STATE_INITIALIZING || old(self).meta(f).state
                == STATE_DRAINING,
            !old(self).draining_queue[t as int]@.contains(f),
            old(self).frame(f).writable(v.id(), v.node, &v.ledger, ISet::empty()),
            v.timelines.retired(old(self).meta(f).last_write_ticket),
        ensures
            final(self).preserves_initializing(old(self), Some(f)),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).consistent(v),
            final(self).meta(f).state == STATE_EMPTY,
            final(self).meta(f).queued(),
            final(self).frame(f).token == old(self).frame(f).token,
            final(self).ready_queue[t as int]@ == old(self).ready_queue[t as int]@.push(f),
            final(self).draining_queue == old(self).draining_queue,
    {
        assert(old(self).valid_frame(f, v));
        let last = self.metas[f.0 as usize].last_write_ticket;
        let mut m = FrameSyncMeta::empty();
        m.last_write_ticket = last;
        self.metas.set(f.0 as usize, m);
        self.ready_queue[t].push(f);
        proof {
            let tracked g = self.ghosts.borrow_mut().tracked_borrow_mut(f.0 as nat);
            g.bound_key = None;
            g.bound_version = None;
            g.cert = None;
        }
        assert(self.ghosts@.dom() =~= old(self).ghosts@.dom());
        assert(2u8 & 2u8 != 0u8 && 2u8 & 1u8 == 0u8) by (bit_vector);
        assert(self.frame(f).frontier() == old(self).frame(f).frontier());
        assert(m.valid_tickets(&v.timelines)) by {
            assert(old(self).meta(f).ticket(4) == last);
        }
        assert(self.frame(f).frontier().readers.subset_of(v.ledger.citations(v.node))) by {
            assert(old(self).frame(f).frontier().readers.subset_of(
                v.ledger.citations(v.node).union(ISet::empty()),
            ));
        }
        assert(self.frame(f).covered(m, &v.timelines, v.node, &v.ledger));
        assert(self.valid_frame(f, v));
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &v.timelines,
        ) by {
            assert(old(self).tier_queues(u, &v.timelines));
        }
        proof {
            self.restore_frame(*old(self), f, v);
        }
    }

    fn set_cursor(&mut self, t: usize, i: usize, Ghost(v): Ghost<KernelView>)
        requires
            t < old(self).num_tiers,
            old(self).consistent(v),
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).consistent(v),
            final(self).ready_queue == old(self).ready_queue,
            final(self).draining_queue == old(self).draining_queue,
    {
        self.draining_cursor[t] = i;
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &v.timelines,
        ) by {
            assert(old(self).tier_queues(u, &v.timelines));
        }
    }

    pub fn alloc_ready_frame(&mut self, tier: TierId, Ghost(v): Ghost<KernelView>) -> (r: Option<
        FrameId,
    >)
        requires
            old(self).consistent(v),
            tier.valid(old(self).num_tiers),
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).consistent(v),
            final(self).ghosts == old(self).ghosts,
            r matches Some(f) ==> old(self).meta(f).state == STATE_EMPTY,
            r.is_none() ==> *final(self) == *old(self),
            r matches Some(f) ==> !final(self).meta(f).overflow(),
            r.is_none() == (old(self).ready_queue[tier.index() as int].len() == 0),
            r matches Some(f) ==> f.0 < final(self).metas.len() && f == old(
                self,
            ).ready_queue[tier.index() as int]@.last() && final(self).meta(f).state == STATE_EMPTY
                && !final(self).meta(f).queued() && final(self).meta(f).reusable(&v.timelines)
                && final(self).frame(f).tier == tier,
    {
        let t = tier.as_index();
        assert(self.tier_queues(t as int, &v.timelines));
        if self.ready_queue[t].is_empty() {
            return None;
        }
        let f = self.ready_queue[t].pop().unwrap();
        let ghost idx = old(self).ready_queue[t as int].len() - 1;
        assert(old(self).ready_queue[t as int][idx] == f);
        assert(old(self).valid_frame(f, v));
        assert(self.frame(f).tier == tier);
        let m = &mut self.metas[f.0 as usize];
        let flags = m.flags;
        m.flags = flags & !FLAG_IN_READY_QUEUE;
        assert((flags & !2u8) & 2u8 == 0u8 && ((flags & !2u8) & 1u8) == (flags & 1u8))
            by (bit_vector);
        proof {
            self.preserve_coverage(*old(self), f, v);
        }
        assert(self.valid_frame(f, v));
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &v.timelines,
        ) by {
            assert(old(self).tier_queues(u, &v.timelines));
            assert forall|i: int| 0 <= i < self.ready_queue[u].len() implies (
            #[trigger] self.ready_queue[u][i]) != f by {
                assert(old(self).ready_queue[u][i] == self.ready_queue[u][i]);
            }
        }
        proof {
            self.restore_frame(*old(self), f, v);
        }
        Some(f)
    }

    pub fn begin_initializing(
        &mut self,
        f: FrameId,
        _key: ObjectKey,
        version: VersionId,
        table: &TimelineTable,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).consistent(v),
            version.well_formed(),
            *table == v.timelines,
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).frame(f).token == old(self).frame(f).token,
            final(self).meta(f).flags == old(self).meta(f).flags,
            final(self).meta(f).last_write_ticket == old(self).meta(f).last_write_ticket,
            final(self).meta(f).inline_read_tickets == old(self).meta(f).inline_read_tickets,
            f.0 < old(self).metas.len() && old(self).meta(f).state == STATE_EMPTY && !old(
                self,
            ).meta(f).queued() && old(self).meta(f).reusable(table) && version != VersionId::NONE
                ==> r.is_ok(),
            final(self).consistent(v),
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> final(self).meta(f).state == STATE_INITIALIZING && final(self).frame(
                f,
            ).bound_key == Some(_key) && final(self).frame(f).bound_version == Some(version),
    {
        if f.0 as usize >= self.metas.len() {
            return Err(CoreError::InvalidArgument);
        }
        let index = f.0 as usize;
        if self.metas[index].state != STATE_EMPTY || self.metas[index].flags & FLAG_IN_READY_QUEUE
            != 0 {
            return Err(CoreError::WouldConflict);
        }
        if version.seq == 0 && version.origin_node == 0 {
            return Err(CoreError::InvalidArgument);
        }
        assert(self.valid_frame(f, v));
        if !self.is_reusable_observed(table, f) {
            return Err(CoreError::WouldConflict);
        }
        self.bind_initializing(f, _key, version, Ghost(v));
        Ok(())
    }

    /// Commit a backend-neutral ordered handoff using the victim's position in its draining queue.
    pub fn begin_ordered_initializing(
        &mut self,
        f: FrameId,
        tier: TierId,
        queue_index: usize,
        key: ObjectKey,
        version: VersionId,
        Ghost(after): Ghost<ISet<TransferEvent>>,
        Ghost(covered): Ghost<ISet<TransferEvent>>,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).consistent(v),
            version.well_formed(),
            tier.valid(old(self).num_tiers),
            f.0 < old(self).metas.len(),
            old(self).meta(f).ordered_eligible(),
            old(self).meta(f).ordered(&v.timelines, covered),
            after.subset_of(v.ledger.submitted.dom()),
            covered.subset_of(
                foundation::tracked::closure_from(
                    v.ledger.submitted.map(),
                    after.union(v.ledger.citations(v.node)),
                ),
            ),
        ensures
            final(self).consistent(v),
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> final(self).meta(f).state == STATE_INITIALIZING && final(self).frame(
                f,
            ).bound_key == Some(key) && final(self).frame(f).bound_version == Some(version)
                && final(self).frame(f).writable(v.id(), v.node, &v.ledger, covered)
                && final(self).frame(f).write_admitted_in(v.node, &v.ledger, after),
    {
        let t = tier.as_index();
        if queue_index >= self.draining_queue[t].len() || self.draining_queue[t][queue_index].0
            != f.0 || (version.origin_node == 0 && version.seq == 0) {
            return Err(CoreError::InvalidArgument);
        }
        proof {
            assert(self.tier_queues(t as int, &v.timelines));
            self.lemma_ordered_handoff_admits_writer(
                f,
                &v.timelines,
                v.node,
                &v.ledger,
                after,
                covered,
            );
        }
        self.remove_draining(t, queue_index, Ghost(v));
        self.bind_initializing(f, key, version, Ghost(v));
        Ok(())
    }

    /// Transition a parked initializing frame to published after its write is completed.
    pub fn publish_frame(
        &mut self,
        f: FrameId,
        writer: Ticket,
        Ghost(cert): Ghost<ValueCertificate>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).except_frame(f, v),
            old(self).versions@.contains_key(old(self).frame(f).bound_version->Some_0) ==> old(
                self,
            ).versions@[old(self).frame(f).bound_version->Some_0] == cert.bytes,
            old(self).meta(f).state == STATE_INITIALIZING,
            !old(self).meta(f).queued(),
            old(self).frame(f).binding(old(self).meta(f), &v.timelines),
            old(self).frame(f).authority(old(self).meta(f), v.id()),
            old(self).frame(f).completed_writes(cert),
            cert.slot == old(self).frame(f).slot,
            cert.bytes.len() == cert.slot.slot_len,
            v.timelines.issued(writer.timeline_id(), writer.sequence()),
            writer != Ticket::NONE,
            cert.writer == v.timelines.events@[(writer.timeline_id(), writer.sequence())],
            old(self).frame(f).tier.valid(old(self).num_tiers),
            cert.slot.loc.node == v.node,
            cert.slot.slot_len == old(self).frame_size,
            cert.slot.is_full(),
        ensures
            final(self).num_tiers == old(self).num_tiers,
            final(self).frame_size == old(self).frame_size,
            final(self).versions@ == old(self).versions@.insert(
                old(self).frame(f).bound_version.unwrap(),
                cert.bytes,
            ),
            final(self).frame(f) == (FrameGhost { cert: Some(cert), ..old(self).frame(f) }),
            final(self).consistent(v),
            final(self).bindings == old(self).bindings,
            final(self).metas.len() == old(self).metas.len(),
            final(self).frame(f).bound_version == old(self).frame(f).bound_version,
            final(self).frame(f).bound_key == old(self).frame(f).bound_key,
            final(self).frame(f).tier == old(self).frame(f).tier,
            forall|g: FrameId|
                g != f && g.0 < old(self).metas.len() ==> #[trigger] final(self).meta(g) == old(
                    self,
                ).meta(g) && final(self).frame(g) == old(self).frame(g),
            old(self).versions@.submap_of(final(self).versions@),
            final(self).versions@[final(self).frame(f).bound_version->Some_0] == cert.bytes,
            final(self).meta(f).state == STATE_PUBLISHED,
            final(self).frame(f).cert == Some(cert),
            final(self).meta(f).last_write_ticket == writer,
    {
        proof {
            self.versions = Ghost(
                self.versions@.insert(self.frame(f).bound_version->Some_0, cert.bytes),
            );
        }
        let m = &mut self.metas[f.0 as usize];
        m.state = STATE_PUBLISHED;
        m.last_write_ticket = writer;
        m.inline_read_tickets = [Ticket::NONE;4];
        m.flags = 0;
        proof {
            self.ghosts.borrow_mut().tracked_borrow_mut(f.0 as nat).cert = Some(cert);
            assert(self.ghosts@.dom() =~= old(self).ghosts@.dom());
            self.queues_unchanged_off_frame(*old(self), f, v);
            self.restore_frame(*old(self), f, v);
        }
    }

    /// Re-install a checked-out `SlotToken` and publish the initializing frame in one step.
    pub fn publish_with_token(
        &mut self,
        f: FrameId,
        Tracked(tok): Tracked<SlotToken>,
        writer: Ticket,
        Ghost(cert): Ghost<ValueCertificate>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).meta(f).state == STATE_INITIALIZING,
            !old(self).meta(f).queued(),
            old(self).frame(f).token is None,
            old(self).versions@.contains_key(old(self).frame(f).bound_version->Some_0) ==> old(
                self,
            ).versions@[old(self).frame(f).bound_version->Some_0] == cert.bytes,
            slot_is_owned(tok, v.id(), old(self).frame(f).slot.loc),
            slot_frontier(tok).readers.is_empty(),
            slot_frontier(tok).last_writer == Some(cert.writer),
            cert.slot == old(self).frame(f).slot,
            cert.bytes.len() == cert.slot.slot_len,
            v.timelines.issued(writer.timeline_id(), writer.sequence()),
            writer != Ticket::NONE,
            cert.writer == v.timelines.events@[(writer.timeline_id(), writer.sequence())],
        ensures
            final(self).num_tiers == old(self).num_tiers,
            final(self).frame_size == old(self).frame_size,
            final(self).versions@ == old(self).versions@.insert(
                old(self).frame(f).bound_version.unwrap(),
                cert.bytes,
            ),
            final(self).frame(f) == (FrameGhost {
                token: Some(tok),
                cert: Some(cert),
                ..old(self).frame(f)
            }),
            final(self).consistent(v),
            final(self).bindings == old(self).bindings,
            final(self).metas.len() == old(self).metas.len(),
            final(self).frame(f).bound_version == old(self).frame(f).bound_version,
            final(self).frame(f).bound_key == old(self).frame(f).bound_key,
            final(self).frame(f).tier == old(self).frame(f).tier,
            forall|g: FrameId|
                g != f && g.0 < old(self).metas.len() ==> #[trigger] final(self).meta(g) == old(
                    self,
                ).meta(g) && final(self).frame(g) == old(self).frame(g),
            old(self).versions@.submap_of(final(self).versions@),
            final(self).versions@[final(self).frame(f).bound_version->Some_0] == cert.bytes,
            final(self).meta(f).state == STATE_PUBLISHED,
            final(self).frame(f).cert == Some(cert),
            final(self).meta(f).last_write_ticket == writer,
    {
        proof {
            assert(old(self).valid_frame(f, v));
            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
            g.token = Some(tok);
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@.dom() =~= old(self).ghosts@.dom());
            assert(self.frame(f).slot == old(self).frame(f).slot);
            assert(self.frame(f).tier == old(self).frame(f).tier);
            assert(self.frame(f).cert == old(self).frame(f).cert);
            assert(self.values_agree());
            assert(self.separated());
            self.queues_unchanged_off_frame(*old(self), f, v);
            assert(self.except_frame(f, v));
        }
        self.publish_frame(f, writer, Ghost(cert), Ghost(v));
    }

    pub fn evict_frame(&mut self, f: FrameId, tier: TierId, Ghost(v): Ghost<KernelView>) -> (r:
        Result<(), CoreError>)
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).frame(f).tier == tier,
        ensures
            final(self).num_tiers == old(self).num_tiers,
            final(self).versions == old(self).versions,
            final(self).frame_size == old(self).frame_size,
            final(self).metas.len() == old(self).metas.len(),
            final(self).consistent(v),
            final(self).bindings == old(self).bindings,
            r.is_ok() == (old(self).meta(f).state == STATE_PUBLISHED),
            forall|g: FrameId|
                g != f && g.0 < old(self).metas.len() ==> #[trigger] final(self).meta(g) == old(
                    self,
                ).meta(g),
            r.is_ok() ==> final(self).meta(f).state == STATE_DRAINING,
            final(self).meta(f).active_remote_leases == old(self).meta(f).active_remote_leases,
            final(self).ghosts == old(self).ghosts,
            r.is_err() ==> *final(self) == *old(self),
    {
        let i = f.0 as usize;
        if self.metas[i].state != STATE_PUBLISHED {
            return Err(CoreError::StaleReplica);
        }
        assert(self.valid_frame(f, v));
        let t = tier.as_index();
        self.draining_queue[t].push(f);
        self.metas[i].state = STATE_DRAINING;
        self.metas[i].flags &= !FLAG_IN_READY_QUEUE;
        let ghost flags = old(self).meta(f).flags;
        assert((flags & !2u8) & 2u8 == 0u8 && ((flags & !2u8) & 1u8) == (flags & 1u8))
            by (bit_vector);
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &v.timelines,
        ) by {
            assert(old(self).tier_queues(u, &v.timelines));
        }
        proof {
            self.preserve_coverage(*old(self), f, v);
            self.restore_frame(*old(self), f, v);
        }
        Ok(())
    }

    pub fn abort_initializing(
        &mut self,
        f: FrameId,
        tier: TierId,
        table: &TimelineTable,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).consistent(v),
            *table == v.timelines,
            f.0 < old(self).metas.len(),
            old(self).frame(f).tier == tier,
            old(self).frame(f).token is Some,
        ensures
            final(self).preserves_initializing(old(self), Some(f)),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).consistent(v),
            r.is_ok() ==> final(self).meta(f).state == STATE_EMPTY && final(self).meta(f).queued(),
            r.is_err() ==> *final(self) == *old(self),
    {
        let m = &self.metas[f.0 as usize];
        if m.state != STATE_INITIALIZING {
            return Err(CoreError::WouldConflict);
        }
        assert(self.valid_frame(f, v));
        for i in 0..5
            invariant
                forall|j: int| 0 <= j < i ==> table.retired(#[trigger] m.ticket(j)),
        {
            let k = m.dependency(i);
            if !table.is_retired(k) {
                return Err(CoreError::WouldConflict);
            }
        }
        if m.flags & FLAG_READ_OVERFLOW != 0 && !table.overflow_all_retired() {
            return Err(CoreError::WouldConflict);
        }
        let t = tier.as_index();
        proof {
            assert(m.ticket(4) == m.last_write_ticket);
            self.frame(f).prove_frontier_ordered(*m, table, v.node, &v.ledger, ISet::empty());
            assert(self.tier_queues(t as int, table));
        }
        self.return_to_ready(f, t, Ghost(v));
        Ok(())
    }

    pub fn reclaim_draining(
        &mut self,
        table: &TimelineTable,
        tier: TierId,
        budget: usize,
        Ghost(v): Ghost<KernelView>,
    ) -> (reclaimed: usize)
        requires
            old(self).consistent(v),
            tier.valid(old(self).num_tiers),
            v.timelines == *table,
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).consistent(v),
            reclaimed <= budget,
            final(self).ready_queue[tier.index() as int].len() == old(
                self,
            ).ready_queue[tier.index() as int].len() + reclaimed,
            final(self).draining_queue[tier.index() as int].len() + reclaimed == old(
                self,
            ).draining_queue[tier.index() as int].len(),
    {
        let t = tier.as_index();
        let limit = budget.min(self.draining_queue[t].len());
        let mut reclaimed = 0;
        for _visited in 0..limit
            invariant
                self.consistent(v),
                self.same_directory(old(self)),
                self.preserves_initializing(old(self), None),
                self.same_other_pins(old(self), None),
                reclaimed <= _visited,
                self.ready_queue[t as int].len() == old(self).ready_queue[t as int].len()
                    + reclaimed,
                self.draining_queue[t as int].len() + reclaimed == old(
                    self,
                ).draining_queue[t as int].len(),
        {
            let n = self.draining_queue[t].len();
            let i = if self.draining_cursor[t] >= n {
                0
            } else {
                self.draining_cursor[t]
            };
            let f = self.draining_queue[t][i];
            assert(self.tier_queues(t as int, table));
            assert(self.valid_frame(f, v));
            if self.is_reusable_observed(table, f) {
                proof {
                    self.lemma_observed_reusable_admits_any_write(f, table, v.node, &v.ledger);
                }
                self.remove_draining(t, i, Ghost(v));
                self.return_to_ready(f, t, Ghost(v));
                reclaimed += 1;
                self.set_cursor(t, i, Ghost(v));
            } else {
                self.set_cursor(t, i + 1, Ghost(v));
            }
        }
        reclaimed
    }
}

} // verus!
