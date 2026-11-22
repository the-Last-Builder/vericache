use super::*;

verus! {

impl FrameTable {
    pub open spec fn replaced_token(&self, before: Self, f: FrameId) -> bool {
        &&& self.num_tiers == before.num_tiers && self.bindings == before.bindings && self.metas
            == before.metas && self.versions == before.versions && self.frame_size
            == before.frame_size
        &&& self.ready_queue == before.ready_queue && self.draining_queue == before.draining_queue
            && self.draining_cursor == before.draining_cursor
        &&& self.ghosts@.dom() == before.ghosts@.dom()
        &&& self.frame(f) == (FrameGhost { token: self.frame(f).token, ..before.frame(f) })
        &&& forall|i: nat|
            i < self.metas.len() && i != f.0 ==> (#[trigger] self.ghosts@[i]) == before.ghosts@[i]
    }

    /// Check out the exclusive `SlotToken` from an initializing frame while keeping
    /// `FrameTable::consistent(v)` intact.
    pub fn checkout_initializing_token(&mut self, _f: FrameId, Ghost(v): Ghost<KernelView>) -> (tok:
        Tracked<SlotToken>)
        requires
            old(self).consistent(v),
            _f.0 < old(self).metas.len(),
            old(self).meta(_f).state == STATE_INITIALIZING,
            !old(self).meta(_f).queued(),
            old(self).frame(_f).token is Some,
        ensures
            final(self).consistent(v),
            final(self).replaced_token(*old(self), _f),
            final(self).same_directory(old(self)),
            final(self).preserves_initializing(old(self), Some(_f)),
            final(self).same_other_pins(old(self), None),
            final(self).frame(_f).token is None,
            Some(tok@) == old(self).frame(_f).token,
    {
        let ghost before = *self;
        proof {
            assert(before.valid_frame(_f, v));
        }
        let tracked tok;
        proof {
            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(_f.0 as nat);
            tok = g.token.tracked_take();
            self.ghosts.borrow_mut().tracked_insert(_f.0 as nat, g);
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
            assert(self.valid_frame(_f, v));
            self.queues_unchanged_off_frame(before, _f, v);
            self.restore_frame(before, _f, v);
        }
        Tracked(tok)
    }

    /// Return a checked-out `SlotToken` into an initializing frame after an observed
    /// or aborted write and move the frame back to `ready_queue`.
    pub fn abort_checked_out_write(
        &mut self,
        f: FrameId,
        tier: TierId,
        Tracked(tok): Tracked<SlotToken>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).meta(f).state == STATE_INITIALIZING,
            old(self).frame(f).tier == tier,
            !old(self).meta(f).queued(),
            old(self).frame(f).token is None,
            slot_is_owned(tok, v.id(), old(self).frame(f).slot.loc),
            write_admitted(slot_frontier(tok), v.ledger.citations(v.node)),
        ensures
            final(self).consistent(v),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).preserves_initializing(old(self), Some(f)),
            final(self).meta(f).state == STATE_EMPTY && final(self).meta(f).queued(),
            final(self).frame(f).token == Some(tok),
    {
        let ghost before = *self;
        proof {
            assert(before.valid_frame(f, v));
            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
            g.token = Some(tok);
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
        }
        self.metas[f.0 as usize].last_write_ticket = Ticket::NONE;
        proof {
            assert forall|i: int| 0 <= i < 5 implies v.timelines.valid_ticket(
                #[trigger] self.meta(f).ticket(i),
            ) by {
                assert(v.timelines.valid_ticket(before.meta(f).ticket(i)));
            }
            assert(self.frame(f).writable(v.id(), v.node, &v.ledger, ISet::empty())) by {
                assert(v.ledger.citations(v.node).union(ISet::empty()) =~= v.ledger.citations(
                    v.node,
                ));
            }
            assert(self.valid_frame(f, v));
            self.queues_unchanged_off_frame(before, f, v);
            self.restore_frame(before, f, v);
        }
        let t = tier.as_index();
        proof {
            assert(self.tier_queues(t as int, &v.timelines));
        }
        self.return_to_ready(f, t, Ghost(v));
    }

    /// A reaped completion, including a short or failed I/O, retires a parked
    /// initializing frame's exact write even when an earlier request holds the
    /// timeline prefix back.
    pub fn discard_observed_write(&mut self, f: FrameId, tier: TierId, Ghost(v): Ghost<KernelView>)
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).meta(f).state == STATE_INITIALIZING,
            old(self).frame(f).tier == tier,
            !old(self).meta(f).queued(),
            old(self).frame(f).writable(v.id(), v.node, &v.ledger, ISet::empty()),
        ensures
            final(self).consistent(v),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).preserves_initializing(old(self), Some(f)),
            final(self).meta(f).state == STATE_EMPTY && final(self).meta(f).queued(),
    {
        assert(self.valid_frame(f, v));
        let t = tier.as_index();
        assert(self.tier_queues(t as int, &v.timelines));
        self.metas[f.0 as usize].last_write_ticket = Ticket::NONE;
        proof {
            assert forall|i: int| 0 <= i < 5 implies v.timelines.valid_ticket(
                #[trigger] self.meta(f).ticket(i),
            ) by {
                assert(v.timelines.valid_ticket(old(self).meta(f).ticket(i)));
            }
            assert(self.frame(f).frontier().readers.subset_of(v.ledger.citations(v.node))) by {
                assert(old(self).frame(f).frontier().readers.subset_of(
                    v.ledger.citations(v.node).union(ISet::empty()),
                ));
            }
            assert(self.valid_frame(f, v));
            self.queues_unchanged_off_frame(*old(self), f, v);
            self.restore_frame(*old(self), f, v);
        }
        self.return_to_ready(f, t, Ghost(v));
    }

    /// Park a checked-out initializing frame back into `FrameTable` with its
    /// write ticket so asynchronous completion can later publish or discard it.
    pub fn park_initializing_write(
        &mut self,
        f: FrameId,
        writer: Ticket,
        Tracked(tok): Tracked<SlotToken>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len(),
            old(self).meta(f).state == STATE_INITIALIZING && !old(self).meta(f).queued(),
            old(self).frame(f).token is None,
            v.timelines.issued(writer.timeline_id(), writer.sequence()),
            writer != Ticket::NONE,
            slot_is_owned(tok, v.id(), old(self).frame(f).slot.loc),
            slot_frontier(tok).readers.subset_of(v.ledger.citations(v.node)),
            slot_frontier(tok).last_writer matches Some(w) ==> v.ledger.citations(v.node).contains(
                w,
            ) || v.timelines.ticket_covers(writer, w),
        ensures
            final(self).consistent(v),
            final(self).preserves_initializing(old(self), Some(f)),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), None),
            final(self).meta(f).state == STATE_INITIALIZING && !final(self).meta(f).queued(),
            final(self).meta(f).last_write_ticket == writer,
            forall|j: int|
                0 <= j < 4 ==> (#[trigger] final(self).meta(f).inline_read_tickets[j])
                    == Ticket::NONE,
            !final(self).meta(f).overflow(),
            final(self).frame(f) == (FrameGhost { token: Some(tok), ..old(self).frame(f) }),
            final(self).bindings == old(self).bindings,
            forall|other: FrameId|
                other.0 < old(self).metas.len() && other != f ==> #[trigger] final(self).meta(other)
                    == old(self).meta(other) && final(self).frame(other) == old(self).frame(other),
    {
        let ghost before = *self;
        let m = &mut self.metas[f.0 as usize];
        m.last_write_ticket = writer;
        m.inline_read_tickets = [Ticket::NONE;4];
        m.flags = 0;
        assert(0u8 & 1u8 == 0u8 && 0u8 & 2u8 == 0u8) by (bit_vector);
        proof {
            assert(before.valid_frame(f, v));
            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
            g.token = Some(tok);
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
            assert(self.valid_frame(f, v));
            self.queues_unchanged_off_frame(before, f, v);
            self.restore_frame(before, f, v);
        }
    }
}

} // verus!
