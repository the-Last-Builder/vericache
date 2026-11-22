use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl FrameTable {
    pub(super) fn accepts_lease_return(&self, f: FrameId, slot: u32, generation: u32) -> (r: bool)
        ensures
            r == (f.0 < self.metas.len() && self.meta(f).active_remote_leases != 0 && (self.meta(
                f,
            ).mw_slot == u32::MAX || (self.meta(f).mw_slot == slot && self.meta(f).mw_gen
                == generation))),
    {
        let idx = f.0 as usize;
        if idx >= self.metas.len() {
            return false;
        }
        let m = &self.metas[idx];
        m.active_remote_leases != 0 && (m.mw_slot == u32::MAX || (m.mw_slot == slot && m.mw_gen
            == generation))
    }

    pub(super) fn pin_share(
        &mut self,
        f: FrameId,
        remote: Option<(u32, u32)>,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(names): Tracked<&mut ShareSource>,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<Tracked<ShareCap>, CoreError>)
        requires
            old(self).consistent(v),
            instance.id() == v.id(),
            old(names).well_formed(),
            old(names).id() == v.id(),
            old(names).node() == v.node,
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), Some(f)),
            final(self).metas.len() == old(self).metas.len(),
            final(self).consistent(v),
            final(names).well_formed(),
            final(names).id() == v.id(),
            final(names).node() == v.node,
            r.is_ok() && !old(self).frame(f).live_shares().contains(old(names).next_share())
                ==> final(names).names() == old(names).names().tail(),
            r.is_ok() == (f.0 < old(self).metas.len() && old(self).meta(f).state == STATE_PUBLISHED
                && match remote {
                None => old(self).meta(f).active_local_reads < u32::MAX,
                Some((slot, _)) => old(self).meta(f).active_remote_leases < u16::MAX && slot
                    != u32::MAX,
            }),
            r.is_ok() ==> final(self).meta(f).state == STATE_PUBLISHED,
            remote matches Some((slot, generation)) ==> (r.is_ok() && old(self).meta(
                f,
            ).active_remote_leases == 0 ==> final(self).meta(f).mw_slot == slot && final(self).meta(
                f,
            ).mw_gen == generation),
            match r {
                Err(_) => *final(self) == *old(self) && *final(names) == *old(names),
                Ok(cap) => {
                    let share = cap@.share();
                    let g = old(self).frame(f);
                    &&& !g.live_shares().contains(share)
                    &&& !g.live_shares().contains(old(names).next_share()) ==> share == old(
                        names,
                    ).next_share()
                    &&& cap@.is_unbound()
                    &&& cap@.instance_id() == v.id()
                    &&& cap@.loc() == g.slot.loc
                    &&& cap@.frontier() == (Frontier {
                        last_writer: g.frontier().last_writer,
                        readers: ISet::empty(),
                    })
                    &&& final(self).frame(f).slot == g.slot
                    &&& final(self).frame(f).local_shares == match remote {
                        None => g.local_shares.insert(share, LocalShareEntry { share }),
                        Some(_) => g.local_shares,
                    }
                    &&& final(self).frame(f).remote_shares == match remote {
                        Some((slot, generation)) => g.remote_shares.insert(
                            share,
                            RemoteShareEntry { mw_slot: slot, mw_gen: generation },
                        ),
                        None => g.remote_shares,
                    }
                },
            },
    {
        let idx = f.0 as usize;
        if idx >= self.metas.len() || self.metas[idx].state != STATE_PUBLISHED {
            return Err(CoreError::StaleReplica);
        }
        if (remote.is_none() && self.metas[idx].active_local_reads == u32::MAX) || (remote.is_some()
            && self.metas[idx].active_remote_leases == u16::MAX) {
            return Err(CoreError::CapacityExhausted);
        }
        if let Some((slot, _)) = remote {
            if slot == u32::MAX {
                return Err(CoreError::InvalidArgument);
            }
        }
        assert(self.valid_frame(f, v));
        let ghost prior = self.frame(f);
        let tracked mut g = self.ghosts.borrow_mut().tracked_remove(idx as nat);
        let tracked name = Self::allocate_share(
            names,
            g.local_shares.dom().union(g.remote_shares.dom()),
        );
        let ghost share = name.element();
        let tracked cur_tok;
        proof {
            cur_tok = g.token.tracked_take();
        }
        let tracked (Tracked(parent), Tracked(cap)) = if self.meta(f).active_local_reads == 0
            && self.meta(f).active_remote_leases == 0 {
            assert(prior.local_shares.dom() =~= Set::empty());
            assert(prior.remote_shares.dom() =~= Set::empty());
            assert(prior.live_shares() =~= ISet::empty());
            instance.lend_slot(cur_tok, name)
        } else {
            instance.lend_more_slot(cur_tok, name)
        };
        proof {
            g.token = Some(parent);
        }
        match remote {
            Some((slot, generation)) => {
                let m = &mut self.metas[idx];
                m.active_remote_leases += 1;
                if m.active_remote_leases == 1 {
                    m.mw_slot = slot;
                    m.mw_gen = generation;
                } else {
                    m.mw_slot = u32::MAX;
                    m.mw_gen = 0;
                }
                proof {
                    g.remote_shares = g.remote_shares.insert(
                        share,
                        RemoteShareEntry { mw_slot: slot, mw_gen: generation },
                    );
                }
            },
            None => {
                self.metas[idx].active_local_reads += 1;
                proof {
                    g.local_shares = g.local_shares.insert(share, LocalShareEntry { share });
                }
            },
        }
        proof {
            assert(g.live_shares() =~= prior.live_shares().insert(share));
            assert(g.slot_tok().value() == (Slot::Lent {
                base: prior.frontier(),
                shares: g.live_shares(),
            }));
            self.ghosts.borrow_mut().tracked_insert(idx as nat, g);
            assert(self.ghosts@.dom() =~= old(self).ghosts@.dom());
            assert forall|i: int| 0 <= i < 5 implies v.timelines.valid_ticket(
                #[trigger] self.meta(f).ticket(i),
            ) by {
                assert(self.meta(f).ticket(i) == old(self).meta(f).ticket(i));
            }
            g.preserve_readers(
                prior,
                self.meta(f),
                old(self).meta(f),
                &v.timelines,
                &v.timelines,
                v.node,
                &v.ledger,
            );
            assert(g.local_shares.len() == self.meta(f).active_local_reads);
            assert(g.remote_shares.len() == self.meta(f).active_remote_leases);
            assert(g.local_shares.dom().disjoint(g.remote_shares.dom()));
            if remote.is_some() && self.meta(f).mw_slot != u32::MAX {
                assert(g.remote_shares.contains_key(share));
                assert(g.remote_shares[share].mw_slot == self.meta(f).mw_slot);
            } else if self.meta(f).mw_slot != u32::MAX {
                let m = old(self).meta(f);
                let cached = choose|s|
                    prior.remote_shares.contains_key(s) && #[trigger] prior.remote_shares[s].mw_slot
                        == m.mw_slot && prior.remote_shares[s].mw_gen == m.mw_gen;
                assert(g.remote_shares[cached] == prior.remote_shares[cached]);
            }
            assert(g.authority(self.meta(f), v.id()));
            assert(g.binding(self.meta(f), &v.timelines));
            self.refresh_lease(*old(self), f, v);
        }
        Ok(Tracked(cap))
    }

    /// Install the parent `SlotToken` produced by a share return or a hardware fence.
    pub(super) fn finish_return(
        &mut self,
        f: FrameId,
        remote: bool,
        Ghost(share): Ghost<TransferShare>,
        Ghost(before): Ghost<FrameTable>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            before.consistent(v),
            before.versions == old(self).versions,
            before.frame(f).has_pin(share, remote),
            f.0 < before.metas.len(),
            old(self).metas == before.metas,
            old(self).bindings == before.bindings,
            old(self).num_tiers == before.num_tiers,
            old(self).ready_queue == before.ready_queue,
            old(self).draining_queue == before.draining_queue,
            old(self).draining_cursor == before.draining_cursor,
            old(self).frame_size == before.frame_size,
            old(self).ghosts@.dom() == before.ghosts@.dom(),
            forall|i: nat|
                i < before.metas.len() && i != f.0 ==> #[trigger] old(self).ghosts@[i]
                    == before.ghosts@[i],
            old(self).frame(f) == (FrameGhost {
                token: old(self).frame(f).token,
                ..before.frame(f)
            }),
            old(self).frame(f).token is Some,
            old(self).frame(f).slot_tok().well_formed(),
            old(self).frame(f).slot_tok().instance_id() == v.id(),
            old(self).frame(f).slot_tok().key() == before.frame(f).slot.loc,
            old(self).frame(f).frontier().last_writer == before.frame(f).frontier().last_writer,
            old(self).frame(f).slot_tok().value() == (if before.frame(f).live_shares().remove(
                share,
            ).is_empty() {
                Slot::Owned(old(self).frame(f).frontier())
            } else {
                Slot::Lent {
                    base: old(self).frame(f).frontier(),
                    shares: before.frame(f).live_shares().remove(share),
                }
            }),
            old(self).frame(f).covered(before.meta(f), &v.timelines, v.node, &v.ledger),
        ensures
            forall|g: FrameId|
                g != f && g.0 < before.metas.len() ==> (#[trigger] final(self).meta(g))
                    == before.meta(g),
            final(self).meta(f).inline_read_tickets == before.meta(f).inline_read_tickets,
            final(self).meta(f).flags == before.meta(f).flags,
            final(self).same_directory(&before),
            final(self).preserves_initializing(&before, None),
            final(self).same_other_pins(&before, Some(f)),
            final(self).metas.len() == before.metas.len(),
            f.0 < before.metas.len() ==> final(self).frame(f).slot == before.frame(f).slot,
            final(self).consistent(v),
            final(self).frame(f).local_shares == (if remote {
                before.frame(f).local_shares
            } else {
                before.frame(f).local_shares.remove(share)
            }),
            final(self).frame(f).remote_shares == (if remote {
                before.frame(f).remote_shares.remove(share)
            } else {
                before.frame(f).remote_shares
            }),
            final(self).frame(f).token == old(self).frame(f).token,
    {
        let idx = f.0 as usize;
        assert(before.valid_frame(f, v));
        if remote {
            self.metas[idx].active_remote_leases -= 1;
            self.metas[idx].mw_slot = u32::MAX;
            self.metas[idx].mw_gen = 0;
        } else {
            self.metas[idx].active_local_reads -= 1;
        }
        proof {
            let tracked g = self.ghosts.borrow_mut().tracked_borrow_mut(idx as nat);
            if remote {
                g.remote_shares = g.remote_shares.remove(share);
            } else {
                g.local_shares = g.local_shares.remove(share);
            }
            assert(g.live_shares() =~= before.frame(f).live_shares().remove(share));
            assert(g.live_shares().is_empty() == (g.local_shares.len() == 0 && g.remote_shares.len()
                == 0)) by {
                if g.live_shares().is_empty() {
                    assert forall|s| !#[trigger] g.local_shares.dom().contains(s) by {
                        assert(!g.live_shares().contains(s));
                    }
                    assert(g.local_shares.dom() =~= Set::empty());
                    assert forall|s| !#[trigger] g.remote_shares.dom().contains(s) by {
                        assert(!g.live_shares().contains(s));
                    }
                    assert(g.remote_shares.dom() =~= Set::empty());
                } else if g.local_shares.len() == 0 && g.remote_shares.len() == 0 {
                    assert(g.live_shares() =~= ISet::empty());
                }
            }
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
            assert(before.valid_frame(f, v));
            self.preserve_coverage(*old(self), f, v);
            assert(self.frame(f).frontier().last_writer == before.frame(f).frontier().last_writer);
            self.refresh_lease(before, f, v);
        }
    }

    pub(super) fn unpin(
        &mut self,
        table: &mut TimelineTable,
        f: FrameId,
        remote: bool,
        Tracked(cap): Tracked<ShareCap>,
        k: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Tracked<ReturnedShare>)
        requires
            old(self).consistent(v),
            instance.id() == v.id(),
            v.timelines == *old(table),
            f.0 < old(self).metas.len(),
            old(self).frame(f).has_pin(cap.share(), remote),
            old(self).returned_reads(f, cap, k, v),
            old(table).valid_ticket(k),
        ensures
            forall|g: FrameId|
                g != f && g.0 < old(self).metas.len() ==> (#[trigger] final(self).meta(g)) == old(
                    self,
                ).meta(g),
            k != Ticket::NONE ==> final(self).meta(f).bounds_reader(
                final(table),
                k.timeline_id(),
                k.sequence(),
            ),
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), Some(f)),
            final(self).metas.len() == old(self).metas.len(),
            f.0 < old(self).metas.len() ==> final(self).frame(f).slot == old(self).frame(f).slot,
            final(self).consistent(KernelView::of(final(table), &v.ledger, v.node)),
            final(self).frame(f).local_shares == (if remote {
                old(self).frame(f).local_shares
            } else {
                old(self).frame(f).local_shares.remove(cap.share())
            }),
            final(self).frame(f).remote_shares == (if remote {
                old(self).frame(f).remote_shares.remove(cap.share())
            } else {
                old(self).frame(f).remote_shares
            }),
            final(table).only_watermarks(old(table)),
            r@.instance_id() == v.id(),
            r@.share() == cap.share(),
            r@.loc() == old(self).frame(f).slot.loc,
    {
        let ghost share = cap.share();
        proof {
            assert(self.valid_frame(f, v));
        }
        if !k.is_none() {
            self.record_read_ticket(table, f, k, Ghost(v));
        } else {
            assert(Ticket::NONE.sequence() == 0) by (compute);
        }
        let ghost view = KernelView::of(table, &v.ledger, v.node);
        let ghost before = *self;
        let ghost m = self.meta(f);
        proof {
            assert(self.valid_frame(f, view));
        }
        let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
        let tracked lent_tok;
        proof {
            lent_tok = g.token.tracked_take();
            assert(before.frame(f).live_shares().contains(share));
        }
        let tracked (Tracked(parent), Tracked(receipt)) = instance.give_back_slot(lent_tok, cap);
        proof {
            g.token = Some(parent);
            let ids = before.frame(f).live_shares();
            assert(before.frame(f).slot_tok().value() == (Slot::Lent {
                base: before.frame(f).frontier(),
                shares: ids,
            }));
            assert(g.frontier() == (Frontier {
                last_writer: before.frame(f).frontier().last_writer,
                readers: before.frame(f).frontier().readers.union(cap.frontier().readers),
            }));
            assert(g.slot_tok().value() == if ids.remove(share).is_empty() {
                Slot::Owned(g.frontier())
            } else {
                Slot::Lent { base: g.frontier(), shares: ids.remove(share) }
            });
            assert(before.valid_frame(f, view));
            assert forall|e| #[trigger] g.frontier().readers.contains(e) implies m.accounts(
                table,
                v.node,
                &v.ledger,
                e,
            ) by {
                if !before.frame(f).frontier().readers.contains(e) && !v.ledger.citations(
                    v.node,
                ).contains(e) {
                    assert(cap.frontier().readers.contains(e));
                    let s = choose|s: u64|
                        1 <= s <= k.sequence() && #[trigger] v.timelines.events@[(
                            k.timeline_id(),
                            s,
                        )] == e;
                    let t = k.timeline_id();
                    if !m.overflow() {
                        let i = choose|i: int|
                            0 <= i < 4 && {
                                let p = #[trigger] m.inline_read_tickets[i];
                                p != Ticket::NONE && p.timeline_id() == t && k.sequence()
                                    <= p.sequence()
                            };
                        assert(s <= m.inline_read_tickets[i].sequence());
                    }
                    assert(m.bounds_reader(table, t, s));
                    assert(table.events@[(t, s)] == e);
                }
            }
            assert(g.covered(m, table, v.node, &v.ledger));
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
        }
        self.finish_return(f, remote, Ghost(share), Ghost(before), Ghost(view));
        Tracked(receipt)
    }

    /// Select a fresh name from the infinite supply against the finite set of live pins.
    /// The selection and its termination proof are erased.
    proof fn allocate_share(
        tracked names: &mut ShareSource,
        live: Set<TransferShare>,
    ) -> (tracked token: UnusedShare)
        requires
            old(names).well_formed(),
        ensures
            final(names).well_formed(),
            final(names).id() == old(names).id(),
            final(names).node() == old(names).node(),
            token.instance_id() == old(names).id(),
            !live.contains(token.element()),
            old(names).names().contains(token.element().seq),
            !live.contains(old(names).next_share()) ==> token.element() == old(names).next_share()
                && final(names).names() == old(names).names().tail(),
        decreases live.len(),
    {
        let head = names.next_share();
        let tracked first = names.allocate();
        if live.contains(head) {
            let tail = names.names();
            let tracked other = Self::allocate_share(names, live.remove(head));
            let i = choose|i: nat| #[trigger] (tail.name)(i) == other.element().seq;
            assert((old(names).names().name)(i + 1) == other.element().seq);
            other
        } else {
            assert((old(names).names().name)(0) == first.element().seq);
            first
        }
    }
}

} // verus!
