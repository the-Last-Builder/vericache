use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

/// Backend-agnostic handle for an in-flight fenced revocation of a remote lease.
/// The caller extracts the single-key `SlotToken` (`Lent`), performs its hardware
/// fence (e.g., RDMA MW invalidation or QP error transition) to obtain the updated
/// `SlotToken` and `RevokedToken`, records the fence event in the ledger, and
/// finishes via `finish_fenced_revocation`.
pub struct PendingRevocation {
    pub frame: FrameId,
    pub slot: u32,
    pub generation: u32,
    pub share: Ghost<TransferShare>,
    pub slot_ref: Ghost<SlotRef>,
    pub token: Tracked<Option<SlotToken>>,
    pub snapshot: Ghost<FrameTable>,
}

impl PendingRevocation {
    pub fn take_token(&mut self) -> (tok: Tracked<SlotToken>)
        requires
            old(self).token@ is Some,
        ensures
            Some(tok@) == old(self).token@,
            final(self).token@ is None,
            final(self).frame == old(self).frame,
            final(self).slot == old(self).slot,
            final(self).generation == old(self).generation,
            final(self).share == old(self).share,
            final(self).slot_ref == old(self).slot_ref,
            final(self).snapshot == old(self).snapshot,
    {
        let tracked tok;
        proof {
            tok = self.token.borrow_mut().tracked_take();
        }
        Tracked(tok)
    }
}

impl FrameTable {
    pub fn record_read_ticket(
        &mut self,
        table: &TimelineTable,
        f: FrameId,
        k: Ticket,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).consistent(v),
            v.timelines == *table,
            f.0 < old(self).metas.len(),
            old(self).meta(f).state >= STATE_PUBLISHED,
            table.valid_ticket(k),
            k != Ticket::NONE,
        ensures
            forall|g: FrameId|
                g != f && g.0 < old(self).metas.len() ==> (#[trigger] final(self).meta(g)) == old(
                    self,
                ).meta(g) && final(self).ovf(g) == old(self).ovf(g),
            final(self).preserves_initializing(old(self), Some(f)),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), Some(f)),
            final(self).consistent(v),
            final(self).metas.len() == old(self).metas.len(),
            final(self).ghosts == old(self).ghosts,
            final(self).meta(f).bounds_reader(final(self).ovf(f), k.timeline_id(), k.sequence()),
            final(self).meta(f).last_write_ticket == old(self).meta(f).last_write_ticket,
            final(self).meta(f).active_remote_leases == old(self).meta(f).active_remote_leases,
            final(self).meta(f).active_local_reads == old(self).meta(f).active_local_reads,
            final(self).meta(f).state == old(self).meta(f).state,
            final(self).meta(f).mw_slot == old(self).meta(f).mw_slot,
            final(self).meta(f).mw_gen == old(self).meta(f).mw_gen,
    {
        assert(self.valid_frame(f, v));
        let mut ovf = self.overflow_seqs[f.0 as usize];
        self.metas[f.0 as usize].note_read(&mut ovf, table, k);
        self.overflow_seqs.set(f.0 as usize, ovf);
        let ghost id = v.id();
        let ghost ledger = v.ledger;
        let ghost node = v.node;
        assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts@[i].valid(
            self.metas[i as int],
            self.overflow_seqs[i as int],
            self.num_tiers,
            self.tier_frame_size(self.ghosts@[i].tier),
            id,
            node,
            table,
            &ledger,
        ) by {
            let g = self.ghosts@[i];
            let m = self.metas[i as int];
            let cov = self.overflow_seqs[i as int];
            let prior = old(self).metas[i as int];
            let prior_cov = old(self).overflow_seqs[i as int];
            assert(g.valid(
                prior,
                prior_cov,
                old(self).num_tiers,
                old(self).tier_frame_size(g.tier),
                id,
                node,
                table,
                &ledger,
            ));
            if i != f.0 {
                assert forall|t: u8, s: u64|
                    table.issued(t, s) && prior.bounds_reader(
                        prior_cov,
                        t,
                        s,
                    ) implies #[trigger] m.bounds_reader(cov, t, s) by {}
            }
            g.preserve_readers(g, m, cov, prior, prior_cov, table, table, node, &ledger);
        }
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            table,
        ) by {
            assert(old(self).tier_queues(u, table));
            assert forall|i: int| 0 <= i < self.ready_queue[u].len() implies self.meta(
                #[trigger] self.ready_queue[u][i],
            ).reusable(self.ovf(self.ready_queue[u][i]), table) by {
                let p = self.ready_queue[u][i];
                assert(p != f);
                assert(old(self).meta(p).reusable(old(self).ovf(p), table));
            }
        }
    }

    pub fn pin_local_share(
        &mut self,
        f: FrameId,
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
            f.0 < old(self).metas.len() ==> final(self).frame(f).slot == old(self).frame(f).slot,
            final(self).consistent(v),
            r.is_ok() == (f.0 < old(self).metas.len() && old(self).meta(f).state == STATE_PUBLISHED
                && old(self).meta(f).active_local_reads < u32::MAX),
            r.is_ok() ==> final(self).meta(f).state == STATE_PUBLISHED,
            r.is_err() ==> *final(self) == *old(self) && *final(names) == *old(names),
            final(names).well_formed(),
            final(names).id() == v.id(),
            final(names).node() == v.node,
            r.is_ok() && !old(self).frame(f).live_shares().contains(old(names).next_share())
                ==> final(names).names() == old(names).names().tail(),
            r matches Ok(cap) ==> {
                &&& (!old(self).frame(f).live_shares().contains(old(names).next_share())
                    ==> cap@.share() == old(names).next_share())
                &&& cap@.is_local()
                &&& cap@.instance_id() == v.id()
                &&& cap@.loc() == old(self).frame(f).slot.loc
                &&& cap@.frontier() == (Frontier {
                    last_writer: old(self).frame(f).frontier().last_writer,
                    readers: ISet::empty(),
                })
                &&& !old(self).frame(f).live_shares().contains(cap@.share())
                &&& final(self).frame(f).remote_shares == old(self).frame(f).remote_shares
                &&& final(self).frame(f).local_shares == old(self).frame(f).local_shares.insert(
                    cap@.share(),
                    LocalShareEntry { share: cap@.share() },
                )
            },
    {
        let Tracked(cap) = self.pin_share(f, None, Tracked(instance), Tracked(names), Ghost(v))?;
        let tracked local = cap.into_local();
        Ok(Tracked(local))
    }

    pub fn pin_remote_lease(
        &mut self,
        f: FrameId,
        slot: u32,
        generation: u32,
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
            f.0 < old(self).metas.len() ==> final(self).frame(f).slot == old(self).frame(f).slot,
            final(self).consistent(v),
            r.is_ok() == (f.0 < old(self).metas.len() && old(self).meta(f).state == STATE_PUBLISHED
                && old(self).meta(f).active_remote_leases < u16::MAX && slot != u32::MAX),
            r.is_ok() ==> final(self).meta(f).state == STATE_PUBLISHED,
            r.is_err() ==> *final(self) == *old(self) && *final(names) == *old(names),
            r.is_ok() && old(self).meta(f).active_remote_leases == 0 ==> final(self).meta(f).mw_slot
                == slot && final(self).meta(f).mw_gen == generation,
            final(names).well_formed(),
            final(names).id() == v.id(),
            final(names).node() == v.node,
            r.is_ok() && !old(self).frame(f).live_shares().contains(old(names).next_share())
                ==> final(names).names() == old(names).names().tail(),
            r matches Ok(cap) ==> {
                &&& (!old(self).frame(f).live_shares().contains(old(names).next_share())
                    ==> cap@.share() == old(names).next_share())
                &&& cap@.is_unbound()
                &&& cap@.instance_id() == v.id()
                &&& cap@.loc() == old(self).frame(f).slot.loc
                &&& cap@.frontier() == (Frontier {
                    last_writer: old(self).frame(f).frontier().last_writer,
                    readers: ISet::empty(),
                })
                &&& !old(self).frame(f).live_shares().contains(cap@.share())
                &&& final(self).frame(f).local_shares == old(self).frame(f).local_shares
                &&& final(self).frame(f).remote_shares == old(self).frame(f).remote_shares.insert(
                    cap@.share(),
                    RemoteShareEntry { mw_slot: slot, mw_gen: generation },
                )
            },
    {
        self.pin_share(f, Some((slot, generation)), Tracked(instance), Tracked(names), Ghost(v))
    }

    pub fn unpin_local_share(
        &mut self,
        table: &TimelineTable,
        f: FrameId,
        Tracked(cap): Tracked<ShareCap>,
        k: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).consistent(v),
            v.timelines == *table,
            instance.id() == v.id(),
            f.0 < old(self).metas.len(),
            old(self).frame(f).local_shares.contains_key(cap.share()),
            old(self).returned_reads(f, cap, k, v),
            table.valid_ticket(k),
        ensures
            forall|g: FrameId|
                g != f && g.0 < old(self).metas.len() ==> (#[trigger] final(self).meta(g)) == old(
                    self,
                ).meta(g) && final(self).ovf(g) == old(self).ovf(g),
            k != Ticket::NONE ==> final(self).meta(f).bounds_reader(
                final(self).ovf(f),
                k.timeline_id(),
                k.sequence(),
            ),
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), Some(f)),
            final(self).metas.len() == old(self).metas.len(),
            f.0 < old(self).metas.len() ==> final(self).frame(f).slot == old(self).frame(f).slot,
            final(self).consistent(v),
            final(self).frame(f).local_shares == old(self).frame(f).local_shares.remove(
                cap.share(),
            ),
            final(self).frame(f).remote_shares == old(self).frame(f).remote_shares,
    {
        let _ = self.unpin(table, f, false, Tracked(cap), k, Tracked(instance), Ghost(v));
        proof {
            assert forall|g: FrameId|
                #![auto]
                g != f && g.0 < old(self).metas.len() implies self.meta(g) == old(self).meta(g)
                && self.ovf(g) == old(self).ovf(g) by {
                assert(self.meta(g) == old(self).meta(g));
                assert(self.ovf(g) == old(self).ovf(g));
            };
        }
    }

    pub fn unpin_remote_lease(
        &mut self,
        table: &TimelineTable,
        f: FrameId,
        slot: u32,
        generation: u32,
        Tracked(cap): Tracked<ShareCap>,
        k: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<Tracked<ReturnedShare>, (CoreError, Tracked<ShareCap>)>)
        requires
            old(self).consistent(v),
            v.timelines == *table,
            instance.id() == v.id(),
            f.0 < old(self).metas.len() && old(self).meta(f).active_remote_leases > 0 ==> old(
                self,
            ).frame(f).matches_lease(cap.share(), slot, generation),
            f.0 < old(self).metas.len() && old(self).meta(f).active_remote_leases > 0 ==> old(
                self,
            ).returned_reads(f, cap, k, v) && table.valid_ticket(k),
        ensures
            final(self).preserves_initializing(old(self), None),
            final(self).same_directory(old(self)),
            final(self).same_other_pins(old(self), Some(f)),
            final(self).metas.len() == old(self).metas.len(),
            f.0 < old(self).metas.len() ==> final(self).frame(f).slot == old(self).frame(f).slot,
            final(self).consistent(v),
            f.0 >= old(self).metas.len() || old(self).meta(f).active_remote_leases == 0
                ==> r matches Err((CoreError::StaleReplica, _)),
            f.0 < old(self).metas.len() && old(self).meta(f).active_remote_leases > 0 ==> r.is_ok(),
            match r {
                Ok(receipt) => final(self).frame(f).local_shares == old(self).frame(f).local_shares
                    && final(self).frame(f).remote_shares == old(self).frame(
                    f,
                ).remote_shares.remove(cap.share()) && receipt@.share() == cap.share()
                    && receipt@.instance_id() == v.id() && receipt@.loc() == old(self).frame(
                    f,
                ).slot.loc,
                Err((_, back)) => back@ == cap && *final(self) == *old(self),
            },
    {
        proof {
            if f.0 < self.metas.len() && self.meta(f).active_remote_leases > 0 {
                assert(self.valid_frame(f, v));
                if self.meta(f).mw_slot != u32::MAX {
                    let s = choose|s|
                        self.frame(f).remote_shares.contains_key(s) && #[trigger] self.frame(
                            f,
                        ).remote_shares[s].mw_slot == self.meta(f).mw_slot && self.frame(
                            f,
                        ).remote_shares[s].mw_gen == self.meta(f).mw_gen;
                    if s != cap.share() {
                        assert(self.frame(f).remote_shares.remove(s).contains_key(cap.share()));
                        assert(self.frame(f).remote_shares.len() >= 2);
                    }
                }
            }
        }
        if !self.accepts_lease_return(f, slot, generation) {
            return Err((CoreError::StaleReplica, Tracked(cap)));
        }
        Ok(self.unpin(table, f, true, Tracked(cap), k, Tracked(instance), Ghost(v)))
    }

    /// Begin a backend-agnostic fenced revocation by validating the lease and
    /// extracting the frame's lent `SlotToken` into a `PendingRevocation`.
    pub fn begin_fenced_revocation(
        &mut self,
        f: FrameId,
        slot: u32,
        generation: u32,
        Ghost(share): Ghost<TransferShare>,
        Ghost(v): Ghost<KernelView>,
    ) -> (r: Result<PendingRevocation, CoreError>)
        requires
            old(self).consistent(v),
            f.0 < old(self).metas.len() && old(self).meta(f).active_remote_leases > 0 ==> old(
                self,
            ).frame(f).matches_lease(share, slot, generation),
        ensures
            f.0 >= old(self).metas.len() || old(self).meta(f).active_remote_leases == 0
                ==> r matches Err(CoreError::StaleReplica),
            f.0 < old(self).metas.len() && old(self).meta(f).active_remote_leases > 0 ==> r.is_ok(),
            r.is_err() ==> *final(self) == *old(self),
            r matches Ok(pending) ==> {
                &&& f.0 < old(self).metas.len()
                &&& old(self).meta(f).active_remote_leases > 0
                &&& old(self).valid_frame(f, v)
                &&& pending.frame == f
                &&& pending.slot == slot
                &&& pending.generation == generation
                &&& pending.share@ == share
                &&& pending.slot_ref@ == old(self).frame(f).slot
                &&& pending.token@ == old(self).frame(f).token
                &&& pending.token@ is Some
                &&& slot_is_lent_with(pending.token@->Some_0, v.id(), pending.slot_ref@.loc, share)
                &&& pending.snapshot@ == *old(self)
                &&& final(self).replaced_token(*old(self), f)
                &&& final(self).frame(f).token is None
            },
    {
        proof {
            if f.0 < self.metas.len() && self.meta(f).active_remote_leases > 0 {
                assert(self.valid_frame(f, v));
                if self.meta(f).mw_slot != u32::MAX {
                    let s = choose|s|
                        self.frame(f).remote_shares.contains_key(s) && #[trigger] self.frame(
                            f,
                        ).remote_shares[s].mw_slot == self.meta(f).mw_slot && self.frame(
                            f,
                        ).remote_shares[s].mw_gen == self.meta(f).mw_gen;
                    if s != share {
                        assert(self.frame(f).remote_shares.remove(s).contains_key(share));
                        assert(self.frame(f).remote_shares.len() >= 2);
                    }
                }
            }
        }
        if !self.accepts_lease_return(f, slot, generation) {
            return Err(CoreError::StaleReplica);
        }
        let ghost before = *self;
        let ghost slot_ref = self.frame(f).slot;
        proof {
            assert(self.valid_frame(f, v));
        }
        let tracked tok;
        proof {
            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
            tok = g.token.tracked_take();
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
            assert(before.frame(f).live_shares().contains(share));
            assert(slot_is_lent_with(tok, v.id(), slot_ref.loc, share));
        }
        Ok(
            PendingRevocation {
                frame: f,
                slot,
                generation,
                share: Ghost(share),
                slot_ref: Ghost(slot_ref),
                token: Tracked(Some(tok)),
                snapshot: Ghost(before),
            },
        )
    }

    /// Roll back an unexecuted fenced revocation and restore `FrameTable::consistent`.
    pub fn cancel_fenced_revocation(
        &mut self,
        _pending: PendingRevocation,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            _pending.snapshot@.consistent(v),
            _pending.frame.0 < _pending.snapshot@.metas.len(),
            old(self).replaced_token(_pending.snapshot@, _pending.frame),
            _pending.token@ == _pending.snapshot@.frame(_pending.frame).token,
        ensures
            final(self).consistent(v),
            *final(self) == _pending.snapshot@,
    {
        let ghost f = _pending.frame;
        let tracked token = _pending.token.get();
        proof {
            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
            g.token = token;
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@ =~= _pending.snapshot@.ghosts@);
        }
    }

    /// Complete a fenced revocation after the backend has fenced the remote lease
    /// (producing `revoked_slot` and `revoked_token`) and recorded `fence_event` in `ledger`.
    pub fn finish_fenced_revocation(
        &mut self,
        pending: PendingRevocation,
        Tracked(revoked_slot): Tracked<SlotToken>,
        Tracked(revoked_token): Tracked<&RevokedToken>,
        Ghost(fence_event): Ghost<TransferEvent>,
        Tracked(ledger): Tracked<&KnowledgeLedger>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            pending.snapshot@.consistent(v),
            pending.frame.0 < pending.snapshot@.metas.len(),
            pending.snapshot@.meta(pending.frame).active_remote_leases > 0,
            pending.snapshot@.frame(pending.frame).matches_lease(
                pending.share@,
                pending.slot,
                pending.generation,
            ),
            old(self).replaced_token(pending.snapshot@, pending.frame),
            ledger.valid(v.id(), v.node),
            v.ledger.submitted.map().submap_of(ledger.submitted.map()),
            v.ledger.knows.set().subset_of(ledger.knows.set()),
            v.ledger.succeeded.dom().subset_of(ledger.succeeded.dom()),
            ledger.citations(v.node).contains(fence_event),
            revoked_token.instance_id() == v.id(),
            revoked_token.key() == (
                pending.snapshot@.frame(pending.frame).slot.loc,
                pending.share@,
            ),
            revoked_token.value() == fence_event,
            revoked_slot.well_formed(),
            revoked_slot.instance_id() == v.id(),
            revoked_slot.key() == pending.snapshot@.frame(pending.frame).slot.loc,
            revoked_slot.value() == fold_share(
                pending.snapshot@.frame(pending.frame).slot_tok().value(),
                pending.share@,
                Frontier { last_writer: None, readers: ISet::empty().insert(fence_event) },
            ),
        ensures
            final(self).metas.len() == pending.snapshot@.metas.len(),
            final(self).frame(pending.frame).slot == pending.snapshot@.frame(pending.frame).slot,
            final(self).consistent(KernelView::of(&v.timelines, ledger, v.node)),
            final(self).frame(pending.frame).local_shares == pending.snapshot@.frame(
                pending.frame,
            ).local_shares,
            final(self).frame(pending.frame).remote_shares == pending.snapshot@.frame(
                pending.frame,
            ).remote_shares.remove(pending.share@),
            final(self).same_directory(&pending.snapshot@),
            final(self).preserves_initializing(&pending.snapshot@, None),
            final(self).same_other_pins(&pending.snapshot@, Some(pending.frame)),
    {
        let f = pending.frame;
        let ghost before = pending.snapshot@;
        let ghost share = pending.share@;
        let ghost view = KernelView::of(&v.timelines, ledger, v.node);
        proof {
            let prior = before.frame(f);
            let m = before.meta(f);
            assert forall|t: u8, s: u64|
                t < v.timelines.count && 1 <= s
                    <= v.timelines.confirmed_seq[t as int] implies #[trigger] v.timelines.observed(
                t,
                s,
                v.node,
                ledger.knows,
            ) by {
                assert(v.timelines.observed(t, s, v.node, v.ledger.knows));
            }
            assert forall|t: u8, s: u64|
                v.timelines.issued(t, s) implies #[trigger] v.timelines.events@.contains_key((t, s))
                && ledger.submitted.dom().contains(v.timelines.events@[(t, s)]) by {
                assert(v.timelines.events@.contains_key((t, s)));
                assert(v.ledger.submitted.dom().contains(v.timelines.events@[(t, s)]));
            }
            assert(view.timelines.well_formed(v.id(), v.node, ledger));
            assert(v.ledger.citations(v.node).subset_of(ledger.citations(v.node)));
            assert(view.extends(v));
            before.lemma_timeline_progress(v, view);
            assert(before.valid_frame(f, view));

            let tracked mut g = self.ghosts.borrow_mut().tracked_remove(f.0 as nat);
            g.token = Some(revoked_slot);
            let ids = prior.live_shares();
            assert(prior.slot_tok().value() == (Slot::Lent {
                base: prior.frontier(),
                shares: ids,
            }));
            assert(prior.frontier().readers.union(ISet::empty().insert(fence_event))
                =~= prior.frontier().readers.insert(fence_event));
            assert(g.frontier() == (Frontier {
                last_writer: prior.frontier().last_writer,
                readers: prior.frontier().readers.insert(fence_event),
            }));
            assert(g.slot_tok().value() == if ids.remove(share).is_empty() {
                Slot::Owned(g.frontier())
            } else {
                Slot::Lent { base: g.frontier(), shares: ids.remove(share) }
            });
            assert forall|e| #[trigger] g.frontier().readers.contains(e) implies m.accounts(
                self.ovf(f),
                &v.timelines,
                v.node,
                ledger,
                e,
            ) by {
                if e != fence_event {
                    assert(prior.frontier().readers.contains(e));
                }
            }
            assert(g.covered(m, self.ovf(f), &v.timelines, v.node, ledger));
            self.ghosts.borrow_mut().tracked_insert(f.0 as nat, g);
            assert(self.ghosts@.dom() =~= before.ghosts@.dom());
        }
        self.finish_return(f, true, Ghost(share), Ghost(before), Ghost(view));
    }
}

} // verus!
