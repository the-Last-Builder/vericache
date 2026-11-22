#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    pub proof fn lemma_service_well_formed(frames: &FrameTable, catalog: &Catalog, v: KernelView)
        requires
            frames.consistent(v),
            catalog.well_formed(frames),
            ServiceSpec::valid_versions(frames.frame_size as nat, frames.versions@),
        ensures
            Self::service(frames, catalog).well_formed(),
    {
        let s = Self::service(frames, catalog);
        assert forall|k: ObjectKey| #[trigger]
            s.catalog.contains_key(k) implies s.versions.contains_key(s.catalog[k]) by {
            assert(catalog.view().contains_key(k));
            let t = choose|t: int|
                0 <= t < catalog.num_tiers() && #[trigger] catalog.slot(k, t).is_some();
            let f = catalog.slot(k, t).unwrap();
            assert(frames.valid_frame(f, v));
            assert(frames.meta(f).state == STATE_PUBLISHED);
            assert(frames.frame(f).cert.is_some());
            assert(frames.frame(f).version_agrees(frames.versions@));
        }
        assert(ServiceSpec::directory_valid(s.catalog, s.versions));
    }

    pub proof fn lemma_same_directory_preserves_state(
        catalog: &Catalog,
        before: &FrameTable,
        after: &FrameTable,
        v_before: KernelView,
        v_after: KernelView,
    )
        requires
            before.consistent(v_before),
            after.consistent(v_after),
            after.same_directory(before),
            catalog.well_formed(before),
            Self::certified(before, v_before),
            v_after.ledger == v_before.ledger,
            v_after.node == v_before.node,
        ensures
            catalog.well_formed(after),
            Self::certified(after, v_after),
            Self::service(after, catalog) == Self::service(before, catalog),
    {
        assert forall|k: ObjectKey, t: int| 0 <= t < catalog.num_tiers() implies (
        #[trigger] catalog.slot(k, t) matches Some(f) ==> {
            &&& f.0 < after.metas.len()
            &&& after.meta(f).state == STATE_PUBLISHED
            &&& after.frame(f).tier.index() == t
            &&& after.frame(f).bound_key == Some(k)
            &&& after.frame(f).bound_version == Some(catalog.view()[k].version)
        }) by {
            if let Some(f) = catalog.slot(k, t) {
                assert(before.meta(f).state == STATE_PUBLISHED);
                assert(before.published(f) == after.published(f));
                assert(after.meta(f).state == STATE_PUBLISHED);
                assert(after.valid_frame(f, v_after));
            }
        }
        assert forall|f: FrameId|
            f.0 < after.metas.len() && (#[trigger] after.meta(f)).state == STATE_PUBLISHED implies {
            let g = after.frame(f);
            &&& g.bound_key.is_some()
            &&& g.tier.valid(catalog.num_tiers())
            &&& catalog.slot(g.bound_key.unwrap(), g.tier.index() as int) == Some(f)
        } by {
            assert(before.published(f) == after.published(f));
            assert(before.meta(f).state == STATE_PUBLISHED);
            assert(before.valid_frame(f, v_before));
            assert(after.valid_frame(f, v_after));
        }
        assert forall|f: FrameId|
            f.0 < after.metas.len() && (#[trigger] after.meta(f)).state == STATE_PUBLISHED implies {
            let g = after.frame(f);
            g.cert matches Some(cert) && {
                &&& replica_valid(v_after.ledger, v_after.node, cert)
                &&& event_succeeded(v_after.ledger, v_after.node, cert.writer)
                &&& v_after.ledger.citations(v_after.node).contains(cert.writer)
                &&& g.bound_version matches Some(ver)
                &&& after.versions@.contains_key(ver)
                &&& after.versions@[ver] == cert.bytes
            }
        } by {
            assert(before.published(f) == after.published(f));
            assert(before.meta(f).state == STATE_PUBLISHED);
            assert(before.valid_frame(f, v_before));
            assert(after.valid_frame(f, v_after));
        }
    }

    pub proof fn lemma_non_published_write_disjoint(&self, target: FrameId, writes: ISet<SlotLoc>)
        requires
            self.frames.separated(),
            target.0 < self.frames.metas.len(),
            self.frames.meta(target).state != STATE_PUBLISHED,
            writes.subset_of(ISet::empty().insert(self.frames.frame(target).slot.loc)),
        ensures
            self.disjoint_from_published(writes),
    {
        assert forall|f: FrameId|
            f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                == STATE_PUBLISHED implies !writes.contains(self.frames.frame(f).slot.loc) by {
            assert(f.0 != target.0);
            if f.0 < target.0 {
                assert(self.frames.ghosts@[f.0 as nat].slot.loc
                    != self.frames.ghosts@[target.0 as nat].slot.loc);
            } else {
                assert(self.frames.ghosts@[target.0 as nat].slot.loc
                    != self.frames.ghosts@[f.0 as nat].slot.loc);
            }
        }
    }

    /// Any frontier admitted by `self.citations()` is also admitted by the closure
    /// generated from `after.union(self.citations())`.
    pub proof fn lemma_citations_admit_closure(
        &self,
        fr: Frontier<TransferDomain>,
        after: ISet<TransferEvent>,
    )
        requires
            self.well_formed(),
        ensures
            read_admitted(fr, self.citations()) ==> read_admitted(
                fr,
                foundation::tracked::closure_from(
                    self.ledger@.submitted.map(),
                    after.union(self.citations()),
                ),
            ),
            write_admitted(fr, self.citations()) ==> write_admitted(
                fr,
                foundation::tracked::closure_from(
                    self.ledger@.submitted.map(),
                    after.union(self.citations()),
                ),
            ),
            self.citations().subset_of(self.ledger@.submitted.dom()),
            foundation::tracked::knowledge_of(self.node(), self.citations()).subset_of(
                self.ledger@.knows.set(),
            ),
    {
        let ordered = foundation::tracked::closure_from(
            self.ledger@.submitted.map(),
            after.union(self.citations()),
        );
        assert(self.citations().subset_of(ordered));
    }

    pub proof fn preserve_after_ledger_growth(&self, before: Self)
        requires
            before.well_formed(),
            self.node_id == before.node_id,
            self.tiers == before.tiers,
            self.frames == before.frames,
            self.catalog == before.catalog,
            self.version_seq_start == before.version_seq_start,
            self.next_version_seq == before.next_version_seq,
            self.version_seq_limit == before.version_seq_limit,
            self.active_leases == before.active_leases,
            self.free_lease_slots == before.free_lease_slots,
            self.lease_generation == before.lease_generation,
            self.spec == before.spec,
            self.version_registry == before.version_registry,
            self.kernel_view().extends(before.kernel_view()),
            before.ledger@.submitted.map().submap_of(self.ledger@.submitted.map()),
            revoked_submap(before.ledger@.revoked, self.ledger@.revoked),
            before.ledger@.knows.set().subset_of(self.ledger@.knows.set()),
            before.ledger@.succeeded.dom().subset_of(self.ledger@.succeeded.dom()),
            complete_receipts(self.ledger@),
            forall|e: TransferEvent, f: FrameId|
                self.ledger@.submitted.dom().contains(e)
                    && !before.ledger@.submitted.dom().contains(e) && f.0 < self.frames.metas.len()
                    && (#[trigger] self.frames.meta(f)).state == STATE_PUBLISHED ==> !(
                #[trigger] self.ledger@.submitted.map()[e]).access.writes.contains(
                    self.frames.frame(f).slot.loc,
                ),
        ensures
            self.well_formed(),
    {
        self.frames.lemma_timeline_progress(before.kernel_view(), self.kernel_view());
        assert forall|f: FrameId|
            f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                == STATE_PUBLISHED implies {
            let g = self.frames.frame(f);
            g.cert matches Some(cert) && {
                &&& replica_valid(self.ledger@, self.node(), cert)
                &&& event_succeeded(self.ledger@, self.node(), cert.writer)
                &&& self.ledger@.citations(self.node()).contains(cert.writer)
                &&& g.bound_version matches Some(ver)
                &&& self.frames.versions@.contains_key(ver)
                &&& self.frames.versions@[ver] == cert.bytes
            }
        } by {
            assert(before.frames.valid_frame(f, before.kernel_view()));
            let cert = self.frames.frame(f).cert.unwrap();
            assert(cert.slot == self.frames.frame(f).slot);
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.submitted.dom().contains(e)
                    && !before.ledger@.submitted.dom().contains(
                    e,
                ) implies !self.ledger@.submitted.map()[e].access.writes.contains(
                cert.slot.loc,
            ) by {
                assert(self.frames.meta(f).state == STATE_PUBLISHED);
            }
            replica_survives_disjoint_writes(before.ledger@, self.ledger@, self.node(), cert);
        }
    }

    /// Issue a FIFO ticket on `timeline` for a disjoint submitted event while
    /// preserving `CacheCore::well_formed()`.
    pub fn issue_ticket(&mut self, timeline: u8, Tracked(receipt): Tracked<Submitted>) -> (r:
        Result<Ticket, CoreError>)
        requires
            old(self).well_formed(),
            receipt.instance_id() == old(self).instance_id(),
            receipt.key().observer == old(self).node(),
            receipt.value().preds.subset_of(old(self).ledger@.submitted.dom()),
            old(self).disjoint_from_published(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            match r {
                Err(_) => *final(self) == *old(self) && (timeline >= old(self).timelines.count
                    || old(self).timelines.next_seq[timeline as int] > MAX_TICKET_SEQ),
                Ok(ticket) => {
                    let s = old(self).timelines.next_seq[timeline as int];
                    &&& timeline < old(self).timelines.count
                    &&& ticket.timeline_id() == timeline && ticket.sequence() == s
                    &&& ticket != Ticket::NONE
                    &&& final(self).timelines.count == old(self).timelines.count
                    &&& final(self).timelines.kinds == old(self).timelines.kinds
                    &&& final(self).timelines.issued(timeline, s)
                    &&& final(self).timelines.events@ == old(self).timelines.events@.insert(
                        (timeline, s),
                        receipt.key(),
                    )
                    &&& final(self).timelines.events@[(timeline, s)] == receipt.key()
                    &&& final(self).timelines.next_seq@ == old(self).timelines.next_seq@.update(
                        timeline as int,
                        (s + 1) as u64,
                    )
                    &&& final(self).timelines.confirmed_seq == old(self).timelines.confirmed_seq
                    &&& final(self).kernel_view().extends(old(self).kernel_view())
                    &&& final(self).ledger@.submitted.map() == old(
                        self,
                    ).ledger@.submitted.map().insert(receipt.key(), receipt.value())
                    &&& old(self).ledger@.submitted.map().submap_of(
                        final(self).ledger@.submitted.map(),
                    )
                    &&& final(self).ledger@.knows == old(self).ledger@.knows
                    &&& final(self).ledger@.succeeded == old(self).ledger@.succeeded
                    &&& final(self).ledger@.revoked == old(self).ledger@.revoked
                },
            },
    {
        let ghost before = *self;
        let ghost event = receipt.key();
        let ticket = self.timelines.issue_ticket(
            timeline,
            Ghost(event),
            Tracked(receipt),
            Tracked(self.ledger.borrow_mut()),
        )?;
        proof {
            assert(before.ledger@.citations(self.node()).subset_of(
                self.ledger@.citations(self.node()),
            ));
            assert(self.kernel_view().extends(before.kernel_view()));
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.submitted.dom().contains(
                    e,
                ) implies self.ledger@.submitted.map()[e].preds.subset_of(
                self.ledger@.submitted.dom(),
            ) by {
                if before.ledger@.submitted.dom().contains(e) {
                    assert(before.ledger@.submitted.map()[e] == self.ledger@.submitted.map()[e]);
                    assert(before.ledger@.submitted.map()[e].preds.subset_of(
                        before.ledger@.submitted.dom(),
                    ));
                } else {
                    assert(e == event);
                }
            }
            self.preserve_after_ledger_growth(before);
        }
        Ok(ticket)
    }

    /// Advance a timeline's confirmed sequence number with a completion
    /// observation `Knows` while preserving `CacheCore::well_formed()`.
    pub fn mark_timeline_complete(
        &mut self,
        timeline: u8,
        new_confirmed: u64,
        Tracked(knows): Tracked<Knows>,
    )
        requires
            old(self).well_formed(),
            knows.instance_id() == old(self).instance_id(),
            timeline < old(self).timelines.count,
            old(self).timelines.confirmed_seq[timeline as int] <= new_confirmed < old(
                self,
            ).timelines.next_seq[timeline as int],
            forall|s: u64|
                old(self).timelines.confirmed_seq[timeline as int] < s <= new_confirmed
                    ==> #[trigger] old(self).timelines.observed(
                    timeline,
                    s,
                    old(self).node(),
                    knows,
                ) || old(self).timelines.observed(
                    timeline,
                    s,
                    old(self).node(),
                    old(self).ledger@.knows,
                ),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).timelines.count == old(self).timelines.count,
            final(self).timelines.kinds == old(self).timelines.kinds,
            final(self).timelines.confirmed_seq@ == old(self).timelines.confirmed_seq@.update(
                timeline as int,
                new_confirmed,
            ),
            final(self).timelines.next_seq == old(self).timelines.next_seq,
            final(self).timelines.events == old(self).timelines.events,
            final(self).ledger@.submitted == old(self).ledger@.submitted,
            final(self).ledger@.succeeded == old(self).ledger@.succeeded,
            final(self).ledger@.revoked == old(self).ledger@.revoked,
            final(self).ledger@.knows.set() == old(self).ledger@.knows.set().union(knows.set()),
    {
        let ghost before = *self;
        let ghost node = self.node();
        self.timelines.advance_confirmed(
            timeline,
            new_confirmed,
            Tracked(knows),
            Tracked(self.ledger.borrow_mut()),
            Ghost(node),
        );
        proof {
            assert(before.ledger@.citations(self.node()).subset_of(
                self.ledger@.citations(self.node()),
            ));
            assert(self.kernel_view().extends(before.kernel_view()));
            self.preserve_after_ledger_growth(before);
        }
    }

    pub fn record_submitted(&mut self, Tracked(receipt): Tracked<Submitted>)
        requires
            old(self).well_formed(),
            receipt.instance_id() == old(self).instance_id(),
            receipt.value().preds.subset_of(old(self).ledger@.submitted.dom()),
            old(self).disjoint_from_published(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger@.submitted.map() == old(self).ledger@.submitted.map().insert(
                receipt.key(),
                receipt.value(),
            ),
            old(self).ledger@.submitted.map().submap_of(final(self).ledger@.submitted.map()),
            final(self).ledger@.knows == old(self).ledger@.knows,
            final(self).ledger@.succeeded == old(self).ledger@.succeeded,
            final(self).ledger@.revoked == old(self).ledger@.revoked,
    {
        self.record_disjoint_event(Tracked(receipt), Tracked(None), Tracked(None));
    }

    pub fn record_known(&mut self, Tracked(observation): Tracked<Knows>)
        requires
            old(self).well_formed(),
            observation.instance_id() == old(self).instance_id(),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger@.submitted == old(self).ledger@.submitted,
            final(self).ledger@.succeeded == old(self).ledger@.succeeded,
            final(self).ledger@.revoked == old(self).ledger@.revoked,
            final(self).ledger@.knows.set() == old(self).ledger@.knows.set().union(
                observation.set(),
            ),
    {
        let ghost before = *self;
        proof {
            self.ledger.borrow_mut().record_knows(observation);
            assert forall|t: u8, s: u64|
                t < self.timelines.count && 1 <= s
                    <= self.timelines.confirmed_seq[t as int] implies #[trigger] self.timelines.observed(
            t, s, self.node(), self.ledger@.knows) by {
                assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
                assert(before.ledger@.knows.set().contains(
                    (before.node(), before.timelines.events@[(t, s)]),
                ));
            }
            assert(before.ledger@.citations(self.node()).subset_of(
                self.ledger@.citations(self.node()),
            ));
            assert(self.kernel_view().extends(before.kernel_view()));
            self.preserve_after_ledger_growth(before);
        }
    }

    pub fn record_succeeded(&mut self, Tracked(witness): Tracked<Succeeded>)
        requires
            old(self).well_formed(),
            witness.instance_id() == old(self).instance_id(),
            old(self).ledger@.submitted.dom().contains(witness.event()),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger@.submitted == old(self).ledger@.submitted,
            final(self).ledger@.knows == old(self).ledger@.knows,
            final(self).ledger@.revoked == old(self).ledger@.revoked,
            final(self).ledger@.succeeded.witnesses == old(self).ledger@.succeeded.witnesses.insert(
                witness.event(),
                witness,
            ),
    {
        let ghost before = *self;
        proof {
            self.ledger.borrow_mut().record_succeeded(witness);
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.succeeded.contains(e) implies self.ledger@.submitted.dom().contains(
                e,
            ) by {
                if !before.ledger@.succeeded.contains(e) {
                    assert(e == witness.event());
                }
            }
            assert(self.ledger@.valid(self.instance_id(), self.node()));
            assert(self.kernel_view().extends(before.kernel_view()));
            self.preserve_after_ledger_growth(before);
        }
    }

    pub fn record_disjoint_event(
        &mut self,
        Tracked(receipt): Tracked<Submitted>,
        Tracked(observation): Tracked<Option<Knows>>,
        Tracked(witness): Tracked<Option<Succeeded>>,
    )
        requires
            old(self).well_formed(),
            receipt.instance_id() == old(self).instance_id(),
            receipt.value().preds.subset_of(old(self).ledger@.submitted.dom()),
            observation matches Some(k) ==> k.instance_id() == old(self).instance_id(),
            witness matches Some(w) ==> w.instance_id() == old(self).instance_id() && w.event()
                == receipt.key(),
            old(self).disjoint_from_published(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger@.submitted.map() == old(self).ledger@.submitted.map().insert(
                receipt.key(),
                receipt.value(),
            ),
            old(self).ledger@.submitted.map().submap_of(final(self).ledger@.submitted.map()),
            observation is None ==> final(self).ledger@.knows == old(self).ledger@.knows,
            witness is None ==> final(self).ledger@.succeeded == old(self).ledger@.succeeded,
            final(self).ledger@.revoked == old(self).ledger@.revoked,
            final(self).ledger@.knows.set() == match observation {
                Some(k) => old(self).ledger@.knows.set().union(k.set()),
                None => old(self).ledger@.knows.set(),
            },
            final(self).ledger@.succeeded.witnesses == match witness {
                Some(w) => old(self).ledger@.succeeded.witnesses.insert(w.event(), w),
                None => old(self).ledger@.succeeded.witnesses,
            },
    {
        let ghost before = *self;
        proof {
            self.ledger.borrow_mut().record_submitted(receipt);
            if let Some(k) = observation {
                self.ledger.borrow_mut().record_knows(k);
            }
            if let Some(w) = witness {
                self.ledger.borrow_mut().record_succeeded(w);
            }
            assert forall|t: u8, s: u64|
                self.timelines.issued(t, s) implies #[trigger] self.timelines.events@.contains_key(
                (t, s),
            ) && self.ledger@.submitted.dom().contains(self.timelines.events@[(t, s)]) by {
                assert(before.timelines.events@.contains_key((t, s)));
                assert(before.ledger@.submitted.dom().contains(before.timelines.events@[(t, s)]));
            }
            assert forall|t: u8, s: u64|
                t < self.timelines.count && 1 <= s
                    <= self.timelines.confirmed_seq[t as int] implies #[trigger] self.timelines.observed(
            t, s, self.node(), self.ledger@.knows) by {
                assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
                assert(before.ledger@.knows.set().contains(
                    (before.node(), before.timelines.events@[(t, s)]),
                ));
            }
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.succeeded.contains(e) implies self.ledger@.submitted.dom().contains(
                e,
            ) by {
                if !before.ledger@.succeeded.contains(e) {
                    assert(e == receipt.key());
                }
            }
            assert(self.ledger@.valid(self.instance_id(), self.node()));
            assert(self.timelines.well_formed(self.instance_id(), self.node(), &self.ledger@));
            assert(before.ledger@.citations(self.node()).subset_of(
                self.ledger@.citations(self.node()),
            ));
            assert(self.kernel_view().extends(before.kernel_view()));
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.submitted.dom().contains(
                    e,
                ) implies self.ledger@.submitted.map()[e].preds.subset_of(
                self.ledger@.submitted.dom(),
            ) by {
                if before.ledger@.submitted.dom().contains(e) {
                    assert(before.ledger@.submitted.map()[e] == self.ledger@.submitted.map()[e]);
                    assert(before.ledger@.submitted.map()[e].preds.subset_of(
                        before.ledger@.submitted.dom(),
                    ));
                } else {
                    assert(e == receipt.key());
                }
            }
            self.preserve_after_ledger_growth(before);
        }
    }

    pub fn record_revoked_map(&mut self, Tracked(revoked): Tracked<RevokedMap>)
        requires
            old(self).well_formed(),
            revoked.instance_id() == old(self).instance_id(),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger@.submitted == old(self).ledger@.submitted,
            final(self).ledger@.knows == old(self).ledger@.knows,
            final(self).ledger@.succeeded == old(self).ledger@.succeeded,
            final(self).ledger@.revoked.map() == old(self).ledger@.revoked.map().union_prefer_right(
                revoked.map(),
            ),
            old(self).ledger@.revoked.map().submap_of(final(self).ledger@.revoked.map()),
            revoked_submap(old(self).ledger@.revoked, final(self).ledger@.revoked),
            revoked.map().submap_of(final(self).ledger@.revoked.map()),
            revoked_submap(revoked, final(self).ledger@.revoked),
    {
        let ghost before = *self;
        proof {
            self.ledger.borrow_mut().record_revoked_map(revoked);
            assert(self.ledger@.valid(self.instance_id(), self.node()));
            assert(self.timelines.well_formed(self.instance_id(), self.node(), &self.ledger@));
            assert(self.kernel_view().extends(before.kernel_view()));
            self.preserve_after_ledger_growth(before);
        }
    }

    proof fn prove_published_frames_certified(
        tracked ghosts: &mut Map<nat, FrameGhost>,
        metas: Seq<FrameSyncMeta>,
        n: nat,
        before_ledger: KnowledgeLedger,
        tracked after_ledger: &KnowledgeLedger,
        node: NodeId,
        tracked instance: &TransferInstance,
    )
        requires
            n <= metas.len(),
            forall|i: nat| i < metas.len() ==> #[trigger] old(ghosts).contains_key(i),
            before_ledger.valid(instance.id(), node),
            after_ledger.valid(instance.id(), node),
            before_ledger.submitted.map().submap_of(after_ledger.submitted.map()),
            revoked_submap(before_ledger.revoked, after_ledger.revoked),
            before_ledger.knows.set().subset_of(after_ledger.knows.set()),
            before_ledger.succeeded.dom().subset_of(after_ledger.succeeded.dom()),
            forall|i: nat|
                i < n && (#[trigger] metas[i as int]).state == STATE_PUBLISHED ==> {
                    let g = old(ghosts)[i];
                    &&& g.cert matches Some(cert)
                    &&& g.token matches Some(tok)
                    &&& tok.well_formed() && tok.instance_id() == instance.id() && tok.key()
                        == cert.slot.loc
                    &&& foundation::tracked::slot_base_frontier(tok.value()).last_writer == Some(
                        cert.writer,
                    )
                    &&& replica_valid(before_ledger, node, cert)
                    &&& before_ledger.citations(node).contains(cert.writer)
                },
        ensures
            *final(ghosts) == *old(ghosts),
            forall|i: nat|
                i < n && (#[trigger] metas[i as int]).state == STATE_PUBLISHED ==> {
                    let g = old(ghosts)[i];
                    replica_valid(*after_ledger, node, g.cert.unwrap())
                },
        decreases n,
    {
        if n == 0 {
            return;
        }
        Self::prove_published_frames_certified(
            ghosts,
            metas,
            (n - 1) as nat,
            before_ledger,
            after_ledger,
            node,
            instance,
        );
        let idx = (n - 1) as nat;
        if metas[idx as int].state == STATE_PUBLISHED {
            let tracked mut fg = ghosts.tracked_remove(idx);
            let ghost cert = fg.cert.unwrap();
            let ghost loc = cert.slot.loc;
            let ghost w = cert.writer;
            let tracked mut tok_opt: Option<SlotToken> = None;
            vstd::modes::tracked_swap(&mut fg.token, &mut tok_opt);
            let tracked mut tok = tok_opt.tracked_unwrap();
            if exists|e: TransferEvent|
                after_ledger.submitted.dom().contains(e) && (
                #[trigger] after_ledger.submitted.map()[e]).access.writes.contains(loc) && e != w
                    && !after_ledger.submitted.map()[w].preds.contains(e) {
                let e = choose|e: TransferEvent|
                    after_ledger.submitted.dom().contains(e) && (
                    #[trigger] after_ledger.submitted.map()[e]).access.writes.contains(loc) && e
                        != w && !after_ledger.submitted.map()[w].preds.contains(e);
                instance.slot_writer_is_latest(w, e, &mut tok, &after_ledger.submitted);
                assert(false);
            }
            fg.token = Some(tok);
            ghosts.tracked_insert(idx, fg);
            assert(*ghosts =~= *old(ghosts));
            assert(before_ledger.citations(node).subset_of(after_ledger.citations(node)));
            assert(event_succeeded(*after_ledger, node, w));
            assert forall|h, run|
                foundation::history::reachable(h) && foundation::value::valid_run(h, run)
                    && ledger_consistent_with_history(
                    *after_ledger,
                    node,
                    h,
                    run,
                ) implies #[trigger] certificate_valid_in(cert, h, run)
                && recorded_writer_certified(cert, *after_ledger, node, h) by {
                ledger_history_restrict(before_ledger, *after_ledger, node, h, run);
                assert(certificate_valid_in(cert, h, run));
                assert(recorded_writer_certified(cert, before_ledger, node, h));
                assert forall|e: TransferEvent|
                    after_ledger.submitted.dom().contains(e) && (#[trigger] (h.access)(
                        e,
                    )).writes.contains(loc) && e != w implies (h.happens_before)(e, w) by {
                    assert(after_ledger.submitted.map()[e].access == (h.access)(e));
                    assert(after_ledger.submitted.map()[w].preds.contains(e));
                }
            }
        }
    }

    /// Merge a peer's `Statement` components (`submitted`, `knows`, `succeeded`)
    /// into `self.ledger`, proving all published local frames retain valid
    /// certificates via `TransferInstance::slot_writer_is_latest`.
    pub fn merge_peer_statement(
        &mut self,
        Tracked(submitted): Tracked<SubmittedMap>,
        Tracked(knows): Tracked<Knows>,
        Tracked(succeeded): Tracked<SucceededMap>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            submitted.instance_id() == old(self).instance_id(),
            knows.instance_id() == old(self).instance_id(),
            succeeded.instance_id == old(self).instance_id(),
            succeeded.well_formed(),
            complete_receipts_map(submitted),
            forall|e| #[trigger] succeeded.contains(e) ==> submitted.dom().contains(e),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).kernel_view().extends(old(self).kernel_view()),
            old(self).ledger@.submitted.map().submap_of(final(self).ledger@.submitted.map()),
            submitted.map().submap_of(final(self).ledger@.submitted.map()),
            old(self).ledger@.knows.set().subset_of(final(self).ledger@.knows.set()),
            knows.set().subset_of(final(self).ledger@.knows.set()),
            old(self).ledger@.succeeded.dom().subset_of(final(self).ledger@.succeeded.dom()),
            succeeded.dom().subset_of(final(self).ledger@.succeeded.dom()),
            final(self).ledger@.revoked == old(self).ledger@.revoked,
    {
        let ghost before = *self;
        proof {
            self.ledger.borrow_mut().record_submitted_map(submitted);
            self.ledger.borrow_mut().record_knows(knows);
            self.ledger.borrow_mut().record_succeeded_map(succeeded);
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.succeeded.contains(e) implies self.ledger@.submitted.dom().contains(
                e,
            ) by {
                if !before.ledger@.succeeded.contains(e) {
                    assert(succeeded.contains(e));
                    assert(submitted.dom().contains(e));
                }
            }
            assert(self.ledger@.valid(self.instance_id(), self.node()));
            self.timelines.lemma_well_formed_ledger_growth(
                self.instance_id(),
                self.node(),
                &before.ledger@,
                &self.ledger@,
            );
            assert(before.ledger@.citations(self.node()).subset_of(
                self.ledger@.citations(self.node()),
            ));
            assert(self.kernel_view().extends(before.kernel_view()));
            assert forall|e: TransferEvent| #[trigger]
                self.ledger@.submitted.dom().contains(
                    e,
                ) implies self.ledger@.submitted.map()[e].preds.subset_of(
                self.ledger@.submitted.dom(),
            ) by {
                if before.ledger@.submitted.dom().contains(e) {
                    assert(before.ledger@.submitted.map()[e] == self.ledger@.submitted.map()[e]);
                    assert(before.ledger@.submitted.map()[e].preds.subset_of(
                        before.ledger@.submitted.dom(),
                    ));
                } else {
                    assert(submitted.dom().contains(e));
                    assert(submitted.map()[e] == self.ledger@.submitted.map()[e]);
                    assert(submitted.map()[e].preds.subset_of(submitted.dom()));
                }
            }
            self.frames.lemma_timeline_progress(before.kernel_view(), self.kernel_view());
            assert forall|i: nat|
                i < self.frames.metas.len() && (#[trigger] self.frames.metas[i as int]).state
                    == STATE_PUBLISHED implies {
                let g = self.frames.ghosts@[i];
                &&& g.cert matches Some(cert)
                &&& g.token matches Some(tok)
                &&& tok.well_formed() && tok.instance_id() == instance.id() && tok.key()
                    == cert.slot.loc
                &&& foundation::tracked::slot_base_frontier(tok.value()).last_writer == Some(
                    cert.writer,
                )
                &&& replica_valid(before.ledger@, self.node(), cert)
                &&& before.ledger@.citations(self.node()).contains(cert.writer)
            } by {
                let f = FrameId(i as u32);
                assert(before.frames.valid_frame(f, before.kernel_view()));
                assert(before.frames.meta(f).state == STATE_PUBLISHED);
            }
            let ghost node = self.node();
            Self::prove_published_frames_certified(
                self.frames.ghosts.borrow_mut(),
                self.frames.metas@,
                self.frames.metas.len() as nat,
                before.ledger@,
                self.ledger.borrow(),
                node,
                instance,
            );
            assert(self.frames == before.frames);
            assert forall|f: FrameId|
                f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                    == STATE_PUBLISHED implies {
                let g = self.frames.frame(f);
                g.cert matches Some(cert) && {
                    &&& replica_valid(self.ledger@, self.node(), cert)
                    &&& event_succeeded(self.ledger@, self.node(), cert.writer)
                    &&& self.ledger@.citations(self.node()).contains(cert.writer)
                    &&& g.bound_version matches Some(ver)
                    &&& self.frames.versions@.contains_key(ver)
                    &&& self.frames.versions@[ver] == cert.bytes
                }
            } by {
                assert(self.frames.metas[f.0 as int].state == STATE_PUBLISHED);
                assert(before.frames.meta(f).state == STATE_PUBLISHED);
            }
        }
    }
}

} // verus!
