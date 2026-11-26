#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    pub(crate) proof fn lemma_service_well_formed(
        frames: &FrameTable,
        catalog: &Catalog,
        v: KernelView,
    )
        requires
            frames.consistent(v),
            catalog.well_formed(frames),
            ServiceSpec::valid_versions(frames.frame_size as nat, frames.versions_spec()@),
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
            assert(frames.frame(f).version_agrees(frames.versions_spec()@));
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
                &&& after.versions_spec()@.contains_key(ver)
                &&& after.versions_spec()@[ver] == cert.bytes
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
            self.frames_spec().separated(),
            target.0 < self.frames_spec().metas_spec().len(),
            self.frames_spec().meta(target).state != STATE_PUBLISHED,
            writes.subset_of(ISet::empty().insert(self.frames_spec().frame(target).slot.loc)),
        ensures
            self.disjoint_from_published(writes),
    {
        assert forall|f: FrameId|
            f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                == STATE_PUBLISHED implies !writes.contains(self.frames.frame(f).slot.loc) by {
            assert(f.0 != target.0);
            if f.0 < target.0 {
                assert(self.frames.ghosts_spec()@[f.0 as nat].slot.loc
                    != self.frames.ghosts_spec()@[target.0 as nat].slot.loc);
            } else {
                assert(self.frames.ghosts_spec()@[target.0 as nat].slot.loc
                    != self.frames.ghosts_spec()@[f.0 as nat].slot.loc);
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
                    self.ledger_spec()@.submitted.map(),
                    after.union(self.citations()),
                ),
            ),
            write_admitted(fr, self.citations()) ==> write_admitted(
                fr,
                foundation::tracked::closure_from(
                    self.ledger_spec()@.submitted.map(),
                    after.union(self.citations()),
                ),
            ),
            self.citations().subset_of(self.ledger_spec()@.submitted.dom()),
            foundation::tracked::knowledge_of(self.node(), self.citations()).subset_of(
                self.ledger_spec()@.knows.set(),
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
            self.node_id_spec() == before.node_id_spec(),
            self.tiers_spec() == before.tiers_spec(),
            self.frames_spec() == before.frames_spec(),
            self.catalog_spec() == before.catalog_spec(),
            self.version_seq_start_spec() == before.version_seq_start_spec(),
            self.next_version_seq_spec() == before.next_version_seq_spec(),
            self.version_seq_limit_spec() == before.version_seq_limit_spec(),
            self.active_leases_spec() == before.active_leases_spec(),
            self.free_lease_slots_spec() == before.free_lease_slots_spec(),
            self.lease_generation_spec() == before.lease_generation_spec(),
            self.spec_spec() == before.spec_spec(),
            self.version_registry_spec() == before.version_registry_spec(),
            self.kernel_view().extends(before.kernel_view()),
            before.ledger_spec()@.submitted.map().submap_of(self.ledger_spec()@.submitted.map()),
            revoked_submap(before.ledger_spec()@.revoked, self.ledger_spec()@.revoked),
            before.ledger_spec()@.knows.set().subset_of(self.ledger_spec()@.knows.set()),
            before.ledger_spec()@.succeeded.dom().subset_of(self.ledger_spec()@.succeeded.dom()),
            complete_receipts(self.ledger_spec()@),
            forall|e: TransferEvent, f: FrameId|
                self.ledger_spec()@.submitted.dom().contains(e)
                    && !before.ledger_spec()@.submitted.dom().contains(e) && f.0
                    < self.frames_spec().metas_spec().len() && (#[trigger] self.frames_spec().meta(
                    f,
                )).state == STATE_PUBLISHED ==> !(
                #[trigger] self.ledger_spec()@.submitted.map()[e]).access.writes.contains(
                    self.frames_spec().frame(f).slot.loc,
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
                &&& self.frames.versions_spec()@.contains_key(ver)
                &&& self.frames.versions_spec()@[ver] == cert.bytes
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
    pub fn issue_ticket(
        &mut self,
        timeline: TimelineId,
        Tracked(receipt): Tracked<Submitted>,
    ) -> (r: Result<Ticket, CoreError>)
        requires
            old(self).well_formed(),
            receipt.instance_id() == old(self).instance_id(),
            receipt.key().observer == old(self).node(),
            receipt.value().preds.subset_of(old(self).ledger_spec()@.submitted.dom()),
            old(self).disjoint_from_published(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            match r {
                Err(_) => *final(self) == *old(self) && (timeline >= old(
                    self,
                ).timelines_spec().count_spec() || old(
                    self,
                ).timelines_spec().next_seq_spec()[timeline as int] > MAX_TICKET_SEQ),
                Ok(ticket) => {
                    let s = old(self).timelines_spec().next_seq_spec()[timeline as int];
                    &&& timeline < old(self).timelines_spec().count_spec()
                    &&& ticket.timeline_id() == timeline && ticket.sequence() == s
                    &&& ticket != Ticket::NONE
                    &&& final(self).timelines_spec().count_spec() == old(
                        self,
                    ).timelines_spec().count_spec()
                    &&& final(self).timelines_spec().kinds_spec() == old(
                        self,
                    ).timelines_spec().kinds_spec()
                    &&& final(self).timelines_spec().issued(timeline, s)
                    &&& final(self).timelines_spec().events_spec()@ == old(
                        self,
                    ).timelines_spec().events_spec()@.insert((timeline, s), receipt.key())
                    &&& final(self).timelines_spec().events_spec()@[(timeline, s)] == receipt.key()
                    &&& final(self).timelines_spec().next_seq_spec()@ == old(
                        self,
                    ).timelines_spec().next_seq_spec()@.update(timeline as int, (s + 1) as u64)
                    &&& final(self).timelines_spec().confirmed_seq_spec() == old(
                        self,
                    ).timelines_spec().confirmed_seq_spec()
                    &&& final(self).kernel_view().extends(old(self).kernel_view())
                    &&& final(self).ledger_spec()@.submitted.map() == old(
                        self,
                    ).ledger_spec()@.submitted.map().insert(receipt.key(), receipt.value())
                    &&& old(self).ledger_spec()@.submitted.map().submap_of(
                        final(self).ledger_spec()@.submitted.map(),
                    )
                    &&& final(self).ledger_spec()@.knows == old(self).ledger_spec()@.knows
                    &&& final(self).ledger_spec()@.succeeded == old(self).ledger_spec()@.succeeded
                    &&& final(self).ledger_spec()@.revoked == old(self).ledger_spec()@.revoked
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
        timeline: TimelineId,
        new_confirmed: u64,
        Tracked(knows): Tracked<Knows>,
    )
        requires
            old(self).well_formed(),
            knows.instance_id() == old(self).instance_id(),
            timeline < old(self).timelines_spec().count_spec(),
            old(self).timelines_spec().confirmed_seq_spec()[timeline as int] <= new_confirmed < old(
                self,
            ).timelines_spec().next_seq_spec()[timeline as int],
            forall|s: u64|
                old(self).timelines_spec().confirmed_seq_spec()[timeline as int] < s
                    <= new_confirmed ==> #[trigger] old(self).timelines_spec().observed(
                    timeline,
                    s,
                    old(self).node(),
                    knows,
                ) || old(self).timelines_spec().observed(
                    timeline,
                    s,
                    old(self).node(),
                    old(self).ledger_spec()@.knows,
                ),
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).timelines_spec().count_spec() == old(self).timelines_spec().count_spec(),
            final(self).timelines_spec().kinds_spec() == old(self).timelines_spec().kinds_spec(),
            final(self).timelines_spec().confirmed_seq_spec()@ == old(
                self,
            ).timelines_spec().confirmed_seq_spec()@.update(timeline as int, new_confirmed),
            final(self).timelines_spec().next_seq_spec() == old(
                self,
            ).timelines_spec().next_seq_spec(),
            final(self).timelines_spec().events_spec() == old(self).timelines_spec().events_spec(),
            final(self).ledger_spec()@.submitted == old(self).ledger_spec()@.submitted,
            final(self).ledger_spec()@.succeeded == old(self).ledger_spec()@.succeeded,
            final(self).ledger_spec()@.revoked == old(self).ledger_spec()@.revoked,
            final(self).ledger_spec()@.knows.set() == old(self).ledger_spec()@.knows.set().union(
                knows.set(),
            ),
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

    pub(crate) fn record_submitted(&mut self, Tracked(receipt): Tracked<Submitted>)
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

    pub(crate) fn record_known(&mut self, Tracked(observation): Tracked<Knows>)
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
            assert forall|t: TimelineId, s: u64|
                t < self.timelines.count && 1 <= s
                    <= self.timelines.confirmed_seq[t as int] implies #[trigger] self.timelines.observed(
            t, s, self.node(), self.ledger@.knows) by {
                assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
                assert(before.ledger@.knows.set().contains(
                    (before.node(), before.timelines.events_spec()@[(t, s)]),
                ));
            }
            assert(before.ledger@.citations(self.node()).subset_of(
                self.ledger@.citations(self.node()),
            ));
            assert(self.kernel_view().extends(before.kernel_view()));
            self.preserve_after_ledger_growth(before);
        }
    }

    pub(crate) fn record_succeeded(&mut self, Tracked(witness): Tracked<Succeeded>)
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
            receipt.value().preds.subset_of(old(self).ledger_spec()@.submitted.dom()),
            observation matches Some(k) ==> k.instance_id() == old(self).instance_id(),
            witness matches Some(w) ==> w.instance_id() == old(self).instance_id() && w.event()
                == receipt.key(),
            old(self).disjoint_from_published(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger_spec()@.submitted.map() == old(
                self,
            ).ledger_spec()@.submitted.map().insert(receipt.key(), receipt.value()),
            old(self).ledger_spec()@.submitted.map().submap_of(
                final(self).ledger_spec()@.submitted.map(),
            ),
            observation is None ==> final(self).ledger_spec()@.knows == old(
                self,
            ).ledger_spec()@.knows,
            witness is None ==> final(self).ledger_spec()@.succeeded == old(
                self,
            ).ledger_spec()@.succeeded,
            final(self).ledger_spec()@.revoked == old(self).ledger_spec()@.revoked,
            final(self).ledger_spec()@.knows.set() == match observation {
                Some(k) => old(self).ledger_spec()@.knows.set().union(k.set()),
                None => old(self).ledger_spec()@.knows.set(),
            },
            final(self).ledger_spec()@.succeeded.witnesses == match witness {
                Some(w) => old(self).ledger_spec()@.succeeded.witnesses.insert(w.event(), w),
                None => old(self).ledger_spec()@.succeeded.witnesses,
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
            assert forall|t: TimelineId, s: u64|
                self.timelines.issued(
                    t,
                    s,
                ) implies #[trigger] self.timelines.events_spec()@.contains_key((t, s))
                && self.ledger@.submitted.dom().contains(self.timelines.events_spec()@[(t, s)]) by {
                assert(before.timelines.events_spec()@.contains_key((t, s)));
                assert(before.ledger@.submitted.dom().contains(
                    before.timelines.events_spec()@[(t, s)],
                ));
            }
            assert forall|t: TimelineId, s: u64|
                t < self.timelines.count && 1 <= s
                    <= self.timelines.confirmed_seq[t as int] implies #[trigger] self.timelines.observed(
            t, s, self.node(), self.ledger@.knows) by {
                assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
                assert(before.ledger@.knows.set().contains(
                    (before.node(), before.timelines.events_spec()@[(t, s)]),
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
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).kernel_view().extends(old(self).kernel_view()),
            final(self).ledger_spec()@.submitted == old(self).ledger_spec()@.submitted,
            final(self).ledger_spec()@.knows == old(self).ledger_spec()@.knows,
            final(self).ledger_spec()@.succeeded == old(self).ledger_spec()@.succeeded,
            final(self).ledger_spec()@.revoked.map() == old(
                self,
            ).ledger_spec()@.revoked.map().union_prefer_right(revoked.map()),
            old(self).ledger_spec()@.revoked.map().submap_of(
                final(self).ledger_spec()@.revoked.map(),
            ),
            revoked_submap(old(self).ledger_spec()@.revoked, final(self).ledger_spec()@.revoked),
            revoked.map().submap_of(final(self).ledger_spec()@.revoked.map()),
            revoked_submap(revoked, final(self).ledger_spec()@.revoked),
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
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).kernel_view().extends(old(self).kernel_view()),
            old(self).ledger_spec()@.submitted.map().submap_of(
                final(self).ledger_spec()@.submitted.map(),
            ),
            submitted.map().submap_of(final(self).ledger_spec()@.submitted.map()),
            old(self).ledger_spec()@.knows.set().subset_of(final(self).ledger_spec()@.knows.set()),
            knows.set().subset_of(final(self).ledger_spec()@.knows.set()),
            old(self).ledger_spec()@.succeeded.dom().subset_of(
                final(self).ledger_spec()@.succeeded.dom(),
            ),
            succeeded.dom().subset_of(final(self).ledger_spec()@.succeeded.dom()),
            final(self).ledger_spec()@.revoked == old(self).ledger_spec()@.revoked,
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
                let g = self.frames.ghosts_spec()@[i];
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
                    &&& self.frames.versions_spec()@.contains_key(ver)
                    &&& self.frames.versions_spec()@[ver] == cert.bytes
                }
            } by {
                assert(self.frames.metas[f.0 as int].state == STATE_PUBLISHED);
                assert(before.frames.meta(f).state == STATE_PUBLISHED);
            }
        }
    }
}

} // verus!
