#[cfg(verus_only)]
use super::poll::{lemma_rdma_core_after_submit, lemma_rdma_remove_range_peel};
use super::*;
use crate::timeline::DataOperation;
#[cfg(verus_only)]
use crate::timeline::{window_matches, BITMAP_CAPACITY};
use transfer::rdma;

verus! {

impl<'a> RdmaBackend<'a> {
    /// Preflight the next data WR ID before posting an RDMA read/write.
    pub(crate) fn next_data_id(&self, core: &CacheCore) -> (r: BackendResult<u64>)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            r matches Ok(s) ==> s == core.timelines_spec().next_seq_spec()[self.timeline_id as int]
                && s <= MAX_TICKET_SEQ && s
                <= core.timelines_spec().confirmed_seq_spec()[self.timeline_id as int]
                + BITMAP_CAPACITY,
    {
        proof {
            lemma_core_wf_facts(core);
        }
        core.timelines().next_data_id(self.timeline_id)
    }

    /// Preflight a contiguous batch of `count` data WR IDs before calling
    /// `rdma::post_send_batch`.
    pub(crate) fn next_data_ids(&self, core: &CacheCore, count: usize) -> (r: BackendResult<u64>)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            r matches Ok(s) ==> s == core.timelines_spec().next_seq_spec()[self.timeline_id as int]
                && count > 0 && s + count - 1 <= MAX_TICKET_SEQ && s + count - 1
                <= core.timelines_spec().confirmed_seq_spec()[self.timeline_id as int]
                + BITMAP_CAPACITY,
    {
        proof {
            lemma_core_wf_facts(core);
        }
        core.timelines().next_data_ids(self.timeline_id, count)
    }

    /// Atomically record the accepted prefix returned by `rdma::post_send_batch`
    /// into `self.in_flight_ops`, `core.timelines_spec()`, and `core.ledger` (M7!).
    #[verifier::rlimit(80)]
    pub(crate) fn record_data_batch(
        &mut self,
        core: &mut CacheCore,
        frames: &[FrameId],
        operations: &[DataOperation],
        posted: Vec<rdma::PostedBatchItem>,
    ) -> (tickets: Vec<Ticket>)
        requires
            old(core).well_formed(),
            old(self).timeline_id < old(core).timelines_spec().count_spec(),
            old(core).timelines_spec().kinds_spec()[old(self).timeline_id as int] == (
            TimelineKind::RdmaQueuePair { peer_node: old(self).peer_node }),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.state() is ReadyToSend ==> old(self).qp.peer() == old(self).peer_node
                && old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().id() == old(core).instance_id(),
            old(self).qp.source().node() == old(core).node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == old(core).node(),
            old(self).cq.infos().instance_id() == old(core).instance_id(),
            old(self).window_pool.well_formed(),
            old(self).window_pool.node() == old(core).node(),
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                old(core),
                old(self).window_pool.pd(),
            ),
            rdma_leases_valid(
                old(self).active_leases@,
                &old(self).window_pool,
                &old(self).qp,
                old(core),
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                rdma::CqSnapshot::of(old(self).cq),
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &old(self).window_pool),
            completed_pulls_wf(
                old(self).completed_pulls,
                &old(core).timelines_spec(),
                old(self).timeline_id,
                old(self).cq.observer(),
                &old(core).ledger_spec()@,
            ),
            old(self).bitmap.well_formed(),
            old(self).bitmap.observations(
                &old(core).timelines_spec(),
                old(self).timeline_id,
                old(core).node(),
                &old(core).ledger_spec()@,
            ),
            CONTROL_WR_ID_BASE <= old(self).next_control_wr_id <= CONTROL_WR_ID_LIMIT,
            forall|s: u64|
                old(self).cq.in_flight().contains_key(s) && !((#[trigger] old(
                    self,
                ).cq.in_flight()[s]).kind is DataTransfer) ==> CONTROL_WR_ID_BASE <= s < old(
                    self,
                ).next_control_wr_id,
            posted.len() <= frames.len(),
            posted.len() <= operations.len(),
            ({
                let t = old(self).timeline_id;
                let base = old(core).timelines_spec().next_seq_spec()[t as int];
                let n = posted.len() as int;
                &&& n == 0 || (base + n - 1 <= MAX_TICKET_SEQ && base + n - 1 <= old(
                    core,
                ).timelines_spec().confirmed_seq_spec()[t as int] + BITMAP_CAPACITY)
                &&& forall|i: int|
                    0 <= i < n ==> {
                        let item = #[trigger] posted[i];
                        let s = (base + i) as u64;
                        &&& item.wr_id == s
                        &&& item.submitted@.instance_id() == old(core).instance_id()
                        &&& item.submitted@.key().observer == old(core).node()
                        &&& forall|e: TransferEvent| #[trigger]
                            item.submitted@.value().preds.contains(e) ==> old(
                                core,
                            ).ledger_spec()@.submitted.dom().contains(e) || exists|k: int|
                                0 <= k < i && (#[trigger] posted[k]).submitted@.key() == e
                        &&& old(core).disjoint_from_published(item.submitted@.value().access.writes)
                        &&& old(self).pending().contains_key(s)
                        &&& old(self).pending()[s] == item.submitted@.key()
                    }
                &&& window_matches(
                    old(self).in_flight_ops,
                    old(self).bitmap,
                    &old(core).timelines_spec(),
                    t,
                    rdma_remove_range(old(self).pending(), base, n),
                )
            }),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec() == old(core).frames_spec(),
            final(core).catalog_spec() == old(core).catalog_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
            final(core).version_seq_limit_spec() == old(core).version_seq_limit_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).timeline_id),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).completed_pulls == old(self).completed_pulls,
            final(core).ledger_spec()@.knows == old(core).ledger_spec()@.knows,
            final(core).ledger_spec()@.succeeded == old(core).ledger_spec()@.succeeded,
            final(core).ledger_spec()@.revoked == old(core).ledger_spec()@.revoked,
            old(core).ledger_spec()@.submitted.map().submap_of(
                final(core).ledger_spec()@.submitted.map(),
            ),
            posted.len() == 1 ==> final(core).ledger_spec()@.submitted.map() == old(
                core,
            ).ledger_spec()@.submitted.map().insert(
                posted[0].submitted@.key(),
                posted[0].submitted@.value(),
            ),
            tickets.len() == posted.len(),
            forall|i: int|
                0 <= i < tickets.len() ==> {
                    let t = old(self).timeline_id;
                    let s = (old(core).timelines_spec().next_seq_spec()[t as int] + i) as u64;
                    &&& (#[trigger] tickets[i]).timeline_id() == t
                    &&& tickets[i].sequence() == s
                    &&& tickets[i] != Ticket::NONE
                    &&& final(core).timelines_spec().issued(t, s)
                    &&& final(core).timelines_spec().events_spec()@.contains_key((t, s))
                    &&& final(core).timelines_spec().events_spec()@[(t, s)]
                        == posted[i].submitted@.key()
                },
            forall|i: int|
                0 <= i < posted.len() ==> {
                    let item = #[trigger] posted@[i];
                    let event = item.submitted@.key();
                    &&& final(core).ledger_spec()@.submitted.dom().contains(event)
                    &&& final(core).ledger_spec()@.submitted.map()[event] == item.submitted@.value()
                },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_core = *core;
        let ghost full = self.pending();
        let ghost items = posted@;
        let ghost base = core.timelines_spec().next_seq_spec()[self.timeline_id as int];
        let t = self.timeline_id;
        let n = posted.len();
        if n == 0 {
            proof {
                reveal(rdma_leases_valid);
                lemma_core_wf_facts(core);
                lemma_core_timelines_wf(core);
                reveal(rdma_window_wf);
                assert(rdma_remove_range(full, base, 0) =~= full) by {
                    broadcast use vstd::map::group_map_lemmas;

                }
                assert(rdma::CqSnapshot::of(self.cq).pending == self.cq.in_flight());
                assert(rdma_window_wf(
                    self.in_flight_ops,
                    self.bitmap,
                    &core.timelines_spec(),
                    t,
                    &core.ledger_spec()@,
                    rdma::CqSnapshot::of(self.cq),
                    core.instance_id(),
                    self.next_control_wr_id,
                ));
            }
            proof {
                lemma_peer_core_step_refl(core, t);
            }
            return Vec::new();
        }
        let mut tickets: Vec<Ticket> = Vec::with_capacity(n);
        let mut iter = posted.into_iter();
        proof {
            lemma_peer_core_step_refl(core, t);
        }
        while tickets.len() < n
            invariant
                peer_core_step(&before_core, core, t),
                tickets.len() <= n == items.len(),
                n <= frames.len(),
                n <= operations.len(),
                t == self.timeline_id == old(self).timeline_id,
                t < core.timelines_spec().count_spec(),
                base == before_core.timelines_spec().next_seq_spec()[t as int],
                full == self.pending(),
                base + n - 1 <= MAX_TICKET_SEQ,
                base + n - 1 <= core.timelines_spec().confirmed_seq_spec()[t as int]
                    + BITMAP_CAPACITY,
                vstd::std_specs::iter::IteratorSpec::remaining(&iter) == items.skip(
                    tickets.len() as int,
                ),
                core.well_formed(),
                before_core.well_formed(),
                core.node_id_spec() == before_core.node_id_spec(),
                core.instance_id() == before_core.instance_id(),
                core.tiers_spec() == before_core.tiers_spec(),
                core.frames_spec() == before_core.frames_spec(),
                core.catalog_spec() == before_core.catalog_spec(),
                core.spec_spec() == before_core.spec_spec(),
                core.next_version_seq_spec() == before_core.next_version_seq_spec(),
                core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
                core.active_leases_spec() == before_core.active_leases_spec(),
                core.version_registry_spec() == before_core.version_registry_spec(),
                core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == before_core.timelines_spec().kinds_spec(),
                core.timelines_spec().confirmed_seq_spec()
                    == before_core.timelines_spec().confirmed_seq_spec(),
                core.timelines_spec().next_seq_spec()@
                    == before_core.timelines_spec().next_seq_spec()@.update(
                    t as int,
                    (base + tickets.len()) as u64,
                ),
                forall|q: u64|
                    before_core.timelines_spec().issued(t, q)
                        ==> #[trigger] core.timelines_spec().events_spec()@[(t, q)]
                        == before_core.timelines_spec().events_spec()@[(t, q)],
                before_core.ledger_spec()@.submitted.map().submap_of(
                    core.ledger_spec()@.submitted.map(),
                ),
                core.ledger_spec()@.knows == before_core.ledger_spec()@.knows,
                core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
                core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
                tickets.len() == 0 ==> core.ledger_spec()@.submitted.map()
                    == before_core.ledger_spec()@.submitted.map(),
                tickets.len() == 1 ==> core.ledger_spec()@.submitted.map()
                    == before_core.ledger_spec()@.submitted.map().insert(
                    items[0].submitted@.key(),
                    items[0].submitted@.value(),
                ),
                self.bitmap.well_formed(),
                self.bitmap.observations(
                    &core.timelines_spec(),
                    t,
                    core.node(),
                    &core.ledger_spec()@,
                ),
                window_matches(
                    self.in_flight_ops,
                    self.bitmap,
                    &core.timelines_spec(),
                    t,
                    rdma_remove_range(full, (base + tickets.len()) as u64, n - tickets.len()),
                ),
                forall|j: int|
                    0 <= j < n ==> {
                        let item = #[trigger] items[j];
                        let s = (base + j) as u64;
                        &&& item.wr_id == s
                        &&& item.submitted@.instance_id() == core.instance_id()
                        &&& item.submitted@.key().observer == core.node()
                        &&& forall|e: TransferEvent| #[trigger]
                            item.submitted@.value().preds.contains(e)
                                ==> before_core.ledger_spec()@.submitted.dom().contains(e)
                                || exists|k: int|
                                0 <= k < j && (#[trigger] items[k]).submitted@.key() == e
                        &&& before_core.disjoint_from_published(
                            item.submitted@.value().access.writes,
                        )
                        &&& full.contains_key(s)
                        &&& full[s] == item.submitted@.key()
                    },
                forall|j: int|
                    0 <= j < tickets.len() ==> {
                        let s = (base + j) as u64;
                        &&& (#[trigger] tickets[j]).timeline_id() == t
                        &&& tickets[j].sequence() == s
                        &&& tickets[j] != Ticket::NONE
                        &&& core.timelines_spec().events_spec()@.contains_key((t, s))
                        &&& core.timelines_spec().events_spec()@[(t, s)]
                            == items[j].submitted@.key()
                        &&& core.ledger_spec()@.submitted.dom().contains(items[j].submitted@.key())
                        &&& core.ledger_spec()@.submitted.map()[items[j].submitted@.key()]
                            == items[j].submitted@.value()
                    },
            decreases n - tickets.len(),
        {
            let i = tickets.len();
            let item = iter.next().unwrap();
            let ghost step_core = *core;
            let ghost s = (base + i) as u64;
            let ghost next_pending = rdma_remove_range(full, (s + 1) as u64, n - i - 1);
            let ghost ev = item.submitted@.key();
            let ghost val = item.submitted@.value();
            proof {
                assert(item == items[i as int]);
                lemma_core_wf_facts(core);
                lemma_core_timelines_wf(core);
                lemma_rdma_remove_range_peel(full, base, n as int, i as int);
            }
            let ticket = core.record_submission(
                &mut self.in_flight_ops,
                Ghost(self.bitmap),
                t,
                frames[i],
                operations[i],
                Ghost(next_pending),
                item.submitted,
            );
            let ghost prior_tickets = tickets@;
            tickets.push(ticket);
            proof {
                assert(val.preds.subset_of(step_core.ledger_spec()@.submitted.dom())) by {
                    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

                    assert forall|e: TransferEvent| #[trigger]
                        val.preds.contains(
                            e,
                        ) implies step_core.ledger_spec()@.submitted.dom().contains(e) by {
                        if before_core.ledger_spec()@.submitted.dom().contains(e) {
                            assert(before_core.ledger_spec()@.submitted.map().submap_of(
                                step_core.ledger_spec()@.submitted.map(),
                            ));
                        } else {
                            let k = choose|k: int|
                                0 <= k < i as int && (#[trigger] items[k]).submitted@.key() == e;
                            assert(tickets[k].timeline_id() == t);
                            assert(step_core.ledger_spec()@.submitted.dom().contains(
                                items[k].submitted@.key(),
                            ));
                        }
                    }
                }
                lemma_rdma_core_after_submit(&step_core, core, t, ev, val);
                vstd::imap_lib::lemma_submap_of_trans(
                    before_core.ledger_spec()@.submitted.map(),
                    step_core.ledger_spec()@.submitted.map(),
                    core.ledger_spec()@.submitted.map(),
                );
                assert(core.timelines_spec().next_seq_spec()@
                    =~= before_core.timelines_spec().next_seq_spec()@.update(
                    t as int,
                    (s + 1) as u64,
                ));
                assert(peer_core_step(&before_core, core, t)) by {
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::same_layout);
                    broadcast use {
                        vstd::map::group_map_lemmas,
                        vstd::imap::group_imap_lemmas,
                        vstd::iset::group_iset_lemmas,
                    };

                    assert forall|u: TimelineId, q: u64|
                        u < before_core.timelines_spec().count_spec() && u != t && 1 <= q
                            < before_core.timelines_spec().next_seq_spec()[u as int] implies #[trigger] core.timelines_spec().events_spec()@[(
                    u, q)] == before_core.timelines_spec().events_spec()@[(u, q)] by {
                        assert(step_core.timelines_spec().events_spec()@[(u, q)]
                            == before_core.timelines_spec().events_spec()@[(u, q)]);
                    }
                }
                assert forall|j: int| 0 <= j < tickets.len() implies {
                    let q = (base + j) as u64;
                    &&& (#[trigger] tickets[j]).timeline_id() == t
                    &&& tickets[j].sequence() == q
                    &&& tickets[j] != Ticket::NONE
                    &&& core.timelines_spec().events_spec()@.contains_key((t, q))
                    &&& core.timelines_spec().events_spec()@[(t, q)] == items[j].submitted@.key()
                    &&& core.ledger_spec()@.submitted.dom().contains(items[j].submitted@.key())
                    &&& core.ledger_spec()@.submitted.map()[items[j].submitted@.key()]
                        == items[j].submitted@.value()
                } by {
                    broadcast use {vstd::map::group_map_lemmas, vstd::imap::group_imap_lemmas};

                    if j < i {
                        assert(tickets[j] == prior_tickets[j]);
                        assert(step_core.ledger_spec()@.submitted.dom().contains(
                            items[j].submitted@.key(),
                        ));
                    }
                }
            }
        }
        proof {
            reveal(rdma_leases_valid);
            lemma_core_wf_facts(core);
            lemma_core_timelines_wf(core);
            reveal(rdma_window_wf);
            assert(rdma_remove_range(full, (base + n) as u64, 0) =~= full) by {
                broadcast use vstd::map::group_map_lemmas;

            }
            assert(rdma::CqSnapshot::of(self.cq).pending == self.cq.in_flight());
            assert(rdma_window_wf(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines_spec(),
                t,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            ));
            lemma_rdma_leases_same_frames(
                self.active_leases@,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines_spec(),
                &core.timelines_spec(),
                t,
                self.cq.observer(),
                &before_core.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
            assert forall|i: int| 0 <= i < n implies {
                let item = #[trigger] items[i];
                let event = item.submitted@.key();
                &&& core.ledger_spec()@.submitted.dom().contains(event)
                &&& core.ledger_spec()@.submitted.map()[event] == item.submitted@.value()
            } by {
                assert(tickets[i].timeline_id() == t);
            }
        }
        tickets
    }
}

} // verus!
