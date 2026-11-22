#[cfg(verus_only)]
use super::poll::{lemma_rdma_core_after_submit, lemma_rdma_remove_range_peel};
use super::*;
use crate::timeline::{record_submission, DataOperation};
#[cfg(verus_only)]
use crate::timeline::{window_matches, BITMAP_CAPACITY};
use transfer::rdma;

verus! {

impl<'a> RdmaBackend<'a> {
    /// Preflight the next data WR ID before posting an RDMA read/write.
    pub fn next_data_id(&self, core: &CacheCore) -> (r: BackendResult<u64>)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            r matches Ok(s) ==> s == core.timelines.next_seq[self.timeline_id as int] && s
                <= MAX_TICKET_SEQ && s <= core.timelines.confirmed_seq[self.timeline_id as int]
                + BITMAP_CAPACITY,
    {
        proof {
            lemma_core_wf_facts(core);
        }
        core.timelines.next_data_id(self.timeline_id)
    }

    /// Preflight a contiguous batch of `count` data WR IDs before calling
    /// `rdma::post_send_batch`.
    pub fn next_data_ids(&self, core: &CacheCore, count: usize) -> (r: BackendResult<u64>)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            r matches Ok(s) ==> s == core.timelines.next_seq[self.timeline_id as int] && count > 0
                && s + count - 1 <= MAX_TICKET_SEQ && s + count - 1
                <= core.timelines.confirmed_seq[self.timeline_id as int] + BITMAP_CAPACITY,
    {
        proof {
            lemma_core_wf_facts(core);
        }
        core.timelines.next_data_ids(self.timeline_id, count)
    }

    /// Atomically record the accepted prefix returned by `rdma::post_send_batch`
    /// into `self.in_flight_ops`, `core.timelines`, and `core.ledger` (M7!).
    pub fn record_data_batch(
        &mut self,
        core: &mut CacheCore,
        frames: &[FrameId],
        operations: &[DataOperation],
        posted: Vec<rdma::PostedBatchItem>,
    ) -> (tickets: Vec<Ticket>)
        requires
            old(core).well_formed(),
            old(self).timeline_id < old(core).timelines.count,
            old(core).timelines.kinds[old(self).timeline_id as int] == (
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
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                old(core),
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                old(core),
                old(self).window_pool.pd(),
            ),
            rdma_leases_valid(
                old(self).active_leases@,
                &old(self).window_pool,
                &old(self).qp,
                old(core),
            ),
            old(self).bitmap.well_formed(),
            old(self).bitmap.observations(
                &old(core).timelines,
                old(self).timeline_id,
                old(core).node(),
                &old(core).ledger@,
            ),
            CONTROL_WR_ID_BASE <= old(self).next_control_wr_id,
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
                let base = old(core).timelines.next_seq[t as int];
                let n = posted.len() as int;
                &&& n == 0 || (base + n - 1 <= MAX_TICKET_SEQ && base + n - 1 <= old(
                    core,
                ).timelines.confirmed_seq[t as int] + BITMAP_CAPACITY)
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
                            ).ledger@.submitted.dom().contains(e) || exists|k: int|
                                0 <= k < i && (#[trigger] posted[k]).submitted@.key() == e
                        &&& old(core).disjoint_from_published(item.submitted@.value().access.writes)
                        &&& old(self).pending().contains_key(s)
                        &&& old(self).pending()[s] == item.submitted@.key()
                    }
                &&& window_matches(
                    old(self).in_flight_ops,
                    old(self).bitmap,
                    &old(core).timelines,
                    t,
                    rdma_remove_range(old(self).pending(), base, n),
                )
            }),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).timeline_id),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(core).ledger@.knows == old(core).ledger@.knows,
            final(core).ledger@.succeeded == old(core).ledger@.succeeded,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.submitted.map().submap_of(final(core).ledger@.submitted.map()),
            posted.len() == 1 ==> final(core).ledger@.submitted.map() == old(
                core,
            ).ledger@.submitted.map().insert(
                posted[0].submitted@.key(),
                posted[0].submitted@.value(),
            ),
            tickets.len() == posted.len(),
            forall|i: int|
                0 <= i < tickets.len() ==> {
                    let t = old(self).timeline_id;
                    let s = (old(core).timelines.next_seq[t as int] + i) as u64;
                    &&& (#[trigger] tickets[i]).timeline_id() == t
                    &&& tickets[i].sequence() == s
                    &&& tickets[i] != Ticket::NONE
                    &&& final(core).timelines.issued(t, s)
                    &&& final(core).timelines.events@.contains_key((t, s))
                    &&& final(core).timelines.events@[(t, s)] == posted[i].submitted@.key()
                },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_core = *core;
        let ghost full = self.pending();
        let ghost items = posted@;
        let ghost base = core.timelines.next_seq[self.timeline_id as int];
        let t = self.timeline_id;
        let n = posted.len();
        if n == 0 {
            proof {
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
                    &core.timelines,
                    t,
                    &core.ledger@,
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
                t < core.timelines.count,
                base == before_core.timelines.next_seq[t as int],
                full == self.pending(),
                base + n - 1 <= MAX_TICKET_SEQ,
                base + n - 1 <= core.timelines.confirmed_seq[t as int] + BITMAP_CAPACITY,
                vstd::std_specs::iter::IteratorSpec::remaining(&iter) == items.skip(
                    tickets.len() as int,
                ),
                core.well_formed(),
                before_core.well_formed(),
                core.node_id == before_core.node_id,
                core.instance_id() == before_core.instance_id(),
                core.tiers == before_core.tiers,
                core.frames == before_core.frames,
                core.catalog == before_core.catalog,
                core.spec == before_core.spec,
                core.next_version_seq == before_core.next_version_seq,
                core.version_seq_limit == before_core.version_seq_limit,
                core.active_leases == before_core.active_leases,
                core.version_registry == before_core.version_registry,
                core.timelines.count == before_core.timelines.count,
                core.timelines.kinds == before_core.timelines.kinds,
                core.timelines.confirmed_seq == before_core.timelines.confirmed_seq,
                core.timelines.overflow_watermark == before_core.timelines.overflow_watermark,
                core.timelines.next_seq@ == before_core.timelines.next_seq@.update(
                    t as int,
                    (base + tickets.len()) as u64,
                ),
                before_core.ledger@.submitted.map().submap_of(core.ledger@.submitted.map()),
                core.ledger@.knows == before_core.ledger@.knows,
                core.ledger@.succeeded == before_core.ledger@.succeeded,
                core.ledger@.revoked == before_core.ledger@.revoked,
                tickets.len() == 0 ==> core.ledger@.submitted.map()
                    == before_core.ledger@.submitted.map(),
                tickets.len() == 1 ==> core.ledger@.submitted.map()
                    == before_core.ledger@.submitted.map().insert(
                    items[0].submitted@.key(),
                    items[0].submitted@.value(),
                ),
                self.bitmap.well_formed(),
                self.bitmap.observations(&core.timelines, t, core.node(), &core.ledger@),
                window_matches(
                    self.in_flight_ops,
                    self.bitmap,
                    &core.timelines,
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
                                ==> before_core.ledger@.submitted.dom().contains(e) || exists|
                                k: int,
                            |
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
                        &&& core.timelines.events@.contains_key((t, s))
                        &&& core.timelines.events@[(t, s)] == items[j].submitted@.key()
                        &&& core.ledger@.submitted.dom().contains(items[j].submitted@.key())
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
            let ticket = record_submission(
                &mut self.in_flight_ops,
                Ghost(self.bitmap),
                &mut core.timelines,
                t,
                frames[i],
                operations[i],
                Ghost(next_pending),
                item.submitted,
                Tracked(core.ledger.borrow_mut()),
            );
            let ghost prior_tickets = tickets@;
            tickets.push(ticket);
            proof {
                assert(val.preds.subset_of(step_core.ledger@.submitted.dom())) by {
                    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

                    assert forall|e: TransferEvent| #[trigger]
                        val.preds.contains(e) implies step_core.ledger@.submitted.dom().contains(
                        e,
                    ) by {
                        if before_core.ledger@.submitted.dom().contains(e) {
                            assert(before_core.ledger@.submitted.map().submap_of(
                                step_core.ledger@.submitted.map(),
                            ));
                        } else {
                            let k = choose|k: int|
                                0 <= k < i as int && (#[trigger] items[k]).submitted@.key() == e;
                            assert(tickets[k].timeline_id() == t);
                            assert(step_core.ledger@.submitted.dom().contains(
                                items[k].submitted@.key(),
                            ));
                        }
                    }
                }
                lemma_rdma_core_after_submit(&step_core, core, t, ev, val);
                vstd::imap_lib::lemma_submap_of_trans(
                    before_core.ledger@.submitted.map(),
                    step_core.ledger@.submitted.map(),
                    core.ledger@.submitted.map(),
                );
                assert(core.timelines.next_seq@ =~= before_core.timelines.next_seq@.update(
                    t as int,
                    (s + 1) as u64,
                ));
                assert(peer_core_step(&before_core, core, t)) by {
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                    broadcast use {
                        vstd::map::group_map_lemmas,
                        vstd::imap::group_imap_lemmas,
                        vstd::iset::group_iset_lemmas,
                    };

                    assert forall|u: u8, q: u64|
                        u < before_core.timelines.count && u != t && 1 <= q
                            < before_core.timelines.next_seq[u as int] implies #[trigger] core.timelines.events@[(
                    u, q)] == before_core.timelines.events@[(u, q)] by {
                        assert(step_core.timelines.events@[(u, q)]
                            == before_core.timelines.events@[(u, q)]);
                    }
                }
                assert forall|j: int| 0 <= j < tickets.len() implies {
                    let q = (base + j) as u64;
                    &&& (#[trigger] tickets[j]).timeline_id() == t
                    &&& tickets[j].sequence() == q
                    &&& tickets[j] != Ticket::NONE
                    &&& core.timelines.events@.contains_key((t, q))
                    &&& core.timelines.events@[(t, q)] == items[j].submitted@.key()
                    &&& core.ledger@.submitted.dom().contains(items[j].submitted@.key())
                } by {
                    broadcast use {vstd::map::group_map_lemmas, vstd::imap::group_imap_lemmas};

                    if j < i {
                        assert(tickets[j] == prior_tickets[j]);
                        assert(step_core.ledger@.submitted.dom().contains(
                            items[j].submitted@.key(),
                        ));
                    }
                }
            }
        }
        proof {
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
                &core.timelines,
                t,
                &core.ledger@,
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
            lemma_rdma_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
        }
        tickets
    }
}

} // verus!
