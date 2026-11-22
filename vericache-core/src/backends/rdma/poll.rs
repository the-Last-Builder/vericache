use super::*;

verus! {

/// Pure proof helper: `completion_prefix` is prefix-closed.
proof fn lemma_rdma_valid_prefix(
    cs: Seq<rdma::Completed>,
    i: int,
    n: int,
    before: rdma::CqSnapshot,
    id: InstanceId,
)
    requires
        0 <= i <= n <= cs.len(),
        rdma::completion_prefix(cs, n, before, id).1,
    ensures
        rdma::completion_prefix(cs, i, before, id).1,
    decreases n - i,
{
    if i < n {
        lemma_rdma_valid_prefix(cs, i, n - 1, before, id);
    }
}

/// Pure proof helper: removing a control WR from `before` leaves `rdma_data_events`
/// unchanged and preserves control-WR-ID bounds.
proof fn lemma_rdma_control_reap_window(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    ledger: &KnowledgeLedger,
    before: rdma::CqSnapshot,
    after: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
    wr_id: u64,
)
    requires
        rdma_window_wf(ops, bitmap, table, t, ledger, before, id, next_control_wr_id),
        before.pending.contains_key(wr_id),
        !(before.pending[wr_id].kind is DataTransfer),
        after.observer == before.observer,
        after.pending == before.pending.remove(wr_id),
    ensures
        rdma_data_events(after) == rdma_data_events(before),
        rdma_window_wf(ops, bitmap, table, t, ledger, after, id, next_control_wr_id),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_window_wf);
    assert(rdma_data_events(after) =~= rdma_data_events(before));
}

/// Pure proof helper: removing a data WR from `before` removes exactly `s` from
/// `rdma_data_events` and preserves control-WR-ID bounds.
proof fn lemma_rdma_data_reap_events(before: rdma::CqSnapshot, after: rdma::CqSnapshot, s: u64)
    requires
        before.pending.contains_key(s),
        before.pending[s].kind is DataTransfer,
        after.observer == before.observer,
        after.pending == before.pending.remove(s),
    ensures
        rdma_data_events(before).contains_key(s),
        rdma_data_events(before)[s] == before.pending[s].kind->DataTransfer_event,
        rdma_data_events(after) == rdma_data_events(before).remove(s),
        closure_from(before.infos, ISet::empty().insert(rdma_data_events(before)[s])).contains(
            rdma_data_events(before)[s],
        ),
{
    broadcast use {vstd::map::group_map_lemmas, vstd::iset::group_iset_lemmas};

    assert(rdma_data_events(after) =~= rdma_data_events(before).remove(s));
}

/// Pure proof helper: restore `CacheCore::well_formed()` after `complete_data`
/// advances `core.timelines` and `core.ledger` (no writes added to `submitted`).
proof fn lemma_rdma_core_after_complete(before_core: &CacheCore, after_core: &CacheCore, t: u8)
    requires
        before_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.tiers == before_core.tiers,
        after_core.frames == before_core.frames,
        after_core.catalog == before_core.catalog,
        after_core.version_seq_start == before_core.version_seq_start,
        after_core.next_version_seq == before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.active_leases == before_core.active_leases,
        after_core.free_lease_slots == before_core.free_lease_slots,
        after_core.lease_generation == before_core.lease_generation,
        after_core.spec == before_core.spec,
        after_core.version_registry == before_core.version_registry,
        after_core.timelines.well_formed(
            before_core.instance_id(),
            before_core.node(),
            &after_core.ledger@,
        ),
        after_core.timelines.polled(&before_core.timelines, t),
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.kinds == before_core.timelines.kinds,
        after_core.timelines.next_seq == before_core.timelines.next_seq,
        after_core.timelines.events == before_core.timelines.events,
        after_core.timelines.overflow_watermark == before_core.timelines.overflow_watermark,
        after_core.ledger@.submitted == before_core.ledger@.submitted,
        after_core.ledger@.revoked == before_core.ledger@.revoked,
        before_core.ledger@.knows.set().subset_of(after_core.ledger@.knows.set()),
        before_core.ledger@.succeeded.dom().subset_of(after_core.ledger@.succeeded.dom()),
    ensures
        after_core.well_formed(),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    lemma_core_wf_facts(before_core);
    assert(complete_receipts(after_core.ledger@)) by {
        assert(complete_receipts(before_core.ledger@));
    }
    assert(after_core.kernel_view().extends(before_core.kernel_view())) by {
        broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    }
    assert(before_core.ledger@.submitted.map().submap_of(after_core.ledger@.submitted.map())) by {
        broadcast use vstd::imap::group_imap_lemmas;

    }
    after_core.preserve_after_ledger_growth(*before_core);
}

/// Pure proof helper: restore `CacheCore::well_formed()` after `record_submission`
/// appends one disjoint data event to `core.timelines` and `core.ledger`.
pub(super) proof fn lemma_rdma_core_after_submit(
    before_core: &CacheCore,
    after_core: &CacheCore,
    t: u8,
    ev: TransferEvent,
    val: EventInfo<TransferDomain>,
)
    requires
        before_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.tiers == before_core.tiers,
        after_core.frames == before_core.frames,
        after_core.catalog == before_core.catalog,
        after_core.version_seq_start == before_core.version_seq_start,
        after_core.next_version_seq == before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.active_leases == before_core.active_leases,
        after_core.free_lease_slots == before_core.free_lease_slots,
        after_core.lease_generation == before_core.lease_generation,
        after_core.spec == before_core.spec,
        after_core.version_registry == before_core.version_registry,
        after_core.timelines.well_formed(
            before_core.instance_id(),
            before_core.node(),
            &after_core.ledger@,
        ),
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.kinds == before_core.timelines.kinds,
        after_core.timelines.confirmed_seq == before_core.timelines.confirmed_seq,
        after_core.timelines.overflow_watermark == before_core.timelines.overflow_watermark,
        after_core.timelines.next_seq@ == before_core.timelines.next_seq@.update(
            t as int,
            (before_core.timelines.next_seq[t as int] + 1) as u64,
        ),
        after_core.timelines.events@ == before_core.timelines.events@.insert(
            (t, before_core.timelines.next_seq[t as int]),
            ev,
        ),
        after_core.ledger@.submitted.map() == before_core.ledger@.submitted.map().insert(ev, val),
        before_core.ledger@.submitted.map().submap_of(after_core.ledger@.submitted.map()),
        after_core.ledger@.knows == before_core.ledger@.knows,
        after_core.ledger@.succeeded == before_core.ledger@.succeeded,
        after_core.ledger@.revoked == before_core.ledger@.revoked,
        val.preds.subset_of(before_core.ledger@.submitted.dom()),
        before_core.disjoint_from_published(val.access.writes),
    ensures
        after_core.well_formed(),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    lemma_core_wf_facts(before_core);
    assert(complete_receipts(after_core.ledger@)) by {
        broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        assert forall|e: TransferEvent| #[trigger]
            after_core.ledger@.submitted.dom().contains(
                e,
            ) implies after_core.ledger@.submitted.map()[e].preds.subset_of(
            after_core.ledger@.submitted.dom(),
        ) by {
            if e != ev {
                assert(before_core.ledger@.submitted.dom().contains(e));
                assert(before_core.ledger@.submitted.map()[e]
                    == after_core.ledger@.submitted.map()[e]);
            }
        }
    }
    assert(after_core.kernel_view().extends(before_core.kernel_view())) by {
        broadcast use {
            vstd::map::group_map_lemmas,
            vstd::imap::group_imap_lemmas,
            vstd::iset::group_iset_lemmas,
        };

    }
    assert forall|e: TransferEvent, f: FrameId|
        after_core.ledger@.submitted.dom().contains(e)
            && !before_core.ledger@.submitted.dom().contains(e) && f.0
            < after_core.frames.metas.len() && (#[trigger] after_core.frames.meta(f)).state
            == STATE_PUBLISHED implies !(
    #[trigger] after_core.ledger@.submitted.map()[e]).access.writes.contains(
        after_core.frames.frame(f).slot.loc,
    ) by {
        broadcast use vstd::imap::group_imap_lemmas;

        assert(e == ev);
        assert(before_core.frames.meta(f).state == STATE_PUBLISHED);
    }
    assert(before_core.ledger@.knows.set().subset_of(after_core.ledger@.knows.set())) by {
        broadcast use vstd::iset::group_iset_lemmas;

    }
    assert(before_core.ledger@.succeeded.dom().subset_of(after_core.ledger@.succeeded.dom())) by {
        broadcast use vstd::iset::group_iset_lemmas;

    }
    after_core.preserve_after_ledger_growth(*before_core);
}

/// Pure proof helper for `rdma_remove_range` peel step in `record_data_batch`.
pub(super) proof fn lemma_rdma_remove_range_peel(
    full: Map<u64, TransferEvent>,
    base: u64,
    n: int,
    i: int,
)
    requires
        0 <= i < n,
        base + n - 1 <= MAX_TICKET_SEQ,
        full.contains_key((base + i) as u64),
    ensures
        ({
            let s = (base + i) as u64;
            let next_pending = rdma_remove_range(full, (s + 1) as u64, n - i - 1);
            &&& next_pending.contains_key(s)
            &&& next_pending[s] == full[s]
            &&& next_pending.remove(s) == rdma_remove_range(full, s, n - i)
        }),
{
    broadcast use vstd::map::group_map_lemmas;

    let s = (base + i) as u64;
    let next_pending = rdma_remove_range(full, (s + 1) as u64, n - i - 1);
    assert(next_pending.remove(s) =~= rdma_remove_range(full, s, n - i));
}

impl<'a> RdmaBackend<'a> {
    pub open spec fn delivers(c: rdma::Completed, result: PeerCompletion) -> bool {
        match c {
            rdma::Completed::Data { wr_id, .. } => result matches PeerCompletion::Data(op, Ok(()))
                && op.sequence == wr_id,
            rdma::Completed::DataFailed {
                wr_id,
                status,
                ..
            } => result matches PeerCompletion::Data(op, Err(e)) && op.sequence == wr_id && e
                == status,
            rdma::Completed::BindFailed { wr_id, status, witness } => result == (
            PeerCompletion::BindFailed { wr_id, status, witness }),
            _ => result == PeerCompletion::Control(c),
        }
    }

    pub open spec fn delivers_batch(
        cs: Seq<rdma::Completed>,
        out: Seq<PeerCompletion>,
        before: rdma::CompletionQueue,
        after: rdma::CompletionQueue,
        id: InstanceId,
    ) -> bool {
        &&& rdma::completions_describe(cs, before, after, id)
        &&& cs.len() == out.len()
        &&& forall|i: int| 0 <= i < out.len() ==> #[trigger] Self::delivers(cs[i], out[i])
    }

    /// Dispatch one reaped CQ completion into `core.timelines`, `core.ledger`,
    /// `self.bitmap`, and `self.in_flight_ops`.
    fn dispatch_one(
        ops: &mut [Option<PendingOperation>; 512],
        bitmap: &mut SlidingBitmap,
        core: &mut CacheCore,
        t: u8,
        next_control_wr_id: u64,
        c: rdma::Completed,
        Ghost(before): Ghost<rdma::CqSnapshot>,
        Ghost(after): Ghost<rdma::CqSnapshot>,
    ) -> (result: PeerCompletion)
        requires
            old(core).well_formed(),
            rdma_window_wf(
                *old(ops),
                *old(bitmap),
                &old(core).timelines,
                t,
                &old(core).ledger@,
                before,
                old(core).instance_id(),
                next_control_wr_id,
            ),
            before.observer == old(core).node(),
            after.observer == before.observer,
            c.describes_snapshot(before, after, old(core).instance_id()),
        ensures
            Self::delivers(c, result),
            final(core).well_formed(),
            rdma_window_wf(
                *final(ops),
                *final(bitmap),
                &final(core).timelines,
                t,
                &final(core).ledger@,
                after,
                final(core).instance_id(),
                next_control_wr_id,
            ),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).version_registry == old(core).version_registry,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.events == old(core).timelines.events,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            peer_core_step(old(core), final(core), t),
            final(core).ledger@.submitted == old(core).ledger@.submitted,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            old(core).ledger@.succeeded.dom().subset_of(final(core).ledger@.succeeded.dom()),
            result matches PeerCompletion::Data(op, _) ==> rdma_data_events(before).contains_key(
                op.sequence,
            ) && final(core).ledger@.knows.set().contains(
                (old(core).node(), rdma_data_events(before)[op.sequence]),
            ),
            result matches PeerCompletion::Data(op, Ok(())) ==> rdma_data_events(
                before,
            ).contains_key(op.sequence) && final(core).ledger@.succeeded.contains(
                rdma_data_events(before)[op.sequence],
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = next_control_wr_id;
        let ghost before_core = *core;
        let ghost id = core.instance_id();
        let ghost node = core.node();
        let (s, outcome, known, success) = match c {
            rdma::Completed::Data { wr_id, knows, succeeded: _witness, .. } => (
                wr_id,
                Ok(()),
                knows,
                Tracked(Some(_witness.get())),
            ),
            rdma::Completed::DataFailed { wr_id, status, knows, .. } => (
                wr_id,
                Err(status),
                knows,
                Tracked(None),
            ),
            rdma::Completed::BindFailed { wr_id, status, witness } => {
                proof {
                    lemma_rdma_control_reap_window(
                        *ops,
                        *bitmap,
                        &core.timelines,
                        t,
                        &core.ledger@,
                        before,
                        after,
                        id,
                        next_control_wr_id,
                        wr_id,
                    );
                    lemma_peer_core_step_refl(core, t);
                }
                return PeerCompletion::BindFailed { wr_id, status, witness };
            },
            control => {
                proof {
                    lemma_rdma_control_reap_window(
                        *ops,
                        *bitmap,
                        &core.timelines,
                        t,
                        &core.ledger@,
                        before,
                        after,
                        id,
                        next_control_wr_id,
                        control.id(),
                    );
                    lemma_peer_core_step_refl(core, t);
                }
                return PeerCompletion::Control(control);
            },
        };
        let ghost pending = rdma_data_events(before);
        proof {
            reveal(rdma_window_wf);
            lemma_rdma_data_reap_events(before, after, s);
            assert(core.timelines.events@[(t, s)] == pending[s]);
            assert(core.timelines.observed(t, s, node, known@));
        }
        let op = complete_data(
            ops,
            bitmap,
            &mut core.timelines,
            t,
            s,
            Ghost(pending),
            Ghost(node),
            known,
            success,
            Tracked(core.ledger.borrow_mut()),
        );
        proof {
            assert(before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set())) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            assert(before_core.ledger@.succeeded.dom().subset_of(core.ledger@.succeeded.dom())) by {
                broadcast use vstd::map::group_map_lemmas;

            }
            lemma_rdma_core_after_complete(&before_core, core, t);
            reveal(rdma_window_wf);
            assert forall|k: u64|
                after.pending.contains_key(k) && !((
                #[trigger] after.pending[k]).kind is DataTransfer) implies CONTROL_WR_ID_BASE <= k
                < next_control_wr_id by {
                broadcast use vstd::map::group_map_lemmas;

                assert(before.pending.contains_key(k));
            }
            if outcome.is_ok() {
                broadcast use vstd::map::group_map_lemmas;

                assert(core.ledger@.succeeded.contains(pending[s]));
            }
            reveal(peer_core_step_except);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        }
        PeerCompletion::Data(op, outcome)
    }

    /// Poll at most one completion from the RDMA CQ and advance `core`.
    pub fn poll_cq(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<Option<PeerCompletion>>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
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
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(core).timelines.events == old(core).timelines.events,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            peer_core_step(old(core), final(core), old(self).timeline_id),
            final(core).ledger@.submitted == old(core).ledger@.submitted,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            old(core).ledger@.succeeded.dom().subset_of(final(core).ledger@.succeeded.dom()),
            r matches Ok(None) ==> final(self).cq == old(self).cq && *final(core) == *old(core),
            r matches Ok(Some(result)) ==> exists|c: rdma::Completed|
                c.describes(old(self).cq, final(self).cq, instance.id())
                    && #[trigger] Self::delivers(c, result),
            r matches Ok(Some(PeerCompletion::Data(op, _))) ==> old(self).pending().contains_key(
                op.sequence,
            ) && final(core).ledger@.knows.set().contains(
                (old(core).node(), old(self).pending()[op.sequence]),
            ),
            r matches Ok(Some(PeerCompletion::Data(op, Ok(()))))
                ==> final(core).ledger@.succeeded.contains(old(self).pending()[op.sequence]),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_core = *core;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let c = match rdma::poll_completion_queue(&mut self.cq, Tracked(instance)) {
            Err(e) => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                }
                return Err(rdma_error(e));
            },
            Ok(rdma::PollStatus::Empty) => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                }
                return Ok(None);
            },
            Ok(rdma::PollStatus::Completed(c)) => c,
        };
        let result = Self::dispatch_one(
            &mut self.in_flight_ops,
            &mut self.bitmap,
            core,
            self.timeline_id,
            self.next_control_wr_id,
            c,
            Ghost(before_cq),
            Ghost(rdma::CqSnapshot::of(self.cq)),
        );
        proof {
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
            assert(Self::delivers(c, result));
        }
        Ok(Some(result))
    }

    /// Poll a batch of up to `max_entries` completions from the RDMA CQ.
    pub fn poll_cq_batch(
        &mut self,
        core: &mut CacheCore,
        max_entries: usize,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<(Vec<PeerCompletion>, Ghost<Seq<rdma::Completed>>)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
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
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).active_leases == old(self).active_leases,
            r matches Ok((v, cs)) ==> Self::delivers_batch(
                cs@,
                v@,
                old(self).cq,
                final(self).cq,
                instance.id(),
            ) && v.len() <= max_entries,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_core = *core;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let cs = match rdma::poll_completion_queue_batch(
            &mut self.cq,
            max_entries,
            Tracked(instance),
        ) {
            Err(e) => return Err(rdma_error(e)),
            Ok(cs) => cs,
        };
        let ghost batch = cs@;
        if cs.is_empty() {
            let empty = Vec::new();
            proof {
                assert(rdma::completion_prefix(batch, 0, before_cq, instance.id()).0 == before_cq);
                assert(Self::delivers_batch(batch, empty@, old(self).cq, self.cq, instance.id()));
            }
            return Ok((empty, Ghost(batch)));
        }
        let batch_len = cs.len();
        let mut out: Vec<PeerCompletion> = Vec::with_capacity(batch_len);
        let mut iter = cs.into_iter();
        let ghost mut snapshot = before_cq;
        while out.len() < batch_len
            invariant
                out.len() <= batch.len(),
                batch_len == batch.len(),
                batch.len() <= max_entries,
                rdma::completions_describe(batch, old(self).cq, self.cq, instance.id()),
                before_cq == rdma::CqSnapshot::of(old(self).cq),
                forall|j: int| 0 <= j < out.len() ==> #[trigger] Self::delivers(batch[j], out[j]),
                vstd::std_specs::iter::IteratorSpec::remaining(&iter) == batch.skip(
                    out.len() as int,
                ),
                snapshot == rdma::completion_prefix(
                    batch,
                    out.len() as int,
                    before_cq,
                    instance.id(),
                ).0,
                snapshot.observer == core.node(),
                core.well_formed(),
                core.node_id == before_core.node_id,
                core.instance_id() == before_core.instance_id(),
                core.tiers == before_core.tiers,
                core.frames == before_core.frames,
                core.catalog == before_core.catalog,
                core.spec == before_core.spec,
                core.next_version_seq == before_core.next_version_seq,
                core.version_seq_limit == before_core.version_seq_limit,
                core.version_registry == before_core.version_registry,
                core.timelines.count == before_core.timelines.count,
                core.timelines.kinds == before_core.timelines.kinds,
                instance.id() == core.instance_id(),
                rdma_window_wf(
                    self.in_flight_ops,
                    self.bitmap,
                    &core.timelines,
                    self.timeline_id,
                    &core.ledger@,
                    snapshot,
                    core.instance_id(),
                    self.next_control_wr_id,
                ),
            decreases batch.len() - out.len(),
        {
            let ghost i = out.len() as int;
            let c = iter.next().unwrap();
            let ghost after = snapshot.reap(c.id());
            proof {
                assert(c == batch[i]);
                lemma_rdma_valid_prefix(batch, i + 1, batch.len() as int, before_cq, instance.id());
                assert(rdma::completion_prefix(batch, i + 1, before_cq, instance.id()) == (
                    after,
                    true,
                ));
            }
            let result = Self::dispatch_one(
                &mut self.in_flight_ops,
                &mut self.bitmap,
                core,
                self.timeline_id,
                self.next_control_wr_id,
                c,
                Ghost(snapshot),
                Ghost(after),
            );
            let ghost prefix = out@;
            out.push(result);
            proof {
                assert forall|j: int| 0 <= j < out.len() implies #[trigger] Self::delivers(
                    batch[j],
                    out[j],
                ) by {
                    if j < i {
                        assert(out[j] == prefix[j]);
                    }
                }
                snapshot = after;
            }
        }
        proof {
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
            assert(Self::delivers_batch(batch, out@, old(self).cq, self.cq, instance.id()));
        }
        Ok((out, Ghost(batch)))
    }
}

} // verus!
