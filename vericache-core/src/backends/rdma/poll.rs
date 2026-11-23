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

/// Pure proof helper: `completion_prefix` removes a subset of pending WR IDs
/// without altering any remaining pending entries or `observer`.
proof fn lemma_rdma_completion_prefix_submap(
    cs: Seq<rdma::Completed>,
    n: int,
    before: rdma::CqSnapshot,
    id: InstanceId,
)
    requires
        0 <= n <= cs.len(),
        rdma::completion_prefix(cs, n, before, id).1,
    ensures
        ({
            let after = rdma::completion_prefix(cs, n, before, id).0;
            &&& after.observer == before.observer
            &&& forall|k: u64| #[trigger]
                after.pending.contains_key(k) ==> before.pending.contains_key(k) && after.pending[k]
                    == before.pending[k]
        }),
    decreases n,
{
    if n > 0 {
        let prev = rdma::completion_prefix(cs, n - 1, before, id).0;
        lemma_rdma_completion_prefix_submap(cs, n - 1, before, id);
        let c = cs[n - 1];
        assert(c.describes_snapshot(prev, prev.reap(c.id()), id));
        broadcast use vstd::map::group_map_lemmas;

    }
}

/// Helper: preserve `lease_index_spec` for all `!has_active_wr()` leases across an
/// update at `idx` where `before[idx].has_active_wr()` and `mw_slot` is preserved.
proof fn lemma_nonactive_leases_after_update(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        after.len() == before.len(),
        before[idx].has_active_wr(),
        forall|j: int| 0 <= j < before.len() && j != idx ==> (#[trigger] after[j]) == before[j],
    ensures
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && !(before[rdma_lease_index_spec(
                before,
                h,
            )->Some_0]).has_active_wr() ==> (rdma_lease_index_spec(after, h) matches Some(j)
                && after[j] == before[rdma_lease_index_spec(before, h)->Some_0]),
{
    reveal(rdma_leases_disjoint);
    assert forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && !(before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0]).has_active_wr() implies (rdma_lease_index_spec(after, h) matches Some(j)
        && after[j] == before[rdma_lease_index_spec(before, h)->Some_0]) by {
        let i = rdma_lease_index_spec(before, h).unwrap();
        assert(i != idx);
        assert(after[i] == before[i]);
        assert(after[i].mw_slot as usize == h.mw_slot && after[i].generation == h.generation);
        let j = rdma_lease_index_spec(after, h).unwrap();
        if j != i {
            assert(after[i].mw_slot == after[j].mw_slot);
            if i < j {
                assert(after[i].mw_slot != after[j].mw_slot);
            } else {
                assert(after[j].mw_slot != after[i].mw_slot);
            }
        }
    }
}

/// Helper: preserve `lease_index_spec` for all `!has_active_wr()` leases across a
/// `swap_remove(idx)` when `before[idx].has_active_wr()`.
proof fn lemma_nonactive_leases_after_swap_remove(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        before[idx].has_active_wr(),
        after.len() == before.len() - 1,
        forall|j: int|
            0 <= j < after.len() ==> (#[trigger] after[j]) == if j == idx {
                before[before.len() - 1]
            } else {
                before[j]
            },
    ensures
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && !(before[rdma_lease_index_spec(
                before,
                h,
            )->Some_0]).has_active_wr() ==> (rdma_lease_index_spec(after, h) matches Some(j)
                && after[j] == before[rdma_lease_index_spec(before, h)->Some_0]),
{
    reveal(rdma_leases_disjoint);
    let last = before.len() - 1;
    assert forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && !(before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0]).has_active_wr() implies (rdma_lease_index_spec(after, h) matches Some(j)
        && after[j] == before[rdma_lease_index_spec(before, h)->Some_0]) by {
        let i = rdma_lease_index_spec(before, h).unwrap();
        assert(i != idx);
        let target = if i == last {
            idx
        } else {
            i
        };
        assert(after[target] == before[i]);
        assert(after[target].mw_slot as usize == h.mw_slot && after[target].generation
            == h.generation);
        let j = rdma_lease_index_spec(after, h).unwrap();
        if j != target {
            if j < target {
                assert(after[j].mw_slot != after[target].mw_slot);
            } else {
                assert(after[target].mw_slot != after[j].mw_slot);
            }
        }
    }
}

pub open spec fn rdma_lease_is_binding_like(l: ActiveRemoteLease) -> bool {
    ||| l.state == RemoteLeaseState::Binding
    ||| l.state == RemoteLeaseState::BindReady
    ||| l.state == RemoteLeaseState::BindFailedReady
    ||| (l.state == RemoteLeaseState::ControlFailedState && l.share_cap is Some)
}

pub open spec fn rdma_lease_binding_preserved(
    ol: ActiveRemoteLease,
    nl: ActiveRemoteLease,
) -> bool {
    &&& nl.frame == ol.frame
    &&& nl.mw_slot == ol.mw_slot
    &&& nl.generation == ol.generation
    &&& nl.wr_id == ol.wr_id
    &&& nl.share == ol.share
    &&& nl.share_cap == ol.share_cap
    &&& rdma_lease_is_binding_like(nl)
}

proof fn lemma_binding_leases_after_update(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        after.len() == before.len(),
        rdma_lease_is_binding_like(before[idx]) ==> rdma_lease_binding_preserved(
            before[idx],
            after[idx],
        ),
        forall|j: int| 0 <= j < before.len() && j != idx ==> (#[trigger] after[j]) == before[j],
    ensures
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && rdma_lease_is_binding_like(
                before[rdma_lease_index_spec(before, h)->Some_0],
            ) ==> (rdma_lease_index_spec(after, h) matches Some(j) && rdma_lease_binding_preserved(
                before[rdma_lease_index_spec(before, h)->Some_0],
                after[j],
            )),
{
    reveal(rdma_leases_disjoint);
    assert forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && rdma_lease_is_binding_like(
            before[rdma_lease_index_spec(before, h)->Some_0],
        ) implies (rdma_lease_index_spec(after, h) matches Some(j) && rdma_lease_binding_preserved(
        before[rdma_lease_index_spec(before, h)->Some_0],
        after[j],
    )) by {
        let i = rdma_lease_index_spec(before, h).unwrap();
        assert(after[i].mw_slot as usize == h.mw_slot && after[i].generation == h.generation);
        let j = rdma_lease_index_spec(after, h).unwrap();
        if j != i {
            if j < i {
                assert(after[j].mw_slot != after[i].mw_slot);
            } else {
                assert(after[i].mw_slot != after[j].mw_slot);
            }
        }
    }
}

proof fn lemma_unbinding_revoking_after_update(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        after.len() == before.len(),
        after[idx].state != RemoteLeaseState::Unbinding,
        after[idx].state != RemoteLeaseState::Revoking,
        forall|j: int| 0 <= j < before.len() && j != idx ==> (#[trigger] after[j]) == before[j],
    ensures
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && before[rdma_lease_index_spec(
                before,
                h,
            )->Some_0].state == RemoteLeaseState::Unbinding && rdma_lease_index_spec(
                after,
                h,
            ) is Some && after[rdma_lease_index_spec(after, h)->Some_0].state
                == RemoteLeaseState::Unbinding ==> after[rdma_lease_index_spec(after, h)->Some_0]
                == before[rdma_lease_index_spec(before, h)->Some_0],
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && before[rdma_lease_index_spec(
                before,
                h,
            )->Some_0].state == RemoteLeaseState::Revoking && rdma_lease_index_spec(
                after,
                h,
            ) is Some && after[rdma_lease_index_spec(after, h)->Some_0].state
                == RemoteLeaseState::Revoking ==> after[rdma_lease_index_spec(after, h)->Some_0]
                == before[rdma_lease_index_spec(before, h)->Some_0],
{
    reveal(rdma_leases_disjoint);
    assert forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && rdma_lease_index_spec(after, h) is Some && (
        after[rdma_lease_index_spec(after, h)->Some_0].state == RemoteLeaseState::Unbinding
            || after[rdma_lease_index_spec(after, h)->Some_0].state
            == RemoteLeaseState::Revoking) implies after[rdma_lease_index_spec(after, h)->Some_0]
        == before[rdma_lease_index_spec(before, h)->Some_0] by {
        let i0 = rdma_lease_index_spec(before, h).unwrap();
        let j = rdma_lease_index_spec(after, h).unwrap();
        assert(j != idx);
        assert(after[j] == before[j]);
        if j != i0 {
            assert(before[j].mw_slot == before[i0].mw_slot);
        }
    }
}

proof fn lemma_unbinding_revoking_after_swap_remove(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        after.len() == before.len() - 1,
        forall|j: int|
            0 <= j < after.len() ==> (#[trigger] after[j]) == if j == idx {
                before[before.len() - 1]
            } else {
                before[j]
            },
    ensures
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && before[rdma_lease_index_spec(
                before,
                h,
            )->Some_0].state == RemoteLeaseState::Unbinding && rdma_lease_index_spec(
                after,
                h,
            ) is Some && after[rdma_lease_index_spec(after, h)->Some_0].state
                == RemoteLeaseState::Unbinding ==> after[rdma_lease_index_spec(after, h)->Some_0]
                == before[rdma_lease_index_spec(before, h)->Some_0],
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && before[rdma_lease_index_spec(
                before,
                h,
            )->Some_0].state == RemoteLeaseState::Revoking && rdma_lease_index_spec(
                after,
                h,
            ) is Some && after[rdma_lease_index_spec(after, h)->Some_0].state
                == RemoteLeaseState::Revoking ==> after[rdma_lease_index_spec(after, h)->Some_0]
                == before[rdma_lease_index_spec(before, h)->Some_0],
{
    reveal(rdma_leases_disjoint);
    assert forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && rdma_lease_index_spec(
            after,
            h,
        ) is Some implies after[rdma_lease_index_spec(after, h)->Some_0]
        == before[rdma_lease_index_spec(before, h)->Some_0] by {
        let i0 = rdma_lease_index_spec(before, h).unwrap();
        let j = rdma_lease_index_spec(after, h).unwrap();
        let src = if j == idx {
            (before.len() - 1) as int
        } else {
            j
        };
        assert(after[j] == before[src]);
        if src != i0 {
            assert(before[src].mw_slot == before[i0].mw_slot);
        }
    }
}

proof fn lemma_binding_leases_after_swap_remove(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        !rdma_lease_is_binding_like(before[idx]),
        after.len() == before.len() - 1,
        forall|j: int|
            0 <= j < after.len() ==> (#[trigger] after[j]) == if j == idx {
                before[before.len() - 1]
            } else {
                before[j]
            },
    ensures
        forall|h: RemoteLeaseHandle| #[trigger]
            rdma_lease_index_spec(before, h) is Some && rdma_lease_is_binding_like(
                before[rdma_lease_index_spec(before, h)->Some_0],
            ) ==> (rdma_lease_index_spec(after, h) matches Some(j) && rdma_lease_binding_preserved(
                before[rdma_lease_index_spec(before, h)->Some_0],
                after[j],
            )),
{
    reveal(rdma_leases_disjoint);
    let last = before.len() - 1;
    assert forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && rdma_lease_is_binding_like(
            before[rdma_lease_index_spec(before, h)->Some_0],
        ) implies (rdma_lease_index_spec(after, h) matches Some(j) && rdma_lease_binding_preserved(
        before[rdma_lease_index_spec(before, h)->Some_0],
        after[j],
    )) by {
        let i = rdma_lease_index_spec(before, h).unwrap();
        assert(i != idx);
        let target = if i == last {
            idx
        } else {
            i
        };
        assert(after[target] == before[i]);
        assert(after[target].mw_slot as usize == h.mw_slot && after[target].generation
            == h.generation);
        let j = rdma_lease_index_spec(after, h).unwrap();
        if j != target {
            if j < target {
                assert(after[j].mw_slot != after[target].mw_slot);
            } else {
                assert(after[target].mw_slot != after[j].mw_slot);
            }
        }
    }
}

/// Pure proof helper: removing a control WR from `before` leaves `rdma_data_events`
/// unchanged and preserves control-WR-ID bounds.
proof fn lemma_rdma_control_reap_window(
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
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
        ops: &mut [Option<PendingOperation>; BITMAP_SLOTS],
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
            ) && rdma_data_events(after) == rdma_data_events(before).remove(op.sequence)
                && final(core).ledger@.knows.set().contains(
                (old(core).node(), rdma_data_events(before)[op.sequence]),
            ),
            result matches PeerCompletion::Data(op, Ok(())) ==> rdma_data_events(
                before,
            ).contains_key(op.sequence) && final(core).ledger@.succeeded.contains(
                rdma_data_events(before)[op.sequence],
            ),
            result matches PeerCompletion::BindFailed { wr_id, .. } ==> !rdma_data_events(
                before,
            ).contains_key(wr_id) && rdma_data_events(after) == rdma_data_events(before),
            result matches PeerCompletion::Control(ctrl) ==> !rdma_data_events(before).contains_key(
                ctrl.id(),
            ) && rdma_data_events(after) == rdma_data_events(before),
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
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(self).completed_pulls == old(self).completed_pulls,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
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
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(rdma_error(e));
            },
            Ok(rdma::PollStatus::Empty) => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
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
            broadcast use vstd::map::group_map_lemmas;

            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
            );
            lemma_rdma_leases_same_frames(
                self.active_leases@,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
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
            assert(rdma::CqSnapshot::of(self.cq).pending == before_cq.pending.remove(c.id()));
            lemma_control_leases_match_cq_after_cq_shrink(
                self.active_leases@,
                &self.window_pool,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
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
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(self).completed_pulls == old(self).completed_pulls,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
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
                core.timelines.next_seq == before_core.timelines.next_seq,
                core.timelines.events == before_core.timelines.events,
                before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set()),
                before_core.ledger@.succeeded.dom().subset_of(core.ledger@.succeeded.dom()),
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
                broadcast use {vstd::iset::group_iset_lemmas, vstd::map::group_map_lemmas};

                assert(before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set()));
                assert(before_core.ledger@.succeeded.dom().subset_of(core.ledger@.succeeded.dom()));
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
            lemma_rdma_completion_prefix_submap(
                batch,
                batch.len() as int,
                before_cq,
                instance.id(),
            );
            assert(rdma::CqSnapshot::of(self.cq) == rdma::completion_prefix(
                batch,
                batch.len() as int,
                before_cq,
                instance.id(),
            ).0);
            lemma_control_leases_match_cq_after_cq_shrink(
                self.active_leases@,
                &self.window_pool,
                before_cq,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
            );
            lemma_rdma_leases_same_frames(
                self.active_leases@,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
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
            assert(Self::delivers_batch(batch, out@, old(self).cq, self.cq, instance.id()));
        }
        Ok((out, Ghost(batch)))
    }
}

#[verifier::opaque]
spec fn pump_cq_lease_post(before: Seq<ActiveRemoteLease>, after: Seq<ActiveRemoteLease>) -> bool {
    &&& forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && !(before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0]).has_active_wr() ==> (rdma_lease_index_spec(after, h) matches Some(j) && after[j]
            == before[rdma_lease_index_spec(before, h)->Some_0])
    &&& forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && rdma_lease_is_binding_like(
            before[rdma_lease_index_spec(before, h)->Some_0],
        ) ==> (rdma_lease_index_spec(after, h) matches Some(j) && rdma_lease_binding_preserved(
            before[rdma_lease_index_spec(before, h)->Some_0],
            after[j],
        ))
    &&& forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0].state == RemoteLeaseState::Unbinding && rdma_lease_index_spec(after, h) is Some
            && after[rdma_lease_index_spec(after, h)->Some_0].state == RemoteLeaseState::Unbinding
            ==> after[rdma_lease_index_spec(after, h)->Some_0] == before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0]
    &&& forall|h: RemoteLeaseHandle| #[trigger]
        rdma_lease_index_spec(before, h) is Some && before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0].state == RemoteLeaseState::Revoking && rdma_lease_index_spec(after, h) is Some
            && after[rdma_lease_index_spec(after, h)->Some_0].state == RemoteLeaseState::Revoking
            ==> after[rdma_lease_index_spec(after, h)->Some_0] == before[rdma_lease_index_spec(
            before,
            h,
        )->Some_0]
    &&& forall|j: int|
        0 <= j < after.len() && !(#[trigger] after[j]).is_unpinned() ==> exists|i: int|
            0 <= i < before.len() && !(#[trigger] before[i]).is_unpinned() && before[i].frame
                == after[j].frame && before[i].share@ == after[j].share@
}

proof fn lemma_pump_cq_lease_post_refl(leases: Seq<ActiveRemoteLease>)
    ensures
        pump_cq_lease_post(leases, leases),
{
    reveal(pump_cq_lease_post);
    assert forall|j: int|
        0 <= j < leases.len() && !(#[trigger] leases[j]).is_unpinned() implies exists|i: int|
        0 <= i < leases.len() && !(#[trigger] leases[i]).is_unpinned() && leases[i].frame
            == leases[j].frame && leases[i].share@ == leases[j].share@ by {
        assert(!leases[j].is_unpinned());
    }
}

proof fn lemma_pump_cq_lease_post_update(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        after.len() == before.len(),
        before[idx].has_active_wr(),
        rdma_lease_is_binding_like(before[idx]) ==> rdma_lease_binding_preserved(
            before[idx],
            after[idx],
        ),
        after[idx].state != RemoteLeaseState::Unbinding,
        after[idx].state != RemoteLeaseState::Revoking,
        !after[idx].is_unpinned() ==> !before[idx].is_unpinned(),
        after[idx].frame == before[idx].frame,
        after[idx].share@ == before[idx].share@,
        forall|j: int| 0 <= j < before.len() && j != idx ==> (#[trigger] after[j]) == before[j],
    ensures
        pump_cq_lease_post(before, after),
{
    lemma_nonactive_leases_after_update(before, after, idx);
    lemma_binding_leases_after_update(before, after, idx);
    lemma_unbinding_revoking_after_update(before, after, idx);
    reveal(pump_cq_lease_post);
    assert forall|j: int|
        0 <= j < after.len() && !(#[trigger] after[j]).is_unpinned() implies exists|i: int|
        0 <= i < before.len() && !(#[trigger] before[i]).is_unpinned() && before[i].frame
            == after[j].frame && before[i].share@ == after[j].share@ by {
        if j == idx {
            assert(!before[idx].is_unpinned());
        } else {
            assert(after[j] == before[j]);
            assert(!before[j].is_unpinned());
        }
    }
}

proof fn lemma_pump_cq_lease_post_swap_remove(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        rdma_leases_disjoint(before),
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        before[idx].has_active_wr(),
        !rdma_lease_is_binding_like(before[idx]),
        after.len() == before.len() - 1,
        forall|j: int|
            0 <= j < after.len() ==> (#[trigger] after[j]) == if j == idx {
                before[before.len() - 1]
            } else {
                before[j]
            },
    ensures
        pump_cq_lease_post(before, after),
{
    lemma_nonactive_leases_after_swap_remove(before, after, idx);
    lemma_binding_leases_after_swap_remove(before, after, idx);
    lemma_unbinding_revoking_after_swap_remove(before, after, idx);
    reveal(pump_cq_lease_post);
    assert forall|j: int|
        0 <= j < after.len() && !(#[trigger] after[j]).is_unpinned() implies exists|i: int|
        0 <= i < before.len() && !(#[trigger] before[i]).is_unpinned() && before[i].frame
            == after[j].frame && before[i].share@ == after[j].share@ by {
        let oi = if j == idx {
            before.len() - 1
        } else {
            j
        };
        assert(after[j] == before[oi]);
        assert(!before[oi].is_unpinned());
    }
}

#[verifier::opaque]
spec fn pump_cq_ring_post<'a>(
    old_s: &RdmaBackend<'a>,
    new_s: &RdmaBackend<'a>,
    wr_id: u64,
) -> bool {
    &&& forall|k: u64| #[trigger]
        new_s.pending().contains_key(k) ==> (old_s.pending().contains_key(k) && new_s.pending()[k]
            == old_s.pending()[k])
    &&& forall|k: u64| #[trigger]
        old_s.pending().contains_key(k) ==> ((new_s.pending().contains_key(k) && new_s.pending()[k]
            == old_s.pending()[k]) || new_s.completed_pull_spec(k) is Some)
    &&& forall|k: u64| #[trigger]
        old_s.completed_pull_spec(k) is Some && !old_s.pending().contains_key(k) && (forall|w: u64|
         #[trigger]
            old_s.pending().contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64)) != (k % (
            BITMAP_CAPACITY as u64))) ==> new_s.completed_pull_spec(k) == old_s.completed_pull_spec(
            k,
        )
    &&& forall|k: u64|
        k != wr_id && #[trigger] old_s.pending().contains_key(k) ==> (new_s.pending().contains_key(
            k,
        ) && new_s.pending()[k] == old_s.pending()[k])
    &&& (old_s.pending().contains_key(wr_id) ==> {
        &&& new_s.completed_pull_spec(wr_id) is Some
        &&& forall|q: u64|
            (q % (BITMAP_CAPACITY as u64)) != (wr_id % (BITMAP_CAPACITY as u64))
                ==> new_s.completed_pull_spec(q) == old_s.completed_pull_spec(q)
    })
    &&& (!old_s.pending().contains_key(wr_id) ==> new_s.completed_pulls == old_s.completed_pulls)
}

proof fn lemma_pump_cq_ring_post_data<'a>(
    old_s: &RdmaBackend<'a>,
    new_s: &RdmaBackend<'a>,
    seq: u64,
    st: Result<(), i32>,
)
    requires
        old_s.pending().contains_key(seq),
        new_s.pending() == old_s.pending().remove(seq),
        new_s.completed_pulls[(seq % BITMAP_CAPACITY) as int] == Some((seq, st)),
        forall|i: int|
            0 <= i < BITMAP_CAPACITY && i != (seq % BITMAP_CAPACITY) as int ==> (
            #[trigger] new_s.completed_pulls[i]) == old_s.completed_pulls[i],
    ensures
        pump_cq_ring_post(old_s, new_s, seq),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(pump_cq_ring_post);
    assert(new_s.completed_pull_spec(seq) is Some);
    assert forall|k: u64| #[trigger]
        old_s.completed_pull_spec(k) is Some && !old_s.pending().contains_key(k) && (forall|w: u64|
         #[trigger]
            old_s.pending().contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64)) != (k % (
            BITMAP_CAPACITY as u64))) implies new_s.completed_pull_spec(k)
        == old_s.completed_pull_spec(k) by {
        assert(seq != k);
        assert((seq % (BITMAP_CAPACITY as u64)) != (k % (BITMAP_CAPACITY as u64)));
        let ks = (k % BITMAP_CAPACITY) as int;
        assert(ks != (seq % BITMAP_CAPACITY) as int);
        assert(new_s.completed_pulls[ks] == old_s.completed_pulls[ks]);
    }
    assert forall|q: u64|
        (q % (BITMAP_CAPACITY as u64)) != (seq % (
        BITMAP_CAPACITY as u64)) implies new_s.completed_pull_spec(q) == old_s.completed_pull_spec(
        q,
    ) by {
        let qs = (q % BITMAP_CAPACITY) as int;
        assert(qs != (seq % BITMAP_CAPACITY) as int);
        assert(new_s.completed_pulls[qs] == old_s.completed_pulls[qs]);
    }
}

proof fn lemma_pump_cq_ring_post_control<'a>(
    old_s: &RdmaBackend<'a>,
    new_s: &RdmaBackend<'a>,
    wr_id: u64,
)
    requires
        !old_s.pending().contains_key(wr_id),
        new_s.pending() == old_s.pending(),
        new_s.completed_pulls == old_s.completed_pulls,
    ensures
        pump_cq_ring_post(old_s, new_s, wr_id),
{
    reveal(pump_cq_ring_post);
}

proof fn lemma_pump_cq_export_posts<'a>(
    old_s: &RdmaBackend<'a>,
    new_s: &RdmaBackend<'a>,
    wr_id: u64,
)
    requires
        pump_cq_lease_post(old_s.active_leases@, new_s.active_leases@),
        pump_cq_ring_post(old_s, new_s, wr_id),
    ensures
        forall|k: u64| #[trigger]
            new_s.pending().contains_key(k) ==> (old_s.pending().contains_key(k)
                && new_s.pending()[k] == old_s.pending()[k]),
        forall|k: u64| #[trigger]
            old_s.pending().contains_key(k) ==> ((new_s.pending().contains_key(k)
                && new_s.pending()[k] == old_s.pending()[k]) || new_s.completed_pull_spec(
                k,
            ) is Some),
        forall|k: u64| #[trigger]
            old_s.completed_pull_spec(k) is Some && !old_s.pending().contains_key(k) && (forall|
                w: u64,
            | #[trigger]
                old_s.pending().contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64)) != (k
                    % (BITMAP_CAPACITY as u64))) ==> new_s.completed_pull_spec(k)
                == old_s.completed_pull_spec(k),
        forall|h: RemoteLeaseHandle| #[trigger]
            old_s.lease_index_spec(h) is Some && !(old_s.active_leases@[old_s.lease_index_spec(
                h,
            )->Some_0]).has_active_wr() ==> (new_s.lease_index_spec(h) matches Some(j)
                && new_s.active_leases@[j] == old_s.active_leases@[old_s.lease_index_spec(
                h,
            )->Some_0]),
        forall|h: RemoteLeaseHandle| #[trigger]
            old_s.lease_index_spec(h) is Some && rdma_lease_is_binding_like(
                old_s.active_leases@[old_s.lease_index_spec(h)->Some_0],
            ) ==> (new_s.lease_index_spec(h) matches Some(j) && rdma_lease_binding_preserved(
                old_s.active_leases@[old_s.lease_index_spec(h)->Some_0],
                new_s.active_leases@[j],
            )),
        forall|h: RemoteLeaseHandle| #[trigger]
            old_s.lease_index_spec(h) is Some && old_s.active_leases@[old_s.lease_index_spec(
                h,
            )->Some_0].state == RemoteLeaseState::Unbinding && new_s.lease_index_spec(h) is Some
                && new_s.active_leases@[new_s.lease_index_spec(h)->Some_0].state
                == RemoteLeaseState::Unbinding ==> new_s.active_leases@[new_s.lease_index_spec(
                h,
            )->Some_0] == old_s.active_leases@[old_s.lease_index_spec(h)->Some_0],
        forall|h: RemoteLeaseHandle| #[trigger]
            old_s.lease_index_spec(h) is Some && old_s.active_leases@[old_s.lease_index_spec(
                h,
            )->Some_0].state == RemoteLeaseState::Revoking && new_s.lease_index_spec(h) is Some
                && new_s.active_leases@[new_s.lease_index_spec(h)->Some_0].state
                == RemoteLeaseState::Revoking ==> new_s.active_leases@[new_s.lease_index_spec(
                h,
            )->Some_0] == old_s.active_leases@[old_s.lease_index_spec(h)->Some_0],
        forall|k: u64|
            k != wr_id && #[trigger] old_s.pending().contains_key(k) ==> (
            new_s.pending().contains_key(k) && new_s.pending()[k] == old_s.pending()[k]),
        old_s.pending().contains_key(wr_id) ==> {
            &&& new_s.completed_pull_spec(wr_id) is Some
            &&& forall|q: u64|
                (q % (BITMAP_CAPACITY as u64)) != (wr_id % (BITMAP_CAPACITY as u64))
                    ==> new_s.completed_pull_spec(q) == old_s.completed_pull_spec(q)
        },
        !old_s.pending().contains_key(wr_id) ==> new_s.completed_pulls == old_s.completed_pulls,
        forall|j: int|
            0 <= j < new_s.active_leases@.len() && !(
            #[trigger] new_s.active_leases@[j]).is_unpinned() ==> exists|i: int|
                0 <= i < old_s.active_leases@.len() && !(
                #[trigger] old_s.active_leases@[i]).is_unpinned() && old_s.active_leases@[i].frame
                    == new_s.active_leases@[j].frame && old_s.active_leases@[i].share@
                    == new_s.active_leases@[j].share@,
{
    reveal(pump_cq_lease_post);
    reveal(pump_cq_ring_post);
}

impl<'a> RdmaBackend<'a> {
    fn find_control_lease_by_wr_id(&self, wr_id: u64) -> (found: Option<usize>)
        ensures
            found matches Some(idx) ==> idx < self.active_leases@.len()
                && self.active_leases@[idx as int].wr_id == wr_id
                && self.active_leases@[idx as int].has_active_wr(),
    {
        let mut i: usize = 0;
        while i < self.active_leases.len()
            invariant
                i <= self.active_leases@.len(),
            decreases self.active_leases.len() - i,
        {
            let l = &self.active_leases[i];
            if l.wr_id == wr_id
                && matches!(
                    l.state,
                    RemoteLeaseState::Binding | RemoteLeaseState::Unbinding
                        | RemoteLeaseState::Revoking
                ) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn apply_bind_failed_cqe(
        &mut self,
        core: &CacheCore,
        wr_id: u64,
        status: i32,
        witness: Tracked<rdma::BindFailedWitness>,
        Ghost(before_cq): Ghost<rdma::CqSnapshot>,
        Ghost(after_cq): Ghost<rdma::CqSnapshot>,
    )
        requires
            core.well_formed(),
            before_cq.observer == core.node(),
            after_cq == rdma::CqSnapshot::of(old(self).cq),
            after_cq.pending == before_cq.pending.remove(wr_id),
            before_cq.pending.contains_key(wr_id),
            before_cq.pending[wr_id].kind is BindWindow,
            forall|w: rdma::MemoryWindow|
                w.state() is Binding && w.state()->Binding_wr_id == wr_id && w.id()
                    == before_cq.pending[wr_id].kind->BindWindow_window_id && w.node()
                    == before_cq.observer ==> (#[trigger] witness@.matches(w)),
            old(self).timeline_id < core.timelines.count,
            core.timelines.kinds[old(self).timeline_id as int] == (TimelineKind::RdmaQueuePair {
                peer_node: old(self).peer_node,
            }),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.state() is ReadyToSend ==> old(self).qp.peer() == old(self).peer_node
                && old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().id() == core.instance_id(),
            old(self).qp.source().node() == core.node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == core.node(),
            old(self).cq.infos().instance_id() == core.instance_id(),
            old(self).window_pool.well_formed(),
            old(self).window_pool.node() == core.node(),
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &core.timelines,
                old(self).timeline_id,
                &core.ledger@,
                after_cq,
                core.instance_id(),
                old(self).next_control_wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &core.timelines,
                old(self).timeline_id,
                old(self).cq.observer(),
                &core.ledger@,
            ),
            rdma_leases_disjoint(old(self).active_leases@),
            rdma_leases_valid(
                old(self).active_leases@,
                &old(self).window_pool,
                &old(self).qp,
                core,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                before_cq,
                old(self).next_control_wr_id,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                after_cq,
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &old(self).window_pool),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool == old(self).window_pool,
            final(self).completed_pulls == old(self).completed_pulls,
            pump_cq_lease_post(old(self).active_leases@, final(self).active_leases@),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(ActiveRemoteLease::valid);
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        if let Some(idx) = self.find_control_lease_by_wr_id(wr_id) {
            proof {
                lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, idx as int);
                assert(before_leases[idx as int].state == RemoteLeaseState::Binding
                    && witness@.matches(
                    before_pool.slots[before_leases[idx as int].mw_slot as int].window->Some_0,
                )) by {
                    reveal(ActiveRemoteLease::valid);
                    reveal(window_pool_slots_wf);
                    reveal(rdma_control_leases_match_cq);
                };
            }
            self.active_leases[idx].bind_failed_witness = Some(witness);
            self.active_leases[idx].staged_error = Some(status);
            self.active_leases[idx].state = RemoteLeaseState::BindFailedReady;
            proof {
                assert(self.active_leases@[idx as int].valid(&self.window_pool, &self.qp, core))
                    by {
                    reveal(ActiveRemoteLease::valid);
                };
                assert(core.frames.same_directory(&core.frames)) by {
                    reveal(FrameTable::same_directory);
                };
                assert(core.frames.same_other_pins(
                    &core.frames,
                    Some(before_leases[idx as int].frame),
                )) by {
                    reveal(FrameTable::same_other_pins);
                };
                lemma_rdma_leases_update_in_place(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                    &self.qp,
                    core,
                    core,
                    None,
                );
                lemma_control_leases_match_cq_after_cq_remove_update(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                    before_cq,
                    after_cq,
                    self.next_control_wr_id,
                    Some(wr_id),
                );
                lemma_pool_slots_owned_after_update_in_place(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                );
                assert(rdma_leases_disjoint(self.active_leases@)) by {
                    reveal(rdma_leases_valid);
                };
                lemma_pump_cq_lease_post_update(before_leases, self.active_leases@, idx as int);
            }
        } else {
            proof {
                lemma_pump_cq_lease_post_refl(before_leases);
            }
        }
        proof {
            assert(self.well_formed(core)) by {
                reveal(RdmaBackend::well_formed);
            };
        }
    }

    fn apply_window_bound_cqe(
        &mut self,
        core: &CacheCore,
        wr_id: u64,
        witness: Tracked<rdma::BindWitness>,
        Ghost(before_cq): Ghost<rdma::CqSnapshot>,
        Ghost(after_cq): Ghost<rdma::CqSnapshot>,
    )
        requires
            core.well_formed(),
            before_cq.observer == core.node(),
            after_cq == rdma::CqSnapshot::of(old(self).cq),
            after_cq.pending == before_cq.pending.remove(wr_id),
            before_cq.pending.contains_key(wr_id),
            before_cq.pending[wr_id].kind is BindWindow,
            forall|w: rdma::MemoryWindow|
                w.state() is Binding && w.state()->Binding_wr_id == wr_id && w.id()
                    == before_cq.pending[wr_id].kind->BindWindow_window_id && w.node()
                    == before_cq.observer ==> (#[trigger] witness@.matches(w)),
            old(self).timeline_id < core.timelines.count,
            core.timelines.kinds[old(self).timeline_id as int] == (TimelineKind::RdmaQueuePair {
                peer_node: old(self).peer_node,
            }),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.state() is ReadyToSend ==> old(self).qp.peer() == old(self).peer_node
                && old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().id() == core.instance_id(),
            old(self).qp.source().node() == core.node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == core.node(),
            old(self).cq.infos().instance_id() == core.instance_id(),
            old(self).window_pool.well_formed(),
            old(self).window_pool.node() == core.node(),
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &core.timelines,
                old(self).timeline_id,
                &core.ledger@,
                after_cq,
                core.instance_id(),
                old(self).next_control_wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &core.timelines,
                old(self).timeline_id,
                old(self).cq.observer(),
                &core.ledger@,
            ),
            rdma_leases_disjoint(old(self).active_leases@),
            rdma_leases_valid(
                old(self).active_leases@,
                &old(self).window_pool,
                &old(self).qp,
                core,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                before_cq,
                old(self).next_control_wr_id,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                after_cq,
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &old(self).window_pool),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool == old(self).window_pool,
            final(self).completed_pulls == old(self).completed_pulls,
            pump_cq_lease_post(old(self).active_leases@, final(self).active_leases@),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(ActiveRemoteLease::valid);
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        if let Some(idx) = self.find_control_lease_by_wr_id(wr_id) {
            proof {
                lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, idx as int);
                assert(before_leases[idx as int].state == RemoteLeaseState::Binding
                    && witness@.matches(
                    before_pool.slots[before_leases[idx as int].mw_slot as int].window->Some_0,
                )) by {
                    reveal(ActiveRemoteLease::valid);
                    reveal(window_pool_slots_wf);
                    reveal(rdma_control_leases_match_cq);
                };
            }
            self.active_leases[idx].bind_witness = Some(witness);
            self.active_leases[idx].state = RemoteLeaseState::BindReady;
            proof {
                assert(self.active_leases@[idx as int].valid(&self.window_pool, &self.qp, core))
                    by {
                    reveal(ActiveRemoteLease::valid);
                };
                assert(core.frames.same_directory(&core.frames)) by {
                    reveal(FrameTable::same_directory);
                };
                assert(core.frames.same_other_pins(
                    &core.frames,
                    Some(before_leases[idx as int].frame),
                )) by {
                    reveal(FrameTable::same_other_pins);
                };
                lemma_rdma_leases_update_in_place(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                    &self.qp,
                    core,
                    core,
                    None,
                );
                lemma_control_leases_match_cq_after_cq_remove_update(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                    before_cq,
                    after_cq,
                    self.next_control_wr_id,
                    Some(wr_id),
                );
                lemma_pool_slots_owned_after_update_in_place(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                );
                assert(rdma_leases_disjoint(self.active_leases@)) by {
                    reveal(rdma_leases_valid);
                };
                lemma_pump_cq_lease_post_update(before_leases, self.active_leases@, idx as int);
            }
        } else {
            proof {
                lemma_pump_cq_lease_post_refl(before_leases);
            }
        }
        proof {
            assert(self.well_formed(core)) by {
                reveal(RdmaBackend::well_formed);
            };
        }
    }

    fn apply_window_invalidated_cqe(
        &mut self,
        core: &CacheCore,
        wr_id: u64,
        witness: Tracked<rdma::InvalidateWitness>,
        Ghost(before_cq): Ghost<rdma::CqSnapshot>,
        Ghost(after_cq): Ghost<rdma::CqSnapshot>,
    )
        requires
            core.well_formed(),
            before_cq.observer == core.node(),
            after_cq == rdma::CqSnapshot::of(old(self).cq),
            after_cq.pending == before_cq.pending.remove(wr_id),
            before_cq.pending.contains_key(wr_id),
            before_cq.pending[wr_id].kind is LocalInvalidate,
            forall|w: rdma::MemoryWindow|
                w.state() is Invalidating && w.state()->Invalidating_wr_id == wr_id && w.id()
                    == before_cq.pending[wr_id].kind->LocalInvalidate_window_id && w.node()
                    == before_cq.observer ==> (#[trigger] witness@.matches(w)),
            old(self).timeline_id < core.timelines.count,
            core.timelines.kinds[old(self).timeline_id as int] == (TimelineKind::RdmaQueuePair {
                peer_node: old(self).peer_node,
            }),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.state() is ReadyToSend ==> old(self).qp.peer() == old(self).peer_node
                && old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().id() == core.instance_id(),
            old(self).qp.source().node() == core.node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == core.node(),
            old(self).cq.infos().instance_id() == core.instance_id(),
            old(self).window_pool.well_formed(),
            old(self).window_pool.node() == core.node(),
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &core.timelines,
                old(self).timeline_id,
                &core.ledger@,
                after_cq,
                core.instance_id(),
                old(self).next_control_wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &core.timelines,
                old(self).timeline_id,
                old(self).cq.observer(),
                &core.ledger@,
            ),
            rdma_leases_disjoint(old(self).active_leases@),
            rdma_leases_valid(
                old(self).active_leases@,
                &old(self).window_pool,
                &old(self).qp,
                core,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                before_cq,
                old(self).next_control_wr_id,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                after_cq,
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &old(self).window_pool),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).completed_pulls == old(self).completed_pulls,
            pump_cq_lease_post(old(self).active_leases@, final(self).active_leases@),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(ActiveRemoteLease::valid);
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        if let Some(idx) = self.find_control_lease_by_wr_id(wr_id) {
            proof {
                lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, idx as int);
                assert((before_leases[idx as int].state == RemoteLeaseState::Unbinding
                    || before_leases[idx as int].state == RemoteLeaseState::Revoking)
                    && witness@.matches(
                    before_pool.slots[before_leases[idx as int].mw_slot as int].window->Some_0,
                )) by {
                    reveal(ActiveRemoteLease::valid);
                    reveal(window_pool_slots_wf);
                    reveal(rdma_control_leases_match_cq);
                };
            }
            if matches!(self.active_leases[idx].state, RemoteLeaseState::Unbinding) {
                proof {
                    reveal(ActiveRemoteLease::valid);
                }
                let lease = self.active_leases.swap_remove(idx);
                let Tracked(returned) = lease.returned_share.unwrap();
                self.window_pool.finish_invalidate(
                    lease.mw_slot,
                    lease.generation,
                    witness,
                    Tracked(Some(returned)),
                );
                proof {
                    assert(core.frames.same_directory(&core.frames)) by {
                        reveal(FrameTable::same_directory);
                    };
                    assert(core.frames.same_other_pins(&core.frames, Some(lease.frame))) by {
                        reveal(FrameTable::same_other_pins);
                    };
                    lemma_rdma_leases_swap_remove(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        &self.qp,
                        core,
                        core,
                        None,
                    );
                    lemma_control_leases_match_cq_after_swap_remove(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                        &before_pool,
                        &self.window_pool,
                        before_cq,
                        after_cq,
                        self.next_control_wr_id,
                        Some(wr_id),
                    );
                    lemma_pool_slots_owned_after_swap_remove(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                        &before_pool,
                        &self.window_pool,
                        lease.mw_slot,
                    );
                    assert(rdma_leases_disjoint(self.active_leases@)) by {
                        reveal(rdma_leases_valid);
                    };
                    lemma_rdma_all_slabs_same_directory(
                        &self.host_mr,
                        &self.gpu_mr,
                        self.shared_host_mr,
                        self.shared_gpu_mr,
                        core,
                        core,
                        self.window_pool.pd(),
                    );
                    lemma_pump_cq_lease_post_swap_remove(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                    );
                }
            } else {
                self.active_leases[idx].inv_witness = Some(witness);
                self.active_leases[idx].state = RemoteLeaseState::RevokeInvReady;
                proof {
                    assert(self.active_leases@[idx as int].valid(&self.window_pool, &self.qp, core))
                        by {
                        reveal(ActiveRemoteLease::valid);
                    };
                    assert(core.frames.same_directory(&core.frames)) by {
                        reveal(FrameTable::same_directory);
                    };
                    assert(core.frames.same_other_pins(
                        &core.frames,
                        Some(before_leases[idx as int].frame),
                    )) by {
                        reveal(FrameTable::same_other_pins);
                    };
                    lemma_rdma_leases_update_in_place(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                        &before_pool,
                        &self.window_pool,
                        &self.qp,
                        core,
                        core,
                        None,
                    );
                    lemma_control_leases_match_cq_after_cq_remove_update(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                        &before_pool,
                        &self.window_pool,
                        before_cq,
                        after_cq,
                        self.next_control_wr_id,
                        Some(wr_id),
                    );
                    lemma_pool_slots_owned_after_update_in_place(
                        before_leases,
                        self.active_leases@,
                        idx as int,
                        &before_pool,
                        &self.window_pool,
                    );
                    assert(rdma_leases_disjoint(self.active_leases@)) by {
                        reveal(rdma_leases_valid);
                    };
                    lemma_pump_cq_lease_post_update(before_leases, self.active_leases@, idx as int);
                }
            }
        } else {
            proof {
                lemma_pump_cq_lease_post_refl(before_leases);
            }
        }
        proof {
            assert(self.well_formed(core)) by {
                reveal(RdmaBackend::well_formed);
            };
        }
    }

    fn apply_control_failed_cqe(
        &mut self,
        core: &CacheCore,
        wr_id: u64,
        status: i32,
        Ghost(before_cq): Ghost<rdma::CqSnapshot>,
        Ghost(after_cq): Ghost<rdma::CqSnapshot>,
    )
        requires
            core.well_formed(),
            after_cq == rdma::CqSnapshot::of(old(self).cq),
            after_cq.pending == before_cq.pending.remove(wr_id),
            before_cq.pending.contains_key(wr_id),
            old(self).timeline_id < core.timelines.count,
            core.timelines.kinds[old(self).timeline_id as int] == (TimelineKind::RdmaQueuePair {
                peer_node: old(self).peer_node,
            }),
            old(self).qp.cq_id() == old(self).cq.id(),
            old(self).qp.state() is ReadyToSend ==> old(self).qp.peer() == old(self).peer_node
                && old(self).qp.pd_id() == old(self).window_pool.pd(),
            old(self).qp.source().id() == core.instance_id(),
            old(self).qp.source().node() == core.node(),
            old(self).cq.well_formed(),
            old(self).cq.observer() == core.node(),
            old(self).cq.infos().instance_id() == core.instance_id(),
            old(self).window_pool.well_formed(),
            old(self).window_pool.node() == core.node(),
            !(old(self).host_mr is Some && old(self).shared_host_mr is Some),
            !(old(self).gpu_mr is Some && old(self).shared_gpu_mr is Some),
            old(self).host_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).gpu_mr matches Some(ref s) ==> s.well_formed(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_host_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            old(self).shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                core,
                old(self).window_pool.pd(),
            ),
            rdma_window_wf(
                old(self).in_flight_ops,
                old(self).bitmap,
                &core.timelines,
                old(self).timeline_id,
                &core.ledger@,
                after_cq,
                core.instance_id(),
                old(self).next_control_wr_id,
            ),
            completed_pulls_wf(
                old(self).completed_pulls,
                &core.timelines,
                old(self).timeline_id,
                old(self).cq.observer(),
                &core.ledger@,
            ),
            rdma_leases_disjoint(old(self).active_leases@),
            rdma_leases_valid(
                old(self).active_leases@,
                &old(self).window_pool,
                &old(self).qp,
                core,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                before_cq,
                old(self).next_control_wr_id,
            ),
            rdma_control_leases_match_cq(
                old(self).active_leases@,
                &old(self).window_pool,
                after_cq,
                old(self).next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(old(self).active_leases@, &old(self).window_pool),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool == old(self).window_pool,
            final(self).completed_pulls == old(self).completed_pulls,
            pump_cq_lease_post(old(self).active_leases@, final(self).active_leases@),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(ActiveRemoteLease::valid);
        let ghost before_leases = self.active_leases@;
        let ghost before_pool = self.window_pool;
        if let Some(idx) = self.find_control_lease_by_wr_id(wr_id) {
            proof {
                lemma_rdma_lease_extract(before_leases, &before_pool, &self.qp, core, idx as int);
            }
            self.active_leases[idx].staged_error = Some(status);
            self.active_leases[idx].state = RemoteLeaseState::ControlFailedState;
            proof {
                assert(self.active_leases@[idx as int].valid(&self.window_pool, &self.qp, core) && (
                !self.active_leases@[idx as int].is_unpinned()
                    ==> !before_leases[idx as int].is_unpinned()) && (rdma_lease_is_binding_like(
                    before_leases[idx as int],
                ) ==> rdma_lease_binding_preserved(
                    before_leases[idx as int],
                    self.active_leases@[idx as int],
                ))) by {
                    reveal(ActiveRemoteLease::valid);
                };
                assert(core.frames.same_directory(&core.frames)) by {
                    reveal(FrameTable::same_directory);
                };
                assert(core.frames.same_other_pins(
                    &core.frames,
                    Some(before_leases[idx as int].frame),
                )) by {
                    reveal(FrameTable::same_other_pins);
                };
                lemma_rdma_leases_update_in_place(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                    &self.qp,
                    core,
                    core,
                    None,
                );
                lemma_control_leases_match_cq_after_cq_remove_update(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                    before_cq,
                    after_cq,
                    self.next_control_wr_id,
                    Some(wr_id),
                );
                lemma_pool_slots_owned_after_update_in_place(
                    before_leases,
                    self.active_leases@,
                    idx as int,
                    &before_pool,
                    &self.window_pool,
                );
                assert(rdma_leases_disjoint(self.active_leases@)) by {
                    reveal(rdma_leases_valid);
                };
                lemma_pump_cq_lease_post_update(before_leases, self.active_leases@, idx as int);
            }
        } else {
            proof {
                lemma_pump_cq_lease_post_refl(before_leases);
            }
        }
        proof {
            assert(self.well_formed(core)) by {
                reveal(RdmaBackend::well_formed);
            };
        }
    }

    proof fn lemma_pump_cq_entry_facts(&self, core: &CacheCore)
        requires
            self.well_formed(core),
        ensures
            self.timeline_id < core.timelines.count,
            core.timelines.kinds[self.timeline_id as int] == (TimelineKind::RdmaQueuePair {
                peer_node: self.peer_node,
            }),
            self.qp.cq_id() == self.cq.id(),
            self.qp.state() is ReadyToSend ==> self.qp.peer() == self.peer_node && self.qp.pd_id()
                == self.window_pool.pd(),
            self.qp.source().id() == core.instance_id(),
            self.qp.source().node() == core.node(),
            self.cq.well_formed(),
            self.cq.observer() == core.node(),
            self.cq.infos().instance_id() == core.instance_id(),
            self.window_pool.well_formed(),
            self.window_pool.node() == core.node(),
            !(self.host_mr is Some && self.shared_host_mr is Some),
            !(self.gpu_mr is Some && self.shared_gpu_mr is Some),
            self.host_mr matches Some(ref s) ==> s.well_formed(core, self.window_pool.pd()),
            self.gpu_mr matches Some(ref s) ==> s.well_formed(core, self.window_pool.pd()),
            self.shared_host_mr matches Some(s) ==> s.well_formed_shared(
                core,
                self.window_pool.pd(),
            ),
            self.shared_gpu_mr matches Some(s) ==> s.well_formed_shared(
                core,
                self.window_pool.pd(),
            ),
            rdma_window_wf(
                self.in_flight_ops,
                self.bitmap,
                &core.timelines,
                self.timeline_id,
                &core.ledger@,
                rdma::CqSnapshot::of(self.cq),
                core.instance_id(),
                self.next_control_wr_id,
            ),
            completed_pulls_wf(
                self.completed_pulls,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &core.ledger@,
            ),
            rdma_leases_disjoint(self.active_leases@),
            rdma_leases_valid(self.active_leases@, &self.window_pool, &self.qp, core),
            rdma_control_leases_match_cq(
                self.active_leases@,
                &self.window_pool,
                rdma::CqSnapshot::of(self.cq),
                self.next_control_wr_id,
            ),
            rdma_pool_slots_owned_by_leases(self.active_leases@, &self.window_pool),
    {
        hide(CacheCore::well_formed);
        reveal(RdmaBackend::well_formed);
    }
}

impl<'a> RdmaBackend<'a> {
    /// Non-dropping single-CQE router: polls at most one CQE from `self.cq`,
    /// routes data completions into `self.completed_pulls` and control completions
    /// directly onto the matching `ActiveRemoteLease` in `self.active_leases`
    /// (auto-finalizing `Unbinding` invalidations cleanly), never discarding
    /// any unrelated data or control CQE.
    #[verifier::rlimit(350)]
    pub fn pump_cq(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<Option<u64>>)
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
            peer_core_step(old(core), final(core), old(self).timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.events == old(core).timelines.events,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).ledger@.submitted == old(core).ledger@.submitted,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            old(core).ledger@.succeeded.dom().subset_of(final(core).ledger@.succeeded.dom()),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).cq.in_flight().dom().subset_of(old(self).cq.in_flight().dom()),
            final(self).cq.in_flight().len() <= old(self).cq.in_flight().len(),
            forall|k: u64| #[trigger]
                final(self).pending().contains_key(k) ==> (old(self).pending().contains_key(k)
                    && final(self).pending()[k] == old(self).pending()[k]),
            forall|k: u64| #[trigger]
                old(self).pending().contains_key(k) ==> ((final(self).pending().contains_key(k)
                    && final(self).pending()[k] == old(self).pending()[k])
                    || final(self).completed_pull_spec(k) is Some),
            forall|k: u64| #[trigger]
                old(self).completed_pull_spec(k) is Some && !old(self).pending().contains_key(k)
                    && (forall|w: u64| #[trigger]
                    old(self).pending().contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64))
                        != (k % (BITMAP_CAPACITY as u64))) ==> final(self).completed_pull_spec(k)
                    == old(self).completed_pull_spec(k),
            forall|h: RemoteLeaseHandle| #[trigger]
                old(self).lease_index_spec(h) is Some && !(old(self).active_leases@[old(
                    self,
                ).lease_index_spec(h)->Some_0]).has_active_wr() ==> (final(self).lease_index_spec(
                    h,
                ) matches Some(j) && final(self).active_leases@[j] == old(self).active_leases@[old(
                    self,
                ).lease_index_spec(h)->Some_0]),
            forall|h: RemoteLeaseHandle| #[trigger]
                old(self).lease_index_spec(h) is Some && rdma_lease_is_binding_like(
                    old(self).active_leases@[old(self).lease_index_spec(h)->Some_0],
                ) ==> (final(self).lease_index_spec(h) matches Some(j)
                    && rdma_lease_binding_preserved(
                    old(self).active_leases@[old(self).lease_index_spec(h)->Some_0],
                    final(self).active_leases@[j],
                )),
            forall|h: RemoteLeaseHandle| #[trigger]
                old(self).lease_index_spec(h) is Some && old(self).active_leases@[old(
                    self,
                ).lease_index_spec(h)->Some_0].state == RemoteLeaseState::Unbinding
                    && final(self).lease_index_spec(h) is Some
                    && final(self).active_leases@[final(self).lease_index_spec(h)->Some_0].state
                    == RemoteLeaseState::Unbinding
                    ==> final(self).active_leases@[final(self).lease_index_spec(h)->Some_0] == old(
                    self,
                ).active_leases@[old(self).lease_index_spec(h)->Some_0],
            forall|h: RemoteLeaseHandle| #[trigger]
                old(self).lease_index_spec(h) is Some && old(self).active_leases@[old(
                    self,
                ).lease_index_spec(h)->Some_0].state == RemoteLeaseState::Revoking
                    && final(self).lease_index_spec(h) is Some
                    && final(self).active_leases@[final(self).lease_index_spec(h)->Some_0].state
                    == RemoteLeaseState::Revoking
                    ==> final(self).active_leases@[final(self).lease_index_spec(h)->Some_0] == old(
                    self,
                ).active_leases@[old(self).lease_index_spec(h)->Some_0],
            r is Err ==> {
                &&& *final(core) == *old(core)
                &&& *final(self) == *old(self)
            },
            r matches Ok(None) ==> {
                &&& *final(core) == *old(core)
                &&& *final(self) == *old(self)
            },
            r matches Ok(Some(wr_id)) ==> {
                &&& old(self).cq.in_flight().contains_key(wr_id)
                &&& !final(self).cq.in_flight().contains_key(wr_id)
                &&& final(self).cq.in_flight().len() < old(self).cq.in_flight().len()
                &&& forall|k: u64|
                    k != wr_id && #[trigger] old(self).pending().contains_key(k) ==> (
                    final(self).pending().contains_key(k) && final(self).pending()[k] == old(
                        self,
                    ).pending()[k])
                &&& (old(self).pending().contains_key(wr_id) ==> {
                    &&& final(self).completed_pull_spec(wr_id) is Some
                    &&& forall|q: u64|
                        (q % (BITMAP_CAPACITY as u64)) != (wr_id % (BITMAP_CAPACITY as u64))
                            ==> final(self).completed_pull_spec(q) == old(self).completed_pull_spec(
                            q,
                        )
                })
                &&& (!old(self).pending().contains_key(wr_id) ==> final(self).completed_pulls
                    == old(self).completed_pulls)
            },
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(ActiveRemoteLease::valid);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            self.lemma_pump_cq_entry_facts(core);
        }
        let ghost before_core = *core;
        let ghost before_cq = rdma::CqSnapshot::of(self.cq);
        let ghost before_leases = self.active_leases@;
        let ghost before_pulls = self.completed_pulls;
        let c = match rdma::poll_completion_queue(&mut self.cq, Tracked(instance)) {
            Err(e) => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Err(rdma_error(e));
            },
            Ok(rdma::PollStatus::Empty) => {
                proof {
                    lemma_peer_core_step_refl(core, self.timeline_id);
                    lemma_foreign_shares_refl(core, self.active_leases@);
                }
                return Ok(None);
            },
            Ok(rdma::PollStatus::Completed(c)) => c,
        };
        let ghost after_cq = rdma::CqSnapshot::of(self.cq);
        proof {
            broadcast use vstd::map::group_map_lemmas;

            assert(after_cq.pending == before_cq.pending.remove(c.id()));
            assert(after_cq.pending.dom().subset_of(before_cq.pending.dom()));
            assert(after_cq.pending.len() < before_cq.pending.len());
        }
        let result = Self::dispatch_one(
            &mut self.in_flight_ops,
            &mut self.bitmap,
            core,
            self.timeline_id,
            self.next_control_wr_id,
            c,
            Ghost(before_cq),
            Ghost(after_cq),
        );
        proof {
            lemma_rdma_all_slabs_same_directory(
                &self.host_mr,
                &self.gpu_mr,
                self.shared_host_mr,
                self.shared_gpu_mr,
                &before_core,
                core,
                self.window_pool.pd(),
            );
            lemma_completed_pulls_after_step(
                self.completed_pulls,
                &before_core.timelines,
                &core.timelines,
                self.timeline_id,
                self.cq.observer(),
                &before_core.ledger@,
                &core.ledger@,
            );
            lemma_rdma_leases_same_frames(
                self.active_leases@,
                &self.window_pool,
                &self.qp,
                &before_core,
                core,
            );
            lemma_control_leases_match_cq_after_cq_shrink(
                self.active_leases@,
                &self.window_pool,
                before_cq,
                after_cq,
                self.next_control_wr_id,
            );
        }
        let out_wr_id: u64 = match result {
            PeerCompletion::Data(op, st) => {
                let seq = op.sequence;
                let slot = (seq % BITMAP_CAPACITY) as usize;
                self.completed_pulls[slot] = Some((seq, st));
                proof {
                    assert(before_cq.pending.contains_key(seq));
                    assert(before_core.timelines.issued(self.timeline_id, seq)) by {
                        reveal(rdma_window_wf);
                    };
                    assert(core.timelines.issued(self.timeline_id, seq));
                    assert(core.timelines.events@[(self.timeline_id, seq)] == rdma_data_events(
                        before_cq,
                    )[seq]) by {
                        reveal(rdma_window_wf);
                    };
                    assert(completed_pulls_wf(
                        self.completed_pulls,
                        &core.timelines,
                        self.timeline_id,
                        self.cq.observer(),
                        &core.ledger@,
                    ));
                    assert forall|k: int| 0 <= k < BITMAP_CAPACITY && k != slot as int implies (
                    #[trigger] self.completed_pulls[k]) == before_pulls[k] by {}
                    assert(self.well_formed(core)) by {
                        reveal(RdmaBackend::well_formed);
                    };
                    lemma_pump_cq_lease_post_refl(before_leases);
                    lemma_pump_cq_ring_post_data(&old(self), &*self, seq, st);
                }
                seq
            },
            PeerCompletion::BindFailed { wr_id, status, witness } => {
                self.apply_bind_failed_cqe(
                    core,
                    wr_id,
                    status,
                    witness,
                    Ghost(before_cq),
                    Ghost(after_cq),
                );
                proof {
                    lemma_pump_cq_ring_post_control(&old(self), &*self, wr_id);
                }
                wr_id
            },
            PeerCompletion::Control(c_ctrl) => {
                match c_ctrl {
                    rdma::Completed::WindowBound { wr_id, witness } => {
                        self.apply_window_bound_cqe(
                            core,
                            wr_id,
                            witness,
                            Ghost(before_cq),
                            Ghost(after_cq),
                        );
                        proof {
                            lemma_pump_cq_ring_post_control(&old(self), &*self, wr_id);
                        }
                        wr_id
                    },
                    rdma::Completed::WindowInvalidated { wr_id, witness } => {
                        self.apply_window_invalidated_cqe(
                            core,
                            wr_id,
                            witness,
                            Ghost(before_cq),
                            Ghost(after_cq),
                        );
                        proof {
                            lemma_pump_cq_ring_post_control(&old(self), &*self, wr_id);
                        }
                        wr_id
                    },
                    rdma::Completed::ControlFailed { wr_id, status } => {
                        self.apply_control_failed_cqe(
                            core,
                            wr_id,
                            status,
                            Ghost(before_cq),
                            Ghost(after_cq),
                        );
                        proof {
                            lemma_pump_cq_ring_post_control(&old(self), &*self, wr_id);
                        }
                        wr_id
                    },
                    _ => {
                        assert(false);
                        0
                    },
                }
            },
        };
        proof {
            lemma_pump_cq_export_posts(&old(self), &*self, out_wr_id);
            lemma_foreign_shares_same_pins(&before_core, core, before_leases, self.active_leases@);
        }
        Ok(Some(out_wr_id))
    }

    /// Drain all currently available CQ completions by repeatedly calling `pump_cq`.
    #[verifier::rlimit(400)]
    pub fn pump_cq_all(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<usize>)
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
            peer_core_step(old(core), final(core), old(self).timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).active_leases@,
                final(self).active_leases@,
            ),
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.events == old(core).timelines.events,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).ledger@.submitted == old(core).ledger@.submitted,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            old(core).ledger@.succeeded.dom().subset_of(final(core).ledger@.succeeded.dom()),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            final(self).window_pool.pd() == old(self).window_pool.pd(),
            final(self).cq.in_flight().dom().subset_of(old(self).cq.in_flight().dom()),
            final(self).cq.in_flight().len() <= old(self).cq.in_flight().len(),
            forall|k: u64| #[trigger]
                final(self).pending().contains_key(k) ==> (old(self).pending().contains_key(k)
                    && final(self).pending()[k] == old(self).pending()[k]),
            forall|k: u64| #[trigger]
                old(self).pending().contains_key(k) ==> ((final(self).pending().contains_key(k)
                    && final(self).pending()[k] == old(self).pending()[k])
                    || final(self).completed_pull_spec(k) is Some),
            forall|k: u64| #[trigger]
                old(self).completed_pull_spec(k) is Some && !old(self).pending().contains_key(k)
                    && (forall|w: u64| #[trigger]
                    old(self).pending().contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64))
                        != (k % (BITMAP_CAPACITY as u64))) ==> final(self).completed_pull_spec(k)
                    == old(self).completed_pull_spec(k),
            forall|h: RemoteLeaseHandle| #[trigger]
                old(self).lease_index_spec(h) is Some && !(old(self).active_leases@[old(
                    self,
                ).lease_index_spec(h)->Some_0]).has_active_wr() ==> (final(self).lease_index_spec(
                    h,
                ) matches Some(j) && final(self).active_leases@[j] == old(self).active_leases@[old(
                    self,
                ).lease_index_spec(h)->Some_0]),
            forall|h: RemoteLeaseHandle| #[trigger]
                old(self).lease_index_spec(h) is Some && rdma_lease_is_binding_like(
                    old(self).active_leases@[old(self).lease_index_spec(h)->Some_0],
                ) ==> (final(self).lease_index_spec(h) matches Some(j)
                    && rdma_lease_binding_preserved(
                    old(self).active_leases@[old(self).lease_index_spec(h)->Some_0],
                    final(self).active_leases@[j],
                )),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_core = *core;
        let ghost before_cq_dom = self.cq.in_flight().dom();
        let ghost before_cq_len = self.cq.in_flight().len();
        let ghost before_leases = self.active_leases@;
        let ghost before_pending = self.pending();
        let ghost before_ops = self.in_flight_ops;
        let ghost before_snap = rdma::CqSnapshot::of(self.cq);
        let mut remaining = self.cq.in_flight_count();
        let mut count: usize = 0;
        proof {
            lemma_peer_core_step_refl(core, self.timeline_id);
            lemma_foreign_shares_refl(core, self.active_leases@);
        }
        while remaining > 0
            invariant
                before_core == *old(core),
                old(core).well_formed(),
                old(self).well_formed(old(core)),
                before_cq_dom == old(self).cq.in_flight().dom(),
                before_cq_len == old(self).cq.in_flight().len(),
                before_cq_len <= usize::MAX,
                before_leases == old(self).active_leases@,
                before_pending == old(self).pending(),
                before_ops == old(self).in_flight_ops,
                before_snap == rdma::CqSnapshot::of(old(self).cq),
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id == before_core.node_id,
                core.instance_id() == before_core.instance_id(),
                core.tiers == before_core.tiers,
                core.frames == before_core.frames,
                core.catalog == before_core.catalog,
                core.spec == before_core.spec,
                core.next_version_seq == before_core.next_version_seq,
                core.version_seq_limit == before_core.version_seq_limit,
                core.version_registry == before_core.version_registry,
                peer_core_step(&before_core, core, self.timeline_id),
                peer_foreign_shares_preserved(
                    &before_core,
                    core,
                    before_leases,
                    self.active_leases@,
                ),
                core.timelines.count == before_core.timelines.count,
                core.timelines.kinds == before_core.timelines.kinds,
                core.timelines.events == before_core.timelines.events,
                core.timelines.next_seq == before_core.timelines.next_seq,
                core.ledger@.submitted == before_core.ledger@.submitted,
                core.ledger@.revoked == before_core.ledger@.revoked,
                before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set()),
                before_core.ledger@.succeeded.dom().subset_of(core.ledger@.succeeded.dom()),
                self.timeline_id == old(self).timeline_id,
                self.peer_node == old(self).peer_node,
                self.qp == old(self).qp,
                self.host_mr == old(self).host_mr,
                self.gpu_mr == old(self).gpu_mr,
                self.shared_host_mr == old(self).shared_host_mr,
                self.shared_gpu_mr == old(self).shared_gpu_mr,
                self.next_control_wr_id == old(self).next_control_wr_id,
                self.window_pool.pd() == old(self).window_pool.pd(),
                self.cq.in_flight().dom().subset_of(before_cq_dom),
                self.cq.in_flight().len() <= before_cq_len,
                forall|k: u64| #[trigger]
                    self.pending().contains_key(k) ==> (before_pending.contains_key(k)
                        && self.pending()[k] == before_pending[k]),
                forall|k: u64| #[trigger]
                    before_pending.contains_key(k) ==> ((self.pending().contains_key(k)
                        && self.pending()[k] == before_pending[k]) || self.completed_pull_spec(
                        k,
                    ) is Some),
                forall|k: u64| #[trigger]
                    old(self).completed_pull_spec(k) is Some && !before_pending.contains_key(k) && (
                    forall|w: u64| #[trigger]
                        before_pending.contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64))
                            != (k % (BITMAP_CAPACITY as u64))) ==> self.completed_pull_spec(k)
                        == old(self).completed_pull_spec(k),
                count + remaining == before_cq_len,
                forall|h: RemoteLeaseHandle| #[trigger]
                    rdma_lease_index_spec(before_leases, h) is Some && !(
                    before_leases[rdma_lease_index_spec(before_leases, h)->Some_0]).has_active_wr()
                        ==> (self.lease_index_spec(h) matches Some(j) && self.active_leases@[j]
                        == before_leases[rdma_lease_index_spec(before_leases, h)->Some_0]),
                forall|h: RemoteLeaseHandle| #[trigger]
                    rdma_lease_index_spec(before_leases, h) is Some && rdma_lease_is_binding_like(
                        before_leases[rdma_lease_index_spec(before_leases, h)->Some_0],
                    ) ==> (self.lease_index_spec(h) matches Some(j) && rdma_lease_binding_preserved(
                        before_leases[rdma_lease_index_spec(before_leases, h)->Some_0],
                        self.active_leases@[j],
                    )),
            decreases remaining,
        {
            let ghost step_core = *core;
            let ghost step_pending = self.pending();
            let ghost step_rdma = *self;
            match self.pump_cq(core, Tracked(instance)) {
                Ok(None) => break,
                Ok(Some(wr_id)) => {
                    count += 1;
                    remaining -= 1;
                    proof {
                        lemma_peer_core_step_trans(
                            &before_core,
                            &step_core,
                            core,
                            self.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            &before_core,
                            &step_core,
                            core,
                            before_leases,
                            step_rdma.active_leases@,
                            self.active_leases@,
                        );
                        assert(self.cq.in_flight().dom().subset_of(before_cq_dom)) by {
                            broadcast use vstd::iset::group_iset_lemmas;

                        };
                        assert(before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set()))
                            by {
                            broadcast use vstd::iset::group_iset_lemmas;

                        };
                        assert(before_core.ledger@.succeeded.dom().subset_of(
                            core.ledger@.succeeded.dom(),
                        )) by {
                            broadcast use vstd::iset::group_iset_lemmas;

                        };
                        assert forall|k: u64| #[trigger] before_pending.contains_key(k) implies ((
                        self.pending().contains_key(k) && self.pending()[k] == before_pending[k])
                            || self.completed_pull_spec(k) is Some) by {
                            broadcast use {
                                vstd::map::group_map_lemmas,
                                vstd::set::group_set_lemmas,
                            };

                            reveal(rdma_window_wf);
                            if step_pending.contains_key(k) {
                                if k != wr_id {
                                    assert(self.pending().contains_key(k));
                                }
                            } else {
                                if step_pending.contains_key(wr_id) {
                                    assert(before_pending.contains_key(wr_id));
                                    assert(wr_id != k);
                                    assert(before_snap.pending.contains_key(k));
                                    assert(before_snap.pending.contains_key(wr_id));
                                    assert(before_ops[(k % (
                                    BITMAP_CAPACITY as u64)) as int] matches Some(opk)
                                        && opk.sequence == k);
                                    assert(before_ops[(wr_id % (
                                    BITMAP_CAPACITY as u64)) as int] matches Some(opw)
                                        && opw.sequence == wr_id);
                                    assert((k % (BITMAP_CAPACITY as u64)) != (wr_id % (
                                    BITMAP_CAPACITY as u64)));
                                }
                            }
                        }
                        assert forall|k: u64| #[trigger]
                            old(self).completed_pull_spec(k) is Some
                                && !before_pending.contains_key(k) && (forall|w: u64| #[trigger]
                                before_pending.contains_key(w) && w != k ==> (w % (
                                BITMAP_CAPACITY as u64)) != (k % (
                                BITMAP_CAPACITY as u64))) implies self.completed_pull_spec(k)
                            == old(self).completed_pull_spec(k) by {
                            assert(!step_pending.contains_key(k));
                            assert forall|w: u64| #[trigger]
                                step_pending.contains_key(w) && w != k implies (w % (
                            BITMAP_CAPACITY as u64)) != (k % (BITMAP_CAPACITY as u64)) by {
                                assert(before_pending.contains_key(w));
                            }
                            assert(step_rdma.completed_pull_spec(k) == old(
                                self,
                            ).completed_pull_spec(k));
                        }
                        assert forall|h: RemoteLeaseHandle| #[trigger]
                            rdma_lease_index_spec(before_leases, h) is Some && !(
                            before_leases[rdma_lease_index_spec(
                                before_leases,
                                h,
                            )->Some_0]).has_active_wr() implies (self.lease_index_spec(
                            h,
                        ) matches Some(j) && self.active_leases@[j]
                            == before_leases[rdma_lease_index_spec(before_leases, h)->Some_0]) by {
                            let i0 = rdma_lease_index_spec(before_leases, h)->Some_0;
                            assert(step_rdma.lease_index_spec(h) is Some);
                            let j1 = step_rdma.lease_index_spec(h)->Some_0;
                            assert(step_rdma.active_leases@[j1] == before_leases[i0]);
                            assert(!step_rdma.active_leases@[j1].has_active_wr());
                            assert(self.lease_index_spec(h) is Some);
                            let j2 = self.lease_index_spec(h)->Some_0;
                            assert(self.active_leases@[j2] == step_rdma.active_leases@[j1]);
                        }
                        assert forall|h: RemoteLeaseHandle| #[trigger]
                            rdma_lease_index_spec(before_leases, h) is Some
                                && rdma_lease_is_binding_like(
                                before_leases[rdma_lease_index_spec(before_leases, h)->Some_0],
                            ) implies (self.lease_index_spec(h) matches Some(j)
                            && rdma_lease_binding_preserved(
                            before_leases[rdma_lease_index_spec(before_leases, h)->Some_0],
                            self.active_leases@[j],
                        )) by {
                            let i0 = rdma_lease_index_spec(before_leases, h)->Some_0;
                            assert(step_rdma.lease_index_spec(h) is Some);
                            let j1 = step_rdma.lease_index_spec(h)->Some_0;
                            assert(rdma_lease_binding_preserved(
                                before_leases[i0],
                                step_rdma.active_leases@[j1],
                            ));
                            assert(rdma_lease_is_binding_like(step_rdma.active_leases@[j1]));
                            assert(self.lease_index_spec(h) is Some);
                            let j2 = self.lease_index_spec(h)->Some_0;
                            assert(rdma_lease_binding_preserved(
                                step_rdma.active_leases@[j1],
                                self.active_leases@[j2],
                            ));
                        }
                    }
                },
                Err(e) => return Err(e),
            }
        }
        Ok(count)
    }
}

} // verus!
