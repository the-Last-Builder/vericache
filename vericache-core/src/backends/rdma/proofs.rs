use super::*;
use vstd::prelude::*;

verus! {

/// Establish initial `rdma_window_wf` for a freshly created `RdmaBackend` with
/// an empty CQ and freshly registered timeline (`next_seq == 1`, `confirmed_seq == 0`).
pub(super) proof fn lemma_rdma_init_window_wf(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
)
    requires
        t < table.count,
        table.well_formed(id, cq.observer, ledger),
        table.next_seq[t as int] == 1,
        table.confirmed_seq[t as int] == 0,
        bitmap.well_formed(),
        forall|j: int| !#[trigger] bitmap.bit_set(j),
        forall|j: int| 0 <= j < BITMAP_CAPACITY ==> #[trigger] ops[j] == None,
        cq.pending.is_empty(),
    ensures
        rdma_window_wf(ops, bitmap, table, t, ledger, cq, id, CONTROL_WR_ID_BASE),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_window_wf);
    assert(rdma_data_events(cq) =~= Map::<u64, TransferEvent>::empty());
    assert(bitmap.observations(table, t, cq.observer, ledger));
    assert(window_matches(ops, bitmap, table, t, rdma_data_events(cq)));
}

/// Extract the fresh-control-WR-ID guarantee (`!cq.in_flight().contains_key(next_control_wr_id)`)
/// from `rdma_window_wf`.
pub(super) proof fn lemma_rdma_control_wr_id_fresh(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
)
    requires
        rdma_window_wf(ops, bitmap, table, t, ledger, cq, id, next_control_wr_id),
    ensures
        CONTROL_WR_ID_BASE <= next_control_wr_id,
        !cq.pending.contains_key(next_control_wr_id),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_window_wf);
    if cq.pending.contains_key(next_control_wr_id) {
        if cq.pending[next_control_wr_id].kind is DataTransfer {
            assert(rdma_data_events(cq).contains_key(next_control_wr_id));
            assert(next_control_wr_id < table.next_seq[t as int]);
            assert(table.next_seq[t as int] <= MAX_TICKET_SEQ + 1);
            assert(false);
        } else {
            assert(next_control_wr_id < next_control_wr_id);
        }
    }
}

/// Preserve `rdma_window_wf` when a control WR (`BindWindow` or `LocalInvalidate`)
/// is inserted into `cq` at `wr_id == next_control_wr_id` and `next_control_wr_id`
/// advances by 1.
pub(super) proof fn lemma_rdma_window_after_control_post(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    ledger: &KnowledgeLedger,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    id: InstanceId,
    wr_id: u64,
    req: rdma::PendingWorkRequest,
)
    requires
        rdma_window_wf(ops, bitmap, table, t, ledger, before_cq, id, wr_id),
        wr_id < u64::MAX,
        !(req.kind is DataTransfer),
        after_cq.observer == before_cq.observer,
        after_cq.pending == before_cq.pending.insert(wr_id, req),
    ensures
        rdma_window_wf(ops, bitmap, table, t, ledger, after_cq, id, (wr_id + 1) as u64),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_window_wf);
    assert(rdma_data_events(after_cq) =~= rdma_data_events(before_cq));
}

/// Preserve `rdma_window_wf` when `table` only changes watermarks (`only_watermarks`)
/// and `ledger`, `cq`, `ops`, `bitmap` are unchanged (e.g., `unpin_remote_lease`).
pub(super) proof fn lemma_rdma_window_after_watermarks(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    before_table: &TimelineTable,
    after_table: &TimelineTable,
    t: u8,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
)
    requires
        rdma_window_wf(ops, bitmap, before_table, t, ledger, cq, id, next_control_wr_id),
        after_table.well_formed(id, cq.observer, ledger),
        after_table.only_watermarks(before_table),
    ensures
        rdma_window_wf(ops, bitmap, after_table, t, ledger, cq, id, next_control_wr_id),
{
    reveal(rdma_window_wf);
}

/// Preserve `rdma_window_wf` when `ledger` grows by a fence event (`finish_fenced_revocation`)
/// while `table`, `cq`, `ops`, `bitmap` are unchanged.
pub(super) proof fn lemma_rdma_window_after_fence_ledger(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    before_ledger: &KnowledgeLedger,
    after_ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
)
    requires
        rdma_window_wf(ops, bitmap, table, t, before_ledger, cq, id, next_control_wr_id),
        table.well_formed(id, cq.observer, after_ledger),
        before_ledger.knows.set().subset_of(after_ledger.knows.set()),
    ensures
        rdma_window_wf(ops, bitmap, table, t, after_ledger, cq, id, next_control_wr_id),
{
    broadcast use vstd::iset::group_iset_lemmas;

    reveal(rdma_window_wf);
    assert(bitmap.observations(table, t, cq.observer, after_ledger)) by {
        assert forall|i: int| #[trigger] bitmap.bit_set(i) implies {
            let s = (table.confirmed_seq[t as int] + 1 + i) as u64;
            table.issued(t, s) && table.observed(t, s, cq.observer, after_ledger.knows)
        } by {
            let s = (table.confirmed_seq[t as int] + 1 + i) as u64;
            assert(table.observed(t, s, cq.observer, before_ledger.knows));
        }
    }
}

/// Establish `rdma_leases_valid` for an empty lease list.
pub(super) proof fn lemma_rdma_leases_empty<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
)
    requires
        leases.len() == 0,
    ensures
        rdma_leases_valid(leases, pool, qp, core),
{
    reveal(rdma_leases_disjoint);
    reveal(rdma_leases_valid);
}

/// Extract single-entry validity from `rdma_leases_valid`.
pub(super) proof fn lemma_rdma_lease_extract<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
    idx: int,
)
    requires
        rdma_leases_valid(leases, pool, qp, core),
        0 <= idx < leases.len(),
    ensures
        leases[idx].valid(pool, qp, core),
{
    reveal(rdma_leases_valid);
}

/// Prove that any checked-out MW slot `mw_slot` (whose state was `Free` before
/// checkout) is disjoint from every existing active remote lease.
pub(super) proof fn lemma_rdma_free_slot_not_in_leases<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
    mw_slot: u32,
)
    requires
        rdma_leases_valid(leases, pool, qp, core),
        (mw_slot as int) < pool.slots.len(),
        pool.slots[mw_slot as int].state == MwSlotState::Free,
    ensures
        forall|i: int| 0 <= i < leases.len() ==> (#[trigger] leases[i]).mw_slot != mw_slot,
{
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).mw_slot
        != mw_slot by {
        assert(leases[i].valid(pool, qp, core));
    }
}

/// Preserve one `ActiveRemoteLease` across a `CacheCore` + `WindowPool` update
/// that only touches `touched_slot` (disjoint from `lease.mw_slot`) and optionally
/// adds or removes one remote share on `touched_frame` (disjoint from `lease.share@`).
proof fn lemma_rdma_one_lease_preserved<'a>(
    lease: &ActiveRemoteLease,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_qp: &rdma::QueuePair<'a>,
    after_qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    touched_slot: Option<u32>,
    touched_frame: Option<FrameId>,
    removed_share: Option<TransferShare>,
    added_share: Option<TransferShare>,
)
    requires
        lease.valid(before_pool, before_qp, before_core),
        before_core.well_formed(),
        after_core.well_formed(),
        after_qp.id() == before_qp.id(),
        after_core.instance_id() == before_core.instance_id(),
        after_pool.slots.len() == before_pool.slots.len(),
        touched_slot != Some(lease.mw_slot),
        forall|j: int|
            0 <= j < after_pool.slots.len() && (touched_slot matches Some(ts) ==> j != ts as int)
                ==> (#[trigger] after_pool.slots[j]) == before_pool.slots[j],
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.same_other_pins(&before_core.frames, touched_frame),
        touched_frame == Some(lease.frame) && lease.state != RemoteLeaseState::Unbinding ==> {
            let tf = lease.frame;
            &&& removed_share is None || added_share is None
            &&& match removed_share {
                None => true,
                Some(rs) => {
                    &&& rs != lease.share@
                    &&& after_core.frames.frame(tf).remote_shares == before_core.frames.frame(
                        tf,
                    ).remote_shares.remove(rs)
                },
            }
            &&& match added_share {
                None => true,
                Some(as_) => {
                    &&& as_ != lease.share@
                    &&& after_core.frames.frame(tf).remote_shares.contains_key(as_)
                    &&& forall|s: TransferShare|
                        s != as_ && #[trigger] before_core.frames.frame(
                            tf,
                        ).remote_shares.contains_key(s) ==> after_core.frames.frame(
                            tf,
                        ).remote_shares.contains_key(s) && after_core.frames.frame(
                            tf,
                        ).remote_shares[s] == before_core.frames.frame(tf).remote_shares[s]
                },
            }
            &&& (removed_share is None && added_share is None) ==> after_core.frames.frame(
                tf,
            ).remote_shares == before_core.frames.frame(tf).remote_shares && after_core.frames.meta(
                tf,
            ).active_remote_leases == before_core.frames.meta(tf).active_remote_leases
        },
    ensures
        lease.valid(after_pool, after_qp, after_core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(FrameTable::same_directory);
    hide(FrameTable::same_other_pins);
    assert(after_core.frames.metas.len() == before_core.frames.metas.len()
        && after_core.frames.frame(lease.frame).slot == before_core.frames.frame(lease.frame).slot)
        by {
        reveal(FrameTable::same_directory);
    }
    assert(after_pool.slots[lease.mw_slot as int] == before_pool.slots[lease.mw_slot as int]);
    if lease.state != RemoteLeaseState::Unbinding {
        if touched_frame != Some(lease.frame) {
            assert(after_core.frames.meta(lease.frame).active_remote_leases
                == before_core.frames.meta(lease.frame).active_remote_leases
                && after_core.frames.frame(lease.frame).remote_shares == before_core.frames.frame(
                lease.frame,
            ).remote_shares) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::same_other_pins);
            }
        } else {
            assert(after_core.frames.frame(lease.frame).matches_lease(
                lease.share@,
                lease.mw_slot,
                lease.generation,
            )) by {
                broadcast use vstd::map::group_map_lemmas;

            }
            assert(after_core.frames.meta(lease.frame).active_remote_leases > 0) by {
                lemma_frame_remote_share_implies_nonzero(after_core, lease.frame, lease.share@);
            }
        }
    }
}

/// Pure helper: if a published/well-formed frame contains `share` in `remote_shares`,
/// then `meta.active_remote_leases > 0`.
pub(super) proof fn lemma_frame_remote_share_implies_nonzero(
    core: &CacheCore,
    f: FrameId,
    share: TransferShare,
)
    requires
        core.well_formed(),
        f.0 < core.frames.metas.len(),
        core.frames.frame(f).remote_shares.contains_key(share),
    ensures
        core.frames.meta(f).active_remote_leases > 0,
{
    assert(core.frames.valid_frame(f, core.kernel_view()));
    assert(core.frames.frame(f).authority(core.frames.meta(f), core.instance_id()));
    broadcast use vstd::map::group_map_lemmas;

    assert(core.frames.frame(f).remote_shares.len() >= 1);
}

/// Push one newly valid `ActiveRemoteLease` (whose `mw_slot` was disjoint from
/// all prior leases) after `pin_remote_lease` + `post_bind`.
pub(super) proof fn lemma_rdma_leases_push<'a>(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    new_lease: ActiveRemoteLease,
)
    requires
        rdma_leases_valid(before, before_pool, qp, before_core),
        after == before.push(new_lease),
        new_lease.valid(after_pool, qp, after_core),
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.instance_id() == before_core.instance_id(),
        after_pool.slots.len() == before_pool.slots.len(),
        forall|i: int|
            0 <= i < before.len() ==> (#[trigger] before[i]).mw_slot != new_lease.mw_slot,
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != new_lease.mw_slot as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.same_other_pins(&before_core.frames, Some(new_lease.frame)),
        !before_core.frames.frame(new_lease.frame).remote_shares.contains_key(new_lease.share@),
        after_core.frames.frame(new_lease.frame).remote_shares.contains_key(new_lease.share@),
        forall|s: TransferShare|
            s != new_lease.share@ && #[trigger] before_core.frames.frame(
                new_lease.frame,
            ).remote_shares.contains_key(s) ==> after_core.frames.frame(
                new_lease.frame,
            ).remote_shares.contains_key(s) && after_core.frames.frame(
                new_lease.frame,
            ).remote_shares[s] == before_core.frames.frame(new_lease.frame).remote_shares[s],
    ensures
        rdma_leases_valid(after, after_pool, qp, after_core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(ActiveRemoteLease::valid);
    reveal(rdma_leases_disjoint);
    reveal(rdma_leases_valid);
    let n = before.len();
    assert forall|i: int, j: int| 0 <= i < j < after.len() implies (#[trigger] after[i]).mw_slot
        != (#[trigger] after[j]).mw_slot by {
        if j < n {
            assert(after[i] == before[i] && after[j] == before[j]);
        } else {
            assert(after[i] == before[i] && after[j] == new_lease);
        }
    }
    assert forall|i: int| 0 <= i < after.len() implies (#[trigger] after[i]).valid(
        after_pool,
        qp,
        after_core,
    ) by {
        if i < n {
            assert(after[i] == before[i]);
            if before[i].frame == new_lease.frame && before[i].state
                != RemoteLeaseState::Unbinding {
                assert(before[i].share@ != new_lease.share@) by {
                    reveal(ActiveRemoteLease::valid);
                    assert(before[i].valid(before_pool, qp, before_core));
                    assert(before_core.frames.frame(new_lease.frame).remote_shares.contains_key(
                        before[i].share@,
                    ));
                }
            }
            lemma_rdma_one_lease_preserved(
                &before[i],
                before_pool,
                after_pool,
                qp,
                qp,
                before_core,
                after_core,
                Some(new_lease.mw_slot),
                Some(new_lease.frame),
                None,
                Some(new_lease.share@),
            );
        }
    }
}

/// Update one lease in place at `idx` (e.g., `Binding -> Bound`, `Bound -> Unbinding`,
/// `Bound -> Revoking`) while preserving all other leases.
pub(super) proof fn lemma_rdma_leases_update_in_place<'a>(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    removed_share: Option<TransferShare>,
)
    requires
        rdma_leases_valid(before, before_pool, qp, before_core),
        0 <= idx < before.len(),
        after.len() == before.len(),
        after[idx].mw_slot == before[idx].mw_slot,
        after[idx].valid(after_pool, qp, after_core),
        forall|j: int| 0 <= j < before.len() && j != idx ==> (#[trigger] after[j]) == before[j],
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.instance_id() == before_core.instance_id(),
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != before[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.same_other_pins(&before_core.frames, Some(before[idx].frame)),
        match removed_share {
            None => after_core.frames == before_core.frames,
            Some(rs) => {
                &&& rs == before[idx].share@
                &&& before_core.frames.frame(before[idx].frame).matches_lease(
                    rs,
                    before[idx].mw_slot,
                    before[idx].generation,
                )
                &&& after_core.frames.frame(before[idx].frame).remote_shares
                    == before_core.frames.frame(before[idx].frame).remote_shares.remove(rs)
            },
        },
    ensures
        rdma_leases_valid(after, after_pool, qp, after_core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(ActiveRemoteLease::valid);
    reveal(rdma_leases_disjoint);
    reveal(rdma_leases_valid);
    let slot_i = before[idx].mw_slot;
    assert forall|i: int, j: int| 0 <= i < j < after.len() implies (#[trigger] after[i]).mw_slot
        != (#[trigger] after[j]).mw_slot by {
        assert(after[i].mw_slot == before[i].mw_slot);
        assert(after[j].mw_slot == before[j].mw_slot);
    }
    assert forall|i: int| 0 <= i < after.len() implies (#[trigger] after[i]).valid(
        after_pool,
        qp,
        after_core,
    ) by {
        if i != idx {
            assert(after[i] == before[i]);
            assert(before[i].mw_slot != slot_i);
            if let Some(rs) = removed_share {
                if before[i].frame == before[idx].frame && before[i].state
                    != RemoteLeaseState::Unbinding {
                    assert(rs != before[i].share@) by {
                        reveal(ActiveRemoteLease::valid);
                        assert(before[i].valid(before_pool, qp, before_core));
                        assert(before_core.frames.frame(before[idx].frame).matches_lease(
                            before[i].share@,
                            before[i].mw_slot,
                            before[i].generation,
                        ));
                    }
                }
            }
            lemma_rdma_one_lease_preserved(
                &before[i],
                before_pool,
                after_pool,
                qp,
                qp,
                before_core,
                after_core,
                Some(slot_i),
                Some(before[idx].frame),
                removed_share,
                None,
            );
        }
    }
}

/// Remove one completed/failed/revoked lease at `idx` (`swap_remove`) while
/// preserving all remaining leases.
pub(super) proof fn lemma_rdma_leases_swap_remove<'a>(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_qp: &rdma::QueuePair<'a>,
    after_qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    removed_share: Option<TransferShare>,
)
    requires
        rdma_leases_valid(before, before_pool, before_qp, before_core),
        0 <= idx < before.len(),
        after.len() == before.len() - 1,
        forall|j: int|
            0 <= j < after.len() ==> (#[trigger] after[j]) == if j == idx {
                before[before.len() - 1]
            } else {
                before[j]
            },
        before_core.well_formed(),
        after_core.well_formed(),
        after_qp.id() == before_qp.id(),
        after_core.instance_id() == before_core.instance_id(),
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != before[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.same_other_pins(&before_core.frames, Some(before[idx].frame)),
        match removed_share {
            None => after_core.frames == before_core.frames,
            Some(rs) => {
                &&& rs == before[idx].share@
                &&& before_core.frames.frame(before[idx].frame).matches_lease(
                    rs,
                    before[idx].mw_slot,
                    before[idx].generation,
                )
                &&& after_core.frames.frame(before[idx].frame).remote_shares
                    == before_core.frames.frame(before[idx].frame).remote_shares.remove(rs)
            },
        },
    ensures
        rdma_leases_valid(after, after_pool, after_qp, after_core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(ActiveRemoteLease::valid);
    reveal(rdma_leases_disjoint);
    reveal(rdma_leases_valid);
    let last = before.len() - 1;
    let slot_i = before[idx].mw_slot;
    assert forall|i: int, j: int| 0 <= i < j < after.len() implies (#[trigger] after[i]).mw_slot
        != (#[trigger] after[j]).mw_slot by {
        let bi = if i == idx {
            last
        } else {
            i
        };
        let bj = if j == idx {
            last
        } else {
            j
        };
        assert(bi != bj);
        assert(after[i] == before[bi] && after[j] == before[bj]);
        if bi < bj {
            assert(before[bi].mw_slot != before[bj].mw_slot);
        } else {
            assert(before[bj].mw_slot != before[bi].mw_slot);
        }
    }
    assert forall|i: int| 0 <= i < after.len() implies (#[trigger] after[i]).valid(
        after_pool,
        after_qp,
        after_core,
    ) by {
        let bi = if i == idx {
            last
        } else {
            i
        };
        assert(bi != idx);
        assert(after[i] == before[bi]);
        assert(before[bi].mw_slot != slot_i) by {
            if bi < idx {
                assert(before[bi].mw_slot != before[idx].mw_slot);
            } else {
                assert(before[idx].mw_slot != before[bi].mw_slot);
            }
        }
        if let Some(rs) = removed_share {
            if before[bi].frame == before[idx].frame && before[bi].state
                != RemoteLeaseState::Unbinding {
                assert(rs != before[bi].share@) by {
                    reveal(ActiveRemoteLease::valid);
                    assert(before[bi].valid(before_pool, before_qp, before_core));
                    assert(before_core.frames.frame(before[idx].frame).matches_lease(
                        before[bi].share@,
                        before[bi].mw_slot,
                        before[bi].generation,
                    ));
                }
            }
        }
        lemma_rdma_one_lease_preserved(
            &before[bi],
            before_pool,
            after_pool,
            before_qp,
            after_qp,
            before_core,
            after_core,
            Some(slot_i),
            Some(before[idx].frame),
            removed_share,
            None,
        );
    }
}

/// Preserve all active remote leases when `pool` acknowledges/replenishes a
/// `Quarantined`/`Vacant` slot (disjoint from all active leases) and `core` is untouched.
pub(super) proof fn lemma_rdma_leases_after_pool_retire_replenish<'a>(
    leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
    slot_idx: u32,
)
    requires
        rdma_leases_valid(leases, before_pool, qp, core),
        core.well_formed(),
        (slot_idx as int) < before_pool.slots.len(),
        before_pool.slots[slot_idx as int].state is Quarantined
            || before_pool.slots[slot_idx as int].state == MwSlotState::Vacant,
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != slot_idx as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
    ensures
        rdma_leases_valid(leases, after_pool, qp, core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        after_pool,
        qp,
        core,
    ) by {
        assert(leases[i].valid(before_pool, qp, core));
        assert(leases[i].mw_slot != slot_idx);
        assert(after_pool.slots[leases[i].mw_slot as int]
            == before_pool.slots[leases[i].mw_slot as int]);
    }
}

/// Preserve all active remote leases across `unquarantine_all` and/or QP state
/// transitions that keep `qp.id()` unchanged.
pub(super) proof fn lemma_rdma_leases_after_qp_state_change<'a>(
    leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_qp: &rdma::QueuePair<'a>,
    after_qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
)
    requires
        rdma_leases_valid(leases, before_pool, before_qp, core),
        after_qp.id() == before_qp.id(),
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && !(before_pool.slots[j].state is Quarantined) ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
    ensures
        rdma_leases_valid(leases, after_pool, after_qp, core),
{
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        after_pool,
        after_qp,
        core,
    ) by {
        assert(leases[i].valid(before_pool, before_qp, core));
        assert(!(before_pool.slots[leases[i].mw_slot as int].state is Quarantined));
        assert(after_pool.slots[leases[i].mw_slot as int]
            == before_pool.slots[leases[i].mw_slot as int]);
    }
}

/// Preserve `RegisteredSlab` invariants when `core.frames` is unchanged or has
/// `same_directory`.
pub(super) proof fn lemma_rdma_slabs_same_directory<'a>(
    host_mr: &Option<RegisteredSlab<'a>>,
    gpu_mr: &Option<RegisteredSlab<'a>>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    pd_id: nat,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.tiers == before_core.tiers,
        after_core.frames.same_directory(&before_core.frames),
        host_mr matches Some(s) ==> s.well_formed(before_core, pd_id),
        gpu_mr matches Some(s) ==> s.well_formed(before_core, pd_id),
    ensures
        host_mr matches Some(s) ==> s.well_formed(after_core, pd_id),
        gpu_mr matches Some(s) ==> s.well_formed(after_core, pd_id),
{
    if let Some(ref s) = *host_mr {
        lemma_slab_bound_same_directory(
            before_core,
            after_core,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    if let Some(ref s) = *gpu_mr {
        lemma_slab_bound_same_directory(
            before_core,
            after_core,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
}

/// Preserve `rdma_leases_valid` when `pool.slots@` is unchanged.
pub(super) proof fn lemma_rdma_leases_same_pool_slots<'a>(
    leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
)
    requires
        rdma_leases_valid(leases, before_pool, qp, core),
        after_pool.slots@ == before_pool.slots@,
    ensures
        rdma_leases_valid(leases, after_pool, qp, core),
{
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        after_pool,
        qp,
        core,
    ) by {
        assert(leases[i].valid(before_pool, qp, core));
        assert(after_pool.slots[leases[i].mw_slot as int]
            == before_pool.slots[leases[i].mw_slot as int]);
    }
}

/// Preserve `rdma_leases_valid` when only `core.timelines` and `core.ledger`
/// advance (e.g., `poll_cq`, `record_data_submission`, `record_data_batch`) and
/// `core.frames` is completely unchanged.
pub(super) proof fn lemma_rdma_leases_same_frames<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        rdma_leases_valid(leases, pool, qp, before_core),
        after_core.instance_id() == before_core.instance_id(),
        after_core.frames == before_core.frames,
    ensures
        rdma_leases_valid(leases, pool, qp, after_core),
{
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        pool,
        qp,
        after_core,
    ) by {
        assert(leases[i].valid(pool, qp, before_core));
    }
}

/// Prove `rdma_leases_preserved_except` across a disjoint `swap_remove`.
pub(super) proof fn lemma_rdma_leases_swap_remove_preserves(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    idx: int,
    removed: RemoteLeaseHandle,
)
    requires
        rdma_leases_disjoint(before),
        0 <= idx < before.len(),
        before[idx].mw_slot as usize == removed.mw_slot,
        before[idx].generation == removed.generation,
        after.len() == before.len() - 1,
        forall|j: int|
            0 <= j < after.len() ==> (#[trigger] after[j]) == if j == idx {
                before[before.len() - 1]
            } else {
                before[j]
            },
    ensures
        rdma_leases_preserved_except(before, after, removed),
{
    reveal(rdma_leases_disjoint);
    assert forall|k: int|
        0 <= k < before.len() && !((#[trigger] before[k]).mw_slot as usize == removed.mw_slot
            && before[k].generation == removed.generation) implies exists|j: int|
        0 <= j < after.len() && (#[trigger] after[j]) == before[k] by {
        assert(k != idx);
        let j = if k == before.len() - 1 {
            idx
        } else {
            k
        };
        assert(after[j] == before[k]);
    }
    assert forall|j: int| 0 <= j < after.len() implies !((#[trigger] after[j]).mw_slot as usize
        == removed.mw_slot && after[j].generation == removed.generation) by {
        let bi = if j == idx {
            before.len() - 1
        } else {
            j
        };
        assert(bi != idx);
        assert(after[j] == before[bi]);
        if bi < idx {
            assert(before[bi].mw_slot != before[idx].mw_slot);
        } else {
            assert(before[idx].mw_slot != before[bi].mw_slot);
        }
    }
}

} // verus!
