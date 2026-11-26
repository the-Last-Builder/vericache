use super::*;
use vstd::prelude::*;

verus! {

/// Establish initial `rdma_window_wf` for a freshly created `RdmaBackend` with
/// an empty CQ and freshly registered timeline (`next_seq == 1`, `confirmed_seq == 0`).
pub(super) proof fn lemma_rdma_init_window_wf(
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: TimelineId,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
)
    requires
        t < table.count_spec(),
        table.well_formed(id, cq.observer, ledger),
        table.next_seq_spec()[t as int] == 1,
        table.confirmed_seq_spec()[t as int] == 0,
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
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: TimelineId,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
)
    requires
        rdma_window_wf(ops, bitmap, table, t, ledger, cq, id, next_control_wr_id),
    ensures
        CONTROL_WR_ID_BASE <= next_control_wr_id <= CONTROL_WR_ID_LIMIT,
        !cq.pending.contains_key(next_control_wr_id),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_window_wf);
    if cq.pending.contains_key(next_control_wr_id) {
        if cq.pending[next_control_wr_id].kind is DataTransfer {
            assert(rdma_data_events(cq).contains_key(next_control_wr_id));
            assert(next_control_wr_id < table.next_seq_spec()[t as int]);
            assert(table.next_seq_spec()[t as int] <= MAX_TICKET_SEQ + 1);
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
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: TimelineId,
    ledger: &KnowledgeLedger,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    id: InstanceId,
    wr_id: u64,
    req: rdma::PendingWorkRequest,
)
    requires
        rdma_window_wf(ops, bitmap, table, t, ledger, before_cq, id, wr_id),
        wr_id < CONTROL_WR_ID_LIMIT,
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
/// and `ledger`, `cq`, `ops`, `bitmap` are unchanged (e.g., `unpin_fenced_lease`).
pub(super) proof fn lemma_rdma_window_after_watermarks(
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
    bitmap: SlidingBitmap,
    before_table: &TimelineTable,
    after_table: &TimelineTable,
    t: TimelineId,
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
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: TimelineId,
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
            let s = (table.confirmed_seq_spec()[t as int] + 1 + i) as u64;
            table.issued(t, s) && table.observed(t, s, cq.observer, after_ledger.knows)
        } by {
            let s = (table.confirmed_seq_spec()[t as int] + 1 + i) as u64;
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
    reveal(rdma_pinned_shares_disjoint);
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
/// that only touches `touched_slot` (disjoint from `lease.mw_slot`) and preserves
/// the lease's fenced share if it touches the lease's frame.
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
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().same_other_pins(&before_core.frames_spec(), touched_frame),
        touched_frame == Some(lease.frame) && !lease.is_unpinned()
            ==> after_core.frames_spec().frame(lease.frame).has_fenced_share(lease.share@),
    ensures
        lease.valid(after_pool, after_qp, after_core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(FrameTable::same_directory);
    hide(FrameTable::same_other_pins);
    assert(after_core.frames_spec().metas_spec().len()
        == before_core.frames_spec().metas_spec().len() && after_core.frames_spec().frame(
        lease.frame,
    ).slot == before_core.frames_spec().frame(lease.frame).slot) by {
        reveal(FrameTable::same_directory);
    }
    assert(after_pool.slots[lease.mw_slot as int] == before_pool.slots[lease.mw_slot as int]);
    if !lease.is_unpinned() {
        if touched_frame != Some(lease.frame) {
            assert(after_core.frames_spec().meta(lease.frame).active_fenced_leases
                == before_core.frames_spec().meta(lease.frame).active_fenced_leases
                && after_core.frames_spec().frame(lease.frame).fenced_shares
                == before_core.frames_spec().frame(lease.frame).fenced_shares) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::same_other_pins);
            }
        } else {
            assert(after_core.frames_spec().meta(lease.frame).active_fenced_leases > 0) by {
                lemma_frame_fenced_share_implies_nonzero(after_core, lease.frame, lease.share@);
            }
        }
    }
}

/// A recorded fenced share contributes to the active lease count.
pub(in crate::backends) proof fn lemma_frame_fenced_share_implies_nonzero(
    core: &CacheCore,
    f: FrameId,
    share: TransferShare,
)
    requires
        core.well_formed(),
        f.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().frame(f).has_fenced_share(share),
    ensures
        core.frames_spec().meta(f).active_fenced_leases > 0,
{
    assert(core.frames_spec().valid_frame(f, core.kernel_view()));
    assert(core.frames_spec().frame(f).authority(core.frames_spec().meta(f), core.instance_id()));
    vstd::set::lemma_set_contains_len(core.frames_spec().frame(f).fenced_shares, share);
}

/// Push one newly valid `ActiveRemoteLease` (whose `mw_slot` was disjoint from
/// all prior leases) after `pin_fenced_lease` + `post_bind`.
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
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().same_other_pins(&before_core.frames_spec(), Some(new_lease.frame)),
        !before_core.frames_spec().frame(new_lease.frame).has_fenced_share(new_lease.share@),
        after_core.frames_spec().frame(new_lease.frame).fenced_shares
            == before_core.frames_spec().frame(new_lease.frame).fenced_shares.insert(
            new_lease.share@,
        ),
    ensures
        rdma_leases_valid(after, after_pool, qp, after_core),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(ActiveRemoteLease::valid);
    reveal(rdma_leases_disjoint);
    reveal(rdma_pinned_shares_disjoint);
    reveal(rdma_leases_valid);
    let n = before.len();
    assert forall|i: int, j: int| 0 <= i < j < after.len() implies {
        &&& (#[trigger] after[i]).mw_slot != (#[trigger] after[j]).mw_slot
        &&& !after[i].is_unpinned() && !after[j].is_unpinned() && after[i].frame == after[j].frame
            ==> after[i].share@ != after[j].share@
    } by {
        if j < n {
            assert(after[i] == before[i] && after[j] == before[j]);
        } else {
            assert(after[i] == before[i] && after[j] == new_lease);
            if !before[i].is_unpinned() && before[i].frame == new_lease.frame {
                assert(before_core.frames_spec().frame(new_lease.frame).has_fenced_share(
                    before[i].share@,
                )) by {
                    reveal(ActiveRemoteLease::valid);
                }
            }
        }
    }
    assert forall|i: int| 0 <= i < after.len() implies (#[trigger] after[i]).valid(
        after_pool,
        qp,
        after_core,
    ) by {
        if i < n {
            assert(after[i] == before[i]);
            if before[i].frame == new_lease.frame && !before[i].is_unpinned() {
                assert(after_core.frames_spec().frame(new_lease.frame).has_fenced_share(
                    before[i].share@,
                )) by {
                    reveal(ActiveRemoteLease::valid);
                    broadcast use vstd::set::group_set_lemmas;

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
        after[idx].frame == before[idx].frame,
        after[idx].share@ == before[idx].share@,
        !after[idx].is_unpinned() ==> !before[idx].is_unpinned(),
        after[idx].valid(after_pool, qp, after_core),
        forall|j: int| 0 <= j < before.len() && j != idx ==> (#[trigger] after[j]) == before[j],
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.instance_id() == before_core.instance_id(),
        after_pool.slots.len() == before_pool.slots.len(),
        forall|j: int|
            0 <= j < after_pool.slots.len() && j != before[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[j]) == before_pool.slots[j],
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().same_other_pins(
            &before_core.frames_spec(),
            Some(before[idx].frame),
        ),
        match removed_share {
            None => after_core.frames_spec() == before_core.frames_spec(),
            Some(rs) => {
                &&& rs == before[idx].share@
                &&& !before[idx].is_unpinned()
                &&& after_core.frames_spec().frame(before[idx].frame).fenced_shares
                    == before_core.frames_spec().frame(before[idx].frame).fenced_shares.remove(rs)
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
    reveal(rdma_pinned_shares_disjoint);
    reveal(rdma_leases_valid);
    let slot_i = before[idx].mw_slot;
    assert forall|i: int, j: int| 0 <= i < j < after.len() implies {
        &&& (#[trigger] after[i]).mw_slot != (#[trigger] after[j]).mw_slot
        &&& !after[i].is_unpinned() && !after[j].is_unpinned() && after[i].frame == after[j].frame
            ==> after[i].share@ != after[j].share@
    } by {
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
            if before[i].frame == before[idx].frame && !before[i].is_unpinned() {
                assert(after_core.frames_spec().frame(before[idx].frame).has_fenced_share(
                    before[i].share@,
                )) by {
                    reveal(ActiveRemoteLease::valid);
                    broadcast use vstd::set::group_set_lemmas;

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
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().same_other_pins(
            &before_core.frames_spec(),
            Some(before[idx].frame),
        ),
        match removed_share {
            None => after_core.frames_spec() == before_core.frames_spec(),
            Some(rs) => {
                &&& rs == before[idx].share@
                &&& !before[idx].is_unpinned()
                &&& after_core.frames_spec().frame(before[idx].frame).fenced_shares
                    == before_core.frames_spec().frame(before[idx].frame).fenced_shares.remove(rs)
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
    reveal(rdma_pinned_shares_disjoint);
    reveal(rdma_leases_valid);
    let last = before.len() - 1;
    let slot_i = before[idx].mw_slot;
    assert forall|i: int, j: int| 0 <= i < j < after.len() implies {
        &&& (#[trigger] after[i]).mw_slot != (#[trigger] after[j]).mw_slot
        &&& !after[i].is_unpinned() && !after[j].is_unpinned() && after[i].frame == after[j].frame
            ==> after[i].share@ != after[j].share@
    } by {
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
        if before[bi].frame == before[idx].frame && !before[bi].is_unpinned() {
            assert(after_core.frames_spec().frame(before[idx].frame).has_fenced_share(
                before[bi].share@,
            )) by {
                reveal(ActiveRemoteLease::valid);
                broadcast use vstd::set::group_set_lemmas;

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

/// Preserve `RegisteredSlab` invariants when `core.frames_spec()` is unchanged or has
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
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.instance_id() == before_core.instance_id(),
        after_core.tiers_spec() == before_core.tiers_spec(),
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
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

/// Preserve `rdma_leases_valid` when only `core.timelines_spec()` and `core.ledger`
/// advance (e.g., `poll_cq`, `record_data_submission`, `record_data_batch`) and
/// `core.frames_spec()` is completely unchanged.
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
        after_core.frames_spec() == before_core.frames_spec(),
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

/// Preserve all 4 RDMA slab invariants (owned + shared) when `core.frames_spec()` has `same_directory`.
pub(super) proof fn lemma_rdma_all_slabs_same_directory<'a>(
    host_mr: &Option<RegisteredSlab<'a>>,
    gpu_mr: &Option<RegisteredSlab<'a>>,
    shared_host_mr: Option<&'a RegisteredSlab<'a>>,
    shared_gpu_mr: Option<&'a RegisteredSlab<'a>>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    pd_id: nat,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.instance_id() == before_core.instance_id(),
        after_core.tiers_spec() == before_core.tiers_spec(),
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        host_mr matches Some(s) ==> s.well_formed(before_core, pd_id),
        gpu_mr matches Some(s) ==> s.well_formed(before_core, pd_id),
        shared_host_mr matches Some(s) ==> s.well_formed_shared(before_core, pd_id),
        shared_gpu_mr matches Some(s) ==> s.well_formed_shared(before_core, pd_id),
    ensures
        host_mr matches Some(s) ==> s.well_formed(after_core, pd_id),
        gpu_mr matches Some(s) ==> s.well_formed(after_core, pd_id),
        shared_host_mr matches Some(s) ==> s.well_formed_shared(after_core, pd_id),
        shared_gpu_mr matches Some(s) ==> s.well_formed_shared(after_core, pd_id),
{
    lemma_rdma_slabs_same_directory(host_mr, gpu_mr, before_core, after_core, pd_id);
    if let Some(s) = shared_host_mr {
        s.lemma_shared_wf_iff(before_core, pd_id);
        lemma_slab_bound_same_directory(
            before_core,
            after_core,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
        s.lemma_shared_wf_iff(after_core, pd_id);
    }
    if let Some(s) = shared_gpu_mr {
        s.lemma_shared_wf_iff(before_core, pd_id);
        lemma_slab_bound_same_directory(
            before_core,
            after_core,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
        s.lemma_shared_wf_iff(after_core, pd_id);
    }
}

/// Preserve `completed_pulls_wf` across any `CacheCore` step that preserves prior
/// timeline events and grows `ledger.knows` and `ledger.succeeded`.
pub(super) proof fn lemma_completed_pulls_after_step(
    completed_pulls: [Option<(u64, Result<(), i32>)>; BITMAP_SLOTS],
    before_table: &TimelineTable,
    after_table: &TimelineTable,
    t: TimelineId,
    observer: NodeId,
    before_ledger: &KnowledgeLedger,
    after_ledger: &KnowledgeLedger,
)
    requires
        completed_pulls_wf(completed_pulls, before_table, t, observer, before_ledger),
        t < before_table.count_spec(),
        after_table.count_spec() == before_table.count_spec(),
        after_table.next_seq_spec()[t as int] >= before_table.next_seq_spec()[t as int],
        forall|s: u64|
            before_table.issued(t, s) ==> #[trigger] after_table.events_spec()@[(t, s)]
                == before_table.events_spec()@[(t, s)],
        before_ledger.knows.set().subset_of(after_ledger.knows.set()),
        before_ledger.succeeded.dom().subset_of(after_ledger.succeeded.dom()),
    ensures
        completed_pulls_wf(completed_pulls, after_table, t, observer, after_ledger),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    assert forall|i: int| 0 <= i < BITMAP_SLOTS implies match #[trigger] completed_pulls[i] {
        None => true,
        Some((seq, st)) => {
            &&& ((seq as int) % (BITMAP_CAPACITY as int)) == i
            &&& after_table.issued(t, seq)
            &&& after_ledger.knows.set().contains((observer, after_table.events_spec()@[(t, seq)]))
            &&& (st.is_ok() ==> after_ledger.succeeded.dom().contains(
                after_table.events_spec()@[(t, seq)],
            ))
        },
    } by {
        if let Some((seq, st)) = completed_pulls[i] {
            assert(before_table.issued(t, seq));
            assert(after_table.events_spec()@[(t, seq)] == before_table.events_spec()@[(t, seq)]);
        }
    }
}

/// Preserve `rdma_pool_slots_owned_by_leases` after replacing slot `idx` via `swap_remove`
/// when `removed_slot` moved out of `Binding | Bound | Invalidating`.
pub(super) proof fn lemma_pool_slots_owned_after_swap_remove<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    removed_slot: u32,
)
    requires
        rdma_pool_slots_owned_by_leases(before_leases, before_pool),
        rdma_leases_disjoint(before_leases),
        0 <= idx < before_leases.len(),
        before_leases[idx].mw_slot == removed_slot,
        after_leases.len() == before_leases.len() - 1,
        forall|j: int|
            0 <= j < after_leases.len() ==> (#[trigger] after_leases[j]) == if j == idx {
                before_leases[before_leases.len() - 1]
            } else {
                before_leases[j]
            },
        after_pool.slots.len() == before_pool.slots.len(),
        (removed_slot as int) < after_pool.slots.len(),
        !(after_pool.slots[removed_slot as int].state is Binding),
        !(after_pool.slots[removed_slot as int].state is Bound),
        !(after_pool.slots[removed_slot as int].state is Invalidating),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != removed_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
    ensures
        rdma_pool_slots_owned_by_leases(after_leases, after_pool),
{
    reveal(rdma_pool_slots_owned_by_leases);
    reveal(rdma_leases_disjoint);
    assert forall|s: int|
        0 <= s < after_pool.slots.len() && ((#[trigger] after_pool.slots[s]).state is Binding
            || after_pool.slots[s].state is Bound
            || after_pool.slots[s].state is Invalidating) implies exists|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).mw_slot as int == s by {
        assert(s != removed_slot as int);
        assert(before_pool.slots[s] == after_pool.slots[s]);
        let bi = choose|k: int|
            0 <= k < before_leases.len() && (#[trigger] before_leases[k]).mw_slot as int == s;
        assert(bi != idx);
        let j = if bi == before_leases.len() - 1 {
            idx
        } else {
            bi
        };
        assert(after_leases[j] == before_leases[bi]);
        assert(after_leases[j].mw_slot as int == s);
    }
}

/// Preserve `rdma_pool_slots_owned_by_leases` after pushing one newly checked-out lease
/// whose `mw_slot` was previously `Free` (so `pool.slots` only changed at `new_slot`).
pub(super) proof fn lemma_pool_slots_owned_after_push<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    new_slot: u32,
)
    requires
        rdma_pool_slots_owned_by_leases(before_leases, before_pool),
        after_leases.len() == before_leases.len() + 1,
        forall|j: int|
            0 <= j < before_leases.len() ==> (#[trigger] after_leases[j]) == before_leases[j],
        after_leases[before_leases.len() as int].mw_slot == new_slot,
        after_pool.slots.len() == before_pool.slots.len(),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != new_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
    ensures
        rdma_pool_slots_owned_by_leases(after_leases, after_pool),
{
    reveal(rdma_pool_slots_owned_by_leases);
    assert forall|s: int|
        0 <= s < after_pool.slots.len() && ((#[trigger] after_pool.slots[s]).state is Binding
            || after_pool.slots[s].state is Bound
            || after_pool.slots[s].state is Invalidating) implies exists|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).mw_slot as int == s by {
        if s == new_slot as int {
            let last = before_leases.len() as int;
            assert(after_leases[last].mw_slot as int == s);
        } else {
            assert(before_pool.slots[s] == after_pool.slots[s]);
            let bi = choose|k: int|
                0 <= k < before_leases.len() && (#[trigger] before_leases[k]).mw_slot as int == s;
            assert(after_leases[bi] == before_leases[bi]);
            assert(after_leases[bi].mw_slot as int == s);
        }
    }
}

/// Preserve `rdma_pool_slots_owned_by_leases` after updating lease `idx` in place while
/// preserving `mw_slot` and only touching `pool.slots[slot]`.
pub(super) proof fn lemma_pool_slots_owned_after_update_in_place<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
)
    requires
        rdma_pool_slots_owned_by_leases(before_leases, before_pool),
        0 <= idx < before_leases.len(),
        after_leases.len() == before_leases.len(),
        after_leases[idx].mw_slot == before_leases[idx].mw_slot,
        forall|j: int|
            0 <= j < before_leases.len() && j != idx ==> (#[trigger] after_leases[j])
                == before_leases[j],
        after_pool.slots.len() == before_pool.slots.len(),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != before_leases[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
    ensures
        rdma_pool_slots_owned_by_leases(after_leases, after_pool),
{
    reveal(rdma_pool_slots_owned_by_leases);
    let touched = before_leases[idx].mw_slot as int;
    assert forall|s: int|
        0 <= s < after_pool.slots.len() && ((#[trigger] after_pool.slots[s]).state is Binding
            || after_pool.slots[s].state is Bound
            || after_pool.slots[s].state is Invalidating) implies exists|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).mw_slot as int == s by {
        if s == touched {
            assert(after_leases[idx].mw_slot as int == s);
        } else {
            assert(before_pool.slots[s] == after_pool.slots[s]);
            let bi = choose|k: int|
                0 <= k < before_leases.len() && (#[trigger] before_leases[k]).mw_slot as int == s;
            assert(after_leases[bi].mw_slot as int == s);
        }
    }
}

/// Preserve `rdma_control_leases_match_cq` when removing `removed_wr_id` from `cq` and
/// removing lease `idx` via `swap_remove`.
pub(super) proof fn lemma_control_leases_match_cq_after_swap_remove<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    next_control_wr_id: u64,
    removed_wr_id: Option<u64>,
)
    requires
        rdma_control_leases_match_cq(before_leases, before_pool, before_cq, next_control_wr_id),
        0 <= idx < before_leases.len(),
        after_leases.len() == before_leases.len() - 1,
        forall|j: int|
            0 <= j < after_leases.len() ==> (#[trigger] after_leases[j]) == if j == idx {
                before_leases[before_leases.len() - 1]
            } else {
                before_leases[j]
            },
        after_pool.slots.len() == before_pool.slots.len(),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != before_leases[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
        removed_wr_id matches Some(rw) ==> before_leases[idx].has_active_wr()
            && before_leases[idx].wr_id == rw,
        after_cq.pending == match removed_wr_id {
            None => before_cq.pending,
            Some(rw) => before_cq.pending.remove(rw),
        },
        rdma_leases_disjoint(before_leases),
    ensures
        rdma_control_leases_match_cq(after_leases, after_pool, after_cq, next_control_wr_id),
{
    reveal(rdma_control_leases_match_cq);
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_leases_disjoint);
    let last = before_leases.len() - 1;
    let slot_i = before_leases[idx].mw_slot;
    assert forall|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() implies {
        let l = after_leases[i];
        let w = after_pool.slots[l.mw_slot as int].window;
        &&& CONTROL_WR_ID_BASE <= l.wr_id < next_control_wr_id
        &&& (l.state == RemoteLeaseState::Binding && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is BindWindow
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->BindWindow_window_id
                == win.id())
        }
        &&& ((l.state == RemoteLeaseState::Unbinding || l.state == RemoteLeaseState::Revoking)
            && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is LocalInvalidate
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->LocalInvalidate_window_id
                == win.id())
        }
    } by {
        let bi = if i == idx {
            last
        } else {
            i
        };
        assert(bi != idx);
        assert(after_leases[i] == before_leases[bi]);
        if bi < idx {
            assert(before_leases[bi].mw_slot != slot_i);
        } else {
            assert(slot_i != before_leases[bi].mw_slot);
        }
        assert(after_pool.slots[after_leases[i].mw_slot as int]
            == before_pool.slots[before_leases[bi].mw_slot as int]);
        if let Some(rw) = removed_wr_id {
            if bi < idx {
                assert(before_leases[bi].wr_id != before_leases[idx].wr_id);
            } else {
                assert(before_leases[idx].wr_id != before_leases[bi].wr_id);
            }
            assert(after_leases[i].wr_id != rw);
        }
    }
    assert forall|i: int, j: int|
        0 <= i < j < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() && (
        #[trigger] after_leases[j]).has_active_wr() implies after_leases[i].wr_id
        != after_leases[j].wr_id by {
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
        if bi < bj {
            assert(before_leases[bi].wr_id != before_leases[bj].wr_id);
        } else {
            assert(before_leases[bj].wr_id != before_leases[bi].wr_id);
        }
    }
}

/// Preserve `rdma_control_leases_match_cq` when pushing a new active control lease at
/// `wr_id == before_next_wr_id` and inserting `(wr_id, req)` into `cq`.
pub(super) proof fn lemma_control_leases_match_cq_after_push_bind<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    before_next_wr_id: u64,
    new_lease: ActiveRemoteLease,
    req: rdma::PendingWorkRequest,
)
    requires
        rdma_control_leases_match_cq(before_leases, before_pool, before_cq, before_next_wr_id),
        CONTROL_WR_ID_BASE <= before_next_wr_id < CONTROL_WR_ID_LIMIT,
        after_leases == before_leases.push(new_lease),
        new_lease.state == RemoteLeaseState::Binding,
        new_lease.wr_id == before_next_wr_id,
        after_pool.slots.len() == before_pool.slots.len(),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != new_lease.mw_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
        forall|i: int|
            0 <= i < before_leases.len() ==> (#[trigger] before_leases[i]).mw_slot
                != new_lease.mw_slot,
        after_cq.pending == before_cq.pending.insert(before_next_wr_id, req),
        req.kind is BindWindow,
        (new_lease.mw_slot as int) < after_pool.slots.len(),
        after_pool.slots[new_lease.mw_slot as int].window matches Some(win)
            ==> req.kind->BindWindow_window_id == win.id(),
    ensures
        rdma_control_leases_match_cq(
            after_leases,
            after_pool,
            after_cq,
            (before_next_wr_id + 1) as u64,
        ),
{
    reveal(rdma_control_leases_match_cq);
    broadcast use vstd::map::group_map_lemmas;

    let n = before_leases.len();
    assert forall|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() implies {
        let l = after_leases[i];
        let w = after_pool.slots[l.mw_slot as int].window;
        &&& CONTROL_WR_ID_BASE <= l.wr_id < (before_next_wr_id + 1) as u64
        &&& (l.state == RemoteLeaseState::Binding && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is BindWindow
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->BindWindow_window_id
                == win.id())
        }
        &&& ((l.state == RemoteLeaseState::Unbinding || l.state == RemoteLeaseState::Revoking)
            && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is LocalInvalidate
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->LocalInvalidate_window_id
                == win.id())
        }
    } by {
        if i < n {
            assert(after_leases[i] == before_leases[i]);
            assert(before_leases[i].wr_id < before_next_wr_id);
            assert(after_leases[i].wr_id != before_next_wr_id);
            assert(after_pool.slots[after_leases[i].mw_slot as int]
                == before_pool.slots[before_leases[i].mw_slot as int]);
        }
    }
    assert forall|i: int, j: int|
        0 <= i < j < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() && (
        #[trigger] after_leases[j]).has_active_wr() implies after_leases[i].wr_id
        != after_leases[j].wr_id by {
        if j < n {
            assert(before_leases[i].wr_id != before_leases[j].wr_id);
        } else {
            assert(before_leases[i].wr_id < before_next_wr_id);
            assert(after_leases[j].wr_id == before_next_wr_id);
        }
    }
}

/// Preserve `rdma_control_leases_match_cq` when updating lease `idx` in place to a
/// newly-posted control WR (`Unbinding` or `Revoking`) at `wr_id == before_next_wr_id`.
pub(super) proof fn lemma_control_leases_match_cq_after_update_invalidate<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    before_next_wr_id: u64,
    req: rdma::PendingWorkRequest,
)
    requires
        rdma_control_leases_match_cq(before_leases, before_pool, before_cq, before_next_wr_id),
        rdma_leases_disjoint(before_leases),
        CONTROL_WR_ID_BASE <= before_next_wr_id < CONTROL_WR_ID_LIMIT,
        0 <= idx < before_leases.len(),
        !before_leases[idx].has_active_wr(),
        after_leases.len() == before_leases.len(),
        after_leases[idx].mw_slot == before_leases[idx].mw_slot,
        after_leases[idx].state == RemoteLeaseState::Unbinding || after_leases[idx].state
            == RemoteLeaseState::Revoking,
        after_leases[idx].wr_id == before_next_wr_id,
        forall|j: int|
            0 <= j < before_leases.len() && j != idx ==> (#[trigger] after_leases[j])
                == before_leases[j],
        after_pool.slots.len() == before_pool.slots.len(),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != before_leases[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
        after_cq.pending == before_cq.pending.insert(before_next_wr_id, req),
        req.kind is LocalInvalidate,
        (before_leases[idx].mw_slot as int) < before_pool.slots.len(),
        after_pool.slots[after_leases[idx].mw_slot as int].window matches Some(win)
            ==> req.kind->LocalInvalidate_window_id == win.id(),
    ensures
        rdma_control_leases_match_cq(
            after_leases,
            after_pool,
            after_cq,
            (before_next_wr_id + 1) as u64,
        ),
{
    reveal(rdma_control_leases_match_cq);
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_leases_disjoint);
    let slot_i = before_leases[idx].mw_slot;
    assert forall|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() implies {
        let l = after_leases[i];
        let w = after_pool.slots[l.mw_slot as int].window;
        &&& CONTROL_WR_ID_BASE <= l.wr_id < (before_next_wr_id + 1) as u64
        &&& (l.state == RemoteLeaseState::Binding && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is BindWindow
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->BindWindow_window_id
                == win.id())
        }
        &&& ((l.state == RemoteLeaseState::Unbinding || l.state == RemoteLeaseState::Revoking)
            && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is LocalInvalidate
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->LocalInvalidate_window_id
                == win.id())
        }
    } by {
        if i != idx {
            assert(after_leases[i] == before_leases[i]);
            assert(before_leases[i].wr_id < before_next_wr_id);
            if i < idx {
                assert(before_leases[i].mw_slot != slot_i);
            } else {
                assert(slot_i != before_leases[i].mw_slot);
            }
            assert(after_pool.slots[after_leases[i].mw_slot as int]
                == before_pool.slots[before_leases[i].mw_slot as int]);
        }
    }
    assert forall|i: int, j: int|
        0 <= i < j < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() && (
        #[trigger] after_leases[j]).has_active_wr() implies after_leases[i].wr_id
        != after_leases[j].wr_id by {
        if i != idx && j != idx {
            assert(before_leases[i].wr_id != before_leases[j].wr_id);
        } else if i == idx {
            assert(before_leases[j].wr_id < before_next_wr_id);
        } else {
            assert(before_leases[i].wr_id < before_next_wr_id);
        }
    }
}

/// Preserve `rdma_control_leases_match_cq` when optionally removing `removed_wr_id`
/// from `cq` and updating lease `idx` in place to a state without an active WR.
pub(super) proof fn lemma_control_leases_match_cq_after_cq_remove_update<'a>(
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    next_control_wr_id: u64,
    removed_wr_id: Option<u64>,
)
    requires
        rdma_control_leases_match_cq(before_leases, before_pool, before_cq, next_control_wr_id),
        rdma_leases_disjoint(before_leases),
        0 <= idx < before_leases.len(),
        after_leases.len() == before_leases.len(),
        after_leases[idx].mw_slot == before_leases[idx].mw_slot,
        !after_leases[idx].has_active_wr(),
        forall|j: int|
            0 <= j < before_leases.len() && j != idx ==> (#[trigger] after_leases[j])
                == before_leases[j],
        after_pool.slots.len() == before_pool.slots.len(),
        forall|s: int|
            0 <= s < after_pool.slots.len() && s != before_leases[idx].mw_slot as int ==> (
            #[trigger] after_pool.slots[s]) == before_pool.slots[s],
        removed_wr_id matches Some(rw) ==> before_leases[idx].has_active_wr()
            && before_leases[idx].wr_id == rw,
        after_cq.pending == match removed_wr_id {
            None => before_cq.pending,
            Some(rw) => before_cq.pending.remove(rw),
        },
    ensures
        rdma_control_leases_match_cq(after_leases, after_pool, after_cq, next_control_wr_id),
{
    reveal(rdma_control_leases_match_cq);
    broadcast use vstd::map::group_map_lemmas;

    reveal(rdma_leases_disjoint);
    let slot_i = before_leases[idx].mw_slot;
    assert forall|i: int|
        0 <= i < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() implies {
        let l = after_leases[i];
        let w = after_pool.slots[l.mw_slot as int].window;
        &&& CONTROL_WR_ID_BASE <= l.wr_id < next_control_wr_id
        &&& (l.state == RemoteLeaseState::Binding && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is BindWindow
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->BindWindow_window_id
                == win.id())
        }
        &&& ((l.state == RemoteLeaseState::Unbinding || l.state == RemoteLeaseState::Revoking)
            && after_cq.pending.contains_key(l.wr_id)) ==> {
            &&& after_cq.pending[l.wr_id].kind is LocalInvalidate
            &&& (w matches Some(win) ==> after_cq.pending[l.wr_id].kind->LocalInvalidate_window_id
                == win.id())
        }
    } by {
        assert(i != idx);
        assert(after_leases[i] == before_leases[i]);
        if i < idx {
            assert(before_leases[i].mw_slot != slot_i);
        } else {
            assert(slot_i != before_leases[i].mw_slot);
        }
        assert(after_pool.slots[after_leases[i].mw_slot as int]
            == before_pool.slots[before_leases[i].mw_slot as int]);
        if let Some(rw) = removed_wr_id {
            if i < idx {
                assert(before_leases[i].wr_id != before_leases[idx].wr_id);
            } else {
                assert(before_leases[idx].wr_id != before_leases[i].wr_id);
            }
            assert(after_leases[i].wr_id != rw);
        }
    }
    assert forall|i: int, j: int|
        0 <= i < j < after_leases.len() && (#[trigger] after_leases[i]).has_active_wr() && (
        #[trigger] after_leases[j]).has_active_wr() implies after_leases[i].wr_id
        != after_leases[j].wr_id by {
        assert(i != idx && j != idx);
        assert(before_leases[i].wr_id != before_leases[j].wr_id);
    }
}

/// Preserve `rdma_control_leases_match_cq` whenever CQ pending map shrinks cleanly
/// (`after_cq.pending` is a submap of `before_cq.pending`, e.g., after reaping completions).
pub(super) proof fn lemma_control_leases_match_cq_after_cq_shrink<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    before_cq: rdma::CqSnapshot,
    after_cq: rdma::CqSnapshot,
    next_control_wr_id: u64,
)
    requires
        rdma_control_leases_match_cq(leases, pool, before_cq, next_control_wr_id),
        forall|k: u64| #[trigger]
            after_cq.pending.contains_key(k) ==> before_cq.pending.contains_key(k)
                && after_cq.pending[k] == before_cq.pending[k],
    ensures
        rdma_control_leases_match_cq(leases, pool, after_cq, next_control_wr_id),
{
    reveal(rdma_control_leases_match_cq);
}

/// Preserve `rdma_control_leases_match_cq` and `rdma_pool_slots_owned_by_leases`
/// when `after_pool.slots@ == before_pool.slots@`.
pub(super) proof fn lemma_rdma_opaque_same_pool_slots<'a>(
    leases: Seq<ActiveRemoteLease>,
    before_pool: &WindowPool<'a>,
    after_pool: &WindowPool<'a>,
    cq: rdma::CqSnapshot,
    next_control_wr_id: u64,
)
    requires
        rdma_control_leases_match_cq(leases, before_pool, cq, next_control_wr_id),
        rdma_pool_slots_owned_by_leases(leases, before_pool),
        after_pool.slots@ == before_pool.slots@,
    ensures
        rdma_control_leases_match_cq(leases, after_pool, cq, next_control_wr_id),
        rdma_pool_slots_owned_by_leases(leases, after_pool),
{
    reveal(rdma_control_leases_match_cq);
    reveal(rdma_pool_slots_owned_by_leases);
}

pub(super) proof fn lemma_swap_remove_foreign_shares_same_pins(
    before_core: &CacheCore,
    after_core: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        0 <= idx < before_leases.len(),
        after_leases.len() == before_leases.len() - 1,
        forall|j: int|
            0 <= j < after_leases.len() ==> (#[trigger] after_leases[j]) == if j == idx {
                before_leases[before_leases.len() - 1]
            } else {
                before_leases[j]
            },
        after_core.frames_spec().metas_spec().len() == before_core.frames_spec().metas_spec().len(),
        after_core.frames_spec().preserves_fenced_leases(&before_core.frames_spec()),
    ensures
        peer_foreign_shares_preserved(before_core, after_core, before_leases, after_leases),
{
    assert forall|j: int|
        0 <= j < after_leases.len() && !(#[trigger] after_leases[j]).is_unpinned() implies exists|
        i: int,
    |
        0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
            && before_leases[i].frame == after_leases[j].frame && before_leases[i].share@
            == after_leases[j].share@ by {
        let oi = if j == idx {
            before_leases.len() - 1
        } else {
            j
        };
        assert(after_leases[j] == before_leases[oi]);
        assert(!before_leases[oi].is_unpinned());
    }
    lemma_foreign_shares_same_pins(before_core, after_core, before_leases, after_leases);
}

pub(super) proof fn lemma_foreign_shares_identical_pins(
    before_core: &CacheCore,
    after_core: &CacheCore,
    leases: Seq<ActiveRemoteLease>,
)
    requires
        after_core.frames_spec().metas_spec().len() == before_core.frames_spec().metas_spec().len(),
        after_core.frames_spec().preserves_fenced_leases(&before_core.frames_spec()),
    ensures
        peer_foreign_shares_preserved(before_core, after_core, leases, leases),
{
    assert forall|j: int|
        0 <= j < leases.len() && !(#[trigger] leases[j]).is_unpinned() implies exists|i: int|
        0 <= i < leases.len() && !(#[trigger] leases[i]).is_unpinned() && leases[i].frame
            == leases[j].frame && leases[i].share@ == leases[j].share@ by {
        assert(!leases[j].is_unpinned());
    }
    lemma_foreign_shares_same_pins(before_core, after_core, leases, leases);
}

pub(super) proof fn lemma_swap_remove_foreign_shares_after_unpin(
    before_core: &CacheCore,
    after_core: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        0 <= idx < before_leases.len(),
        !before_leases[idx].is_unpinned(),
        before_leases[idx].frame.0 < before_core.frames_spec().metas_spec().len(),
        after_leases.len() == before_leases.len() - 1,
        forall|j: int|
            0 <= j < after_leases.len() ==> (#[trigger] after_leases[j]) == if j == idx {
                before_leases[before_leases.len() - 1]
            } else {
                before_leases[j]
            },
        after_core.frames_spec().metas_spec().len() == before_core.frames_spec().metas_spec().len(),
        after_core.frames_spec().same_other_pins(
            &before_core.frames_spec(),
            Some(before_leases[idx].frame),
        ),
        after_core.frames_spec().frame(before_leases[idx].frame).fenced_shares
            == before_core.frames_spec().frame(before_leases[idx].frame).fenced_shares.remove(
            before_leases[idx].share@,
        ),
    ensures
        peer_foreign_shares_preserved(before_core, after_core, before_leases, after_leases),
{
    assert forall|j: int|
        0 <= j < after_leases.len() && !(#[trigger] after_leases[j]).is_unpinned() implies exists|
        i: int,
    |
        0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
            && before_leases[i].frame == after_leases[j].frame && before_leases[i].share@
            == after_leases[j].share@ by {
        let oi = if j == idx {
            before_leases.len() - 1
        } else {
            j
        };
        assert(after_leases[j] == before_leases[oi]);
        assert(!before_leases[oi].is_unpinned());
    }
    lemma_foreign_shares_after_unpin(before_core, after_core, before_leases, after_leases, idx);
}

} // verus!
