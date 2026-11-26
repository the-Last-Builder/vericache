//! Verified Milestone 3 RDMA peer session (`PeerSession<'a, T>`):
//! exclusively uses `Channel::recv_any` with constructive `VeriCacheAttachment`
//! frames, driving Donor (`serve_fetch_request`, `handle_read_release`,
//! `revoke_lease_unilateral`), Initiator (`send_fetch_request`,
//! `fetch_remote_frame`), and QP recovery (`reset_and_reconnect`).
#[cfg(verus_only)]
use super::ops::bind_reservations_preserved;
use super::ops::PreparedLeaseBind;
use super::protocol::{
    decode_control_payload, decode_read_grant_meta, decode_read_release_meta,
    encode_control_payload, encode_read_grant_meta, encode_read_release_meta, ControlPayload,
    ReadGrantMeta, ReadReleaseMeta, VeriCacheAttachment,
};
#[cfg(verus_only)]
use super::protocol::{lemma_replica_valid_to_statement, lemma_statement_valid_in_receiver_ledger};
use super::pull::{
    finish_pull_read, finish_pull_read_batch, post_pull_read, post_pull_read_batch,
    InFlightRdmaPull, PreparedPull, PullBatchPost, PullOutcome, PullTarget,
};
#[cfg(verus_only)]
use super::pull::{
    lemma_observed_pull_batch_entry, lemma_posted_pull_batch_entry, observed_pull_batch_matches,
};
use super::*;
use crate::frame::STATE_PUBLISHED;
#[cfg(verus_only)]
use crate::provenance::{
    complete_receipts, event_succeeded, replica_valid, revoked_submap, ValueCertificate,
};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, knowledge_of, ordered_or_absent, Frontier};
#[cfg(verus_only)]
use transfer::context::cluster_version_instance_id;
use transfer::context::{
    EventSourceToken, RevokedMap, ShareCap, ShareSource, SubmittedMap, SucceededMap, SyncContext,
    TransferInstance, VersionBinding,
};
use transfer::rdma;
use transfer::transport::{
    ByteTransport, Channel, ChannelAttachment, ChannelError, EndpointOrFrame, GrantKind, GrantTag,
    Received, ReceivedFrame, Statement,
};
use vstd::prelude::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

proof fn lemma_foreign_shares_identical_leases(
    before: &CacheCore,
    after: &CacheCore,
    leases: Seq<ActiveRemoteLease>,
)
    requires
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
        after.frames_spec().preserves_fenced_leases(&before.frames_spec()),
    ensures
        peer_foreign_shares_preserved(before, after, leases, leases),
{
    assert forall|j: int|
        0 <= j < leases.len() && !(#[trigger] leases[j]).is_unpinned() implies exists|i: int|
        0 <= i < leases.len() && !(#[trigger] leases[i]).is_unpinned() && leases[i].frame
            == leases[j].frame && leases[i].share@ == leases[j].share@ by {
        assert(!leases[j].is_unpinned());
    }
    lemma_foreign_shares_same_pins(before, after, leases, leases);
}

proof fn lemma_poll_bind_witness_matches<'a>(
    rdma: &RdmaBackend<'a>,
    core: &CacheCore,
    handle: RemoteLeaseHandle,
    bind_wr_id: u64,
    before_cq: rdma::CompletionQueue,
    c: rdma::Completed,
)
    requires
        rdma.well_formed(core),
        rdma.lease_index_spec(handle) matches Some(idx) && rdma.active_leases@[idx].wr_id
            == bind_wr_id && rdma.active_leases@[idx].state == RemoteLeaseState::Binding,
        before_cq.observer() == core.node(),
        before_cq.in_flight().contains_key(bind_wr_id),
        before_cq.in_flight()[bind_wr_id].kind is BindWindow,
        rdma.lease_matches_window_state(handle, RemoteLeaseState::Binding) matches Some(w)
            ==> before_cq.in_flight()[bind_wr_id].kind->BindWindow_window_id == w.id(),
        c.describes(before_cq, rdma.cq, core.instance_id()),
        c.id() == bind_wr_id,
    ensures
        c matches rdma::Completed::WindowBound { witness, .. } ==> (rdma.lease_matches_window_state(
            handle,
            RemoteLeaseState::Binding,
        ) matches Some(w) ==> witness@.matches(w)),
        c matches rdma::Completed::BindFailed { witness, .. } ==> (rdma.lease_matches_window_state(
            handle,
            RemoteLeaseState::Binding,
        ) matches Some(w) ==> witness@.matches(w)),
{
    reveal(rdma_leases_valid);
    reveal(window_pool_slots_wf);
}

proof fn lemma_poll_inv_witness_matches<'a>(
    rdma: &RdmaBackend<'a>,
    core: &CacheCore,
    handle: RemoteLeaseHandle,
    expected_state: RemoteLeaseState,
    inv_wr_id: u64,
    before_cq: rdma::CompletionQueue,
    c: rdma::Completed,
)
    requires
        rdma.well_formed(core),
        expected_state == RemoteLeaseState::Unbinding || expected_state
            == RemoteLeaseState::Revoking,
        rdma.lease_index_spec(handle) matches Some(idx) && rdma.active_leases@[idx].wr_id
            == inv_wr_id && rdma.active_leases@[idx].state == expected_state,
        before_cq.observer() == core.node(),
        before_cq.in_flight().contains_key(inv_wr_id),
        before_cq.in_flight()[inv_wr_id].kind is LocalInvalidate,
        rdma.lease_matches_window_state(handle, expected_state) matches Some(w)
            ==> before_cq.in_flight()[inv_wr_id].kind->LocalInvalidate_window_id == w.id(),
        c.describes(before_cq, rdma.cq, core.instance_id()),
        c.id() == inv_wr_id,
    ensures
        c matches rdma::Completed::WindowInvalidated { witness, .. } ==> (
        rdma.lease_matches_window_state(handle, expected_state) matches Some(w)
            ==> witness@.matches(w)),
{
    reveal(rdma_leases_valid);
    reveal(window_pool_slots_wf);
}

proof fn lemma_poll_keeps_bind(
    before: rdma::CompletionQueue,
    after: rdma::CompletionQueue,
    c: rdma::Completed,
    id: InstanceId,
    keep: u64,
)
    requires
        c.describes(before, after, id),
        before.in_flight().contains_key(keep),
        before.in_flight()[keep].kind is BindWindow,
        !(c matches rdma::Completed::WindowBound { wr_id, .. } && wr_id == keep),
        !(c matches rdma::Completed::BindFailed { wr_id, .. } && wr_id == keep),
        !(c matches rdma::Completed::ControlFailed { wr_id, .. } && wr_id == keep),
    ensures
        after.in_flight().contains_key(keep),
        after.in_flight()[keep] == before.in_flight()[keep],
{
    assert(c.id() != keep);
}

proof fn lemma_poll_keeps_data(
    before: rdma::CompletionQueue,
    after: rdma::CompletionQueue,
    c: rdma::Completed,
    id: InstanceId,
    keep: u64,
)
    requires
        c.describes(before, after, id),
        before.in_flight().contains_key(keep),
        before.in_flight()[keep].kind is DataTransfer,
        !(c matches rdma::Completed::Data { wr_id, .. } && wr_id == keep),
        !(c matches rdma::Completed::DataFailed { wr_id, .. } && wr_id == keep),
    ensures
        after.in_flight().contains_key(keep),
        after.in_flight()[keep] == before.in_flight()[keep],
{
    assert(c.id() != keep);
}

proof fn lemma_swap_removed_grant(old: Seq<ActiveGrant>, new: Seq<ActiveGrant>, pos: int)
    requires
        0 <= pos < old.len(),
        new == old.update(pos, old.last()).drop_last(),
        forall|i: int, j: int|
            0 <= i < j < old.len() ==> (#[trigger] old[i]).grant_tag != (
            #[trigger] old[j]).grant_tag && old[i].handle != old[j].handle,
    ensures
        forall|i: int|
            0 <= i < new.len() ==> (#[trigger] new[i]) == old[if i == pos {
                old.len() - 1
            } else {
                i
            }],
        forall|i: int|
            0 <= i < new.len() ==> (#[trigger] new[i]).handle != old[pos].handle && new[i].grant_tag
                != old[pos].grant_tag,
        forall|i: int, j: int|
            0 <= i < j < new.len() ==> (#[trigger] new[i]).grant_tag != (
            #[trigger] new[j]).grant_tag && new[i].handle != new[j].handle,
{
    let last = old.len() - 1;
    assert forall|i: int| 0 <= i < new.len() implies (#[trigger] new[i]).handle != old[pos].handle
        && new[i].grant_tag != old[pos].grant_tag by {
        let k = if i == pos {
            last
        } else {
            i
        };
        assert(new[i] == old[k]);
        if k < pos {
            assert(old[k].handle != old[pos].handle);
        } else {
            assert(old[pos].handle != old[k].handle);
        }
    }
    assert forall|i: int, j: int| 0 <= i < j < new.len() implies (#[trigger] new[i]).grant_tag != (
    #[trigger] new[j]).grant_tag && new[i].handle != new[j].handle by {
        let ki = if i == pos {
            last
        } else {
            i
        };
        let kj = if j == pos {
            last
        } else {
            j
        };
        assert(new[i] == old[ki]);
        assert(new[j] == old[kj]);
        assert(ki != kj);
        if ki < kj {
            assert(old[ki].handle != old[kj].handle);
        } else {
            assert(old[kj].handle != old[ki].handle);
        }
    }
}

proof fn lemma_lease_index_preserved_prefix(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    h: RemoteLeaseHandle,
    idx: int,
)
    requires
        rdma_leases_disjoint(after),
        0 <= idx < before.len(),
        before.len() <= after.len(),
        after.take(before.len() as int) == before,
        before[idx].mw_slot as usize == h.mw_slot && before[idx].generation == h.generation,
    ensures
        rdma_lease_index_spec(after, h) == Some(idx),
{
    reveal(rdma_leases_disjoint);
    assert(after[idx] == before[idx]);
    assert(exists|i: int|
        0 <= i < after.len() && (#[trigger] after[i]).mw_slot as usize == h.mw_slot
            && after[i].generation == h.generation);
    let chosen = rdma_lease_index_spec(after, h).unwrap();
    if chosen != idx {
        if chosen < idx {
            assert(after[chosen].mw_slot != after[idx].mw_slot);
        } else {
            assert(after[idx].mw_slot != after[chosen].mw_slot);
        }
    }
}

proof fn lemma_lease_index_preserved_update(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    changed_idx: int,
    h: RemoteLeaseHandle,
    idx: int,
)
    requires
        rdma_leases_disjoint(after),
        after.len() == before.len(),
        0 <= idx < before.len(),
        idx != changed_idx,
        forall|j: int|
            0 <= j < before.len() && j != changed_idx ==> (#[trigger] after[j]) == before[j],
        before[idx].mw_slot as usize == h.mw_slot && before[idx].generation == h.generation,
    ensures
        rdma_lease_index_spec(after, h) == Some(idx),
        after[idx] == before[idx],
{
    reveal(rdma_leases_disjoint);
    assert(after[idx] == before[idx]);
    assert(exists|i: int|
        0 <= i < after.len() && (#[trigger] after[i]).mw_slot as usize == h.mw_slot
            && after[i].generation == h.generation);
    let chosen = rdma_lease_index_spec(after, h).unwrap();
    if chosen != idx {
        if chosen < idx {
            assert(after[chosen].mw_slot != after[idx].mw_slot);
        } else {
            assert(after[idx].mw_slot != after[chosen].mw_slot);
        }
    }
}

proof fn lemma_lease_index_preserved_remove(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    removed: RemoteLeaseHandle,
    h: RemoteLeaseHandle,
    idx: int,
)
    requires
        rdma_leases_disjoint(after),
        rdma_leases_preserved_except(before, after, removed),
        0 <= idx < before.len(),
        before[idx].mw_slot as usize == h.mw_slot && before[idx].generation == h.generation,
        h != removed,
    ensures
        rdma_lease_index_spec(after, h) matches Some(j) && after[j] == before[idx],
{
    reveal(rdma_leases_disjoint);
    assert(!(before[idx].mw_slot as usize == removed.mw_slot && before[idx].generation
        == removed.generation));
    let j = choose|j: int| 0 <= j < after.len() && (#[trigger] after[j]) == before[idx];
    assert(exists|i: int|
        0 <= i < after.len() && (#[trigger] after[i]).mw_slot as usize == h.mw_slot
            && after[i].generation == h.generation);
    let chosen = rdma_lease_index_spec(after, h).unwrap();
    if chosen != j {
        if chosen < j {
            assert(after[chosen].mw_slot != after[j].mw_slot);
        } else {
            assert(after[j].mw_slot != after[chosen].mw_slot);
        }
    }
}

proof fn lemma_rdma_leases_after_begin_write<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        rdma_leases_valid(leases, pool, qp, before_core),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.instance_id() == before_core.instance_id(),
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_core.frames_spec().same_other_pins(&before_core.frames_spec(), None),
    ensures
        rdma_leases_valid(leases, pool, qp, after_core),
{
    reveal(rdma_leases_valid);
    reveal(FrameTable::same_directory);
    reveal(FrameTable::same_other_pins);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        pool,
        qp,
        after_core,
    ) by {
        assert(leases[i].valid(pool, qp, before_core));
        let f = leases[i].frame;
        assert(after_core.frames_spec().frame(f).slot == before_core.frames_spec().frame(f).slot);
        assert(after_core.frames_spec().published(f) == before_core.frames_spec().published(f));
        assert(after_core.frames_spec().frame(f).fenced_shares == before_core.frames_spec().frame(
            f,
        ).fenced_shares);
        assert(after_core.frames_spec().meta(f).active_fenced_leases
            == before_core.frames_spec().meta(f).active_fenced_leases);
    }
}

proof fn lemma_peer_grants_after_lease_update(
    grants: Seq<ActiveGrant>,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    changed_handle: RemoteLeaseHandle,
    changed_idx: int,
    sent: Map<GrantTag, GrantKind>,
    next_grant_tag: GrantTag,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        0 <= changed_idx < before_leases.len(),
        after_leases.len() == before_leases.len(),
        rdma_lease_index_spec(before_leases, changed_handle) == Some(changed_idx),
        rdma_leases_disjoint(after_leases),
        forall|k: int|
            0 <= k < before_leases.len() && k != changed_idx ==> (#[trigger] after_leases[k])
                == before_leases[k],
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.handle != changed_handle
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < before_core.frames_spec().metas_spec().len()
                &&& rdma_lease_index_spec(before_leases, g.handle) matches Some(idx) && {
                    let lease = before_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: before_core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
    ensures
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
                &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
                    let lease = after_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: after_core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
{
    reveal(FrameTable::same_directory);
    assert forall|i: int| 0 <= i < grants.len() implies {
        let g = #[trigger] grants[i];
        &&& g.grant_tag < next_grant_tag
        &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
        &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
            let lease = after_leases[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& sent.contains_key(g.grant_tag)
            &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                slot: after_core.frames_spec().frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = grants[i];
        let old_idx = rdma_lease_index_spec(before_leases, g.handle).unwrap();
        assert(old_idx != changed_idx);
        lemma_lease_index_preserved_update(
            before_leases,
            after_leases,
            changed_idx,
            g.handle,
            old_idx,
        );
    }
}

proof fn lemma_peer_grants_after_lease_remove(
    grants: Seq<ActiveGrant>,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    removed_handle: RemoteLeaseHandle,
    sent: Map<GrantTag, GrantKind>,
    next_grant_tag: GrantTag,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        after_core.frames_spec().same_directory(&before_core.frames_spec()),
        rdma_leases_disjoint(after_leases),
        rdma_leases_preserved_except(before_leases, after_leases, removed_handle),
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.handle != removed_handle
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < before_core.frames_spec().metas_spec().len()
                &&& rdma_lease_index_spec(before_leases, g.handle) matches Some(idx) && {
                    let lease = before_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: before_core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
    ensures
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
                &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
                    let lease = after_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: after_core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
{
    reveal(FrameTable::same_directory);
    assert forall|i: int| 0 <= i < grants.len() implies {
        let g = #[trigger] grants[i];
        &&& g.grant_tag < next_grant_tag
        &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
        &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
            let lease = after_leases[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& sent.contains_key(g.grant_tag)
            &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                slot: after_core.frames_spec().frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = grants[i];
        let old_idx = rdma_lease_index_spec(before_leases, g.handle).unwrap();
        lemma_lease_index_preserved_remove(
            before_leases,
            after_leases,
            removed_handle,
            g.handle,
            old_idx,
        );
    }
}

proof fn lemma_peer_grants_same_layout(
    grants: Seq<ActiveGrant>,
    leases: Seq<ActiveRemoteLease>,
    sent: Map<GrantTag, GrantKind>,
    next_grant_tag: GrantTag,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < before_core.frames_spec().metas_spec().len()
                &&& rdma_lease_index_spec(leases, g.handle) matches Some(idx) && {
                    let lease = leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: before_core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
    ensures
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
                &&& rdma_lease_index_spec(leases, g.handle) matches Some(idx) && {
                    let lease = leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: after_core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
{
    reveal(FrameTable::same_layout);
    assert forall|i: int| 0 <= i < grants.len() implies {
        let g = #[trigger] grants[i];
        &&& g.grant_tag < next_grant_tag
        &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
        &&& rdma_lease_index_spec(leases, g.handle) matches Some(idx) && {
            let lease = leases[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& sent.contains_key(g.grant_tag)
            &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                slot: after_core.frames_spec().frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = grants[i];
        assert(after_core.frames_spec().frame(g.frame).slot == before_core.frames_spec().frame(
            g.frame,
        ).slot);
    }
}

/// Monotonicity helper: `statement_replica_valid` lifts from bind-time snapshots
/// `(sub1, succ1, facts1)` to any larger `(sub2, succ2, facts2)` satisfying `core.well_formed()`.
proof fn lemma_statement_replica_valid_monotone(
    sub1: SubmittedMap,
    sub2: SubmittedMap,
    succ1: SucceededMap,
    succ2: SucceededMap,
    revoked: RevokedMap,
    facts1: ISet<TransferEvent>,
    facts2: ISet<TransferEvent>,
    cert: ValueCertificate,
)
    requires
        super::protocol::statement_replica_valid(sub1, succ1, revoked, facts1, cert),
        sub1.map().submap_of(sub2.map()),
        succ1.dom().subset_of(succ2.dom()),
        facts1.subset_of(facts2),
        facts2.subset_of(sub2.dom()),
        forall|e: TransferEvent| #[trigger] succ2.contains(e) ==> sub2.dom().contains(e),
    ensures
        super::protocol::statement_replica_valid(sub2, succ2, revoked, facts2, cert),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    assert(sub1.dom().contains(cert.writer));
    assert(sub2.dom().contains(cert.writer));
    if !cert.writer.source.succeeds_on_observe() {
        assert(succ1.contains(cert.writer));
        assert(succ1.dom().contains(cert.writer));
        assert(succ2.dom().contains(cert.writer));
    }
    assert forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && super::protocol::statement_consistent_with_history(
            sub2,
            succ2,
            revoked,
            facts2,
            h,
            run,
        ) implies #[trigger] crate::provenance::certificate_valid_in(cert, h, run) by {
        assert forall|e: TransferEvent| #[trigger] sub1.dom().contains(e) implies {
            let info = sub1.map()[e];
            &&& sub2.dom().contains(e)
            &&& sub2.map()[e] == info
        } by {}
        assert forall|e: TransferEvent|
            e.source.succeeds_on_observe() && #[trigger] facts1.contains(
                e,
            ) implies sub1.dom().contains(e) && (run.succeeded)(e) && (run.end)(e).is_some() by {
            assert(facts2.contains(e));
        }
        assert forall|e: TransferEvent|
            !e.source.succeeds_on_observe() && #[trigger] succ1.contains(
                e,
            ) implies sub1.dom().contains(e) && (run.succeeded)(e) && (run.end)(e).is_some() by {
            assert(succ1.dom().contains(e));
            assert(succ2.dom().contains(e));
            assert(succ2.contains(e));
        }
        assert(super::protocol::statement_consistent_with_history(
            sub1,
            succ1,
            revoked,
            facts1,
            h,
            run,
        ));
    }
}

/// Preserve `pending_outbound_grants` validity across any step leaving their binding-like
/// RDMA leases valid (with `same_layout` + ledger growth).
proof fn lemma_outbound_grants_preserved(
    pending: Seq<PendingOutboundGrant>,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    before_core: &CacheCore,
    after_core: &CacheCore,
)
    requires
        rdma_leases_disjoint(after_leases),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.instance_id() == before_core.instance_id(),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        before_core.ledger_spec()@.submitted.map().submap_of(
            after_core.ledger_spec()@.submitted.map(),
        ),
        before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
        before_core.ledger_spec()@.succeeded.dom().subset_of(
            after_core.ledger_spec()@.succeeded.dom(),
        ),
        forall|i: int|
            0 <= i < pending.len() ==> {
                let p = #[trigger] pending[i];
                &&& p.frame.0 < before_core.frames_spec().metas_spec().len()
                &&& p.frame_size == before_core.frames_spec().tier_frame_size(
                    before_core.frames_spec().frame(p.frame).tier,
                ) as u64
                &&& p.cert@.slot == before_core.frames_spec().frame(p.frame).slot
                &&& p.cert@.slot.is_full()
                &&& p.cert@.slot.well_formed()
                &&& p.cert@.bytes.len() == p.frame_size
                &&& p.bind_revoked@.instance_id() == before_core.instance_id()
                &&& p.bind_sub@.map().submap_of(before_core.ledger_spec()@.submitted.map())
                &&& p.bind_succ@.dom().subset_of(before_core.ledger_spec()@.succeeded.dom())
                &&& (forall|e: TransferEvent| #[trigger]
                    p.bind_facts@.contains(e)
                        ==> before_core.ledger_spec()@.submitted.map().contains_key(e)
                        && before_core.ledger_spec()@.knows.set().contains((before_core.node(), e)))
                &&& p.bind_facts@.contains(p.cert@.writer)
                &&& super::protocol::statement_replica_valid(
                    p.bind_sub@,
                    p.bind_succ@,
                    p.bind_revoked@,
                    p.bind_facts@,
                    p.cert@,
                )
                &&& (p.version_binding@ matches Some(vb) && vb.instance_id()
                    == cluster_version_instance_id() && vb.key() == (
                    p.version.origin_node as NodeId,
                    p.version.seq,
                ) && vb.value() == p.cert@.bytes)
                &&& rdma_lease_index_spec(before_leases, p.handle) matches Some(bidx) && {
                    let bl = before_leases[bidx];
                    &&& bl.frame == p.frame
                    &&& bl.wr_id == p.bind_wr_id
                    &&& super::poll::rdma_lease_is_binding_like(bl)
                    &&& bl.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(
                        p.cert@.writer,
                    )
                    &&& rdma_lease_index_spec(after_leases, p.handle) matches Some(aidx) && {
                        let al = after_leases[aidx];
                        &&& al.frame == bl.frame
                        &&& al.wr_id == bl.wr_id
                        &&& super::poll::rdma_lease_is_binding_like(al)
                        &&& al.share_cap == bl.share_cap
                    }
                }
            },
    ensures
        forall|i: int|
            0 <= i < pending.len() ==> {
                let p = #[trigger] pending[i];
                &&& p.frame.0 < after_core.frames_spec().metas_spec().len()
                &&& p.frame_size == after_core.frames_spec().tier_frame_size(
                    after_core.frames_spec().frame(p.frame).tier,
                ) as u64
                &&& p.cert@.slot == after_core.frames_spec().frame(p.frame).slot
                &&& p.cert@.slot.is_full()
                &&& p.cert@.slot.well_formed()
                &&& p.cert@.bytes.len() == p.frame_size
                &&& p.bind_revoked@.instance_id() == after_core.instance_id()
                &&& p.bind_sub@.map().submap_of(after_core.ledger_spec()@.submitted.map())
                &&& p.bind_succ@.dom().subset_of(after_core.ledger_spec()@.succeeded.dom())
                &&& (forall|e: TransferEvent| #[trigger]
                    p.bind_facts@.contains(e)
                        ==> after_core.ledger_spec()@.submitted.map().contains_key(e)
                        && after_core.ledger_spec()@.knows.set().contains((after_core.node(), e)))
                &&& p.bind_facts@.contains(p.cert@.writer)
                &&& super::protocol::statement_replica_valid(
                    p.bind_sub@,
                    p.bind_succ@,
                    p.bind_revoked@,
                    p.bind_facts@,
                    p.cert@,
                )
                &&& (p.version_binding@ matches Some(vb) && vb.instance_id()
                    == cluster_version_instance_id() && vb.key() == (
                    p.version.origin_node as NodeId,
                    p.version.seq,
                ) && vb.value() == p.cert@.bytes)
                &&& rdma_lease_index_spec(after_leases, p.handle) matches Some(aidx) && {
                    let al = after_leases[aidx];
                    &&& al.frame == p.frame
                    &&& al.wr_id == p.bind_wr_id
                    &&& super::poll::rdma_lease_is_binding_like(al)
                    &&& al.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(
                        p.cert@.writer,
                    )
                }
            },
{
    assert forall|i: int| 0 <= i < pending.len() implies {
        let p = #[trigger] pending[i];
        &&& p.frame.0 < after_core.frames_spec().metas_spec().len()
        &&& p.frame_size == after_core.frames_spec().tier_frame_size(
            after_core.frames_spec().frame(p.frame).tier,
        ) as u64
        &&& p.cert@.slot == after_core.frames_spec().frame(p.frame).slot
        &&& p.cert@.slot.is_full()
        &&& p.cert@.slot.well_formed()
        &&& p.cert@.bytes.len() == p.frame_size
        &&& p.bind_revoked@.instance_id() == after_core.instance_id()
        &&& p.bind_sub@.map().submap_of(after_core.ledger_spec()@.submitted.map())
        &&& p.bind_succ@.dom().subset_of(after_core.ledger_spec()@.succeeded.dom())
        &&& (forall|e: TransferEvent| #[trigger]
            p.bind_facts@.contains(e) ==> after_core.ledger_spec()@.submitted.map().contains_key(e)
                && after_core.ledger_spec()@.knows.set().contains((after_core.node(), e)))
        &&& p.bind_facts@.contains(p.cert@.writer)
        &&& super::protocol::statement_replica_valid(
            p.bind_sub@,
            p.bind_succ@,
            p.bind_revoked@,
            p.bind_facts@,
            p.cert@,
        )
        &&& (p.version_binding@ matches Some(vb) && vb.instance_id()
            == cluster_version_instance_id() && vb.key() == (
            p.version.origin_node as NodeId,
            p.version.seq,
        ) && vb.value() == p.cert@.bytes)
        &&& rdma_lease_index_spec(after_leases, p.handle) matches Some(aidx) && {
            let al = after_leases[aidx];
            &&& al.frame == p.frame
            &&& al.wr_id == p.bind_wr_id
            &&& super::poll::rdma_lease_is_binding_like(al)
            &&& al.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(p.cert@.writer)
        }
    } by {
        broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        let p = pending[i];
        let bidx = rdma_lease_index_spec(before_leases, p.handle).unwrap();
        let aidx = rdma_lease_index_spec(after_leases, p.handle).unwrap();
        assert(after_leases[aidx].frame == before_leases[bidx].frame);
        assert(after_leases[aidx].wr_id == before_leases[bidx].wr_id);
        assert(super::poll::rdma_lease_is_binding_like(after_leases[aidx]));
        assert(after_leases[aidx].share_cap == before_leases[bidx].share_cap);
        vstd::imap_lib::lemma_submap_of_trans(
            p.bind_sub@.map(),
            before_core.ledger_spec()@.submitted.map(),
            after_core.ledger_spec()@.submitted.map(),
        );
        assert(p.bind_succ@.dom().subset_of(after_core.ledger_spec()@.succeeded.dom())) by {
            assert forall|k: TransferEvent| #[trigger]
                p.bind_succ@.dom().contains(
                    k,
                ) implies after_core.ledger_spec()@.succeeded.dom().contains(k) by {
                assert(before_core.ledger_spec()@.succeeded.dom().contains(k));
            };
        };
        assert forall|e: TransferEvent| #[trigger]
            p.bind_facts@.contains(
                e,
            ) implies after_core.ledger_spec()@.submitted.map().contains_key(e)
            && after_core.ledger_spec()@.knows.set().contains((after_core.node(), e)) by {
            assert(before_core.ledger_spec()@.submitted.map().dom().contains(e));
            assert(before_core.ledger_spec()@.submitted.map().contains_key(e));
            assert(after_core.ledger_spec()@.submitted.map().dom().contains(e));
            assert(before_core.ledger_spec()@.knows.set().contains((before_core.node(), e)));
        };
        assert(p.version_binding@@ matches Some(vb) && vb.instance_id()
            == cluster_version_instance_id() && vb.key() == (
            p.version.origin_node as NodeId,
            p.version.seq,
        ) && vb.value() == p.cert@.bytes);
    }
}

/// Donor-side record mapping a sent `GrantTag` to its bound RDMA lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveGrant {
    pub grant_tag: GrantTag,
    pub handle: RemoteLeaseHandle,
    pub frame: FrameId,
    pub soft_revoking: bool,
}

/// Multiplexed peer session progress counters returned by `PeerSession::poll_progress`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerProgress {
    pub cq_processed: usize,
    pub frames_received: usize,
    pub releases_processed: usize,
    pub grants_completed: usize,
    pub windows_replenished: usize,
}

impl PeerProgress {
    pub fn empty() -> (r: Self)
        ensures
            r.cq_processed == 0,
            r.frames_received == 0,
            r.releases_processed == 0,
            r.grants_completed == 0,
            r.windows_replenished == 0,
    {
        Self {
            cq_processed: 0,
            frames_received: 0,
            releases_processed: 0,
            grants_completed: 0,
            windows_replenished: 0,
        }
    }

    /// Combine counters from multiple peers or pump passes.
    pub fn accumulate(&mut self, other: Self) {
        self.cq_processed = self.cq_processed.saturating_add(other.cq_processed);
        self.frames_received = self.frames_received.saturating_add(other.frames_received);
        self.releases_processed = self.releases_processed.saturating_add(other.releases_processed);
        self.grants_completed = self.grants_completed.saturating_add(other.grants_completed);
        self.windows_replenished = self.windows_replenished.saturating_add(
            other.windows_replenished,
        );
    }
}

/// Outbound MW-bind response queued on `PeerSession` awaiting CQ completion.
pub struct PendingOutboundGrant {
    pub request_id: u64,
    pub key: ObjectKey,
    pub frame: FrameId,
    pub version: VersionId,
    pub frame_size: u64,
    pub handle: RemoteLeaseHandle,
    pub bind_wr_id: u64,
    pub send_on_ready: bool,
    pub cert: Ghost<ValueCertificate>,
    pub bind_sub: Ghost<SubmittedMap>,
    pub bind_succ: Ghost<SucceededMap>,
    pub bind_revoked: Tracked<RevokedMap>,
    pub bind_facts: Ghost<ISet<TransferEvent>>,
    pub version_binding: Tracked<Option<VersionBinding>>,
}

impl PendingOutboundGrant {
    pub open(crate) spec fn valid(&self, rdma: &RdmaBackend, core: &CacheCore) -> bool {
        &&& self.frame.0 < core.frames_spec().metas_spec().len()
        &&& self.frame_size == core.frames_spec().tier_frame_size(
            core.frames_spec().frame(self.frame).tier,
        ) as u64
        &&& self.cert@.slot == core.frames_spec().frame(self.frame).slot
        &&& self.cert@.slot.is_full()
        &&& self.cert@.slot.well_formed()
        &&& self.cert@.bytes.len() == self.frame_size
        &&& self.bind_revoked@.instance_id() == core.instance_id()
        &&& self.bind_sub@.map().submap_of(core.ledger_spec()@.submitted.map())
        &&& self.bind_succ@.dom().subset_of(core.ledger_spec()@.succeeded.dom())
        &&& forall|e: TransferEvent| #[trigger]
            self.bind_facts@.contains(e) ==> core.ledger_spec()@.submitted.map().contains_key(e)
                && core.ledger_spec()@.knows.set().contains((core.node(), e))
        &&& self.bind_facts@.contains(self.cert@.writer)
        &&& super::protocol::statement_replica_valid(
            self.bind_sub@,
            self.bind_succ@,
            self.bind_revoked@,
            self.bind_facts@,
            self.cert@,
        )
        &&& (self.version_binding@ matches Some(vb) && vb.instance_id()
            == cluster_version_instance_id() && vb.key() == (
            self.version.origin_node as NodeId,
            self.version.seq,
        ) && vb.value() == self.cert@.bytes)
        &&& rdma.lease_index_spec(self.handle) matches Some(idx) && {
            let lease = rdma.active_leases@[idx];
            &&& lease.frame == self.frame
            &&& lease.wr_id == self.bind_wr_id
            &&& super::poll::rdma_lease_is_binding_like(lease)
            &&& lease.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(
                self.cert@.writer,
            )
        }
    }
}

/// Buffered inbound `READ_GRANT` received out-of-order on `Channel`.
pub struct PendingInboundGrant {
    pub tag: GrantTag,
    pub meta: ReadGrantMeta,
    pub cap: rdma::RemoteCapability,
    pub share: Tracked<ShareCap>,
    pub statement: Tracked<Statement>,
    pub cert: Ghost<ValueCertificate>,
    pub revoked: Tracked<RevokedMap>,
    pub version_binding: Tracked<Option<VersionBinding>>,
}

impl PendingInboundGrant {
    pub open spec fn valid<T: ByteTransport>(
        &self,
        core: &CacheCore,
        channel: &Channel<T, VeriCacheAttachment>,
        peer_node: NodeId,
    ) -> bool {
        &&& channel.received().contains_key(self.tag)
        &&& channel.received()[self.tag] == (GrantKind::ReadShare {
            slot: self.cap.slot_ref(),
            share_id: self.share@.share(),
        })
        &&& self.statement@.valid(core.instance_id(), core.node())
        &&& self.statement@.knows.set() == knowledge_of(core.node(), self.statement@.closure())
        &&& self.cap.well_formed()
        &&& self.cap.slot_ref().loc.node == peer_node
        &&& self.cap.access() == rdma::RemoteAccess::ReadOnly
        &&& self.cap.share() == Some(self.share@.share())
        &&& self.cap.slot_ref() == self.cert@.slot
        &&& self.cert@.slot.is_full()
        &&& self.cert@.bytes.len() == self.meta.frame_size
        &&& self.share@.is_fenced()
        &&& self.share@.instance_id() == core.instance_id()
        &&& self.share@.loc() == self.cap.slot_ref().loc
        &&& self.share@.frontier().last_writer == Some(self.cert@.writer)
        &&& transfer::transport::covers(self.share@.frontier(), self.statement@.closure())
        &&& self.revoked@.instance_id() == core.instance_id()
        &&& crate::provenance::complete_receipts_map(self.statement@.submitted)
        &&& super::protocol::statement_replica_valid(
            self.statement@.submitted,
            self.statement@.succeeded,
            self.revoked@,
            self.statement@.facts,
            self.cert@,
        )
        &&& (self.version_binding@ matches Some(b) && b.instance_id()
            == cluster_version_instance_id() && b.key() == (
            self.meta.version.origin_node as NodeId,
            self.meta.version.seq,
        ) && b.value() == self.cert@.bytes)
    }
}

/// Decoded frame received from `Channel::recv_any`.
#[allow(inconsistent_fields)]
pub enum PeerMessage {
    Control { payload: ControlPayload, statement: Tracked<Statement> },
    ReadGrant {
        tag: GrantTag,
        meta: ReadGrantMeta,
        cap: rdma::RemoteCapability,
        share: Tracked<ShareCap>,
        statement: Tracked<Statement>,
        cert: Ghost<ValueCertificate>,
        revoked: Tracked<RevokedMap>,
        version_binding: Tracked<Option<VersionBinding>>,
    },
    ReadRelease {
        tag: GrantTag,
        meta: ReadReleaseMeta,
        share: Tracked<ShareCap>,
        statement: Tracked<Statement>,
    },
    Unexpected,
}

/// Outcome of donor `serve_fetch_request`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeOutcome {
    Granted { grant_tag: GrantTag, handle: RemoteLeaseHandle },
    Miss,
}

/// Outcome of initiator `fetch_remote_frame`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchOutcome {
    Hit { frame: FrameId, version: VersionId },
    Miss,
}

pub struct PullGrantReturn {
    pub tag: GrantTag,
    pub request_id: u64,
    pub binding: Tracked<VersionBinding>,
    pub frontier: Ghost<foundation::tracked::Frontier<TransferDomain>>,
}

impl PullGrantReturn {
    pub open spec fn owns_grant(
        &self,
        received: Map<GrantTag, GrantKind>,
        share: ShareCap,
    ) -> bool {
        &&& received.contains_key(self.tag)
        &&& received[self.tag] is ReadShare
        &&& received[self.tag]->ReadShare_slot.loc == share.loc()
        &&& received[self.tag]->ReadShare_share_id == share.share()
    }

    pub open spec fn matches_version(&self, version: VersionId, bytes: Seq<u8>) -> bool {
        &&& self.binding@.instance_id() == cluster_version_instance_id()
        &&& self.binding@.key() == (version.origin_node as NodeId, version.seq)
        &&& self.binding@.value() == bytes
    }

    pub open spec fn matches_posted(
        &self,
        pull: &InFlightRdmaPull,
        received: Map<GrantTag, GrantKind>,
        core: &CacheCore,
    ) -> bool {
        &&& self.owns_grant(received, pull.share_cap@)
        &&& self.matches_version(pull.handle.version, pull.copy@.source.bytes)
        &&& transfer::transport::covers(self.frontier@, core.citations())
        &&& pull.share_cap@.frontier() == (foundation::tracked::Frontier {
            last_writer: self.frontier@.last_writer,
            readers: self.frontier@.readers.insert(pull.writer@),
        })
    }

    pub open spec fn matches_prepared(
        &self,
        plan: &PreparedPull,
        received: Map<GrantTag, GrantKind>,
        core: &CacheCore,
    ) -> bool {
        &&& self.owns_grant(received, plan.share@)
        &&& self.matches_version(plan.handle.version, plan.source@.bytes)
        &&& transfer::transport::covers(self.frontier@, core.citations())
        &&& plan.share@.frontier() == self.frontier@
    }
}

#[verifier::opaque]
pub(crate) open spec fn pull_batch_targets_distinct(
    pulls: Seq<InFlightRdmaPull>,
    routes: Seq<PullGrantReturn>,
) -> bool {
    forall|a: int, b: int|
        0 <= a < b < pulls.len() ==> {
            &&& (#[trigger] pulls[a]).handle.frame != (#[trigger] pulls[b]).handle.frame
            &&& routes[a].tag != routes[b].tag
        }
}

proof fn lemma_pull_batch_targets_distinct(
    pulls: Seq<InFlightRdmaPull>,
    routes: Seq<PullGrantReturn>,
    a: int,
    b: int,
)
    requires
        pull_batch_targets_distinct(pulls, routes),
        0 <= a < b < pulls.len(),
    ensures
        pulls[a].handle.frame != pulls[b].handle.frame,
        routes[a].tag != routes[b].tag,
{
    reveal(pull_batch_targets_distinct);
}

pub(crate) ghost struct PullBatchView {
    pub pulls: Seq<InFlightRdmaPull>,
    pub routes: Seq<PullGrantReturn>,
    pub statuses: Seq<Result<(), i32>>,
    pub plans: Seq<PreparedPull>,
    pub before: CacheCore,
    pub visible: bool,
}

impl PullBatchView {
    pub(crate) open spec fn valid_entry<'a, T: ByteTransport>(
        self,
        i: int,
        session: &PeerSession<'a, T>,
        core: &CacheCore,
    ) -> bool {
        let p = self.pulls[i];
        &&& p.valid(&session.rdma, core)
        &&& p.handle.frame == self.plans[i].handle.frame
        &&& p.handle.frame.0 < self.before.frames_spec().metas_spec().len()
        &&& self.before.frames_spec().meta(p.handle.frame).state != STATE_INITIALIZING
        &&& self.routes[i].matches_posted(&p, session.channel.received(), core)
        &&& (i < self.statuses.len() ==> observed_batch_status(&p, core, self.statuses[i]))
        &&& (self.visible ==> p.batch_visible())
    }

    #[verifier::opaque]
    pub(crate) open spec fn valid_suffix<'a, T: ByteTransport>(
        self,
        start: int,
        session: &PeerSession<'a, T>,
        core: &CacheCore,
    ) -> bool {
        forall|i: int|
            start <= i < self.pulls.len() ==> #[trigger] self.valid_entry(i, session, core)
    }

    proof fn lemma_entry<'a, T: ByteTransport>(
        self,
        start: int,
        i: int,
        session: &PeerSession<'a, T>,
        core: &CacheCore,
    )
        requires
            start <= i < self.pulls.len(),
            self.valid_suffix(start, session, core),
        ensures
            self.valid_entry(i, session, core),
    {
        reveal(PullBatchView::valid_suffix);
    }
}

pub struct PendingBatchPull {
    pub pull: InFlightRdmaPull,
    pub route: PullGrantReturn,
    pub status: Option<Result<(), i32>>,
}

/// Results carry request IDs since misses and rejected reservations can resolve
/// before a posted read. On CQ or GPU visibility errors, pending pulls stay owned.
pub struct FetchBatchPulls {
    pub results: Vec<(u64, Result<FetchOutcome, CoreError>)>,
    pub pending: Vec<PendingBatchPull>,
    pub wait_error: Option<CoreError>,
}

pub open(crate) spec fn observed_batch_status(
    p: &InFlightRdmaPull,
    core: &CacheCore,
    status: Result<(), i32>,
) -> bool {
    &&& p.completion_observed(core)
    &&& core.citations().contains(p.writer@)
    &&& status.is_ok() ==> event_succeeded(core.ledger_spec()@, core.node(), p.writer@)
}

proof fn lemma_batch_status_growth(
    before: &CacheCore,
    after: &CacheCore,
    t: TimelineId,
    retired: Option<FrameId>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        peer_core_step_except(before, after, t, retired),
    ensures
        before.citations().subset_of(after.citations()),
        forall|p: InFlightRdmaPull, status: Result<(), i32>| #[trigger]
            observed_batch_status(&p, before, status) ==> observed_batch_status(&p, after, status),
{
    hide(CacheCore::well_formed);
    reveal(peer_core_step_except);
    reveal(core_transition);
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    assert forall|e: TransferEvent| #[trigger]
        before.citations().contains(e) implies after.citations().contains(e) by {
        assert(before.ledger_spec()@.submitted.dom().contains(e));
        assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
    }
    assert(before.citations().subset_of(after.citations()));
    assert forall|p: InFlightRdmaPull, status: Result<(), i32>| #[trigger]
        observed_batch_status(&p, before, status) implies observed_batch_status(
        &p,
        after,
        status,
    ) by {
        let writer = p.writer@;
        assert(before.citations().contains(writer));
        assert(after.citations().contains(writer));
        assert(p.completion_observed(after));
        if status.is_ok() && !writer.source.succeeds_on_observe() {
            assert(before.ledger_spec()@.succeeded.dom().contains(writer));
            assert(after.ledger_spec()@.succeeded.dom().contains(writer));
        }
    }
}

/// Combined RDMA backend + authenticated `Channel<T, VeriCacheAttachment>` session
/// with one remote peer node.
fn append_fetch_entries(queue: &mut Vec<(u64, ObjectKey)>, entries: &[(u64, ObjectKey)])
    ensures
        final(queue)@ == old(queue)@ + entries@,
{
    let mut i = 0;
    while i < entries.len()
        invariant
            i <= entries.len(),
            queue@ == old(queue)@ + entries@.take(i as int),
        decreases entries.len() - i,
    {
        queue.push(entries[i]);
        assert(queue@ =~= old(queue)@ + entries@.take(i + 1));
        i += 1;
    }
    assert(entries@.take(i as int) =~= entries@);
}

pub struct PeerSession<'a, T: ByteTransport> {
    pub(crate) rdma: RdmaBackend<'a>,
    pub(crate) channel: Channel<T, VeriCacheAttachment>,
    pub(crate) next_grant_tag: u64,
    pub(crate) next_request_id: u64,
    pub(crate) active_grants: Vec<ActiveGrant>,
    pub(crate) pending_outbound_grants: Vec<PendingOutboundGrant>,
    pub(crate) pending_inbound_fetches: Vec<(u64, ObjectKey)>,
    pub(crate) pending_inbound_grants: Vec<PendingInboundGrant>,
    pub(crate) pending_inbound_misses: Vec<u64>,
    pub(crate) pending_revoke_notices: Vec<GrantTag>,
    pub(crate) pending_revoke_acks: Vec<GrantTag>,
    pub(crate) pending_reconnect_barriers: Vec<u64>,
    pub(crate) quarantined_grants: Vec<(GrantTag, u32)>,
    pub(crate) reconnect_epoch: u64,
    pub(crate) peer_reconnect_epoch: u64,
}

impl<'a, T: ByteTransport> PeerSession<'a, T> {
    pub open(crate) spec fn peer_reconnect_epoch_spec(&self) -> u64 {
        self.peer_reconnect_epoch
    }

    pub open(crate) spec fn quarantined_grants_spec(&self) -> Vec<(GrantTag, u32)> {
        self.quarantined_grants
    }

    pub open(crate) spec fn pending_reconnect_barriers_spec(&self) -> Vec<u64> {
        self.pending_reconnect_barriers
    }

    pub open(crate) spec fn pending_revoke_acks_spec(&self) -> Vec<GrantTag> {
        self.pending_revoke_acks
    }

    pub open(crate) spec fn pending_revoke_notices_spec(&self) -> Vec<GrantTag> {
        self.pending_revoke_notices
    }

    pub open(crate) spec fn pending_inbound_misses_spec(&self) -> Vec<u64> {
        self.pending_inbound_misses
    }

    pub open(crate) spec fn reconnect_epoch_spec(&self) -> u64 {
        self.reconnect_epoch
    }

    pub open(crate) spec fn pending_inbound_grants_spec(&self) -> Vec<PendingInboundGrant> {
        self.pending_inbound_grants
    }

    pub open(crate) spec fn pending_outbound_grants_spec(&self) -> Vec<PendingOutboundGrant> {
        self.pending_outbound_grants
    }

    pub open(crate) spec fn active_grants_spec(&self) -> Vec<ActiveGrant> {
        self.active_grants
    }

    pub open(crate) spec fn pending_inbound_fetches_spec(&self) -> Vec<(u64, ObjectKey)> {
        self.pending_inbound_fetches
    }

    pub open(crate) spec fn channel_spec(&self) -> Channel<T, VeriCacheAttachment> {
        self.channel
    }

    pub open(crate) spec fn next_grant_tag_spec(&self) -> u64 {
        self.next_grant_tag
    }

    pub open(crate) spec fn rdma_spec(&self) -> RdmaBackend<'a> {
        self.rdma
    }

    pub open(crate) spec fn next_request_id_spec(&self) -> u64 {
        self.next_request_id
    }
}

#[verifier::opaque]
pub open spec fn pull_tracked_or_done_state(
    pending: Map<u64, TransferEvent>,
    completed_pulls: [Option<(u64, Result<(), i32>)>; BITMAP_SLOTS],
    seq: u64,
    writer: TransferEvent,
) -> bool {
    &&& (forall|w: u64| #[trigger]
        pending.contains_key(w) && w != seq ==> (w % (BITMAP_CAPACITY as u64)) != (seq % (
        BITMAP_CAPACITY as u64)))
    &&& (pending.contains_key(seq) ==> pending[seq] == writer)
    &&& (pending.contains_key(seq) || match completed_pulls[((seq as int) % (
    BITMAP_CAPACITY as int))] {
        Some((s, _)) => s == seq,
        None => false,
    })
}

pub open(crate) spec fn pull_tracked_or_done<'a>(
    rdma: &RdmaBackend<'a>,
    seq: u64,
    writer: TransferEvent,
) -> bool {
    pull_tracked_or_done_state(rdma.pending(), rdma.completed_pulls, seq, writer)
}

pub open spec fn inbound_tag_unbuffered(grants: Seq<PendingInboundGrant>, k: GrantTag) -> bool {
    forall|i: int| 0 <= i < grants.len() ==> (#[trigger] grants[i]).tag != k
}

pub open spec fn inbound_grant_in_seq(
    seq: Seq<PendingInboundGrant>,
    g: PendingInboundGrant,
) -> bool {
    exists|j: int| 0 <= j < seq.len() && (#[trigger] seq[j]) == g
}

proof fn lemma_same_pulls_preserves_pull_tracked<'a>(
    before: &RdmaBackend<'a>,
    after: &RdmaBackend<'a>,
)
    requires
        after.pending() == before.pending(),
        after.completed_pulls == before.completed_pulls,
    ensures
        forall|seq: u64, writer: TransferEvent| #[trigger]
            pull_tracked_or_done(before, seq, writer) ==> pull_tracked_or_done(after, seq, writer),
{
}

proof fn lemma_pump_cq_all_preserves_pull_tracked<'a>(
    before: &RdmaBackend<'a>,
    after: &RdmaBackend<'a>,
)
    requires
        forall|k: u64| #[trigger]
            after.pending().contains_key(k) ==> (before.pending().contains_key(k)
                && after.pending()[k] == before.pending()[k]),
        forall|k: u64| #[trigger]
            before.pending().contains_key(k) ==> ((after.pending().contains_key(k)
                && after.pending()[k] == before.pending()[k]) || after.completed_pull_spec(
                k,
            ) is Some),
        forall|k: u64| #[trigger]
            before.completed_pull_spec(k) is Some && !before.pending().contains_key(k) && (forall|
                w: u64,
            | #[trigger]
                before.pending().contains_key(w) && w != k ==> (w % (BITMAP_CAPACITY as u64)) != (k
                    % (BITMAP_CAPACITY as u64))) ==> after.completed_pull_spec(k)
                == before.completed_pull_spec(k),
    ensures
        forall|seq: u64, writer: TransferEvent| #[trigger]
            pull_tracked_or_done(before, seq, writer) ==> pull_tracked_or_done(after, seq, writer),
{
    reveal(pull_tracked_or_done_state);
    assert forall|seq: u64, writer: TransferEvent| #[trigger]
        pull_tracked_or_done(before, seq, writer) implies pull_tracked_or_done(
        after,
        seq,
        writer,
    ) by {
        if before.pending().contains_key(seq) {
            assert(before.pending()[seq] == writer);
        } else {
            assert(before.completed_pull_spec(seq) is Some);
        }
    }
}

proof fn lemma_pull_retained_after_peer_step(
    prev: &CacheCore,
    next: &CacheCore,
    rdma_tl: TimelineId,
    copy: &crate::provenance::GuardedReplicaCopy,
    retired: Option<FrameId>,
)
    requires
        prev.well_formed(),
        next.well_formed(),
        peer_core_step_except(prev, next, rdma_tl, retired),
        copy.retained(prev.ledger_spec()@, prev.node()),
    ensures
        prev.citations().subset_of(next.citations()),
        copy.before.citations(next.node()).subset_of(next.citations()),
        copy.retained(next.ledger_spec()@, next.node()),
{
    hide(CacheCore::well_formed);
    reveal(peer_core_step_except);
    reveal(core_transition);
    lemma_core_wf_facts(prev);
    lemma_core_wf_facts(next);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    assert(prev.citations().subset_of(next.citations())) by {
        assert forall|e: TransferEvent| #[trigger]
            prev.citations().contains(e) implies next.citations().contains(e) by {
            assert(prev.ledger_spec()@.knows.set().contains((prev.node(), e)));
        }
    }
    vstd::imap_lib::lemma_submap_of_trans(
        copy.before.submitted.map(),
        prev.ledger_spec()@.submitted.map(),
        next.ledger_spec()@.submitted.map(),
    );
    assert(revoked_submap(copy.before.revoked, next.ledger_spec()@.revoked)) by {
        assert forall|k: (SlotLoc, TransferShare)| #[trigger]
            copy.before.revoked.dom().contains(k) implies (
        next.ledger_spec()@.revoked.dom().contains(k) && next.ledger_spec()@.revoked.map()[k]
            == copy.before.revoked.map()[k]) by {
            assert(prev.ledger_spec()@.revoked.dom().contains(k));
        }
    }
    assert(copy.before.succeeded.dom().subset_of(next.ledger_spec()@.succeeded.dom())) by {
        assert forall|e: TransferEvent| #[trigger]
            copy.before.succeeded.dom().contains(
                e,
            ) implies next.ledger_spec()@.succeeded.dom().contains(e) by {
            assert(prev.ledger_spec()@.succeeded.dom().contains(e));
        }
    }
    assert(copy.before.citations(next.node()).subset_of(next.citations())) by {
        assert forall|e: TransferEvent| #[trigger]
            copy.before.citations(next.node()).contains(e) implies next.citations().contains(e) by {
            assert(prev.citations().contains(e));
        }
    }
    assert(next.ledger_spec()@.submitted.dom().contains(copy.writer));
    assert(next.ledger_spec()@.submitted.dom().contains(copy.source.writer));
    assert(next.ledger_spec()@.submitted.map()[copy.writer]
        == prev.ledger_spec()@.submitted.map()[copy.writer]);
    assert(next.ledger_spec()@.submitted.map()[copy.writer].preds.subset_of(
        next.ledger_spec()@.submitted.dom(),
    )) by {
        assert forall|e: TransferEvent| #[trigger]
            next.ledger_spec()@.submitted.map()[copy.writer].preds.contains(
                e,
            ) implies next.ledger_spec()@.submitted.dom().contains(e) by {
            assert(prev.ledger_spec()@.submitted.dom().contains(e));
        }
    }
}

proof fn lemma_pull_valid_after_peer_step<'a>(
    prev_rdma: &RdmaBackend<'a>,
    next_rdma: &RdmaBackend<'a>,
    prev_core: &CacheCore,
    next_core: &CacheCore,
    pull: &InFlightRdmaPull,
    retired: Option<FrameId>,
)
    requires
        prev_core.well_formed(),
        next_core.well_formed(),
        peer_core_step_except(prev_core, next_core, prev_rdma.timeline_id, retired),
        retired != Some(pull.handle.frame),
        next_core.timelines_spec().count_spec() == prev_core.timelines_spec().count_spec(),
        next_core.timelines_spec().events_spec() == prev_core.timelines_spec().events_spec(),
        next_core.timelines_spec().next_seq_spec() == prev_core.timelines_spec().next_seq_spec(),
        next_rdma.timeline_id == prev_rdma.timeline_id,
        next_rdma.host_mr == prev_rdma.host_mr,
        next_rdma.gpu_mr == prev_rdma.gpu_mr,
        next_rdma.shared_host_mr == prev_rdma.shared_host_mr,
        next_rdma.shared_gpu_mr == prev_rdma.shared_gpu_mr,
        pull.valid(prev_rdma, prev_core),
    ensures
        pull.valid(next_rdma, next_core),
        prev_core.citations().subset_of(next_core.citations()),
{
    hide(CacheCore::well_formed);
    reveal(peer_core_step_except);
    reveal(core_transition);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    lemma_pull_retained_after_peer_step(
        prev_core,
        next_core,
        prev_rdma.timeline_id,
        &pull.copy@,
        retired,
    );
    assert(next_core.timelines_spec().issued(next_rdma.timeline_id, pull.wr_id));
    assert(next_core.frames_spec().meta(pull.handle.frame) == prev_core.frames_spec().meta(
        pull.handle.frame,
    ));
    assert(next_core.frames_spec().frame(pull.handle.frame) == prev_core.frames_spec().frame(
        pull.handle.frame,
    ));
    assert(next_rdma.effective_gpu_mr_spec() == prev_rdma.effective_gpu_mr_spec());
}

proof fn lemma_pump_cq_preserves_pull_tracked<'a>(
    before: &RdmaBackend<'a>,
    after: &RdmaBackend<'a>,
    seq: u64,
    writer: TransferEvent,
    wr_id: u64,
)
    requires
        pull_tracked_or_done(before, seq, writer),
        after.cq.in_flight().dom().subset_of(before.cq.in_flight().dom()),
        forall|k: u64| #[trigger]
            after.pending().contains_key(k) ==> (before.pending().contains_key(k)
                && after.pending()[k] == before.pending()[k]),
        forall|k: u64|
            k != wr_id && #[trigger] before.pending().contains_key(k) ==> (
            after.pending().contains_key(k) && after.pending()[k] == before.pending()[k]),
        before.pending().contains_key(wr_id) ==> (after.completed_pull_spec(wr_id) is Some
            && forall|q: u64|
            (q % (BITMAP_CAPACITY as u64)) != (wr_id % (BITMAP_CAPACITY as u64))
                ==> after.completed_pull_spec(q) == before.completed_pull_spec(q)),
        !before.pending().contains_key(wr_id) ==> after.completed_pulls == before.completed_pulls,
    ensures
        pull_tracked_or_done(after, seq, writer),
{
    reveal(pull_tracked_or_done_state);
    if before.pending().contains_key(seq) {
        if seq == wr_id {
            assert(after.completed_pull_spec(seq) is Some);
        } else {
            assert(after.pending().contains_key(seq));
            assert(after.pending()[seq] == writer);
        }
    } else {
        assert(before.completed_pull_spec(seq) is Some);
        if before.pending().contains_key(wr_id) {
            assert(wr_id != seq);
            assert((wr_id % (BITMAP_CAPACITY as u64)) != (seq % (BITMAP_CAPACITY as u64)));
            assert(after.completed_pull_spec(seq) == before.completed_pull_spec(seq));
        }
    }
}

proof fn lemma_rdma_wf_implies_pull_tracked<'a>(
    rdma: &RdmaBackend<'a>,
    core: &CacheCore,
    seq: u64,
    writer: TransferEvent,
)
    requires
        rdma.well_formed(core),
        rdma.pending().contains_key(seq),
        rdma.pending()[seq] == writer,
    ensures
        pull_tracked_or_done(rdma, seq, writer),
{
    reveal(pull_tracked_or_done_state);
    reveal(rdma_window_wf);
    let cq_snap = rdma::CqSnapshot::of(rdma.cq);
    assert(cq_snap.pending.contains_key(seq));
    assert(cq_snap.pending[seq].kind is DataTransfer);
    assert(rdma.in_flight_ops[(seq % (BITMAP_CAPACITY as u64)) as int] matches Some(ops)
        && ops.sequence == seq);
    assert forall|w: u64| #[trigger] rdma.pending().contains_key(w) && w != seq implies (w % (
    BITMAP_CAPACITY as u64)) != (seq % (BITMAP_CAPACITY as u64)) by {
        assert(cq_snap.pending.contains_key(w));
        assert(cq_snap.pending[w].kind is DataTransfer);
        assert(rdma.in_flight_ops[(w % (BITMAP_CAPACITY as u64)) as int] matches Some(opw)
            && opw.sequence == w);
        if (w % (BITMAP_CAPACITY as u64)) == (seq % (BITMAP_CAPACITY as u64)) {
            assert(w == seq);
        }
    }
}

enum BindAwaitStatus {
    Ready,
    BindFailed,
    PendingOrFailed,
}

impl<'a, T: ByteTransport> PeerSession<'a, T> {
    pub open(crate) spec fn session_queues_valid(&self, core: &CacheCore) -> bool {
        &&& forall|i: int, j: int|
            0 <= i < j < self.pending_outbound_grants@.len() ==> (
            #[trigger] self.pending_outbound_grants@[i]).handle != (
            #[trigger] self.pending_outbound_grants@[j]).handle
        &&& forall|i: int|
            0 <= i < self.pending_outbound_grants@.len() ==> (
            #[trigger] self.pending_outbound_grants@[i]).valid(&self.rdma, core)
        &&& forall|i: int|
            0 <= i < self.pending_inbound_grants@.len() ==> (
            #[trigger] self.pending_inbound_grants@[i]).valid(
                core,
                &self.channel,
                self.rdma.peer_node,
            )
    }

    pub open(crate) spec fn well_formed(&self, core: &CacheCore) -> bool {
        &&& self.rdma.well_formed(core)
        &&& self.channel.local() == core.node()
        &&& self.channel.peer() == self.rdma.peer_node
        &&& self.channel.id() == core.instance_id()
        &&& forall|t: GrantTag| #[trigger]
            self.channel.sent().contains_key(t) ==> t < self.next_grant_tag
        &&& forall|i: int, j: int|
            0 <= i < j < self.active_grants@.len() ==> (#[trigger] self.active_grants@[i]).grant_tag
                != (#[trigger] self.active_grants@[j]).grant_tag && self.active_grants@[i].handle
                != self.active_grants@[j].handle
        &&& forall|i: int|
            0 <= i < self.active_grants@.len() ==> {
                let g = #[trigger] self.active_grants@[i];
                &&& g.grant_tag < self.next_grant_tag
                &&& g.frame.0 < core.frames_spec().metas_spec().len()
                &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                    let lease = self.rdma.active_leases@[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& self.channel.sent().contains_key(g.grant_tag)
                    &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                        slot: core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            }
        &&& self.session_queues_valid(core)
    }

    /// Take ownership of the RDMA backend after detaching this session.
    pub fn into_rdma(self) -> (r: RdmaBackend<'a>)
        ensures
            r == self.rdma_spec(),
    {
        self.rdma
    }

    pub fn rdma(&self) -> (backend: &RdmaBackend<'a>)
        ensures
            *backend == self.rdma_spec(),
    {
        &self.rdma
    }

    pub fn new(
        rdma: RdmaBackend<'a>,
        channel: Channel<T, VeriCacheAttachment>,
        core: &CacheCore,
    ) -> (s: Self)
        requires
            core.well_formed(),
            rdma.well_formed(core),
            channel.local() == core.node(),
            channel.peer() == rdma.peer_node_spec(),
            channel.id() == core.instance_id(),
            channel.sent() == Map::<GrantTag, GrantKind>::empty(),
        ensures
            s.well_formed(core),
            s.rdma_spec() == rdma,
            s.channel_spec() == channel,
            s.next_grant_tag_spec() == 1,
            s.next_request_id_spec() == 1,
            s.active_grants_spec()@.len() == 0,
            s.pending_outbound_grants_spec()@.len() == 0,
            s.pending_inbound_grants_spec()@.len() == 0,
            s.reconnect_epoch_spec() == 0,
    {
        let _ = core;
        Self {
            rdma,
            channel,
            next_grant_tag: 1,
            next_request_id: 1,
            active_grants: Vec::new(),
            pending_outbound_grants: Vec::new(),
            pending_inbound_fetches: Vec::new(),
            pending_inbound_grants: Vec::new(),
            pending_inbound_misses: Vec::new(),
            pending_revoke_notices: Vec::new(),
            pending_revoke_acks: Vec::new(),
            pending_reconnect_barriers: Vec::new(),
            quarantined_grants: Vec::new(),
            reconnect_epoch: 0,
            peer_reconnect_epoch: 0,
        }
    }

    /// Receive and decode the next peer message using `Channel::recv_any` exclusively.
    pub(crate) fn recv_message(
        &mut self,
        core: &CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerMessage, ChannelError<T::Error>>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
        ensures
            final(self).rdma == old(self).rdma,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            final(self).channel.same_session(old(self).channel),
            r.is_err() ==> final(self).well_formed(core) && final(self).channel.closed(),
            r matches Ok(PeerMessage::Control { .. }) ==> final(self).well_formed(core),
            r matches Ok(PeerMessage::Unexpected) ==> final(self).well_formed(core),
            r matches Ok(
                PeerMessage::ReadGrant {
                    tag,
                    meta,
                    cap,
                    share,
                    statement,
                    cert,
                    revoked,
                    version_binding,
                },
            ) ==> {
                &&& final(self).well_formed(core)
                &&& final(self).channel.received().contains_key(tag)
                &&& final(self).channel.received()[tag] == (GrantKind::ReadShare {
                    slot: cap.slot_ref(),
                    share_id: share@.share(),
                })
                &&& statement@.valid(core.instance_id(), core.node())
                &&& statement@.knows.set() == knowledge_of(core.node(), statement@.closure())
                &&& cap.well_formed()
                &&& cap.slot_ref().loc.node == old(self).rdma.peer_node
                &&& cap.access() == rdma::RemoteAccess::ReadOnly
                &&& cap.share() == Some(share@.share())
                &&& cap.slot_ref() == cert@.slot
                &&& cert@.slot.is_full()
                &&& cert@.bytes.len() == meta.frame_size
                &&& share@.is_fenced()
                &&& share@.instance_id() == core.instance_id()
                &&& share@.loc() == cap.slot_ref().loc
                &&& share@.frontier().last_writer == Some(cert@.writer)
                &&& transfer::transport::covers(share@.frontier(), statement@.closure())
                &&& revoked@.instance_id() == core.instance_id()
                &&& crate::provenance::complete_receipts_map(statement@.submitted)
                &&& super::protocol::statement_replica_valid(
                    statement@.submitted,
                    statement@.succeeded,
                    revoked@,
                    statement@.facts,
                    cert@,
                )
                &&& (version_binding@ matches Some(b) && b.instance_id()
                    == cluster_version_instance_id() && b.key() == (
                    meta.version.origin_node as NodeId,
                    meta.version.seq,
                ) && b.value() == cert@.bytes)
            },
            r matches Ok(PeerMessage::ReadRelease { tag, share, statement, .. }) ==> {
                &&& old(self).session_queues_valid(core)
                &&& final(self).session_queues_valid(core)
                &&& old(self).channel.sent().contains_key(tag)
                &&& final(self).channel.sent() == old(self).channel.sent().remove(tag)
                &&& final(self).channel.received() == old(self).channel.received()
                &&& old(self).channel.sent()[tag] matches GrantKind::ReadShare { slot, share_id }
                    && share@.is_fenced() && share@.instance_id() == core.instance_id()
                    && share@.loc() == slot.loc && share@.share() == share_id
                    && transfer::transport::covers(share@.frontier(), statement@.closure())
                &&& crate::provenance::complete_receipts_map(statement@.submitted)
                &&& statement@.valid(core.instance_id(), core.node())
                &&& statement@.knows.set() == knowledge_of(core.node(), statement@.closure())
            },
    {
        let _ = core;
        let frame = self.channel.recv_any(Tracked(instance))?;
        match frame {
            ReceivedFrame::Notice(rec) => {
                let Received { user_bytes, statement, .. } = rec;
                match decode_control_payload(&user_bytes) {
                    Some(payload) => Ok(PeerMessage::Control { payload, statement }),
                    None => Ok(PeerMessage::Unexpected),
                }
            },
            ReceivedFrame::ReadGrant(rec) => {
                let Received { tag, user_bytes, capability, share, statement, attachment, .. } =
                    rec;
                let Some(meta) = decode_read_grant_meta(&user_bytes) else {
                    return Err(ChannelError::InvalidFrame);
                };
                let cap = capability.unwrap();
                let tracked sh = share.get().tracked_unwrap();
                let tracked att = attachment.get();
                let ghost cert;
                let tracked revoked;
                let tracked version_binding;
                proof {
                    match att {
                        VeriCacheAttachment::ReadGrant {
                            meta: att_meta,
                            cert: c,
                            revoked: r,
                            version_binding: vb,
                        } => {
                            super::protocol::lemma_encode_read_grant_meta_injective(meta, att_meta);
                            cert = c;
                            revoked = r;
                            version_binding = vb;
                        },
                        _ => {
                            assert(false);
                            cert = arbitrary();
                            revoked = proof_from_false();
                            version_binding = None;
                        },
                    }
                }
                Ok(
                    PeerMessage::ReadGrant {
                        tag,
                        meta,
                        cap,
                        share: Tracked(sh),
                        statement,
                        cert: Ghost(cert),
                        revoked: Tracked(revoked),
                        version_binding: Tracked(version_binding),
                    },
                )
            },
            ReceivedFrame::ReadRelease(rec) => {
                let Received { tag, user_bytes, share, statement, .. } = rec;
                let Some(meta) = decode_read_release_meta(&user_bytes) else {
                    return Err(ChannelError::InvalidFrame);
                };
                let tracked sh = share.get().tracked_unwrap();
                Ok(PeerMessage::ReadRelease { tag, meta, share: Tracked(sh), statement })
            },
            ReceivedFrame::WriteGrant(_) | ReceivedFrame::WriteRelease(_) => {
                Ok(PeerMessage::Unexpected)
            },
        }
    }

    /// Send a `ControlPayload` notice on `self.channel`.
    pub(crate) fn send_control(
        &mut self,
        core: &CacheCore,
        payload: ControlPayload,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), ChannelError<T::Error>>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
        ensures
            final(self).well_formed(core),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma == old(self).rdma,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            final(self).channel.same_session(old(self).channel),
            final(self).channel.sent() == old(self).channel.sent(),
            final(self).channel.received() == old(self).channel.received(),
    {
        if let ControlPayload::FetchBatch { ref entries } = payload {
            if entries.is_empty() || entries.len() > super::protocol::MAX_FETCH_BATCH {
                return Err(ChannelError::InvalidFrame);
            }
        }
        let bytes = encode_control_payload(payload);
        let ghost cited = core.citations();
        let tracked att = VeriCacheAttachment::Control { payload };
        proof {
            lemma_core_wf_facts(core);
        }
        match self.channel.send_notice_with(
            &bytes,
            Tracked(instance),
            Tracked(&core.borrow_ledger().submitted),
            Tracked(&core.borrow_ledger().knows),
            Tracked(&core.borrow_ledger().succeeded),
            Ghost(cited),
            Tracked(att),
        ) {
            Ok(()) => Ok(()),
            Err((e, _)) => Err(e),
        }
    }

    /// Initiator step 1: send a `ControlPayload::FetchRequest` for `key` and
    /// return the allocated `request_id`.
    pub fn send_fetch_request(
        &mut self,
        core: &CacheCore,
        key: ObjectKey,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, ChannelError<T::Error>>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
            old(self).next_request_id_spec() < u64::MAX,
        ensures
            final(self).well_formed(core),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().qp_spec() == old(self).rdma_spec().qp_spec(),
            final(self).rdma_spec() == old(self).rdma_spec(),
            final(self).next_grant_tag_spec() == old(self).next_grant_tag_spec(),
            final(self).active_grants_spec() == old(self).active_grants_spec(),
            final(self).pending_outbound_grants_spec() == old(self).pending_outbound_grants_spec(),
            final(self).pending_inbound_fetches_spec() == old(self).pending_inbound_fetches_spec(),
            final(self).pending_inbound_grants_spec() == old(self).pending_inbound_grants_spec(),
            final(self).pending_inbound_misses_spec() == old(self).pending_inbound_misses_spec(),
            final(self).pending_revoke_notices_spec() == old(self).pending_revoke_notices_spec(),
            final(self).pending_revoke_acks_spec() == old(self).pending_revoke_acks_spec(),
            final(self).pending_reconnect_barriers_spec() == old(
                self,
            ).pending_reconnect_barriers_spec(),
            final(self).quarantined_grants_spec() == old(self).quarantined_grants_spec(),
            final(self).reconnect_epoch_spec() == old(self).reconnect_epoch_spec(),
            final(self).peer_reconnect_epoch_spec() == old(self).peer_reconnect_epoch_spec(),
            final(self).channel_spec().same_session(old(self).channel_spec()),
            final(self).channel_spec().sent() == old(self).channel_spec().sent(),
            final(self).channel_spec().received() == old(self).channel_spec().received(),
    {
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        self.send_control(
            core,
            ControlPayload::FetchRequest { request_id, key },
            Tracked(instance),
        )?;
        Ok(request_id)
    }

    /// Allocate consecutive request IDs and emit a single fetch notice.
    pub(crate) fn send_fetch_batch(
        &mut self,
        core: &CacheCore,
        keys: &[ObjectKey],
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, ChannelError<T::Error>>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
        ensures
            final(self).well_formed(core),
            final(self).rdma == old(self).rdma,
            final(self).channel.same_session(old(self).channel),
            final(self).channel.sent() == old(self).channel.sent(),
            final(self).channel.received() == old(self).channel.received(),
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            r matches Ok(base) ==> base == old(self).next_request_id && final(self).next_request_id
                == base + keys.len(),
    {
        if keys.is_empty() || keys.len() > super::protocol::MAX_FETCH_BATCH {
            return Err(ChannelError::InvalidFrame);
        }
        let base = self.next_request_id;
        if keys.len() as u64 > u64::MAX - base {
            return Err(ChannelError::InvalidFrame);
        }
        let mut entries = Vec::with_capacity(keys.len());
        let mut i = 0;
        while i < keys.len()
            invariant
                i == entries.len() <= keys.len(),
                base + keys.len() <= u64::MAX,
                forall|j: int|
                    0 <= j < i ==> #[trigger] entries@[j] == ((base + j) as u64, keys@[j]),
            decreases keys.len() - i,
        {
            entries.push((base + i as u64, keys[i]));
            i += 1;
        }
        self.next_request_id = base + keys.len() as u64;
        self.send_control(core, ControlPayload::FetchBatch { entries }, Tracked(instance))?;
        Ok(base)
    }

    fn find_grant_index(&self, grant_tag: GrantTag) -> (r: Option<usize>)
        ensures
            r matches Some(idx) ==> (idx as int) < self.active_grants@.len()
                && self.active_grants@[idx as int].grant_tag == grant_tag,
            r is None ==> forall|i: int|
                0 <= i < self.active_grants@.len() ==> (#[trigger] self.active_grants@[i]).grant_tag
                    != grant_tag,
    {
        let mut i: usize = 0;
        while i < self.active_grants.len()
            invariant
                i <= self.active_grants.len(),
                forall|j: int|
                    0 <= j < i ==> (#[trigger] self.active_grants@[j]).grant_tag != grant_tag,
            decreases self.active_grants.len() - i,
        {
            if self.active_grants[i].grant_tag == grant_tag {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Lease `handle` is in `RemoteLeaseState::Binding` for `frame` with retained
    /// `ShareCap` last writer `writer` (whether or not the bind CQE has been reaped).
    pub open(crate) spec fn lease_is_binding(
        &self,
        handle: RemoteLeaseHandle,
        frame: FrameId,
        wr_id: u64,
        writer: TransferEvent,
    ) -> bool {
        self.rdma.lease_index_spec(handle) matches Some(idx) && {
            let lease = self.rdma.active_leases@[idx];
            &&& lease.frame == frame
            &&& lease.wr_id == wr_id
            &&& lease.state == RemoteLeaseState::Binding
            &&& lease.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(writer)
        }
    }

    /// Lease `handle` is a pending MW bind of `frame` whose bind WR is still in flight.
    pub open(crate) spec fn binding_lease(
        &self,
        handle: RemoteLeaseHandle,
        frame: FrameId,
        wr_id: u64,
        writer: TransferEvent,
    ) -> bool {
        &&& self.lease_is_binding(handle, frame, wr_id, writer)
        &&& self.rdma.cq.in_flight().contains_key(wr_id)
        &&& self.rdma.cq.in_flight()[wr_id].kind is BindWindow
        &&& (self.rdma.lease_matches_window_state(handle, RemoteLeaseState::Binding) matches Some(w)
            ==> self.rdma.cq.in_flight()[wr_id].kind->BindWindow_window_id == w.id())
    }

    /// Donor helper: pin published `frame` and post a Type-2 MW bind for it.
    #[verifier::rlimit(250)]
    fn serve_bind(
        &mut self,
        core: &mut CacheCore,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(RemoteLeaseHandle, u64), CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec() == old(core).timelines_spec(),
            final(core).frames_spec().same_directory(&old(core).frames_spec()),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.pending() == old(self).rdma.pending(),
            final(self).rdma.completed_pulls == old(self).rdma.completed_pulls,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            r matches Ok((handle, wr_id)) ==> {
                &&& old(self).rdma.lease_index_spec(handle) is None
                &&& final(core).frames_spec().published(frame) matches Some((_, _, c, _))
                    && final(self).binding_lease(handle, frame, wr_id, c.writer)
            },
    {
        let mut reserved = None;
        self.serve_bind_with(core, frame, Tracked(shares), Tracked(instance), false, &mut reserved)
    }

    #[verifier::rlimit(250)]
    fn serve_bind_with(
        &mut self,
        core: &mut CacheCore,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
        prepare: bool,
        reserved: &mut Option<PreparedLeaseBind>,
    ) -> (r: Result<(RemoteLeaseHandle, u64), CoreError>)
        requires
            old(core).well_formed(),
            old(reserved).is_none(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec() == old(core).timelines_spec(),
            final(core).frames_spec().same_directory(&old(core).frames_spec()),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(core).frames_spec().same_directory(&old(core).frames_spec()),
            bind_reservations_preserved(&old(self).rdma, &final(self).rdma),
            r.is_err() ==> final(reserved).is_none(),
            r.is_ok() ==> {
                &&& final(reserved).is_some() == prepare
                &&& (*final(reserved) matches Some(p) ==> p.frame == frame && p.valid(
                    &final(self).rdma,
                    final(core),
                ) && old(self).rdma.next_control_wr_id <= p.bind.id() && p.bind.id()
                    < final(self).rdma.next_control_wr_id)
            },
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.pending() == old(self).rdma.pending(),
            final(self).rdma.completed_pulls == old(self).rdma.completed_pulls,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            r matches Ok((handle, wr_id)) ==> {
                &&& old(self).rdma.lease_index_spec(handle) is None
                &&& final(core).frames_spec().published(frame) matches Some((_, _, c, _))
                    && final(self).binding_lease(handle, frame, wr_id, c.writer)
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_leases = self.rdma.active_leases@;
        let ghost before_core = *core;
        match self.rdma.bind_remote_read_lease_with(
            core,
            frame,
            Tracked(shares),
            Tracked(instance),
            prepare,
            reserved,
        ) {
            Ok((handle, _mw_slot, _gen, bind_wr_id)) => {
                proof {
                    assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                        let g = #[trigger] self.active_grants@[i];
                        &&& g.grant_tag < self.next_grant_tag
                        &&& g.frame.0 < core.frames_spec().metas_spec().len()
                        &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                            let lease = self.rdma.active_leases@[idx];
                            &&& lease.frame == g.frame
                            &&& lease.state == RemoteLeaseState::Bound
                            &&& self.channel.sent().contains_key(g.grant_tag)
                            &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                                slot: core.frames_spec().frame(g.frame).slot,
                                share_id: lease.share@,
                            })
                        }
                    } by {
                        reveal(FrameTable::same_directory);
                        let g = self.active_grants@[i];
                        let old_idx = rdma_lease_index_spec(before_leases, g.handle).unwrap();
                        lemma_lease_index_preserved_prefix(
                            before_leases,
                            self.rdma.active_leases@,
                            g.handle,
                            old_idx,
                        );
                    }
                    if let Some(old_idx) = rdma_lease_index_spec(before_leases, handle) {
                        lemma_lease_index_preserved_prefix(
                            before_leases,
                            self.rdma.active_leases@,
                            handle,
                            old_idx,
                        );
                        assert(self.rdma.lease_index_spec(handle) == Some(old_idx));
                    }
                    assert forall|i: int| 0 <= i < self.pending_outbound_grants@.len() implies {
                        let p = #[trigger] self.pending_outbound_grants@[i];
                        &&& rdma_lease_index_spec(before_leases, p.handle) matches Some(bidx) && {
                            let bl = before_leases[bidx];
                            &&& rdma_lease_index_spec(
                                self.rdma.active_leases@,
                                p.handle,
                            ) matches Some(aidx)
                            &&& self.rdma.active_leases@[aidx].frame == bl.frame
                            &&& self.rdma.active_leases@[aidx].wr_id == bl.wr_id
                            &&& super::poll::rdma_lease_is_binding_like(
                                self.rdma.active_leases@[aidx],
                            )
                            &&& self.rdma.active_leases@[aidx].share_cap == bl.share_cap
                        }
                    } by {
                        let p = self.pending_outbound_grants@[i];
                        let bidx = rdma_lease_index_spec(before_leases, p.handle).unwrap();
                        lemma_lease_index_preserved_prefix(
                            before_leases,
                            self.rdma.active_leases@,
                            p.handle,
                            bidx,
                        );
                    }
                    assert(core.frames_spec().same_layout(&before_core.frames_spec())) by {
                        reveal(FrameTable::same_directory);
                        reveal(FrameTable::same_layout);
                    }
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        before_leases,
                        self.rdma.active_leases@,
                        &before_core,
                        core,
                    );
                    reveal(rdma_leases_valid);
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                Ok((handle, bind_wr_id))
            },
            Err(e) => {
                proof {
                    reveal(rdma_leases_valid);
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                Err(e)
            },
        }
    }

    /// Donor helper: drive `pump_cq` (without discarding unrelated completions)
    /// until `handle` transitions to `BindReady`, `BindFailedReady`, or fails/times out.
    fn serve_await_bind(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        frame: FrameId,
        bind_wr_id: u64,
        Ghost(writer): Ghost<TransferEvent>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<BindAwaitStatus, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).binding_lease(handle, frame, bind_wr_id, writer),
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
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            final(core).ledger_spec()@.submitted == old(core).ledger_spec()@.submitted,
            final(core).ledger_spec()@.revoked == old(core).ledger_spec()@.revoked,
            old(core).ledger_spec()@.knows.set().subset_of(final(core).ledger_spec()@.knows.set()),
            old(core).ledger_spec()@.succeeded.dom().subset_of(
                final(core).ledger_spec()@.succeeded.dom(),
            ),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            r matches Ok(BindAwaitStatus::Ready) ==> (final(self).rdma.lease_index_spec(
                handle,
            ) matches Some(idx) && final(self).rdma.active_leases@[idx].state
                == RemoteLeaseState::BindReady && final(self).rdma.active_leases@[idx].frame
                == frame && (final(self).rdma.active_leases@[idx].share_cap matches Some(sc)
                && sc@.frontier().last_writer == Some(writer))),
            r matches Ok(BindAwaitStatus::BindFailed) ==> (final(self).rdma.lease_index_spec(
                handle,
            ) matches Some(idx) && final(self).rdma.active_leases@[idx].state
                == RemoteLeaseState::BindFailedReady),
            ((r matches Ok(BindAwaitStatus::PendingOrFailed)) || r.is_err()) ==> (
            final(self).rdma.lease_index_spec(handle) matches Some(idx) && {
                let l = final(self).rdma.active_leases@[idx];
                &&& l.frame == frame
                &&& l.wr_id == bind_wr_id
                &&& super::poll::rdma_lease_is_binding_like(l)
                &&& l.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(writer)
            }),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut spins: u32 = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec() == old(core).frames_spec(),
                core.catalog_spec() == old(core).catalog_spec(),
                core.spec_spec() == old(core).spec_spec(),
                core.next_version_seq_spec() == old(core).next_version_seq_spec(),
                core.version_seq_limit_spec() == old(core).version_seq_limit_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                core.ledger_spec()@.submitted == old(core).ledger_spec()@.submitted,
                core.ledger_spec()@.revoked == old(core).ledger_spec()@.revoked,
                old(core).ledger_spec()@.knows.set().subset_of(core.ledger_spec()@.knows.set()),
                old(core).ledger_spec()@.succeeded.dom().subset_of(
                    core.ledger_spec()@.succeeded.dom(),
                ),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
                self.rdma.lease_index_spec(handle) matches Some(lidx) && {
                    let l = self.rdma.active_leases@[lidx];
                    &&& l.frame == frame
                    &&& l.wr_id == bind_wr_id
                    &&& l.state == RemoteLeaseState::Binding
                    &&& l.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(writer)
                },
            decreases 1_000_000 - spins,
        {
            let ghost step_core = *core;
            let ghost step_leases = self.rdma.active_leases@;
            let ghost step_self = *self;
            let ghost step_rdma = self.rdma;
            let pump_res = self.rdma.pump_cq(core, Tracked(instance));
            proof {
                lemma_foreign_shares_trans(
                    old(core),
                    &step_core,
                    core,
                    old(self).rdma.active_leases@,
                    step_leases,
                    self.rdma.active_leases@,
                );
            }
            match pump_res {
                Err(e) => return Err(e),
                Ok(None) => {
                    spins += 1;
                },
                Ok(Some(_)) => {
                    proof {
                        reveal(rdma_leases_valid);
                        reveal(FrameTable::same_directory);
                        reveal(FrameTable::same_layout);
                        lemma_peer_core_step_trans(
                            old(core),
                            &step_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        broadcast use {vstd::iset::group_iset_lemmas, vstd::map::group_map_lemmas};

                        assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                            let g = #[trigger] self.active_grants@[i];
                            &&& g.grant_tag < self.next_grant_tag
                            &&& g.frame.0 < core.frames_spec().metas_spec().len()
                            &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                                let lease = self.rdma.active_leases@[idx];
                                &&& lease.frame == g.frame
                                &&& lease.state == RemoteLeaseState::Bound
                                &&& self.channel.sent().contains_key(g.grant_tag)
                                &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                                    slot: core.frames_spec().frame(g.frame).slot,
                                    share_id: lease.share@,
                                })
                            }
                        } by {
                            let g = self.active_grants@[i];
                            assert(step_self.active_grants@[i] == g);
                            assert(step_rdma.lease_index_spec(g.handle) is Some);
                            let oidx = step_rdma.lease_index_spec(g.handle)->Some_0;
                            assert(step_leases[oidx].state == RemoteLeaseState::Bound);
                            assert(!step_leases[oidx].has_active_wr());
                        }
                        assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                            reveal(FrameTable::same_layout);
                        }
                        assert forall|i: int| 0 <= i < self.pending_outbound_grants@.len() implies {
                            let p = #[trigger] self.pending_outbound_grants@[i];
                            &&& rdma_lease_index_spec(step_leases, p.handle) matches Some(bidx) && {
                                let bl = step_leases[bidx];
                                &&& rdma_lease_index_spec(
                                    self.rdma.active_leases@,
                                    p.handle,
                                ) matches Some(aidx)
                                &&& self.rdma.active_leases@[aidx].frame == bl.frame
                                &&& self.rdma.active_leases@[aidx].wr_id == bl.wr_id
                                &&& super::poll::rdma_lease_is_binding_like(
                                    self.rdma.active_leases@[aidx],
                                )
                                &&& self.rdma.active_leases@[aidx].share_cap == bl.share_cap
                            }
                        } by {
                            let p = self.pending_outbound_grants@[i];
                            assert(step_self.pending_outbound_grants@[i] == p);
                            assert(step_rdma.lease_index_spec(p.handle) is Some);
                            let bidx = step_rdma.lease_index_spec(p.handle)->Some_0;
                            assert(super::poll::rdma_lease_is_binding_like(step_leases[bidx]));
                        }
                        lemma_outbound_grants_preserved(
                            self.pending_outbound_grants@,
                            step_leases,
                            self.rdma.active_leases@,
                            &step_core,
                            core,
                        );
                    }
                    match self.rdma.lease_state(handle) {
                        Some(RemoteLeaseState::BindReady) => return Ok(BindAwaitStatus::Ready),
                        Some(RemoteLeaseState::BindFailedReady) => {
                            return Ok(BindAwaitStatus::BindFailed);
                        },
                        Some(RemoteLeaseState::Binding) => {
                            spins += 1;
                        },
                        _ => return Ok(BindAwaitStatus::PendingOrFailed),
                    }
                },
            }
        }
        Ok(BindAwaitStatus::PendingOrFailed)
    }

    /// Donor helper: complete a `BindReady` MW lease, construct `VeriCacheAttachment::ReadGrant`,
    /// and transmit `READ_GRANT` (cleaning up cleanly via unpin/unbind if send returns `Closed`).
    #[verifier::rlimit(250)]
    fn finish_send_bound_grant(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        frame: FrameId,
        request_id: u64,
        key: ObjectKey,
        version: VersionId,
        Ghost(cert): Ghost<ValueCertificate>,
        Ghost(bind_sub): Ghost<SubmittedMap>,
        Ghost(bind_succ): Ghost<SucceededMap>,
        Tracked(revoked): Tracked<RevokedMap>,
        Ghost(bind_facts): Ghost<ISet<TransferEvent>>,
        Tracked(version_binding): Tracked<Option<VersionBinding>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == core.instance_id(),
            old(self).next_grant_tag < u64::MAX,
            frame.0 < core.frames_spec().metas_spec().len(),
            old(self).rdma.lease_index_spec(handle) matches Some(idx) && {
                let lease = old(self).rdma.active_leases@[idx];
                &&& lease.frame == frame
                &&& lease.state == RemoteLeaseState::BindReady
                &&& lease.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(
                    cert.writer,
                )
            },
            forall|i: int|
                0 <= i < old(self).pending_outbound_grants@.len() ==> (#[trigger] old(
                    self,
                ).pending_outbound_grants@[i]).handle != handle,
            revoked.instance_id() == core.instance_id(),
            bind_sub.map().submap_of(core.ledger_spec()@.submitted.map()),
            bind_succ.dom().subset_of(core.ledger_spec()@.succeeded.dom()),
            forall|e: TransferEvent| #[trigger]
                bind_facts.contains(e) ==> core.ledger_spec()@.submitted.map().contains_key(e)
                    && core.ledger_spec()@.knows.set().contains((core.node(), e)),
            bind_facts.contains(cert.writer),
            cert.slot == core.frames_spec().frame(frame).slot,
            cert.slot.is_full(),
            cert.slot.well_formed(),
            cert.bytes.len() == core.frames_spec().tier_frame_size(
                core.frames_spec().frame(frame).tier,
            ) as u64,
            super::protocol::statement_replica_valid(
                bind_sub,
                bind_succ,
                revoked,
                bind_facts,
                cert,
            ),
            version_binding matches Some(vb) && vb.instance_id() == cluster_version_instance_id()
                && vb.key() == (version.origin_node as NodeId, version.seq) && vb.value()
                == cert.bytes,
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            final(self).channel.received() == old(self).channel.received(),
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        hide(ActiveRemoteLease::valid);
        hide(super::protocol::statement_replica_valid);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost pre_bind_self = *self;
        let ghost hidx = self.rdma.lease_index_spec(handle).unwrap();
        proof {
            self.lemma_peer_wf_lease_facts(core);
            self.lemma_active_grants_exclude_nonbound_handle(core, handle, hidx);
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        let (cap, Tracked(share_cap)) = self.rdma.complete_ready_bind(core, handle)?;
        proof {
            assert(core.frames_spec().same_directory(&core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            };
            Self::lemma_session_wf_after_outbound_lease_update(
                &pre_bind_self,
                &*self,
                core,
                core,
                handle,
                hidx,
            );
            lemma_same_pulls_preserves_pull_tracked(&old(self).rdma, &self.rdma);
        }
        let grant_tag = self.next_grant_tag;
        proof {
            lemma_core_wf_facts(core);
            lemma_initializing_frame_slot_valid(core, frame);
        }
        let (_, frame_tier) = core.frame_binding(frame).unwrap();
        let meta = ReadGrantMeta {
            request_id,
            key,
            version,
            frame_size: core.registered_tier_frame_size(frame_tier) as u64,
        };
        let meta_bytes = encode_read_grant_meta(meta);
        let ghost cited = core.citations();
        let tracked att = VeriCacheAttachment::ReadGrant { meta, cert, revoked, version_binding };
        proof {
            self.lemma_read_grant_attachment_valid(
                core,
                grant_tag,
                frame,
                frame_tier,
                meta,
                meta_bytes@,
                cap,
                share_cap,
                cert,
                bind_sub,
                bind_succ,
                revoked,
                bind_facts,
                att,
                cited,
            );
        }
        let ghost pre_send_self = *self;
        match self.channel.send_remote_read_grant_with(
            grant_tag,
            &cap,
            &meta_bytes,
            Tracked(instance),
            Tracked(&core.borrow_ledger().submitted),
            Tracked(&core.borrow_ledger().knows),
            Tracked(&core.borrow_ledger().succeeded),
            Ghost(cited),
            Tracked(share_cap),
            Tracked(att),
        ) {
            Ok(()) => {
                self.next_grant_tag += 1;
                self.active_grants.push(
                    ActiveGrant { grant_tag, handle, frame, soft_revoking: false },
                );
                proof {
                    Self::lemma_session_wf_after_grant_sent(
                        &pre_send_self,
                        &*self,
                        core,
                        grant_tag,
                        handle,
                        hidx,
                        frame,
                        cap.slot_ref(),
                        share_cap.share(),
                    );
                }
                Ok(ServeOutcome::Granted { grant_tag, handle })
            },
            Err((send_err, ret_trk)) => {
                proof {
                    Self::lemma_session_wf_after_send_transport_err(&pre_send_self, &*self, core);
                }
                if matches!(send_err, ChannelError::Closed) {
                    let tracked (sh, _) = ret_trk.get().tracked_unwrap();
                    let ghost before_u_self = *self;
                    let ghost before_u_core = *core;
                    let ghost before_u_leases = self.rdma.active_leases@;
                    proof {
                        self.lemma_peer_wf_lease_facts(core);
                    }
                    if let Ok(inv_wr_id) = self.rdma.unpin_and_unbind_lease(
                        core,
                        handle,
                        Tracked(sh),
                        Ticket::NONE,
                        Tracked(instance),
                    ) {
                        proof {
                            lemma_foreign_shares_trans(
                                old(core),
                                &before_u_core,
                                core,
                                old(self).rdma.active_leases@,
                                before_u_leases,
                                self.rdma.active_leases@,
                            );
                            Self::lemma_session_wf_after_outbound_lease_update(
                                &before_u_self,
                                &*self,
                                &before_u_core,
                                core,
                                handle,
                                hidx,
                            );
                            lemma_same_pulls_preserves_pull_tracked(&old(self).rdma, &self.rdma);
                        }
                        let ghost before_await_core = *core;
                        let ghost before_await_leases = self.rdma.active_leases@;
                        let _ = self.await_unbind_invalidate(
                            core,
                            handle,
                            inv_wr_id,
                            Tracked(instance),
                        );
                        proof {
                            lemma_peer_core_step_trans(
                                &before_u_core,
                                &before_await_core,
                                core,
                                self.rdma.timeline_id,
                            );
                            lemma_foreign_shares_trans(
                                old(core),
                                &before_await_core,
                                core,
                                old(self).rdma.active_leases@,
                                before_await_leases,
                                self.rdma.active_leases@,
                            );
                        }
                    } else {
                        proof {
                            lemma_foreign_shares_trans(
                                old(core),
                                &before_u_core,
                                core,
                                old(self).rdma.active_leases@,
                                before_u_leases,
                                self.rdma.active_leases@,
                            );
                            Self::lemma_session_wf_same_leases(&before_u_self, &*self, core);
                        }
                    }
                }
                Err(rdma_error(rdma::RdmaError(-1)))
            },
        }
    }

    /// Donor: serve a `FetchRequest { request_id, key }` by looking up `key` in
    /// `tier`, binding an RDMA Type-2 MW (polling CQ via `pump_cq` until bind
    /// finishes), and sending a `READ_GRANT` with constructive `VeriCacheAttachment::ReadGrant`
    /// (or `FetchMiss` if absent, bind fails, or bind times out cleanly with async tracking).
    fn lookup_serve_frame(
        core: &CacheCore,
        key: ObjectKey,
        primary_tier: TierId,
        fallback_tier: Option<TierId>,
    ) -> (r: Option<(FrameId, VersionId)>)
        requires
            core.well_formed(),
        ensures
            r matches Some((f, version)) ==> {
                &&& f.0 < core.frames_spec().metas_spec().len()
                &&& core.frames_spec().meta(f).state == STATE_PUBLISHED
                &&& core.frames_spec().frame(f).bound_key == Some(key)
                &&& core.frames_spec().frame(f).bound_version == Some(version)
            },
    {
        match core.lookup_tier(key, primary_tier) {
            Some(hit) => Some(hit),
            None => match fallback_tier {
                Some(fb) => core.lookup_tier(key, fb),
                None => None,
            },
        }
    }

    /// Donor-side handler for `FetchRequest { request_id, key }` checking `primary_tier`
    /// first and optionally falling back to `fallback_tier` (e.g., GPU HBM -> Host DRAM).
    #[verifier::rlimit(250)]
    pub(crate) fn serve_fetch_request_multi(
        &mut self,
        core: &mut CacheCore,
        primary_tier: TierId,
        fallback_tier: Option<TierId>,
        request_id: u64,
        key: ObjectKey,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).next_grant_tag < u64::MAX,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        hide(ActiveRemoteLease::valid);
        hide(super::protocol::statement_replica_valid);
        hide(replica_valid);
        hide(complete_receipts);
        let Some((frame, version)) = Self::lookup_serve_frame(
            core,
            key,
            primary_tier,
            fallback_tier,
        ) else {
            let _ = self.send_control(
                core,
                ControlPayload::FetchMiss { request_id },
                Tracked(instance),
            );
            proof {
                lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                lemma_foreign_shares_refl(core, self.rdma.active_leases@);
            }
            return Ok(ServeOutcome::Miss);
        };
        let ghost before_bind_core = *core;
        let ghost before_bind_self = *self;
        let Ok((handle, bind_wr_id)) = self.serve_bind(
            core,
            frame,
            Tracked(shares),
            Tracked(instance),
        ) else {
            let _ = self.send_control(
                core,
                ControlPayload::FetchMiss { request_id },
                Tracked(instance),
            );
            return Ok(ServeOutcome::Miss);
        };
        let ghost after_bind_core = *core;
        let ghost after_bind_leases = self.rdma.active_leases@;
        proof {
            before_bind_self.lemma_fresh_handle_excluded_from_session(&before_bind_core, handle);
            reveal(FrameTable::same_directory);
            lemma_core_wf_facts(core);
            lemma_published_frame_replica_valid(core, frame);
            lemma_published_frame_slot_valid(core, frame);
            lemma_initializing_frame_slot_valid(core, frame);
            lemma_citations_closure(core);
            assert(core.frames_spec().published(frame) == before_bind_core.frames_spec().published(
                frame,
            ));
        }
        let (_, frame_tier) = core.frame_binding(frame).unwrap();
        let frame_size = core.registered_tier_frame_size(frame_tier) as u64;
        let ghost cert = core.frames_spec().published(frame).unwrap().2;
        let ghost bind_sub = core.ledger_spec()@.submitted;
        let ghost bind_succ = core.ledger_spec()@.succeeded;
        let tracked bind_revoked = instance.duplicate_revoked(&core.borrow_ledger().revoked);
        let ghost bind_facts = core.citations();
        let Tracked(version_binding) = core.version_binding_for(version);
        proof {
            lemma_replica_valid_to_statement(core.ledger_spec()@, core.node(), cert);
            super::protocol::lemma_statement_replica_valid_revoked_ext(
                core.ledger_spec()@.submitted,
                core.ledger_spec()@.succeeded,
                core.ledger_spec()@.revoked,
                bind_revoked,
                core.citations(),
                cert,
            );
        }
        let await_res = self.serve_await_bind(
            core,
            handle,
            frame,
            bind_wr_id,
            Ghost(cert.writer),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans(old(core), &after_bind_core, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &after_bind_core,
                core,
                old(self).rdma.active_leases@,
                after_bind_leases,
                self.rdma.active_leases@,
            );
        }
        match await_res {
            Ok(BindAwaitStatus::Ready) => {
                let ghost before_send_core = *core;
                let ghost before_send_leases = self.rdma.active_leases@;
                let res = self.finish_send_bound_grant(
                    core,
                    handle,
                    frame,
                    request_id,
                    key,
                    version,
                    Ghost(cert),
                    Ghost(bind_sub),
                    Ghost(bind_succ),
                    Tracked(bind_revoked),
                    Ghost(bind_facts),
                    Tracked(version_binding),
                    Tracked(instance),
                );
                proof {
                    lemma_peer_core_step_trans(
                        old(core),
                        &before_send_core,
                        core,
                        self.rdma.timeline_id,
                    );
                    lemma_foreign_shares_trans(
                        old(core),
                        &before_send_core,
                        core,
                        old(self).rdma.active_leases@,
                        before_send_leases,
                        self.rdma.active_leases@,
                    );
                }
                res
            },
            Ok(BindAwaitStatus::BindFailed) => {
                let ghost pre_fail_self = *self;
                let ghost pre_fail_core = *core;
                let ghost pre_fail_leases = self.rdma.active_leases@;
                proof {
                    self.lemma_peer_wf_lease_facts(core);
                }
                let cfb_ok = self.rdma.complete_failed_bind(
                    core,
                    handle,
                    Tracked(instance),
                ).is_ok();
                proof {
                    lemma_foreign_shares_trans(
                        old(core),
                        &pre_fail_core,
                        core,
                        old(self).rdma.active_leases@,
                        pre_fail_leases,
                        self.rdma.active_leases@,
                    );
                }
                if cfb_ok {
                    proof {
                        Self::lemma_session_wf_after_outbound_lease_remove(
                            &pre_fail_self,
                            &*self,
                            &pre_fail_core,
                            core,
                            handle,
                        );
                        assert(peer_core_step(&pre_fail_core, core, self.rdma.timeline_id)) by {
                            reveal(peer_core_step_except);
                            reveal(core_transition);
                            reveal(FrameTable::same_directory);
                            reveal(FrameTable::same_layout);
                        };
                        lemma_peer_core_step_trans(
                            old(core),
                            &pre_fail_core,
                            core,
                            self.rdma.timeline_id,
                        );
                    }
                }
                let _ = self.send_control(
                    core,
                    ControlPayload::FetchMiss { request_id },
                    Tracked(instance),
                );
                Ok(ServeOutcome::Miss)
            },
            Ok(BindAwaitStatus::PendingOrFailed) => {
                let ghost pre_push_self = *self;
                let item = PendingOutboundGrant {
                    request_id,
                    key,
                    frame,
                    version,
                    frame_size,
                    handle,
                    bind_wr_id,
                    send_on_ready: false,
                    cert: Ghost(cert),
                    bind_sub: Ghost(bind_sub),
                    bind_succ: Ghost(bind_succ),
                    bind_revoked: Tracked(bind_revoked),
                    bind_facts: Ghost(bind_facts),
                    version_binding: Tracked(version_binding),
                };
                let ghost item_snap = item;
                self.pending_outbound_grants.push(item);
                proof {
                    Self::lemma_session_wf_after_push_outbound_grant(
                        &pre_push_self,
                        &*self,
                        core,
                        &item_snap,
                    );
                }
                let _ = self.send_control(
                    core,
                    ControlPayload::FetchMiss { request_id },
                    Tracked(instance),
                );
                Ok(ServeOutcome::Miss)
            },
            Err(e) => {
                let ghost pre_push_self = *self;
                let item = PendingOutboundGrant {
                    request_id,
                    key,
                    frame,
                    version,
                    frame_size,
                    handle,
                    bind_wr_id,
                    send_on_ready: false,
                    cert: Ghost(cert),
                    bind_sub: Ghost(bind_sub),
                    bind_succ: Ghost(bind_succ),
                    bind_revoked: Tracked(bind_revoked),
                    bind_facts: Ghost(bind_facts),
                    version_binding: Tracked(version_binding),
                };
                let ghost item_snap = item;
                self.pending_outbound_grants.push(item);
                proof {
                    Self::lemma_session_wf_after_push_outbound_grant(
                        &pre_push_self,
                        &*self,
                        core,
                        &item_snap,
                    );
                }
                Err(e)
            },
        }
    }

    /// Donor: handle an incoming `ReadRelease` by cooperatively unpinning and
    /// invalidating the bound MW (pumping CQ for `WindowInvalidated` without
    /// dropping any unrelated completions).
    #[verifier::rlimit(250)]
    pub(crate) fn handle_read_release(
        &mut self,
        core: &mut CacheCore,
        grant_tag: GrantTag,
        Tracked(share_cap): Tracked<ShareCap>,
        Tracked(statement): Tracked<Statement>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(self).rdma.well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).channel.local() == old(core).node(),
            old(self).channel.peer() == old(self).rdma.peer_node,
            old(self).channel.id() == old(core).instance_id(),
            old(self).session_queues_valid(old(core)),
            instance.id() == old(core).instance_id(),
            statement.valid(old(core).instance_id(), old(core).node()),
            crate::provenance::complete_receipts_map(statement.submitted),
            statement.knows.set() == knowledge_of(old(core).node(), statement.closure()),
            forall|t: GrantTag| #[trigger]
                old(self).channel.sent().contains_key(t) ==> t < old(self).next_grant_tag,
            forall|i: int, j: int|
                0 <= i < j < old(self).active_grants@.len() ==> (#[trigger] old(
                    self,
                ).active_grants@[i]).grant_tag != (#[trigger] old(self).active_grants@[j]).grant_tag
                    && old(self).active_grants@[i].handle != old(self).active_grants@[j].handle,
            !old(self).channel.sent().contains_key(grant_tag),
            forall|i: int|
                0 <= i < old(self).active_grants@.len() ==> {
                    let g = #[trigger] old(self).active_grants@[i];
                    &&& g.grant_tag < old(self).next_grant_tag
                    &&& g.frame.0 < old(core).frames_spec().metas_spec().len()
                    &&& old(self).rdma.lease_index_spec(g.handle) matches Some(idx) && {
                        let lease = old(self).rdma.active_leases@[idx];
                        &&& lease.frame == g.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& (g.grant_tag != grant_tag ==> old(self).channel.sent().contains_key(
                            g.grant_tag,
                        ) && old(self).channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                            slot: old(core).frames_spec().frame(g.frame).slot,
                            share_id: lease.share@,
                        }))
                        &&& (g.grant_tag == grant_tag ==> share_cap.is_fenced()
                            && share_cap.instance_id() == old(core).instance_id() && share_cap.loc()
                            == old(core).frames_spec().frame(g.frame).slot.loc && share_cap.share()
                            == lease.share@ && transfer::transport::covers(
                            share_cap.frontier(),
                            statement.closure(),
                        ))
                    }
                },
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).active_grants@.len() < old(self).active_grants@.len()
                || final(self).active_grants == old(self).active_grants,
            r.is_ok() ==> final(self).active_grants@.len() + 1 == old(self).active_grants@.len(),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        hide(ActiveRemoteLease::valid);
        hide(super::protocol::statement_replica_valid);
        hide(replica_valid);
        hide(complete_receipts);
        let Some(pos) = self.find_grant_index(grant_tag) else {
            proof {
                self.lemma_session_wf_after_release_lookup_miss(core, grant_tag);
                lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                lemma_foreign_shares_refl(core, self.rdma.active_leases@);
                lemma_same_pulls_preserves_pull_tracked(&old(self).rdma, &self.rdma);
            }
            return Err(CoreError::InvalidArgument);
        };
        let ghost entry_self = *self;
        let ghost stmt_sub = statement.submitted;
        let ghost stmt_facts = statement.facts;
        let ghost stmt_closure = statement.closure();
        let grant = self.active_grants.swap_remove(pos);
        proof {
            Self::lemma_session_wf_after_release_swap_remove(
                &entry_self,
                &*self,
                core,
                grant_tag,
                pos as int,
                grant,
                share_cap,
                stmt_closure,
            );
        }
        let ghost hidx = self.rdma.lease_index_spec(grant.handle).unwrap();
        // Merge peer statement so `share_cap.frontier().readers` are in `core.citations()`.
        let ghost before_merge = *core;
        core.merge_peer_statement(
            Tracked(statement.submitted),
            Tracked(statement.knows),
            Tracked(statement.succeeded),
            Tracked(instance),
        );
        proof {
            self.lemma_session_wf_after_merge_statement(
                &before_merge,
                core,
                stmt_sub,
                stmt_facts,
                stmt_closure,
            );
            lemma_foreign_shares_identical_leases(&before_merge, core, self.rdma.active_leases@);
        }
        let ghost pre_unbind_self = *self;
        let ghost pre_unbind_core = *core;
        let ghost pre_unbind_leases = self.rdma.active_leases@;
        let inv_wr_id = match self.rdma.unpin_and_unbind_lease(
            core,
            grant.handle,
            Tracked(share_cap),
            Ticket::NONE,
            Tracked(instance),
        ) {
            Ok(w) => w,
            Err((e, _)) => {
                proof {
                    lemma_foreign_shares_trans(
                        &before_merge,
                        &pre_unbind_core,
                        core,
                        old(self).rdma.active_leases@,
                        pre_unbind_leases,
                        self.rdma.active_leases@,
                    );
                    Self::lemma_session_wf_same_leases(&pre_unbind_self, &*self, core);
                    lemma_same_pulls_preserves_pull_tracked(&old(self).rdma, &self.rdma);
                }
                return Err(e);
            },
        };
        proof {
            lemma_peer_core_step_trans(
                &before_merge,
                &pre_unbind_core,
                core,
                self.rdma.timeline_id,
            );
            lemma_foreign_shares_trans(
                &before_merge,
                &pre_unbind_core,
                core,
                old(self).rdma.active_leases@,
                pre_unbind_leases,
                self.rdma.active_leases@,
            );
            Self::lemma_session_wf_after_outbound_lease_update(
                &pre_unbind_self,
                &*self,
                &pre_unbind_core,
                core,
                grant.handle,
                hidx,
            );
            lemma_same_pulls_preserves_pull_tracked(&old(self).rdma, &self.rdma);
        }
        let ghost before_await = *core;
        let ghost before_await_leases = self.rdma.active_leases@;
        // Drive `pump_cq` for `WindowInvalidated`.
        let res = self.await_unbind_invalidate(core, grant.handle, inv_wr_id, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(&before_merge, &before_await, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                &before_merge,
                &before_await,
                core,
                old(self).rdma.active_leases@,
                before_await_leases,
                self.rdma.active_leases@,
            );
        }
        res
    }

    /// Donor helper: drive `pump_cq` until `Unbinding` lease `handle` completes
    /// (`WindowInvalidated` auto-finalized by `pump_cq`, removing `handle` from `active_leases`).
    #[verifier::rlimit(350)]
    fn await_unbind_invalidate(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        inv_wr_id: u64,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).rdma.lease_index_spec(handle) matches Some(idx) && old(
                self,
            ).rdma.active_leases@[idx].wr_id == inv_wr_id && old(
                self,
            ).rdma.active_leases@[idx].state == RemoteLeaseState::Unbinding,
            forall|i: int|
                0 <= i < old(self).active_grants@.len() ==> (#[trigger] old(
                    self,
                ).active_grants@[i]).handle != handle,
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut spins: u32 = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec() == old(core).frames_spec(),
                core.catalog_spec() == old(core).catalog_spec(),
                core.spec_spec() == old(core).spec_spec(),
                core.next_version_seq_spec() == old(core).next_version_seq_spec(),
                core.version_seq_limit_spec() == old(core).version_seq_limit_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                core.ledger_spec()@.submitted == old(core).ledger_spec()@.submitted,
                core.ledger_spec()@.revoked == old(core).ledger_spec()@.revoked,
                old(core).ledger_spec()@.knows.set().subset_of(core.ledger_spec()@.knows.set()),
                old(core).ledger_spec()@.succeeded.dom().subset_of(
                    core.ledger_spec()@.succeeded.dom(),
                ),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
                forall|seq: u64, writer: TransferEvent| #[trigger]
                    pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                        &self.rdma,
                        seq,
                        writer,
                    ),
                self.rdma.lease_index_spec(handle) matches Some(idx)
                    && self.rdma.active_leases@[idx].wr_id == inv_wr_id
                    && self.rdma.active_leases@[idx].state == RemoteLeaseState::Unbinding,
            decreases 1_000_000 - spins,
        {
            let ghost step_core = *core;
            let ghost step_leases = self.rdma.active_leases@;
            let ghost step_rdma = self.rdma;
            let pump_res = self.rdma.pump_cq(core, Tracked(instance));
            proof {
                lemma_foreign_shares_trans(
                    old(core),
                    &step_core,
                    core,
                    old(self).rdma.active_leases@,
                    step_leases,
                    self.rdma.active_leases@,
                );
            }
            match pump_res {
                Err(e) => return Err(e),
                Ok(None) => {
                    spins += 1;
                },
                Ok(Some(cqe_id)) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        assert(old(core).ledger_spec()@.knows.set().subset_of(
                            core.ledger_spec()@.knows.set(),
                        )) by {
                            broadcast use vstd::iset::group_iset_lemmas;

                        };
                        assert(old(core).ledger_spec()@.succeeded.dom().subset_of(
                            core.ledger_spec()@.succeeded.dom(),
                        )) by {
                            broadcast use vstd::iset::group_iset_lemmas;

                        };
                        assert forall|seq: u64, writer: TransferEvent| #[trigger]
                            pull_tracked_or_done(
                                &old(self).rdma,
                                seq,
                                writer,
                            ) implies pull_tracked_or_done(&self.rdma, seq, writer) by {
                            lemma_pump_cq_preserves_pull_tracked(
                                &step_rdma,
                                &self.rdma,
                                seq,
                                writer,
                                cqe_id,
                            );
                        }
                        assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                            let g = #[trigger] self.active_grants@[i];
                            &&& g.grant_tag < self.next_grant_tag
                            &&& g.frame.0 < core.frames_spec().metas_spec().len()
                            &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                                let lease = self.rdma.active_leases@[idx];
                                &&& lease.frame == g.frame
                                &&& lease.state == RemoteLeaseState::Bound
                                &&& self.channel.sent().contains_key(g.grant_tag)
                                &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                                    slot: core.frames_spec().frame(g.frame).slot,
                                    share_id: lease.share@,
                                })
                            }
                        } by {
                            reveal(rdma_leases_valid);
                            let g = self.active_grants@[i];
                            let oidx = rdma_lease_index_spec(step_leases, g.handle).unwrap();
                            assert(step_leases[oidx].state == RemoteLeaseState::Bound);
                            assert(!step_leases[oidx].has_active_wr());
                        }
                        assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                            reveal(FrameTable::same_layout);
                        }
                        lemma_outbound_grants_preserved(
                            self.pending_outbound_grants@,
                            step_leases,
                            self.rdma.active_leases@,
                            &step_core,
                            core,
                        );
                    }
                    match self.rdma.lease_state(handle) {
                        None => return Ok(()),
                        Some(RemoteLeaseState::Unbinding) => {
                            spins += 1;
                        },
                        Some(RemoteLeaseState::ControlFailedState) | _ => {
                            return Err(rdma_error(rdma::RdmaError(-1)));
                        },
                    }
                },
            }
        }
        Err(rdma_error(rdma::RdmaError(11)))
    }

    /// Donor: unilaterally revoke an active grant via fenced local MW
    /// invalidation (`Quarantined` until peer `RevokeAck`) and send `RevokeNotice`.
    #[verifier::rlimit(250)]
    pub fn revoke_lease_unilateral(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        grant_tag: GrantTag,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u32, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).rdma_spec().qp_spec().ready(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend,
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().qp_spec() == old(self).rdma_spec().qp_spec(),
            r.is_ok() ==> final(self).active_grants_spec()@.len() + 1 == old(
                self,
            ).active_grants_spec()@.len(),
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        hide(ActiveRemoteLease::valid);
        hide(super::protocol::statement_replica_valid);
        hide(replica_valid);
        hide(complete_receipts);
        let Some(pos) = self.find_grant_index(grant_tag) else {
            proof {
                lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                lemma_foreign_shares_refl(core, self.rdma.active_leases@);
            }
            return Err(CoreError::InvalidArgument);
        };
        let ghost entry_self = *self;
        let grant = self.active_grants.swap_remove(pos);
        proof {
            Self::lemma_session_wf_after_revoke_swap_remove(
                &entry_self,
                &*self,
                core,
                pos as int,
                grant,
            );
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        let ghost hidx = self.rdma.lease_index_spec(grant.handle).unwrap();
        let ghost pre_rev_self = *self;
        let _inv_wr_id = match self.rdma.begin_fenced_invalidate(core, grant.handle) {
            Ok(w) => w,
            Err(e) => {
                proof {
                    Self::lemma_session_wf_same_leases(&pre_rev_self, &*self, core);
                }
                return Err(e);
            },
        };
        proof {
            assert(core.frames_spec().same_directory(&core.frames_spec())) by {
                reveal(FrameTable::same_directory);
            };
            Self::lemma_session_wf_after_outbound_lease_update(
                &pre_rev_self,
                &*self,
                core,
                core,
                grant.handle,
                hidx,
            );
        }
        let mut spins: u32 = 0;
        let mut ready = false;
        let ghost post_remove_grants = self.active_grants;
        while spins < 1_000_000 && !ready
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec() == old(core).frames_spec(),
                core.catalog_spec() == old(core).catalog_spec(),
                core.spec_spec() == old(core).spec_spec(),
                core.next_version_seq_spec() == old(core).next_version_seq_spec(),
                core.version_seq_limit_spec() == old(core).version_seq_limit_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                core.ledger_spec()@.submitted == old(core).ledger_spec()@.submitted,
                core.ledger_spec()@.revoked == old(core).ledger_spec()@.revoked,
                old(core).ledger_spec()@.knows.set().subset_of(core.ledger_spec()@.knows.set()),
                old(core).ledger_spec()@.succeeded.dom().subset_of(
                    core.ledger_spec()@.succeeded.dom(),
                ),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == post_remove_grants,
                self.active_grants@.len() + 1 == old(self).active_grants@.len(),
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
                forall|i: int|
                    0 <= i < self.active_grants@.len() ==> (
                    #[trigger] self.active_grants@[i]).handle != grant.handle,
                forall|k: int|
                    0 <= k < self.pending_outbound_grants@.len() ==> (
                    #[trigger] self.pending_outbound_grants@[k]).handle != grant.handle,
            decreases
                    1_000_000 - spins + (if ready {
                        0int
                    } else {
                        1int
                    }),
        {
            let ghost step_core = *core;
            let ghost step_leases = self.rdma.active_leases@;
            let ghost step_self = *self;
            proof {
                self.lemma_peer_wf_lease_facts(core);
            }
            let pump_res = self.rdma.pump_cq(core, Tracked(instance));
            proof {
                lemma_foreign_shares_trans(
                    old(core),
                    &step_core,
                    core,
                    old(self).rdma.active_leases@,
                    step_leases,
                    self.rdma.active_leases@,
                );
            }
            match pump_res {
                Err(e) => return Err(e),
                Ok(None) => {
                    spins += 1;
                },
                Ok(Some(_)) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        Self::lemma_session_wf_after_pump_for_revoke(
                            &step_self,
                            &*self,
                            old(core),
                            &step_core,
                            core,
                        );
                        self.lemma_peer_wf_lease_facts(core);
                    }
                    match self.rdma.lease_state(grant.handle) {
                        Some(RemoteLeaseState::RevokeInvReady) => {
                            ready = true;
                        },
                        Some(RemoteLeaseState::Revoking) => {
                            spins += 1;
                        },
                        _ => return Err(rdma_error(rdma::RdmaError(-1))),
                    }
                },
            }
        }
        if !ready {
            return Err(rdma_error(rdma::RdmaError(11)));
        }
        let ghost pre_fin_self = *self;
        let ghost pre_fin_core = *core;
        let ghost pre_fin_leases = self.rdma.active_leases@;
        proof {
            self.lemma_peer_wf_lease_facts(core);
        }
        let mw_slot = match self.rdma.finish_ready_fenced_revoke(
            core,
            host,
            grant.handle,
            Tracked(instance),
        ) {
            Ok(s) => s,
            Err(e) => {
                proof {
                    lemma_foreign_shares_trans(
                        old(core),
                        &pre_fin_core,
                        core,
                        old(self).rdma.active_leases@,
                        pre_fin_leases,
                        self.rdma.active_leases@,
                    );
                    Self::lemma_session_wf_same_leases(&pre_fin_self, &*self, core);
                }
                return Err(e);
            },
        };
        proof {
            lemma_peer_core_step_trans(old(core), &pre_fin_core, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &pre_fin_core,
                core,
                old(self).rdma.active_leases@,
                pre_fin_leases,
                self.rdma.active_leases@,
            );
            Self::lemma_session_wf_after_outbound_lease_remove(
                &pre_fin_self,
                &*self,
                &pre_fin_core,
                core,
                grant.handle,
            );
        }
        let ghost pre_q_self = *self;
        self.quarantined_grants.push((grant_tag, mw_slot));
        proof {
            Self::lemma_session_wf_same_active_and_queues(&pre_q_self, &*self, core);
        }
        let _ = self.send_control(
            core,
            ControlPayload::RevokeNotice { grant_tag },
            Tracked(instance),
        );
        Ok(mw_slot)
    }

    /// Initiator helper: drain CQ via `pump_cq` until `pull.wr_id` completes
    /// (never discarding unrelated data or control CQEs).
    #[verifier::rlimit(250)]
    fn fetch_await_pull(
        &mut self,
        core: &mut CacheCore,
        pull: &InFlightRdmaPull,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Result<(), i32>, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            pull.valid(&old(self).rdma, old(core)),
            old(self).rdma.pending().contains_key(pull.wr_id) || old(self).rdma.completed_pull_spec(
                pull.wr_id,
            ) is Some,
            old(self).rdma.pending().contains_key(pull.wr_id) ==> old(
                self,
            ).rdma.pending()[pull.wr_id] == pull.writer@,
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
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            pull.valid(&final(self).rdma, final(core)),
            old(core).ledger_spec()@.submitted == final(core).ledger_spec()@.submitted,
            old(core).ledger_spec()@.revoked == final(core).ledger_spec()@.revoked,
            old(core).citations().subset_of(final(core).citations()),
            r matches Ok(st) ==> {
                &&& final(core).ledger_spec()@.knows.set().contains(
                    (final(core).node(), pull.writer@),
                )
                &&& final(core).citations().contains(pull.writer@)
                &&& (st.is_ok() ==> event_succeeded(
                    final(core).ledger_spec()@,
                    final(core).node(),
                    pull.writer@,
                ))
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut spins: u32 = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec() == old(core).frames_spec(),
                core.catalog_spec() == old(core).catalog_spec(),
                core.spec_spec() == old(core).spec_spec(),
                core.next_version_seq_spec() == old(core).next_version_seq_spec(),
                core.version_seq_limit_spec() == old(core).version_seq_limit_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
                pull.valid(&self.rdma, core),
                old(core).ledger_spec()@.submitted == core.ledger_spec()@.submitted,
                old(core).ledger_spec()@.revoked == core.ledger_spec()@.revoked,
                old(core).citations().subset_of(core.citations()),
                self.rdma.pending().contains_key(pull.wr_id) || self.rdma.completed_pull_spec(
                    pull.wr_id,
                ) is Some,
                self.rdma.pending().contains_key(pull.wr_id) ==> self.rdma.pending()[pull.wr_id]
                    == pull.writer@,
            decreases 1_000_000 - spins,
        {
            let ghost step_core = *core;
            let ghost step_leases = self.rdma.active_leases@;
            let ghost step_self = *self;
            let ghost step_rdma = self.rdma;
            if let Some(st) = self.rdma.take_completed_pull(core, pull.wr_id) {
                proof {
                    lemma_core_wf_facts(core);
                    lemma_core_timelines_wf(core);
                    assert(core.ledger_spec()@.knows.set().contains((core.node(), pull.writer@)));
                    assert(core.ledger_spec()@.submitted.dom().contains(pull.writer@));
                    assert(core.citations().contains(pull.writer@));
                    if st.is_ok() {
                        assert(core.ledger_spec()@.succeeded.dom().contains(pull.writer@));
                        assert(event_succeeded(core.ledger_spec()@, core.node(), pull.writer@));
                    }
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        step_leases,
                        self.rdma.active_leases@,
                        &step_core,
                        core,
                    );
                }
                return Ok(st);
            }
            let ghost pre_pump_rdma = self.rdma;
            let pump_res = self.rdma.pump_cq(core, Tracked(instance));
            proof {
                lemma_foreign_shares_trans(
                    old(core),
                    &step_core,
                    core,
                    old(self).rdma.active_leases@,
                    step_leases,
                    self.rdma.active_leases@,
                );
            }
            match pump_res {
                Err(e) => return Err(e),
                Ok(None) => {
                    spins += 1;
                },
                Ok(Some(wr_id)) => {
                    proof {
                        reveal(rdma_leases_valid);
                        reveal(FrameTable::same_directory);
                        reveal(FrameTable::same_layout);
                        lemma_peer_core_step_trans(
                            old(core),
                            &step_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        broadcast use {vstd::iset::group_iset_lemmas, vstd::map::group_map_lemmas};

                        assert(core.frames_spec().same_layout(&step_core.frames_spec()));
                        assert(step_core.citations().subset_of(core.citations()));
                        assert(pull.copy@.before.citations(core.node()).subset_of(
                            core.citations(),
                        ));
                        lemma_pull_valid_after_peer_step(
                            &step_rdma,
                            &self.rdma,
                            &step_core,
                            core,
                            pull,
                            None,
                        );
                        assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                            let g = #[trigger] self.active_grants@[i];
                            &&& g.grant_tag < self.next_grant_tag
                            &&& g.frame.0 < core.frames_spec().metas_spec().len()
                            &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                                let lease = self.rdma.active_leases@[idx];
                                &&& lease.frame == g.frame
                                &&& lease.state == RemoteLeaseState::Bound
                                &&& self.channel.sent().contains_key(g.grant_tag)
                                &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                                    slot: core.frames_spec().frame(g.frame).slot,
                                    share_id: lease.share@,
                                })
                            }
                        } by {
                            let g = self.active_grants@[i];
                            assert(step_self.active_grants@[i] == g);
                            assert(step_rdma.lease_index_spec(g.handle) is Some);
                            assert(pre_pump_rdma.active_leases@ == step_leases);
                            assert(pre_pump_rdma.lease_index_spec(g.handle) is Some);
                            let oidx = pre_pump_rdma.lease_index_spec(g.handle)->Some_0;
                            assert(step_leases[oidx].state == RemoteLeaseState::Bound);
                            assert(!step_leases[oidx].has_active_wr());
                        }
                        assert forall|i: int| 0 <= i < self.pending_outbound_grants@.len() implies {
                            let p = #[trigger] self.pending_outbound_grants@[i];
                            &&& rdma_lease_index_spec(step_leases, p.handle) matches Some(bidx) && {
                                let bl = step_leases[bidx];
                                &&& rdma_lease_index_spec(
                                    self.rdma.active_leases@,
                                    p.handle,
                                ) matches Some(aidx)
                                &&& self.rdma.active_leases@[aidx].frame == bl.frame
                                &&& self.rdma.active_leases@[aidx].wr_id == bl.wr_id
                                &&& super::poll::rdma_lease_is_binding_like(
                                    self.rdma.active_leases@[aidx],
                                )
                                &&& self.rdma.active_leases@[aidx].share_cap == bl.share_cap
                            }
                        } by {
                            let p = self.pending_outbound_grants@[i];
                            assert(step_self.pending_outbound_grants@[i] == p);
                            assert(step_rdma.lease_index_spec(p.handle) is Some);
                            assert(pre_pump_rdma.active_leases@ == step_leases);
                            assert(pre_pump_rdma.lease_index_spec(p.handle) is Some);
                            let bidx = pre_pump_rdma.lease_index_spec(p.handle)->Some_0;
                            assert(super::poll::rdma_lease_is_binding_like(
                                pre_pump_rdma.active_leases@[bidx],
                            ));
                            assert(self.rdma.lease_index_spec(p.handle) is Some);
                        }
                        lemma_outbound_grants_preserved(
                            self.pending_outbound_grants@,
                            step_leases,
                            self.rdma.active_leases@,
                            &step_core,
                            core,
                        );
                    }
                    if wr_id == pull.wr_id {
                        if let Some(st) = self.rdma.take_completed_pull(core, pull.wr_id) {
                            proof {
                                lemma_core_wf_facts(core);
                                lemma_core_timelines_wf(core);
                                assert(core.citations().contains(pull.writer@));
                                if st.is_ok() {
                                    assert(event_succeeded(
                                        core.ledger_spec()@,
                                        core.node(),
                                        pull.writer@,
                                    ));
                                }
                            }
                            return Ok(st);
                        }
                    }
                    spins += 1;
                },
            }
        }
        Err(rdma_error(rdma::RdmaError(11)))
    }

    fn remove_inbound_grants_by_tag(&mut self, tag: GrantTag)
        ensures
            final(self).rdma == old(self).rdma,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            final(self).pending_inbound_grants@.len() <= old(self).pending_inbound_grants@.len(),
            forall|k: GrantTag|
                k != tag && #[trigger] old(self).channel.received().contains_key(k) ==> (
                final(self).channel.received().contains_key(k) && final(self).channel.received()[k]
                    == old(self).channel.received()[k]),
            forall|k: GrantTag| #[trigger]
                inbound_tag_unbuffered(old(self).pending_inbound_grants@, k)
                    ==> inbound_tag_unbuffered(final(self).pending_inbound_grants@, k),
            forall|i: int|
                0 <= i < final(self).pending_inbound_grants@.len() ==> (
                #[trigger] final(self).pending_inbound_grants@[i]).tag != tag && exists|j: int|
                    0 <= j < old(self).pending_inbound_grants@.len() && old(
                        self,
                    ).pending_inbound_grants@[j] == final(self).pending_inbound_grants@[i],
    {
        let mut i: usize = 0;
        let ghost orig_grants = old(self).pending_inbound_grants@;
        proof {
            assert(orig_grants == old(self).pending_inbound_grants@);
            assert forall|k: int|
                0 <= k < self.pending_inbound_grants@.len() implies inbound_grant_in_seq(
                orig_grants,
                #[trigger] self.pending_inbound_grants@[k],
            ) by {
                assert(0 <= k < orig_grants.len() && orig_grants[k]
                    == self.pending_inbound_grants@[k]);
            }
        }
        while i < self.pending_inbound_grants.len()
            invariant
                i <= self.pending_inbound_grants@.len(),
                orig_grants == old(self).pending_inbound_grants@,
                self.pending_inbound_grants@.len() <= old(self).pending_inbound_grants@.len(),
                self.rdma == old(self).rdma,
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
                forall|k: int|
                    0 <= k < i ==> (#[trigger] self.pending_inbound_grants@[k]).tag != tag,
                forall|k: int|
                    0 <= k < self.pending_inbound_grants@.len() ==> inbound_grant_in_seq(
                        orig_grants,
                        #[trigger] self.pending_inbound_grants@[k],
                    ),
            decreases self.pending_inbound_grants@.len() - i,
        {
            if self.pending_inbound_grants[i].tag == tag {
                let ghost before_sr = self.pending_inbound_grants@;
                let _ = self.pending_inbound_grants.swap_remove(i);
                proof {
                    assert forall|k: int|
                        0 <= k < self.pending_inbound_grants@.len() implies inbound_grant_in_seq(
                        orig_grants,
                        #[trigger] self.pending_inbound_grants@[k],
                    ) by {
                        let old_idx = if k == i as int {
                            (before_sr.len() - 1) as int
                        } else {
                            k
                        };
                        assert(0 <= old_idx < before_sr.len());
                        assert(before_sr[old_idx] == self.pending_inbound_grants@[k]);
                        assert(inbound_grant_in_seq(orig_grants, before_sr[old_idx]));
                    }
                }
            } else {
                i += 1;
            }
        }
        proof {
            assert forall|i: int| 0 <= i < self.pending_inbound_grants@.len() implies exists|j: int|

                0 <= j < old(self).pending_inbound_grants@.len() && old(
                    self,
                ).pending_inbound_grants@[j] == (#[trigger] self.pending_inbound_grants@[i]) by {
                assert(inbound_grant_in_seq(orig_grants, self.pending_inbound_grants@[i]));
                let j = choose|j: int|
                    0 <= j < orig_grants.len() && (#[trigger] orig_grants[j])
                        == self.pending_inbound_grants@[i];
                assert(0 <= j < old(self).pending_inbound_grants@.len() && old(
                    self,
                ).pending_inbound_grants@[j] == self.pending_inbound_grants@[i]);
            }
        }
    }

    pub(crate) fn take_inbound_grant(&mut self, core: &CacheCore, request_id: u64) -> (r: Option<
        PendingInboundGrant,
    >)
        requires
            core.well_formed(),
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).rdma == old(self).rdma,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            r matches Some(g) ==> {
                &&& g.meta.request_id == request_id
                &&& g.valid(core, &final(self).channel, final(self).rdma.peer_node)
            },
    {
        let mut i: usize = 0;
        let ghost orig_grants = old(self).pending_inbound_grants@;
        proof {
            assert forall|k: int| 0 <= k < self.pending_inbound_grants@.len() implies exists|j: int|

                0 <= j < old(self).pending_inbound_grants@.len() && old(
                    self,
                ).pending_inbound_grants@[j] == (#[trigger] self.pending_inbound_grants@[k]) by {
                let g = self.pending_inbound_grants@[k];
                assert(0 <= k < orig_grants.len() && orig_grants[k] == g);
            }
        }
        while i < self.pending_inbound_grants.len()
            invariant
                i <= self.pending_inbound_grants@.len(),
                core.well_formed(),
                self.well_formed(core),
                self.rdma == old(self).rdma,
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            decreases self.pending_inbound_grants@.len() - i,
        {
            if self.pending_inbound_grants[i].meta.request_id == request_id {
                let g = self.pending_inbound_grants.swap_remove(i);
                return Some(g);
            }
            i += 1;
        }
        None
    }

    pub(crate) fn take_inbound_miss(&mut self, request_id: u64) -> (r: bool)
        ensures
            final(self).rdma == old(self).rdma,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
    {
        let mut i: usize = 0;
        while i < self.pending_inbound_misses.len()
            invariant
                i <= self.pending_inbound_misses@.len(),
                self.rdma == old(self).rdma,
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            decreases self.pending_inbound_misses@.len() - i,
        {
            if self.pending_inbound_misses[i] == request_id {
                let _ = self.pending_inbound_misses.swap_remove(i);
                return true;
            }
            i += 1;
        }
        false
    }

    /// Initiator helper: send a `READ_RELEASE` returning `share_cap` for `tag`,
    /// removing any matching `tag` entry from `pending_inbound_grants`.
    fn fetch_release_grant(
        &mut self,
        core: &CacheCore,
        tag: GrantTag,
        request_id: u64,
        Tracked(share_cap): Tracked<ShareCap>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
            old(self).channel.received().contains_key(tag),
            old(self).channel.received()[tag] is ReadShare,
            share_cap.is_fenced(),
            share_cap.instance_id() == core.instance_id(),
            share_cap.loc() == old(self).channel.received()[tag]->ReadShare_slot.loc,
            share_cap.share() == old(self).channel.received()[tag]->ReadShare_share_id,
            transfer::transport::covers(share_cap.frontier(), core.citations()),
        ensures
            final(self).well_formed(core),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma == old(self).rdma,
            final(self).channel.same_session(old(self).channel),
            final(self).channel.sent() == old(self).channel.sent(),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            final(self).pending_inbound_grants@.len() <= old(self).pending_inbound_grants@.len(),
            forall|k: GrantTag|
                k != tag && #[trigger] old(self).channel.received().contains_key(k) ==> (
                final(self).channel.received().contains_key(k) && final(self).channel.received()[k]
                    == old(self).channel.received()[k]),
            forall|k: GrantTag| #[trigger]
                inbound_tag_unbuffered(old(self).pending_inbound_grants@, k)
                    ==> inbound_tag_unbuffered(final(self).pending_inbound_grants@, k),
            forall|i: int|
                0 <= i < final(self).pending_inbound_grants@.len() ==> (
                #[trigger] final(self).pending_inbound_grants@[i]).tag != tag,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost pre_inbound = self.pending_inbound_grants@;
        self.remove_inbound_grants_by_tag(tag);
        let rel_meta = ReadReleaseMeta { request_id };
        let rel_bytes = encode_read_release_meta(rel_meta);
        let ghost cited = core.citations();
        let tracked att = VeriCacheAttachment::ReadRelease { meta: rel_meta };
        proof {
            lemma_core_wf_facts(core);
            lemma_citations_closure(core);
            reveal(complete_receipts);
            assert(transfer::transport::covers(
                share_cap.frontier(),
                closure_from(core.ledger_spec()@.submitted.map(), cited),
            ));
        }
        let _ = self.channel.send_remote_read_release_with(
            tag,
            &rel_bytes,
            Tracked(instance),
            Tracked(&core.borrow_ledger().submitted),
            Tracked(&core.borrow_ledger().knows),
            Tracked(&core.borrow_ledger().succeeded),
            Ghost(cited),
            Tracked(share_cap),
            Tracked(att),
        );
        proof {
            broadcast use vstd::map::group_map_lemmas;

            assert forall|i: int| 0 <= i < self.pending_inbound_grants@.len() implies (
            #[trigger] self.pending_inbound_grants@[i]).valid(
                core,
                &self.channel,
                self.rdma.peer_node,
            ) by {
                let g = self.pending_inbound_grants@[i];
                let j = choose|j: int| 0 <= j < pre_inbound.len() && pre_inbound[j] == g;
                assert(pre_inbound[j].valid(core, &old(self).channel, self.rdma.peer_node));
                assert(g.tag != tag);
            }
            assert forall|k: GrantTag| #[trigger]
                inbound_tag_unbuffered(pre_inbound, k) implies inbound_tag_unbuffered(
                self.pending_inbound_grants@,
                k,
            ) by {
                assert forall|i: int| 0 <= i < self.pending_inbound_grants@.len() implies (
                #[trigger] self.pending_inbound_grants@[i]).tag != k by {
                    let g = self.pending_inbound_grants@[i];
                    let j = choose|j: int| 0 <= j < pre_inbound.len() && pre_inbound[j] == g;
                    assert(pre_inbound[j].tag != k);
                }
            }
        }
    }

    pub(crate) fn discard_read_grant(
        &mut self,
        core: &mut CacheCore,
        g: PendingInboundGrant,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(self).well_formed(old(core)),
            old(core).well_formed(),
            old(core).all_versions_bound(),
            g.valid(old(core), &old(self).channel, old(self).rdma.peer_node),
            instance.id() == old(core).instance_id(),
        ensures
            final(self).well_formed(final(core)),
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).rdma == old(self).rdma,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
    {
        hide(CacheCore::well_formed);
        let ghost before = *core;
        let tracked stmt = g.statement.get();
        let ghost stmt_sub = stmt.submitted;
        let ghost stmt_facts = stmt.facts;
        let ghost stmt_closure = stmt.closure();
        core.merge_peer_statement(
            Tracked(stmt.submitted),
            Tracked(stmt.knows),
            Tracked(stmt.succeeded),
            Tracked(instance),
        );
        proof {
            self.lemma_session_wf_after_merge_statement(
                &before,
                core,
                stmt_sub,
                stmt_facts,
                stmt_closure,
            );
            lemma_foreign_shares_identical_leases(&before, core, self.rdma.active_leases@);
            assert(transfer::transport::covers(g.share@.frontier(), core.citations()));
        }
        self.fetch_release_grant(core, g.tag, g.meta.request_id, g.share, Tracked(instance));
    }

    /// Drain all stale buffered `pending_inbound_grants`, merging statements and
    /// releasing every `ShareCap` over `self.channel`.
    #[verifier::rlimit(200)]
    fn drain_stale_inbound_grants(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
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
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma == old(self).rdma,
            final(self).channel.same_session(old(self).channel),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants@.len() == 0,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        while self.pending_inbound_grants.len() > 0
            invariant
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec() == old(core).frames_spec(),
                core.catalog_spec() == old(core).catalog_spec(),
                core.spec_spec() == old(core).spec_spec(),
                core.next_version_seq_spec() == old(core).next_version_seq_spec(),
                core.version_seq_limit_spec() == old(core).version_seq_limit_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma == old(self).rdma,
                self.channel.same_session(old(self).channel),
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_revoke_acks == old(self).pending_revoke_acks,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.quarantined_grants == old(self).quarantined_grants,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            decreases self.pending_inbound_grants@.len(),
        {
            let g = self.pending_inbound_grants.pop().unwrap();
            let ghost step_core = *core;
            let tracked stmt = g.statement.get();
            let ghost stmt_sub = stmt.submitted;
            let ghost stmt_facts = stmt.facts;
            let ghost stmt_closure = stmt.closure();
            core.merge_peer_statement(
                Tracked(stmt.submitted),
                Tracked(stmt.knows),
                Tracked(stmt.succeeded),
                Tracked(instance),
            );
            proof {
                lemma_rdma_window_after_fence_ledger(
                    self.rdma.in_flight_ops,
                    self.rdma.bitmap,
                    &core.timelines_spec(),
                    self.rdma.timeline_id,
                    &step_core.ledger_spec()@,
                    &core.ledger_spec()@,
                    rdma::CqSnapshot::of(self.rdma.cq),
                    core.instance_id(),
                    self.rdma.next_control_wr_id,
                );
                lemma_completed_pulls_after_step(
                    self.rdma.completed_pulls,
                    &step_core.timelines_spec(),
                    &core.timelines_spec(),
                    self.rdma.timeline_id,
                    self.rdma.cq.observer(),
                    &step_core.ledger_spec()@,
                    &core.ledger_spec()@,
                );
                assert(core.frames_spec().same_directory(&step_core.frames_spec())) by {
                    reveal(FrameTable::same_directory);
                }
                assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                    reveal(FrameTable::same_layout);
                }
                lemma_rdma_all_slabs_same_directory(
                    &self.rdma.host_mr,
                    &self.rdma.gpu_mr,
                    self.rdma.shared_host_mr,
                    self.rdma.shared_gpu_mr,
                    &step_core,
                    core,
                    self.rdma.window_pool.pd(),
                );
                lemma_rdma_leases_same_frames(
                    self.rdma.active_leases@,
                    &self.rdma.window_pool,
                    &self.rdma.qp,
                    &step_core,
                    core,
                );
                lemma_outbound_grants_preserved(
                    self.pending_outbound_grants@,
                    self.rdma.active_leases@,
                    self.rdma.active_leases@,
                    &step_core,
                    core,
                );
                assert(peer_core_step(&step_core, core, self.rdma.timeline_id)) by {
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::preserves_initializing);
                }
                lemma_peer_core_step_trans(old(core), &step_core, core, self.rdma.timeline_id);
                assert(stmt_closure.subset_of(core.citations())) by {
                    assert forall|e: TransferEvent| #[trigger]
                        stmt_closure.contains(e) implies core.citations().contains(e) by {
                        assert(stmt_sub.dom().contains(e)) by {
                            if !stmt_facts.contains(e) {
                                let m = choose|m: TransferEvent|
                                    stmt_facts.contains(m) && (
                                    #[trigger] stmt_sub.map()[m]).preds.contains(e);
                                assert(stmt_sub.dom().contains(m));
                                assert(stmt_sub.map()[m].preds.subset_of(stmt_sub.dom()));
                            }
                        }
                        assert(core.ledger_spec()@.submitted.dom().contains(e));
                        assert(core.ledger_spec()@.knows.set().contains((core.node(), e)));
                    }
                }
                assert(transfer::transport::covers(g.share@.frontier(), core.citations()));
            }
            self.fetch_release_grant(core, g.tag, g.meta.request_id, g.share, Tracked(instance));
        }
        proof {
            lemma_foreign_shares_identical_leases(old(core), core, self.rdma.active_leases@);
        }
    }

    #[verifier::rlimit(120)]
    fn fetch_prepare_pull(
        &mut self,
        core: &mut CacheCore,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        meta: ReadGrantMeta,
        cap: rdma::RemoteCapability,
        Tracked(share): Tracked<ShareCap>,
        Tracked(statement): Tracked<Statement>,
        Ghost(cert): Ghost<ValueCertificate>,
        Tracked(revoked): Tracked<RevokedMap>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PreparedPull, (CoreError, Tracked<ShareCap>)>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(core).instance_id(),
            statement.valid(old(core).instance_id(), old(core).node()),
            statement.knows.set() == knowledge_of(old(core).node(), statement.closure()),
            crate::provenance::complete_receipts_map(statement.submitted),
            cap.well_formed(),
            cap.slot_ref().loc.node == old(self).rdma.peer_node,
            cap.access() == rdma::RemoteAccess::ReadOnly,
            cap.share() == Some(share.share()),
            cap.slot_ref() == cert.slot,
            cert.slot.is_full(),
            cert.bytes.len() == meta.frame_size,
            share.is_fenced(),
            share.instance_id() == old(core).instance_id(),
            share.loc() == cap.slot_ref().loc,
            share.frontier().last_writer == Some(cert.writer),
            transfer::transport::covers(share.frontier(), statement.closure()),
            revoked.instance_id() == old(core).instance_id(),
            super::protocol::statement_replica_valid(
                statement.submitted,
                statement.succeeded,
                revoked,
                statement.facts,
                cert,
            ),
            match target {
                PullTarget::Host => old(self).rdma.effective_host_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma.effective_gpu_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            final(self).rdma == old(self).rdma,
            final(core).frames_spec().same_directory(&old(core).frames_spec()),
            final(core).frames_spec().preserves_fenced_leases(&old(core).frames_spec()),
            final(core).timelines_spec() == old(core).timelines_spec(),
            final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
            old(core).citations().subset_of(final(core).citations()),
            transfer::transport::covers(share.frontier(), final(core).citations()),
            r matches Ok(p) ==> {
                &&& p.valid(&final(self).rdma, final(core), target)
                &&& p.handle.frame.0 < old(core).frames_spec().metas_spec().len()
                &&& old(core).frames_spec().meta(p.handle.frame).state != STATE_INITIALIZING
                &&& p.handle.version == meta.version
                &&& p.source@ == cert
                &&& p.share@ == share
                &&& p.remote == cap
            },
            r matches Err((_, sh)) ==> sh@ == share,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost stmt_sub = statement.submitted;
        let ghost stmt_succ = statement.succeeded;
        let ghost stmt_facts = statement.facts;
        let ghost stmt_closure = statement.closure();
        let ghost orig_share_frontier = share.frontier();
        let ghost rev_ghost = revoked;
        let ghost before_merge = *core;
        core.merge_peer_statement(
            Tracked(statement.submitted),
            Tracked(statement.knows),
            Tracked(statement.succeeded),
            Tracked(instance),
        );
        core.record_revoked_map(Tracked(revoked));
        proof {
            lemma_rdma_window_after_fence_ledger(
                self.rdma.in_flight_ops,
                self.rdma.bitmap,
                &core.timelines_spec(),
                self.rdma.timeline_id,
                &before_merge.ledger_spec()@,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(self.rdma.cq),
                core.instance_id(),
                self.rdma.next_control_wr_id,
            );
            lemma_completed_pulls_after_step(
                self.rdma.completed_pulls,
                &before_merge.timelines_spec(),
                &core.timelines_spec(),
                self.rdma.timeline_id,
                self.rdma.cq.observer(),
                &before_merge.ledger_spec()@,
                &core.ledger_spec()@,
            );
            assert(core.frames_spec().same_directory(&before_merge.frames_spec())) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames_spec().same_layout(&before_merge.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            lemma_rdma_all_slabs_same_directory(
                &self.rdma.host_mr,
                &self.rdma.gpu_mr,
                self.rdma.shared_host_mr,
                self.rdma.shared_gpu_mr,
                &before_merge,
                core,
                self.rdma.window_pool.pd(),
            );
            lemma_rdma_leases_same_frames(
                self.rdma.active_leases@,
                &self.rdma.window_pool,
                &self.rdma.qp,
                &before_merge,
                core,
            );
            lemma_outbound_grants_preserved(
                self.pending_outbound_grants@,
                self.rdma.active_leases@,
                self.rdma.active_leases@,
                &before_merge,
                core,
            );
            lemma_statement_valid_in_receiver_ledger(
                core.ledger_spec()@,
                core.node(),
                stmt_sub,
                stmt_succ,
                rev_ghost,
                stmt_facts,
                cert,
            );
            assert(stmt_closure.subset_of(core.citations())) by {
                assert forall|e: TransferEvent| #[trigger]
                    stmt_closure.contains(e) implies core.citations().contains(e) by {
                    assert(stmt_sub.dom().contains(e)) by {
                        if !stmt_facts.contains(e) {
                            let m = choose|m: TransferEvent|
                                stmt_facts.contains(m) && (
                                #[trigger] stmt_sub.map()[m]).preds.contains(e);
                            assert(stmt_sub.dom().contains(m));
                            assert(stmt_sub.map()[m].preds.subset_of(stmt_sub.dom()));
                        }
                    }
                    assert(core.ledger_spec()@.submitted.dom().contains(e));
                    assert(core.ledger_spec()@.knows.set().contains((core.node(), e)));
                }
            }
            assert(transfer::transport::covers(orig_share_frontier, core.citations()));
        }
        if meta.frame_size != core.registered_tier_frame_size(dst_tier) as u64 {
            proof {
                reveal(peer_core_step_except);
                reveal(core_transition);
                reveal(FrameTable::same_layout);
                reveal(FrameTable::preserves_initializing);
                lemma_foreign_shares_identical_leases(old(core), core, self.rdma.active_leases@);
            }
            return Err((CoreError::InvalidArgument, Tracked(share)));
        }
        let ghost before_bw = *core;
        let (handle, dst_tok) = match core.begin_write_version(dst_tier, key, meta.version) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                    lemma_foreign_shares_identical_leases(
                        old(core),
                        core,
                        self.rdma.active_leases@,
                    );
                }
                return Err((e, Tracked(share)));
            },
        };
        proof {
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            lemma_completed_pulls_after_step(
                self.rdma.completed_pulls,
                &before_bw.timelines_spec(),
                &core.timelines_spec(),
                self.rdma.timeline_id,
                self.rdma.cq.observer(),
                &before_bw.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_rdma_all_slabs_same_directory(
                &self.rdma.host_mr,
                &self.rdma.gpu_mr,
                self.rdma.shared_host_mr,
                self.rdma.shared_gpu_mr,
                &before_bw,
                core,
                self.rdma.window_pool.pd(),
            );
            lemma_rdma_leases_after_begin_write(
                self.rdma.active_leases@,
                &self.rdma.window_pool,
                &self.rdma.qp,
                &before_bw,
                core,
            );
            lemma_outbound_grants_preserved(
                self.pending_outbound_grants@,
                self.rdma.active_leases@,
                self.rdma.active_leases@,
                &before_bw,
                core,
            );
            assert(peer_core_step(&before_merge, core, self.rdma.timeline_id)) by {
                reveal(peer_core_step_except);
                reveal(core_transition);
                reveal(FrameTable::preserves_initializing);
            }
        }
        let owns = match target {
            PullTarget::Host => self.rdma.effective_host_mr().unwrap().owns_frame(handle.frame),
            PullTarget::GpuGdr => self.rdma.effective_gpu_mr().unwrap().owns_frame(handle.frame),
        };
        if !owns {
            let ghost before_ab = *core;
            let ghost ab_frame = handle.frame;
            core.abort_write(handle, dst_tok);
            proof {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::same_layout);
                lemma_completed_pulls_after_step(
                    self.rdma.completed_pulls,
                    &before_ab.timelines_spec(),
                    &core.timelines_spec(),
                    self.rdma.timeline_id,
                    self.rdma.cq.observer(),
                    &before_ab.ledger_spec()@,
                    &core.ledger_spec()@,
                );
                lemma_rdma_all_slabs_same_directory(
                    &self.rdma.host_mr,
                    &self.rdma.gpu_mr,
                    self.rdma.shared_host_mr,
                    self.rdma.shared_gpu_mr,
                    &before_ab,
                    core,
                    self.rdma.window_pool.pd(),
                );
                lemma_rdma_leases_after_begin_write(
                    self.rdma.active_leases@,
                    &self.rdma.window_pool,
                    &self.rdma.qp,
                    &before_ab,
                    core,
                );
                lemma_outbound_grants_preserved(
                    self.pending_outbound_grants@,
                    self.rdma.active_leases@,
                    self.rdma.active_leases@,
                    &before_ab,
                    core,
                );
                assert(peer_core_step_except(
                    &before_ab,
                    core,
                    self.rdma.timeline_id,
                    Some(ab_frame),
                )) by {
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::preserves_initializing);
                }
                lemma_peer_core_step_trans_retire_new_init(
                    &before_merge,
                    &before_ab,
                    core,
                    self.rdma.timeline_id,
                    ab_frame,
                );
                lemma_foreign_shares_identical_leases(old(core), core, self.rdma.active_leases@);
            }
            return Err((CoreError::UnsupportedTier, Tracked(share)));
        }
        proof {
            reveal(FrameTable::same_layout);
            reveal(FrameTable::same_directory);
            lemma_foreign_shares_identical_leases(old(core), core, self.rdma.active_leases@);
        }
        Ok(
            PreparedPull {
                handle,
                dst: dst_tok,
                share: Tracked(share),
                remote: cap,
                source: Ghost(cert),
            },
        )
    }

    /// Initiator helper: merge peer statement + revocations, reserve a destination
    /// frame in `dst_tier`, and post the RDMA READ pull (supporting both owned
    /// and shared host/GPU RDMA slabs).
    #[verifier::rlimit(250)]
    pub(crate) fn fetch_post_pull(
        &mut self,
        core: &mut CacheCore,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        meta: ReadGrantMeta,
        cap: &rdma::RemoteCapability,
        Tracked(share): Tracked<ShareCap>,
        Tracked(statement): Tracked<Statement>,
        Ghost(cert): Ghost<ValueCertificate>,
        Tracked(revoked): Tracked<RevokedMap>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<InFlightRdmaPull, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            meta.frame_size == old(core).frames_spec().tier_frame_size(dst_tier) as u64,
            instance.id() == old(core).instance_id(),
            statement.valid(old(core).instance_id(), old(core).node()),
            statement.knows.set() == knowledge_of(old(core).node(), statement.closure()),
            crate::provenance::complete_receipts_map(statement.submitted),
            cap.well_formed(),
            cap.slot_ref().loc.node == old(self).rdma.peer_node,
            cap.access() == rdma::RemoteAccess::ReadOnly,
            cap.share() == Some(share.share()),
            cap.slot_ref() == cert.slot,
            cert.slot.is_full(),
            cert.bytes.len() == meta.frame_size,
            share.is_fenced(),
            share.instance_id() == old(core).instance_id(),
            share.loc() == cap.slot_ref().loc,
            share.frontier().last_writer == Some(cert.writer),
            transfer::transport::covers(share.frontier(), statement.closure()),
            revoked.instance_id() == old(core).instance_id(),
            super::protocol::statement_replica_valid(
                statement.submitted,
                statement.succeeded,
                revoked,
                statement.facts,
                cert,
            ),
            match target {
                PullTarget::Host => old(self).rdma.effective_host_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma.effective_gpu_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            r matches Ok(pull) ==> {
                &&& pull.valid(&final(self).rdma, final(core))
                &&& pull.handle.frame.0 < old(core).frames_spec().metas_spec().len()
                &&& old(core).frames_spec().meta(pull.handle.frame).state != STATE_INITIALIZING
                &&& pull.handle.version == meta.version
                &&& pull.copy@.source == cert
                &&& pull.share_cap@.loc() == cap.slot_ref().loc
                &&& pull.share_cap@.share() == share.share()
                &&& pull.share_cap@.frontier() == (foundation::tracked::Frontier {
                    last_writer: share.frontier().last_writer,
                    readers: share.frontier().readers.insert(pull.writer@),
                })
                &&& transfer::transport::covers(share.frontier(), final(core).citations())
                &&& final(self).rdma.cq.in_flight().contains_key(pull.wr_id)
                &&& final(self).rdma.cq.in_flight()[pull.wr_id] == (rdma::PendingWorkRequest {
                    kind: rdma::PendingKind::DataTransfer { event: pull.writer@ },
                })
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_merge = *core;
        let PreparedPull { handle, dst: dst_tok, share, remote, source } =
            match self.fetch_prepare_pull(
            core,
            key,
            dst_tier,
            target,
            meta,
            cap.descriptor_copy(),
            Tracked(share),
            Tracked(statement),
            Ghost(cert),
            Tracked(revoked),
            Tracked(instance),
        ) {
            Ok(p) => p,
            Err((e, _)) => return Err(e),
        };
        let ghost before_post = *core;
        match post_pull_read(
            &mut self.rdma,
            core,
            Tracked(instance),
            target,
            handle,
            dst_tok,
            &remote,
            share,
            source,
        ) {
            Ok(p) => {
                proof {
                    lemma_peer_core_step_trans(
                        &before_merge,
                        &before_post,
                        core,
                        self.rdma.timeline_id,
                    );
                    reveal(FrameTable::same_layout);
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                        &before_post,
                        core,
                    );
                    lemma_foreign_shares_identical_leases(
                        &before_post,
                        core,
                        self.rdma.active_leases@,
                    );
                    lemma_foreign_shares_trans(
                        old(core),
                        &before_post,
                        core,
                        old(self).rdma.active_leases@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                    );
                }
                Ok(p)
            },
            Err((e, h, tok, _sh)) => {
                let ghost before_ab = *core;
                let ghost ab_frame = h.frame;
                core.abort_write(h, tok);
                proof {
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::same_layout);
                    lemma_completed_pulls_after_step(
                        self.rdma.completed_pulls,
                        &before_ab.timelines_spec(),
                        &core.timelines_spec(),
                        self.rdma.timeline_id,
                        self.rdma.cq.observer(),
                        &before_ab.ledger_spec()@,
                        &core.ledger_spec()@,
                    );
                    lemma_rdma_all_slabs_same_directory(
                        &self.rdma.host_mr,
                        &self.rdma.gpu_mr,
                        self.rdma.shared_host_mr,
                        self.rdma.shared_gpu_mr,
                        &before_ab,
                        core,
                        self.rdma.window_pool.pd(),
                    );
                    lemma_rdma_leases_after_begin_write(
                        self.rdma.active_leases@,
                        &self.rdma.window_pool,
                        &self.rdma.qp,
                        &before_ab,
                        core,
                    );
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                        &before_ab,
                        core,
                    );
                    assert(peer_core_step_except(
                        &before_ab,
                        core,
                        self.rdma.timeline_id,
                        Some(ab_frame),
                    )) by {
                        reveal(peer_core_step_except);
                        reveal(core_transition);
                        reveal(FrameTable::preserves_initializing);
                    }
                    lemma_peer_core_step_trans_retire_new_init(
                        &before_merge,
                        &before_ab,
                        core,
                        self.rdma.timeline_id,
                        ab_frame,
                    );
                    lemma_foreign_shares_identical_leases(
                        &before_post,
                        core,
                        self.rdma.active_leases@,
                    );
                    lemma_foreign_shares_trans(
                        old(core),
                        &before_post,
                        core,
                        old(self).rdma.active_leases@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                    );
                }
                Err(e)
            },
        }
    }

    fn cancel_batch_pull(
        &mut self,
        core: &mut CacheCore,
        target: PullTarget,
        plan: PreparedPull,
        route: PullGrantReturn,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            plan.valid(&old(self).rdma, old(core), target),
            route.owns_grant(old(self).channel.received(), plan.share@),
            transfer::transport::covers(plan.share@.frontier(), old(core).citations()),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(self).rdma == old(self).rdma,
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).timelines_spec() == old(core).timelines_spec(),
            final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
            peer_core_step_except(
                old(core),
                final(core),
                old(self).rdma.timeline_id,
                Some(plan.handle.frame),
            ),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            forall|k: GrantTag|
                k != route.tag && #[trigger] old(self).channel.received().contains_key(k)
                    ==> final(self).channel.received().contains_key(k)
                    && final(self).channel.received()[k] == old(self).channel.received()[k],
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before = *core;
        let ghost rdma = self.rdma;
        core.abort_write(plan.handle, plan.dst);
        proof {
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
            reveal(peer_core_step_except);
            reveal(core_transition);
            lemma_core_wf_facts(core);
            lemma_rdma_all_slabs_same_directory(
                &self.rdma.host_mr,
                &self.rdma.gpu_mr,
                self.rdma.shared_host_mr,
                self.rdma.shared_gpu_mr,
                &before,
                core,
                self.rdma.window_pool.pd(),
            );
            lemma_rdma_leases_after_begin_write(
                self.rdma.active_leases@,
                &self.rdma.window_pool,
                &self.rdma.qp,
                &before,
                core,
            );
            lemma_completed_pulls_after_step(
                self.rdma.completed_pulls,
                &before.timelines_spec(),
                &core.timelines_spec(),
                self.rdma.timeline_id,
                self.rdma.cq.observer(),
                &before.ledger_spec()@,
                &core.ledger_spec()@,
            );
            lemma_outbound_grants_preserved(
                self.pending_outbound_grants@,
                rdma.active_leases@,
                self.rdma.active_leases@,
                &before,
                core,
            );
            lemma_foreign_shares_identical_leases(&before, core, self.rdma.active_leases@);
        }
        self.fetch_release_grant(core, route.tag, route.request_id, plan.share, Tracked(instance));
    }

    /// Reap the batch's data statuses while leaving its destination tokens sealed.
    fn await_batch_completions(
        &mut self,
        core: &mut CacheCore,
        pulls: &[InFlightRdmaPull],
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: (Vec<Result<(), i32>>, Option<CoreError>))
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            forall|i: int|
                0 <= i < pulls.len() ==> {
                    let p = #[trigger] pulls@[i];
                    &&& p.valid(&old(self).rdma, old(core))
                    &&& pull_tracked_or_done(&old(self).rdma, p.wr_id, p.writer@)
                },
            forall|a: int, b: int|
                0 <= a < b < pulls.len() ==> (#[trigger] pulls@[a]).wr_id != (
                #[trigger] pulls@[b]).wr_id,
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).channel == old(self).channel,
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            old(core).citations().subset_of(final(core).citations()),
            r.0.len() <= pulls.len(),
            r.1.is_none() ==> r.0.len() == pulls.len(),
            forall|i: int|
                0 <= i < pulls.len() ==> {
                    let p = #[trigger] pulls@[i];
                    &&& p.valid(&final(self).rdma, final(core))
                    &&& i < r.0.len() ==> observed_batch_status(&p, final(core), r.0[i])
                    &&& i >= r.0.len() ==> pull_tracked_or_done(
                        &final(self).rdma,
                        p.wr_id,
                        p.writer@,
                    )
                },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut statuses: Vec<Result<(), i32>> = Vec::with_capacity(pulls.len());
        let mut spins = 0u32;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while statuses.len() < pulls.len() && spins < 1_000_000
            invariant
                statuses.len() <= pulls.len(),
                spins <= 1_000_000,
                self.well_formed(core),
                core.well_formed(),
                core.all_versions_bound(),
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.channel == old(self).channel,
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                instance.id() == core.instance_id(),
                core.instance_id() == old(core).instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                peer_core_step(old(core), core, self.rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                old(core).citations().subset_of(core.citations()),
                forall|j: int|
                    0 <= j < pulls.len() ==> {
                        let p = #[trigger] pulls@[j];
                        &&& p.valid(&self.rdma, core)
                        &&& j >= statuses.len() ==> pull_tracked_or_done(
                            &self.rdma,
                            p.wr_id,
                            p.writer@,
                        )
                        &&& j < statuses.len() ==> observed_batch_status(&p, core, statuses[j])
                    },
                forall|a: int, b: int|
                    0 <= a < b < pulls.len() ==> (#[trigger] pulls@[a]).wr_id != (
                    #[trigger] pulls@[b]).wr_id,
            decreases 1_000_000 - spins, pulls.len() - statuses.len(),
        {
            let i = statuses.len();
            let p = &pulls[i];
            let ghost pre = *self;
            if let Some(status) = self.rdma.take_completed_pull(core, p.wr_id) {
                proof {
                    Self::lemma_session_wf_same_leases(&pre, self, core);
                    lemma_core_wf_facts(core);
                    lemma_core_timelines_wf(core);
                    assert(observed_batch_status(p, core, status));
                    assert forall|j: int| i < j < pulls.len() implies pull_tracked_or_done(
                        &self.rdma,
                        (#[trigger] pulls@[j]).wr_id,
                        pulls@[j].writer@,
                    ) by {
                        let other = pulls@[j];
                        let seq = other.wr_id;
                        assert(pull_tracked_or_done(&pre.rdma, seq, other.writer@));
                        assert(seq != p.wr_id);
                        assert(self.rdma.pending() == pre.rdma.pending());
                        assert(self.rdma.completed_pull_spec(seq) == pre.rdma.completed_pull_spec(
                            seq,
                        ));
                        reveal(pull_tracked_or_done_state);
                        if !pre.rdma.pending().contains_key(seq) {
                            assert(pre.rdma.completed_pull_spec(seq).is_some());
                            assert(self.rdma.completed_pull_spec(seq).is_some());
                        }
                    }
                }
                statuses.push(status);
            } else {
                let ghost before = *core;
                let ghost rdma = self.rdma;
                proof {
                    Self::lemma_session_wf_same_leases(&pre, self, core);
                    assert forall|j: int| i <= j < pulls.len() implies pull_tracked_or_done(
                        &self.rdma,
                        (#[trigger] pulls@[j]).wr_id,
                        pulls@[j].writer@,
                    ) by {
                        let other = pulls@[j];
                        let seq = other.wr_id;
                        assert(pull_tracked_or_done(&pre.rdma, seq, other.writer@));
                        assert(self.rdma.pending() == pre.rdma.pending());
                        reveal(pull_tracked_or_done_state);
                        if j == i {
                            assert(pre.rdma.completed_pull_spec(p.wr_id).is_none());
                            assert(pre.rdma.pending().contains_key(seq));
                        } else if !pre.rdma.pending().contains_key(seq) {
                            assert(seq != p.wr_id);
                            assert(pre.rdma.completed_pull_spec(seq).is_some());
                            assert(self.rdma.completed_pull_spec(seq)
                                == pre.rdma.completed_pull_spec(seq));
                        }
                    }
                }
                let progress = self.pump_session_cq(core, Tracked(instance));
                proof {
                    lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &before,
                        core,
                        old(self).rdma.active_leases@,
                        rdma.active_leases@,
                        self.rdma.active_leases@,
                    );
                    lemma_batch_status_growth(&before, core, self.rdma.timeline_id, None);
                    assert forall|j: int| 0 <= j < pulls.len() implies {
                        let p = #[trigger] pulls@[j];
                        &&& p.valid(&self.rdma, core)
                        &&& j >= i ==> pull_tracked_or_done(&self.rdma, p.wr_id, p.writer@)
                        &&& j < i ==> observed_batch_status(&p, core, statuses[j])
                    } by {
                        lemma_pull_valid_after_peer_step(
                            &rdma,
                            &self.rdma,
                            &before,
                            core,
                            &pulls@[j],
                            None,
                        );
                    }
                }
                if let Err(e) = progress {
                    return (statuses, Some(e));
                }
                spins += 1;
            }
        }
        if statuses.len() == pulls.len() {
            (statuses, None)
        } else {
            (statuses, Some(rdma_error(rdma::RdmaError(11))))
        }
    }

    /// Reserve destinations for received grants and collect errors for failed reservations.
    fn prepare_batch_pulls(
        &mut self,
        core: &mut CacheCore,
        dst_tier: TierId,
        target: PullTarget,
        grants: Vec<PendingInboundGrant>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (out: (
        Vec<PreparedPull>,
        Vec<PullGrantReturn>,
        Vec<(u64, Result<FetchOutcome, CoreError>)>,
    ))
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            old(self).pull_tier_valid(dst_tier, target),
            forall|i: int|
                0 <= i < grants.len() ==> (#[trigger] grants[i]).valid(
                    old(core),
                    &old(self).channel,
                    old(self).rdma.peer_node,
                ),
            forall|i: int, j: int|
                0 <= i < j < grants.len() ==> (#[trigger] grants[i]).tag != (
                #[trigger] grants[j]).tag,
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(self).rdma == old(self).rdma,
            final(core).instance_id() == old(core).instance_id(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            out.0.len() == out.1.len(),
            out.0.len() + out.2.len() == grants.len(),
            forall|j: int|
                0 <= j < out.0.len() ==> {
                    let p = #[trigger] out.0[j];
                    &&& p.valid(&final(self).rdma, final(core), target)
                    &&& out.1[j].matches_prepared(&p, final(self).channel.received(), final(core))
                    &&& p.handle.frame.0 < old(core).frames_spec().metas_spec().len()
                    &&& old(core).frames_spec().meta(p.handle.frame).state != STATE_INITIALIZING
                },
            forall|a: int, b: int|
                0 <= a < b < out.0.len() ==> {
                    &&& (#[trigger] out.0[a]).handle.frame != (#[trigger] out.0[b]).handle.frame
                    &&& out.1[a].tag != out.1[b].tag
                },
    {
        hide(CacheCore::well_formed);
        hide(CacheCore::all_versions_bound);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(rdma_lease_index_spec);
        hide(replica_valid);
        hide(complete_receipts);
        hide(super::protocol::statement_replica_valid);
        let count = grants.len();
        let ghost received = grants@;
        let mut incoming = grants.into_iter();
        let mut plans: Vec<PreparedPull> = Vec::with_capacity(count);
        let mut routes: Vec<PullGrantReturn> = Vec::with_capacity(count);
        let mut results = Vec::with_capacity(count);
        let mut i = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while i < count
            invariant
                i <= count == received.len(),
                plans.len() == routes.len(),
                plans.len() + results.len() == i,
                vstd::std_specs::iter::IteratorSpec::remaining(&incoming) == received.skip(
                    i as int,
                ),
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma == old(self).rdma,
                instance.id() == core.instance_id(),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                self.rdma.qp.source().well_formed(),
                self.rdma.qp.effective_max_rd_atomic() > 0,
                core.node() == old(core).node(),
                core.frames_spec().metas_spec().len() == old(core).frames_spec().metas_spec().len(),
                self.pull_tier_valid(dst_tier, target),
                core.frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
                self.rdma.peer_node != core.node(),
                core.frames_spec().tier_frame_size(dst_tier) == old(
                    core,
                ).frames_spec().tier_frame_size(dst_tier),
                peer_core_step(old(core), core, self.rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                forall|j: int|
                    i <= j < count ==> (#[trigger] received[j]).valid(
                        core,
                        &self.channel,
                        self.rdma.peer_node,
                    ),
                forall|j: int|
                    0 <= j < plans.len() ==> {
                        let p = #[trigger] plans[j];
                        let route = routes[j];
                        &&& p.valid(&self.rdma, core, target)
                        &&& route.matches_prepared(&p, self.channel.received(), core)
                        &&& p.handle.frame.0 < old(core).frames_spec().metas_spec().len()
                        &&& old(core).frames_spec().meta(p.handle.frame).state != STATE_INITIALIZING
                        &&& exists|k: int| 0 <= k < i && (#[trigger] received[k]).tag == route.tag
                    },
                forall|a: int, b: int|
                    0 <= a < b < plans.len() ==> {
                        &&& (#[trigger] plans[a]).handle.frame != (#[trigger] plans[b]).handle.frame
                        &&& routes[a].tag != routes[b].tag
                    },
                forall|a: int, b: int|
                    0 <= a < b < count ==> (#[trigger] received[a]).tag != (
                    #[trigger] received[b]).tag,
            decreases count - i,
        {
            let g = incoming.next().unwrap();
            let ghost before = *core;
            let ghost pre = *self;
            let ghost frontier = g.share@.frontier();
            let result = self.fetch_prepare_pull(
                core,
                g.meta.key,
                dst_tier,
                target,
                g.meta,
                g.cap,
                g.share,
                g.statement,
                g.cert,
                g.revoked,
                Tracked(instance),
            );
            proof {
                reveal(FrameTable::same_layout);
                reveal(FrameTable::preserves_initializing);
                reveal(peer_core_step_except);
                reveal(core_transition);
                lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
                assert forall|j: int| 0 <= j < plans.len() implies {
                    let p = #[trigger] plans[j];
                    &&& p.valid(&self.rdma, core, target)
                    &&& routes[j].matches_prepared(&p, self.channel.received(), core)
                    &&& (result matches Ok(new) ==> new.handle.frame != p.handle.frame)
                    &&& g.tag != routes[j].tag
                } by {
                    let p = plans[j];
                    p.lemma_after_step(&before, core, &pre.rdma, &self.rdma, target, None);
                    let k = choose|k: int|
                        0 <= k < i && (#[trigger] received[k]).tag == routes[j].tag;
                    assert(g == received[i as int]);
                }
            }
            match result {
                Ok(plan) => {
                    let tracked binding = g.version_binding.get().tracked_unwrap();
                    let route = PullGrantReturn {
                        tag: g.tag,
                        request_id: g.meta.request_id,
                        binding: Tracked(binding),
                        frontier: Ghost(frontier),
                    };
                    proof {
                        if old(core).frames_spec().meta(plan.handle.frame).state
                            == STATE_INITIALIZING {
                            assert(before.frames_spec().meta(plan.handle.frame) == old(
                                core,
                            ).frames_spec().meta(plan.handle.frame));
                            assert(false);
                        }
                    }
                    plans.push(plan);
                    routes.push(route);
                },
                Err((err, share)) => {
                    self.fetch_release_grant(
                        core,
                        g.tag,
                        g.meta.request_id,
                        share,
                        Tracked(instance),
                    );
                    results.push((g.meta.request_id, Err(err)));
                },
            }
            proof {
                lemma_foreign_shares_trans(
                    old(core),
                    &before,
                    core,
                    old(self).rdma.active_leases@,
                    pre.rdma.active_leases@,
                    self.rdma.active_leases@,
                );
                assert forall|j: int| i < j < count implies (#[trigger] received[j]).valid(
                    core,
                    &self.channel,
                    self.rdma.peer_node,
                ) by {
                    assert(received[j].tag != g.tag);
                }
            }
            i += 1;
        }
        proof {
            assert(core.node_id_spec() == old(core).node_id_spec()) by {
                reveal(peer_core_step_except);
                reveal(core_transition);
            }
        }
        (plans, routes, results)
    }

    /// Submit the prepared reads and preserve the channel's buffered grants.
    #[verifier::rlimit(30)]
    fn post_batch_pulls(
        &mut self,
        core: &mut CacheCore,
        target: PullTarget,
        plans: Vec<PreparedPull>,
        routes: &[PullGrantReturn],
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (out: PullBatchPost)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            plans.len() == routes.len(),
            forall|j: int|
                0 <= j < plans.len() ==> {
                    let p = #[trigger] plans[j];
                    &&& p.valid(&old(self).rdma, old(core), target)
                    &&& routes@[j].matches_prepared(&p, old(self).channel.received(), old(core))
                },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).instance_id() == old(core).instance_id(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(self).channel == old(self).channel,
            final(self).rdma.active_leases == old(self).rdma.active_leases,
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            rdma::read_batch_qp_frame(&final(self).rdma.qp, &old(self).rdma.qp),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            out.posted.len() + out.unposted.len() == plans.len(),
            out.unposted@ == plans@.skip(out.posted.len() as int),
            forall|j: int|
                0 <= j < out.posted.len() ==> {
                    let p = #[trigger] out.posted[j];
                    &&& p.valid(&final(self).rdma, final(core)) && p.target == target
                    &&& !p.visibility_flushed && p.handle == plans[j].handle
                    &&& p.wr_id == old(core).timelines_spec().next_seq_spec()[old(
                        self,
                    ).rdma.timeline_id as int] + j
                    &&& routes@[j].matches_posted(&p, final(self).channel.received(), final(core))
                    &&& pull_tracked_or_done(&final(self).rdma, p.wr_id, p.writer@)
                },
            forall|j: int|
                0 <= j < out.unposted.len() ==> {
                    let p = #[trigger] out.unposted[j];
                    &&& p.valid(&final(self).rdma, final(core), target)
                    &&& routes@[out.posted.len() + j].matches_prepared(
                        &p,
                        final(self).channel.received(),
                        final(core),
                    )
                },
    {
        hide(CacheCore::well_formed);
        hide(CacheCore::all_versions_bound);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(InFlightRdmaPull::valid);
        hide(PreparedPull::valid);
        hide(PullGrantReturn::matches_posted);
        hide(PullGrantReturn::matches_prepared);
        hide(rdma_lease_index_spec);
        hide(replica_valid);
        hide(complete_receipts);
        hide(super::protocol::statement_replica_valid);
        let ghost plan_view = plans@;
        let ghost before = *core;
        let ghost prior = *self;
        proof {
            assert(self.rdma.well_formed(core)) by {
                reveal(PeerSession::well_formed);
            }
        }
        let post = post_pull_read_batch(&mut self.rdma, core, Tracked(instance), target, plans);
        let n = post.posted.len();
        proof {
            assert(self.well_formed(core)) by {
                reveal(PeerSession::well_formed);
                reveal(FrameTable::same_layout);
                assert(rdma_leases_disjoint(self.rdma.active_leases@)) by {
                    reveal(RdmaBackend::well_formed);
                }
                lemma_outbound_grants_preserved(
                    self.pending_outbound_grants@,
                    prior.rdma.active_leases@,
                    self.rdma.active_leases@,
                    &before,
                    core,
                );
            }
            lemma_foreign_shares_identical_leases(&before, core, self.rdma.active_leases@);
            assert forall|j: int| 0 <= j < n implies {
                let p = #[trigger] post.posted[j];
                &&& p.valid(&self.rdma, core) && p.target == target && !p.visibility_flushed
                &&& p.handle == plan_view[j].handle
                &&& p.wr_id
                    == before.timelines_spec().next_seq_spec()[prior.rdma.timeline_id as int] + j
                &&& routes@[j].matches_posted(&p, self.channel.received(), core)
                &&& pull_tracked_or_done(&self.rdma, p.wr_id, p.writer@)
            } by {
                lemma_posted_pull_batch_entry(
                    post.posted@,
                    plan_view,
                    &self.rdma,
                    core,
                    target,
                    before.timelines_spec().next_seq_spec()[prior.rdma.timeline_id as int],
                    j,
                );
                assert(routes@[j].matches_posted(&post.posted[j], self.channel.received(), core))
                    by {
                    reveal(PullGrantReturn::matches_posted);
                    reveal(PullGrantReturn::matches_prepared);
                    assert(routes@[j].matches_prepared(
                        &plan_view[j],
                        prior.channel.received(),
                        &before,
                    ));
                    assert(post.posted[j].copy@.source == plan_view[j].source@);
                }
                lemma_rdma_wf_implies_pull_tracked(
                    &self.rdma,
                    core,
                    post.posted[j].wr_id,
                    post.posted[j].writer@,
                );
            }
            assert forall|j: int| 0 <= j < post.unposted.len() implies {
                let p = #[trigger] post.unposted[j];
                &&& p.valid(&self.rdma, core, target)
                &&& routes@[n + j].matches_prepared(&p, self.channel.received(), core)
            } by {
                assert(post.unposted[j] == plan_view[(n + j) as int]);
                assert(routes@[n + j].matches_prepared(
                    &post.unposted[j],
                    self.channel.received(),
                    core,
                )) by {
                    reveal(PullGrantReturn::matches_prepared);
                    assert(routes@[n + j].matches_prepared(
                        &plan_view[(n + j) as int],
                        prior.channel.received(),
                        &before,
                    ));
                }
                post.unposted[j].lemma_after_step(
                    &before,
                    core,
                    &prior.rdma,
                    &self.rdma,
                    target,
                    None,
                );
            }
        }
        post
    }

    /// Return grants for unposted reads and append their posting errors.
    fn cancel_unposted_pulls(
        &mut self,
        core: &mut CacheCore,
        target: PullTarget,
        mut unposted: Vec<PreparedPull>,
        mut routes: Vec<PullGrantReturn>,
        mut results: Vec<(u64, Result<FetchOutcome, CoreError>)>,
        post_error: CoreError,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(batch_core): Ghost<CacheCore>,
        Ghost(batch_rdma): Ghost<RdmaBackend<'a>>,
        Ghost(plan_view): Ghost<Seq<PreparedPull>>,
        Ghost(posted): Ghost<Seq<InFlightRdmaPull>>,
    ) -> (out: (Vec<PullGrantReturn>, Vec<(u64, Result<FetchOutcome, CoreError>)>))
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).rdma.timeline_id == batch_rdma.timeline_id,
            peer_core_step(&batch_core, old(core), batch_rdma.timeline_id),
            peer_foreign_shares_preserved(
                &batch_core,
                old(core),
                batch_rdma.active_leases@,
                old(self).rdma.active_leases@,
            ),
            posted.len() + unposted.len() == routes.len() == plan_view.len(),
            unposted@ == plan_view.skip(posted.len() as int),
            forall|j: int|
                0 <= j < routes.len() ==> {
                    let frame = (#[trigger] plan_view[j]).handle.frame;
                    &&& frame.0 < batch_core.frames_spec().metas_spec().len()
                    &&& batch_core.frames_spec().meta(frame).state != STATE_INITIALIZING
                },
            forall|a: int, b: int|
                0 <= a < b < routes.len() ==> (#[trigger] routes[a]).tag != (
                #[trigger] routes[b]).tag && (#[trigger] plan_view[a]).handle.frame != (
                #[trigger] plan_view[b]).handle.frame,
            forall|j: int|
                0 <= j < posted.len() ==> {
                    let p = #[trigger] posted[j];
                    &&& p.valid(&old(self).rdma, old(core)) && p.target == target
                    &&& !p.visibility_flushed && p.handle == plan_view[j].handle
                    &&& routes[j].matches_posted(&p, old(self).channel.received(), old(core))
                    &&& pull_tracked_or_done(&old(self).rdma, p.wr_id, p.writer@)
                },
            forall|j: int|
                0 <= j < unposted.len() ==> {
                    let p = #[trigger] unposted[j];
                    &&& p.valid(&old(self).rdma, old(core), target)
                    &&& routes[posted.len() + j].matches_prepared(
                        &p,
                        old(self).channel.received(),
                        old(core),
                    )
                },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(self).rdma == old(self).rdma,
            final(core).node_id_spec() == batch_core.node_id_spec(),
            final(core).instance_id() == batch_core.instance_id(),
            peer_core_step(&batch_core, final(core), batch_rdma.timeline_id),
            peer_foreign_shares_preserved(
                &batch_core,
                final(core),
                batch_rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            out.0.len() == posted.len(),
            out.0@ == routes@.take(posted.len() as int),
            out.1.len() + posted.len() == results.len() + routes.len(),
            forall|j: int|
                0 <= j < posted.len() ==> {
                    let p = #[trigger] posted[j];
                    &&& p.valid(&final(self).rdma, final(core)) && p.target == target
                    &&& !p.visibility_flushed && p.handle == plan_view[j].handle
                    &&& out.0[j].matches_posted(&p, final(self).channel.received(), final(core))
                    &&& pull_tracked_or_done(&final(self).rdma, p.wr_id, p.writer@)
                },
    {
        hide(CacheCore::well_formed);
        hide(CacheCore::all_versions_bound);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(InFlightRdmaPull::valid);
        hide(rdma_lease_index_spec);
        hide(replica_valid);
        hide(complete_receipts);
        hide(super::protocol::statement_replica_valid);
        let ghost n = posted.len();
        let ghost route_view = routes@;
        let ghost count = results@.len() + routes@.len();
        assert(unposted@ =~= plan_view.subrange(n as int, routes.len() as int));
        let ghost after_post_rdma = self.rdma;
        while !unposted.is_empty()
            invariant
                n + unposted.len() == routes.len() <= route_view.len() == plan_view.len(),
                n == posted.len(),
                results.len() + routes.len() == count,
                routes@ == route_view.take(routes.len() as int),
                unposted@ == plan_view.subrange(n as int, routes.len() as int),
                self.rdma == after_post_rdma,
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                peer_core_step(&batch_core, core, self.rdma.timeline_id),
                peer_foreign_shares_preserved(
                    &batch_core,
                    core,
                    batch_rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                forall|j: int|
                    0 <= j < routes.len() ==> {
                        &&& batch_core.frames_spec().meta(
                            (#[trigger] plan_view[j]).handle.frame,
                        ).state != STATE_INITIALIZING
                        &&& plan_view[j].handle.frame.0
                            < batch_core.frames_spec().metas_spec().len()
                    },
                forall|a: int, b: int|
                    0 <= a < b < route_view.len() ==> (#[trigger] route_view[a]).tag != (
                    #[trigger] route_view[b]).tag && (#[trigger] plan_view[a]).handle.frame != (
                    #[trigger] plan_view[b]).handle.frame,
                forall|j: int|
                    0 <= j < n ==> {
                        let p = #[trigger] posted[j];
                        &&& p.valid(&self.rdma, core) && p.target == target && !p.visibility_flushed
                        &&& p.handle == plan_view[j].handle
                        &&& routes[j].matches_posted(&p, self.channel.received(), core)
                        &&& pull_tracked_or_done(&self.rdma, p.wr_id, p.writer@)
                    },
                forall|j: int|
                    0 <= j < unposted.len() ==> {
                        let p = #[trigger] unposted[j];
                        &&& p.valid(&self.rdma, core, target)
                        &&& routes[n + j].matches_prepared(&p, self.channel.received(), core)
                    },
            decreases unposted.len(),
        {
            let ghost cut = routes.len() - 1;
            let plan = unposted.pop().unwrap();
            let route = routes.pop().unwrap();
            proof {
                assert(plan == plan_view[cut]);
                assert(route == route_view[cut]);
                assert(routes@ =~= route_view.take(cut));
                assert(unposted@ =~= plan_view.subrange(n as int, cut));
            }
            let request_id = route.request_id;
            let ghost frame = plan.handle.frame;
            let ghost tag = route.tag;
            let ghost before_cancel = *core;
            self.cancel_batch_pull(core, target, plan, route, Tracked(instance));
            proof {
                lemma_peer_core_step_trans_retire_new_init(
                    &batch_core,
                    &before_cancel,
                    core,
                    self.rdma.timeline_id,
                    frame,
                );
                lemma_foreign_shares_trans(
                    &batch_core,
                    &before_cancel,
                    core,
                    batch_rdma.active_leases@,
                    self.rdma.active_leases@,
                    self.rdma.active_leases@,
                );
                assert forall|j: int| 0 <= j < n implies {
                    let p = #[trigger] posted[j];
                    &&& p.valid(&self.rdma, core)
                    &&& routes[j].matches_posted(&p, self.channel.received(), core)
                } by {
                    assert(routes[j] == route_view[j]);
                    assert(routes[j].tag != tag);
                    lemma_pull_valid_after_peer_step(
                        &self.rdma,
                        &self.rdma,
                        &before_cancel,
                        core,
                        &posted[j],
                        Some(frame),
                    );
                }
                assert forall|j: int| 0 <= j < unposted.len() implies {
                    let p = #[trigger] unposted[j];
                    &&& p.valid(&self.rdma, core, target)
                    &&& routes[n + j].matches_prepared(&p, self.channel.received(), core)
                } by {
                    assert(routes[n + j] == route_view[n + j]);
                    assert(routes[n + j].tag != tag);
                    unposted[j].lemma_after_step(
                        &before_cancel,
                        core,
                        &self.rdma,
                        &self.rdma,
                        target,
                        Some(frame),
                    );
                }
            }
            results.push((request_id, Err(post_error)));
        }
        proof {
            assert(core.node_id_spec() == batch_core.node_id_spec() && core.instance_id()
                == batch_core.instance_id()) by {
                reveal(peer_core_step_except);
                reveal(core_transition);
            }
        }
        (routes, results)
    }

    /// Pull a batch of received grants. Entries can arrive in any order.
    #[verifier::rlimit(160)]
    pub(crate) fn fetch_batch_from_grants(
        &mut self,
        core: &mut CacheCore,
        dst_tier: TierId,
        target: PullTarget,
        grants: Vec<PendingInboundGrant>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: FetchBatchPulls)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            old(self).pull_tier_valid(dst_tier, target),
            forall|i: int|
                0 <= i < grants.len() ==> (#[trigger] grants[i]).valid(
                    old(core),
                    &old(self).channel,
                    old(self).rdma.peer_node,
                ),
            forall|i: int, j: int|
                0 <= i < j < grants.len() ==> (#[trigger] grants[i]).tag != (
                #[trigger] grants[j]).tag,
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(self).well_formed(final(core)),
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            r.pending.len() == 0 || r.wait_error.is_some(),
            r.results.len() + r.pending.len() == grants.len(),
    {
        hide(CacheCore::well_formed);
        hide(CacheCore::all_versions_bound);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(InFlightRdmaPull::valid);
        hide(rdma_lease_index_spec);
        hide(replica_valid);
        hide(complete_receipts);
        hide(super::protocol::statement_replica_valid);
        let (plans, routes, results) = self.prepare_batch_pulls(
            core,
            dst_tier,
            target,
            grants,
            Tracked(instance),
        );
        let ghost plan_view = plans@;
        let ghost route_view = routes@;
        let ghost before_post = *core;
        let ghost before_leases = self.rdma.active_leases@;
        let post = self.post_batch_pulls(core, target, plans, routes.as_slice(), Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before_post, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_post,
                core,
                old(self).rdma.active_leases@,
                before_leases,
                self.rdma.active_leases@,
            );
        }
        let posted = post.posted;
        let (routes, results) = self.cancel_unposted_pulls(
            core,
            target,
            post.unposted,
            routes,
            results,
            post.error.unwrap_or(CoreError::CapacityExhausted),
            Tracked(instance),
            Ghost(*old(core)),
            Ghost(old(self).rdma),
            Ghost(plan_view),
            Ghost(posted@),
        );
        proof {
            assert forall|a: int, b: int| 0 <= a < b < posted.len() implies {
                &&& (#[trigger] posted[a]).wr_id != (#[trigger] posted[b]).wr_id
                &&& posted[a].handle.frame != posted[b].handle.frame
                &&& routes[a].tag != routes[b].tag
            } by {
                assert(posted[a].wr_id == before_post.timelines_spec().next_seq_spec()[old(
                    self,
                ).rdma.timeline_id as int] + a);
                assert(posted[b].wr_id == before_post.timelines_spec().next_seq_spec()[old(
                    self,
                ).rdma.timeline_id as int] + b);
                assert(routes[a] == route_view[a]);
                assert(routes[b] == route_view[b]);
            }
        }
        self.finish_posted_pulls(
            core,
            target,
            posted,
            routes,
            results,
            Tracked(instance),
            Ghost(*old(core)),
            Ghost(old(self).rdma),
            Ghost(plan_view),
        )
    }

    /// Await the posted reads, then publish completed destinations or return sealed pulls.
    #[verifier::rlimit(30)]
    fn finish_posted_pulls(
        &mut self,
        core: &mut CacheCore,
        target: PullTarget,
        posted: Vec<InFlightRdmaPull>,
        routes: Vec<PullGrantReturn>,
        mut results: Vec<(u64, Result<FetchOutcome, CoreError>)>,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(batch_core): Ghost<CacheCore>,
        Ghost(batch_rdma): Ghost<RdmaBackend<'a>>,
        Ghost(plan_view): Ghost<Seq<PreparedPull>>,
    ) -> (r: FetchBatchPulls)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(core).instance_id() == batch_core.instance_id(),
            old(core).node_id_spec() == batch_core.node_id_spec(),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            rdma::read_batch_qp_frame(&old(self).rdma.qp, &batch_rdma.qp),
            old(self).rdma.timeline_id == batch_rdma.timeline_id,
            old(self).rdma.peer_node == batch_rdma.peer_node,
            peer_core_step(&batch_core, old(core), batch_rdma.timeline_id),
            peer_foreign_shares_preserved(
                &batch_core,
                old(core),
                batch_rdma.active_leases@,
                old(self).rdma.active_leases@,
            ),
            posted.len() == routes.len() <= plan_view.len(),
            forall|j: int|
                0 <= j < posted.len() ==> {
                    let p = #[trigger] posted[j];
                    &&& p.valid(&old(self).rdma, old(core))
                    &&& p.target == target && !p.visibility_flushed
                    &&& p.handle == plan_view[j].handle
                    &&& p.handle.frame.0 < batch_core.frames_spec().metas_spec().len()
                    &&& batch_core.frames_spec().meta(p.handle.frame).state != STATE_INITIALIZING
                    &&& routes[j].matches_posted(&p, old(self).channel.received(), old(core))
                    &&& pull_tracked_or_done(&old(self).rdma, p.wr_id, p.writer@)
                },
            forall|a: int, b: int|
                0 <= a < b < posted.len() ==> {
                    &&& (#[trigger] posted[a]).wr_id != (#[trigger] posted[b]).wr_id
                    &&& posted[a].handle.frame != posted[b].handle.frame
                    &&& routes[a].tag != routes[b].tag
                },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(core).instance_id() == batch_core.instance_id(),
            final(core).node_id_spec() == batch_core.node_id_spec(),
            final(self).well_formed(final(core)),
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            rdma::read_batch_qp_frame(&final(self).rdma.qp, &batch_rdma.qp),
            final(self).rdma.timeline_id == batch_rdma.timeline_id,
            final(self).rdma.peer_node == batch_rdma.peer_node,
            peer_core_step(&batch_core, final(core), batch_rdma.timeline_id),
            peer_foreign_shares_preserved(
                &batch_core,
                final(core),
                batch_rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            r.pending.len() == 0 || r.wait_error.is_some(),
            r.results.len() + r.pending.len() == results.len() + posted.len(),
    {
        hide(CacheCore::well_formed);
        hide(CacheCore::all_versions_bound);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(InFlightRdmaPull::valid);
        hide(PullGrantReturn::matches_posted);
        hide(rdma_lease_index_spec);
        hide(replica_valid);
        hide(complete_receipts);
        hide(super::protocol::statement_replica_valid);
        let n = posted.len();
        let ghost count = results@.len() + posted@.len();
        let ghost before_wait = *core;
        let ghost before_wait_rdma = self.rdma;
        let ghost before_wait_channel = self.channel;
        let (statuses, wait_error) = self.await_batch_completions(
            core,
            posted.as_slice(),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans(&batch_core, &before_wait, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                &batch_core,
                &before_wait,
                core,
                batch_rdma.active_leases@,
                before_wait_rdma.active_leases@,
                self.rdma.active_leases@,
            );
        }
        let ghost pre_flush = posted@;
        let (posted, wait_error) = if wait_error.is_none() {
            proof {
                assert(self.rdma.well_formed(core)) by {
                    reveal(PeerSession::well_formed);
                }
                assert forall|j: int| 0 <= j < n implies {
                    let p = #[trigger] pre_flush[j];
                    &&& p.valid(&self.rdma, core) && p.target == target
                    &&& p.completion_observed(core)
                    &&& target is GpuGdr ==> !p.batch_visible()
                } by {
                    assert(observed_batch_status(&pre_flush[j], core, statuses[j]));
                }
                lemma_core_frames_separated(core);
                assert forall|a: int, b: int| 0 <= a < b < n implies (
                #[trigger] pre_flush[a]).handle.slot@.loc != (
                #[trigger] pre_flush[b]).handle.slot@.loc by {
                    reveal(InFlightRdmaPull::valid);
                    let left = pre_flush[a];
                    let right = pre_flush[b];
                    assert(left.handle == plan_view[a].handle);
                    assert(right.handle == plan_view[b].handle);
                    assert(left.handle.frame != right.handle.frame);
                    assert(left.handle.slot@ == core.frames_spec().frame(left.handle.frame).slot);
                    assert(right.handle.slot@ == core.frames_spec().frame(right.handle.frame).slot);
                }
            }
            finish_pull_read_batch(&self.rdma, core, Tracked(instance), target, posted)
        } else {
            proof {
                assert(observed_pull_batch_matches(posted@, pre_flush, &self.rdma, core, false))
                    by {
                    reveal(observed_pull_batch_matches);
                }
            }
            (posted, wait_error)
        };
        let ghost all_pulls = posted@;
        let ghost all_routes = routes@;
        let ghost batch = PullBatchView {
            pulls: all_pulls,
            routes: all_routes,
            statuses: statuses@,
            plans: plan_view,
            before: batch_core,
            visible: wait_error.is_none(),
        };
        assert(n == all_pulls.len() == all_routes.len() <= plan_view.len());
        proof {
            assert(pull_batch_targets_distinct(all_pulls, all_routes)) by {
                reveal(pull_batch_targets_distinct);
                assert forall|a: int, b: int| 0 <= a < b < n implies {
                    &&& (#[trigger] all_pulls[a]).handle.frame != (
                    #[trigger] all_pulls[b]).handle.frame
                    &&& all_routes[a].tag != all_routes[b].tag
                } by {
                    lemma_observed_pull_batch_entry(
                        all_pulls,
                        pre_flush,
                        &self.rdma,
                        core,
                        wait_error.is_none(),
                        a,
                    );
                    lemma_observed_pull_batch_entry(
                        all_pulls,
                        pre_flush,
                        &self.rdma,
                        core,
                        wait_error.is_none(),
                        b,
                    );
                }
            }
            assert(batch.valid_suffix(0, self, core)) by {
                reveal(PullBatchView::valid_suffix);
                assert forall|j: int| 0 <= j < n implies #[trigger] batch.valid_entry(
                    j,
                    self,
                    core,
                ) by {
                    let p = all_pulls[j];
                    let previous = pre_flush[j];
                    lemma_observed_pull_batch_entry(
                        all_pulls,
                        pre_flush,
                        &self.rdma,
                        core,
                        wait_error.is_none(),
                        j,
                    );
                    assert(all_routes[j].matches_posted(&p, self.channel.received(), core)) by {
                        reveal(PullGrantReturn::matches_posted);
                        assert(all_routes[j].matches_posted(
                            &previous,
                            before_wait_channel.received(),
                            &before_wait,
                        ));
                    }
                    if j < statuses.len() {
                        assert(observed_batch_status(&previous, core, statuses[j]));
                    }
                }
            }
        }
        let mut pulls = posted.into_iter();
        let mut routes = routes.into_iter();
        let mut pending = Vec::new();
        let mut k = 0;
        proof {
            assert(core.node_id_spec() == batch_core.node_id_spec()) by {
                reveal(peer_core_step_except);
                reveal(core_transition);
            }
        }
        while k < n
            invariant
                k <= n == all_pulls.len() == all_routes.len(),
                n <= plan_view.len(),
                results.len() + pending.len() + n - k == count,
                vstd::std_specs::iter::IteratorSpec::remaining(&pulls) == all_pulls.skip(k as int),
                vstd::std_specs::iter::IteratorSpec::remaining(&routes) == all_routes.skip(
                    k as int,
                ),
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.instance_id() == batch_core.instance_id(),
                core.node_id_spec() == batch_core.node_id_spec(),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                rdma::read_batch_qp_frame(&self.rdma.qp, &batch_rdma.qp),
                self.rdma.timeline_id == batch_rdma.timeline_id,
                self.rdma.peer_node == batch_rdma.peer_node,
                peer_core_step(&batch_core, core, self.rdma.timeline_id),
                peer_foreign_shares_preserved(
                    &batch_core,
                    core,
                    batch_rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                pending.len() == 0 || wait_error.is_some(),
                statuses.len() <= n,
                wait_error.is_none() ==> statuses.len() == n,
                batch == (PullBatchView {
                    pulls: all_pulls,
                    routes: all_routes,
                    statuses: statuses@,
                    plans: plan_view,
                    before: batch_core,
                    visible: wait_error.is_none(),
                }),
                pull_batch_targets_distinct(all_pulls, all_routes),
                batch.valid_suffix(k as int, self, core),
            decreases n - k,
        {
            let pull = pulls.next().unwrap();
            let route = routes.next().unwrap();
            if wait_error.is_some() {
                let status = if k < statuses.len() {
                    Some(statuses[k])
                } else {
                    None
                };
                pending.push(PendingBatchPull { pull, route, status });
                assert(batch.valid_suffix(k as int + 1, self, core)) by {
                    reveal(PullBatchView::valid_suffix);
                }
            } else {
                let status = statuses[k];
                let ghost before_finish = *core;
                let ghost prior_rdma = self.rdma;
                let ghost prior_session = *self;
                let ghost frame = pull.handle.frame;
                let ghost tag = route.tag;
                proof {
                    assert(pull == all_pulls[k as int]);
                    assert(route == all_routes[k as int]);
                    batch.lemma_entry(k as int, k as int, self, core);
                    assert(pull.valid(&self.rdma, core));
                    assert(route.matches_posted(&pull, self.channel.received(), core));
                    assert(route.owns_grant(self.channel.received(), pull.share_cap@)
                        && route.matches_version(pull.handle.version, pull.copy@.source.bytes)
                        && transfer::transport::covers(route.frontier@, core.citations())
                        && pull.share_cap@.frontier() == (Frontier {
                        last_writer: route.frontier@.last_writer,
                        readers: route.frontier@.readers.insert(pull.writer@),
                    })) by {
                        reveal(PullGrantReturn::matches_posted);
                    }
                    assert(observed_batch_status(&pull, core, status));
                    assert(core.citations().contains(pull.writer@));
                }
                let outcome = self.finish_observed_pull(
                    core,
                    route.tag,
                    route.request_id,
                    pull,
                    status,
                    route.binding,
                    route.frontier,
                    Tracked(instance),
                );
                proof {
                    lemma_peer_core_step_trans_retire_new_init(
                        &batch_core,
                        &before_finish,
                        core,
                        self.rdma.timeline_id,
                        frame,
                    );
                    lemma_foreign_shares_trans(
                        &batch_core,
                        &before_finish,
                        core,
                        batch_rdma.active_leases@,
                        prior_rdma.active_leases@,
                        self.rdma.active_leases@,
                    );
                    lemma_batch_status_growth(
                        &before_finish,
                        core,
                        self.rdma.timeline_id,
                        Some(frame),
                    );
                    assert(batch.valid_suffix(k as int + 1, self, core)) by {
                        reveal(PullBatchView::valid_suffix);
                        assert forall|j: int| k < j < n implies #[trigger] batch.valid_entry(
                            j,
                            self,
                            core,
                        ) by {
                            lemma_pull_batch_targets_distinct(all_pulls, all_routes, k as int, j);
                            batch.lemma_entry(k as int, j, &prior_session, &before_finish);
                            assert(all_pulls[j].handle.frame != frame);
                            assert(all_routes[j].tag != tag);
                            lemma_pull_valid_after_peer_step(
                                &prior_rdma,
                                &self.rdma,
                                &before_finish,
                                core,
                                &all_pulls[j],
                                Some(frame),
                            );
                            assert(all_routes[j].matches_posted(
                                &all_pulls[j],
                                self.channel.received(),
                                core,
                            )) by {
                                reveal(PullGrantReturn::matches_posted);
                            }
                        }
                    }
                }
                results.push((route.request_id, outcome));
            }
            k += 1;
        }
        FetchBatchPulls { results, pending, wait_error }
    }

    pub open(crate) spec fn pull_tier_valid(&self, tier: TierId, target: PullTarget) -> bool {
        match target {
            PullTarget::Host => self.rdma.effective_host_mr_spec() matches Some(s) && s.tier == tier
                && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
            PullTarget::GpuGdr => self.rdma.effective_gpu_mr_spec() matches Some(s) && s.tier
                == tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
        }
    }

    /// Initiator helper: await CQ completion (via `pump_cq` / `completed_pulls`),
    /// verify version binding, finish/abort the pull, register the version binding
    /// on hit, and send `READ_RELEASE`.
    #[verifier::rlimit(250)]
    pub(crate) fn fetch_complete_pull(
        &mut self,
        core: &mut CacheCore,
        tag: GrantTag,
        req_id: u64,
        pull: InFlightRdmaPull,
        Tracked(vb): Tracked<VersionBinding>,
        Ghost(orig_share_frontier): Ghost<foundation::tracked::Frontier<TransferDomain>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            pull.valid(&old(self).rdma, old(core)),
            old(self).rdma.pending().contains_key(pull.wr_id) || old(self).rdma.completed_pull_spec(
                pull.wr_id,
            ) is Some,
            old(self).rdma.pending().contains_key(pull.wr_id) ==> old(
                self,
            ).rdma.pending()[pull.wr_id] == pull.writer@,
            old(self).channel.received().contains_key(tag),
            old(self).channel.received()[tag] is ReadShare,
            pull.share_cap@.loc() == old(self).channel.received()[tag]->ReadShare_slot.loc,
            pull.share_cap@.share() == old(self).channel.received()[tag]->ReadShare_share_id,
            pull.share_cap@.frontier() == (foundation::tracked::Frontier {
                last_writer: orig_share_frontier.last_writer,
                readers: orig_share_frontier.readers.insert(pull.writer@),
            }),
            transfer::transport::covers(orig_share_frontier, old(core).citations()),
            vb.instance_id() == cluster_version_instance_id(),
            vb.key() == (pull.handle.version.origin_node as NodeId, pull.handle.version.seq),
            vb.value() == pull.copy@.source.bytes,
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step_except(
                old(core),
                final(core),
                old(self).rdma.timeline_id,
                Some(pull.handle.frame),
            ),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost pull_frame = pull.handle.frame;
        let status = match self.fetch_await_pull(core, &pull, Tracked(instance)) {
            Ok(s) => s,
            Err(e) => {
                proof {
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::preserves_initializing);
                }
                return Err(e);
            },
        };
        let ghost before_finish = *core;
        let ghost before_finish_leases = self.rdma.active_leases@;
        let res = self.finish_observed_pull(
            core,
            tag,
            req_id,
            pull,
            status,
            Tracked(vb),
            Ghost(orig_share_frontier),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_then_except(
                old(core),
                &before_finish,
                core,
                self.rdma.timeline_id,
                Some(pull_frame),
            );
            lemma_foreign_shares_trans(
                old(core),
                &before_finish,
                core,
                old(self).rdma.active_leases@,
                before_finish_leases,
                self.rdma.active_leases@,
            );
        }
        res
    }

    /// Finish a pull whose CQ status has already been reaped.
    #[verifier::rlimit(150)]
    pub(crate) fn finish_observed_pull(
        &mut self,
        core: &mut CacheCore,
        tag: GrantTag,
        req_id: u64,
        pull: InFlightRdmaPull,
        status: Result<(), i32>,
        Tracked(vb): Tracked<VersionBinding>,
        Ghost(orig_share_frontier): Ghost<foundation::tracked::Frontier<TransferDomain>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            pull.valid(&old(self).rdma, old(core)),
            pull.completion_observed(old(core)),
            old(core).citations().contains(pull.writer@),
            status.is_ok() ==> event_succeeded(
                old(core).ledger_spec()@,
                old(core).node(),
                pull.writer@,
            ),
            old(self).channel.received().contains_key(tag),
            old(self).channel.received()[tag] is ReadShare,
            pull.share_cap@.loc() == old(self).channel.received()[tag]->ReadShare_slot.loc,
            pull.share_cap@.share() == old(self).channel.received()[tag]->ReadShare_share_id,
            pull.share_cap@.frontier() == (foundation::tracked::Frontier {
                last_writer: orig_share_frontier.last_writer,
                readers: orig_share_frontier.readers.insert(pull.writer@),
            }),
            transfer::transport::covers(orig_share_frontier, old(core).citations()),
            vb.instance_id() == cluster_version_instance_id(),
            vb.key() == (pull.handle.version.origin_node as NodeId, pull.handle.version.seq),
            vb.value() == pull.copy@.source.bytes,
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step_except(
                old(core),
                final(core),
                old(self).rdma.timeline_id,
                Some(pull.handle.frame),
            ),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_revoke_acks == old(self).pending_revoke_acks,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).quarantined_grants == old(self).quarantined_grants,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            final(core).timelines_spec() == old(core).timelines_spec(),
            final(self).rdma == old(self).rdma,
            forall|k: GrantTag|
                k != tag && #[trigger] old(self).channel.received().contains_key(k)
                    ==> final(self).channel.received().contains_key(k)
                    && final(self).channel.received()[k] == old(self).channel.received()[k],
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost pull_frame = pull.handle.frame;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        core.verify_version_binding(pull.handle.version, Tracked(&vb));
        let ghost before_fin_core = *core;
        let ghost pull_writer = pull.writer@;
        let outcome = match finish_pull_read(&self.rdma, core, Tracked(instance), pull, status) {
            Ok(o) => o,
            Err((e, _)) => {
                proof {
                    lemma_peer_core_step_then_except(
                        old(core),
                        &before_fin_core,
                        core,
                        self.rdma.timeline_id,
                        Some(pull_frame),
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_peer_core_step_then_except(
                old(core),
                &before_fin_core,
                core,
                self.rdma.timeline_id,
                Some(pull_frame),
            );
            lemma_peer_grants_same_layout(
                self.active_grants@,
                self.rdma.active_leases@,
                self.channel.sent(),
                self.next_grant_tag,
                &before_fin_core,
                core,
            );
            assert(core.frames_spec().same_layout(&before_fin_core.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            reveal(peer_core_step_except);
            reveal(core_transition);
            broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

            lemma_outbound_grants_preserved(
                self.pending_outbound_grants@,
                self.rdma.active_leases@,
                self.rdma.active_leases@,
                &before_fin_core,
                core,
            );
            lemma_foreign_shares_identical_leases(&before_fin_core, core, self.rdma.active_leases@);
            lemma_foreign_shares_trans(
                old(core),
                &before_fin_core,
                core,
                old(self).rdma.active_leases@,
                self.rdma.active_leases@,
                self.rdma.active_leases@,
            );
        }
        match outcome {
            PullOutcome::Published { frame, version, share_cap } => {
                let ghost before_reg = *core;
                core.try_register_version_binding(version, Tracked(Some(vb)));
                proof {
                    assert(core.frames_spec().same_directory(&before_reg.frames_spec())) by {
                        reveal(FrameTable::same_directory);
                    }
                    assert(core.frames_spec().same_layout(&before_reg.frames_spec())) by {
                        reveal(FrameTable::same_layout);
                    }
                    lemma_foreign_shares_identical_leases(
                        &before_reg,
                        core,
                        self.rdma.active_leases@,
                    );
                    lemma_foreign_shares_trans(
                        old(core),
                        &before_reg,
                        core,
                        old(self).rdma.active_leases@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                    );
                    lemma_completed_pulls_after_step(
                        self.rdma.completed_pulls,
                        &before_reg.timelines_spec(),
                        &core.timelines_spec(),
                        self.rdma.timeline_id,
                        self.rdma.cq.observer(),
                        &before_reg.ledger_spec()@,
                        &core.ledger_spec()@,
                    );
                    lemma_rdma_all_slabs_same_directory(
                        &self.rdma.host_mr,
                        &self.rdma.gpu_mr,
                        self.rdma.shared_host_mr,
                        self.rdma.shared_gpu_mr,
                        &before_reg,
                        core,
                        self.rdma.window_pool.pd(),
                    );
                    lemma_rdma_leases_same_frames(
                        self.rdma.active_leases@,
                        &self.rdma.window_pool,
                        &self.rdma.qp,
                        &before_reg,
                        core,
                    );
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                        &before_reg,
                        core,
                    );
                    assert(peer_core_step(&before_reg, core, self.rdma.timeline_id)) by {
                        reveal(peer_core_step_except);
                        reveal(core_transition);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                    }
                    lemma_peer_core_step_except_then(
                        old(core),
                        &before_reg,
                        core,
                        self.rdma.timeline_id,
                        Some(pull_frame),
                    );
                    assert(core.all_versions_bound()) by {
                        assert forall|v: VersionId| #[trigger]
                            core.frames_spec().versions_spec()@.contains_key(
                                v,
                            ) implies core.has_version_binding(v) by {
                            if v != version {
                                assert(before_fin_core.frames_spec().versions_spec()@.contains_key(
                                    v,
                                ));
                                assert(before_fin_core.has_version_binding(v));
                                assert(before_reg.has_version_binding(v));
                            }
                        }
                    }
                    assert(core.citations().contains(pull_writer));
                    assert(transfer::transport::covers(share_cap@.frontier(), core.citations()));
                }
                self.fetch_release_grant(core, tag, req_id, share_cap, Tracked(instance));
                Ok(FetchOutcome::Hit { frame, version })
            },
            PullOutcome::Aborted { status: code, share_cap } => {
                proof {
                    assert(core.citations().contains(pull_writer));
                    assert(transfer::transport::covers(share_cap@.frontier(), core.citations()));
                    assert(core.all_versions_bound()) by {
                        reveal(FrameTable::same_directory);
                    }
                }
                self.fetch_release_grant(core, tag, req_id, share_cap, Tracked(instance));
                Err(rdma_error(rdma::RdmaError(code)))
            },
        }
    }

    /// Internal helper: route one decoded `ReceivedFrame` into `PeerSession` queues
    /// (handling `ReadRelease` via `handle_read_release` when `unbind_leases` is true,
    /// and never silently dropping `ShareCap`, `VersionBinding`, or notices).
    #[verifier::rlimit(300)]
    fn route_received_frame(
        &mut self,
        core: &mut CacheCore,
        Ghost(pre_chan): Ghost<Channel<T, VeriCacheAttachment>>,
        frame: ReceivedFrame<VeriCacheAttachment>,
        unbind_leases: bool,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(bool, bool), CoreError>)
        requires
            old(core).well_formed(),
            old(self).rdma.well_formed(old(core)),
            unbind_leases ==> (old(self).rdma.qp.ready() && old(
                self,
            ).rdma.qp.state() is ReadyToSend),
            instance.id() == old(core).instance_id(),
            old(self).channel.local() == old(core).node(),
            old(self).channel.peer() == old(self).rdma.peer_node,
            old(self).channel.id() == old(core).instance_id(),
            old(self).channel.same_session(pre_chan),
            Channel::<T, VeriCacheAttachment>::frame_delivery(pre_chan, old(self).channel, frame),
            forall|t: GrantTag| #[trigger]
                pre_chan.sent().contains_key(t) ==> t < old(self).next_grant_tag,
            forall|i: int, j: int|
                0 <= i < j < old(self).active_grants@.len() ==> (#[trigger] old(
                    self,
                ).active_grants@[i]).grant_tag != (#[trigger] old(self).active_grants@[j]).grant_tag
                    && old(self).active_grants@[i].handle != old(self).active_grants@[j].handle,
            forall|i: int|
                0 <= i < old(self).active_grants@.len() ==> {
                    let g = #[trigger] old(self).active_grants@[i];
                    &&& g.grant_tag < old(self).next_grant_tag
                    &&& g.frame.0 < old(core).frames_spec().metas_spec().len()
                    &&& old(self).rdma.lease_index_spec(g.handle) matches Some(idx) && {
                        let lease = old(self).rdma.active_leases@[idx];
                        &&& lease.frame == g.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& pre_chan.sent().contains_key(g.grant_tag)
                        &&& pre_chan.sent()[g.grant_tag] == (GrantKind::ReadShare {
                            slot: old(core).frames_spec().frame(g.frame).slot,
                            share_id: lease.share@,
                        })
                    }
                },
            forall|i: int, j: int|
                0 <= i < j < old(self).pending_outbound_grants@.len() ==> (#[trigger] old(
                    self,
                ).pending_outbound_grants@[i]).handle != (#[trigger] old(
                    self,
                ).pending_outbound_grants@[j]).handle,
            forall|i: int|
                0 <= i < old(self).pending_outbound_grants@.len() ==> (#[trigger] old(
                    self,
                ).pending_outbound_grants@[i]).valid(&old(self).rdma, old(core)),
            forall|i: int|
                0 <= i < old(self).pending_inbound_grants@.len() ==> (#[trigger] old(
                    self,
                ).pending_inbound_grants@[i]).valid(old(core), &pre_chan, old(self).rdma.peer_node),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            pre_chan.received().submap_of(final(self).channel.received()),
            forall|k: GrantTag| #[trigger]
                pre_chan.received().contains_key(k) && inbound_tag_unbuffered(
                    old(self).pending_inbound_grants@,
                    k,
                ) ==> inbound_tag_unbuffered(final(self).pending_inbound_grants@, k),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants@.len() < old(self).active_grants@.len()
                || final(self).active_grants == old(self).active_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        match frame {
            ReceivedFrame::Notice(rec) => {
                let Received { user_bytes, .. } = rec;
                match decode_control_payload(&user_bytes) {
                    Some(ControlPayload::FetchRequest { request_id, key }) => {
                        self.pending_inbound_fetches.push((request_id, key));
                        Ok((true, false))
                    },
                    Some(ControlPayload::FetchBatch { entries }) => {
                        append_fetch_entries(&mut self.pending_inbound_fetches, entries.as_slice());
                        Ok((true, false))
                    },
                    Some(ControlPayload::FetchMiss { request_id }) => {
                        self.pending_inbound_misses.push(request_id);
                        Ok((true, false))
                    },
                    Some(ControlPayload::RevokeNotice { grant_tag }) => {
                        self.pending_revoke_notices.push(grant_tag);
                        Ok((true, false))
                    },
                    Some(ControlPayload::RevokeAck { grant_tag }) => {
                        self.pending_revoke_acks.push(grant_tag);
                        Ok((true, false))
                    },
                    Some(ControlPayload::ReconnectBarrier { epoch }) => {
                        self.pending_reconnect_barriers.push(epoch);
                        if epoch > self.peer_reconnect_epoch {
                            self.peer_reconnect_epoch = epoch;
                        }
                        Ok((true, false))
                    },
                    None => Ok((false, false)),
                }
            },
            ReceivedFrame::ReadGrant(rec) => {
                let Received { tag, user_bytes, capability, share, statement, attachment, .. } =
                    rec;
                let Some(meta) = decode_read_grant_meta(&user_bytes) else {
                    return Err(rdma_error(rdma::RdmaError(-1)));
                };
                let cap = capability.unwrap();
                let tracked sh = share.get().tracked_unwrap();
                let tracked att = attachment.get();
                let ghost cert;
                let tracked revoked;
                let tracked version_binding;
                proof {
                    match att {
                        VeriCacheAttachment::ReadGrant {
                            meta: att_meta,
                            cert: c,
                            revoked: r,
                            version_binding: vb,
                        } => {
                            super::protocol::lemma_encode_read_grant_meta_injective(meta, att_meta);
                            cert = c;
                            revoked = r;
                            version_binding = vb;
                        },
                        _ => {
                            assert(false);
                            cert = arbitrary();
                            revoked = proof_from_false();
                            version_binding = None;
                        },
                    }
                }
                self.pending_inbound_grants.push(
                    PendingInboundGrant {
                        tag,
                        meta,
                        cap,
                        share: Tracked(sh),
                        statement,
                        cert: Ghost(cert),
                        revoked: Tracked(revoked),
                        version_binding: Tracked(version_binding),
                    },
                );
                proof {
                    broadcast use vstd::map::group_map_lemmas;

                    assert(pre_chan.received().submap_of(self.channel.received()));
                }
                Ok((true, false))
            },
            ReceivedFrame::ReadRelease(rec) => {
                let Received { tag, user_bytes, share, statement, .. } = rec;
                let Some(_meta) = decode_read_release_meta(&user_bytes) else {
                    return Err(rdma_error(rdma::RdmaError(-1)));
                };
                let tracked sh = share.get().tracked_unwrap();
                if unbind_leases {
                    match self.handle_read_release(
                        core,
                        tag,
                        Tracked(sh),
                        statement,
                        Tracked(instance),
                    ) {
                        Ok(()) | Err(CoreError::InvalidArgument) => Ok((false, true)),
                        Err(e) => Err(e),
                    }
                } else {
                    if let Some(pos) = self.find_grant_index(tag) {
                        let ghost before_grants = self.active_grants@;
                        let _ = self.active_grants.swap_remove(pos);
                        proof {
                            lemma_swap_removed_grant(
                                before_grants,
                                self.active_grants@,
                                pos as int,
                            );
                        }
                    }
                    let ghost before_merge = *core;
                    let tracked stmt = statement.get();
                    core.merge_peer_statement(
                        Tracked(stmt.submitted),
                        Tracked(stmt.knows),
                        Tracked(stmt.succeeded),
                        Tracked(instance),
                    );
                    proof {
                        lemma_rdma_window_after_fence_ledger(
                            self.rdma.in_flight_ops,
                            self.rdma.bitmap,
                            &core.timelines_spec(),
                            self.rdma.timeline_id,
                            &before_merge.ledger_spec()@,
                            &core.ledger_spec()@,
                            rdma::CqSnapshot::of(self.rdma.cq),
                            core.instance_id(),
                            self.rdma.next_control_wr_id,
                        );
                        lemma_completed_pulls_after_step(
                            self.rdma.completed_pulls,
                            &before_merge.timelines_spec(),
                            &core.timelines_spec(),
                            self.rdma.timeline_id,
                            self.rdma.cq.observer(),
                            &before_merge.ledger_spec()@,
                            &core.ledger_spec()@,
                        );
                        assert(core.frames_spec().same_directory(&before_merge.frames_spec())) by {
                            reveal(FrameTable::same_directory);
                        }
                        assert(core.frames_spec().same_layout(&before_merge.frames_spec())) by {
                            reveal(FrameTable::same_layout);
                        }
                        lemma_rdma_all_slabs_same_directory(
                            &self.rdma.host_mr,
                            &self.rdma.gpu_mr,
                            self.rdma.shared_host_mr,
                            self.rdma.shared_gpu_mr,
                            &before_merge,
                            core,
                            self.rdma.window_pool.pd(),
                        );
                        lemma_rdma_leases_same_frames(
                            self.rdma.active_leases@,
                            &self.rdma.window_pool,
                            &self.rdma.qp,
                            &before_merge,
                            core,
                        );
                        lemma_outbound_grants_preserved(
                            self.pending_outbound_grants@,
                            self.rdma.active_leases@,
                            self.rdma.active_leases@,
                            &before_merge,
                            core,
                        );
                        assert(peer_core_step(&before_merge, core, self.rdma.timeline_id)) by {
                            reveal(peer_core_step_except);
                            reveal(core_transition);
                            reveal(FrameTable::preserves_initializing);
                        }
                        lemma_foreign_shares_identical_leases(
                            &before_merge,
                            core,
                            self.rdma.active_leases@,
                        );
                    }
                    Ok((false, true))
                }
            },
            ReceivedFrame::WriteGrant(_) | ReceivedFrame::WriteRelease(_) => Ok((false, false)),
        }
    }

    /// Blocking-transport-safe single step: drain all currently posted CQEs
    /// (`pump_cq_all`) and process at most ONE frame via `Channel::try_recv_any`.
    pub(crate) fn pump_cq_and_channel_step(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<bool, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            old(self).channel.received().submap_of(final(self).channel.received()),
            forall|k: GrantTag| #[trigger]
                old(self).channel.received().contains_key(k) && inbound_tag_unbuffered(
                    old(self).pending_inbound_grants@,
                    k,
                ) ==> inbound_tag_unbuffered(final(self).pending_inbound_grants@, k),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants@.len() < old(self).active_grants@.len()
                || final(self).active_grants == old(self).active_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost step_core = *core;
        let ghost step_leases = self.rdma.active_leases@;
        let ghost step_rdma = self.rdma;
        let cq_n = match self.rdma.pump_cq_all(core, Tracked(instance)) {
            Ok(n) => n,
            Err(e) => {
                proof {
                    lemma_pump_cq_all_preserves_pull_tracked(&step_rdma, &self.rdma);
                }
                return Err(e);
            },
        };
        proof {
            reveal(rdma_leases_valid);
            assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                let g = #[trigger] self.active_grants@[i];
                &&& g.grant_tag < self.next_grant_tag
                &&& g.frame.0 < core.frames_spec().metas_spec().len()
                &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                    let lease = self.rdma.active_leases@[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& self.channel.sent().contains_key(g.grant_tag)
                    &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                        slot: core.frames_spec().frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            } by {
                let g = self.active_grants@[i];
                let oidx = rdma_lease_index_spec(step_leases, g.handle).unwrap();
                assert(step_leases[oidx].state == RemoteLeaseState::Bound);
                assert(!step_leases[oidx].has_active_wr());
            }
            assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            lemma_outbound_grants_preserved(
                self.pending_outbound_grants@,
                step_leases,
                self.rdma.active_leases@,
                &step_core,
                core,
            );
            lemma_pump_cq_all_preserves_pull_tracked(&step_rdma, &self.rdma);
        }
        let ghost pre_chan = self.channel;
        let ghost after_cq_core = *core;
        let ghost after_cq_leases = self.rdma.active_leases@;
        let frame_opt = match self.channel.try_recv_any(Tracked(instance)) {
            Ok(opt) => opt,
            Err(_) => return Err(rdma_error(rdma::RdmaError(-1))),
        };
        if let Some(frame) = frame_opt {
            match self.route_received_frame(core, Ghost(pre_chan), frame, true, Tracked(instance)) {
                Ok((was_frame, was_rel)) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &after_cq_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &after_cq_core,
                            core,
                            old(self).rdma.active_leases@,
                            after_cq_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    Ok(cq_n > 0 || was_frame || was_rel)
                },
                Err(e) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &after_cq_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &after_cq_core,
                            core,
                            old(self).rdma.active_leases@,
                            after_cq_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    Err(e)
                },
            }
        } else {
            Ok(cq_n > 0)
        }
    }

    /// Bounded non-blocking drain of CQEs + inbound channel frames (does NOT
    /// require `SyncContext` or `ShareSource`). Use `pump_cq_and_channel_step`
    /// when interleaving single frames over blocking `TcpTransport`.
    pub(crate) fn pump_cq_and_channel_basic(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<bool, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants@.len() < old(self).active_grants@.len()
                || final(self).active_grants == old(self).active_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        let progress = self.drain_inbound_frames(core, Tracked(instance))?;
        Ok(
            progress.cq_processed > 0 || progress.frames_received > 0 || progress.releases_processed
                > 0,
        )
    }

    /// Poll CQEs while preserving the session's buffered grants and pulls.
    fn pump_session_cq(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).channel == old(self).channel,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
            final(self).rdma.pending().dom().subset_of(old(self).rdma.pending().dom()),
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(RdmaBackend::pending);
        hide(RdmaBackend::completed_pull_spec);
        hide(rdma_lease_index_spec);
        hide(replica_valid);
        hide(complete_receipts);
        hide(super::protocol::statement_replica_valid);
        let ghost step_core = *core;
        let ghost step_leases = self.rdma.active_leases@;
        let ghost step_rdma = self.rdma;
        proof {
            assert(self.rdma.well_formed(core)) by {
                reveal(PeerSession::well_formed);
            }
        }
        let result = self.rdma.pump_cq_all(core, Tracked(instance));
        proof {
            assert(self.well_formed(core)) by {
                reveal(PeerSession::well_formed);
                assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                    let g = #[trigger] self.active_grants@[i];
                    &&& g.grant_tag < self.next_grant_tag
                    &&& g.frame.0 < core.frames_spec().metas_spec().len()
                    &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                        let lease = self.rdma.active_leases@[idx];
                        &&& lease.frame == g.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& self.channel.sent().contains_key(g.grant_tag)
                        &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                            slot: core.frames_spec().frame(g.frame).slot,
                            share_id: lease.share@,
                        })
                    }
                } by {
                    let g = self.active_grants@[i];
                    let before_idx = step_rdma.lease_index_spec(g.handle).unwrap();
                    let before_lease = step_leases[before_idx];
                    assert(before_lease.state == RemoteLeaseState::Bound);
                    assert(!before_lease.has_active_wr());
                    assert(self.rdma.lease_index_spec(g.handle) matches Some(idx)
                        && self.rdma.active_leases@[idx] == before_lease);
                }
                assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                    reveal(FrameTable::same_layout);
                }
                assert(rdma_leases_disjoint(self.rdma.active_leases@)) by {
                    reveal(RdmaBackend::well_formed);
                }
                lemma_outbound_grants_preserved(
                    self.pending_outbound_grants@,
                    step_leases,
                    self.rdma.active_leases@,
                    &step_core,
                    core,
                );
            }
            lemma_pump_cq_all_preserves_pull_tracked(&step_rdma, &self.rdma);
        }
        result
    }

    /// Drain a bounded burst of frames after polling currently available CQEs.
    fn drain_inbound_frames(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants@.len() < old(self).active_grants@.len()
                || final(self).active_grants == old(self).active_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut progress = PeerProgress::empty();
        progress.cq_processed = self.pump_session_cq(core, Tracked(instance))?;
        let mut budget: usize = 64;
        while budget > 0
            invariant
                budget <= 64,
                core.well_formed(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.channel.same_session(old(self).channel),
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants@.len() < old(self).active_grants@.len() || self.active_grants
                    == old(self).active_grants,
                forall|seq: u64, writer: TransferEvent| #[trigger]
                    pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                        &self.rdma,
                        seq,
                        writer,
                    ),
            decreases budget,
        {
            let ghost iter_core = *core;
            let ghost iter_leases = self.rdma.active_leases@;
            let ghost pre_chan = self.channel;
            let frame_opt = match self.channel.try_recv_any(Tracked(instance)) {
                Ok(opt) => opt,
                Err(_) => return Err(rdma_error(rdma::RdmaError(-1))),
            };
            match frame_opt {
                None => break,
                Some(frame) => {
                    let route_res = self.route_received_frame(
                        core,
                        Ghost(pre_chan),
                        frame,
                        true,
                        Tracked(instance),
                    );
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &iter_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &iter_core,
                            core,
                            old(self).rdma.active_leases@,
                            iter_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    match route_res {
                        Ok((was_f, was_r)) => {
                            progress.frames_received = progress.frames_received.saturating_add(
                                if was_f {
                                    1
                                } else {
                                    0
                                },
                            );
                            progress.releases_processed
                                = progress.releases_processed.saturating_add(
                                if was_r {
                                    1
                                } else {
                                    0
                                },
                            );
                            budget -= 1;
                        },
                        Err(e) => return Err(e),
                    }
                },
            }
        }
        Ok(progress)
    }

    /// Donor: apply all received `RevokeAck { grant_tag }` notifications against
    /// `self.quarantined_grants`, transitioning their MW pool slots from
    /// `Quarantined` to `Vacant`.
    #[verifier::rlimit(50)]
    pub(crate) fn acknowledge_pending_revoke_acks(&mut self, core: &CacheCore) -> (n: usize)
        requires
            core.well_formed(),
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.cq == old(self).rdma.cq,
            final(self).rdma.bitmap == old(self).rdma.bitmap,
            final(self).rdma.in_flight_ops == old(self).rdma.in_flight_ops,
            final(self).rdma.completed_pulls == old(self).rdma.completed_pulls,
            final(self).rdma.active_leases == old(self).rdma.active_leases,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut count: usize = 0;
        while self.pending_revoke_acks.len() > 0
            invariant
                core.well_formed(),
                self.well_formed(core),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.cq == old(self).rdma.cq,
                self.rdma.bitmap == old(self).rdma.bitmap,
                self.rdma.in_flight_ops == old(self).rdma.in_flight_ops,
                self.rdma.completed_pulls == old(self).rdma.completed_pulls,
                self.rdma.active_leases == old(self).rdma.active_leases,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            decreases self.pending_revoke_acks@.len(),
        {
            let ack_tag = self.pending_revoke_acks.pop().unwrap();
            let mut qidx: usize = 0;
            let mut found_slot: Option<u32> = None;
            while qidx < self.quarantined_grants.len()
                invariant
                    qidx <= self.quarantined_grants@.len(),
                decreases self.quarantined_grants@.len() - qidx,
            {
                if self.quarantined_grants[qidx].0 == ack_tag {
                    let (_t, mw_slot) = self.quarantined_grants.swap_remove(qidx);
                    found_slot = Some(mw_slot);
                    break;
                }
                qidx += 1;
            }
            if let Some(mw_slot) = found_slot {
                let ghost before_pool = self.rdma.window_pool;
                if self.rdma.window_pool.acknowledge_quarantined(mw_slot).is_ok() {
                    proof {
                        reveal(rdma_leases_valid);
                        assert forall|i: int| 0 <= i < self.rdma.active_leases@.len() implies (
                        #[trigger] self.rdma.active_leases@[i]).mw_slot != mw_slot && (
                        self.rdma.active_leases@[i]).valid(
                            &self.rdma.window_pool,
                            &self.rdma.qp,
                            core,
                        ) by {
                            let l = self.rdma.active_leases@[i];
                            assert(l.valid(&before_pool, &self.rdma.qp, core));
                            assert(self.rdma.window_pool.slots[l.mw_slot as int]
                                == before_pool.slots[l.mw_slot as int]);
                        }
                        assert(rdma_control_leases_match_cq(
                            self.rdma.active_leases@,
                            &self.rdma.window_pool,
                            rdma::CqSnapshot::of(self.rdma.cq),
                            self.rdma.next_control_wr_id,
                        )) by {
                            reveal(rdma_control_leases_match_cq);
                        }
                        assert(rdma_pool_slots_owned_by_leases(
                            self.rdma.active_leases@,
                            &self.rdma.window_pool,
                        )) by {
                            reveal(rdma_pool_slots_owned_by_leases);
                        }
                    }
                    if count < usize::MAX {
                        count += 1;
                    }
                }
            }
        }
        proof {
            reveal(pull_tracked_or_done_state);
        }
        count
    }

    /// Acknowledge pending `RevokeAck`s and replenish any resulting `Vacant`
    /// `window_pool` slots using caller-supplied `EventSourceToken`s for `pd`.
    #[verifier::rlimit(200)]
    pub(crate) fn replenish_acknowledged_windows(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
    ) -> (n: usize)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            pd.id() == old(self).rdma.window_pool.pd(),
            pd.node() == core.node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == pd.node(),
        ensures
            final(self).well_formed(core),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.cq == old(self).rdma.cq,
            final(self).rdma.bitmap == old(self).rdma.bitmap,
            final(self).rdma.in_flight_ops == old(self).rdma.in_flight_ops,
            final(self).rdma.completed_pulls == old(self).rdma.completed_pulls,
            final(self).rdma.active_leases == old(self).rdma.active_leases,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_misses == old(self).pending_inbound_misses,
            final(self).pending_revoke_notices == old(self).pending_revoke_notices,
            final(self).pending_reconnect_barriers == old(self).pending_reconnect_barriers,
            final(self).reconnect_epoch == old(self).reconnect_epoch,
            final(self).peer_reconnect_epoch == old(self).peer_reconnect_epoch,
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node()
                    == pd.node(),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = self.acknowledge_pending_revoke_acks(core);
        let mut replenished: usize = 0;
        let len = self.rdma.window_pool.slots.len();
        let mut idx: usize = 0;
        while idx < len && tokens.len() > 0
            invariant
                idx <= len,
                len == self.rdma.window_pool.slots.len(),
                core.well_formed(),
                self.well_formed(core),
                pd.id() == self.rdma.window_pool.pd(),
                pd.node() == core.node(),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.cq == old(self).rdma.cq,
                self.rdma.bitmap == old(self).rdma.bitmap,
                self.rdma.in_flight_ops == old(self).rdma.in_flight_ops,
                self.rdma.completed_pulls == old(self).rdma.completed_pulls,
                self.rdma.active_leases == old(self).rdma.active_leases,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.channel == old(self).channel,
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
                self.pending_inbound_fetches == old(self).pending_inbound_fetches,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                self.pending_inbound_misses == old(self).pending_inbound_misses,
                self.pending_revoke_notices == old(self).pending_revoke_notices,
                self.pending_reconnect_barriers == old(self).pending_reconnect_barriers,
                self.reconnect_epoch == old(self).reconnect_epoch,
                self.peer_reconnect_epoch == old(self).peer_reconnect_epoch,
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source().is_rdma_qp() && tokens@[k]@.node() == pd.node(),
            decreases len - idx,
        {
            if matches!(self.rdma.window_pool.slots[idx].state, MwSlotState::Vacant) {
                let Tracked(tok) = tokens.pop().unwrap();
                let ghost before_pool = self.rdma.window_pool;
                proof {
                    reveal(window_pool_slots_wf);
                }
                if self.rdma.window_pool.replenish(idx as u32, pd, Tracked(tok)).is_ok() {
                    proof {
                        reveal(rdma_leases_valid);
                        assert forall|i: int| 0 <= i < self.rdma.active_leases@.len() implies (
                        #[trigger] self.rdma.active_leases@[i]).mw_slot != idx as u32 && (
                        self.rdma.active_leases@[i]).valid(
                            &self.rdma.window_pool,
                            &self.rdma.qp,
                            core,
                        ) by {
                            let l = self.rdma.active_leases@[i];
                            assert(l.valid(&before_pool, &self.rdma.qp, core));
                            assert(self.rdma.window_pool.slots[l.mw_slot as int]
                                == before_pool.slots[l.mw_slot as int]);
                        }
                        assert(rdma_control_leases_match_cq(
                            self.rdma.active_leases@,
                            &self.rdma.window_pool,
                            rdma::CqSnapshot::of(self.rdma.cq),
                            self.rdma.next_control_wr_id,
                        )) by {
                            reveal(rdma_control_leases_match_cq);
                        }
                        assert(rdma_pool_slots_owned_by_leases(
                            self.rdma.active_leases@,
                            &self.rdma.window_pool,
                        )) by {
                            reveal(rdma_pool_slots_owned_by_leases);
                        }
                    }
                    if replenished < usize::MAX {
                        replenished += 1;
                    }
                }
            }
            idx += 1;
        }
        proof {
            reveal(pull_tracked_or_done_state);
        }
        replenished
    }

    /// Initiator helper: release any unused buffered `PendingInboundGrant` matching
    /// queued `RevokeNotice { grant_tag }` and reply with `ControlPayload::RevokeAck`.
    #[verifier::rlimit(200)]
    fn send_pending_revoke_acks(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
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
            final(core).timelines_spec() == old(core).timelines_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma == old(self).rdma,
            final(self).channel.same_session(old(self).channel),
            forall|k: GrantTag| #[trigger]
                old(self).channel.received().contains_key(k) && inbound_tag_unbuffered(
                    old(self).pending_inbound_grants@,
                    k,
                ) ==> (final(self).channel.received().contains_key(k)
                    && final(self).channel.received()[k] == old(self).channel.received()[k]
                    && inbound_tag_unbuffered(final(self).pending_inbound_grants@, k)),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_outbound_grants == old(self).pending_outbound_grants,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost orig_received = self.channel.received();
        let ghost orig_inbound = self.pending_inbound_grants@;
        let mut sent_acks: usize = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        while self.pending_revoke_notices.len() > 0
            invariant
                orig_received == old(self).channel.received(),
                orig_inbound == old(self).pending_inbound_grants@,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec() == old(core).frames_spec(),
                core.catalog_spec() == old(core).catalog_spec(),
                core.spec_spec() == old(core).spec_spec(),
                core.next_version_seq_spec() == old(core).next_version_seq_spec(),
                core.version_seq_limit_spec() == old(core).version_seq_limit_spec(),
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec() == old(core).timelines_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma == old(self).rdma,
                self.channel.same_session(old(self).channel),
                forall|k: GrantTag| #[trigger]
                    orig_received.contains_key(k) && inbound_tag_unbuffered(orig_inbound, k) ==> (
                    self.channel.received().contains_key(k) && self.channel.received()[k]
                        == orig_received[k] && inbound_tag_unbuffered(
                        self.pending_inbound_grants@,
                        k,
                    )),
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_outbound_grants == old(self).pending_outbound_grants,
            decreases self.pending_revoke_notices@.len(),
        {
            let grant_tag = self.pending_revoke_notices.remove(0);
            let mut idx: usize = 0;
            let mut matched_idx: Option<usize> = None;
            while idx < self.pending_inbound_grants.len() && matched_idx.is_none()
                invariant
                    idx <= self.pending_inbound_grants@.len(),
                    matched_idx matches Some(m) ==> (m as int) < self.pending_inbound_grants@.len()
                        && self.pending_inbound_grants@[m as int].tag == grant_tag,
                decreases
                        (if matched_idx.is_none() {
                            1int
                        } else {
                            0int
                        }),
                        self.pending_inbound_grants@.len() - idx,
            {
                if self.pending_inbound_grants[idx].tag == grant_tag {
                    matched_idx = Some(idx);
                } else {
                    idx += 1;
                }
            }
            if let Some(m_idx) = matched_idx {
                let ghost pre_swap_inbound = self.pending_inbound_grants@;
                let g = self.pending_inbound_grants.swap_remove(m_idx);
                let ghost step_core = *core;
                let tracked stmt = g.statement.get();
                let ghost stmt_sub = stmt.submitted;
                let ghost stmt_facts = stmt.facts;
                let ghost stmt_closure = stmt.closure();
                core.merge_peer_statement(
                    Tracked(stmt.submitted),
                    Tracked(stmt.knows),
                    Tracked(stmt.succeeded),
                    Tracked(instance),
                );
                proof {
                    lemma_rdma_window_after_fence_ledger(
                        self.rdma.in_flight_ops,
                        self.rdma.bitmap,
                        &core.timelines_spec(),
                        self.rdma.timeline_id,
                        &step_core.ledger_spec()@,
                        &core.ledger_spec()@,
                        rdma::CqSnapshot::of(self.rdma.cq),
                        core.instance_id(),
                        self.rdma.next_control_wr_id,
                    );
                    lemma_completed_pulls_after_step(
                        self.rdma.completed_pulls,
                        &step_core.timelines_spec(),
                        &core.timelines_spec(),
                        self.rdma.timeline_id,
                        self.rdma.cq.observer(),
                        &step_core.ledger_spec()@,
                        &core.ledger_spec()@,
                    );
                    assert(core.frames_spec().same_directory(&step_core.frames_spec())) by {
                        reveal(FrameTable::same_directory);
                    }
                    assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                        reveal(FrameTable::same_layout);
                    }
                    lemma_rdma_all_slabs_same_directory(
                        &self.rdma.host_mr,
                        &self.rdma.gpu_mr,
                        self.rdma.shared_host_mr,
                        self.rdma.shared_gpu_mr,
                        &step_core,
                        core,
                        self.rdma.window_pool.pd(),
                    );
                    lemma_rdma_leases_same_frames(
                        self.rdma.active_leases@,
                        &self.rdma.window_pool,
                        &self.rdma.qp,
                        &step_core,
                        core,
                    );
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        self.rdma.active_leases@,
                        self.rdma.active_leases@,
                        &step_core,
                        core,
                    );
                    assert(peer_core_step(&step_core, core, self.rdma.timeline_id)) by {
                        reveal(peer_core_step_except);
                        reveal(core_transition);
                        reveal(FrameTable::preserves_initializing);
                    }
                    lemma_peer_core_step_trans(old(core), &step_core, core, self.rdma.timeline_id);
                    assert(stmt_closure.subset_of(core.citations())) by {
                        assert forall|e: TransferEvent| #[trigger]
                            stmt_closure.contains(e) implies core.citations().contains(e) by {
                            assert(stmt_sub.dom().contains(e)) by {
                                if !stmt_facts.contains(e) {
                                    let m = choose|m: TransferEvent|
                                        stmt_facts.contains(m) && (
                                        #[trigger] stmt_sub.map()[m]).preds.contains(e);
                                    assert(stmt_sub.dom().contains(m));
                                    assert(stmt_sub.map()[m].preds.subset_of(stmt_sub.dom()));
                                }
                            }
                            assert(core.ledger_spec()@.submitted.dom().contains(e));
                            assert(core.ledger_spec()@.knows.set().contains((core.node(), e)));
                        }
                    }
                    assert(transfer::transport::covers(g.share@.frontier(), core.citations()));
                    assert(!inbound_tag_unbuffered(pre_swap_inbound, grant_tag)) by {
                        assert(pre_swap_inbound[m_idx as int].tag == grant_tag);
                    }
                    assert forall|k: GrantTag| #[trigger]
                        orig_received.contains_key(k) && inbound_tag_unbuffered(
                            orig_inbound,
                            k,
                        ) implies k != grant_tag && inbound_tag_unbuffered(
                        self.pending_inbound_grants@,
                        k,
                    ) by {
                        assert(inbound_tag_unbuffered(pre_swap_inbound, k));
                        assert forall|j: int| 0 <= j < self.pending_inbound_grants@.len() implies (
                        #[trigger] self.pending_inbound_grants@[j]).tag != k by {
                            let src = if j == m_idx as int {
                                (pre_swap_inbound.len() - 1) as int
                            } else {
                                j
                            };
                            assert(self.pending_inbound_grants@[j] == pre_swap_inbound[src]);
                        }
                    }
                }
                self.fetch_release_grant(
                    core,
                    g.tag,
                    g.meta.request_id,
                    g.share,
                    Tracked(instance),
                );
            }
            if self.send_control(
                core,
                ControlPayload::RevokeAck { grant_tag },
                Tracked(instance),
            ).is_err() {
                proof {
                    lemma_foreign_shares_identical_leases(
                        old(core),
                        core,
                        self.rdma.active_leases@,
                    );
                }
                return Err(rdma_error(rdma::RdmaError(-1)));
            }
            if sent_acks < usize::MAX {
                sent_acks += 1;
            }
        }
        proof {
            lemma_foreign_shares_identical_leases(old(core), core, self.rdma.active_leases@);
        }
        Ok(sent_acks)
    }

    pub closed spec fn inbound_fetches(&self) -> Seq<(u64, ObjectKey)> {
        self.pending_inbound_fetches@
    }

    /// Drain received fetch requests so the caller can decide when to serve them.
    pub fn take_inbound_fetches(&mut self, core: &CacheCore) -> (r: Vec<(u64, ObjectKey)>)
        requires
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).rdma_spec() == old(self).rdma_spec(),
            r@ == old(self).inbound_fetches(),
            final(self).inbound_fetches().len() == 0,
    {
        let mut requests = Vec::new();
        core::mem::swap(&mut self.pending_inbound_fetches, &mut requests);
        requests
    }

    /// Number of window binds that fit in the free MW slots and CQ entries.
    pub fn bind_budget(&self, core: &CacheCore) -> (r: usize)
        requires
            self.well_formed(core),
        ensures
            r <= self.rdma_spec().window_pool_spec().free_spec()@.len(),
    {
        let free = self.rdma.window_pool.free.len();
        let cq_avail = self.rdma.cq.capacity_exec().saturating_sub(self.rdma.cq.in_flight_count());
        free.min(cq_avail)
    }

    /// Reply to a fetch with an explicit miss.
    pub fn reject_fetch_request(
        &mut self,
        core: &CacheCore,
        request_id: u64,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
        ensures
            final(self).well_formed(core),
            final(self).rdma_spec() == old(self).rdma_spec(),
            final(self).channel_spec().same_session(old(self).channel_spec()),
            final(self).pending_inbound_fetches_spec() == old(self).pending_inbound_fetches_spec(),
            final(self).channel_spec().sent() == old(self).channel_spec().sent(),
            final(self).channel_spec().received() == old(self).channel_spec().received(),
            final(self).next_grant_tag_spec() == old(self).next_grant_tag_spec(),
            final(self).next_request_id_spec() == old(self).next_request_id_spec(),
            final(self).active_grants_spec() == old(self).active_grants_spec(),
            final(self).pending_outbound_grants_spec() == old(self).pending_outbound_grants_spec(),
            final(self).pending_inbound_grants_spec() == old(self).pending_inbound_grants_spec(),
    {
        match self.send_control(core, ControlPayload::FetchMiss { request_id }, Tracked(instance)) {
            Ok(()) => Ok(()),
            Err(_) => Err(rdma_error(rdma::RdmaError(-1))),
        }
    }

    /// Pin the caller-selected published replica and post its Type-2 MW bind.
    /// Success queues a grant to transmit when the bind completes. An error
    /// leaves the admission decision with the caller.
    #[verifier::rlimit(250)]
    pub fn grant_fetch_from_frame(
        &mut self,
        core: &mut CacheCore,
        request_id: u64,
        key: ObjectKey,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma_spec().qp_spec().ready(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().peer_node_spec() == old(self).rdma_spec().peer_node_spec(),
            final(self).rdma_spec().qp_spec() == old(self).rdma_spec().qp_spec(),
            final(self).rdma_spec().window_pool_spec().pd() == old(
                self,
            ).rdma_spec().window_pool_spec().pd(),
            final(self).rdma_spec().host_mr_spec() == old(self).rdma_spec().host_mr_spec(),
            final(self).rdma_spec().gpu_mr_spec() == old(self).rdma_spec().gpu_mr_spec(),
            final(self).rdma_spec().shared_host_mr_spec() == old(
                self,
            ).rdma_spec().shared_host_mr_spec(),
            final(self).rdma_spec().shared_gpu_mr_spec() == old(
                self,
            ).rdma_spec().shared_gpu_mr_spec(),
            final(self).rdma_spec().pending() == old(self).rdma_spec().pending(),
            final(self).rdma_spec().completed_pulls_spec() == old(
                self,
            ).rdma_spec().completed_pulls_spec(),
            final(self).channel_spec() == old(self).channel_spec(),
            final(self).next_grant_tag_spec() == old(self).next_grant_tag_spec(),
            final(self).next_request_id_spec() == old(self).next_request_id_spec(),
            final(self).active_grants_spec() == old(self).active_grants_spec(),
            final(self).pending_inbound_grants_spec() == old(self).pending_inbound_grants_spec(),
            final(self).pending_inbound_fetches_spec() == old(self).pending_inbound_fetches_spec(),
            match r {
                Err(_) => final(self).pending_outbound_grants_spec() == old(
                    self,
                ).pending_outbound_grants_spec(),
                Ok(()) => {
                    let grants = final(self).pending_outbound_grants_spec()@;
                    &&& grants.len() == old(self).pending_outbound_grants_spec()@.len() + 1
                    &&& grants.last().request_id == request_id
                    &&& grants.last().key == key
                    &&& grants.last().frame == frame
                    &&& grants.last().send_on_ready
                },
            },
    {
        let mut reserved = None;
        self.grant_fetch_from_frame_with(
            core,
            request_id,
            key,
            frame,
            Tracked(shares),
            Tracked(instance),
            false,
            &mut reserved,
        )
    }

    #[verifier::rlimit(250)]
    fn grant_fetch_from_frame_with(
        &mut self,
        core: &mut CacheCore,
        request_id: u64,
        key: ObjectKey,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
        prepare: bool,
        reserved: &mut Option<PreparedLeaseBind>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(reserved).is_none(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(core).frames_spec().same_directory(&old(core).frames_spec()),
            bind_reservations_preserved(&old(self).rdma, &final(self).rdma),
            r.is_err() ==> final(reserved).is_none(),
            r.is_ok() ==> {
                &&& final(reserved).is_some() == prepare
                &&& (*final(reserved) matches Some(p) ==> p.frame == frame && p.valid(
                    &final(self).rdma,
                    final(core),
                ) && old(self).rdma.next_control_wr_id <= p.bind.id() && p.bind.id()
                    < final(self).rdma.next_control_wr_id)
            },
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.pending() == old(self).rdma.pending(),
            final(self).rdma.completed_pulls == old(self).rdma.completed_pulls,
            final(self).channel == old(self).channel,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            final(self).pending_inbound_fetches == old(self).pending_inbound_fetches,
            match r {
                Err(_) => final(self).pending_outbound_grants == old(self).pending_outbound_grants,
                Ok(()) => {
                    let grants = final(self).pending_outbound_grants@;
                    &&& grants.len() == old(self).pending_outbound_grants@.len() + 1
                    &&& grants.last().request_id == request_id
                    &&& grants.last().key == key
                    &&& grants.last().frame == frame
                    &&& grants.last().send_on_ready
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_core_wf_facts(core);
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        let idx = frame.0 as usize;
        if idx >= core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        let (frame_key, _) = core.frame_binding(frame).unwrap();
        if core.frame_meta(frame).unwrap().state != STATE_PUBLISHED || frame_key != key {
            return Err(CoreError::StaleReplica);
        }
        if self.next_grant_tag == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let version = VersionId::unpack(core.frame_meta(frame).unwrap().version_packed);
        proof {
            assert(core.frames_spec().valid_frame(frame, core.kernel_view())) by {
                reveal(CacheCore::well_formed);
            }
            lemma_version_pack_unpack(
                core.frames_spec().frame(frame).bound_version.unwrap(),
                core.frames_spec().meta(frame).version_packed,
            );
        }
        let bind = self.serve_bind_with(
            core,
            frame,
            Tracked(shares),
            Tracked(instance),
            prepare,
            reserved,
        );
        proof {
            reveal(FrameTable::same_directory);
        }
        let (handle, bind_wr_id) = bind?;
        proof {
            old(self).lemma_fresh_handle_excluded_from_session(old(core), handle);
            lemma_core_wf_facts(core);
            lemma_published_frame_replica_valid(core, frame);
            lemma_published_frame_slot_valid(core, frame);
            lemma_initializing_frame_slot_valid(core, frame);
            lemma_citations_closure(core);
        }
        let (_, frame_tier) = core.frame_binding(frame).unwrap();
        let frame_size = core.registered_tier_frame_size(frame_tier) as u64;
        let ghost cert = core.frames_spec().published(frame).unwrap().2;
        let ghost bind_sub = core.ledger_spec()@.submitted;
        let ghost bind_succ = core.ledger_spec()@.succeeded;
        let tracked bind_revoked = instance.duplicate_revoked(&core.borrow_ledger().revoked);
        let ghost bind_facts = core.citations();
        let Tracked(version_binding) = core.version_binding_for(version);
        proof {
            lemma_replica_valid_to_statement(core.ledger_spec()@, core.node(), cert);
        }
        let grant = PendingOutboundGrant {
            request_id,
            key,
            frame,
            version,
            frame_size,
            handle,
            bind_wr_id,
            send_on_ready: true,
            cert: Ghost(cert),
            bind_sub: Ghost(bind_sub),
            bind_succ: Ghost(bind_succ),
            bind_revoked: Tracked(bind_revoked),
            bind_facts: Ghost(bind_facts),
            version_binding: Tracked(version_binding),
        };
        let ghost before = *self;
        let ghost item = grant;
        self.pending_outbound_grants.push(grant);
        proof {
            Self::lemma_session_wf_after_push_outbound_grant(&before, self, core, &item);
        }
        Ok(())
    }

    /// Admit the batch on this peer's PD and ring one doorbell for its binds.
    /// Rejected entries get a miss. CQ processing sends grants and cleans up
    /// failed binds using the usual lease state machine.
    #[verifier::rlimit(80)]
    pub fn grant_fetch_batch(
        &mut self,
        core: &mut CacheCore,
        serve_tier: TierId,
        entries: &[(u64, ObjectKey)],
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma_spec().qp_spec().ready(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(self).rdma_spec().qp_spec() == old(self).rdma_spec().qp_spec(),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().peer_node_spec() == old(self).rdma_spec().peer_node_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(n) ==> n <= entries.len(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut binds: Vec<PreparedLeaseBind> = Vec::new();
        let mut i = 0;
        let mut reply_error = None;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        if entries.is_empty() || entries.len() > super::protocol::MAX_FETCH_BATCH {
            return Err(CoreError::InvalidArgument);
        }
        while i < entries.len()
            invariant
                binds.len() <= i <= entries.len(),
                self.well_formed(core),
                core.well_formed(),
                core.all_versions_bound(),
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                instance.id() == core.instance_id(),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.id() == core.instance_id(),
                shares.node() == old(shares).node() == core.node(),
                peer_core_step(old(core), core, self.rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                forall|j: int|
                    0 <= j < binds.len() ==> {
                        let b = #[trigger] binds[j];
                        b.valid(&self.rdma, core) && b.bind.id() < self.rdma.next_control_wr_id
                    },
                forall|j: int, k: int|
                    0 <= j < k < binds.len() ==> (#[trigger] binds[j]).bind.id() < (
                    #[trigger] binds[k]).bind.id(),
            decreases entries.len() - i,
        {
            let (request_id, key) = entries[i];
            let chosen = Self::lookup_serve_frame(core, key, serve_tier, None);
            let mut staged = None;
            if let Some((frame, _)) = chosen {
                let ghost before = *core;
                let ghost before_rdma = self.rdma;
                let _ = self.grant_fetch_from_frame_with(
                    core,
                    request_id,
                    key,
                    frame,
                    Tracked(shares),
                    Tracked(instance),
                    true,
                    &mut staged,
                );
                proof {
                    lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &before,
                        core,
                        old(self).rdma.active_leases@,
                        before_rdma.active_leases@,
                        self.rdma.active_leases@,
                    );
                    assert forall|j: int| 0 <= j < binds.len() implies {
                        let b = #[trigger] binds[j];
                        &&& b.valid(&self.rdma, core)
                        &&& b.bind.id() < self.rdma.next_control_wr_id
                        &&& (staged matches Some(p) ==> b.bind.id() < p.bind.id())
                    } by {
                        binds[j].lemma_preserved(&before_rdma, &self.rdma, &before, core);
                    }
                }
            }
            if let Some(bind) = staged {
                binds.push(bind);
            } else if let Err(e) = self.reject_fetch_request(core, request_id, Tracked(instance)) {
                reply_error = Some(e);
            }
            i += 1;
        }
        let (posted, _) = self.rdma.post_prepared_lease_binds(core, binds);
        if let Some(e) = reply_error {
            Err(e)
        } else {
            Ok(posted)
        }
    }

    /// Apply the built-in tier lookup policy used by the legacy duplex APIs.
    #[verifier::rlimit(250)]
    fn queue_pending_inbound_fetches(
        &mut self,
        core: &mut CacheCore,
        primary_tier: TierId,
        fallback_tier: Option<TierId>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).rdma.pending() == old(self).rdma.pending(),
            final(self).rdma.completed_pulls == old(self).rdma.completed_pulls,
            final(self).channel.same_session(old(self).channel),
            final(self).channel.received() == old(self).channel.received(),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants == old(self).active_grants,
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while self.pending_inbound_fetches.len() > 0
            invariant
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                instance.id() == core.instance_id(),
                shares.well_formed(),
                shares.id() == core.instance_id(),
                shares.node() == core.node(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.rdma.pending() == old(self).rdma.pending(),
                self.rdma.completed_pulls == old(self).rdma.completed_pulls,
                self.channel.same_session(old(self).channel),
                self.channel.received() == old(self).channel.received(),
                self.next_grant_tag == old(self).next_grant_tag,
                self.next_request_id == old(self).next_request_id,
                self.active_grants == old(self).active_grants,
                self.pending_inbound_grants == old(self).pending_inbound_grants,
            decreases self.pending_inbound_fetches@.len(),
        {
            let (request_id, key) = self.pending_inbound_fetches.remove(0);
            let grant_err = if let Some((frame, _)) = Self::lookup_serve_frame(
                core,
                key,
                primary_tier,
                fallback_tier,
            ) {
                let ghost before = *core;
                let ghost before_leases = self.rdma.active_leases@;
                let result = self.grant_fetch_from_frame(
                    core,
                    request_id,
                    key,
                    frame,
                    Tracked(shares),
                    Tracked(instance),
                );
                proof {
                    lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &before,
                        core,
                        old(self).rdma.active_leases@,
                        before_leases,
                        self.rdma.active_leases@,
                    );
                }
                result.is_err()
            } else {
                true
            };
            if grant_err {
                let _ = self.reject_fetch_request(core, request_id, Tracked(instance));
            }
        }
        proof {
            reveal(pull_tracked_or_done_state);
        }
    }

    proof fn lemma_peer_wf_lease_facts(&self, core: &CacheCore)
        requires
            self.well_formed(core),
        ensures
            self.rdma.well_formed(core),
            rdma_leases_disjoint(self.rdma.active_leases@),
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::session_queues_valid);
        reveal(PeerSession::well_formed);
        reveal(RdmaBackend::well_formed);
        reveal(rdma_leases_valid);
    }

    proof fn lemma_pop_outbound_grant(
        before: &Self,
        after: &Self,
        core: &CacheCore,
        i: int,
        p: &PendingOutboundGrant,
    )
        requires
            before.well_formed(core),
            0 <= i < before.pending_outbound_grants@.len(),
            *p == before.pending_outbound_grants@[i],
            after.pending_outbound_grants@ == before.pending_outbound_grants@.update(
                i,
                before.pending_outbound_grants@.last(),
            ).drop_last(),
            after.rdma == before.rdma,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
        ensures
            after.well_formed(core),
            after.rdma.well_formed(core),
            forall|k: int|
                0 <= k < after.pending_outbound_grants@.len() ==> (
                #[trigger] after.pending_outbound_grants@[k]).handle != p.handle,
            forall|j: int|
                0 <= j < after.active_grants@.len() ==> (#[trigger] after.active_grants@[j]).handle
                    != p.handle,
            after.rdma.lease_index_spec(p.handle) matches Some(idx) && {
                let lease = after.rdma.active_leases@[idx];
                &&& lease.frame == p.frame
                &&& lease.wr_id == p.bind_wr_id
                &&& super::poll::rdma_lease_is_binding_like(lease)
                &&& (lease.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(
                    p.cert@.writer,
                ))
            },
            p.frame.0 < core.frames_spec().metas_spec().len(),
            p.bind_revoked@.instance_id() == core.instance_id(),
            p.bind_sub@.map().submap_of(core.ledger_spec()@.submitted.map()),
            p.bind_succ@.dom().subset_of(core.ledger_spec()@.succeeded.dom()),
            forall|e: TransferEvent| #[trigger]
                p.bind_facts@.contains(e) ==> core.ledger_spec()@.submitted.map().contains_key(e)
                    && core.ledger_spec()@.knows.set().contains((core.node(), e)),
            p.bind_facts@.contains(p.cert@.writer),
            p.cert@.slot == core.frames_spec().frame(p.frame).slot,
            p.cert@.slot.is_full(),
            p.cert@.slot.well_formed(),
            p.cert@.bytes.len() == core.frames_spec().tier_frame_size(
                core.frames_spec().frame(p.frame).tier,
            ) as u64,
            super::protocol::statement_replica_valid(
                p.bind_sub@,
                p.bind_succ@,
                p.bind_revoked@,
                p.bind_facts@,
                p.cert@,
            ),
            p.version_binding@ matches Some(vb) && vb.instance_id() == cluster_version_instance_id()
                && vb.key() == (p.version.origin_node as NodeId, p.version.seq) && vb.value()
                == p.cert@.bytes,
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingInboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        let old_q = before.pending_outbound_grants@;
        let new_q = after.pending_outbound_grants@;
        let last = (old_q.len() - 1) as int;
        assert(before.pending_outbound_grants@[i].valid(&before.rdma, core));
        assert forall|k: int| 0 <= k < new_q.len() implies {
            &&& (#[trigger] new_q[k]).valid(&after.rdma, core)
            &&& new_q[k].handle != p.handle
        } by {
            let orig_k = if k == i {
                last
            } else {
                k
            };
            assert(new_q[k] == old_q[orig_k]);
            assert(old_q[orig_k].valid(&before.rdma, core));
            if orig_k < i {
                assert(old_q[orig_k].handle != old_q[i].handle);
            } else {
                assert(i < orig_k);
                assert(old_q[i].handle != old_q[orig_k].handle);
            }
        }
        assert forall|k1: int, k2: int| 0 <= k1 < k2 < new_q.len() implies (
        #[trigger] new_q[k1]).handle != (#[trigger] new_q[k2]).handle by {
            let o1 = if k1 == i {
                last
            } else {
                k1
            };
            let o2 = if k2 == i {
                last
            } else {
                k2
            };
            assert(new_q[k1] == old_q[o1]);
            assert(new_q[k2] == old_q[o2]);
            if o1 < o2 {
                assert(old_q[o1].handle != old_q[o2].handle);
            } else {
                assert(o2 < o1);
                assert(old_q[o2].handle != old_q[o1].handle);
            }
        }
        let pidx = after.rdma.lease_index_spec(p.handle).unwrap();
        assert forall|j: int| 0 <= j < after.active_grants@.len() implies (
        #[trigger] after.active_grants@[j]).handle != p.handle by {
            let g = before.active_grants@[j];
            assert(after.active_grants@[j] == g);
            if g.handle == p.handle {
                assert(after.rdma.active_leases@[pidx].state == RemoteLeaseState::Bound);
            }
        }
    }

    proof fn lemma_session_wf_after_outbound_lease_update(
        before: &Self,
        after: &Self,
        before_core: &CacheCore,
        after_core: &CacheCore,
        handle: RemoteLeaseHandle,
        hidx: int,
    )
        requires
            before.well_formed(before_core),
            after.rdma.well_formed(after_core),
            after_core.node_id_spec() == before_core.node_id_spec(),
            after_core.instance_id() == before_core.instance_id(),
            after_core.frames_spec().same_directory(&before_core.frames_spec()),
            *after_core == *before_core || peer_core_step(
                before_core,
                after_core,
                before.rdma.timeline_id,
            ),
            after.rdma.peer_node == before.rdma.peer_node,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
            0 <= hidx < before.rdma.active_leases@.len(),
            before.rdma.lease_index_spec(handle) == Some(hidx),
            after.rdma.active_leases@.len() == before.rdma.active_leases@.len(),
            forall|j: int|
                0 <= j < before.rdma.active_leases@.len() && j != hidx ==> (
                #[trigger] after.rdma.active_leases@[j]) == before.rdma.active_leases@[j],
            forall|j: int|
                0 <= j < before.active_grants@.len() ==> (
                #[trigger] before.active_grants@[j]).handle != handle,
            forall|k: int|
                0 <= k < before.pending_outbound_grants@.len() ==> (
                #[trigger] before.pending_outbound_grants@[k]).handle != handle,
        ensures
            after.well_formed(after_core),
            after_core.frames_spec().versions_spec()@ == before_core.frames_spec().versions_spec()@,
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(super::protocol::statement_replica_valid);
        reveal(peer_core_step_except);
        reveal(core_transition);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        reveal(PendingInboundGrant::valid);
        assert(rdma_leases_disjoint(before.rdma.active_leases@)) by {
            reveal(RdmaBackend::well_formed);
            reveal(rdma_leases_valid);
        }
        assert(rdma_leases_disjoint(after.rdma.active_leases@)) by {
            reveal(RdmaBackend::well_formed);
            reveal(rdma_leases_valid);
        }
        assert(after_core.frames_spec().same_layout(&before_core.frames_spec())) by {
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
        }
        let pre_leases = before.rdma.active_leases@;
        let post_leases = after.rdma.active_leases@;
        lemma_peer_grants_after_lease_update(
            after.active_grants@,
            pre_leases,
            post_leases,
            handle,
            hidx,
            after.channel.sent(),
            after.next_grant_tag,
            before_core,
            after_core,
        );
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies {
            let q = #[trigger] after.pending_outbound_grants@[k];
            &&& rdma_lease_index_spec(pre_leases, q.handle) matches Some(bidx) && {
                let bl = pre_leases[bidx];
                &&& rdma_lease_index_spec(post_leases, q.handle) matches Some(aidx)
                &&& post_leases[aidx].frame == bl.frame
                &&& post_leases[aidx].wr_id == bl.wr_id
                &&& super::poll::rdma_lease_is_binding_like(post_leases[aidx])
                &&& post_leases[aidx].share_cap == bl.share_cap
            }
        } by {
            let q = after.pending_outbound_grants@[k];
            assert(before.pending_outbound_grants@[k] == q);
            assert(q.valid(&before.rdma, before_core));
            let bidx = rdma_lease_index_spec(pre_leases, q.handle).unwrap();
            if bidx == hidx {
                reveal(rdma_leases_disjoint);
                assert(false);
            }
            lemma_lease_index_preserved_update(pre_leases, post_leases, hidx, q.handle, bidx);
        }
        lemma_outbound_grants_preserved(
            after.pending_outbound_grants@,
            pre_leases,
            post_leases,
            before_core,
            after_core,
        );
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[k]).valid(&after.rdma, after_core) by {
            assert(before.pending_outbound_grants@[k] == after.pending_outbound_grants@[k]);
        }
        assert forall|k: int| 0 <= k < after.pending_inbound_grants@.len() implies (
        #[trigger] after.pending_inbound_grants@[k]).valid(
            after_core,
            &after.channel,
            after.rdma.peer_node,
        ) by {
            let g = before.pending_inbound_grants@[k];
            assert(after.pending_inbound_grants@[k] == g);
            assert(g.valid(before_core, &before.channel, before.rdma.peer_node));
        }
    }

    proof fn lemma_session_wf_same_active_and_queues(before: &Self, after: &Self, core: &CacheCore)
        requires
            before.well_formed(core),
            after.rdma == before.rdma,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
        ensures
            after.well_formed(core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
    }

    proof fn lemma_session_wf_same_leases(before: &Self, after: &Self, core: &CacheCore)
        requires
            before.well_formed(core),
            after.rdma.well_formed(core),
            after.rdma.peer_node == before.rdma.peer_node,
            after.rdma.active_leases@ == before.rdma.active_leases@,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
        ensures
            after.well_formed(core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingInboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[k]).valid(&after.rdma, core) by {
            let p = before.pending_outbound_grants@[k];
            assert(after.pending_outbound_grants@[k] == p);
            assert(p.valid(&before.rdma, core));
        }
    }

    proof fn lemma_active_grants_exclude_nonbound_handle(
        &self,
        core: &CacheCore,
        handle: RemoteLeaseHandle,
        hidx: int,
    )
        requires
            self.well_formed(core),
            self.rdma.lease_index_spec(handle) == Some(hidx),
            0 <= hidx < self.rdma.active_leases@.len(),
            self.rdma.active_leases@[hidx].state != RemoteLeaseState::Bound,
        ensures
            forall|j: int|
                0 <= j < self.active_grants@.len() ==> (#[trigger] self.active_grants@[j]).handle
                    != handle,
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        reveal(PeerSession::well_formed);
        assert forall|j: int| 0 <= j < self.active_grants@.len() implies (
        #[trigger] self.active_grants@[j]).handle != handle by {
            let g = self.active_grants@[j];
            if g.handle == handle {
                assert(self.rdma.active_leases@[hidx].state == RemoteLeaseState::Bound);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    proof fn lemma_read_grant_attachment_valid(
        &self,
        core: &CacheCore,
        grant_tag: GrantTag,
        frame: FrameId,
        frame_tier: TierId,
        meta: ReadGrantMeta,
        meta_bytes: Seq<u8>,
        cap: rdma::RemoteCapability,
        share_cap: ShareCap,
        cert: ValueCertificate,
        bind_sub: SubmittedMap,
        bind_succ: SucceededMap,
        revoked: RevokedMap,
        bind_facts: ISet<TransferEvent>,
        att: VeriCacheAttachment,
        cited: ISet<TransferEvent>,
    )
        requires
            core.well_formed(),
            self.well_formed(core),
            grant_tag == self.next_grant_tag,
            frame.0 < core.frames_spec().metas_spec().len(),
            frame_tier == core.frames_spec().bindings_spec()[frame.0 as int].1,
            meta.frame_size == core.frames_spec().tier_frame_size(frame_tier) as u64,
            meta_bytes == super::protocol::encode_read_grant_meta_spec(meta),
            cited == core.citations(),
            cap.well_formed(),
            cap.slot_ref() == core.frames_spec().frame(frame).slot,
            cap.access() == rdma::RemoteAccess::ReadOnly,
            cap.share() == Some(share_cap.share()),
            share_cap.is_fenced(),
            share_cap.instance_id() == core.instance_id(),
            share_cap.loc() == cap.slot_ref().loc,
            share_cap.frontier().last_writer == Some(cert.writer),
            share_cap.frontier().readers == ISet::<TransferEvent>::empty(),
            revoked.instance_id() == core.instance_id(),
            bind_sub.map().submap_of(core.ledger_spec()@.submitted.map()),
            bind_succ.dom().subset_of(core.ledger_spec()@.succeeded.dom()),
            forall|e: TransferEvent| #[trigger]
                bind_facts.contains(e) ==> core.ledger_spec()@.submitted.map().contains_key(e)
                    && core.ledger_spec()@.knows.set().contains((core.node(), e)),
            bind_facts.contains(cert.writer),
            cert.slot == core.frames_spec().frame(frame).slot,
            cert.slot.is_full(),
            cert.slot.well_formed(),
            cert.bytes.len() == core.frames_spec().tier_frame_size(
                core.frames_spec().frame(frame).tier,
            ) as u64,
            super::protocol::statement_replica_valid(
                bind_sub,
                bind_succ,
                revoked,
                bind_facts,
                cert,
            ),
            att matches VeriCacheAttachment::ReadGrant {
                meta: am,
                cert: ac,
                revoked: ar,
                version_binding: avb,
            } && am == meta && ac == cert && ar == revoked && (avb matches Some(vb)
                && vb.instance_id() == cluster_version_instance_id() && vb.key() == (
                meta.version.origin_node as NodeId,
                meta.version.seq,
            ) && vb.value() == cert.bytes),
        ensures
            self.channel.id() == core.instance_id(),
            self.channel.local() == core.node(),
            self.channel.peer() == self.rdma.peer_node,
            !self.channel.sent().contains_key(grant_tag),
            cap.slot_ref().loc.node == self.channel.local(),
            cited.subset_of(core.ledger_spec()@.submitted.dom()),
            knowledge_of(self.channel.local(), cited).subset_of(core.ledger_spec()@.knows.set()),
            ordered_or_absent(
                share_cap.frontier().last_writer,
                closure_from(core.ledger_spec()@.submitted.map(), cited),
            ),
            forall|s: Statement|
                s.submitted.instance_id() == self.channel.id() && s.submitted.map()
                    == core.ledger_spec()@.submitted.map() && s.knows.instance_id()
                    == self.channel.id() && s.knows.set() == core.ledger_spec()@.knows.set()
                    && s.succeeded == core.ledger_spec()@.succeeded && s.facts == cited
                    ==> #[trigger] att.valid(
                    self.channel.id(),
                    self.channel.local(),
                    1,
                    grant_tag,
                    meta_bytes,
                    Some(
                        GrantKind::ReadShare { slot: cap.slot_ref(), share_id: share_cap.share() },
                    ),
                    Some(cap),
                    Some(share_cap.frontier()),
                    s,
                ),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        reveal(PeerSession::well_formed);
        lemma_core_wf_facts(core);
        lemma_initializing_frame_slot_valid(core, frame);
        lemma_citations_closure(core);
        assert(core.frames_spec().bindings_spec()[frame.0 as int].1 == core.frames_spec().frame(
            frame,
        ).tier);
        assert(meta.frame_size == cert.bytes.len());
        lemma_statement_replica_valid_monotone(
            bind_sub,
            core.ledger_spec()@.submitted,
            bind_succ,
            core.ledger_spec()@.succeeded,
            revoked,
            bind_facts,
            cited,
            cert,
        );
        assert(closure_from(core.ledger_spec()@.submitted.map(), cited).contains(cert.writer));
        reveal(complete_receipts);
        assert forall|s: Statement|
            s.submitted.instance_id() == self.channel.id() && s.submitted.map()
                == core.ledger_spec()@.submitted.map() && s.knows.instance_id() == self.channel.id()
                && s.knows.set() == core.ledger_spec()@.knows.set() && s.succeeded
                == core.ledger_spec()@.succeeded && s.facts == cited implies #[trigger] att.valid(
            self.channel.id(),
            self.channel.local(),
            1,
            grant_tag,
            meta_bytes,
            Some(GrantKind::ReadShare { slot: cap.slot_ref(), share_id: share_cap.share() }),
            Some(cap),
            Some(share_cap.frontier()),
            s,
        ) by {
            assert(s.submitted.dom() =~= core.ledger_spec()@.submitted.dom());
            assert(crate::provenance::complete_receipts_map(s.submitted));
            super::protocol::lemma_statement_replica_valid_ext(
                core.ledger_spec()@.submitted,
                s.submitted,
                core.ledger_spec()@.succeeded,
                revoked,
                cited,
                cert,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    proof fn lemma_session_wf_after_grant_sent(
        before: &Self,
        after: &Self,
        core: &CacheCore,
        grant_tag: GrantTag,
        handle: RemoteLeaseHandle,
        hidx: int,
        frame: FrameId,
        cap_slot: SlotRef,
        share_id: TransferShare,
    )
        requires
            before.well_formed(core),
            grant_tag == before.next_grant_tag,
            before.next_grant_tag < u64::MAX,
            after.next_grant_tag == before.next_grant_tag + 1,
            after.rdma == before.rdma,
            after.channel.same_session(before.channel),
            after.channel.sent() == before.channel.sent().insert(
                grant_tag,
                GrantKind::ReadShare { slot: cap_slot, share_id },
            ),
            after.channel.received() == before.channel.received(),
            after.active_grants@ == before.active_grants@.push(
                ActiveGrant { grant_tag, handle, frame, soft_revoking: false },
            ),
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
            frame.0 < core.frames_spec().metas_spec().len(),
            0 <= hidx < before.rdma.active_leases@.len(),
            before.rdma.lease_index_spec(handle) == Some(hidx),
            before.rdma.active_leases@[hidx].frame == frame,
            before.rdma.active_leases@[hidx].state == RemoteLeaseState::Bound,
            before.rdma.active_leases@[hidx].share@ == share_id,
            cap_slot == core.frames_spec().frame(frame).slot,
            forall|j: int|
                0 <= j < before.active_grants@.len() ==> (
                #[trigger] before.active_grants@[j]).handle != handle,
        ensures
            after.well_formed(core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingOutboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingInboundGrant::valid);
        let n = before.active_grants@.len() as int;
        assert forall|t: GrantTag| #[trigger] after.channel.sent().contains_key(t) implies t
            < after.next_grant_tag by {
            if t != grant_tag {
                assert(before.channel.sent().contains_key(t));
            }
        }
        assert forall|i: int, j: int| 0 <= i < j < after.active_grants@.len() implies (
        #[trigger] after.active_grants@[i]).grant_tag != (
        #[trigger] after.active_grants@[j]).grant_tag && after.active_grants@[i].handle
            != after.active_grants@[j].handle by {
            if j == n {
                assert(after.active_grants@[i] == before.active_grants@[i]);
                assert(before.active_grants@[i].grant_tag < before.next_grant_tag);
            } else {
                assert(after.active_grants@[i] == before.active_grants@[i]);
                assert(after.active_grants@[j] == before.active_grants@[j]);
            }
        }
        assert forall|i: int| 0 <= i < after.active_grants@.len() implies {
            let g = #[trigger] after.active_grants@[i];
            &&& g.grant_tag < after.next_grant_tag
            &&& g.frame.0 < core.frames_spec().metas_spec().len()
            &&& after.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                let lease = after.rdma.active_leases@[idx];
                &&& lease.frame == g.frame
                &&& lease.state == RemoteLeaseState::Bound
                &&& after.channel.sent().contains_key(g.grant_tag)
                &&& after.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                    slot: core.frames_spec().frame(g.frame).slot,
                    share_id: lease.share@,
                })
            }
        } by {
            if i < n {
                let g = before.active_grants@[i];
                assert(after.active_grants@[i] == g);
                assert(g.grant_tag != grant_tag);
            }
        }
        assert forall|k: int| 0 <= k < after.pending_inbound_grants@.len() implies (
        #[trigger] after.pending_inbound_grants@[k]).valid(
            core,
            &after.channel,
            after.rdma.peer_node,
        ) by {
            let g = before.pending_inbound_grants@[k];
            assert(after.pending_inbound_grants@[k] == g);
            assert(g.valid(core, &before.channel, before.rdma.peer_node));
        }
    }

    proof fn lemma_session_wf_after_send_transport_err(
        before: &Self,
        after: &Self,
        core: &CacheCore,
    )
        requires
            before.well_formed(core),
            after.rdma == before.rdma,
            after.channel.same_session(before.channel),
            after.channel.sent() == before.channel.sent(),
            after.channel.received() == before.channel.received(),
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
        ensures
            after.well_formed(core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingOutboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingInboundGrant::valid);
        assert forall|k: int| 0 <= k < after.pending_inbound_grants@.len() implies (
        #[trigger] after.pending_inbound_grants@[k]).valid(
            core,
            &after.channel,
            after.rdma.peer_node,
        ) by {
            let g = before.pending_inbound_grants@[k];
            assert(after.pending_inbound_grants@[k] == g);
            assert(g.valid(core, &before.channel, before.rdma.peer_node));
        }
    }

    proof fn lemma_fresh_handle_excluded_from_session(
        &self,
        core: &CacheCore,
        handle: RemoteLeaseHandle,
    )
        requires
            self.well_formed(core),
            self.rdma.lease_index_spec(handle) is None,
        ensures
            forall|j: int|
                0 <= j < self.active_grants@.len() ==> (#[trigger] self.active_grants@[j]).handle
                    != handle,
            forall|k: int|
                0 <= k < self.pending_outbound_grants@.len() ==> (
                #[trigger] self.pending_outbound_grants@[k]).handle != handle,
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingInboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        assert forall|j: int| 0 <= j < self.active_grants@.len() implies (
        #[trigger] self.active_grants@[j]).handle != handle by {
            let g = self.active_grants@[j];
            if g.handle == handle {
                assert(self.rdma.lease_index_spec(g.handle) is Some);
            }
        }
        assert forall|k: int| 0 <= k < self.pending_outbound_grants@.len() implies (
        #[trigger] self.pending_outbound_grants@[k]).handle != handle by {
            let p = self.pending_outbound_grants@[k];
            assert(p.valid(&self.rdma, core));
            if p.handle == handle {
                assert(self.rdma.lease_index_spec(p.handle) is Some);
            }
        }
    }

    proof fn lemma_session_wf_after_push_outbound_grant(
        before: &Self,
        after: &Self,
        core: &CacheCore,
        item: &PendingOutboundGrant,
    )
        requires
            core.well_formed(),
            before.well_formed(core),
            after.rdma == before.rdma,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
            after.pending_outbound_grants@ == before.pending_outbound_grants@.push(*item),
            forall|k: int|
                0 <= k < before.pending_outbound_grants@.len() ==> (
                #[trigger] before.pending_outbound_grants@[k]).handle != item.handle,
            item.frame.0 < core.frames_spec().metas_spec().len(),
            item.frame_size == core.frames_spec().tier_frame_size(
                core.frames_spec().frame(item.frame).tier,
            ) as u64,
            item.cert@.slot == core.frames_spec().frame(item.frame).slot,
            item.cert@.slot.is_full(),
            item.cert@.slot.well_formed(),
            item.cert@.bytes.len() == item.frame_size,
            item.bind_revoked@.instance_id() == core.instance_id(),
            item.bind_sub@.map().submap_of(core.ledger_spec()@.submitted.map()),
            item.bind_succ@.dom().subset_of(core.ledger_spec()@.succeeded.dom()),
            forall|e: TransferEvent| #[trigger]
                item.bind_facts@.contains(e) ==> core.ledger_spec()@.submitted.map().contains_key(e)
                    && core.ledger_spec()@.knows.set().contains((core.node(), e)),
            item.bind_facts@.contains(item.cert@.writer),
            super::protocol::statement_replica_valid(
                item.bind_sub@,
                item.bind_succ@,
                item.bind_revoked@,
                item.bind_facts@,
                item.cert@,
            ),
            item.version_binding@ matches Some(vb) && vb.instance_id()
                == cluster_version_instance_id() && vb.key() == (
                item.version.origin_node as NodeId,
                item.version.seq,
            ) && vb.value() == item.cert@.bytes,
            after.rdma.lease_index_spec(item.handle) matches Some(idx) && {
                let l = after.rdma.active_leases@[idx];
                &&& l.frame == item.frame
                &&& l.wr_id == item.bind_wr_id
                &&& super::poll::rdma_lease_is_binding_like(l)
                &&& l.share_cap matches Some(sc) && sc@.frontier().last_writer == Some(
                    item.cert@.writer,
                )
            },
        ensures
            after.well_formed(core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingInboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        let n = before.pending_outbound_grants@.len() as int;
        assert(item.valid(&after.rdma, core));
        assert forall|i: int, j: int| 0 <= i < j < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[i]).handle != (
        #[trigger] after.pending_outbound_grants@[j]).handle by {
            if j == n {
                assert(after.pending_outbound_grants@[i] == before.pending_outbound_grants@[i]);
            } else {
                assert(after.pending_outbound_grants@[i] == before.pending_outbound_grants@[i]);
                assert(after.pending_outbound_grants@[j] == before.pending_outbound_grants@[j]);
            }
        }
        assert forall|i: int| 0 <= i < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[i]).valid(&after.rdma, core) by {
            if i < n {
                assert(after.pending_outbound_grants@[i] == before.pending_outbound_grants@[i]);
            }
        }
    }

    proof fn lemma_session_wf_after_release_lookup_miss(
        &self,
        core: &CacheCore,
        grant_tag: GrantTag,
    )
        requires
            self.rdma.well_formed(core),
            self.channel.local() == core.node(),
            self.channel.peer() == self.rdma.peer_node,
            self.channel.id() == core.instance_id(),
            self.session_queues_valid(core),
            forall|t: GrantTag| #[trigger]
                self.channel.sent().contains_key(t) ==> t < self.next_grant_tag,
            forall|i: int, j: int|
                0 <= i < j < self.active_grants@.len() ==> (
                #[trigger] self.active_grants@[i]).grant_tag != (
                #[trigger] self.active_grants@[j]).grant_tag && self.active_grants@[i].handle
                    != self.active_grants@[j].handle,
            forall|i: int|
                0 <= i < self.active_grants@.len() ==> (#[trigger] self.active_grants@[i]).grant_tag
                    != grant_tag,
            forall|i: int|
                0 <= i < self.active_grants@.len() ==> {
                    let g = #[trigger] self.active_grants@[i];
                    &&& g.grant_tag < self.next_grant_tag
                    &&& g.frame.0 < core.frames_spec().metas_spec().len()
                    &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                        let lease = self.rdma.active_leases@[idx];
                        &&& lease.frame == g.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& (g.grant_tag != grant_tag ==> self.channel.sent().contains_key(
                            g.grant_tag,
                        ) && self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                            slot: core.frames_spec().frame(g.frame).slot,
                            share_id: lease.share@,
                        }))
                    }
                },
        ensures
            self.well_formed(core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        reveal(PeerSession::well_formed);
        assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
            let g = #[trigger] self.active_grants@[i];
            &&& g.grant_tag < self.next_grant_tag
            &&& g.frame.0 < core.frames_spec().metas_spec().len()
            &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                let lease = self.rdma.active_leases@[idx];
                &&& lease.frame == g.frame
                &&& lease.state == RemoteLeaseState::Bound
                &&& self.channel.sent().contains_key(g.grant_tag)
                &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                    slot: core.frames_spec().frame(g.frame).slot,
                    share_id: lease.share@,
                })
            }
        } by {
            let g = self.active_grants@[i];
            assert(g.grant_tag != grant_tag);
        }
    }

    #[allow(clippy::too_many_arguments)]
    proof fn lemma_session_wf_after_release_swap_remove(
        before: &Self,
        after: &Self,
        core: &CacheCore,
        grant_tag: GrantTag,
        pos: int,
        grant: ActiveGrant,
        share_cap: ShareCap,
        stmt_closure: ISet<TransferEvent>,
    )
        requires
            before.rdma.well_formed(core),
            before.channel.local() == core.node(),
            before.channel.peer() == before.rdma.peer_node,
            before.channel.id() == core.instance_id(),
            before.session_queues_valid(core),
            forall|t: GrantTag| #[trigger]
                before.channel.sent().contains_key(t) ==> t < before.next_grant_tag,
            forall|i: int, j: int|
                0 <= i < j < before.active_grants@.len() ==> (
                #[trigger] before.active_grants@[i]).grant_tag != (
                #[trigger] before.active_grants@[j]).grant_tag && before.active_grants@[i].handle
                    != before.active_grants@[j].handle,
            forall|i: int|
                0 <= i < before.active_grants@.len() ==> {
                    let g = #[trigger] before.active_grants@[i];
                    &&& g.grant_tag < before.next_grant_tag
                    &&& g.frame.0 < core.frames_spec().metas_spec().len()
                    &&& before.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                        let lease = before.rdma.active_leases@[idx];
                        &&& lease.frame == g.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& (g.grant_tag != grant_tag ==> before.channel.sent().contains_key(
                            g.grant_tag,
                        ) && before.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                            slot: core.frames_spec().frame(g.frame).slot,
                            share_id: lease.share@,
                        }))
                        &&& (g.grant_tag == grant_tag ==> share_cap.is_fenced()
                            && share_cap.instance_id() == core.instance_id() && share_cap.loc()
                            == core.frames_spec().frame(g.frame).slot.loc && share_cap.share()
                            == lease.share@ && transfer::transport::covers(
                            share_cap.frontier(),
                            stmt_closure,
                        ))
                    }
                },
            0 <= pos < before.active_grants@.len(),
            grant == before.active_grants@[pos],
            grant.grant_tag == grant_tag,
            after.rdma == before.rdma,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants@ == before.active_grants@.update(
                pos,
                before.active_grants@.last(),
            ).drop_last(),
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
        ensures
            after.well_formed(core),
            after.rdma.well_formed(core),
            grant.frame.0 < core.frames_spec().metas_spec().len(),
            after.rdma.lease_index_spec(grant.handle) matches Some(idx) && {
                let lease = after.rdma.active_leases@[idx];
                &&& 0 <= idx < after.rdma.active_leases@.len()
                &&& lease.frame == grant.frame
                &&& lease.state == RemoteLeaseState::Bound
                &&& lease.share@ == share_cap.share()
            },
            share_cap.is_fenced(),
            share_cap.instance_id() == core.instance_id(),
            share_cap.loc() == core.frames_spec().frame(grant.frame).slot.loc,
            transfer::transport::covers(share_cap.frontier(), stmt_closure),
            forall|j: int|
                0 <= j < after.active_grants@.len() ==> (#[trigger] after.active_grants@[j]).handle
                    != grant.handle,
            forall|k: int|
                0 <= k < after.pending_outbound_grants@.len() ==> (
                #[trigger] after.pending_outbound_grants@[k]).handle != grant.handle,
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingInboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        lemma_swap_removed_grant(before.active_grants@, after.active_grants@, pos);
        let hidx = before.rdma.lease_index_spec(grant.handle).unwrap();
        assert(0 <= hidx < before.rdma.active_leases@.len()) by {
            reveal(RdmaBackend::well_formed);
            reveal(rdma_leases_valid);
        };
        assert forall|i: int| 0 <= i < after.active_grants@.len() implies {
            let g = #[trigger] after.active_grants@[i];
            &&& g.grant_tag < after.next_grant_tag
            &&& g.frame.0 < core.frames_spec().metas_spec().len()
            &&& after.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                let lease = after.rdma.active_leases@[idx];
                &&& lease.frame == g.frame
                &&& lease.state == RemoteLeaseState::Bound
                &&& after.channel.sent().contains_key(g.grant_tag)
                &&& after.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                    slot: core.frames_spec().frame(g.frame).slot,
                    share_id: lease.share@,
                })
            }
        } by {
            let orig_i = if i == pos {
                (before.active_grants@.len() - 1) as int
            } else {
                i
            };
            assert(after.active_grants@[i] == before.active_grants@[orig_i]);
            assert(after.active_grants@[i].grant_tag != grant_tag);
        }
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[k]).handle != grant.handle by {
            let p = before.pending_outbound_grants@[k];
            assert(after.pending_outbound_grants@[k] == p);
            assert(p.valid(&before.rdma, core));
            if p.handle == grant.handle {
                assert(super::poll::rdma_lease_is_binding_like(before.rdma.active_leases@[hidx]));
            }
        }
    }

    proof fn lemma_session_wf_after_merge_statement(
        &self,
        before_core: &CacheCore,
        after_core: &CacheCore,
        stmt_sub: SubmittedMap,
        stmt_facts: ISet<TransferEvent>,
        stmt_closure: ISet<TransferEvent>,
    )
        requires
            before_core.well_formed(),
            after_core.well_formed(),
            self.well_formed(before_core),
            after_core.node_id_spec() == before_core.node_id_spec(),
            after_core.instance_id() == before_core.instance_id(),
            after_core.tiers_spec() == before_core.tiers_spec(),
            after_core.frames_spec() == before_core.frames_spec(),
            after_core.catalog_spec() == before_core.catalog_spec(),
            after_core.timelines_spec() == before_core.timelines_spec(),
            after_core.active_leases_spec() == before_core.active_leases_spec(),
            after_core.free_lease_slots_spec() == before_core.free_lease_slots_spec(),
            after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
            after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            after_core.version_registry_spec() == before_core.version_registry_spec(),
            after_core.spec_spec() == before_core.spec_spec(),
            before_core.ledger_spec()@.submitted.map().submap_of(
                after_core.ledger_spec()@.submitted.map(),
            ),
            stmt_sub.map().submap_of(after_core.ledger_spec()@.submitted.map()),
            before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
            knowledge_of(before_core.node(), stmt_closure).subset_of(
                after_core.ledger_spec()@.knows.set(),
            ),
            before_core.ledger_spec()@.succeeded.dom().subset_of(
                after_core.ledger_spec()@.succeeded.dom(),
            ),
            after_core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
            crate::provenance::complete_receipts_map(stmt_sub),
            stmt_facts.subset_of(stmt_sub.dom()),
            stmt_closure == closure_from(stmt_sub.map(), stmt_facts),
        ensures
            self.well_formed(after_core),
            self.rdma.well_formed(after_core),
            peer_core_step(before_core, after_core, self.rdma.timeline_id),
            stmt_closure.subset_of(after_core.citations()),
            after_core.timelines_spec().valid_ticket(Ticket::NONE),
    {
        hide(CacheCore::well_formed);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(RdmaBackend::well_formed);
        reveal(PendingOutboundGrant::valid);
        reveal(PendingInboundGrant::valid);
        lemma_core_wf_facts(after_core);
        lemma_core_timelines_wf(after_core);
        lemma_rdma_window_after_fence_ledger(
            self.rdma.in_flight_ops,
            self.rdma.bitmap,
            &after_core.timelines_spec(),
            self.rdma.timeline_id,
            &before_core.ledger_spec()@,
            &after_core.ledger_spec()@,
            rdma::CqSnapshot::of(self.rdma.cq),
            after_core.instance_id(),
            self.rdma.next_control_wr_id,
        );
        lemma_completed_pulls_after_step(
            self.rdma.completed_pulls,
            &before_core.timelines_spec(),
            &after_core.timelines_spec(),
            self.rdma.timeline_id,
            self.rdma.cq.observer(),
            &before_core.ledger_spec()@,
            &after_core.ledger_spec()@,
        );
        assert(after_core.frames_spec().same_directory(&before_core.frames_spec())) by {
            reveal(FrameTable::same_directory);
        };
        assert(after_core.frames_spec().same_layout(&before_core.frames_spec())) by {
            reveal(FrameTable::same_layout);
        };
        lemma_rdma_all_slabs_same_directory(
            &self.rdma.host_mr,
            &self.rdma.gpu_mr,
            self.rdma.shared_host_mr,
            self.rdma.shared_gpu_mr,
            before_core,
            after_core,
            self.rdma.window_pool.pd(),
        );
        lemma_rdma_leases_same_frames(
            self.rdma.active_leases@,
            &self.rdma.window_pool,
            &self.rdma.qp,
            before_core,
            after_core,
        );
        lemma_outbound_grants_preserved(
            self.pending_outbound_grants@,
            self.rdma.active_leases@,
            self.rdma.active_leases@,
            before_core,
            after_core,
        );
        assert forall|k: int| 0 <= k < self.pending_outbound_grants@.len() implies (
        #[trigger] self.pending_outbound_grants@[k]).valid(&self.rdma, after_core) by {
            assert(self.pending_outbound_grants@[k].valid(&self.rdma, before_core));
        }
        assert forall|k: int| 0 <= k < self.pending_inbound_grants@.len() implies (
        #[trigger] self.pending_inbound_grants@[k]).valid(
            after_core,
            &self.channel,
            self.rdma.peer_node,
        ) by {
            assert(self.pending_inbound_grants@[k].valid(
                before_core,
                &self.channel,
                self.rdma.peer_node,
            ));
        }
        assert(peer_core_step(before_core, after_core, self.rdma.timeline_id)) by {
            reveal(peer_core_step_except);
            reveal(core_transition);
            reveal(FrameTable::preserves_initializing);
        };
        assert(stmt_closure.subset_of(after_core.citations())) by {
            assert forall|e: TransferEvent| #[trigger]
                stmt_closure.contains(e) implies after_core.citations().contains(e) by {
                assert(stmt_sub.dom().contains(e)) by {
                    if !stmt_facts.contains(e) {
                        let m = choose|m: TransferEvent|
                            stmt_facts.contains(m) && (#[trigger] stmt_sub.map()[m]).preds.contains(
                                e,
                            );
                        assert(stmt_sub.dom().contains(m));
                        assert(stmt_sub.map()[m].preds.subset_of(stmt_sub.dom()));
                    }
                }
                assert(after_core.ledger_spec()@.submitted.dom().contains(e));
                assert(after_core.ledger_spec()@.knows.set().contains((after_core.node(), e)));
            }
        };
    }

    proof fn lemma_session_wf_after_revoke_swap_remove(
        before: &Self,
        after: &Self,
        core: &CacheCore,
        pos: int,
        grant: ActiveGrant,
    )
        requires
            before.well_formed(core),
            0 <= pos < before.active_grants@.len(),
            grant == before.active_grants@[pos],
            after.rdma == before.rdma,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants@ == before.active_grants@.update(
                pos,
                before.active_grants@.last(),
            ).drop_last(),
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
        ensures
            after.well_formed(core),
            after.rdma.well_formed(core),
            after.rdma.lease_index_spec(grant.handle) matches Some(idx) && {
                &&& 0 <= idx < after.rdma.active_leases@.len()
                &&& after.rdma.active_leases@[idx].state == RemoteLeaseState::Bound
            },
            forall|j: int|
                0 <= j < after.active_grants@.len() ==> (#[trigger] after.active_grants@[j]).handle
                    != grant.handle,
            forall|k: int|
                0 <= k < after.pending_outbound_grants@.len() ==> (
                #[trigger] after.pending_outbound_grants@[k]).handle != grant.handle,
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PendingInboundGrant::valid);
        hide(super::protocol::statement_replica_valid);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        lemma_swap_removed_grant(before.active_grants@, after.active_grants@, pos);
        let hidx = before.rdma.lease_index_spec(grant.handle).unwrap();
        assert(0 <= hidx < before.rdma.active_leases@.len()) by {
            reveal(RdmaBackend::well_formed);
            reveal(rdma_leases_valid);
        };
        assert forall|i: int| 0 <= i < after.active_grants@.len() implies {
            let g = #[trigger] after.active_grants@[i];
            &&& g.grant_tag < after.next_grant_tag
            &&& g.frame.0 < core.frames_spec().metas_spec().len()
            &&& after.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                let lease = after.rdma.active_leases@[idx];
                &&& lease.frame == g.frame
                &&& lease.state == RemoteLeaseState::Bound
                &&& after.channel.sent().contains_key(g.grant_tag)
                &&& after.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                    slot: core.frames_spec().frame(g.frame).slot,
                    share_id: lease.share@,
                })
            }
        } by {
            let orig_i = if i == pos {
                (before.active_grants@.len() - 1) as int
            } else {
                i
            };
            assert(after.active_grants@[i] == before.active_grants@[orig_i]);
        }
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[k]).handle != grant.handle by {
            let p = before.pending_outbound_grants@[k];
            assert(after.pending_outbound_grants@[k] == p);
            assert(p.valid(&before.rdma, core));
            if p.handle == grant.handle {
                assert(super::poll::rdma_lease_is_binding_like(before.rdma.active_leases@[hidx]));
            }
        }
    }

    proof fn lemma_session_wf_after_pump_for_revoke(
        before: &Self,
        after: &Self,
        init_core: &CacheCore,
        before_core: &CacheCore,
        after_core: &CacheCore,
    )
        requires
            before.well_formed(before_core),
            after.rdma.well_formed(after_core),
            after_core.node_id_spec() == before_core.node_id_spec(),
            after_core.instance_id() == before_core.instance_id(),
            after_core.frames_spec() == before_core.frames_spec(),
            after_core.ledger_spec()@.submitted == before_core.ledger_spec()@.submitted,
            init_core.ledger_spec()@.knows.set().subset_of(before_core.ledger_spec()@.knows.set()),
            before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
            init_core.ledger_spec()@.succeeded.dom().subset_of(
                before_core.ledger_spec()@.succeeded.dom(),
            ),
            before_core.ledger_spec()@.succeeded.dom().subset_of(
                after_core.ledger_spec()@.succeeded.dom(),
            ),
            after.rdma.peer_node == before.rdma.peer_node,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
            forall|h: RemoteLeaseHandle| #[trigger]
                before.rdma.lease_index_spec(h) is Some && !(
                before.rdma.active_leases@[before.rdma.lease_index_spec(h)->Some_0]).has_active_wr()
                    ==> (after.rdma.lease_index_spec(h) matches Some(j)
                    && after.rdma.active_leases@[j]
                    == before.rdma.active_leases@[before.rdma.lease_index_spec(h)->Some_0]),
            forall|h: RemoteLeaseHandle| #[trigger]
                before.rdma.lease_index_spec(h) is Some && super::poll::rdma_lease_is_binding_like(
                    before.rdma.active_leases@[before.rdma.lease_index_spec(h)->Some_0],
                ) ==> (after.rdma.lease_index_spec(h) matches Some(j)
                    && super::poll::rdma_lease_binding_preserved(
                    before.rdma.active_leases@[before.rdma.lease_index_spec(h)->Some_0],
                    after.rdma.active_leases@[j],
                )),
        ensures
            after.well_formed(after_core),
            init_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
            init_core.ledger_spec()@.succeeded.dom().subset_of(
                after_core.ledger_spec()@.succeeded.dom(),
            ),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(super::protocol::statement_replica_valid);
        reveal(peer_core_step_except);
        reveal(core_transition);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        reveal(PendingInboundGrant::valid);
        assert(init_core.ledger_spec()@.knows.set().subset_of(
            after_core.ledger_spec()@.knows.set(),
        )) by {
            assert forall|x| #[trigger]
                init_core.ledger_spec()@.knows.set().contains(
                    x,
                ) implies after_core.ledger_spec()@.knows.set().contains(x) by {
                assert(before_core.ledger_spec()@.knows.set().contains(x));
            }
        };
        assert(init_core.ledger_spec()@.succeeded.dom().subset_of(
            after_core.ledger_spec()@.succeeded.dom(),
        )) by {
            assert forall|x| #[trigger]
                init_core.ledger_spec()@.succeeded.dom().contains(
                    x,
                ) implies after_core.ledger_spec()@.succeeded.dom().contains(x) by {
                assert(before_core.ledger_spec()@.succeeded.dom().contains(x));
            }
        };
        assert forall|i: int| 0 <= i < after.active_grants@.len() implies {
            let g = #[trigger] after.active_grants@[i];
            &&& g.grant_tag < after.next_grant_tag
            &&& g.frame.0 < after_core.frames_spec().metas_spec().len()
            &&& after.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                let lease = after.rdma.active_leases@[idx];
                &&& lease.frame == g.frame
                &&& lease.state == RemoteLeaseState::Bound
                &&& after.channel.sent().contains_key(g.grant_tag)
                &&& after.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                    slot: after_core.frames_spec().frame(g.frame).slot,
                    share_id: lease.share@,
                })
            }
        } by {
            let g = after.active_grants@[i];
            assert(before.active_grants@[i] == g);
            let oidx = before.rdma.lease_index_spec(g.handle).unwrap();
            assert(before.rdma.active_leases@[oidx].state == RemoteLeaseState::Bound);
            assert(!before.rdma.active_leases@[oidx].has_active_wr());
        }
        assert(after_core.frames_spec().same_layout(&before_core.frames_spec())) by {
            reveal(FrameTable::same_layout);
        };
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies {
            let p = #[trigger] after.pending_outbound_grants@[k];
            &&& rdma_lease_index_spec(before.rdma.active_leases@, p.handle) matches Some(bidx) && {
                let bl = before.rdma.active_leases@[bidx];
                &&& rdma_lease_index_spec(after.rdma.active_leases@, p.handle) matches Some(aidx)
                &&& after.rdma.active_leases@[aidx].frame == bl.frame
                &&& after.rdma.active_leases@[aidx].wr_id == bl.wr_id
                &&& super::poll::rdma_lease_is_binding_like(after.rdma.active_leases@[aidx])
                &&& after.rdma.active_leases@[aidx].share_cap == bl.share_cap
            }
        } by {
            let p = after.pending_outbound_grants@[k];
            assert(before.pending_outbound_grants@[k] == p);
            assert(p.valid(&before.rdma, before_core));
        }
        assert(rdma_leases_disjoint(after.rdma.active_leases@)) by {
            reveal(RdmaBackend::well_formed);
        };
        lemma_outbound_grants_preserved(
            after.pending_outbound_grants@,
            before.rdma.active_leases@,
            after.rdma.active_leases@,
            before_core,
            after_core,
        );
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[k]).valid(&after.rdma, after_core) by {
            assert(before.pending_outbound_grants@[k] == after.pending_outbound_grants@[k]);
        }
        assert forall|k: int| 0 <= k < after.pending_inbound_grants@.len() implies (
        #[trigger] after.pending_inbound_grants@[k]).valid(
            after_core,
            &after.channel,
            after.rdma.peer_node,
        ) by {
            assert(before.pending_inbound_grants@[k] == after.pending_inbound_grants@[k]);
            assert(before.pending_inbound_grants@[k].valid(
                before_core,
                &before.channel,
                before.rdma.peer_node,
            ));
        }
    }

    proof fn lemma_session_wf_after_outbound_lease_remove(
        before: &Self,
        after: &Self,
        before_core: &CacheCore,
        after_core: &CacheCore,
        handle: RemoteLeaseHandle,
    )
        requires
            before.well_formed(before_core),
            after.rdma.well_formed(after_core),
            after_core.node_id_spec() == before_core.node_id_spec(),
            after_core.instance_id() == before_core.instance_id(),
            after_core.frames_spec().same_directory(&before_core.frames_spec()),
            *after_core == *before_core || peer_core_step(
                before_core,
                after_core,
                before.rdma.timeline_id,
            ) || (before_core.ledger_spec()@.submitted.map().submap_of(
                after_core.ledger_spec()@.submitted.map(),
            ) && before_core.ledger_spec()@.knows.set().subset_of(
                after_core.ledger_spec()@.knows.set(),
            ) && before_core.ledger_spec()@.succeeded.dom().subset_of(
                after_core.ledger_spec()@.succeeded.dom(),
            )),
            after.rdma.peer_node == before.rdma.peer_node,
            after.channel == before.channel,
            after.next_grant_tag == before.next_grant_tag,
            after.active_grants == before.active_grants,
            after.pending_outbound_grants == before.pending_outbound_grants,
            after.pending_inbound_grants == before.pending_inbound_grants,
            rdma_leases_preserved_except(
                before.rdma.active_leases@,
                after.rdma.active_leases@,
                handle,
            ),
            forall|j: int|
                0 <= j < before.active_grants@.len() ==> (
                #[trigger] before.active_grants@[j]).handle != handle,
            forall|k: int|
                0 <= k < before.pending_outbound_grants@.len() ==> (
                #[trigger] before.pending_outbound_grants@[k]).handle != handle,
        ensures
            after.well_formed(after_core),
    {
        hide(CacheCore::well_formed);
        hide(RdmaBackend::well_formed);
        hide(super::protocol::statement_replica_valid);
        reveal(peer_core_step_except);
        reveal(core_transition);
        reveal(PeerSession::well_formed);
        reveal(PeerSession::session_queues_valid);
        reveal(PendingOutboundGrant::valid);
        reveal(PendingInboundGrant::valid);
        assert(rdma_leases_disjoint(after.rdma.active_leases@)) by {
            reveal(RdmaBackend::well_formed);
            reveal(rdma_leases_valid);
        }
        assert(after_core.frames_spec().same_layout(&before_core.frames_spec())) by {
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
        }
        let pre_leases = before.rdma.active_leases@;
        let post_leases = after.rdma.active_leases@;
        lemma_peer_grants_after_lease_remove(
            after.active_grants@,
            pre_leases,
            post_leases,
            handle,
            after.channel.sent(),
            after.next_grant_tag,
            before_core,
            after_core,
        );
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies {
            let q = #[trigger] after.pending_outbound_grants@[k];
            &&& rdma_lease_index_spec(pre_leases, q.handle) matches Some(bidx) && {
                let bl = pre_leases[bidx];
                &&& rdma_lease_index_spec(post_leases, q.handle) matches Some(aidx)
                &&& post_leases[aidx].frame == bl.frame
                &&& post_leases[aidx].wr_id == bl.wr_id
                &&& super::poll::rdma_lease_is_binding_like(post_leases[aidx])
                &&& post_leases[aidx].share_cap == bl.share_cap
            }
        } by {
            let q = after.pending_outbound_grants@[k];
            assert(before.pending_outbound_grants@[k] == q);
            assert(q.valid(&before.rdma, before_core));
            let bidx = rdma_lease_index_spec(pre_leases, q.handle).unwrap();
            lemma_lease_index_preserved_remove(pre_leases, post_leases, handle, q.handle, bidx);
        }
        lemma_outbound_grants_preserved(
            after.pending_outbound_grants@,
            pre_leases,
            post_leases,
            before_core,
            after_core,
        );
        assert forall|k: int| 0 <= k < after.pending_outbound_grants@.len() implies (
        #[trigger] after.pending_outbound_grants@[k]).valid(&after.rdma, after_core) by {
            assert(before.pending_outbound_grants@[k] == after.pending_outbound_grants@[k]);
        }
        assert forall|k: int| 0 <= k < after.pending_inbound_grants@.len() implies (
        #[trigger] after.pending_inbound_grants@[k]).valid(
            after_core,
            &after.channel,
            after.rdma.peer_node,
        ) by {
            let g = before.pending_inbound_grants@[k];
            assert(after.pending_inbound_grants@[k] == g);
            assert(g.valid(before_core, &before.channel, before.rdma.peer_node));
        }
    }

    /// Donor helper: advance any `pending_outbound_grants` whose MW bind CQE
    /// (`BindReady` or `BindFailedReady`) has been staged by `pump_cq`.
    #[verifier::rlimit(250)]
    fn process_pending_outbound_grants(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            final(self).channel.received() == old(self).channel.received(),
            final(self).pending_inbound_grants == old(self).pending_inbound_grants,
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        hide(ActiveRemoteLease::valid);
        hide(super::protocol::statement_replica_valid);
        let mut completed: usize = 0;
        let mut i: usize = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        while i < self.pending_outbound_grants.len()
            invariant
                i <= self.pending_outbound_grants@.len(),
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                self.channel.same_session(old(self).channel),
                self.channel.received() == old(self).channel.received(),
                self.pending_inbound_grants == old(self).pending_inbound_grants,
                forall|seq: u64, writer: TransferEvent| #[trigger]
                    pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                        &self.rdma,
                        seq,
                        writer,
                    ),
            decreases self.pending_outbound_grants@.len() - i,
        {
            let ghost iter_rdma = self.rdma;
            let ghost iter_leases = self.rdma.active_leases@;
            proof {
                self.lemma_peer_wf_lease_facts(core);
            }
            let h = self.pending_outbound_grants[i].handle;
            let st = self.rdma.lease_state(h);
            match st {
                Some(RemoteLeaseState::BindReady) => {
                    let ghost pre_pop_self = *self;
                    let p = self.pending_outbound_grants.swap_remove(i);
                    proof {
                        Self::lemma_pop_outbound_grant(&pre_pop_self, self, core, i as int, &p);
                    }
                    let ghost step_core = *core;
                    if p.send_on_ready && self.next_grant_tag < u64::MAX {
                        let Tracked(rev) = p.bind_revoked;
                        let Tracked(vb) = p.version_binding;
                        let res = self.finish_send_bound_grant(
                            core,
                            p.handle,
                            p.frame,
                            p.request_id,
                            p.key,
                            p.version,
                            p.cert,
                            p.bind_sub,
                            p.bind_succ,
                            Tracked(rev),
                            p.bind_facts,
                            Tracked(vb),
                            Tracked(instance),
                        );
                        proof {
                            lemma_peer_core_step_trans(
                                old(core),
                                &step_core,
                                core,
                                self.rdma.timeline_id,
                            );
                            lemma_foreign_shares_trans(
                                old(core),
                                &step_core,
                                core,
                                old(self).rdma.active_leases@,
                                iter_leases,
                                self.rdma.active_leases@,
                            );
                            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                                pull_tracked_or_done(
                                    &old(self).rdma,
                                    seq,
                                    writer,
                                ) implies pull_tracked_or_done(&self.rdma, seq, writer) by {
                                assert(pull_tracked_or_done(&iter_rdma, seq, writer));
                            }
                        }
                        if let Err(e) = res {
                            return Err(e);
                        }
                    } else {
                        let ghost pre_c_self = *self;
                        let ghost hidx = self.rdma.lease_index_spec(p.handle).unwrap();
                        let (_, Tracked(share_cap)) = match self.rdma.complete_ready_bind(
                            core,
                            p.handle,
                        ) {
                            Ok(pair) => pair,
                            Err(e) => {
                                proof {
                                    lemma_foreign_shares_trans(
                                        old(core),
                                        &step_core,
                                        core,
                                        old(self).rdma.active_leases@,
                                        iter_leases,
                                        self.rdma.active_leases@,
                                    );
                                }
                                return Err(e);
                            },
                        };
                        proof {
                            lemma_foreign_shares_trans(
                                old(core),
                                &step_core,
                                core,
                                old(self).rdma.active_leases@,
                                iter_leases,
                                self.rdma.active_leases@,
                            );
                            assert(core.frames_spec().same_directory(&core.frames_spec())) by {
                                reveal(FrameTable::same_directory);
                            }
                            broadcast use {
                                vstd::imap::group_imap_lemmas,
                                vstd::iset::group_iset_lemmas,
                            };

                            Self::lemma_session_wf_after_outbound_lease_update(
                                &pre_c_self,
                                self,
                                core,
                                core,
                                p.handle,
                                hidx,
                            );
                            lemma_same_pulls_preserves_pull_tracked(&iter_rdma, &self.rdma);
                        }
                        let ghost pre_u_self = *self;
                        let ghost before_u_core = *core;
                        let ghost before_u_leases = self.rdma.active_leases@;
                        let ghost u_idx = self.rdma.lease_index_spec(p.handle).unwrap();
                        match self.rdma.unpin_and_unbind_lease(
                            core,
                            p.handle,
                            Tracked(share_cap),
                            Ticket::NONE,
                            Tracked(instance),
                        ) {
                            Ok(inv_wr_id) => {
                                proof {
                                    reveal(peer_core_step_except);
                                    reveal(core_transition);
                                    Self::lemma_session_wf_after_outbound_lease_update(
                                        &pre_u_self,
                                        self,
                                        &before_u_core,
                                        core,
                                        p.handle,
                                        u_idx,
                                    );
                                    lemma_same_pulls_preserves_pull_tracked(&iter_rdma, &self.rdma);
                                    lemma_peer_core_step_trans(
                                        old(core),
                                        &before_u_core,
                                        core,
                                        self.rdma.timeline_id,
                                    );
                                    lemma_foreign_shares_trans(
                                        old(core),
                                        &before_u_core,
                                        core,
                                        old(self).rdma.active_leases@,
                                        before_u_leases,
                                        self.rdma.active_leases@,
                                    );
                                    assert(core.all_versions_bound()) by {
                                        reveal(CacheCore::well_formed);
                                    }
                                }
                                let ghost before_await = *core;
                                let ghost before_await_leases = self.rdma.active_leases@;
                                let ghost before_await_rdma = self.rdma;
                                let await_res = self.await_unbind_invalidate(
                                    core,
                                    p.handle,
                                    inv_wr_id,
                                    Tracked(instance),
                                );
                                proof {
                                    lemma_peer_core_step_trans(
                                        old(core),
                                        &before_await,
                                        core,
                                        self.rdma.timeline_id,
                                    );
                                    lemma_foreign_shares_trans(
                                        old(core),
                                        &before_await,
                                        core,
                                        old(self).rdma.active_leases@,
                                        before_await_leases,
                                        self.rdma.active_leases@,
                                    );
                                    assert(core.all_versions_bound()) by {
                                        reveal(CacheCore::well_formed);
                                    }
                                    assert forall|seq: u64, writer: TransferEvent| #[trigger]
                                        pull_tracked_or_done(
                                            &old(self).rdma,
                                            seq,
                                            writer,
                                        ) implies pull_tracked_or_done(&self.rdma, seq, writer) by {
                                        assert(pull_tracked_or_done(&iter_rdma, seq, writer));
                                        assert(pull_tracked_or_done(
                                            &before_await_rdma,
                                            seq,
                                            writer,
                                        ));
                                    }
                                }
                                if let Err(e) = await_res {
                                    return Err(e);
                                }
                            },
                            Err((e, _)) => {
                                proof {
                                    lemma_foreign_shares_trans(
                                        old(core),
                                        &before_u_core,
                                        core,
                                        old(self).rdma.active_leases@,
                                        before_u_leases,
                                        self.rdma.active_leases@,
                                    );
                                    assert(core.frames_spec().same_directory(
                                        &before_u_core.frames_spec(),
                                    )) by {
                                        reveal(FrameTable::same_directory);
                                    }
                                    broadcast use {
                                        vstd::imap::group_imap_lemmas,
                                        vstd::iset::group_iset_lemmas,
                                    };

                                    Self::lemma_session_wf_after_outbound_lease_update(
                                        &pre_u_self,
                                        self,
                                        &before_u_core,
                                        core,
                                        p.handle,
                                        u_idx,
                                    );
                                    lemma_same_pulls_preserves_pull_tracked(&iter_rdma, &self.rdma);
                                    assert forall|seq: u64, writer: TransferEvent| #[trigger]
                                        pull_tracked_or_done(
                                            &old(self).rdma,
                                            seq,
                                            writer,
                                        ) implies pull_tracked_or_done(&self.rdma, seq, writer) by {
                                        assert(pull_tracked_or_done(&iter_rdma, seq, writer));
                                    }
                                }
                                return Err(e);
                            },
                        }
                        if p.send_on_ready {
                            self.pending_inbound_fetches.push((p.request_id, p.key));
                            proof {
                                assert(self.well_formed(core)) by {
                                    reveal(PeerSession::well_formed);
                                    reveal(PeerSession::session_queues_valid);
                                }
                            }
                        }
                    }
                    if completed < usize::MAX {
                        completed += 1;
                    }
                },
                Some(RemoteLeaseState::BindFailedReady) => {
                    let ghost pre_pop_self = *self;
                    let p = self.pending_outbound_grants.swap_remove(i);
                    proof {
                        Self::lemma_pop_outbound_grant(&pre_pop_self, self, core, i as int, &p);
                    }
                    let ghost pre_fail_self = *self;
                    let ghost pre_fail_core = *core;
                    let ghost pre_fail_leases = self.rdma.active_leases@;
                    let cfb_ok = self.rdma.complete_failed_bind(
                        core,
                        p.handle,
                        Tracked(instance),
                    ).is_ok();
                    proof {
                        lemma_foreign_shares_trans(
                            old(core),
                            &pre_fail_core,
                            core,
                            old(self).rdma.active_leases@,
                            pre_fail_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    if cfb_ok {
                        proof {
                            broadcast use {
                                vstd::imap::group_imap_lemmas,
                                vstd::iset::group_iset_lemmas,
                            };

                            Self::lemma_session_wf_after_outbound_lease_remove(
                                &pre_fail_self,
                                self,
                                &pre_fail_core,
                                core,
                                p.handle,
                            );
                            assert(peer_core_step(&pre_fail_core, core, self.rdma.timeline_id)) by {
                                reveal(peer_core_step_except);
                                reveal(core_transition);
                                reveal(FrameTable::same_directory);
                                reveal(FrameTable::same_layout);
                                reveal(FrameTable::preserves_initializing);
                            }
                            lemma_peer_core_step_trans(
                                old(core),
                                &pre_fail_core,
                                core,
                                self.rdma.timeline_id,
                            );
                            assert(core.all_versions_bound()) by {
                                reveal(CacheCore::well_formed);
                            }
                        }
                    }
                    proof {
                        lemma_same_pulls_preserves_pull_tracked(&iter_rdma, &self.rdma);
                        assert forall|seq: u64, writer: TransferEvent| #[trigger]
                            pull_tracked_or_done(
                                &old(self).rdma,
                                seq,
                                writer,
                            ) implies pull_tracked_or_done(&self.rdma, seq, writer) by {
                            assert(pull_tracked_or_done(&iter_rdma, seq, writer));
                        }
                    }
                    if p.send_on_ready {
                        self.pending_inbound_fetches.push((p.request_id, p.key));
                        proof {
                            assert(self.well_formed(core)) by {
                                reveal(PeerSession::well_formed);
                                reveal(PeerSession::session_queues_valid);
                            }
                        }
                    }
                    if completed < usize::MAX {
                        completed += 1;
                    }
                },
                _ => {
                    i += 1;
                },
            }
        }
        Ok(completed)
    }

    /// Receive a burst of transport frames and respond to cooperative revokes.
    fn pump_inbound_nonblocking(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let progress = self.drain_inbound_frames(core, Tracked(instance))?;
        let ghost before = *core;
        let ghost before_rdma = self.rdma;
        let ack_result = self.send_pending_revoke_acks(core, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before,
                core,
                old(self).rdma.active_leases@,
                before_rdma.active_leases@,
                self.rdma.active_leases@,
            );
            lemma_same_pulls_preserves_pull_tracked(&before_rdma, &self.rdma);
            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) implies pull_tracked_or_done(
                &self.rdma,
                seq,
                writer,
            ) by {
                assert(pull_tracked_or_done(&before_rdma, seq, writer));
            }
        }
        let _ = ack_result?;
        Ok(progress)
    }

    /// Retire binds and invalidations and replenish windows released by the protocol.
    #[verifier::rlimit(250)]
    pub(crate) fn pump_outbound_nonblocking(
        &mut self,
        core: &mut CacheCore,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            pd matches Some(p) ==> p.id() == old(self).rdma.window_pool.pd() && p.node() == old(
                core,
            ).node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    core,
                ).node(),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut progress = PeerProgress::empty();
        progress.cq_processed = self.pump_session_cq(core, Tracked(instance))?;
        let ghost before = *core;
        let ghost before_rdma = self.rdma;
        let grants = self.process_pending_outbound_grants(core, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before,
                core,
                old(self).rdma.active_leases@,
                before_rdma.active_leases@,
                self.rdma.active_leases@,
            );
            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) implies pull_tracked_or_done(
                &self.rdma,
                seq,
                writer,
            ) by {
                assert(pull_tracked_or_done(&before_rdma, seq, writer));
            }
        }
        progress.grants_completed = grants?;
        let ghost before_rdma = self.rdma;
        progress.windows_replenished = if let Some(pd_ref) = pd {
            self.replenish_acknowledged_windows(core, pd_ref, tokens)
        } else {
            let _ = self.acknowledge_pending_revoke_acks(core);
            0
        };
        proof {
            lemma_same_pulls_preserves_pull_tracked(&before_rdma, &self.rdma);
            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) implies pull_tracked_or_done(
                &self.rdma,
                seq,
                writer,
            ) by {
                assert(pull_tracked_or_done(&before_rdma, seq, writer));
            }
        }
        Ok(progress)
    }

    /// Drive CQ and protocol progress and queue received fetches for caller admission.
    pub(crate) fn pump_nonblocking(
        &mut self,
        core: &mut CacheCore,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            pd matches Some(p) ==> p.id() == old(self).rdma.window_pool.pd() && p.node() == old(
                core,
            ).node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    core,
                ).node(),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut progress = self.pump_inbound_nonblocking(core, Tracked(instance))?;
        let ghost before = *core;
        let ghost before_rdma = self.rdma;
        let outbound = self.pump_outbound_nonblocking(core, pd, tokens, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before,
                core,
                old(self).rdma.active_leases@,
                before_rdma.active_leases@,
                self.rdma.active_leases@,
            );
            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) implies pull_tracked_or_done(
                &self.rdma,
                seq,
                writer,
            ) by {
                assert(pull_tracked_or_done(&before_rdma, seq, writer));
            }
        }
        progress.accumulate(outbound?);
        Ok(progress)
    }

    /// Full-duplex progress driver: drains CQEs + inbound channel frames, sends
    /// `RevokeAck`s, initiates async MW binds for inbound `FetchRequest`s, pumps
    /// freshly completed binds, completes outbound `BindReady` grants, and
    /// acknowledges/replenishes quarantined MWs.
    #[verifier::rlimit(250)]
    pub(crate) fn poll_progress_multi(
        &mut self,
        core: &mut CacheCore,
        primary_tier: TierId,
        fallback_tier: Option<TierId>,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            pd matches Some(p) ==> p.id() == old(self).rdma.window_pool.pd() && p.node() == old(
                core,
            ).node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma.host_mr == old(self).rdma.host_mr,
            final(self).rdma.gpu_mr == old(self).rdma.gpu_mr,
            final(self).rdma.shared_host_mr == old(self).rdma.shared_host_mr,
            final(self).rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
            final(self).channel.same_session(old(self).channel),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    core,
                ).node(),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma,
                    seq,
                    writer,
                ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut progress = self.pump_inbound_nonblocking(core, Tracked(instance))?;
        let ghost before = *core;
        let ghost before_rdma = self.rdma;
        self.queue_pending_inbound_fetches(
            core,
            primary_tier,
            fallback_tier,
            Tracked(shares),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before,
                core,
                old(self).rdma.active_leases@,
                before_rdma.active_leases@,
                self.rdma.active_leases@,
            );
            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) implies pull_tracked_or_done(
                &self.rdma,
                seq,
                writer,
            ) by {
                assert(pull_tracked_or_done(&before_rdma, seq, writer));
            }
        }
        let ghost before = *core;
        let ghost before_rdma = self.rdma;
        let outbound = self.pump_outbound_nonblocking(core, pd, tokens, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before,
                core,
                old(self).rdma.active_leases@,
                before_rdma.active_leases@,
                self.rdma.active_leases@,
            );
            assert forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma, seq, writer) implies pull_tracked_or_done(
                &self.rdma,
                seq,
                writer,
            ) by {
                assert(pull_tracked_or_done(&before_rdma, seq, writer));
            }
        }
        progress.accumulate(outbound?);
        Ok(progress)
    }

    /// Initiator: multiplexed remote frame fetch (`FetchRequest` -> full-duplex CQ
    /// & channel step loop -> `fetch_post_pull` -> `fetch_complete_pull`).
    /// Safe over blocking `TcpTransport` (`pump_cq_and_channel_step` performs at
    /// most one `try_recv_any` per loop iteration) and handles concurrent
    /// `ReadRelease`, `FetchRequest`, `RevokeNotice`, `RevokeAck`, and out-of-order
    /// `ReadGrant`/`FetchMiss` messages without choking.
    #[verifier::rlimit(250)]
    pub fn fetch_remote_frame(
        &mut self,
        core: &mut CacheCore,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma_spec().qp_spec().ready(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).rdma_spec().qp_spec().source().well_formed(),
            old(self).rdma_spec().qp_spec().effective_max_rd_atomic() > 0,
            old(self).rdma_spec().peer_node_spec() != old(core).node(),
            old(self).next_request_id_spec() < u64::MAX,
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(core).instance_id(),
            match target {
                PullTarget::Host => old(self).rdma_spec().effective_host_mr_spec() matches Some(s)
                    && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma_spec().effective_gpu_mr_spec() matches Some(s)
                    && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().peer_node_spec() == old(self).rdma_spec().peer_node_spec(),
            final(self).rdma_spec().qp_spec().ready(),
            final(self).rdma_spec().qp_spec().state() is ReadyToSend,
            final(self).rdma_spec().qp_spec().source().well_formed(),
            final(self).rdma_spec().qp_spec().source().source() == old(
                self,
            ).rdma_spec().qp_spec().source().source(),
    {
        hide(CacheCore::well_formed);
        hide(PeerSession::well_formed);
        hide(RdmaBackend::well_formed);
        hide(PeerSession::session_queues_valid);
        hide(PendingOutboundGrant::valid);
        hide(PendingInboundGrant::valid);
        hide(ActiveRemoteLease::valid);
        hide(super::protocol::statement_replica_valid);
        hide(replica_valid);
        hide(complete_receipts);
        let req_id = match self.send_fetch_request(core, key, Tracked(instance)) {
            Ok(id) => id,
            Err(_) => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                    lemma_foreign_shares_refl(core, self.rdma.active_leases@);
                }
                return Err(rdma_error(rdma::RdmaError(-1)));
            },
        };
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        let mut spins: u32 = 0;
        let mut chosen_grant: Option<PendingInboundGrant> = None;
        while spins < 1_000_000 && chosen_grant.is_none()
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                self.rdma.qp.source().well_formed(),
                self.rdma.qp.effective_max_rd_atomic() > 0,
                self.rdma.peer_node != core.node(),
                core.frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                match target {
                    PullTarget::Host => self.rdma.effective_host_mr_spec() matches Some(s) && s.tier
                        == dst_tier && s.mr.access().local_write
                        && s.mr.gpu_device_spec().is_none(),
                    PullTarget::GpuGdr => self.rdma.effective_gpu_mr_spec() matches Some(s)
                        && s.tier == dst_tier && s.mr.access().local_write
                        && s.mr.gpu_device_spec() is Some,
                },
                chosen_grant matches Some(g) ==> g.meta.request_id == req_id && g.valid(
                    core,
                    &self.channel,
                    self.rdma.peer_node,
                ),
            decreases
                    1_000_000 - spins + (if chosen_grant.is_some() {
                        0int
                    } else {
                        1int
                    }),
        {
            if let Some(g) = self.take_inbound_grant(core, req_id) {
                chosen_grant = Some(g);
            } else {
                let ghost pre_miss_self = *self;
                let is_miss = self.take_inbound_miss(req_id);
                proof {
                    Self::lemma_session_wf_same_active_and_queues(&pre_miss_self, &*self, core);
                }
                if is_miss {
                    return Ok(FetchOutcome::Miss);
                }
                let ghost step_core = *core;
                let ghost step_leases = self.rdma.active_leases@;
                match self.pump_cq_and_channel_step(core, Tracked(instance)) {
                    Ok(_) => {
                        proof {
                            assert(core.frames_spec().same_layout(&step_core.frames_spec())) by {
                                reveal(peer_core_step_except);
                                reveal(core_transition);
                                reveal(FrameTable::same_layout);
                            };
                            lemma_peer_core_step_trans(
                                old(core),
                                &step_core,
                                core,
                                self.rdma.timeline_id,
                            );
                            lemma_foreign_shares_trans(
                                old(core),
                                &step_core,
                                core,
                                old(self).rdma.active_leases@,
                                step_leases,
                                self.rdma.active_leases@,
                            );
                        }
                    },
                    Err(e) => {
                        proof {
                            lemma_peer_core_step_trans(
                                old(core),
                                &step_core,
                                core,
                                self.rdma.timeline_id,
                            );
                            lemma_foreign_shares_trans(
                                old(core),
                                &step_core,
                                core,
                                old(self).rdma.active_leases@,
                                step_leases,
                                self.rdma.active_leases@,
                            );
                        }
                        return Err(e);
                    },
                }
                spins += 1;
            }
        }
        let Some(g) = chosen_grant else {
            return Err(rdma_error(rdma::RdmaError(11)));
        };
        proof {
            self.lemma_peer_wf_lease_facts(core);
            self.rdma.lemma_effective_slabs_wf(core);
            assert(dst_tier.valid(core.tiers_spec().len()));
            reveal(PendingInboundGrant::valid);
            assert(self.channel.id() == core.instance_id() && self.channel.local() == core.node())
                by {
                reveal(PeerSession::well_formed);
            };
        }
        if g.meta.frame_size != core.registered_tier_frame_size(dst_tier) as u64 {
            let ghost before_m = *core;
            let ghost before_m_leases = self.rdma.active_leases@;
            let tracked stmt = g.statement.get();
            let ghost stmt_sub = stmt.submitted;
            let ghost stmt_facts = stmt.facts;
            let ghost stmt_closure = stmt.closure();
            core.merge_peer_statement(
                Tracked(stmt.submitted),
                Tracked(stmt.knows),
                Tracked(stmt.succeeded),
                Tracked(instance),
            );
            proof {
                self.lemma_session_wf_after_merge_statement(
                    &before_m,
                    core,
                    stmt_sub,
                    stmt_facts,
                    stmt_closure,
                );
                lemma_peer_core_step_trans(old(core), &before_m, core, self.rdma.timeline_id);
                lemma_foreign_shares_identical_leases(&before_m, core, before_m_leases);
                lemma_foreign_shares_trans(
                    old(core),
                    &before_m,
                    core,
                    old(self).rdma.active_leases@,
                    before_m_leases,
                    self.rdma.active_leases@,
                );
                assert(transfer::transport::covers(g.share@.frontier(), core.citations()));
            }
            self.fetch_release_grant(core, g.tag, req_id, g.share, Tracked(instance));
            return Err(CoreError::InvalidArgument);
        }
        let ghost before_post = *core;
        let ghost before_post_leases = self.rdma.active_leases@;
        let ghost orig_share_frontier = g.share@.frontier();
        let pull = match self.fetch_post_pull(
            core,
            key,
            dst_tier,
            target,
            g.meta,
            &g.cap,
            g.share,
            g.statement,
            g.cert,
            g.revoked,
            Tracked(instance),
        ) {
            Ok(p) => p,
            Err(e) => {
                proof {
                    lemma_peer_core_step_trans(
                        old(core),
                        &before_post,
                        core,
                        self.rdma.timeline_id,
                    );
                    lemma_foreign_shares_trans(
                        old(core),
                        &before_post,
                        core,
                        old(self).rdma.active_leases@,
                        before_post_leases,
                        self.rdma.active_leases@,
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_peer_core_step_trans(old(core), &before_post, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_post,
                core,
                old(self).rdma.active_leases@,
                before_post_leases,
                self.rdma.active_leases@,
            );
        }
        let ghost after_post = *core;
        let ghost after_post_leases = self.rdma.active_leases@;
        let ghost pull_frame = pull.handle.frame;
        let tracked vb = g.version_binding.get().tracked_unwrap();
        let res = self.fetch_complete_pull(
            core,
            g.tag,
            req_id,
            pull,
            Tracked(vb),
            Ghost(orig_share_frontier),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans_retire_new_init(
                &before_post,
                &after_post,
                core,
                self.rdma.timeline_id,
                pull_frame,
            );
            lemma_peer_core_step_trans(old(core), &before_post, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &after_post,
                core,
                old(self).rdma.active_leases@,
                after_post_leases,
                self.rdma.active_leases@,
            );
        }
        res
    }

    /// Full-duplex symmetric remote fetch: while awaiting the requested grant and
    /// RDMA READ completion, actively serves inbound `FetchRequest`s, advances
    /// `pending_outbound_grants`, sends `RevokeAck`s, replenishes windows, and
    /// steps one channel frame per tick (`pump_cq_and_channel_step`), so two
    /// peers can safely call `fetch_remote_frame_duplex` against each other
    /// simultaneously even over blocking `TcpTransport`.
    #[verifier::rlimit(350)]
    pub(crate) fn fetch_remote_frame_duplex_multi(
        &mut self,
        core: &mut CacheCore,
        primary_tier: TierId,
        fallback_tier: Option<TierId>,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(self).next_request_id < u64::MAX,
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            pd matches Some(p) ==> p.id() == old(self).rdma.window_pool.pd() && p.node() == old(
                core,
            ).node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(core).node(),
            match target {
                PullTarget::Host => old(self).rdma.effective_host_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma.effective_gpu_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    core,
                ).node(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let req_id = match self.send_fetch_request(core, key, Tracked(instance)) {
            Ok(id) => id,
            Err(_) => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                    lemma_foreign_shares_refl(core, self.rdma.active_leases@);
                }
                return Err(rdma_error(rdma::RdmaError(-1)));
            },
        };
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        let mut spins: u32 = 0;
        let mut chosen_grant: Option<PendingInboundGrant> = None;
        while spins < 1_000_000 && chosen_grant.is_none()
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.node() == old(shares).node(),
                shares.id() == core.instance_id(),
                shares.node() == core.node(),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                self.rdma.qp.source().well_formed(),
                self.rdma.qp.effective_max_rd_atomic() > 0,
                self.rdma.peer_node != core.node(),
                core.frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                match target {
                    PullTarget::Host => self.rdma.effective_host_mr_spec() matches Some(s) && s.tier
                        == dst_tier && s.mr.access().local_write
                        && s.mr.gpu_device_spec().is_none(),
                    PullTarget::GpuGdr => self.rdma.effective_gpu_mr_spec() matches Some(s)
                        && s.tier == dst_tier && s.mr.access().local_write
                        && s.mr.gpu_device_spec() is Some,
                },
                pd matches Some(p) ==> p.id() == self.rdma.window_pool.pd() && p.node()
                    == core.node(),
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source().is_rdma_qp() && tokens@[k]@.node() == core.node(),
                chosen_grant matches Some(g) ==> g.meta.request_id == req_id && g.valid(
                    core,
                    &self.channel,
                    self.rdma.peer_node,
                ),
            decreases 1_000_000 - spins,
        {
            if let Some(g) = self.take_inbound_grant(core, req_id) {
                chosen_grant = Some(g);
                spins += 1;
            } else {
                if self.take_inbound_miss(req_id) {
                    return Ok(FetchOutcome::Miss);
                }
                let ghost step1_core = *core;
                let ghost step1_leases = self.rdma.active_leases@;
                if let Err(e) = self.send_pending_revoke_acks(core, Tracked(instance)) {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step1_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &step1_core,
                            core,
                            old(self).rdma.active_leases@,
                            step1_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    return Err(e);
                }
                proof {
                    lemma_peer_core_step_trans(old(core), &step1_core, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &step1_core,
                        core,
                        old(self).rdma.active_leases@,
                        step1_leases,
                        self.rdma.active_leases@,
                    );
                }
                let ghost step2_core = *core;
                let ghost step2_leases = self.rdma.active_leases@;
                self.queue_pending_inbound_fetches(
                    core,
                    primary_tier,
                    fallback_tier,
                    Tracked(shares),
                    Tracked(instance),
                );
                proof {
                    lemma_peer_core_step_trans(old(core), &step2_core, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &step2_core,
                        core,
                        old(self).rdma.active_leases@,
                        step2_leases,
                        self.rdma.active_leases@,
                    );
                }
                let ghost step3_core = *core;
                let ghost step3_leases = self.rdma.active_leases@;
                if let Err(e) = self.rdma.pump_cq_all(core, Tracked(instance)) {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step3_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &step3_core,
                            core,
                            old(self).rdma.active_leases@,
                            step3_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    return Err(e);
                }
                proof {
                    reveal(rdma_leases_valid);
                    lemma_peer_core_step_trans(old(core), &step3_core, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &step3_core,
                        core,
                        old(self).rdma.active_leases@,
                        step3_leases,
                        self.rdma.active_leases@,
                    );
                    assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                        let g = #[trigger] self.active_grants@[i];
                        &&& g.grant_tag < self.next_grant_tag
                        &&& g.frame.0 < core.frames_spec().metas_spec().len()
                        &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                            let lease = self.rdma.active_leases@[idx];
                            &&& lease.frame == g.frame
                            &&& lease.state == RemoteLeaseState::Bound
                            &&& self.channel.sent().contains_key(g.grant_tag)
                            &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                                slot: core.frames_spec().frame(g.frame).slot,
                                share_id: lease.share@,
                            })
                        }
                    } by {
                        let g = self.active_grants@[i];
                        let oidx = rdma_lease_index_spec(step3_leases, g.handle).unwrap();
                        assert(step3_leases[oidx].state == RemoteLeaseState::Bound);
                        assert(!step3_leases[oidx].has_active_wr());
                    }
                    assert(core.frames_spec().same_layout(&step3_core.frames_spec())) by {
                        reveal(FrameTable::same_layout);
                    }
                    lemma_outbound_grants_preserved(
                        self.pending_outbound_grants@,
                        step3_leases,
                        self.rdma.active_leases@,
                        &step3_core,
                        core,
                    );
                }
                let ghost step4_core = *core;
                let ghost step4_leases = self.rdma.active_leases@;
                if let Err(e) = self.process_pending_outbound_grants(core, Tracked(instance)) {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step4_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &step4_core,
                            core,
                            old(self).rdma.active_leases@,
                            step4_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    return Err(e);
                }
                proof {
                    lemma_peer_core_step_trans(old(core), &step4_core, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &step4_core,
                        core,
                        old(self).rdma.active_leases@,
                        step4_leases,
                        self.rdma.active_leases@,
                    );
                }
                if let Some(pd_ref) = pd {
                    let _ = self.replenish_acknowledged_windows(core, pd_ref, tokens);
                } else {
                    let _ = self.acknowledge_pending_revoke_acks(core);
                }
                let ghost step5_core = *core;
                let ghost step5_leases = self.rdma.active_leases@;
                if let Err(e) = self.pump_cq_and_channel_step(core, Tracked(instance)) {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step5_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &step5_core,
                            core,
                            old(self).rdma.active_leases@,
                            step5_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    return Err(e);
                }
                proof {
                    lemma_peer_core_step_trans(old(core), &step5_core, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &step5_core,
                        core,
                        old(self).rdma.active_leases@,
                        step5_leases,
                        self.rdma.active_leases@,
                    );
                }
                spins += 1;
            }
        }
        let Some(g) = chosen_grant else {
            return Err(rdma_error(rdma::RdmaError(11)));
        };
        proof {
            self.rdma.lemma_effective_slabs_wf(core);
            assert(dst_tier.valid(core.tiers_spec().len()));
        }
        if g.meta.frame_size != core.registered_tier_frame_size(dst_tier) as u64 {
            let ghost before_m = *core;
            let ghost before_m_leases = self.rdma.active_leases@;
            let tracked stmt = g.statement.get();
            let ghost stmt_sub = stmt.submitted;
            let ghost stmt_facts = stmt.facts;
            let ghost stmt_closure = stmt.closure();
            core.merge_peer_statement(
                Tracked(stmt.submitted),
                Tracked(stmt.knows),
                Tracked(stmt.succeeded),
                Tracked(instance),
            );
            proof {
                lemma_rdma_window_after_fence_ledger(
                    self.rdma.in_flight_ops,
                    self.rdma.bitmap,
                    &core.timelines_spec(),
                    self.rdma.timeline_id,
                    &before_m.ledger_spec()@,
                    &core.ledger_spec()@,
                    rdma::CqSnapshot::of(self.rdma.cq),
                    core.instance_id(),
                    self.rdma.next_control_wr_id,
                );
                lemma_completed_pulls_after_step(
                    self.rdma.completed_pulls,
                    &before_m.timelines_spec(),
                    &core.timelines_spec(),
                    self.rdma.timeline_id,
                    self.rdma.cq.observer(),
                    &before_m.ledger_spec()@,
                    &core.ledger_spec()@,
                );
                assert(core.frames_spec().same_directory(&before_m.frames_spec())) by {
                    reveal(FrameTable::same_directory);
                }
                assert(core.frames_spec().same_layout(&before_m.frames_spec())) by {
                    reveal(FrameTable::same_layout);
                }
                lemma_rdma_all_slabs_same_directory(
                    &self.rdma.host_mr,
                    &self.rdma.gpu_mr,
                    self.rdma.shared_host_mr,
                    self.rdma.shared_gpu_mr,
                    &before_m,
                    core,
                    self.rdma.window_pool.pd(),
                );
                lemma_rdma_leases_same_frames(
                    self.rdma.active_leases@,
                    &self.rdma.window_pool,
                    &self.rdma.qp,
                    &before_m,
                    core,
                );
                lemma_outbound_grants_preserved(
                    self.pending_outbound_grants@,
                    self.rdma.active_leases@,
                    self.rdma.active_leases@,
                    &before_m,
                    core,
                );
                assert(peer_core_step(&before_m, core, self.rdma.timeline_id)) by {
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::preserves_initializing);
                }
                lemma_peer_core_step_trans(old(core), &before_m, core, self.rdma.timeline_id);
                lemma_foreign_shares_identical_leases(&before_m, core, before_m_leases);
                lemma_foreign_shares_trans(
                    old(core),
                    &before_m,
                    core,
                    old(self).rdma.active_leases@,
                    before_m_leases,
                    self.rdma.active_leases@,
                );
                assert(stmt_closure.subset_of(core.citations())) by {
                    assert forall|e: TransferEvent| #[trigger]
                        stmt_closure.contains(e) implies core.citations().contains(e) by {
                        assert(stmt_sub.dom().contains(e)) by {
                            if !stmt_facts.contains(e) {
                                let m = choose|m: TransferEvent|
                                    stmt_facts.contains(m) && (
                                    #[trigger] stmt_sub.map()[m]).preds.contains(e);
                                assert(stmt_sub.dom().contains(m));
                                assert(stmt_sub.map()[m].preds.subset_of(stmt_sub.dom()));
                            }
                        }
                        assert(core.ledger_spec()@.submitted.dom().contains(e));
                        assert(core.ledger_spec()@.knows.set().contains((core.node(), e)));
                    }
                }
                assert(transfer::transport::covers(g.share@.frontier(), core.citations()));
            }
            self.fetch_release_grant(core, g.tag, req_id, g.share, Tracked(instance));
            return Err(CoreError::InvalidArgument);
        }
        self.remove_inbound_grants_by_tag(g.tag);
        let ghost before_post = *core;
        let ghost before_post_leases = self.rdma.active_leases@;
        let ghost orig_share_frontier = g.share@.frontier();
        let pull = match self.fetch_post_pull(
            core,
            key,
            dst_tier,
            target,
            g.meta,
            &g.cap,
            g.share,
            g.statement,
            g.cert,
            g.revoked,
            Tracked(instance),
        ) {
            Ok(p) => p,
            Err(e) => {
                proof {
                    lemma_peer_core_step_trans(
                        old(core),
                        &before_post,
                        core,
                        self.rdma.timeline_id,
                    );
                    lemma_foreign_shares_trans(
                        old(core),
                        &before_post,
                        core,
                        old(self).rdma.active_leases@,
                        before_post_leases,
                        self.rdma.active_leases@,
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_peer_core_step_trans(old(core), &before_post, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_post,
                core,
                old(self).rdma.active_leases@,
                before_post_leases,
                self.rdma.active_leases@,
            );
            lemma_rdma_wf_implies_pull_tracked(&self.rdma, core, pull.wr_id, pull.writer@);
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        let ghost after_post = *core;
        let ghost pull_frame = pull.handle.frame;
        let mut pull_spins: u32 = 0;
        while pull_spins < 1_000_000
            invariant
                pull_spins <= 1_000_000,
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.node() == old(shares).node(),
                shares.id() == core.instance_id(),
                shares.node() == core.node(),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                self.rdma.qp.source().well_formed(),
                self.rdma.qp.source().source() == old(self).rdma.qp.source().source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                peer_core_step(&after_post, core, self.rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.host_mr == old(self).rdma.host_mr,
                self.rdma.gpu_mr == old(self).rdma.gpu_mr,
                self.rdma.shared_host_mr == old(self).rdma.shared_host_mr,
                self.rdma.shared_gpu_mr == old(self).rdma.shared_gpu_mr,
                pull.valid(&self.rdma, core),
                pull_tracked_or_done(&self.rdma, pull.wr_id, pull.writer@),
                self.channel.received().contains_key(g.tag),
                self.channel.received()[g.tag] is ReadShare,
                inbound_tag_unbuffered(self.pending_inbound_grants@, g.tag),
                pull.share_cap@.loc() == self.channel.received()[g.tag]->ReadShare_slot.loc,
                pull.share_cap@.share() == self.channel.received()[g.tag]->ReadShare_share_id,
                pull.share_cap@.frontier() == (foundation::tracked::Frontier {
                    last_writer: orig_share_frontier.last_writer,
                    readers: orig_share_frontier.readers.insert(pull.writer@),
                }),
                transfer::transport::covers(orig_share_frontier, core.citations()),
                pd matches Some(p) ==> p.id() == self.rdma.window_pool.pd() && p.node()
                    == core.node(),
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source().is_rdma_qp() && tokens@[k]@.node() == core.node(),
            decreases 1_000_000 - pull_spins,
        {
            if self.rdma.has_completed_pull(pull.wr_id) {
                break;
            }
            let ghost pstep1 = *core;
            let ghost pstep1_leases = self.rdma.active_leases@;
            let ghost pstep1_rdma = self.rdma;
            let revoke_ack_res = self.send_pending_revoke_acks(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(&after_post, &pstep1, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &pstep1,
                    core,
                    old(self).rdma.active_leases@,
                    pstep1_leases,
                    self.rdma.active_leases@,
                );
                lemma_pull_valid_after_peer_step(
                    &pstep1_rdma,
                    &self.rdma,
                    &pstep1,
                    core,
                    &pull,
                    None,
                );
            }
            if revoke_ack_res.is_err() {
                break;
            }
            let ghost pstep2 = *core;
            let ghost pstep2_leases = self.rdma.active_leases@;
            let ghost pstep2_rdma = self.rdma;
            self.queue_pending_inbound_fetches(
                core,
                primary_tier,
                fallback_tier,
                Tracked(shares),
                Tracked(instance),
            );
            proof {
                lemma_peer_core_step_trans(&after_post, &pstep2, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &pstep2,
                    core,
                    old(self).rdma.active_leases@,
                    pstep2_leases,
                    self.rdma.active_leases@,
                );
                lemma_pull_valid_after_peer_step(
                    &pstep2_rdma,
                    &self.rdma,
                    &pstep2,
                    core,
                    &pull,
                    None,
                );
            }
            let ghost pstep3 = *core;
            let ghost pstep3_leases = self.rdma.active_leases@;
            let ghost pstep3_rdma = self.rdma;
            let cq_res = self.rdma.pump_cq_all(core, Tracked(instance));
            proof {
                reveal(rdma_leases_valid);
                reveal(FrameTable::same_layout);
                lemma_peer_core_step_trans(&after_post, &pstep3, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &pstep3,
                    core,
                    old(self).rdma.active_leases@,
                    pstep3_leases,
                    self.rdma.active_leases@,
                );
                lemma_pull_valid_after_peer_step(
                    &pstep3_rdma,
                    &self.rdma,
                    &pstep3,
                    core,
                    &pull,
                    None,
                );
                assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                    let ag = #[trigger] self.active_grants@[i];
                    &&& ag.grant_tag < self.next_grant_tag
                    &&& ag.frame.0 < core.frames_spec().metas_spec().len()
                    &&& self.rdma.lease_index_spec(ag.handle) matches Some(idx) && {
                        let lease = self.rdma.active_leases@[idx];
                        &&& lease.frame == ag.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& self.channel.sent().contains_key(ag.grant_tag)
                        &&& self.channel.sent()[ag.grant_tag] == (GrantKind::ReadShare {
                            slot: core.frames_spec().frame(ag.frame).slot,
                            share_id: lease.share@,
                        })
                    }
                } by {
                    let ag = self.active_grants@[i];
                    let oidx = rdma_lease_index_spec(pstep3_leases, ag.handle).unwrap();
                    assert(pstep3_leases[oidx].state == RemoteLeaseState::Bound);
                    assert(!pstep3_leases[oidx].has_active_wr());
                }
                lemma_outbound_grants_preserved(
                    self.pending_outbound_grants@,
                    pstep3_leases,
                    self.rdma.active_leases@,
                    &pstep3,
                    core,
                );
                lemma_pump_cq_all_preserves_pull_tracked(&pstep3_rdma, &self.rdma);
            }
            if cq_res.is_err() {
                break;
            }
            let ghost pstep4 = *core;
            let ghost pstep4_leases = self.rdma.active_leases@;
            let ghost pstep4_rdma = self.rdma;
            let ob_res = self.process_pending_outbound_grants(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(&after_post, &pstep4, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &pstep4,
                    core,
                    old(self).rdma.active_leases@,
                    pstep4_leases,
                    self.rdma.active_leases@,
                );
                lemma_pull_valid_after_peer_step(
                    &pstep4_rdma,
                    &self.rdma,
                    &pstep4,
                    core,
                    &pull,
                    None,
                );
            }
            if ob_res.is_err() {
                break;
            }
            if let Some(pd_ref) = pd {
                let _ = self.replenish_acknowledged_windows(core, pd_ref, tokens);
            } else {
                let _ = self.acknowledge_pending_revoke_acks(core);
            }
            if self.rdma.has_completed_pull(pull.wr_id) {
                break;
            }
            let ghost pstep5 = *core;
            let ghost pstep5_leases = self.rdma.active_leases@;
            let ghost pstep5_rdma = self.rdma;
            let step_res = self.pump_cq_and_channel_step(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(&after_post, &pstep5, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &pstep5,
                    core,
                    old(self).rdma.active_leases@,
                    pstep5_leases,
                    self.rdma.active_leases@,
                );
                lemma_pull_valid_after_peer_step(
                    &pstep5_rdma,
                    &self.rdma,
                    &pstep5,
                    core,
                    &pull,
                    None,
                );
                assert(self.channel.received().contains_key(g.tag));
            }
            if step_res.is_err() {
                break;
            }
            pull_spins += 1;
        }
        let ghost before_complete = *core;
        let ghost before_complete_leases = self.rdma.active_leases@;
        proof {
            reveal(pull_tracked_or_done_state);
        }
        let tracked vb = g.version_binding.get().tracked_unwrap();
        let res = self.fetch_complete_pull(
            core,
            g.tag,
            req_id,
            pull,
            Tracked(vb),
            Ghost(orig_share_frontier),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans(
                &before_post,
                &after_post,
                &before_complete,
                self.rdma.timeline_id,
            );
            lemma_peer_core_step_trans_retire_new_init(
                &before_post,
                &before_complete,
                core,
                self.rdma.timeline_id,
                pull_frame,
            );
            lemma_peer_core_step_trans(old(core), &before_post, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_complete,
                core,
                old(self).rdma.active_leases@,
                before_complete_leases,
                self.rdma.active_leases@,
            );
        }
        res
    }

    /// Donor: mark an active grant as `soft_revoking` and send a cooperative
    /// `ControlPayload::RevokeNotice { grant_tag }` advising the peer to return
    /// `ReadRelease` promptly before unilateral MW invalidation escalates.
    pub(crate) fn send_soft_revoke_notice(
        &mut self,
        core: &CacheCore,
        grant_tag: GrantTag,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
        ensures
            final(self).well_formed(core),
            final(self).rdma == old(self).rdma,
            final(self).channel.same_session(old(self).channel),
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).next_request_id == old(self).next_request_id,
            final(self).active_grants@.len() == old(self).active_grants@.len(),
            forall|i: int|
                0 <= i < old(self).active_grants@.len() ==> {
                    &&& (#[trigger] final(self).active_grants@[i]).grant_tag == old(
                        self,
                    ).active_grants@[i].grant_tag
                    &&& final(self).active_grants@[i].handle == old(self).active_grants@[i].handle
                    &&& final(self).active_grants@[i].frame == old(self).active_grants@[i].frame
                },
    {
        let Some(pos) = self.find_grant_index(grant_tag) else {
            return Err(CoreError::InvalidArgument);
        };
        let g = self.active_grants[pos];
        self.active_grants.set(
            pos,
            ActiveGrant {
                grant_tag: g.grant_tag,
                handle: g.handle,
                frame: g.frame,
                soft_revoking: true,
            },
        );
        if self.send_control(
            core,
            ControlPayload::RevokeNotice { grant_tag },
            Tracked(instance),
        ).is_err() {
            return Err(rdma_error(rdma::RdmaError(-1)));
        }
        Ok(())
    }

    /// Two-phase cooperative-first revocation: sends `RevokeNotice { grant_tag }`,
    /// pumps CQ + channel steps for up to `spin_budget` iterations hoping the
    /// peer responds with `ReadRelease` (`Ok(None)` — zero MW quarantine), and
    /// escalates cleanly to unilateral fenced MW invalidation (`Ok(Some(mw_slot))`)
    /// if the grace window expires or the channel closes.
    #[verifier::rlimit(250)]
    pub(crate) fn revoke_lease_cooperative_first(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        grant_tag: GrantTag,
        spin_budget: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Option<u32>, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            r.is_ok() ==> final(self).active_grants@.len() < old(self).active_grants@.len(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        let Some(init_pos) = self.find_grant_index(grant_tag) else {
            return Err(CoreError::InvalidArgument);
        };
        let _ = self.send_soft_revoke_notice(core, grant_tag, Tracked(instance));
        proof {
            assert(self.active_grants@[init_pos as int].grant_tag == grant_tag);
        }
        let ghost post_notice_len = self.active_grants@.len();
        let mut spins: u32 = 0;
        while spins < spin_budget
            invariant
                spins <= spin_budget,
                post_notice_len == old(self).active_grants@.len(),
                core.well_formed(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.active_grants@.len() <= post_notice_len,
                self.active_grants@.len() == post_notice_len ==> exists|i: int|
                    0 <= i < self.active_grants@.len() && (
                    #[trigger] self.active_grants@[i]).grant_tag == grant_tag,
            decreases spin_budget - spins,
        {
            if self.find_grant_index(grant_tag).is_none() {
                return Ok(None);
            }
            let ghost step_core = *core;
            let ghost step_leases = self.rdma.active_leases@;
            let ghost step_grants = self.active_grants;
            match self.pump_cq_and_channel_step(core, Tracked(instance)) {
                Ok(_) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &step_core,
                            core,
                            old(self).rdma.active_leases@,
                            step_leases,
                            self.rdma.active_leases@,
                        );
                        if self.active_grants@.len() == post_notice_len {
                            assert(self.active_grants == step_grants);
                        }
                    }
                    if self.find_grant_index(grant_tag).is_none() {
                        return Ok(None);
                    }
                    spins += 1;
                },
                Err(_) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &step_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &step_core,
                            core,
                            old(self).rdma.active_leases@,
                            step_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    break;
                },
            }
        }
        if self.find_grant_index(grant_tag).is_none() {
            return Ok(None);
        }
        let ghost before_uni = *core;
        let ghost before_uni_leases = self.rdma.active_leases@;
        let uni_res = self.revoke_lease_unilateral(core, host, grant_tag, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before_uni, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_uni,
                core,
                old(self).rdma.active_leases@,
                before_uni_leases,
                self.rdma.active_leases@,
            );
        }
        let mw_slot = uni_res?;
        Ok(Some(mw_slot))
    }

    /// Donor: drain and revoke all grants referencing `frame` (marking any
    /// in-flight `PendingOutboundGrant` for `frame` as cancelled `send_on_ready = false`,
    /// pumping to complete pending binds if ready, then cooperatively-first revoking
    /// every active grant for `frame`).
    pub(crate) fn drain_and_evict_frame_leases(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        frame: FrameId,
        spin_budget: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            r.is_ok() ==> forall|i: int|
                0 <= i < final(self).active_grants@.len() ==> (
                #[trigger] final(self).active_grants@[i]).frame != frame,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        // Step 1: cancel `send_on_ready` on any queued outbound binds for `frame`.
        let mut p_idx: usize = 0;
        while p_idx < self.pending_outbound_grants.len()
            invariant
                p_idx <= self.pending_outbound_grants@.len(),
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                core.timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
                core.timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
                core.timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
                core.timelines_spec().next_seq_spec() == old(core).timelines_spec().next_seq_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
            decreases self.pending_outbound_grants@.len() - p_idx,
        {
            let grant = &self.pending_outbound_grants[p_idx];
            if grant.frame == frame && grant.send_on_ready {
                self.pending_inbound_fetches.push((grant.request_id, grant.key));
                self.pending_outbound_grants[p_idx].send_on_ready = false;
            }
            p_idx += 1;
        }
        // Step 2: drive `pump_cq` + `process_pending_outbound_grants` up to `spin_budget`
        // so ready binds for `frame` are unbound immediately.
        let mut drain_spins: u32 = 0;
        while drain_spins < spin_budget
            invariant
                drain_spins <= spin_budget,
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
            decreases spin_budget - drain_spins,
        {
            let mut has_pending_for_frame = false;
            let mut k: usize = 0;
            while k < self.pending_outbound_grants.len()
                invariant
                    k <= self.pending_outbound_grants@.len(),
                decreases self.pending_outbound_grants@.len() - k,
            {
                if self.pending_outbound_grants[k].frame == frame {
                    has_pending_for_frame = true;
                    break;
                }
                k += 1;
            }
            if !has_pending_for_frame {
                break;
            }
            let ghost step_core = *core;
            let ghost step_leases = self.rdma.active_leases@;
            let _ = self.pump_cq_and_channel_step(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(old(core), &step_core, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &step_core,
                    core,
                    old(self).rdma.active_leases@,
                    step_leases,
                    self.rdma.active_leases@,
                );
            }
            let ghost step2_core = *core;
            let ghost step2_leases = self.rdma.active_leases@;
            if let Err(e) = self.process_pending_outbound_grants(core, Tracked(instance)) {
                proof {
                    lemma_peer_core_step_trans(old(core), &step2_core, core, self.rdma.timeline_id);
                    lemma_foreign_shares_trans(
                        old(core),
                        &step2_core,
                        core,
                        old(self).rdma.active_leases@,
                        step2_leases,
                        self.rdma.active_leases@,
                    );
                }
                return Err(e);
            }
            proof {
                lemma_peer_core_step_trans(old(core), &step2_core, core, self.rdma.timeline_id);
                lemma_foreign_shares_trans(
                    old(core),
                    &step2_core,
                    core,
                    old(self).rdma.active_leases@,
                    step2_leases,
                    self.rdma.active_leases@,
                );
            }
            drain_spins += 1;
        }
        // Step 3: cooperatively-first revoke every active grant on `frame`.
        let mut evicted: usize = 0;
        loop
            invariant
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                self.rdma.qp.ready(),
                self.rdma.qp.state() is ReadyToSend,
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
            decreases self.active_grants@.len(),
        {
            let mut found_tag: Option<GrantTag> = None;
            let mut i: usize = 0;
            while i < self.active_grants.len() && found_tag.is_none()
                invariant
                    i <= self.active_grants@.len(),
                    found_tag is None ==> forall|j: int|
                        0 <= j < i ==> (#[trigger] self.active_grants@[j]).frame != frame,
                decreases
                        (if found_tag.is_none() {
                            1int
                        } else {
                            0int
                        }),
                        self.active_grants@.len() - i,
            {
                if self.active_grants[i].frame.0 == frame.0 {
                    found_tag = Some(self.active_grants[i].grant_tag);
                } else {
                    proof {
                        assert(self.active_grants@[i as int].frame != frame);
                    }
                    i += 1;
                }
            }
            let Some(grant_tag) = found_tag else {
                return Ok(evicted);
            };
            let ghost before_rev = *core;
            let ghost before_rev_leases = self.rdma.active_leases@;
            match self.revoke_lease_cooperative_first(
                core,
                host,
                grant_tag,
                spin_budget,
                Tracked(instance),
            ) {
                Ok(_) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &before_rev,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &before_rev,
                            core,
                            old(self).rdma.active_leases@,
                            before_rev_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    if evicted < usize::MAX {
                        evicted += 1;
                    }
                },
                Err(e) => {
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &before_rev,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &before_rev,
                            core,
                            old(self).rdma.active_leases@,
                            before_rev_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    return Err(e);
                },
            }
        }
    }

    /// Reset the QP (`IBV_QPS_RESET`), clear/recover all active remote leases across
    /// all 8 `RemoteLeaseState` variants (`recover_all_leases_on_qp_reset`), drain
    /// stale buffered grants, bump `reconnect_epoch`, send `ReconnectBarrier` +
    /// local `QueuePairEndpoint`, receive pre-barrier frames + peer endpoint via
    /// `Channel::recv_endpoint_or_any`, and reconnect the QP to `ReadyToSend`.
    #[verifier::rlimit(250)]
    pub fn reset_and_reconnect(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        local_endpoint: &rdma::QueuePairEndpoint,
        psn: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            old(self).rdma_spec().cq_spec().in_flight().is_empty(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend || old(
                self,
            ).rdma_spec().qp_spec().state() is Error,
            old(self).rdma_spec().qp_spec().peer() == old(self).rdma_spec().peer_node_spec(),
            old(self).rdma_spec().qp_spec().pd_id() == old(
                self,
            ).rdma_spec().window_pool_spec().pd(),
            old(self).rdma_spec().qp_spec().source().well_formed(),
            old(self).rdma_spec().qp_spec().source().source().is_rdma_qp(),
            old(self).rdma_spec().qp_spec().source().node() != old(
                self,
            ).rdma_spec().peer_node_spec(),
            local_endpoint.node() == old(core).node(),
            psn <= 0xffffff,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().peer_node_spec() == old(self).rdma_spec().peer_node_spec(),
            final(self).rdma_spec().window_pool_spec().pd() == old(
                self,
            ).rdma_spec().window_pool_spec().pd(),
            final(self).rdma_spec().qp_spec().peer() == old(self).rdma_spec().peer_node_spec(),
            final(self).rdma_spec().qp_spec().pd_id() == old(
                self,
            ).rdma_spec().window_pool_spec().pd(),
            final(self).rdma_spec().qp_spec().source().well_formed(),
            final(self).rdma_spec().qp_spec().source().source().is_rdma_qp(),
            final(self).rdma_spec().qp_spec().source().node() != old(
                self,
            ).rdma_spec().peer_node_spec(),
            r.is_ok() ==> final(self).rdma_spec().qp_spec().ready()
                && final(self).rdma_spec().qp_spec().state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            lemma_foreign_shares_refl(core, self.rdma.active_leases@);
        }
        let Tracked(qp_reset) = self.rdma.reset_qp(core)?;
        self.active_grants.clear();
        self.pending_outbound_grants.clear();
        self.pending_inbound_fetches.clear();
        self.pending_inbound_misses.clear();
        self.pending_revoke_notices.clear();
        self.pending_revoke_acks.clear();
        self.pending_reconnect_barriers.clear();
        self.quarantined_grants.clear();
        // Recover/revoke all active remote leases across all 8 `RemoteLeaseState` variants.
        self.rdma.recover_all_leases_on_qp_reset(
            core,
            host,
            Tracked(&qp_reset),
            Tracked(instance),
        )?;
        // Drain any stale pre-reset `PendingInboundGrant` entries cleanly.
        let ghost before_drain = *core;
        let ghost before_drain_leases = self.rdma.active_leases@;
        self.drain_stale_inbound_grants(core, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before_drain, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_drain,
                core,
                old(self).rdma.active_leases@,
                before_drain_leases,
                self.rdma.active_leases@,
            );
        }
        if self.reconnect_epoch < u64::MAX {
            self.reconnect_epoch += 1;
        }
        let epoch = self.reconnect_epoch;
        if self.send_control(
            core,
            ControlPayload::ReconnectBarrier { epoch },
            Tracked(instance),
        ).is_err() {
            return Err(rdma_error(rdma::RdmaError(-1)));
        }
        if self.channel.send_endpoint(local_endpoint).is_err() {
            return Err(rdma_error(rdma::RdmaError(-1)));
        }
        let mut peer_ep_opt: Option<rdma::QueuePairEndpoint> = None;
        let mut spins: u32 = 0;
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.instance_id() == old(core).instance_id(),
                core.tiers_spec() == old(core).tiers_spec(),
                core.frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
                core.version_registry_spec() == old(core).version_registry_spec(),
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                peer_foreign_shares_preserved(
                    old(core),
                    core,
                    old(self).rdma.active_leases@,
                    self.rdma.active_leases@,
                ),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.rdma.qp.state() == rdma::QueuePairState::Reset,
                self.rdma.qp.peer() == self.rdma.peer_node,
                self.rdma.qp.pd_id() == self.rdma.window_pool.pd(),
                self.rdma.qp.source().well_formed(),
                self.rdma.qp.source().source().is_rdma_qp(),
                self.rdma.qp.source().node() != self.rdma.peer_node,
                peer_ep_opt matches Some(ep) ==> ep.node() == self.rdma.peer_node,
            decreases 1_000_000 - spins,
        {
            let ghost pre_chan = self.channel;
            let ghost iter_core = *core;
            let ghost iter_leases = self.rdma.active_leases@;
            match self.channel.recv_endpoint_or_any(Tracked(instance)) {
                Err(_) => return Err(rdma_error(rdma::RdmaError(-1))),
                Ok(EndpointOrFrame::Endpoint(ep)) => {
                    peer_ep_opt = Some(ep);
                    break;
                },
                Ok(EndpointOrFrame::Frame(frame)) => {
                    let route_res = self.route_received_frame(
                        core,
                        Ghost(pre_chan),
                        frame,
                        false,
                        Tracked(instance),
                    );
                    proof {
                        lemma_peer_core_step_trans(
                            old(core),
                            &iter_core,
                            core,
                            self.rdma.timeline_id,
                        );
                        lemma_foreign_shares_trans(
                            old(core),
                            &iter_core,
                            core,
                            old(self).rdma.active_leases@,
                            iter_leases,
                            self.rdma.active_leases@,
                        );
                    }
                    if let Err(e) = route_res {
                        return Err(e);
                    }
                    spins += 1;
                },
            }
        }
        let Some(peer_ep) = peer_ep_opt else {
            return Err(rdma_error(rdma::RdmaError(11)));
        };
        let ghost before_post_drain = *core;
        let ghost before_post_drain_leases = self.rdma.active_leases@;
        self.drain_stale_inbound_grants(core, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before_post_drain, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_post_drain,
                core,
                old(self).rdma.active_leases@,
                before_post_drain_leases,
                self.rdma.active_leases@,
            );
        }
        self.pending_inbound_fetches.clear();
        self.pending_inbound_misses.clear();
        self.pending_revoke_notices.clear();
        self.pending_revoke_acks.clear();
        self.rdma.reconnect_qp(core, psn, peer_ep)
    }

    /// Full RDMA + session teardown: cleanly drains buffered inbound grants,
    /// clears session queues, transitions QP to Error, drains CQ, resets QP,
    /// recovers all active leases across all 8 states, deallocate all Type-2
    /// MWs in `window_pool`, and releases owned/shared MR handles.
    pub fn shutdown(
        &mut self,
        core: &mut CacheCore,
        host: &mut SyncContext,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().peer_node_spec() == old(self).rdma_spec().peer_node_spec(),
            final(self).rdma_spec().window_pool_spec().pd() == old(
                self,
            ).rdma_spec().window_pool_spec().pd(),
            r.is_ok() ==> {
                &&& final(self).rdma_spec().active_leases_spec()@.len() == 0
                &&& final(self).rdma_spec().qp_spec().state() == rdma::QueuePairState::Reset
                &&& final(self).rdma_spec().cq_spec().in_flight().is_empty()
                &&& final(self).rdma_spec().window_pool_spec().free_spec()@.len() == 0
                &&& final(self).rdma_spec().host_mr_spec() is None
                &&& final(self).rdma_spec().gpu_mr_spec() is None
                &&& final(self).rdma_spec().shared_host_mr_spec() is None
                &&& final(self).rdma_spec().shared_gpu_mr_spec() is None
                &&& final(self).active_grants_spec()@.len() == 0
                &&& final(self).pending_outbound_grants_spec()@.len() == 0
                &&& final(self).pending_inbound_grants_spec()@.len() == 0
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        self.drain_stale_inbound_grants(core, Tracked(instance));
        self.active_grants.clear();
        self.pending_outbound_grants.clear();
        self.pending_inbound_fetches.clear();
        self.pending_inbound_misses.clear();
        self.pending_revoke_notices.clear();
        self.pending_revoke_acks.clear();
        self.pending_reconnect_barriers.clear();
        self.quarantined_grants.clear();
        let ghost before_sd = *core;
        let ghost before_sd_leases = self.rdma.active_leases@;
        let res = self.rdma.shutdown(core, host, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(old(core), &before_sd, core, self.rdma.timeline_id);
            lemma_foreign_shares_trans(
                old(core),
                &before_sd,
                core,
                old(self).rdma.active_leases@,
                before_sd_leases,
                self.rdma.active_leases@,
            );
        }
        res
    }

    /// Single-tier compatibility wrapper over `serve_fetch_request_multi`.
    pub fn serve_fetch_request(
        &mut self,
        core: &mut CacheCore,
        tier: TierId,
        request_id: u64,
        key: ObjectKey,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma_spec().qp_spec().ready(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).next_grant_tag_spec() < u64::MAX,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().qp_spec() == old(self).rdma_spec().qp_spec(),
    {
        self.serve_fetch_request_multi(
            core,
            tier,
            None,
            request_id,
            key,
            Tracked(shares),
            Tracked(instance),
        )
    }

    /// Single-tier compatibility wrapper over `poll_progress_multi`.
    pub fn poll_progress(
        &mut self,
        core: &mut CacheCore,
        serve_tier: TierId,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma_spec().qp_spec().ready(),
            old(self).rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            pd matches Some(p) ==> p.id() == old(self).rdma_spec().window_pool_spec().pd()
                && p.node() == old(core).node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).timelines_spec().count_spec() == old(core).timelines_spec().count_spec(),
            final(core).timelines_spec().kinds_spec() == old(core).timelines_spec().kinds_spec(),
            final(core).timelines_spec().events_spec() == old(core).timelines_spec().events_spec(),
            final(core).timelines_spec().next_seq_spec() == old(
                core,
            ).timelines_spec().next_seq_spec(),
            peer_core_step(old(core), final(core), old(self).rdma_spec().timeline_id_spec()),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma_spec().active_leases_spec()@,
                final(self).rdma_spec().active_leases_spec()@,
            ),
            final(self).rdma_spec().timeline_id_spec() == old(self).rdma_spec().timeline_id_spec(),
            final(self).rdma_spec().peer_node_spec() == old(self).rdma_spec().peer_node_spec(),
            final(self).rdma_spec().qp_spec() == old(self).rdma_spec().qp_spec(),
            final(self).rdma_spec().host_mr_spec() == old(self).rdma_spec().host_mr_spec(),
            final(self).rdma_spec().gpu_mr_spec() == old(self).rdma_spec().gpu_mr_spec(),
            final(self).rdma_spec().shared_host_mr_spec() == old(
                self,
            ).rdma_spec().shared_host_mr_spec(),
            final(self).rdma_spec().shared_gpu_mr_spec() == old(
                self,
            ).rdma_spec().shared_gpu_mr_spec(),
            final(self).channel_spec().same_session(old(self).channel_spec()),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    core,
                ).node(),
            forall|seq: u64, writer: TransferEvent| #[trigger]
                pull_tracked_or_done(&old(self).rdma_spec(), seq, writer) ==> pull_tracked_or_done(
                    &final(self).rdma_spec(),
                    seq,
                    writer,
                ),
    {
        self.poll_progress_multi(
            core,
            serve_tier,
            None,
            pd,
            tokens,
            Tracked(shares),
            Tracked(instance),
        )
    }

    /// Single-tier compatibility wrapper over `fetch_remote_frame_duplex_multi`.
    pub(crate) fn fetch_remote_frame_duplex(
        &mut self,
        core: &mut CacheCore,
        serve_tier: TierId,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(self).next_request_id < u64::MAX,
            old(core).frames_spec().tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            pd matches Some(p) ==> p.id() == old(self).rdma.window_pool.pd() && p.node() == old(
                core,
            ).node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(core).node(),
            match target {
                PullTarget::Host => old(self).rdma.effective_host_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma.effective_gpu_mr_spec() matches Some(s)
                    && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            peer_foreign_shares_preserved(
                old(core),
                final(core),
                old(self).rdma.active_leases@,
                final(self).rdma.active_leases@,
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    core,
                ).node(),
    {
        self.fetch_remote_frame_duplex_multi(
            core,
            serve_tier,
            None,
            pd,
            tokens,
            key,
            dst_tier,
            target,
            Tracked(shares),
            Tracked(instance),
        )
    }
}

} // verus!
