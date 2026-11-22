//! Verified Milestone 3 RDMA peer session (`PeerSession<'a, T>`):
//! exclusively uses `Channel::recv_any` with constructive `VeriCacheAttachment`
//! frames, driving Donor (`serve_fetch_request`, `handle_read_release`,
//! `revoke_lease_unilateral`), Initiator (`send_fetch_request`,
//! `fetch_remote_frame`), and QP recovery (`reset_and_reconnect`).
use super::protocol::{
    decode_control_payload, decode_read_grant_meta, decode_read_release_meta,
    encode_control_payload, encode_read_grant_meta, encode_read_release_meta, ControlPayload,
    ReadGrantMeta, ReadReleaseMeta, VeriCacheAttachment,
};
#[cfg(verus_only)]
use super::protocol::{lemma_replica_valid_to_statement, lemma_statement_valid_in_receiver_ledger};
use super::pull::{finish_pull_read, post_pull_read, InFlightRdmaPull, PullOutcome, PullTarget};
use super::*;
#[cfg(verus_only)]
use crate::frame::STATE_PUBLISHED;
#[cfg(verus_only)]
use crate::provenance::{
    complete_receipts, event_succeeded, replica_valid, revoked_submap, ValueCertificate,
};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, knowledge_of, ordered_or_absent};
#[cfg(verus_only)]
use transfer::context::cluster_version_instance_id;
use transfer::context::{RevokedMap, ShareCap, ShareSource, TransferInstance, VersionBinding};
use transfer::host::HostContext;
use transfer::rdma;
use transfer::transport::{
    ByteTransport, Channel, ChannelAttachment, ChannelError, GrantKind, GrantTag, Received,
    ReceivedFrame, Statement,
};
use vstd::prelude::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

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
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.frames.same_directory(&before_core.frames),
        after_core.frames.same_other_pins(&before_core.frames, None),
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
        assert(after_core.frames.frame(f).slot == before_core.frames.frame(f).slot);
        assert(after_core.frames.published(f) == before_core.frames.published(f));
        assert(after_core.frames.frame(f).remote_shares == before_core.frames.frame(
            f,
        ).remote_shares);
        assert(after_core.frames.meta(f).active_remote_leases == before_core.frames.meta(
            f,
        ).active_remote_leases);
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
        after_core.frames.same_directory(&before_core.frames),
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
                &&& g.frame.0 < before_core.frames.metas.len()
                &&& rdma_lease_index_spec(before_leases, g.handle) matches Some(idx) && {
                    let lease = before_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: before_core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
    ensures
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < after_core.frames.metas.len()
                &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
                    let lease = after_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: after_core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
{
    reveal(FrameTable::same_directory);
    assert forall|i: int| 0 <= i < grants.len() implies {
        let g = #[trigger] grants[i];
        &&& g.grant_tag < next_grant_tag
        &&& g.frame.0 < after_core.frames.metas.len()
        &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
            let lease = after_leases[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& sent.contains_key(g.grant_tag)
            &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                slot: after_core.frames.frame(g.frame).slot,
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
        after_core.frames.same_directory(&before_core.frames),
        rdma_leases_disjoint(after_leases),
        rdma_leases_preserved_except(before_leases, after_leases, removed_handle),
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.handle != removed_handle
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < before_core.frames.metas.len()
                &&& rdma_lease_index_spec(before_leases, g.handle) matches Some(idx) && {
                    let lease = before_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: before_core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
    ensures
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < after_core.frames.metas.len()
                &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
                    let lease = after_leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: after_core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
{
    reveal(FrameTable::same_directory);
    assert forall|i: int| 0 <= i < grants.len() implies {
        let g = #[trigger] grants[i];
        &&& g.grant_tag < next_grant_tag
        &&& g.frame.0 < after_core.frames.metas.len()
        &&& rdma_lease_index_spec(after_leases, g.handle) matches Some(idx) && {
            let lease = after_leases[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& sent.contains_key(g.grant_tag)
            &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                slot: after_core.frames.frame(g.frame).slot,
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
        after_core.frames.same_layout(&before_core.frames),
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < before_core.frames.metas.len()
                &&& rdma_lease_index_spec(leases, g.handle) matches Some(idx) && {
                    let lease = leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: before_core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
    ensures
        forall|i: int|
            0 <= i < grants.len() ==> {
                let g = #[trigger] grants[i];
                &&& g.grant_tag < next_grant_tag
                &&& g.frame.0 < after_core.frames.metas.len()
                &&& rdma_lease_index_spec(leases, g.handle) matches Some(idx) && {
                    let lease = leases[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& sent.contains_key(g.grant_tag)
                    &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                        slot: after_core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            },
{
    reveal(FrameTable::same_layout);
    assert forall|i: int| 0 <= i < grants.len() implies {
        let g = #[trigger] grants[i];
        &&& g.grant_tag < next_grant_tag
        &&& g.frame.0 < after_core.frames.metas.len()
        &&& rdma_lease_index_spec(leases, g.handle) matches Some(idx) && {
            let lease = leases[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& sent.contains_key(g.grant_tag)
            &&& sent[g.grant_tag] == (GrantKind::ReadShare {
                slot: after_core.frames.frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = grants[i];
        assert(after_core.frames.frame(g.frame).slot == before_core.frames.frame(g.frame).slot);
    }
}

/// Donor-side record mapping a sent `GrantTag` to its bound RDMA lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveGrant {
    pub grant_tag: GrantTag,
    pub handle: RemoteLeaseHandle,
    pub frame: FrameId,
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

/// Combined RDMA backend + authenticated `Channel<T, VeriCacheAttachment>` session
/// with one remote peer node.
pub struct PeerSession<'a, T: ByteTransport> {
    pub rdma: RdmaBackend<'a>,
    pub channel: Channel<T, VeriCacheAttachment>,
    pub next_grant_tag: u64,
    pub next_request_id: u64,
    pub active_grants: Vec<ActiveGrant>,
}

impl<'a, T: ByteTransport> PeerSession<'a, T> {
    pub open spec fn well_formed(&self, core: &CacheCore) -> bool {
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
                &&& g.frame.0 < core.frames.metas.len()
                &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                    let lease = self.rdma.active_leases@[idx];
                    &&& lease.frame == g.frame
                    &&& lease.state == RemoteLeaseState::Bound
                    &&& self.channel.sent().contains_key(g.grant_tag)
                    &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                        slot: core.frames.frame(g.frame).slot,
                        share_id: lease.share@,
                    })
                }
            }
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
            channel.peer() == rdma.peer_node,
            channel.id() == core.instance_id(),
            channel.sent() == Map::<GrantTag, GrantKind>::empty(),
        ensures
            s.well_formed(core),
            s.rdma == rdma,
            s.channel == channel,
            s.next_grant_tag == 1,
            s.next_request_id == 1,
            s.active_grants@.len() == 0,
    {
        let _ = core;
        Self { rdma, channel, next_grant_tag: 1, next_request_id: 1, active_grants: Vec::new() }
    }

    /// Receive and decode the next peer message using `Channel::recv_any` exclusively.
    pub fn recv_message(
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
                &&& share@.is_rdma_bound()
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
                &&& old(self).channel.sent().contains_key(tag)
                &&& final(self).channel.sent() == old(self).channel.sent().remove(tag)
                &&& final(self).channel.received() == old(self).channel.received()
                &&& old(self).channel.sent()[tag] matches GrantKind::ReadShare { slot, share_id }
                    && share@.is_rdma_bound() && share@.instance_id() == core.instance_id()
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
    pub fn send_control(
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
            final(self).channel.same_session(old(self).channel),
            final(self).channel.sent() == old(self).channel.sent(),
            final(self).channel.received() == old(self).channel.received(),
    {
        let bytes = encode_control_payload(payload);
        let ghost cited = core.citations();
        let tracked att = VeriCacheAttachment::Control { payload };
        proof {
            lemma_core_wf_facts(core);
        }
        match self.channel.send_notice_with(
            &bytes,
            Tracked(instance),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Tracked(&core.ledger.borrow().succeeded),
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
            old(self).next_request_id < u64::MAX,
        ensures
            final(self).well_formed(core),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).rdma == old(self).rdma,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).active_grants == old(self).active_grants,
            final(self).channel.same_session(old(self).channel),
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
    pub open spec fn lease_is_binding(
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
    pub open spec fn binding_lease(
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
    fn serve_bind(
        &mut self,
        core: &mut CacheCore,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Option<(RemoteLeaseHandle, u64)>)
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).version_registry == old(core).version_registry,
            final(core).frames.same_directory(&old(core).frames),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).active_grants == old(self).active_grants,
            r matches Some((handle, wr_id)) ==> final(core).frames.published(frame) matches Some(
                (_, _, c, _),
            ) && final(self).binding_lease(handle, frame, wr_id, c.writer),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_leases = self.rdma.active_leases@;
        match self.rdma.bind_remote_read_lease(core, frame, Tracked(shares), Tracked(instance)) {
            Ok((handle, _mw_slot, _gen, bind_wr_id)) => {
                proof {
                    reveal(rdma_leases_valid);
                    assert forall|i: int| 0 <= i < self.active_grants@.len() implies {
                        let g = #[trigger] self.active_grants@[i];
                        &&& g.grant_tag < self.next_grant_tag
                        &&& g.frame.0 < core.frames.metas.len()
                        &&& self.rdma.lease_index_spec(g.handle) matches Some(idx) && {
                            let lease = self.rdma.active_leases@[idx];
                            &&& lease.frame == g.frame
                            &&& lease.state == RemoteLeaseState::Bound
                            &&& self.channel.sent().contains_key(g.grant_tag)
                            &&& self.channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                                slot: core.frames.frame(g.frame).slot,
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
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                Some((handle, bind_wr_id))
            },
            Err(_) => {
                proof {
                    reveal(rdma_leases_valid);
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_directory);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                None
            },
        }
    }

    /// Donor helper: recover a failed MW bind (`Completed::BindFailed`).
    fn fail_bound_lease(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        witness: Tracked<rdma::BindFailedWitness>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).rdma.lease_index_spec(handle) matches Some(idx) && old(
                self,
            ).rdma.active_leases@[idx].state == RemoteLeaseState::Binding,
            old(self).rdma.lease_matches_window_state(
                handle,
                RemoteLeaseState::Binding,
            ) matches Some(w) ==> witness@.matches(w),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).version_registry == old(core).version_registry,
            final(core).frames.same_directory(&old(core).frames),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).active_grants == old(self).active_grants,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost pre_leases = self.rdma.active_leases@;
        let ghost pre_core = *core;
        proof {
            let hidx = rdma_lease_index_spec(pre_leases, handle).unwrap();
            assert forall|i: int| 0 <= i < self.active_grants@.len() implies (
            #[trigger] self.active_grants@[i]).handle != handle by {
                let g = self.active_grants@[i];
                if g.handle == handle {
                    assert(pre_leases[hidx].state == RemoteLeaseState::Bound);
                }
            }
        }
        if self.rdma.fail_bind(core, handle, witness, Tracked(instance)).is_ok() {
            proof {
                reveal(rdma_leases_valid);
                lemma_peer_grants_after_lease_remove(
                    self.active_grants@,
                    pre_leases,
                    self.rdma.active_leases@,
                    handle,
                    self.channel.sent(),
                    self.next_grant_tag,
                    &pre_core,
                    core,
                );
            }
        }
        proof {
            reveal(peer_core_step_except);
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
            reveal(FrameTable::preserves_initializing);
        }
    }

    /// Donor helper: poll the CQ until the bind posted as `bind_wr_id` completes.
    /// Returns `Ok(Some(witness))` once bound, `Ok(None)` on bind failure/timeout.
    fn serve_await_bind(
        &mut self,
        core: &mut CacheCore,
        handle: RemoteLeaseHandle,
        frame: FrameId,
        bind_wr_id: u64,
        Ghost(writer): Ghost<TransferEvent>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Option<Tracked<rdma::BindWitness>>, CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(self).binding_lease(handle, frame, bind_wr_id, writer),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).version_registry == old(core).version_registry,
            final(core).frames.same_directory(&old(core).frames),
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).next_grant_tag == old(self).next_grant_tag,
            final(self).active_grants == old(self).active_grants,
            r matches Ok(Some(bw)) ==> {
                &&& final(core).frames == old(core).frames
                &&& final(self).lease_is_binding(handle, frame, bind_wr_id, writer)
                &&& (final(self).rdma.lease_matches_window_state(
                    handle,
                    RemoteLeaseState::Binding,
                ) matches Some(w) ==> bw@.matches(w))
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut spins: u32 = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.version_registry == old(core).version_registry,
                core.frames == old(core).frames,
                self.rdma.qp == old(self).rdma.qp,
                self.next_grant_tag == old(self).next_grant_tag,
                self.active_grants == old(self).active_grants,
                self.rdma.active_leases == old(self).rdma.active_leases,
                self.rdma.window_pool == old(self).rdma.window_pool,
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.binding_lease(handle, frame, bind_wr_id, writer),
            decreases 1_000_000 - spins,
        {
            let ghost before_poll_core = *core;
            let ghost before_poll_cq = self.rdma.cq;
            let polled = self.rdma.poll_cq(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(
                    old(core),
                    &before_poll_core,
                    core,
                    self.rdma.timeline_id,
                );
            }
            let pc = match polled {
                Err(_) => {
                    proof {
                        reveal(FrameTable::same_directory);
                    }
                    return Err(rdma_error(rdma::RdmaError(-1)));
                },
                Ok(None) => None,
                Ok(Some(pc)) => Some(pc),
            };
            if let Some(pc) = pc {
                let ghost c = choose|c: rdma::Completed|
                    c.describes(before_poll_cq, self.rdma.cq, instance.id())
                        && #[trigger] RdmaBackend::delivers(c, pc);
                match pc {
                    PeerCompletion::Control(rdma::Completed::WindowBound { wr_id, witness }) => {
                        if wr_id == bind_wr_id {
                            proof {
                                lemma_poll_bind_witness_matches(
                                    &self.rdma,
                                    core,
                                    handle,
                                    bind_wr_id,
                                    before_poll_cq,
                                    c,
                                );
                                reveal(FrameTable::same_directory);
                            }
                            return Ok(Some(witness));
                        }
                    },
                    PeerCompletion::BindFailed { wr_id, status, witness } => {
                        if wr_id == bind_wr_id {
                            proof {
                                lemma_poll_bind_witness_matches(
                                    &self.rdma,
                                    core,
                                    handle,
                                    bind_wr_id,
                                    before_poll_cq,
                                    c,
                                );
                            }
                            let ghost before_fail_core = *core;
                            self.fail_bound_lease(core, handle, witness, Tracked(instance));
                            proof {
                                lemma_peer_core_step_trans(
                                    old(core),
                                    &before_fail_core,
                                    core,
                                    self.rdma.timeline_id,
                                );
                            }
                            return Ok(None);
                        }
                    },
                    PeerCompletion::Control(rdma::Completed::ControlFailed { wr_id, .. }) => {
                        if wr_id == bind_wr_id {
                            proof {
                                reveal(FrameTable::same_directory);
                            }
                            return Err(rdma_error(rdma::RdmaError(-1)));
                        }
                    },
                    _ => {},
                }
                proof {
                    lemma_poll_keeps_bind(
                        before_poll_cq,
                        self.rdma.cq,
                        c,
                        instance.id(),
                        bind_wr_id,
                    );
                }
            }
            spins += 1;
        }
        proof {
            reveal(FrameTable::same_directory);
        }
        Ok(None)
    }

    /// Donor helper: complete the MW bind and send `READ_GRANT` with a
    /// constructive `VeriCacheAttachment::ReadGrant`.
    fn serve_send_grant(
        &mut self,
        core: &CacheCore,
        handle: RemoteLeaseHandle,
        frame: FrameId,
        request_id: u64,
        key: ObjectKey,
        version: VersionId,
        bw: Tracked<rdma::BindWitness>,
        bind_wr_id: u64,
        Ghost(cert): Ghost<ValueCertificate>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            instance.id() == core.instance_id(),
            old(self).next_grant_tag < u64::MAX,
            old(self).lease_is_binding(handle, frame, bind_wr_id, cert.writer),
            old(self).rdma.lease_matches_window_state(
                handle,
                RemoteLeaseState::Binding,
            ) matches Some(w) ==> bw@.matches(w),
            frame.0 < core.frames.metas.len(),
            core.frames.published(frame) matches Some((_, v, c, _)) && v == version && c == cert,
            core.has_version_binding(version),
        ensures
            final(self).well_formed(core),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost pre_leases = self.rdma.active_leases@;
        let ghost hidx = rdma_lease_index_spec(pre_leases, handle).unwrap();
        proof {
            reveal(rdma_leases_valid);
            assert forall|i: int| 0 <= i < self.active_grants@.len() implies (
            #[trigger] self.active_grants@[i]).handle != handle by {
                let g = self.active_grants@[i];
                if g.handle == handle {
                    assert(pre_leases[hidx].state == RemoteLeaseState::Bound);
                }
            }
        }
        let (cap, Tracked(share_cap)) = self.rdma.complete_bind(core, handle, bw)?;
        proof {
            reveal(rdma_leases_valid);
            assert(core.frames.same_directory(&core.frames)) by {
                reveal(FrameTable::same_directory);
            }
            lemma_peer_grants_after_lease_update(
                self.active_grants@,
                pre_leases,
                self.rdma.active_leases@,
                handle,
                hidx,
                self.channel.sent(),
                self.next_grant_tag,
                core,
                core,
            );
        }
        let grant_tag = self.next_grant_tag;
        let meta = ReadGrantMeta {
            request_id,
            key,
            version,
            frame_size: core.frames.frame_size as u64,
        };
        let meta_bytes = encode_read_grant_meta(meta);
        let ghost cited = core.citations();
        proof {
            lemma_core_wf_facts(core);
        }
        let tracked revoked = instance.duplicate_revoked(&core.ledger.borrow().revoked);
        let Tracked(version_binding) = core.version_binding_for(version);
        proof {
            lemma_published_frame_replica_valid(core, frame);
            lemma_published_frame_slot_valid(core, frame);
            lemma_initializing_frame_slot_valid(core, frame);
            lemma_citations_closure(core);
            assert(cert == core.frames.frame(frame).cert.unwrap());
            lemma_replica_valid_to_statement(core.ledger@, core.node(), cert);
            assert(closure_from(core.ledger@.submitted.map(), cited).contains(cert.writer));
            assert(ordered_or_absent(
                share_cap.frontier().last_writer,
                closure_from(core.ledger@.submitted.map(), cited),
            ));
        }
        let tracked att = VeriCacheAttachment::ReadGrant { meta, cert, revoked, version_binding };
        proof {
            reveal(complete_receipts);
            assert forall|s: Statement|
                s.submitted.instance_id() == self.channel.id() && s.submitted.map()
                    == core.ledger@.submitted.map() && s.knows.instance_id() == self.channel.id()
                    && s.knows.set() == core.ledger@.knows.set() && s.succeeded
                    == core.ledger@.succeeded && s.facts == cited implies #[trigger] att.valid(
                self.channel.id(),
                self.channel.local(),
                1,
                grant_tag,
                meta_bytes@,
                Some(GrantKind::ReadShare { slot: cap.slot_ref(), share_id: share_cap.share() }),
                Some(cap),
                Some(share_cap.frontier()),
                s,
            ) by {
                super::protocol::lemma_statement_replica_valid_ext(
                    core.ledger@.submitted,
                    s.submitted,
                    core.ledger@.succeeded,
                    revoked,
                    cited,
                    cert,
                );
            }
        }
        match self.channel.send_remote_read_grant_with(
            grant_tag,
            &cap,
            &meta_bytes,
            Tracked(instance),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Tracked(&core.ledger.borrow().succeeded),
            Ghost(cited),
            Tracked(share_cap),
            Tracked(att),
        ) {
            Ok(()) => {
                self.next_grant_tag += 1;
                self.active_grants.push(ActiveGrant { grant_tag, handle, frame });
                Ok(ServeOutcome::Granted { grant_tag, handle })
            },
            Err(_) => Err(rdma_error(rdma::RdmaError(-1))),
        }
    }

    /// Donor: serve a `FetchRequest { request_id, key }` by looking up `key` in
    /// `tier`, binding an RDMA Type-2 MW (polling CQ until bind finishes), and
    /// sending a `READ_GRANT` with constructive `VeriCacheAttachment::ReadGrant`
    /// (or `FetchMiss` if absent or bind fails).
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let Some((frame, version)) = core.lookup_tier(key, tier) else {
            let _ = self.send_control(
                core,
                ControlPayload::FetchMiss { request_id },
                Tracked(instance),
            );
            proof {
                lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            }
            return Ok(ServeOutcome::Miss);
        };
        let ghost before_bind_core = *core;
        let Some((handle, bind_wr_id)) = self.serve_bind(
            core,
            frame,
            Tracked(shares),
            Tracked(instance),
        ) else {
            proof {
                reveal(FrameTable::same_directory);
            }
            let _ = self.send_control(
                core,
                ControlPayload::FetchMiss { request_id },
                Tracked(instance),
            );
            return Ok(ServeOutcome::Miss);
        };
        let ghost after_bind_core = *core;
        let ghost cert = core.frames.published(frame).unwrap().2;
        proof {
            reveal(FrameTable::same_directory);
            lemma_published_frame_replica_valid(core, frame);
            assert(core.frames.published(frame) == before_bind_core.frames.published(frame));
        }
        let bw = match self.serve_await_bind(
            core,
            handle,
            frame,
            bind_wr_id,
            Ghost(cert.writer),
            Tracked(instance),
        ) {
            Ok(Some(bw)) => bw,
            Ok(None) => {
                proof {
                    reveal(FrameTable::same_directory);
                    lemma_peer_core_step_trans(
                        old(core),
                        &after_bind_core,
                        core,
                        self.rdma.timeline_id,
                    );
                }
                let _ = self.send_control(
                    core,
                    ControlPayload::FetchMiss { request_id },
                    Tracked(instance),
                );
                return Ok(ServeOutcome::Miss);
            },
            Err(e) => {
                proof {
                    reveal(FrameTable::same_directory);
                    lemma_peer_core_step_trans(
                        old(core),
                        &after_bind_core,
                        core,
                        self.rdma.timeline_id,
                    );
                }
                return Err(e);
            },
        };
        proof {
            reveal(FrameTable::same_directory);
            lemma_peer_core_step_trans(old(core), &after_bind_core, core, self.rdma.timeline_id);
        }
        self.serve_send_grant(
            core,
            handle,
            frame,
            request_id,
            key,
            version,
            bw,
            bind_wr_id,
            Ghost(cert),
            Tracked(instance),
        )
    }

    /// Donor: handle an incoming `ReadRelease` by cooperatively unpinning and
    /// invalidating the bound MW (polling CQ for `WindowInvalidated`).
    pub fn handle_read_release(
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
                    &&& g.frame.0 < old(core).frames.metas.len()
                    &&& old(self).rdma.lease_index_spec(g.handle) matches Some(idx) && {
                        let lease = old(self).rdma.active_leases@[idx];
                        &&& lease.frame == g.frame
                        &&& lease.state == RemoteLeaseState::Bound
                        &&& (g.grant_tag != grant_tag ==> old(self).channel.sent().contains_key(
                            g.grant_tag,
                        ) && old(self).channel.sent()[g.grant_tag] == (GrantKind::ReadShare {
                            slot: old(core).frames.frame(g.frame).slot,
                            share_id: lease.share@,
                        }))
                        &&& (g.grant_tag == grant_tag ==> share_cap.is_rdma_bound()
                            && share_cap.instance_id() == old(core).instance_id() && share_cap.loc()
                            == old(core).frames.frame(g.frame).slot.loc && share_cap.share()
                            == lease.share@ && transfer::transport::covers(
                            share_cap.frontier(),
                            statement.closure(),
                        ))
                    }
                },
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames.versions@ == old(core).frames.versions@,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let Some(pos) = self.find_grant_index(grant_tag) else {
            proof {
                lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            }
            return Err(CoreError::InvalidArgument);
        };
        let grant = self.active_grants.swap_remove(pos);
        proof {
            lemma_swap_removed_grant(old(self).active_grants@, self.active_grants@, pos as int);
        }
        // Merge peer statement so `share_cap.frontier().readers` are in `core.citations()`.
        let ghost before_merge = *core;
        core.merge_peer_statement(
            Tracked(statement.submitted),
            Tracked(statement.knows),
            Tracked(statement.succeeded),
            Tracked(instance),
        );
        proof {
            lemma_rdma_window_after_fence_ledger(
                self.rdma.in_flight_ops,
                self.rdma.bitmap,
                &core.timelines,
                self.rdma.timeline_id,
                &before_merge.ledger@,
                &core.ledger@,
                rdma::CqSnapshot::of(self.rdma.cq),
                core.instance_id(),
                self.rdma.next_control_wr_id,
            );
            assert(core.frames.same_directory(&before_merge.frames)) by {
                reveal(FrameTable::same_directory);
            }
            lemma_rdma_slabs_same_directory(
                &self.rdma.host_mr,
                &self.rdma.gpu_mr,
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
        }
        let ghost pre_unbind_leases = self.rdma.active_leases@;
        let ghost pre_unbind_core = *core;
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
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                return Err(e);
            },
        };
        proof {
            reveal(rdma_leases_valid);
            let changed_idx = rdma_lease_index_spec(pre_unbind_leases, grant.handle).unwrap();
            lemma_peer_grants_after_lease_update(
                self.active_grants@,
                pre_unbind_leases,
                self.rdma.active_leases@,
                grant.handle,
                changed_idx,
                self.channel.sent(),
                self.next_grant_tag,
                &pre_unbind_core,
                core,
            );
            assert(peer_core_step(&before_merge, &pre_unbind_core, self.rdma.timeline_id)) by {
                reveal(peer_core_step_except);
                reveal(FrameTable::same_layout);
                reveal(FrameTable::preserves_initializing);
            }
            lemma_peer_core_step_trans(
                &before_merge,
                &pre_unbind_core,
                core,
                self.rdma.timeline_id,
            );
        }
        let ghost before_await = *core;
        // Poll CQ for `WindowInvalidated`.
        let res = self.await_unbind_invalidate(core, grant.handle, inv_wr_id, Tracked(instance));
        proof {
            lemma_peer_core_step_trans(&before_merge, &before_await, core, self.rdma.timeline_id);
            reveal(FrameTable::same_directory);
        }
        res
    }

    /// Donor helper: poll CQ for `WindowInvalidated` matching `inv_wr_id` and
    /// call `finish_unbind_lease`.
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
            old(self).rdma.cq.in_flight().contains_key(inv_wr_id),
            old(self).rdma.cq.in_flight()[inv_wr_id].kind is LocalInvalidate,
            old(self).rdma.lease_matches_window_state(
                handle,
                RemoteLeaseState::Unbinding,
            ) matches Some(w) ==> old(
                self,
            ).rdma.cq.in_flight()[inv_wr_id].kind->LocalInvalidate_window_id == w.id(),
            forall|i: int|
                0 <= i < old(self).active_grants@.len() ==> (#[trigger] old(
                    self,
                ).active_grants@[i]).handle != handle,
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames.versions@ == old(core).frames.versions@,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let mut spins: u32 = 0;
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.frames.versions@ == old(core).frames.versions@,
                core.version_registry == old(core).version_registry,
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.active_grants == old(self).active_grants,
                self.rdma.lease_index_spec(handle) matches Some(idx)
                    && self.rdma.active_leases@[idx].wr_id == inv_wr_id
                    && self.rdma.active_leases@[idx].state == RemoteLeaseState::Unbinding,
                self.rdma.cq.in_flight().contains_key(inv_wr_id),
                self.rdma.cq.in_flight()[inv_wr_id].kind is LocalInvalidate,
                self.rdma.lease_matches_window_state(
                    handle,
                    RemoteLeaseState::Unbinding,
                ) matches Some(w)
                    ==> self.rdma.cq.in_flight()[inv_wr_id].kind->LocalInvalidate_window_id
                    == w.id(),
            decreases 1_000_000 - spins,
        {
            let ghost before_poll_core = *core;
            let ghost before_poll_cq = self.rdma.cq;
            let poll_res = self.rdma.poll_cq(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(
                    old(core),
                    &before_poll_core,
                    core,
                    self.rdma.timeline_id,
                );
            }
            match poll_res {
                Ok(
                    Some(
                        PeerCompletion::Control(
                            rdma::Completed::WindowInvalidated { wr_id, witness, .. },
                        ),
                    ),
                ) => {
                    if wr_id == inv_wr_id {
                        proof {
                            let c = choose|c: rdma::Completed|
                                c.describes(before_poll_cq, self.rdma.cq, instance.id())
                                    && #[trigger] RdmaBackend::delivers(
                                    c,
                                    PeerCompletion::Control(
                                        rdma::Completed::WindowInvalidated { wr_id, witness },
                                    ),
                                );
                            lemma_poll_inv_witness_matches(
                                &self.rdma,
                                core,
                                handle,
                                RemoteLeaseState::Unbinding,
                                inv_wr_id,
                                before_poll_cq,
                                c,
                            );
                        }
                        let ghost pre_fin_leases = self.rdma.active_leases@;
                        let ghost pre_fin_core = *core;
                        if self.rdma.finish_unbind_lease(core, handle, witness).is_ok() {
                            proof {
                                reveal(rdma_leases_valid);
                                reveal(FrameTable::same_directory);
                                lemma_peer_grants_after_lease_remove(
                                    self.active_grants@,
                                    pre_fin_leases,
                                    self.rdma.active_leases@,
                                    handle,
                                    self.channel.sent(),
                                    self.next_grant_tag,
                                    &pre_fin_core,
                                    core,
                                );
                            }
                        }
                        return Ok(());
                    }
                },
                Ok(Some(PeerCompletion::Control(rdma::Completed::ControlFailed { wr_id, .. }))) => {
                    if wr_id == inv_wr_id {
                        return Err(rdma_error(rdma::RdmaError(-1)));
                    }
                },
                Ok(_) => {},
                Err(e) => return Err(e),
            }
            spins += 1;
        }
        Err(rdma_error(rdma::RdmaError(11)))
    }

    /// Donor: unilaterally revoke an active grant via fenced local MW
    /// invalidation (`Quarantined` until peer `RevokeAck`) and send `RevokeNotice`.
    pub fn revoke_lease_unilateral(
        &mut self,
        core: &mut CacheCore,
        host: &mut HostContext,
        grant_tag: GrantTag,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u32, CoreError>)
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames.versions@ == old(core).frames.versions@,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let Some(pos) = self.find_grant_index(grant_tag) else {
            proof {
                lemma_peer_core_step_refl(core, self.rdma.timeline_id);
            }
            return Err(CoreError::InvalidArgument);
        };
        let grant = self.active_grants.swap_remove(pos);
        proof {
            lemma_swap_removed_grant(old(self).active_grants@, self.active_grants@, pos as int);
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        let ghost pre_rev_leases = self.rdma.active_leases@;
        let ghost pre_rev_core = *core;
        let inv_wr_id = self.rdma.begin_fenced_invalidate(core, grant.handle)?;
        proof {
            reveal(rdma_leases_valid);
            reveal(FrameTable::same_directory);
            let changed_idx = rdma_lease_index_spec(pre_rev_leases, grant.handle).unwrap();
            lemma_peer_grants_after_lease_update(
                self.active_grants@,
                pre_rev_leases,
                self.rdma.active_leases@,
                grant.handle,
                changed_idx,
                self.channel.sent(),
                self.next_grant_tag,
                &pre_rev_core,
                core,
            );
        }
        let mut spins: u32 = 0;
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                instance.id() == core.instance_id(),
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.frames.versions@ == old(core).frames.versions@,
                core.version_registry == old(core).version_registry,
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.rdma.lease_index_spec(grant.handle) matches Some(idx)
                    && self.rdma.active_leases@[idx].wr_id == inv_wr_id
                    && self.rdma.active_leases@[idx].state == RemoteLeaseState::Revoking,
                self.rdma.cq.in_flight().contains_key(inv_wr_id),
                self.rdma.cq.in_flight()[inv_wr_id].kind is LocalInvalidate,
                self.rdma.lease_matches_window_state(
                    grant.handle,
                    RemoteLeaseState::Revoking,
                ) matches Some(w)
                    ==> self.rdma.cq.in_flight()[inv_wr_id].kind->LocalInvalidate_window_id
                    == w.id(),
            decreases 1_000_000 - spins,
        {
            let ghost before_poll_core = *core;
            let ghost before_poll_cq = self.rdma.cq;
            let poll_res = self.rdma.poll_cq(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(
                    old(core),
                    &before_poll_core,
                    core,
                    self.rdma.timeline_id,
                );
            }
            match poll_res {
                Ok(
                    Some(
                        PeerCompletion::Control(
                            rdma::Completed::WindowInvalidated { wr_id, witness, .. },
                        ),
                    ),
                ) => {
                    if wr_id == inv_wr_id {
                        proof {
                            let c = choose|c: rdma::Completed|
                                c.describes(before_poll_cq, self.rdma.cq, instance.id())
                                    && #[trigger] RdmaBackend::delivers(
                                    c,
                                    PeerCompletion::Control(
                                        rdma::Completed::WindowInvalidated { wr_id, witness },
                                    ),
                                );
                            lemma_poll_inv_witness_matches(
                                &self.rdma,
                                core,
                                grant.handle,
                                RemoteLeaseState::Revoking,
                                inv_wr_id,
                                before_poll_cq,
                                c,
                            );
                        }
                        let ghost pre_fin_leases = self.rdma.active_leases@;
                        let ghost pre_fin_core = *core;
                        let mw_slot = self.rdma.finish_fenced_revoke_with_invalidate(
                            core,
                            host,
                            grant.handle,
                            witness,
                            Tracked(instance),
                        )?;
                        proof {
                            reveal(rdma_leases_valid);
                            reveal(FrameTable::same_directory);
                            lemma_peer_core_step_trans(
                                old(core),
                                &pre_fin_core,
                                core,
                                self.rdma.timeline_id,
                            );
                            lemma_peer_grants_after_lease_remove(
                                self.active_grants@,
                                pre_fin_leases,
                                self.rdma.active_leases@,
                                grant.handle,
                                self.channel.sent(),
                                self.next_grant_tag,
                                &pre_fin_core,
                                core,
                            );
                        }
                        let _ = self.send_control(
                            core,
                            ControlPayload::RevokeNotice { grant_tag },
                            Tracked(instance),
                        );
                        return Ok(mw_slot);
                    }
                },
                Ok(Some(PeerCompletion::Control(rdma::Completed::ControlFailed { wr_id, .. }))) => {
                    if wr_id == inv_wr_id {
                        return Err(rdma_error(rdma::RdmaError(-1)));
                    }
                },
                Ok(_) => {},
                Err(e) => return Err(e),
            }
            spins += 1;
        }
        Err(rdma_error(rdma::RdmaError(11)))
    }

    /// Initiator helper: poll the CQ until `pull.wr_id` completes.
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
            old(self).rdma.cq.in_flight().contains_key(pull.wr_id),
            old(self).rdma.cq.in_flight()[pull.wr_id] == (rdma::PendingWorkRequest {
                kind: rdma::PendingKind::DataTransfer { event: pull.writer@ },
            }),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.qp == old(self).rdma.qp,
            final(self).channel == old(self).channel,
            final(self).rdma.active_leases == old(self).rdma.active_leases,
            pull.valid(&final(self).rdma, final(core)),
            old(core).ledger@.submitted == final(core).ledger@.submitted,
            old(core).citations().subset_of(final(core).citations()),
            r matches Ok(st) ==> {
                &&& final(core).ledger@.knows.set().contains((final(core).node(), pull.writer@))
                &&& final(core).citations().contains(pull.writer@)
                &&& (st.is_ok() ==> event_succeeded(
                    final(core).ledger@,
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
        }
        while spins < 1_000_000
            invariant
                spins <= 1_000_000,
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.frames == old(core).frames,
                core.version_registry == old(core).version_registry,
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
                self.rdma.qp == old(self).rdma.qp,
                self.channel == old(self).channel,
                self.rdma.active_leases == old(self).rdma.active_leases,
                pull.valid(&self.rdma, core),
                old(core).ledger@.submitted == core.ledger@.submitted,
                old(core).citations().subset_of(core.citations()),
                self.rdma.cq.in_flight().contains_key(pull.wr_id),
                self.rdma.cq.in_flight()[pull.wr_id] == (rdma::PendingWorkRequest {
                    kind: rdma::PendingKind::DataTransfer { event: pull.writer@ },
                }),
            decreases 1_000_000 - spins,
        {
            let ghost before_poll_core = *core;
            let ghost before_poll_cq = self.rdma.cq;
            let polled = self.rdma.poll_cq(core, Tracked(instance));
            proof {
                lemma_peer_core_step_trans(
                    old(core),
                    &before_poll_core,
                    core,
                    self.rdma.timeline_id,
                );
            }
            match polled {
                Ok(Some(pc)) => {
                    let ghost c = choose|c: rdma::Completed|
                        c.describes(before_poll_cq, self.rdma.cq, instance.id())
                            && #[trigger] RdmaBackend::delivers(c, pc);
                    proof {
                        assert(before_poll_core.citations().subset_of(core.citations()));
                        assert(pull.copy@.before.citations(core.node()).subset_of(
                            core.citations(),
                        ));
                        assert(pull.copy@.retained(core.ledger@, core.node()));
                    }
                    match pc {
                        PeerCompletion::Data(op, st) => {
                            if op.sequence == pull.wr_id {
                                proof {
                                    assert(before_poll_cq.pending().contains_key(pull.wr_id));
                                    assert(core.ledger@.knows.set().contains(
                                        (core.node(), pull.writer@),
                                    ));
                                    assert(core.citations().contains(pull.writer@));
                                }
                                return Ok(st);
                            } else {
                                proof {
                                    lemma_poll_keeps_data(
                                        before_poll_cq,
                                        self.rdma.cq,
                                        c,
                                        instance.id(),
                                        pull.wr_id,
                                    );
                                }
                            }
                        },
                        _ => {
                            proof {
                                lemma_poll_keeps_data(
                                    before_poll_cq,
                                    self.rdma.cq,
                                    c,
                                    instance.id(),
                                    pull.wr_id,
                                );
                            }
                        },
                    }
                },
                Ok(None) => {},
                Err(e) => return Err(e),
            }
            spins += 1;
        }
        Err(rdma_error(rdma::RdmaError(11)))
    }

    /// Initiator helper: send a `READ_RELEASE` returning `share_cap` for `tag`.
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
            share_cap.is_rdma_bound(),
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
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
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
                closure_from(core.ledger@.submitted.map(), cited),
            ));
        }
        let _ = self.channel.send_remote_read_release_with(
            tag,
            &rel_bytes,
            Tracked(instance),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Tracked(&core.ledger.borrow().succeeded),
            Ghost(cited),
            Tracked(share_cap),
            Tracked(att),
        );
    }

    /// Initiator helper: merge peer statement + revocations, reserve a destination
    /// frame in `dst_tier`, and post the RDMA READ pull.
    pub fn fetch_post_pull(
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
            old(core).frames.frame_size as u64 <= rdma::MAX_RDMA_MSG_LEN,
            meta.frame_size == old(core).frames.frame_size as u64,
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
            share.is_rdma_bound(),
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
                PullTarget::Host => old(self).rdma.host_mr matches Some(ref s) && s.tier == dst_tier
                    && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma.gpu_mr matches Some(ref s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
            final(self).channel == old(self).channel,
            r matches Ok(pull) ==> {
                &&& pull.valid(&final(self).rdma, final(core))
                &&& pull.handle.frame.0 < old(core).frames.metas.len()
                &&& old(core).frames.meta(pull.handle.frame).state != STATE_INITIALIZING
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
                &core.timelines,
                self.rdma.timeline_id,
                &before_merge.ledger@,
                &core.ledger@,
                rdma::CqSnapshot::of(self.rdma.cq),
                core.instance_id(),
                self.rdma.next_control_wr_id,
            );
            assert(core.frames.same_directory(&before_merge.frames)) by {
                reveal(FrameTable::same_directory);
            }
            lemma_rdma_slabs_same_directory(
                &self.rdma.host_mr,
                &self.rdma.gpu_mr,
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
            lemma_statement_valid_in_receiver_ledger(
                core.ledger@,
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
                    assert(core.ledger@.submitted.dom().contains(e));
                    assert(core.ledger@.knows.set().contains((core.node(), e)));
                }
            }
            assert(transfer::transport::covers(orig_share_frontier, core.citations()));
        }
        let ghost before_bw = *core;
        let (handle, dst_tok) = match core.begin_write_version(dst_tier, key, meta.version) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                return Err(e);
            },
        };
        proof {
            reveal(FrameTable::same_directory);
            lemma_rdma_slabs_same_directory(
                &self.rdma.host_mr,
                &self.rdma.gpu_mr,
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
            assert(peer_core_step(&before_merge, core, self.rdma.timeline_id)) by {
                reveal(peer_core_step_except);
                reveal(FrameTable::same_layout);
                reveal(FrameTable::preserves_initializing);
            }
        }
        let owns = match target {
            PullTarget::Host => self.rdma.host_mr.as_ref().unwrap().owns_frame(handle.frame),
            PullTarget::GpuGdr => self.rdma.gpu_mr.as_ref().unwrap().owns_frame(handle.frame),
        };
        if !owns {
            let ghost before_ab = *core;
            let ghost ab_frame = handle.frame;
            core.abort_write(handle, dst_tok);
            proof {
                reveal(FrameTable::same_directory);
                lemma_rdma_slabs_same_directory(
                    &self.rdma.host_mr,
                    &self.rdma.gpu_mr,
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
                assert(peer_core_step_except(
                    &before_ab,
                    core,
                    self.rdma.timeline_id,
                    Some(ab_frame),
                )) by {
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                }
                lemma_peer_core_step_trans_retire_new_init(
                    &before_merge,
                    &before_ab,
                    core,
                    self.rdma.timeline_id,
                    ab_frame,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        let ghost before_post = *core;
        match post_pull_read(
            &mut self.rdma,
            core,
            Tracked(instance),
            target,
            handle,
            dst_tok,
            cap,
            Tracked(share),
            Ghost(cert),
        ) {
            Ok(p) => {
                proof {
                    lemma_peer_core_step_trans(
                        &before_merge,
                        &before_post,
                        core,
                        self.rdma.timeline_id,
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
                    lemma_rdma_slabs_same_directory(
                        &self.rdma.host_mr,
                        &self.rdma.gpu_mr,
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
                    assert(peer_core_step_except(
                        &before_ab,
                        core,
                        self.rdma.timeline_id,
                        Some(ab_frame),
                    )) by {
                        reveal(peer_core_step_except);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                    }
                    lemma_peer_core_step_trans_retire_new_init(
                        &before_merge,
                        &before_ab,
                        core,
                        self.rdma.timeline_id,
                        ab_frame,
                    );
                }
                Err(e)
            },
        }
    }

    /// Initiator helper: await CQ completion, verify version binding, finish/abort
    /// the pull, register the version binding on hit, and send `READ_RELEASE`.
    pub fn fetch_complete_pull(
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
            old(self).rdma.cq.in_flight().contains_key(pull.wr_id),
            old(self).rdma.cq.in_flight()[pull.wr_id] == (rdma::PendingWorkRequest {
                kind: rdma::PendingKind::DataTransfer { event: pull.writer@ },
            }),
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            peer_core_step_except(
                old(core),
                final(core),
                old(self).rdma.timeline_id,
                Some(pull.handle.frame),
            ),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp == old(self).rdma.qp,
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
                    reveal(FrameTable::preserves_initializing);
                }
                return Err(e);
            },
        };
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
        }
        match outcome {
            PullOutcome::Published { frame, version, share_cap } => {
                let ghost before_reg = *core;
                core.try_register_version_binding(version, Tracked(Some(vb)));
                proof {
                    assert(core.frames.same_directory(&before_reg.frames)) by {
                        reveal(FrameTable::same_directory);
                    }
                    lemma_rdma_slabs_same_directory(
                        &self.rdma.host_mr,
                        &self.rdma.gpu_mr,
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
                    assert(peer_core_step(&before_reg, core, self.rdma.timeline_id)) by {
                        reveal(peer_core_step_except);
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
                            core.frames.versions@.contains_key(v) implies core.has_version_binding(
                            v,
                        ) by {
                            if v != version {
                                assert(before_fin_core.frames.versions@.contains_key(v));
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

    /// Initiator: end-to-end remote frame fetch (`FetchRequest` -> `recv_any` ->
    /// `merge_peer_statement` + `record_revoked_map` -> `begin_write_version` ->
    /// `post_pull_read` -> `poll_cq` -> `finish_pull_read` -> `ReadRelease`).
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
            old(self).rdma.qp.ready(),
            old(self).rdma.qp.state() is ReadyToSend,
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.effective_max_rd_atomic() > 0,
            old(self).rdma.peer_node != old(core).node(),
            old(self).next_request_id < u64::MAX,
            old(core).frames.frame_size as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(core).instance_id(),
            match target {
                PullTarget::Host => old(self).rdma.host_mr matches Some(ref s) && s.tier == dst_tier
                    && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).rdma.gpu_mr matches Some(ref s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.qp.ready(),
            final(self).rdma.qp.state() is ReadyToSend,
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() == old(self).rdma.qp.source().source(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let req_id = match self.send_fetch_request(core, key, Tracked(instance)) {
            Ok(id) => id,
            Err(_) => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                }
                return Err(rdma_error(rdma::RdmaError(-1)));
            },
        };
        let msg = match self.recv_message(core, Tracked(instance)) {
            Ok(m) => m,
            Err(_) => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                }
                return Err(rdma_error(rdma::RdmaError(-1)));
            },
        };
        match msg {
            PeerMessage::Control { payload: ControlPayload::FetchMiss { .. }, .. } => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                }
                Ok(FetchOutcome::Miss)
            },
            PeerMessage::ReadGrant {
                tag,
                meta,
                cap,
                share,
                statement,
                cert,
                revoked,
                version_binding,
            } => {
                if meta.request_id != req_id || meta.frame_size != core.frames.frame_size as u64 {
                    proof {
                        lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                    }
                    return Err(CoreError::InvalidArgument);
                }
                let ghost orig_share_frontier = share@.frontier();
                let pull = self.fetch_post_pull(
                    core,
                    key,
                    dst_tier,
                    target,
                    meta,
                    &cap,
                    share,
                    statement,
                    cert,
                    revoked,
                    Tracked(instance),
                )?;
                let ghost after_post = *core;
                let ghost pull_frame = pull.handle.frame;
                let tracked vb = version_binding.get().tracked_unwrap();
                let res = self.fetch_complete_pull(
                    core,
                    tag,
                    req_id,
                    pull,
                    Tracked(vb),
                    Ghost(orig_share_frontier),
                    Tracked(instance),
                );
                proof {
                    lemma_peer_core_step_trans_retire_new_init(
                        old(core),
                        &after_post,
                        core,
                        self.rdma.timeline_id,
                        pull_frame,
                    );
                }
                res
            },
            PeerMessage::Control { .. } => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                }
                Err(CoreError::InvalidArgument)
            },
            PeerMessage::ReadRelease { tag, .. } => {
                if let Some(pos) = self.find_grant_index(tag) {
                    let ghost before_grants = self.active_grants@;
                    let _ = self.active_grants.swap_remove(pos);
                    proof {
                        lemma_swap_removed_grant(before_grants, self.active_grants@, pos as int);
                    }
                }
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                }
                Err(CoreError::InvalidArgument)
            },
            PeerMessage::Unexpected => {
                proof {
                    lemma_peer_core_step_refl(core, self.rdma.timeline_id);
                }
                Err(CoreError::InvalidArgument)
            },
        }
    }

    /// Reset the QP (`IBV_QPS_RESET`), unquarantine windows, revoke/recover any
    /// active remote leases on the reset QP, exchange fresh QP endpoints over
    /// `self.channel`, and reconnect the QP to `ReadyToSend`.
    pub fn reset_and_reconnect(
        &mut self,
        core: &mut CacheCore,
        host: &mut HostContext,
        local_endpoint: &rdma::QueuePairEndpoint,
        psn: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            old(host).ready(old(core).instance_id(), old(core).node()),
            old(self).rdma.cq.in_flight().is_empty(),
            old(self).rdma.qp.state() is ReadyToSend || old(self).rdma.qp.state() is Error,
            old(self).rdma.qp.peer() == old(self).rdma.peer_node,
            old(self).rdma.qp.pd_id() == old(self).rdma.window_pool.pd(),
            old(self).rdma.qp.source().well_formed(),
            old(self).rdma.qp.source().source() is RdmaQueuePair,
            old(self).rdma.qp.source().node() != old(self).rdma.peer_node,
            local_endpoint.node() == old(core).node(),
            psn <= 0xffffff,
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(host).ready(final(core).instance_id(), final(core).node()),
            final(host).source@.source() == old(host).source@.source(),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames.versions@ == old(core).frames.versions@,
            final(core).version_registry == old(core).version_registry,
            peer_core_step(old(core), final(core), old(self).rdma.timeline_id),
            final(self).rdma.timeline_id == old(self).rdma.timeline_id,
            final(self).rdma.peer_node == old(self).rdma.peer_node,
            final(self).rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
            final(self).rdma.qp.peer() == old(self).rdma.peer_node,
            final(self).rdma.qp.pd_id() == old(self).rdma.window_pool.pd(),
            final(self).rdma.qp.source().well_formed(),
            final(self).rdma.qp.source().source() is RdmaQueuePair,
            final(self).rdma.qp.source().node() != old(self).rdma.peer_node,
            r.is_ok() ==> final(self).rdma.qp.ready() && final(self).rdma.qp.state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_peer_core_step_refl(core, self.rdma.timeline_id);
        }
        let Tracked(qp_reset) = self.rdma.reset_qp(core)?;
        self.active_grants.clear();
        // Revoke/recover any remaining active leases on the reset QP.
        let mut guard: usize = self.rdma.active_leases.len();
        while self.rdma.active_leases.len() > 0 && guard > 0
            invariant
                core.well_formed(),
                self.well_formed(core),
                self.active_grants@.len() == 0,
                host.ready(core.instance_id(), core.node()),
                host.source@.source() == old(host).source@.source(),
                qp_reset.matches_qp(self.rdma.qp.id(), core.node()),
                self.rdma.qp.state() == rdma::QueuePairState::Reset,
                self.rdma.qp.peer() == self.rdma.peer_node,
                self.rdma.qp.pd_id() == self.rdma.window_pool.pd(),
                self.rdma.window_pool.pd() == old(self).rdma.window_pool.pd(),
                self.rdma.peer_node == old(self).rdma.peer_node,
                self.rdma.qp.source().well_formed(),
                self.rdma.qp.source().source() is RdmaQueuePair,
                self.rdma.qp.source().node() != self.rdma.peer_node,
                instance.id() == core.instance_id(),
                core.node_id == old(core).node_id,
                core.instance_id() == old(core).instance_id(),
                core.tiers == old(core).tiers,
                core.frames.versions@ == old(core).frames.versions@,
                core.version_registry == old(core).version_registry,
                peer_core_step(old(core), core, old(self).rdma.timeline_id),
                self.rdma.timeline_id == old(self).rdma.timeline_id,
            decreases guard,
        {
            guard -= 1;
            let lease_h = RemoteLeaseHandle {
                mw_slot: self.rdma.active_leases[0].mw_slot as usize,
                generation: self.rdma.active_leases[0].generation,
            };
            match self.rdma.active_leases[0].state {
                RemoteLeaseState::Bound | RemoteLeaseState::Revoking => {
                    let ghost before_step = *core;
                    let _ = self.rdma.fenced_revoke_on_qp_reset(
                        core,
                        host,
                        lease_h,
                        Tracked(&qp_reset),
                        Tracked(instance),
                    );
                    proof {
                        reveal(FrameTable::same_directory);
                        lemma_peer_core_step_trans(
                            old(core),
                            &before_step,
                            core,
                            self.rdma.timeline_id,
                        );
                    }
                },
                RemoteLeaseState::Unbinding => {
                    let _ = self.rdma.recover_unbinding_on_qp_reset(
                        core,
                        lease_h,
                        Tracked(&qp_reset),
                    );
                },
                RemoteLeaseState::Binding => {
                    break;
                },
            }
        }
        let peer_ep = match self.channel.exchange_queue_pair_endpoint(
            local_endpoint,
            Tracked(instance),
        ) {
            Ok(ep) => ep,
            Err(_) => return Err(rdma_error(rdma::RdmaError(-1))),
        };
        self.rdma.reconnect_qp(core, psn, peer_ep)
    }
}

} // verus!
