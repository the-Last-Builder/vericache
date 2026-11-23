//! Standalone Milestone 3 RDMA READ pull state machine supporting both
//! host-DRAM (`PullTarget::Host`) and GPUDirect RDMA (`PullTarget::GpuGdr`,
//! enforcing mandatory post-CQE `MemoryRegion::complete_gdr_slot` BAR1 flush).
use super::*;
#[cfg(verus_only)]
use crate::frame::{STATE_INITIALIZING, STATE_PUBLISHED};
use crate::kernel::WriteHandle;
#[cfg(verus_only)]
use crate::provenance::{
    certificate_valid_in, complete_receipts, copied_slice_certificate, event_succeeded,
    ledger_consistent_with_history, replica_valid, revoked_submap, source_slice_valid,
    GuardedReplicaCopy, ValueCertificate,
};
use crate::timeline::DataOperation;
#[cfg(verus_only)]
use foundation::history::reachable;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
#[cfg(verus_only)]
use foundation::value::valid_run;
use transfer::context::{ReadPerm, ShareCap, SlotToken, TransferInstance};
#[cfg(verus_only)]
use transfer::domain::TransferEvent;
use transfer::rdma;
use vstd::prelude::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// Destination slab class for an RDMA READ pull.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PullTarget {
    /// Pull directly into `RdmaBackend::host_mr` via `rdma::post_rdma_read`.
    Host,
    /// Pull directly into `RdmaBackend::gpu_mr` via `rdma::post_rdma_read_gdr`
    /// and seal the destination `SlotToken` until `mr.complete_gdr_slot` flushes BAR1 writes.
    GpuGdr,
}

/// Destination slot token held while an RDMA READ pull is in flight.
pub tracked enum PullSlotToken {
    Host(SlotToken),
    GpuGdr(rdma::PendingGdrSlot),
}

/// Standalone in-flight RDMA READ pull awaiting CQ completion.
pub struct InFlightRdmaPull {
    pub target: PullTarget,
    pub handle: WriteHandle,
    pub ticket: Ticket,
    pub wr_id: u64,
    pub writer: Ghost<TransferEvent>,
    pub copy: Ghost<GuardedReplicaCopy>,
    pub share_cap: Tracked<ShareCap>,
    pub dst_token: Tracked<PullSlotToken>,
}

/// Outcome of `finish_pull_read`.
pub enum PullOutcome {
    Published { frame: FrameId, version: VersionId, share_cap: Tracked<ShareCap> },
    Aborted { status: i32, share_cap: Tracked<ShareCap> },
}

impl InFlightRdmaPull {
    pub open spec fn valid(&self, rdma: &RdmaBackend, core: &CacheCore) -> bool {
        &&& self.ticket != Ticket::NONE
        &&& self.ticket.timeline_id() == rdma.timeline_id
        &&& self.ticket.sequence() == self.wr_id
        &&& core.timelines.issued(rdma.timeline_id, self.wr_id)
        &&& core.timelines.events@[(rdma.timeline_id, self.wr_id)] == self.writer@
        &&& self.handle.frame.0 < core.frames.metas.len()
        &&& core.frames.meta(self.handle.frame).state == STATE_INITIALIZING
        &&& !core.frames.meta(self.handle.frame).queued()
        &&& core.frames.frame(self.handle.frame).token is None
        &&& core.frames.frame(self.handle.frame).tier == self.handle.tier
        &&& core.frames.frame(self.handle.frame).slot == self.handle.slot@
        &&& core.frames.frame(self.handle.frame).bound_key == Some(self.handle.key)
        &&& core.frames.frame(self.handle.frame).bound_version == Some(self.handle.version)
        &&& self.handle.version.well_formed() && self.handle.version != VersionId::NONE
        &&& (self.handle.version.origin_node != core.node_id || self.handle.version.seq
            < core.next_version_seq || self.handle.version.seq >= core.version_seq_limit)
        &&& self.handle.slot@.is_full()
        &&& self.handle.slot@.loc.node == core.node()
        &&& self.handle.slot@.slot_len == core.frames.tier_frame_size(self.handle.tier)
        &&& self.copy@.writer == self.writer@
        &&& self.copy@.destination == self.handle.slot@
        &&& self.copy@.source_slot == self.copy@.source.slot
        &&& self.copy@.source.bytes.len() == self.handle.slot@.slot_len
        &&& self.copy@.retained(core.ledger@, core.node())
        &&& self.share_cap@.is_rdma_bound()
        &&& self.share_cap@.instance_id() == core.instance_id()
        &&& self.share_cap@.loc() == self.copy@.source_slot.loc
        &&& self.share_cap@.share() == self.copy@.share
        &&& self.share_cap@.frontier().last_writer == Some(self.copy@.source.writer)
        &&& self.share_cap@.frontier().readers.contains(self.writer@)
        &&& match (self.target, self.dst_token@) {
            (PullTarget::Host, PullSlotToken::Host(tok)) => slot_owned(
                tok,
                core.instance_id(),
                self.handle.slot@.loc,
                Frontier { last_writer: Some(self.writer@), readers: ISet::empty() },
            ),
            (PullTarget::GpuGdr, PullSlotToken::GpuGdr(pending)) => {
                &&& pending.well_formed()
                &&& pending.writer() == self.writer@
                &&& pending.instance_id() == core.instance_id()
                &&& pending.loc() == self.handle.slot@.loc
                &&& pending.frontier() == (Frontier {
                    last_writer: Some(self.writer@),
                    readers: ISet::empty(),
                })
                &&& rdma.effective_gpu_mr_spec() matches Some(s) && s.owns(self.handle.frame)
                    && s.mr.gpu_device_spec() is Some && s.mr.space() == self.handle.slot@.loc.space
            },
            _ => false,
        }
    }
}

proof fn lemma_post_pull_cq_wf<'a>(rdma: &RdmaBackend<'a>, core: &CacheCore)
    requires
        rdma.well_formed(core),
    ensures
        rdma.cq.well_formed(),
        rdma.cq.observer() == core.node(),
        rdma.cq.infos().instance_id() == core.instance_id(),
{
    hide(CacheCore::well_formed);
    reveal(RdmaBackend::well_formed);
}

proof fn lemma_post_pull_entry_facts<'a>(
    rdma: &RdmaBackend<'a>,
    core: &CacheCore,
    wr_id: u64,
    handle: WriteHandle,
    dst_tok: SlotToken,
    share_cap: ShareCap,
    source_cert: ValueCertificate,
)
    requires
        core.well_formed(),
        rdma.well_formed(core),
        rdma.qp.state() is ReadyToSend,
        wr_id <= MAX_TICKET_SEQ,
        !core.timelines.issued(rdma.timeline_id, wr_id),
        handle.frame.0 < core.frames.metas.len(),
        core.frames.meta(handle.frame).state == STATE_INITIALIZING,
        core.frames.frame(handle.frame).tier == handle.tier,
        core.frames.frame(handle.frame).slot == handle.slot@,
        slot_is_owned(dst_tok, core.instance_id(), handle.slot@.loc),
        write_admitted(slot_frontier(dst_tok), core.citations()),
        read_admitted(share_cap.frontier(), core.citations()),
        core.citations().contains(source_cert.writer),
    ensures
        rdma.qp.cq_id() == rdma.cq.id(),
        rdma.qp.peer() == rdma.peer_node,
        rdma.qp.pd_id() == rdma.window_pool.pd(),
        rdma.qp.source().id() == core.instance_id(),
        rdma.qp.source().node() == core.node(),
        rdma.cq.well_formed(),
        rdma.cq.observer() == core.node(),
        rdma.cq.infos().instance_id() == core.instance_id(),
        !rdma.cq.in_flight().contains_key(wr_id),
        handle.slot@.well_formed(),
        handle.slot@.is_full(),
        handle.slot@.loc.node == core.node(),
        handle.slot@.slot_len == core.frames.tier_frame_size(handle.tier),
        core.disjoint_from_published(ISet::empty().insert(handle.slot@.loc)),
        read_admitted(
            share_cap.frontier(),
            closure_from(core.ledger@.submitted.map(), core.citations()),
        ),
        write_admitted(
            slot_frontier(dst_tok),
            closure_from(core.ledger@.submitted.map(), core.citations()),
        ),
        core.citations().subset_of(core.ledger@.submitted.dom()),
        core.ledger@.submitted.instance_id() == core.instance_id(),
        core.ledger@.knows.instance_id() == core.instance_id(),
        foundation::tracked::knowledge_of(core.node(), core.citations()).subset_of(
            core.ledger@.knows.set(),
        ),
        closure_from(core.ledger@.submitted.map(), core.citations()).contains(source_cert.writer),
        rdma.effective_host_mr_spec() matches Some(s) ==> s.well_formed(
            core,
            rdma.window_pool.pd(),
        ),
        rdma.effective_gpu_mr_spec() matches Some(s) ==> s.well_formed(core, rdma.window_pool.pd()),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    reveal(RdmaBackend::well_formed);
    reveal(rdma_window_wf);
    lemma_core_wf_facts(core);
    lemma_core_timelines_wf(core);
    lemma_core_frames_separated(core);
    lemma_initializing_frame_slot_valid(core, handle.frame);
    core.lemma_non_published_write_disjoint(handle.frame, ISet::empty().insert(handle.slot@.loc));
    rdma.lemma_effective_slabs_wf(core);
    let before_cq = rdma::CqSnapshot::of(rdma.cq);
    if rdma.cq.in_flight().contains_key(wr_id) {
        let req = rdma.cq.in_flight()[wr_id];
        if req.kind is DataTransfer {
            assert(rdma_data_events(before_cq).contains_key(wr_id));
        } else {
            assert(CONTROL_WR_ID_BASE <= wr_id);
        }
    }
    core.lemma_citations_admit_closure(share_cap.frontier(), ISet::empty());
    core.lemma_citations_admit_closure(slot_frontier(dst_tok), ISet::empty());
    lemma_citations_closure(core);
    assert(ISet::empty().union(core.citations()) =~= core.citations());
}

#[allow(clippy::too_many_arguments)]
proof fn lemma_post_pull_prepare_batch<'a>(
    before_core: &CacheCore,
    core: &CacheCore,
    old_rdma: &RdmaBackend<'a>,
    rdma: &RdmaBackend<'a>,
    before_cq: rdma::CqSnapshot,
    wr_id: u64,
    ev: TransferEvent,
    receipt: Submitted,
    cited: ISet<TransferEvent>,
    remote_slot: SlotRef,
    handle: WriteHandle,
    guards: ReadGuards,
)
    requires
        before_core.well_formed(),
        old_rdma.well_formed(before_core),
        old_rdma.qp.ready(),
        old_rdma.qp.state() is ReadyToSend,
        before_cq == rdma::CqSnapshot::of(old_rdma.cq),
        cited == before_core.citations(),
        wr_id == before_core.timelines.next_seq[old_rdma.timeline_id as int],
        wr_id <= MAX_TICKET_SEQ,
        wr_id <= before_core.timelines.confirmed_seq[old_rdma.timeline_id as int] + BITMAP_CAPACITY,
        !before_cq.pending.contains_key(wr_id),
        before_core.disjoint_from_published(ISet::empty().insert(handle.slot@.loc)),
        core.node_id == before_core.node_id,
        core.tiers == before_core.tiers,
        core.frames == before_core.frames,
        core.catalog == before_core.catalog,
        core.timelines == before_core.timelines,
        core.active_leases == before_core.active_leases,
        core.free_lease_slots == before_core.free_lease_slots,
        core.lease_generation == before_core.lease_generation,
        core.version_seq_start == before_core.version_seq_start,
        core.next_version_seq == before_core.next_version_seq,
        core.version_seq_limit == before_core.version_seq_limit,
        core.version_registry == before_core.version_registry,
        core.spec == before_core.spec,
        core.ledger@.submitted.map() == before_core.ledger@.submitted.map(),
        core.ledger@.submitted.instance_id() == before_core.ledger@.submitted.instance_id(),
        core.ledger@.knows == before_core.ledger@.knows,
        core.ledger@.succeeded == before_core.ledger@.succeeded,
        core.ledger@.revoked == before_core.ledger@.revoked,
        rdma.timeline_id == old_rdma.timeline_id,
        rdma.peer_node == old_rdma.peer_node,
        rdma.window_pool == old_rdma.window_pool,
        rdma.host_mr == old_rdma.host_mr,
        rdma.gpu_mr == old_rdma.gpu_mr,
        rdma.shared_host_mr == old_rdma.shared_host_mr,
        rdma.shared_gpu_mr == old_rdma.shared_gpu_mr,
        rdma.in_flight_ops == old_rdma.in_flight_ops,
        rdma.bitmap == old_rdma.bitmap,
        rdma.completed_pulls == old_rdma.completed_pulls,
        rdma.active_leases == old_rdma.active_leases,
        rdma.next_control_wr_id == old_rdma.next_control_wr_id,
        rdma.qp.state() is ReadyToSend,
        rdma.qp.qp_cq_stepped(
            old_rdma.qp,
            rdma.cq,
            old_rdma.cq,
            copy_access(remote_slot, handle.slot@),
            copy_effect(remote_slot, handle.slot@),
            guards,
            wr_id,
            false,
            true,
            before_core.ledger@.submitted,
            cited,
            ev,
            receipt,
        ),
        old_rdma.qp.transfer_ordered(false, before_core.ledger@.submitted, cited) == closure_from(
            before_core.ledger@.submitted.map(),
            cited,
        ),
    ensures
        core.well_formed(),
        peer_core_step(before_core, core, rdma.timeline_id),
        rdma.timeline_id < core.timelines.count,
        core.timelines.kinds[rdma.timeline_id as int] == (TimelineKind::RdmaQueuePair {
            peer_node: rdma.peer_node,
        }),
        rdma.qp.cq_id() == rdma.cq.id(),
        rdma.qp.state() is ReadyToSend ==> rdma.qp.peer() == rdma.peer_node && rdma.qp.pd_id()
            == rdma.window_pool.pd(),
        rdma.qp.source().id() == core.instance_id(),
        rdma.qp.source().node() == core.node(),
        rdma.cq.well_formed(),
        rdma.cq.observer() == core.node(),
        rdma.cq.infos().instance_id() == core.instance_id(),
        rdma.window_pool.well_formed(),
        rdma.window_pool.node() == core.node(),
        !(rdma.host_mr is Some && rdma.shared_host_mr is Some),
        !(rdma.gpu_mr is Some && rdma.shared_gpu_mr is Some),
        rdma.host_mr matches Some(ref s) ==> s.well_formed(core, rdma.window_pool.pd()),
        rdma.gpu_mr matches Some(ref s) ==> s.well_formed(core, rdma.window_pool.pd()),
        rdma.shared_host_mr matches Some(s) ==> s.well_formed_shared(core, rdma.window_pool.pd()),
        rdma.shared_gpu_mr matches Some(s) ==> s.well_formed_shared(core, rdma.window_pool.pd()),
        rdma_leases_valid(rdma.active_leases@, &rdma.window_pool, &rdma.qp, core),
        rdma_control_leases_match_cq(
            rdma.active_leases@,
            &rdma.window_pool,
            rdma::CqSnapshot::of(rdma.cq),
            rdma.next_control_wr_id,
        ),
        rdma_pool_slots_owned_by_leases(rdma.active_leases@, &rdma.window_pool),
        completed_pulls_wf(
            rdma.completed_pulls,
            &core.timelines,
            rdma.timeline_id,
            rdma.cq.observer(),
            &core.ledger@,
        ),
        rdma.bitmap.well_formed(),
        rdma.bitmap.observations(&core.timelines, rdma.timeline_id, core.node(), &core.ledger@),
        CONTROL_WR_ID_BASE <= rdma.next_control_wr_id <= CONTROL_WR_ID_LIMIT,
        forall|s: u64|
            rdma.cq.in_flight().contains_key(s) && !((
            #[trigger] rdma.cq.in_flight()[s]).kind is DataTransfer) ==> CONTROL_WR_ID_BASE <= s
                < rdma.next_control_wr_id,
        receipt.instance_id() == core.instance_id(),
        receipt.key() == ev,
        receipt.key().observer == core.node(),
        receipt.value().preds.subset_of(core.ledger@.submitted.dom()),
        core.disjoint_from_published(receipt.value().access.writes),
        rdma.pending().contains_key(wr_id),
        rdma.pending()[wr_id] == ev,
        window_matches(
            rdma.in_flight_ops,
            rdma.bitmap,
            &core.timelines,
            rdma.timeline_id,
            rdma_remove_range(rdma.pending(), wr_id, 1),
        ),
{
    hide(replica_valid);
    reveal(RdmaBackend::well_formed);
    reveal(rdma_window_wf);
    lemma_core_wf_facts(before_core);
    lemma_citations_closure(before_core);
    assert(core.well_formed()) by {
        reveal(CacheCore::well_formed);
        reveal(complete_receipts);
        assert(before_core.ledger@.submitted.map().submap_of(core.ledger@.submitted.map()));
        assert(revoked_submap(before_core.ledger@.revoked, core.ledger@.revoked));
        assert(before_core.ledger@.knows.set().subset_of(core.ledger@.knows.set()));
        assert(before_core.ledger@.succeeded.dom().subset_of(core.ledger@.succeeded.dom()));
        assert(core.citations() =~= before_core.citations());
        assert(core.citations().subset_of(core.citations()));
        assert(core.kernel_view().extends(before_core.kernel_view())) by {
            assert forall|u: u8|
                u
                    < before_core.timelines.count implies #[trigger] core.timelines.confirmed_seq[u as int]
                >= before_core.timelines.confirmed_seq[u as int] by {}
            assert forall|t: u8, s: u64|
                t < before_core.timelines.count && #[trigger] before_core.timelines.issued(
                    t,
                    s,
                ) implies core.timelines.issued(t, s) && core.timelines.events@[(t, s)]
                == before_core.timelines.events@[(t, s)] by {}
        };
        core.preserve_after_ledger_growth(*before_core);
    }
    lemma_rdma_leases_same_frames(
        rdma.active_leases@,
        &rdma.window_pool,
        &old_rdma.qp,
        before_core,
        core,
    );
    lemma_rdma_leases_after_qp_state_change(
        rdma.active_leases@,
        &rdma.window_pool,
        &rdma.window_pool,
        &old_rdma.qp,
        &rdma.qp,
        core,
    );
    reveal(FrameTable::same_layout);
    lemma_rdma_slabs_after_init_finish(
        &rdma.host_mr,
        &rdma.gpu_mr,
        rdma.shared_host_mr,
        rdma.shared_gpu_mr,
        before_core,
        core,
        rdma.window_pool.pd(),
    );
    assert(completed_pulls_wf(
        rdma.completed_pulls,
        &core.timelines,
        rdma.timeline_id,
        rdma.cq.observer(),
        &core.ledger@,
    ));
    assert(rdma.bitmap.observations(&core.timelines, rdma.timeline_id, core.node(), &core.ledger@));
    assert(rdma.pending().contains_key(wr_id));
    assert(rdma.pending()[wr_id] == ev);
    assert(rdma_remove_range(rdma.pending(), wr_id, 1) =~= rdma_data_events(before_cq)) by {
        assert forall|k: u64|
            (#[trigger] rdma_remove_range(rdma.pending(), wr_id, 1).contains_key(k))
                == rdma_data_events(before_cq).contains_key(k) by {
            if k != wr_id {
                assert(rdma.cq.in_flight().contains_key(k) == before_cq.pending.contains_key(k));
                if before_cq.pending.contains_key(k) {
                    assert(rdma.cq.in_flight()[k] == before_cq.pending[k]);
                }
            }
        }
        assert forall|k: u64| #[trigger]
            rdma_remove_range(rdma.pending(), wr_id, 1).contains_key(k) implies rdma_remove_range(
            rdma.pending(),
            wr_id,
            1,
        )[k] == rdma_data_events(before_cq)[k] by {
            assert(k != wr_id);
            assert(rdma.cq.in_flight()[k] == before_cq.pending[k]);
        }
    };
    assert forall|s: u64|
        rdma.cq.in_flight().contains_key(s) && !((
        #[trigger] rdma.cq.in_flight()[s]).kind is DataTransfer) implies CONTROL_WR_ID_BASE <= s
        < rdma.next_control_wr_id by {
        assert(s != wr_id);
        assert(before_cq.pending.contains_key(s));
        assert(before_cq.pending[s] == rdma.cq.in_flight()[s]);
    }
    assert(rdma_control_leases_match_cq(
        rdma.active_leases@,
        &rdma.window_pool,
        rdma::CqSnapshot::of(rdma.cq),
        rdma.next_control_wr_id,
    )) by {
        reveal(rdma_control_leases_match_cq);
        assert forall|i: int|
            0 <= i < rdma.active_leases@.len() && (
            #[trigger] rdma.active_leases@[i]).has_active_wr() implies {
            let l = rdma.active_leases@[i];
            let w = rdma.window_pool.slots[l.mw_slot as int].window;
            let cq_snap = rdma::CqSnapshot::of(rdma.cq);
            &&& CONTROL_WR_ID_BASE <= l.wr_id < rdma.next_control_wr_id
            &&& (l.state == RemoteLeaseState::Binding && cq_snap.pending.contains_key(l.wr_id))
                ==> {
                &&& cq_snap.pending[l.wr_id].kind is BindWindow
                &&& (w matches Some(win) ==> cq_snap.pending[l.wr_id].kind->BindWindow_window_id
                    == win.id())
            }
            &&& ((l.state == RemoteLeaseState::Unbinding || l.state == RemoteLeaseState::Revoking)
                && cq_snap.pending.contains_key(l.wr_id)) ==> {
                &&& cq_snap.pending[l.wr_id].kind is LocalInvalidate
                &&& (w matches Some(win)
                    ==> cq_snap.pending[l.wr_id].kind->LocalInvalidate_window_id == win.id())
            }
        } by {
            let l = rdma.active_leases@[i];
            assert(l.wr_id != wr_id);
        }
    };
    assert(receipt.value().preds == closure_from(before_core.ledger@.submitted.map(), cited));
    assert(receipt.value().preds.subset_of(core.ledger@.submitted.dom())) by {
        assert forall|e: TransferEvent| #[trigger]
            receipt.value().preds.contains(e) implies core.ledger@.submitted.dom().contains(e) by {
            assert(before_core.ledger@.submitted.dom().contains(e));
            assert(before_core.ledger@.submitted.map().contains_key(e));
        }
    };
    assert(receipt.value().access.writes =~= ISet::empty().insert(handle.slot@.loc));
    assert(core.disjoint_from_published(receipt.value().access.writes));
    assert(peer_core_step(before_core, core, rdma.timeline_id)) by {
        reveal(peer_core_step_except);
        reveal(FrameTable::preserves_initializing);
    };
}

#[allow(clippy::too_many_arguments)]
proof fn lemma_post_pull_finish_valid<'a>(
    before_core: &CacheCore,
    core: &CacheCore,
    rdma: &RdmaBackend<'a>,
    pull: &InFlightRdmaPull,
    source_cert: ValueCertificate,
    receipt: Submitted,
    cited: ISet<TransferEvent>,
)
    requires
        before_core.well_formed(),
        core.well_formed(),
        rdma.well_formed(core),
        core.instance_id() == before_core.instance_id(),
        cited == before_core.citations(),
        cited.contains(source_cert.writer),
        pull.ticket != Ticket::NONE,
        pull.ticket.timeline_id() == rdma.timeline_id,
        pull.ticket.sequence() == pull.wr_id,
        core.timelines.issued(rdma.timeline_id, pull.wr_id),
        core.timelines.events@.dom().contains((rdma.timeline_id, pull.wr_id))
            ==> core.timelines.events@[(rdma.timeline_id, pull.wr_id)] == pull.writer@,
        pull.handle.frame.0 < core.frames.metas.len(),
        core.frames.meta(pull.handle.frame).state == STATE_INITIALIZING,
        !core.frames.meta(pull.handle.frame).queued(),
        core.frames.frame(pull.handle.frame).token is None,
        core.frames.frame(pull.handle.frame).tier == pull.handle.tier,
        core.frames.frame(pull.handle.frame).slot == pull.handle.slot@,
        core.frames.frame(pull.handle.frame).bound_key == Some(pull.handle.key),
        core.frames.frame(pull.handle.frame).bound_version == Some(pull.handle.version),
        pull.handle.version.well_formed() && pull.handle.version != VersionId::NONE,
        pull.handle.version.origin_node != core.node_id || pull.handle.version.seq
            < core.next_version_seq || pull.handle.version.seq >= core.version_seq_limit,
        pull.handle.slot@.is_full(),
        pull.handle.slot@.well_formed(),
        pull.handle.slot@.loc.node == core.node(),
        pull.handle.slot@.slot_len == core.frames.tier_frame_size(pull.handle.tier),
        source_cert.slot.well_formed(),
        source_cert.slot.is_full(),
        source_cert.slot.sub_len == pull.handle.slot@.slot_len,
        source_cert.bytes.len() == pull.handle.slot@.slot_len,
        before_core.ledger@.submitted.dom().contains(source_cert.writer),
        forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                before_core.ledger@,
                before_core.node(),
                h,
                run,
            ) ==> #[trigger] certificate_valid_in(source_cert, h, run),
        pull.copy@ == (GuardedReplicaCopy {
            before: before_core.ledger@,
            source: source_cert,
            source_slot: source_cert.slot,
            share: pull.share_cap@.share(),
            writer: pull.writer@,
            destination: pull.handle.slot@,
        }),
        before_core.ledger@.submitted.map().submap_of(core.ledger@.submitted.map()),
        core.ledger@.knows == before_core.ledger@.knows,
        core.ledger@.succeeded == before_core.ledger@.succeeded,
        core.ledger@.revoked == before_core.ledger@.revoked,
        core.node() == before_core.node(),
        core.ledger@.submitted.dom().contains(pull.writer@),
        core.ledger@.submitted.map()[pull.writer@] == receipt.value(),
        receipt.value().access == copy_access(source_cert.slot, pull.handle.slot@),
        receipt.value().effect == copy_effect(source_cert.slot, pull.handle.slot@),
        receipt.value().read_guards == IMap::empty().insert(
            source_cert.slot.loc,
            pull.share_cap@.share(),
        ),
        receipt.value().preds == closure_from(before_core.ledger@.submitted.map(), cited),
        pull.share_cap@.is_rdma_bound(),
        pull.share_cap@.instance_id() == core.instance_id(),
        pull.share_cap@.loc() == source_cert.slot.loc,
        pull.share_cap@.frontier().last_writer == Some(source_cert.writer),
        pull.share_cap@.frontier().readers.contains(pull.writer@),
        match (pull.target, pull.dst_token@) {
            (PullTarget::Host, PullSlotToken::Host(tok)) => slot_owned(
                tok,
                core.instance_id(),
                pull.handle.slot@.loc,
                Frontier { last_writer: Some(pull.writer@), readers: ISet::empty() },
            ),
            (PullTarget::GpuGdr, PullSlotToken::GpuGdr(pending)) => {
                &&& pending.well_formed()
                &&& pending.writer() == pull.writer@
                &&& pending.instance_id() == core.instance_id()
                &&& pending.loc() == pull.handle.slot@.loc
                &&& pending.frontier() == (Frontier {
                    last_writer: Some(pull.writer@),
                    readers: ISet::empty(),
                })
                &&& rdma.effective_gpu_mr_spec() matches Some(s) && s.owns(pull.handle.frame)
                    && s.mr.gpu_device_spec() is Some && s.mr.space() == pull.handle.slot@.loc.space
            },
            _ => false,
        },
    ensures
        pull.valid(rdma, core),
{
    hide(CacheCore::well_formed);
    hide(RdmaBackend::well_formed);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(core);
    lemma_core_timelines_wf(core);
    lemma_citations_closure(before_core);
    reveal(GuardedReplicaCopy::retained);
    reveal(InFlightRdmaPull::valid);
    assert(before_core.ledger@.valid(core.ledger@.submitted.instance_id(), core.node()));
    assert(revoked_submap(before_core.ledger@.revoked, core.ledger@.revoked));
    assert(before_core.ledger@.citations(core.node()).subset_of(
        core.ledger@.citations(core.node()),
    )) by {
        assert forall|e: TransferEvent| #[trigger]
            before_core.ledger@.citations(core.node()).contains(e) implies core.ledger@.citations(
            core.node(),
        ).contains(e) by {
            assert(before_core.ledger@.submitted.map().contains_key(e));
            assert(core.ledger@.submitted.map().contains_key(e));
        }
    };
    assert(core.ledger@.submitted.dom().contains(source_cert.writer)) by {
        assert(before_core.ledger@.submitted.map().contains_key(source_cert.writer));
    };
    assert(receipt.value().preds.contains(source_cert.writer));
    assert(receipt.value().preds.subset_of(core.ledger@.submitted.dom())) by {
        assert forall|e: TransferEvent| #[trigger]
            receipt.value().preds.contains(e) implies core.ledger@.submitted.dom().contains(e) by {
            assert(before_core.ledger@.submitted.dom().contains(e));
            assert(before_core.ledger@.submitted.map().contains_key(e));
        }
    };
    assert(source_slice_valid(source_cert, source_cert.slot));
    assert(crate::provenance::effect_implies(
        core.ledger@.submitted.map()[pull.writer@].effect,
        copy_effect(source_cert.slot, pull.handle.slot@),
    ));
    assert(pull.copy@.retained(core.ledger@, core.node()));
}

/// Post an RDMA READ pull into a reserved initializing frame (`PullTarget::Host`
/// or `PullTarget::GpuGdr`), recording the submission on `rdma` and `core`.
///
/// On failure, returns `(err, handle, dst_tok, share_cap)` intact without leaking.
pub fn post_pull_read<'a>(
    rdma: &mut RdmaBackend<'a>,
    core: &mut CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    target: PullTarget,
    handle: WriteHandle,
    Tracked(dst_tok): Tracked<SlotToken>,
    remote_cap: &rdma::RemoteCapability,
    Tracked(share_cap): Tracked<ShareCap>,
    Ghost(source_cert): Ghost<ValueCertificate>,
) -> (r: Result<InFlightRdmaPull, (CoreError, WriteHandle, Tracked<SlotToken>, Tracked<ShareCap>)>)
    requires
        old(core).well_formed(),
        old(rdma).well_formed(old(core)),
        instance.id() == old(core).instance_id(),
        old(rdma).qp.ready(),
        old(rdma).qp.state() is ReadyToSend,
        old(rdma).qp.source().well_formed(),
        old(rdma).qp.effective_max_rd_atomic() > 0,
        old(core).frames.tier_frame_size(handle.tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
        handle.frame.0 < old(core).frames.metas.len(),
        old(core).frames.meta(handle.frame).state == STATE_INITIALIZING,
        !old(core).frames.meta(handle.frame).queued(),
        old(core).frames.frame(handle.frame).token is None,
        old(core).frames.frame(handle.frame).tier == handle.tier,
        old(core).frames.frame(handle.frame).slot == handle.slot@,
        old(core).frames.frame(handle.frame).bound_key == Some(handle.key),
        old(core).frames.frame(handle.frame).bound_version == Some(handle.version),
        handle.version.well_formed() && handle.version != VersionId::NONE,
        handle.version.origin_node != old(core).node_id || handle.version.seq < old(
            core,
        ).next_version_seq || handle.version.seq >= old(core).version_seq_limit,
        slot_is_owned(dst_tok, old(core).instance_id(), handle.slot@.loc),
        write_admitted(slot_frontier(dst_tok), old(core).citations()),
        remote_cap.well_formed(),
        remote_cap.slot_ref().loc.node == old(rdma).peer_node,
        remote_cap.slot_ref().loc.node != old(core).node(),
        remote_cap.access() == rdma::RemoteAccess::ReadOnly,
        remote_cap.share() == Some(share_cap.share()),
        remote_cap.slot_ref() == source_cert.slot,
        source_cert.slot.is_full(),
        source_cert.slot.sub_len == old(core).frames.tier_frame_size(handle.tier),
        source_cert.bytes.len() == old(core).frames.tier_frame_size(handle.tier),
        old(core).ledger@.submitted.dom().contains(source_cert.writer),
        old(core).citations().contains(source_cert.writer),
        forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                old(core).ledger@,
                old(core).node(),
                h,
                run,
            ) ==> #[trigger] certificate_valid_in(source_cert, h, run),
        share_cap.is_rdma_bound(),
        share_cap.instance_id() == old(core).instance_id(),
        share_cap.loc() == remote_cap.slot_ref().loc,
        share_cap.frontier().last_writer == Some(source_cert.writer),
        read_admitted(share_cap.frontier(), old(core).citations()),
        match target {
            PullTarget::Host => old(rdma).effective_host_mr_spec() matches Some(s) && s.owns(
                handle.frame,
            ) && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
            PullTarget::GpuGdr => old(rdma).effective_gpu_mr_spec() matches Some(s) && s.owns(
                handle.frame,
            ) && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
        },
    ensures
        final(core).well_formed(),
        final(rdma).well_formed(final(core)),
        final(core).node_id == old(core).node_id,
        final(core).instance_id() == old(core).instance_id(),
        final(core).tiers == old(core).tiers,
        final(core).frames == old(core).frames,
        final(core).catalog == old(core).catalog,
        final(core).spec == old(core).spec,
        final(core).version_registry == old(core).version_registry,
        peer_core_step(old(core), final(core), old(rdma).timeline_id),
        final(rdma).timeline_id == old(rdma).timeline_id,
        final(rdma).peer_node == old(rdma).peer_node,
        final(rdma).qp.ready(),
        final(rdma).qp.state() is ReadyToSend,
        final(rdma).qp.source().well_formed(),
        final(rdma).qp.source().source() == old(rdma).qp.source().source(),
        final(rdma).window_pool == old(rdma).window_pool,
        final(rdma).active_leases == old(rdma).active_leases,
        final(rdma).host_mr == old(rdma).host_mr,
        final(rdma).gpu_mr == old(rdma).gpu_mr,
        final(rdma).shared_host_mr == old(rdma).shared_host_mr,
        final(rdma).shared_gpu_mr == old(rdma).shared_gpu_mr,
        final(rdma).completed_pulls == old(rdma).completed_pulls,
        final(rdma).next_control_wr_id == old(rdma).next_control_wr_id,
        match r {
            Err((_, ret_h, ret_tok, ret_sh)) => {
                &&& ret_h == handle
                &&& ret_tok@ == dst_tok
                &&& ret_sh@ == share_cap
                &&& *final(core) == *old(core)
                &&& *final(rdma) == *old(rdma)
            },
            Ok(pull) => {
                &&& pull.valid(final(rdma), final(core))
                &&& pull.target == target
                &&& pull.handle == handle
                &&& pull.copy@.source == source_cert
                &&& pull.copy@.share == share_cap.share()
                &&& pull.share_cap@.frontier() == (Frontier {
                    last_writer: share_cap.frontier().last_writer,
                    readers: share_cap.frontier().readers.insert(pull.writer@),
                })
                &&& final(rdma).cq.in_flight().contains_key(pull.wr_id)
                &&& final(rdma).cq.in_flight()[pull.wr_id] == (rdma::PendingWorkRequest {
                    kind: rdma::PendingKind::DataTransfer { event: pull.writer@ },
                })
                &&& final(core).ledger@.submitted.map() == old(core).ledger@.submitted.map().insert(
                    pull.writer@,
                    final(core).ledger@.submitted.map()[pull.writer@],
                )
                &&& old(core).ledger@.submitted.map().submap_of(final(core).ledger@.submitted.map())
                &&& final(core).ledger@.knows == old(core).ledger@.knows
                &&& final(core).ledger@.succeeded == old(core).ledger@.succeeded
                &&& final(core).ledger@.revoked == old(core).ledger@.revoked
            },
        },
{
    hide(CacheCore::well_formed);
    hide(RdmaBackend::well_formed);
    hide(InFlightRdmaPull::valid);
    hide(GuardedReplicaCopy::retained);
    hide(replica_valid);
    hide(complete_receipts);
    proof {
        lemma_post_pull_cq_wf(rdma, core);
    }
    if rdma.cq.in_flight_count() >= rdma.cq.capacity_exec() {
        proof {
            lemma_peer_core_step_refl(core, rdma.timeline_id);
        }
        return Err((CoreError::CapacityExhausted, handle, Tracked(dst_tok), Tracked(share_cap)));
    }
    let wr_id = match rdma.next_data_id(core) {
        Ok(s) => s,
        Err(e) => {
            proof {
                lemma_peer_core_step_refl(core, rdma.timeline_id);
            }
            return Err((e, handle, Tracked(dst_tok), Tracked(share_cap)));
        },
    };
    let ghost before_core = *core;
    let ghost before_cq = rdma::CqSnapshot::of(rdma.cq);
    let ghost cited = core.citations();
    let ghost share_id = share_cap.share();
    proof {
        lemma_post_pull_entry_facts(rdma, core, wr_id, handle, dst_tok, share_cap, source_cert);
    }
    rdma.qp.lemma_next_event_unsubmitted(
        Tracked(instance),
        Tracked(&core.ledger.borrow().submitted),
    );
    let tracked mut remote_perm = ReadPerm::Shared(share_cap);
    let ghost expected_guards = remote_perm.read_guards();
    let (ev, receipt, dst_token): (
        Ghost<TransferEvent>,
        Tracked<Submitted>,
        Tracked<PullSlotToken>,
    ) = match target {
        PullTarget::Host => {
            let slab = match (&rdma.host_mr, rdma.shared_host_mr) {
                (Some(s), _) => s,
                (None, Some(s)) => s,
                (None, None) => {
                    let tracked sh = remote_perm.into_shared();
                    proof {
                        assert(false);
                        lemma_peer_core_step_refl(core, rdma.timeline_id);
                    }
                    return Err((CoreError::UnsupportedTier, handle, Tracked(dst_tok), Tracked(sh)));
                },
            };
            proof {
                lemma_slab_frame_slot(
                    core,
                    slab.tier,
                    slab.first_frame,
                    slab.num_frames,
                    slab.mr.space(),
                    handle.frame,
                );
            }
            let slice = slab.mr.slot_slice((handle.frame.0 - slab.first_frame.0) as usize);
            proof {
                assert(slice.slot_ref() == handle.slot@);
                assert(ISet::empty().union(cited) =~= cited);
                assert(rdma.qp.transfer_ordered(false, core.ledger@.submitted, cited)
                    == closure_from(core.ledger@.submitted.map(), cited));
            }
            let tracked mut local_tok = dst_tok;
            let post_res = {
                let tracked ledger_ref = core.ledger.borrow_mut();
                rdma::post_rdma_read(
                    &mut rdma.qp,
                    &mut rdma.cq,
                    Tracked(instance),
                    &slice,
                    remote_cap,
                    wr_id,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(&mut local_tok),
                    Tracked(&mut ledger_ref.submitted),
                    Tracked(&ledger_ref.knows),
                    Ghost(cited),
                )
            };
            match post_res {
                Ok((ev, sub)) => (ev, sub, Tracked(PullSlotToken::Host(local_tok))),
                Err(e) => {
                    let tracked sh = remote_perm.into_shared();
                    proof {
                        assert(*core == before_core);
                        lemma_peer_core_step_refl(core, rdma.timeline_id);
                    }
                    return Err((rdma_error(e), handle, Tracked(local_tok), Tracked(sh)));
                },
            }
        },
        PullTarget::GpuGdr => {
            let slab = match (&rdma.gpu_mr, rdma.shared_gpu_mr) {
                (Some(s), _) => s,
                (None, Some(s)) => s,
                (None, None) => {
                    let tracked sh = remote_perm.into_shared();
                    proof {
                        assert(false);
                        lemma_peer_core_step_refl(core, rdma.timeline_id);
                    }
                    return Err((CoreError::UnsupportedTier, handle, Tracked(dst_tok), Tracked(sh)));
                },
            };
            proof {
                lemma_slab_frame_slot(
                    core,
                    slab.tier,
                    slab.first_frame,
                    slab.num_frames,
                    slab.mr.space(),
                    handle.frame,
                );
            }
            let slice = slab.mr.slot_slice((handle.frame.0 - slab.first_frame.0) as usize);
            proof {
                assert(slice.slot_ref() == handle.slot@);
                assert(ISet::empty().union(cited) =~= cited);
                assert(rdma.qp.transfer_ordered(false, core.ledger@.submitted, cited)
                    == closure_from(core.ledger@.submitted.map(), cited));
            }
            let post_res = {
                let tracked ledger_ref = core.ledger.borrow_mut();
                rdma::post_rdma_read_gdr(
                    &mut rdma.qp,
                    &mut rdma.cq,
                    Tracked(instance),
                    &slice,
                    remote_cap,
                    wr_id,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(dst_tok),
                    Tracked(&mut ledger_ref.submitted),
                    Tracked(&ledger_ref.knows),
                    Ghost(cited),
                )
            };
            match post_res {
                Ok((ev, sub, pending)) => (ev, sub, Tracked(PullSlotToken::GpuGdr(pending.get()))),
                Err((e, ret_tok)) => {
                    let tracked sh = remote_perm.into_shared();
                    proof {
                        assert(*core == before_core);
                        lemma_peer_core_step_refl(core, rdma.timeline_id);
                    }
                    return Err((rdma_error(e), handle, ret_tok, Tracked(sh)));
                },
            }
        },
    };
    let tracked out_share = remote_perm.into_shared();
    let ghost raw_receipt = receipt@;
    let ghost after_post_core = *core;
    proof {
        lemma_post_pull_prepare_batch(
            &before_core,
            core,
            &*old(rdma),
            &*rdma,
            before_cq,
            wr_id,
            ev@,
            raw_receipt,
            cited,
            remote_cap.slot_ref(),
            handle,
            expected_guards,
        );
    }
    let mut posted: Vec<rdma::PostedBatchItem> = Vec::with_capacity(1);
    posted.push(rdma::PostedBatchItem { wr_id, event: ev, submitted: receipt });
    let frames = [handle.frame];
    let ops = [DataOperation::Read];
    let tickets = rdma.record_data_batch(core, &frames, &ops, posted);
    let ticket = tickets[0];
    let ghost copy = GuardedReplicaCopy {
        before: before_core.ledger@,
        source: source_cert,
        source_slot: source_cert.slot,
        share: share_id,
        writer: ev@,
        destination: handle.slot@,
    };
    let out_pull = InFlightRdmaPull {
        target,
        handle,
        ticket,
        wr_id,
        writer: ev,
        copy: Ghost(copy),
        share_cap: Tracked(out_share),
        dst_token,
    };
    proof {
        lemma_post_pull_finish_valid(
            &before_core,
            core,
            rdma,
            &out_pull,
            source_cert,
            raw_receipt,
            cited,
        );
        lemma_peer_core_step_trans(&before_core, &after_post_core, core, rdma.timeline_id);
    }
    Ok(out_pull)
}

/// Complete an in-flight RDMA READ pull after `RdmaBackend::poll_cq` reports
/// `PeerCompletion::Data(op, status)` with `op.sequence == pull.wr_id`.
///
/// For `PullTarget::GpuGdr`, unconditionally executes `mr.complete_gdr_slot`
/// before unsealing the destination `SlotToken`.
/// On `Ok(())`, calls `instance.observe_share_bound` -> `core.record_revoked_map`
/// -> `pull.copy@.certify` -> `core.finish_write` and returns `PullOutcome::Published`.
/// On `Err(code)`, calls `core.abort_write` and returns `PullOutcome::Aborted`.
pub fn finish_pull_read<'a>(
    rdma: &RdmaBackend<'a>,
    core: &mut CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    pull: InFlightRdmaPull,
    status: Result<(), i32>,
) -> (r: Result<PullOutcome, (CoreError, InFlightRdmaPull)>)
    requires
        old(core).well_formed(),
        rdma.well_formed(old(core)),
        instance.id() == old(core).instance_id(),
        pull.valid(rdma, old(core)),
        old(core).ledger@.knows.set().contains((old(core).node(), pull.writer@)),
        status.is_ok() ==> {
            &&& old(core).citations().contains(pull.writer@)
            &&& event_succeeded(old(core).ledger@, old(core).node(), pull.writer@)
            &&& (!old(core).frames.versions@.contains_key(pull.handle.version) || old(
                core,
            ).frames.versions@[pull.handle.version] == pull.copy@.source.bytes)
            &&& pull.handle.version.well_formed()
            &&& pull.handle.version != VersionId::NONE
        },
    ensures
        final(core).well_formed(),
        rdma.well_formed(final(core)),
        final(core).node_id == old(core).node_id,
        final(core).instance_id() == old(core).instance_id(),
        final(core).tiers == old(core).tiers,
        final(core).frames.same_layout(&old(core).frames),
        final(core).frames.preserves_remote_leases(&old(core).frames),
        final(core).version_registry == old(core).version_registry,
        peer_core_step_except(old(core), final(core), rdma.timeline_id, Some(pull.handle.frame)),
        old(core).ledger@.submitted.map().submap_of(final(core).ledger@.submitted.map()),
        old(core).citations().subset_of(final(core).citations()),
        match r {
            Err((_, ret_pull)) => *final(core) == *old(core) && ret_pull.valid(rdma, final(core)),
            Ok(PullOutcome::Published { frame, version, share_cap }) => {
                &&& status.is_ok()
                &&& frame == pull.handle.frame
                &&& version == pull.handle.version
                &&& final(core).frames.meta(frame).state == STATE_PUBLISHED
                &&& final(core).frames.frame(frame).bound_version == Some(version)
                &&& final(core).frames.versions@ == old(core).frames.versions@.insert(
                    version,
                    pull.copy@.source.bytes,
                )
                &&& final(core).frames.versions@.contains_key(version)
                &&& final(core).frames.versions@[version] == pull.copy@.source.bytes
                &&& share_cap@.well_formed()
                &&& share_cap@.is_rdma_bound()
                &&& share_cap@.instance_id() == pull.share_cap@.instance_id()
                &&& share_cap@.loc() == pull.share_cap@.loc()
                &&& share_cap@.share() == pull.share_cap@.share()
                &&& share_cap@.frontier() == pull.share_cap@.frontier()
            },
            Ok(PullOutcome::Aborted { status: err_code, share_cap }) => {
                &&& status == Err(err_code)
                &&& final(core).frames.same_directory(&old(core).frames)
                &&& share_cap@ == pull.share_cap@
            },
        },
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    let InFlightRdmaPull { target, handle, ticket, wr_id, writer, copy, share_cap, dst_token } =
        pull;
    let Tracked(mut tok): Tracked<SlotToken> = match target {
        PullTarget::Host => {
            let tracked t = match dst_token.get() {
                PullSlotToken::Host(t) => t,
                _ => {
                    assert(false);
                    proof_from_false()
                },
            };
            Tracked(t)
        },
        PullTarget::GpuGdr => {
            let tracked pending = match dst_token.get() {
                PullSlotToken::GpuGdr(p) => p,
                _ => {
                    assert(false);
                    proof_from_false()
                },
            };
            proof {
                rdma.lemma_effective_slabs_wf(core);
                lemma_core_wf_facts(core);
            }
            let slab = rdma.effective_gpu_mr().unwrap();
            proof {
                assert(slab.mr.node() == core.node());
                assert(pending.loc().node == core.node());
            }
            match slab.mr.complete_gdr_slot(
                Tracked(instance),
                Tracked(pending),
                Tracked(&core.ledger.borrow().knows),
            ) {
                Ok(t) => t,
                Err((e, pending_back)) => {
                    proof {
                        reveal(FrameTable::same_layout);
                        lemma_peer_core_step_except_refl(
                            core,
                            rdma.timeline_id,
                            Some(handle.frame),
                        );
                    }
                    return Err(
                        (
                            cuda_error(e),
                            InFlightRdmaPull {
                                target,
                                handle,
                                ticket,
                                wr_id,
                                writer,
                                copy,
                                share_cap,
                                dst_token: Tracked(PullSlotToken::GpuGdr(pending_back.get())),
                            },
                        ),
                    );
                },
            }
        },
    };
    match status {
        Ok(()) => {
            let ghost before_rev = *core;
            let tracked mut sh = share_cap.get();
            let tracked revoked_map;
            proof {
                revoked_map =
                instance.observe_share_bound(
                    copy@.source.writer,
                    &mut sh,
                    &core.ledger.borrow().submitted,
                );
            }
            core.record_revoked_map(Tracked(revoked_map));
            let ghost cert;
            proof {
                lemma_core_wf_facts(core);
                lemma_guarded_copy_revoked_growth(
                    copy@,
                    before_rev.ledger@,
                    core.ledger@,
                    core.node(),
                );
                lemma_pull_dst_latest(
                    copy@.writer,
                    copy@.destination.loc,
                    &core.ledger.borrow().submitted,
                    &mut tok,
                    instance,
                );
                lemma_lift_share_bound(
                    core.ledger@.submitted,
                    revoked_map,
                    core.ledger@.revoked,
                    copy@.source_slot.loc,
                    copy@.share,
                    copy@.source.writer,
                );
                cert = copy@.certify(core.ledger@, core.node());
                assert(cert.bytes =~= copy@.source.bytes);
            }
            let frame = handle.frame;
            let version = handle.version;
            let ghost before_fin = *core;
            core.finish_write(handle, Tracked(tok), ticket, Ghost(cert));
            proof {
                lemma_core_wf_facts(core);
                lemma_rdma_window_after_fence_ledger(
                    rdma.in_flight_ops,
                    rdma.bitmap,
                    &core.timelines,
                    rdma.timeline_id,
                    &before_rev.ledger@,
                    &core.ledger@,
                    rdma::CqSnapshot::of(rdma.cq),
                    core.instance_id(),
                    rdma.next_control_wr_id,
                );
                lemma_completed_pulls_after_step(
                    rdma.completed_pulls,
                    &before_rev.timelines,
                    &core.timelines,
                    rdma.timeline_id,
                    rdma.cq.observer(),
                    &before_rev.ledger@,
                    &core.ledger@,
                );
                lemma_rdma_slabs_after_init_finish(
                    &rdma.host_mr,
                    &rdma.gpu_mr,
                    rdma.shared_host_mr,
                    rdma.shared_gpu_mr,
                    &before_rev,
                    core,
                    rdma.window_pool.pd(),
                );
                lemma_rdma_leases_after_init_finish(
                    rdma.active_leases@,
                    &rdma.window_pool,
                    &rdma.qp,
                    &before_rev,
                    core,
                    frame,
                );
                reveal(peer_core_step_except);
                reveal(FrameTable::same_layout);
                reveal(FrameTable::preserves_initializing);
                broadcast use {
                    vstd::map::group_map_lemmas,
                    vstd::imap::group_imap_lemmas,
                    vstd::iset::group_iset_lemmas,
                };

            }
            Ok(PullOutcome::Published { frame, version, share_cap: Tracked(sh) })
        },
        Err(err_code) => {
            let ghost before_abort = *core;
            core.abort_write(handle, Tracked(tok));
            proof {
                lemma_completed_pulls_after_step(
                    rdma.completed_pulls,
                    &before_abort.timelines,
                    &core.timelines,
                    rdma.timeline_id,
                    rdma.cq.observer(),
                    &before_abort.ledger@,
                    &core.ledger@,
                );
                lemma_rdma_slabs_after_init_finish(
                    &rdma.host_mr,
                    &rdma.gpu_mr,
                    rdma.shared_host_mr,
                    rdma.shared_gpu_mr,
                    &before_abort,
                    core,
                    rdma.window_pool.pd(),
                );
                reveal(rdma_leases_valid);
                reveal(FrameTable::same_directory);
                reveal(FrameTable::same_other_pins);
                assert forall|i: int| 0 <= i < rdma.active_leases@.len() implies (
                #[trigger] rdma.active_leases@[i]).valid(&rdma.window_pool, &rdma.qp, core) by {
                    assert(rdma.active_leases@[i].valid(
                        &rdma.window_pool,
                        &rdma.qp,
                        &before_abort,
                    ));
                    let f = rdma.active_leases@[i].frame;
                    assert(core.frames.frame(f).slot == before_abort.frames.frame(f).slot);
                    assert(core.frames.published(f) == before_abort.frames.published(f));
                    assert(core.frames.frame(f).remote_shares == before_abort.frames.frame(
                        f,
                    ).remote_shares);
                    assert(core.frames.meta(f).active_remote_leases == before_abort.frames.meta(
                        f,
                    ).active_remote_leases);
                }
                reveal(peer_core_step_except);
                reveal(FrameTable::same_layout);
                reveal(FrameTable::preserves_initializing);
                broadcast use {
                    vstd::map::group_map_lemmas,
                    vstd::imap::group_imap_lemmas,
                    vstd::iset::group_iset_lemmas,
                };

            }
            Ok(PullOutcome::Aborted { status: err_code, share_cap })
        },
    }
}

proof fn lemma_pull_dst_latest(
    writer: TransferEvent,
    dst_loc: SlotLoc,
    tracked submitted: &transfer::context::SubmittedMap,
    tracked tok: &mut SlotToken,
    tracked instance: &TransferInstance,
)
    requires
        submitted.instance_id() == instance.id(),
        submitted.dom().contains(writer),
        old(tok).well_formed(),
        old(tok).instance_id() == instance.id(),
        old(tok).loc() == dst_loc,
        foundation::tracked::slot_base_frontier(old(tok).value()).last_writer == Some(writer),
    ensures
        final(tok).well_formed(),
        final(tok).instance_id() == old(tok).instance_id(),
        final(tok).loc() == old(tok).loc(),
        final(tok).value() == old(tok).value(),
        forall|e: TransferEvent| #[trigger]
            submitted.dom().contains(e) && submitted.map()[e].access.writes.contains(dst_loc) && e
                != writer ==> submitted.map()[writer].preds.contains(e),
{
    if exists|e: TransferEvent|
        submitted.dom().contains(e) && (#[trigger] submitted.map()[e]).access.writes.contains(
            dst_loc,
        ) && e != writer && !submitted.map()[writer].preds.contains(e) {
        let e = choose|e: TransferEvent|
            submitted.dom().contains(e) && (#[trigger] submitted.map()[e]).access.writes.contains(
                dst_loc,
            ) && e != writer && !submitted.map()[writer].preds.contains(e);
        instance.slot_writer_is_latest(writer, e, tok, submitted);
        assert(false);
    }
}

proof fn lemma_guarded_copy_revoked_growth(
    copy: GuardedReplicaCopy,
    before: crate::provenance::KnowledgeLedger,
    after: crate::provenance::KnowledgeLedger,
    node: NodeId,
)
    requires
        copy.retained(before, node),
        after.submitted == before.submitted,
        after.knows == before.knows,
        after.succeeded == before.succeeded,
        revoked_submap(before.revoked, after.revoked),
    ensures
        copy.retained(after, node),
        after.submitted.dom().contains(copy.writer),
{
}

proof fn lemma_lift_share_bound(
    submitted: transfer::context::SubmittedMap,
    revoked_map: transfer::context::RevokedMap,
    ledger_revoked: transfer::context::RevokedMap,
    loc: SlotLoc,
    share: TransferShare,
    src_writer: TransferEvent,
)
    requires
        revoked_submap(revoked_map, ledger_revoked),
        forall|e: TransferEvent| #[trigger]
            submitted.dom().contains(e) && submitted.map()[e].access.writes.contains(loc) && e
                != src_writer ==> (submitted.map()[src_writer].preds.contains(e) || (
            revoked_map.map().dom().contains((loc, share)) && submitted.map()[e].preds.contains(
                revoked_map.map()[(loc, share)],
            ))),
    ensures
        forall|e: TransferEvent| #[trigger]
            submitted.dom().contains(e) && submitted.map()[e].access.writes.contains(loc) && e
                != src_writer ==> (submitted.map()[src_writer].preds.contains(e) || (
            ledger_revoked.dom().contains((loc, share)) && submitted.map()[e].preds.contains(
                ledger_revoked.map()[(loc, share)],
            ))),
{
    let k = (loc, share);
    if revoked_map.map().dom().contains(k) {
        assert(revoked_map.dom().contains(k));
        assert(ledger_revoked.dom().contains(k));
        assert(ledger_revoked.map()[k] == revoked_map.map()[k]);
    }
}

proof fn lemma_rdma_slabs_after_init_finish<'a>(
    host_mr: &Option<RegisteredSlab<'a>>,
    gpu_mr: &Option<RegisteredSlab<'a>>,
    shared_host_mr: Option<&'a RegisteredSlab<'a>>,
    shared_gpu_mr: Option<&'a RegisteredSlab<'a>>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    pd_id: nat,
)
    requires
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.tiers == before_core.tiers,
        after_core.frames.same_layout(&before_core.frames),
        host_mr matches Some(ref s) ==> s.well_formed(before_core, pd_id),
        gpu_mr matches Some(ref s) ==> s.well_formed(before_core, pd_id),
        shared_host_mr matches Some(s) ==> s.well_formed_shared(before_core, pd_id),
        shared_gpu_mr matches Some(s) ==> s.well_formed_shared(before_core, pd_id),
    ensures
        host_mr matches Some(ref s) ==> s.well_formed(after_core, pd_id),
        gpu_mr matches Some(ref s) ==> s.well_formed(after_core, pd_id),
        shared_host_mr matches Some(s) ==> s.well_formed_shared(after_core, pd_id),
        shared_gpu_mr matches Some(s) ==> s.well_formed_shared(after_core, pd_id),
{
    if let Some(ref s) = *host_mr {
        lemma_slab_bound_same_layout(
            before_core,
            after_core,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    if let Some(ref s) = *gpu_mr {
        lemma_slab_bound_same_layout(
            before_core,
            after_core,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    if let Some(s) = shared_host_mr {
        s.lemma_shared_wf_iff(before_core, pd_id);
        lemma_slab_bound_same_layout(
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
        lemma_slab_bound_same_layout(
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

proof fn lemma_rdma_leases_after_init_finish<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    touched_frame: FrameId,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        rdma_leases_valid(leases, pool, qp, before_core),
        after_core.node_id == before_core.node_id,
        after_core.instance_id() == before_core.instance_id(),
        after_core.frames.same_layout(&before_core.frames),
        touched_frame.0 < before_core.frames.metas.len(),
        before_core.frames.meta(touched_frame).state == STATE_INITIALIZING,
        forall|g: FrameId|
            g.0 < before_core.frames.metas.len() && g != touched_frame ==> {
                &&& (#[trigger] after_core.frames.frame(g)) == before_core.frames.frame(g)
                &&& after_core.frames.meta(g).active_remote_leases == before_core.frames.meta(
                    g,
                ).active_remote_leases
            },
    ensures
        rdma_leases_valid(leases, pool, qp, after_core),
{
    reveal(rdma_leases_valid);
    reveal(CacheCore::well_formed);
    assert forall|i: int| 0 <= i < leases.len() implies (#[trigger] leases[i]).valid(
        pool,
        qp,
        after_core,
    ) by {
        assert(leases[i].valid(pool, qp, before_core));
        let f = leases[i].frame;
        if leases[i].is_unpinned() {
            assert(after_core.frames.frame(f).slot == before_core.frames.frame(f).slot) by {
                reveal(FrameTable::same_layout);
            }
        } else {
            assert(f != touched_frame) by {
                if f == touched_frame {
                    assert(before_core.frames.valid_frame(f, before_core.kernel_view()));
                    assert(before_core.frames.meta(f).active_remote_leases > 0);
                }
            }
            assert(after_core.frames.frame(f) == before_core.frames.frame(f));
            assert(after_core.frames.meta(f).active_remote_leases == before_core.frames.meta(
                f,
            ).active_remote_leases);
        }
    }
}

} // verus!
