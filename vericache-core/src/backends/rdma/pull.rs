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
    GpuVisible(SlotToken),
    BatchPending,
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
    pub visibility_flushed: bool,
}

/// Outcome of `finish_pull_read`.
pub enum PullOutcome {
    Published { frame: FrameId, version: VersionId, share_cap: Tracked<ShareCap> },
    Aborted { status: i32, share_cap: Tracked<ShareCap> },
}

impl InFlightRdmaPull {
    pub open(crate) spec fn valid(&self, rdma: &RdmaBackend, core: &CacheCore) -> bool {
        &&& self.ticket != Ticket::NONE
        &&& self.ticket.timeline_id() == rdma.timeline_id
        &&& self.ticket.sequence() == self.wr_id
        &&& core.timelines_spec().issued(rdma.timeline_id, self.wr_id)
        &&& core.timelines_spec().events_spec()@[(rdma.timeline_id, self.wr_id)] == self.writer@
        &&& self.handle.frame.0 < core.frames_spec().metas_spec().len()
        &&& core.frames_spec().meta(self.handle.frame).state == STATE_INITIALIZING
        &&& !core.frames_spec().meta(self.handle.frame).queued()
        &&& core.frames_spec().frame(self.handle.frame).token is None
        &&& core.frames_spec().frame(self.handle.frame).tier == self.handle.tier
        &&& core.frames_spec().frame(self.handle.frame).slot == self.handle.slot@
        &&& core.frames_spec().frame(self.handle.frame).bound_key == Some(self.handle.key)
        &&& core.frames_spec().frame(self.handle.frame).bound_version == Some(self.handle.version)
        &&& self.handle.version.well_formed() && self.handle.version != VersionId::NONE
        &&& (self.handle.version.origin_node != core.node_id_spec() || self.handle.version.seq
            < core.next_version_seq_spec() || self.handle.version.seq
            >= core.version_seq_limit_spec())
        &&& self.handle.slot@.is_full()
        &&& self.handle.slot@.loc.node == core.node()
        &&& self.handle.slot@.slot_len == core.frames_spec().tier_frame_size(self.handle.tier)
        &&& self.copy@.writer == self.writer@
        &&& self.copy@.destination == self.handle.slot@
        &&& self.copy@.source_slot == self.copy@.source.slot
        &&& self.copy@.source.bytes.len() == self.handle.slot@.slot_len
        &&& self.copy@.retained(core.ledger_spec()@, core.node())
        &&& self.share_cap@.is_fenced()
        &&& self.share_cap@.instance_id() == core.instance_id()
        &&& self.share_cap@.loc() == self.copy@.source_slot.loc
        &&& self.share_cap@.share() == self.copy@.share
        &&& self.share_cap@.frontier().last_writer == Some(self.copy@.source.writer)
        &&& self.share_cap@.frontier().readers.contains(self.writer@)
        &&& match (self.target, self.dst_token@) {
            (PullTarget::Host, PullSlotToken::Host(tok)) => !self.visibility_flushed && slot_owned(
                tok,
                core.instance_id(),
                self.handle.slot@.loc,
                Frontier { last_writer: Some(self.writer@), readers: ISet::empty() },
            ),
            (PullTarget::GpuGdr, PullSlotToken::GpuGdr(pending)) => {
                &&& !self.visibility_flushed
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
            (PullTarget::GpuGdr, PullSlotToken::GpuVisible(tok)) => self.visibility_flushed
                && slot_owned(
                tok,
                core.instance_id(),
                self.handle.slot@.loc,
                Frontier { last_writer: Some(self.writer@), readers: ISet::empty() },
            ),
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
        !core.timelines_spec().issued(rdma.timeline_id, wr_id),
        handle.frame.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(handle.frame).state == STATE_INITIALIZING,
        core.frames_spec().frame(handle.frame).tier == handle.tier,
        core.frames_spec().frame(handle.frame).slot == handle.slot@,
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
        handle.slot@.slot_len == core.frames_spec().tier_frame_size(handle.tier),
        core.disjoint_from_published(ISet::empty().insert(handle.slot@.loc)),
        read_admitted(
            share_cap.frontier(),
            closure_from(core.ledger_spec()@.submitted.map(), core.citations()),
        ),
        write_admitted(
            slot_frontier(dst_tok),
            closure_from(core.ledger_spec()@.submitted.map(), core.citations()),
        ),
        core.citations().subset_of(core.ledger_spec()@.submitted.dom()),
        core.ledger_spec()@.submitted.instance_id() == core.instance_id(),
        core.ledger_spec()@.knows.instance_id() == core.instance_id(),
        foundation::tracked::knowledge_of(core.node(), core.citations()).subset_of(
            core.ledger_spec()@.knows.set(),
        ),
        closure_from(core.ledger_spec()@.submitted.map(), core.citations()).contains(
            source_cert.writer,
        ),
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
        wr_id == before_core.timelines_spec().next_seq_spec()[old_rdma.timeline_id as int],
        wr_id <= MAX_TICKET_SEQ,
        wr_id <= before_core.timelines_spec().confirmed_seq_spec()[old_rdma.timeline_id as int]
            + BITMAP_CAPACITY,
        !before_cq.pending.contains_key(wr_id),
        before_core.disjoint_from_published(ISet::empty().insert(handle.slot@.loc)),
        core.node_id_spec() == before_core.node_id_spec(),
        core.tiers_spec() == before_core.tiers_spec(),
        core.frames_spec() == before_core.frames_spec(),
        core.catalog_spec() == before_core.catalog_spec(),
        core.timelines_spec() == before_core.timelines_spec(),
        core.active_leases_spec() == before_core.active_leases_spec(),
        core.free_lease_slots_spec() == before_core.free_lease_slots_spec(),
        core.lease_generation_spec() == before_core.lease_generation_spec(),
        core.version_seq_start_spec() == before_core.version_seq_start_spec(),
        core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        core.version_registry_spec() == before_core.version_registry_spec(),
        core.spec_spec() == before_core.spec_spec(),
        core.ledger_spec()@.submitted.map() == before_core.ledger_spec()@.submitted.map(),
        core.ledger_spec()@.submitted.instance_id()
            == before_core.ledger_spec()@.submitted.instance_id(),
        core.ledger_spec()@.knows == before_core.ledger_spec()@.knows,
        core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
        core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
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
            before_core.ledger_spec()@.submitted,
            cited,
            ev,
            receipt,
        ),
        old_rdma.qp.transfer_ordered(false, before_core.ledger_spec()@.submitted, cited)
            == closure_from(before_core.ledger_spec()@.submitted.map(), cited),
    ensures
        core.well_formed(),
        peer_core_step(before_core, core, rdma.timeline_id),
        rdma.timeline_id < core.timelines_spec().count_spec(),
        core.timelines_spec().kinds_spec()[rdma.timeline_id as int] == (
        TimelineKind::RdmaQueuePair { peer_node: rdma.peer_node }),
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
            &core.timelines_spec(),
            rdma.timeline_id,
            rdma.cq.observer(),
            &core.ledger_spec()@,
        ),
        rdma.bitmap.well_formed(),
        rdma.bitmap.observations(
            &core.timelines_spec(),
            rdma.timeline_id,
            core.node(),
            &core.ledger_spec()@,
        ),
        CONTROL_WR_ID_BASE <= rdma.next_control_wr_id <= CONTROL_WR_ID_LIMIT,
        forall|s: u64|
            rdma.cq.in_flight().contains_key(s) && !((
            #[trigger] rdma.cq.in_flight()[s]).kind is DataTransfer) ==> CONTROL_WR_ID_BASE <= s
                < rdma.next_control_wr_id,
        receipt.instance_id() == core.instance_id(),
        receipt.key() == ev,
        receipt.key().observer == core.node(),
        receipt.value().preds.subset_of(core.ledger_spec()@.submitted.dom()),
        core.disjoint_from_published(receipt.value().access.writes),
        rdma.pending().contains_key(wr_id),
        rdma.pending()[wr_id] == ev,
        window_matches(
            rdma.in_flight_ops,
            rdma.bitmap,
            &core.timelines_spec(),
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
        assert(before_core.ledger_spec()@.submitted.map().submap_of(
            core.ledger_spec()@.submitted.map(),
        ));
        assert(revoked_submap(before_core.ledger_spec()@.revoked, core.ledger_spec()@.revoked));
        assert(before_core.ledger_spec()@.knows.set().subset_of(core.ledger_spec()@.knows.set()));
        assert(before_core.ledger_spec()@.succeeded.dom().subset_of(
            core.ledger_spec()@.succeeded.dom(),
        ));
        assert(core.citations() =~= before_core.citations());
        assert(core.citations().subset_of(core.citations()));
        assert(core.kernel_view().extends(before_core.kernel_view())) by {
            assert forall|u: TimelineId|
                u
                    < before_core.timelines_spec().count_spec() implies #[trigger] core.timelines_spec().confirmed_seq_spec()[u as int]
                >= before_core.timelines_spec().confirmed_seq_spec()[u as int] by {}
            assert forall|t: TimelineId, s: u64|
                t < before_core.timelines_spec().count_spec()
                    && #[trigger] before_core.timelines_spec().issued(
                    t,
                    s,
                ) implies core.timelines_spec().issued(t, s)
                && core.timelines_spec().events_spec()@[(t, s)]
                == before_core.timelines_spec().events_spec()@[(t, s)] by {}
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
        &core.timelines_spec(),
        rdma.timeline_id,
        rdma.cq.observer(),
        &core.ledger_spec()@,
    ));
    assert(rdma.bitmap.observations(
        &core.timelines_spec(),
        rdma.timeline_id,
        core.node(),
        &core.ledger_spec()@,
    ));
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
    assert(receipt.value().preds == closure_from(
        before_core.ledger_spec()@.submitted.map(),
        cited,
    ));
    assert(receipt.value().preds.subset_of(core.ledger_spec()@.submitted.dom())) by {
        assert forall|e: TransferEvent| #[trigger]
            receipt.value().preds.contains(e) implies core.ledger_spec()@.submitted.dom().contains(
            e,
        ) by {
            assert(before_core.ledger_spec()@.submitted.dom().contains(e));
            assert(before_core.ledger_spec()@.submitted.map().contains_key(e));
        }
    };
    assert(receipt.value().access.writes =~= ISet::empty().insert(handle.slot@.loc));
    assert(core.disjoint_from_published(receipt.value().access.writes));
    assert(peer_core_step(before_core, core, rdma.timeline_id)) by {
        reveal(peer_core_step_except);
        reveal(core_transition);
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
        !pull.visibility_flushed,
        pull.ticket != Ticket::NONE,
        pull.ticket.timeline_id() == rdma.timeline_id,
        pull.ticket.sequence() == pull.wr_id,
        core.timelines_spec().issued(rdma.timeline_id, pull.wr_id),
        core.timelines_spec().events_spec()@.dom().contains((rdma.timeline_id, pull.wr_id))
            ==> core.timelines_spec().events_spec()@[(rdma.timeline_id, pull.wr_id)]
            == pull.writer@,
        pull.handle.frame.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(pull.handle.frame).state == STATE_INITIALIZING,
        !core.frames_spec().meta(pull.handle.frame).queued(),
        core.frames_spec().frame(pull.handle.frame).token is None,
        core.frames_spec().frame(pull.handle.frame).tier == pull.handle.tier,
        core.frames_spec().frame(pull.handle.frame).slot == pull.handle.slot@,
        core.frames_spec().frame(pull.handle.frame).bound_key == Some(pull.handle.key),
        core.frames_spec().frame(pull.handle.frame).bound_version == Some(pull.handle.version),
        pull.handle.version.well_formed() && pull.handle.version != VersionId::NONE,
        pull.handle.version.origin_node != core.node_id_spec() || pull.handle.version.seq
            < core.next_version_seq_spec() || pull.handle.version.seq
            >= core.version_seq_limit_spec(),
        pull.handle.slot@.is_full(),
        pull.handle.slot@.well_formed(),
        pull.handle.slot@.loc.node == core.node(),
        pull.handle.slot@.slot_len == core.frames_spec().tier_frame_size(pull.handle.tier),
        source_cert.slot.well_formed(),
        source_cert.slot.is_full(),
        source_cert.slot.sub_len == pull.handle.slot@.slot_len,
        source_cert.bytes.len() == pull.handle.slot@.slot_len,
        before_core.ledger_spec()@.submitted.dom().contains(source_cert.writer),
        forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                before_core.ledger_spec()@,
                before_core.node(),
                h,
                run,
            ) ==> #[trigger] certificate_valid_in(source_cert, h, run),
        pull.copy@ == (GuardedReplicaCopy {
            before: before_core.ledger_spec()@,
            source: source_cert,
            source_slot: source_cert.slot,
            share: pull.share_cap@.share(),
            writer: pull.writer@,
            destination: pull.handle.slot@,
        }),
        before_core.ledger_spec()@.submitted.map().submap_of(core.ledger_spec()@.submitted.map()),
        core.ledger_spec()@.knows == before_core.ledger_spec()@.knows,
        core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
        core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
        core.node() == before_core.node(),
        core.ledger_spec()@.submitted.dom().contains(pull.writer@),
        core.ledger_spec()@.submitted.map()[pull.writer@] == receipt.value(),
        receipt.value().access == copy_access(source_cert.slot, pull.handle.slot@),
        receipt.value().effect == copy_effect(source_cert.slot, pull.handle.slot@),
        receipt.value().read_guards == IMap::empty().insert(
            source_cert.slot.loc,
            pull.share_cap@.share(),
        ),
        receipt.value().preds == closure_from(before_core.ledger_spec()@.submitted.map(), cited),
        pull.share_cap@.is_fenced(),
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
    assert(before_core.ledger_spec()@.valid(
        core.ledger_spec()@.submitted.instance_id(),
        core.node(),
    ));
    assert(revoked_submap(before_core.ledger_spec()@.revoked, core.ledger_spec()@.revoked));
    assert(before_core.ledger_spec()@.citations(core.node()).subset_of(
        core.ledger_spec()@.citations(core.node()),
    )) by {
        assert forall|e: TransferEvent| #[trigger]
            before_core.ledger_spec()@.citations(core.node()).contains(
                e,
            ) implies core.ledger_spec()@.citations(core.node()).contains(e) by {
            assert(before_core.ledger_spec()@.submitted.map().contains_key(e));
            assert(core.ledger_spec()@.submitted.map().contains_key(e));
        }
    };
    assert(core.ledger_spec()@.submitted.dom().contains(source_cert.writer)) by {
        assert(before_core.ledger_spec()@.submitted.map().contains_key(source_cert.writer));
    };
    assert(receipt.value().preds.contains(source_cert.writer));
    assert(receipt.value().preds.subset_of(core.ledger_spec()@.submitted.dom())) by {
        assert forall|e: TransferEvent| #[trigger]
            receipt.value().preds.contains(e) implies core.ledger_spec()@.submitted.dom().contains(
            e,
        ) by {
            assert(before_core.ledger_spec()@.submitted.dom().contains(e));
            assert(before_core.ledger_spec()@.submitted.map().contains_key(e));
        }
    };
    assert(source_slice_valid(source_cert, source_cert.slot));
    assert(crate::provenance::effect_implies(
        core.ledger_spec()@.submitted.map()[pull.writer@].effect,
        copy_effect(source_cert.slot, pull.handle.slot@),
    ));
    assert(pull.copy@.retained(core.ledger_spec()@, core.node()));
}

/// A reserved destination and the checked provenance carried by its read grant.
pub struct PreparedPull {
    pub handle: WriteHandle,
    pub remote: rdma::RemoteCapability,
    pub dst: Tracked<SlotToken>,
    pub share: Tracked<ShareCap>,
    pub source: Ghost<ValueCertificate>,
}

pub open(crate) spec fn pull_admitted(
    rdma: &RdmaBackend,
    core: &CacheCore,
    target: PullTarget,
    handle: WriteHandle,
    dst_tok: SlotToken,
    remote_cap: &rdma::RemoteCapability,
    share_cap: ShareCap,
    source_cert: ValueCertificate,
) -> bool {
    &&& core.frames_spec().tier_frame_size(handle.tier) as u64 <= rdma::MAX_RDMA_MSG_LEN
    &&& handle.frame.0 < core.frames_spec().metas_spec().len()
    &&& core.frames_spec().meta(handle.frame).state == STATE_INITIALIZING
    &&& !core.frames_spec().meta(handle.frame).queued()
    &&& core.frames_spec().frame(handle.frame).token is None
    &&& core.frames_spec().frame(handle.frame).tier == handle.tier
    &&& core.frames_spec().frame(handle.frame).slot == handle.slot@
    &&& core.frames_spec().frame(handle.frame).bound_key == Some(handle.key)
    &&& core.frames_spec().frame(handle.frame).bound_version == Some(handle.version)
    &&& handle.version.well_formed() && handle.version != VersionId::NONE
    &&& (handle.version.origin_node != core.node_id_spec() || handle.version.seq
        < core.next_version_seq_spec() || handle.version.seq >= core.version_seq_limit_spec())
    &&& slot_is_owned(dst_tok, core.instance_id(), handle.slot@.loc)
    &&& write_admitted(slot_frontier(dst_tok), core.citations())
    &&& remote_cap.well_formed()
    &&& remote_cap.slot_ref().loc.node == rdma.peer_node
    &&& remote_cap.slot_ref().loc.node != core.node()
    &&& remote_cap.access() == rdma::RemoteAccess::ReadOnly
    &&& remote_cap.share() == Some(share_cap.share())
    &&& remote_cap.slot_ref() == source_cert.slot
    &&& source_cert.slot.is_full()
    &&& source_cert.slot.sub_len == core.frames_spec().tier_frame_size(handle.tier)
    &&& source_cert.bytes.len() == core.frames_spec().tier_frame_size(handle.tier)
    &&& core.ledger_spec()@.submitted.dom().contains(source_cert.writer)
    &&& core.citations().contains(source_cert.writer)
    &&& forall|h, run|
        reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
            core.ledger_spec()@,
            core.node(),
            h,
            run,
        ) ==> #[trigger] certificate_valid_in(source_cert, h, run)
    &&& share_cap.is_fenced()
    &&& share_cap.instance_id() == core.instance_id()
    &&& share_cap.loc() == remote_cap.slot_ref().loc
    &&& share_cap.frontier().last_writer == Some(source_cert.writer)
    &&& read_admitted(share_cap.frontier(), core.citations())
    &&& match target {
        PullTarget::Host => rdma.effective_host_mr_spec() matches Some(s) && s.owns(handle.frame)
            && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
        PullTarget::GpuGdr => rdma.effective_gpu_mr_spec() matches Some(s) && s.owns(handle.frame)
            && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
    }
}

/// Post an RDMA READ pull into a reserved initializing frame (`PullTarget::Host`
/// or `PullTarget::GpuGdr`), recording the submission on `rdma` and `core`.
///
/// On failure, returns `(err, handle, dst_tok, share_cap)` intact without leaking.
pub(crate) fn post_pull_read<'a>(
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
        pull_admitted(
            old(rdma),
            old(core),
            target,
            handle,
            dst_tok,
            remote_cap,
            share_cap,
            source_cert,
        ),
    ensures
        final(core).well_formed(),
        final(rdma).well_formed(final(core)),
        final(core).node_id_spec() == old(core).node_id_spec(),
        final(core).instance_id() == old(core).instance_id(),
        final(core).tiers_spec() == old(core).tiers_spec(),
        final(core).frames_spec() == old(core).frames_spec(),
        final(core).catalog_spec() == old(core).catalog_spec(),
        final(core).spec_spec() == old(core).spec_spec(),
        final(core).version_registry_spec() == old(core).version_registry_spec(),
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
                &&& final(core).ledger_spec()@.submitted.map() == old(
                    core,
                ).ledger_spec()@.submitted.map().insert(
                    pull.writer@,
                    final(core).ledger_spec()@.submitted.map()[pull.writer@],
                )
                &&& old(core).ledger_spec()@.submitted.map().submap_of(
                    final(core).ledger_spec()@.submitted.map(),
                )
                &&& final(core).ledger_spec()@.knows == old(core).ledger_spec()@.knows
                &&& final(core).ledger_spec()@.succeeded == old(core).ledger_spec()@.succeeded
                &&& final(core).ledger_spec()@.revoked == old(core).ledger_spec()@.revoked
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
        Tracked(&core.borrow_ledger().submitted),
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
                assert(rdma.qp.transfer_ordered(false, core.ledger_spec()@.submitted, cited)
                    == closure_from(core.ledger_spec()@.submitted.map(), cited));
            }
            let tracked mut local_tok = dst_tok;
            let post_res = {
                let tracked ledger_ref = core.borrow_ledger_mut();
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
                assert(rdma.qp.transfer_ordered(false, core.ledger_spec()@.submitted, cited)
                    == closure_from(core.ledger_spec()@.submitted.map(), cited));
            }
            let post_res = {
                let tracked ledger_ref = core.borrow_ledger_mut();
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
        before: before_core.ledger_spec()@,
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
        visibility_flushed: false,
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

impl PreparedPull {
    pub open spec fn valid(self, rdma: &RdmaBackend, core: &CacheCore, target: PullTarget) -> bool {
        pull_admitted(
            rdma,
            core,
            target,
            self.handle,
            self.dst@,
            &self.remote,
            self.share@,
            self.source@,
        )
    }

    pub(super) proof fn lemma_after_step(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        old_rdma: &RdmaBackend,
        rdma: &RdmaBackend,
        target: PullTarget,
        retired: Option<FrameId>,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            self.valid(old_rdma, before, target),
            peer_core_step_except(before, after, old_rdma.timeline_id, retired),
            retired != Some(self.handle.frame),
            after.next_version_seq_spec() == before.next_version_seq_spec(),
            rdma.peer_node == old_rdma.peer_node,
            rdma.effective_host_mr_spec() == old_rdma.effective_host_mr_spec(),
            rdma.effective_gpu_mr_spec() == old_rdma.effective_gpu_mr_spec(),
        ensures
            self.valid(rdma, after, target),
            before.citations().subset_of(after.citations()),
    {
        hide(CacheCore::well_formed);
        reveal(peer_core_step_except);
        reveal(core_transition);
        reveal(FrameTable::same_layout);
        reveal(FrameTable::preserves_initializing);
        lemma_core_wf_facts(before);
        lemma_core_wf_facts(after);
        assert forall|e: TransferEvent| #[trigger]
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
        }
        assert forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                after.ledger_spec()@,
                after.node(),
                h,
                run,
            ) implies #[trigger] certificate_valid_in(self.source@, h, run) by {
            crate::provenance::ledger_history_restrict(
                before.ledger_spec()@,
                after.ledger_spec()@,
                after.node(),
                h,
                run,
            );
        }
    }
}

/// The posted prefix owns its in-flight destination and share. The suffix
/// still owns each unchanged reservation and can be retried or aborted.
pub struct PullBatchPost {
    pub posted: Vec<InFlightRdmaPull>,
    pub unposted: Vec<PreparedPull>,
    pub error: Option<CoreError>,
}

struct PullBatchMeta {
    handle: WriteHandle,
    remote: rdma::RemoteCapability,
    source: Ghost<ValueCertificate>,
}

struct PostedPullBatch {
    frames: Vec<FrameId>,
    ops: Vec<DataOperation>,
    meta: Vec<PullBatchMeta>,
    raw: rdma::BatchPostOutcome,
    buffers: Tracked<Map<usize, rdma::BatchReadBuffers>>,
    start_buffers: Ghost<Map<usize, rdma::BatchReadBuffers>>,
}

#[verifier::opaque]
spec fn posted_pull_submission(
    core: &CacheCore,
    item: rdma::PostedBatchItem,
    input: PreparedPull,
) -> bool {
    &&& item.event@ == item.submitted@.key()
    &&& item.event@.observer == core.node()
    &&& item.submitted@.instance_id() == core.instance_id()
    &&& item.submitted@.value().access == copy_access(input.source@.slot, input.handle.slot@)
    &&& item.submitted@.value().effect == copy_effect(input.source@.slot, input.handle.slot@)
    &&& item.submitted@.value().read_guards == IMap::empty().insert(
        input.source@.slot.loc,
        input.share@.share(),
    )
    &&& item.submitted@.value().preds == closure_from(
        core.ledger_spec()@.submitted.map(),
        core.citations(),
    )
}

#[verifier::opaque]
spec fn posted_pull_recorded(
    rdma: &RdmaBackend,
    core: &CacheCore,
    before_core: &CacheCore,
    wr_id: int,
    ticket: Ticket,
    item: rdma::PostedBatchItem,
    input: PreparedPull,
) -> bool {
    &&& item.wr_id == wr_id
    &&& ticket.timeline_id() == rdma.timeline_id && ticket.sequence() == item.wr_id && ticket
        != Ticket::NONE
    &&& core.timelines_spec().issued(rdma.timeline_id, item.wr_id)
    &&& core.timelines_spec().events_spec()@.contains_key((rdma.timeline_id, item.wr_id))
    &&& core.timelines_spec().events_spec()@[(rdma.timeline_id, item.wr_id)] == item.event@
    &&& core.ledger_spec()@.submitted.dom().contains(item.event@)
    &&& core.ledger_spec()@.submitted.map()[item.event@] == item.submitted@.value()
    &&& rdma.pending().contains_key(item.wr_id) && rdma.pending()[item.wr_id] == item.event@
    &&& posted_pull_submission(before_core, item, input)
}

proof fn lemma_pull_batch_cq<'a>(
    old_rdma: &RdmaBackend<'a>,
    rdma: &RdmaBackend<'a>,
    core: &CacheCore,
    base: u64,
    items: Seq<rdma::PostedBatchItem>,
)
    requires
        core.well_formed(),
        old_rdma.well_formed(core),
        old_rdma.qp.ready(),
        rdma::read_batch_qp_frame(&rdma.qp, &old_rdma.qp),
        rdma.cq.well_formed(),
        rdma.cq.same_queue(old_rdma.cq),
        rdma::read_batch_items(old_rdma.cq, rdma.cq, base, items),
        rdma.active_leases == old_rdma.active_leases,
        rdma.window_pool == old_rdma.window_pool,
        rdma.next_control_wr_id == old_rdma.next_control_wr_id,
        base == core.timelines_spec().next_seq_spec()[old_rdma.timeline_id as int],
        base + items.len() <= CONTROL_WR_ID_BASE,
    ensures
        rdma_remove_range(rdma.pending(), base, items.len() as int) =~= old_rdma.pending(),
        rdma_leases_valid(rdma.active_leases@, &rdma.window_pool, &rdma.qp, core),
        rdma_control_leases_match_cq(
            rdma.active_leases@,
            &rdma.window_pool,
            rdma::CqSnapshot::of(rdma.cq),
            rdma.next_control_wr_id,
        ),
        forall|s: u64|
            rdma.cq.in_flight().contains_key(s) && !((
            #[trigger] rdma.cq.in_flight()[s]).kind is DataTransfer) ==> CONTROL_WR_ID_BASE <= s
                < rdma.next_control_wr_id,
        forall|i: int|
            0 <= i < items.len() ==> {
                let item = #[trigger] items[i];
                &&& rdma.pending().contains_key(item.wr_id)
                &&& rdma.pending()[item.wr_id] == item.event@
            },
{
    hide(CacheCore::well_formed);
    reveal(RdmaBackend::well_formed);
    reveal(rdma_window_wf);
    reveal(window_matches);
    lemma_core_timelines_wf(core);
    reveal(rdma_control_leases_match_cq);
    let before = old_rdma.cq.in_flight();
    let after = rdma.cq.in_flight();
    let remaining = rdma_remove_range(rdma.pending(), base, items.len() as int);
    lemma_rdma_leases_after_qp_state_change(
        rdma.active_leases@,
        &old_rdma.window_pool,
        &rdma.window_pool,
        &old_rdma.qp,
        &rdma.qp,
        core,
    );
    assert forall|s: u64| base <= s < base + items.len() implies after.contains_key(s) && (
    #[trigger] after[s]).kind is DataTransfer by {
        let i = s - base;
        assert(items[i].wr_id == s);
    }
    assert forall|s: u64| #[trigger]
        remaining.contains_key(s) <==> old_rdma.pending().contains_key(s) by {
        if base <= s < base + items.len() {
            if old_rdma.pending().contains_key(s) {
                assert(s < core.timelines_spec().next_seq_spec()[old_rdma.timeline_id as int]);
            }
        } else {
            assert(after.contains_key(s) == before.contains_key(s));
            if before.contains_key(s) {
                assert(after[s] == before[s]);
            }
        }
    }
    assert forall|s: u64| #[trigger] remaining.contains_key(s) implies remaining[s]
        == old_rdma.pending()[s] by {
        assert(!(base <= s < base + items.len()));
        assert(after[s] == before[s]);
    }
    assert(remaining =~= old_rdma.pending());
    assert forall|s: u64|
        after.contains_key(s) && !((
        #[trigger] after[s]).kind is DataTransfer) implies CONTROL_WR_ID_BASE <= s
        < rdma.next_control_wr_id by {
        assert(!(base <= s < base + items.len()));
        assert(before.contains_key(s) && before[s] == after[s]);
    }
    assert forall|i: int|
        0 <= i < rdma.active_leases.len() && (
        #[trigger] rdma.active_leases[i]).has_active_wr() implies {
        let l = rdma.active_leases[i];
        let w = rdma.window_pool.slot_window(l.mw_slot as int);
        &&& l.mw_slot < rdma.window_pool.slots.len()
        &&& CONTROL_WR_ID_BASE <= l.wr_id < rdma.next_control_wr_id
        &&& (l.state is Binding && after.contains_key(l.wr_id)) ==> {
            &&& after[l.wr_id].kind is BindWindow
            &&& (w matches Some(win) ==> after[l.wr_id].kind->BindWindow_window_id == win.id())
        }
        &&& ((l.state is Unbinding || l.state is Revoking) && after.contains_key(l.wr_id)) ==> {
            &&& after[l.wr_id].kind is LocalInvalidate
            &&& (w matches Some(win) ==> after[l.wr_id].kind->LocalInvalidate_window_id == win.id())
        }
    } by {
        let l = rdma.active_leases[i];
        assert(old_rdma.active_leases[i] == l);
        assert(after.contains_key(l.wr_id) == before.contains_key(l.wr_id));
    }
    assert forall|i: int| 0 <= i < items.len() implies {
        let item = #[trigger] items[i];
        &&& rdma.pending().contains_key(item.wr_id)
        &&& rdma.pending()[item.wr_id] == item.event@
    } by {}
}

/// Post all reserved Host or GPUDirect destinations through one RDMA READ doorbell.
#[verifier::rlimit(120)]
/// Facts about one posted prefix entry and the reservation it consumed.
pub(super) open spec fn posted_pull_matches_input(
    posted: InFlightRdmaPull,
    input: PreparedPull,
    rdma: &RdmaBackend<'_>,
    core: &CacheCore,
    target: PullTarget,
    wr_id: int,
) -> bool {
    &&& posted.valid(rdma, core)
    &&& posted.target == target && posted.handle == input.handle
    &&& posted.copy@.source == input.source@
    &&& posted.share_cap@.share() == input.share@.share()
    &&& posted.share_cap@.loc() == input.share@.loc()
    &&& posted.share_cap@.frontier() == (Frontier {
        last_writer: input.share@.frontier().last_writer,
        readers: input.share@.frontier().readers.insert(posted.writer@),
    })
    &&& !posted.visibility_flushed && posted.wr_id == wr_id
    &&& rdma.pending().contains_key(posted.wr_id)
    &&& rdma.pending()[posted.wr_id] == posted.writer@
}

/// Instantiate the receipt facts at an index without exposing the whole
/// batch quantifier in the caller's solver context.
#[verifier::opaque]
pub(super) open spec fn posted_pull_batch_matches(
    posted: Seq<InFlightRdmaPull>,
    inputs: Seq<PreparedPull>,
    rdma: &RdmaBackend<'_>,
    core: &CacheCore,
    target: PullTarget,
    base: u64,
) -> bool {
    forall|i: int|
        0 <= i < posted.len() ==> posted_pull_matches_input(
            #[trigger] posted[i],
            inputs[i],
            rdma,
            core,
            target,
            base + i,
        )
}

pub(super) proof fn lemma_posted_pull_batch_entry(
    posted: Seq<InFlightRdmaPull>,
    inputs: Seq<PreparedPull>,
    rdma: &RdmaBackend<'_>,
    core: &CacheCore,
    target: PullTarget,
    base: u64,
    i: int,
)
    requires
        0 <= i < posted.len(),
        posted_pull_batch_matches(posted, inputs, rdma, core, target, base),
    ensures
        posted_pull_matches_input(posted[i], inputs[i], rdma, core, target, base + i),
{
    hide(InFlightRdmaPull::valid);
    reveal(posted_pull_batch_matches);
}

pub(super) fn post_pull_read_batch<'a>(
    rdma: &mut RdmaBackend<'a>,
    core: &mut CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    target: PullTarget,
    pulls: Vec<PreparedPull>,
) -> (out: PullBatchPost)
    requires
        old(core).well_formed(),
        old(rdma).well_formed(old(core)),
        instance.id() == old(core).instance_id(),
        old(rdma).qp.ready(),
        old(rdma).qp.state() is ReadyToSend,
        old(rdma).qp.source().well_formed(),
        old(rdma).qp.effective_max_rd_atomic() > 0,
        forall|i: int|
            0 <= i < pulls.len() ==> (#[trigger] pulls[i]).valid(old(rdma), old(core), target),
    ensures
        final(core).well_formed(),
        final(rdma).well_formed(final(core)),
        final(core).all_versions_bound() == old(core).all_versions_bound(),
        final(core).frames_spec() == old(core).frames_spec(),
        final(core).next_version_seq_spec() == old(core).next_version_seq_spec(),
        old(core).citations().subset_of(final(core).citations()),
        old(core).ledger_spec()@.submitted.map().submap_of(
            final(core).ledger_spec()@.submitted.map(),
        ),
        final(core).ledger_spec()@.knows == old(core).ledger_spec()@.knows,
        final(core).ledger_spec()@.succeeded == old(core).ledger_spec()@.succeeded,
        final(core).ledger_spec()@.revoked == old(core).ledger_spec()@.revoked,
        final(core).tiers_spec() == old(core).tiers_spec(),
        final(core).node_id_spec() == old(core).node_id_spec(),
        final(core).instance_id() == old(core).instance_id(),
        peer_core_step(old(core), final(core), old(rdma).timeline_id),
        final(rdma).timeline_id == old(rdma).timeline_id,
        final(rdma).peer_node == old(rdma).peer_node,
        rdma::read_batch_qp_frame(&final(rdma).qp, &old(rdma).qp),
        final(rdma).window_pool == old(rdma).window_pool,
        final(rdma).active_leases == old(rdma).active_leases,
        final(rdma).host_mr == old(rdma).host_mr,
        final(rdma).gpu_mr == old(rdma).gpu_mr,
        final(rdma).shared_host_mr == old(rdma).shared_host_mr,
        final(rdma).shared_gpu_mr == old(rdma).shared_gpu_mr,
        final(rdma).completed_pulls == old(rdma).completed_pulls,
        out.posted.len() + out.unposted.len() == pulls.len(),
        out.error.is_none() ==> out.unposted.len() == 0,
        posted_pull_batch_matches(
            out.posted@,
            pulls@,
            final(rdma),
            final(core),
            target,
            old(core).timelines_spec().next_seq_spec()[old(rdma).timeline_id as int],
        ),
        out.unposted@ == pulls@.skip(out.posted.len() as int),
{
    hide(CacheCore::well_formed);
    hide(InFlightRdmaPull::valid);
    hide(PreparedPull::valid);
    hide(complete_receipts);
    hide(replica_valid);
    hide(GuardedReplicaCopy::retained);
    let count = pulls.len();
    proof {
        lemma_peer_core_step_refl(core, rdma.timeline_id);
        lemma_post_pull_cq_wf(rdma, core);
        lemma_core_wf_facts(core);
    }
    if count == 0 {
        assert(pulls@.skip(0) =~= pulls@);
        assert(posted_pull_batch_matches(
            Seq::empty(),
            pulls@,
            rdma,
            core,
            target,
            old(core).timelines_spec().next_seq_spec()[old(rdma).timeline_id as int],
        )) by {
            reveal(posted_pull_batch_matches);
        }
        return PullBatchPost { posted: Vec::new(), unposted: pulls, error: None };
    }
    if count > rdma.cq.capacity_exec().saturating_sub(rdma.cq.in_flight_count()) {
        assert(pulls@.skip(0) =~= pulls@);
        assert(posted_pull_batch_matches(
            Seq::empty(),
            pulls@,
            rdma,
            core,
            target,
            old(core).timelines_spec().next_seq_spec()[old(rdma).timeline_id as int],
        )) by {
            reveal(posted_pull_batch_matches);
        }
        return PullBatchPost {
            posted: Vec::new(),
            unposted: pulls,
            error: Some(CoreError::CapacityExhausted),
        };
    }
    let base = match rdma.next_data_ids(core, count) {
        Ok(s) => s,
        Err(e) => {
            assert(pulls@.skip(0) =~= pulls@);
            assert(posted_pull_batch_matches(
                Seq::empty(),
                pulls@,
                rdma,
                core,
                target,
                old(core).timelines_spec().next_seq_spec()[old(rdma).timeline_id as int],
            )) by {
                reveal(posted_pull_batch_matches);
            }
            return PullBatchPost { posted: Vec::new(), unposted: pulls, error: Some(e) };
        },
    };
    let ghost before_core = *core;
    let ghost before_rdma = *rdma;
    let ghost inputs = pulls@;
    let batch = prepare_and_post_batch_reads(rdma, core, Tracked(instance), target, base, pulls);
    let n = batch.raw.posted.len();
    let error = match batch.raw.error {
        Some(e) => Some(rdma_error(e)),
        None => None,
    };
    let ghost items = batch.raw.posted@;
    proof {
        assert forall|i: int| 0 <= i < n implies (#[trigger] items[i]).event@
            == items[i].submitted@.key() by {
            reveal(posted_pull_submission);
        }
        lemma_pull_batch_cq(&before_rdma, rdma, core, base, items);
        reveal(RdmaBackend::well_formed);
        reveal(rdma_window_wf);
        lemma_citations_closure(core);
        lemma_core_frames_separated(core);
        assert forall|j: int| 0 <= j < n implies {
            let item = #[trigger] items[j];
            &&& item.wr_id == base + j
            &&& item.submitted@.instance_id() == core.instance_id()
            &&& item.submitted@.key().observer == core.node()
            &&& forall|e: TransferEvent| #[trigger]
                item.submitted@.value().preds.contains(e)
                    ==> core.ledger_spec()@.submitted.dom().contains(e)
            &&& core.disjoint_from_published(item.submitted@.value().access.writes)
            &&& rdma.pending().contains_key(item.wr_id)
            &&& rdma.pending()[item.wr_id] == item.submitted@.key()
        } by {
            reveal(PreparedPull::valid);
            reveal(posted_pull_submission);
            let loc = inputs[j].handle.slot@.loc;
            assert(items[j].submitted@.value().access.writes =~= ISet::empty().insert(loc));
            core.lemma_non_published_write_disjoint(
                inputs[j].handle.frame,
                ISet::empty().insert(loc),
            );
        }
    }
    let tickets = rdma.record_data_batch(
        core,
        batch.frames.as_slice(),
        batch.ops.as_slice(),
        batch.raw.posted,
    );
    proof {
        assert forall|i: int| 0 <= i < n implies posted_pull_recorded(
            rdma,
            core,
            &before_core,
            base + i,
            #[trigger] tickets[i],
            items[i],
            inputs[i],
        ) by {
            reveal(posted_pull_recorded);
            assert(tickets[i].sequence() == base + i);
            assert(core.ledger_spec()@.submitted.dom().contains(items[i].submitted@.key()));
        }
    }
    collect_posted_pulls(
        rdma,
        core,
        target,
        base,
        error,
        batch.meta,
        tickets,
        Tracked(instance),
        batch.buffers,
        Ghost(before_core),
        Ghost(before_rdma),
        Ghost(inputs),
        batch.start_buffers,
        Ghost(items),
    )
}

/// Build the local descriptors and submit the READ doorbell. The CQ owns the
/// accepted prefix, whose receipts will be recorded in the core by the caller.
fn prepare_and_post_batch_reads<'a>(
    rdma: &mut RdmaBackend<'a>,
    core: &mut CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    target: PullTarget,
    base: u64,
    pulls: Vec<PreparedPull>,
) -> (batch: PostedPullBatch)
    requires
        old(core).well_formed(),
        old(rdma).well_formed(old(core)),
        instance.id() == old(core).instance_id(),
        old(rdma).qp.ready(),
        old(rdma).qp.state() is ReadyToSend,
        old(rdma).qp.source().well_formed(),
        old(rdma).qp.effective_max_rd_atomic() > 0,
        0 < pulls.len(),
        old(rdma).cq.pending().len() + pulls.len() <= old(rdma).cq.capacity(),
        base == old(core).timelines_spec().next_seq_spec()[old(rdma).timeline_id as int],
        base + pulls.len() - 1 <= MAX_TICKET_SEQ,
        forall|i: int|
            0 <= i < pulls.len() ==> (#[trigger] pulls[i]).valid(old(rdma), old(core), target),
    ensures
        *final(core) == *old(core),
        final(rdma).timeline_id == old(rdma).timeline_id,
        final(rdma).peer_node == old(rdma).peer_node,
        final(rdma).window_pool == old(rdma).window_pool,
        final(rdma).active_leases == old(rdma).active_leases,
        final(rdma).next_control_wr_id == old(rdma).next_control_wr_id,
        final(rdma).host_mr == old(rdma).host_mr,
        final(rdma).gpu_mr == old(rdma).gpu_mr,
        final(rdma).shared_host_mr == old(rdma).shared_host_mr,
        final(rdma).shared_gpu_mr == old(rdma).shared_gpu_mr,
        final(rdma).completed_pulls == old(rdma).completed_pulls,
        final(rdma).in_flight_ops == old(rdma).in_flight_ops,
        final(rdma).bitmap == old(rdma).bitmap,
        rdma::read_batch_qp_frame(&final(rdma).qp, &old(rdma).qp),
        final(rdma).cq.same_queue(old(rdma).cq),
        final(rdma).cq.well_formed(),
        rdma::read_batch_items(old(rdma).cq, final(rdma).cq, base, batch.raw.posted@),
        batch.frames.len() == batch.ops.len() == batch.meta.len() == pulls.len(),
        batch.raw.posted.len() <= pulls.len(),
        batch.raw.error.is_none() ==> batch.raw.posted.len() == pulls.len(),
        forall|i: int|
            0 <= i < batch.meta.len() ==> {
                let m = #[trigger] batch.meta[i];
                &&& m.handle == pulls[i].handle && m.source == pulls[i].source
                &&& m.remote == pulls[i].remote
            },
        forall|i: usize| i < pulls.len() ==> #[trigger] batch.start_buffers@.contains_key(i),
        batch.buffers@.dom() == batch.start_buffers@.dom(),
        forall|i: usize|
            batch.raw.posted.len() <= i < pulls.len() ==> #[trigger] batch.buffers@[i]
                == batch.start_buffers@[i],
        forall|i: int|
            0 <= i < pulls.len() ==> (#[trigger] batch.start_buffers@[i as usize]) == (
            rdma::BatchReadBuffers {
                remote: ReadPerm::Shared(pulls[i].share@),
                destination: rdma::BatchReadDestination::Owned(pulls[i].dst@),
            }),
        forall|i: int|
            0 <= i < batch.raw.posted.len() ==> posted_pull_submission(
                old(core),
                #[trigger] batch.raw.posted[i],
                pulls[i],
            ),
        forall|i: usize|
            i < batch.raw.posted.len() ==> (#[trigger] batch.buffers@[i]).completed_post(
                batch.start_buffers@[i],
                target is GpuGdr,
                instance.id(),
                pulls[i as int].handle.slot@.loc,
                batch.raw.posted[i as int].event@,
            ),
{
    hide(CacheCore::well_formed);
    hide(complete_receipts);
    hide(replica_valid);
    hide(GuardedReplicaCopy::retained);
    let count = pulls.len();
    let ghost inputs = pulls@;
    let ghost cited = core.citations();
    proof {
        lemma_post_pull_cq_wf(rdma, core);
        lemma_core_wf_facts(core);
        rdma.lemma_effective_slabs_wf(core);
        assert(inputs[0].valid(rdma, core, target));
    }
    let slab = match target {
        PullTarget::Host => match (&rdma.host_mr, rdma.shared_host_mr) {
            (Some(s), _) => s,
            (None, Some(s)) => s,
            _ => {
                assert(false);
                unreachable!()
            },
        },
        PullTarget::GpuGdr => match (&rdma.gpu_mr, rdma.shared_gpu_mr) {
            (Some(s), _) => s,
            (None, Some(s)) => s,
            _ => {
                assert(false);
                unreachable!()
            },
        },
    };
    let gpu = matches!(target, PullTarget::GpuGdr);
    let mut locals: Vec<[rdma::MemoryRegionSlice; 1]> = Vec::with_capacity(count);
    let mut frames = Vec::with_capacity(count);
    let mut ops = Vec::with_capacity(count);
    let mut meta: Vec<PullBatchMeta> = Vec::with_capacity(count);
    let tracked mut buffers = Map::<usize, rdma::BatchReadBuffers>::tracked_empty();
    let mut entries = pulls.into_iter();
    let mut i = 0;
    while i < count
        invariant
            i == locals.len() == frames.len() == ops.len() == meta.len() <= count == inputs.len(),
            vstd::std_specs::iter::IteratorSpec::remaining(&entries) == inputs.skip(i as int),
            forall|j: int| 0 <= j < count ==> (#[trigger] inputs[j]).valid(rdma, core, target),
            forall|j: int|
                0 <= j < i ==> {
                    let m = #[trigger] meta[j];
                    &&& m.handle == inputs[j].handle && m.source == inputs[j].source
                    &&& m.remote == inputs[j].remote && frames[j] == m.handle.frame
                },
            forall|j: int|
                0 <= j < i ==> {
                    let arr = #[trigger] locals[j];
                    &&& arr@[0].well_formed() && arr@[0].slot_ref() == inputs[j].handle.slot@
                    &&& arr@[0].registration() == slab.mr
                },
            forall|j: usize| j < i ==> #[trigger] buffers.contains_key(j),
            forall|j: int|
                0 <= j < i ==> (#[trigger] buffers[j as usize]) == (rdma::BatchReadBuffers {
                    remote: ReadPerm::Shared(inputs[j].share@),
                    destination: rdma::BatchReadDestination::Owned(inputs[j].dst@),
                }),
            core.well_formed(),
            rdma.well_formed(core),
            core.timelines_spec().next_seq_spec()[rdma.timeline_id as int] == base,
            base + count - 1 <= MAX_TICKET_SEQ,
            gpu == (target is GpuGdr),
            slab.well_formed(core, rdma.window_pool.pd()),
            target is Host ==> rdma.effective_host_mr_spec() == Some(*slab),
            target is GpuGdr ==> rdma.effective_gpu_mr_spec() == Some(*slab),
        decreases count - i,
    {
        let p = entries.next().unwrap();
        proof {
            assert(p == inputs[i as int]);
            lemma_slab_frame_slot(
                core,
                slab.tier,
                slab.first_frame,
                slab.num_frames,
                slab.mr.space(),
                p.handle.frame,
            );
            buffers.tracked_insert(
                i,
                rdma::BatchReadBuffers {
                    remote: ReadPerm::Shared(p.share.get()),
                    destination: rdma::BatchReadDestination::Owned(p.dst.get()),
                },
            );
        }
        let local = slab.mr.slot_slice((p.handle.frame.0 - slab.first_frame.0) as usize);
        locals.push([local]);
        frames.push(p.handle.frame);
        ops.push(DataOperation::Read);
        meta.push(PullBatchMeta { handle: p.handle, remote: p.remote, source: p.source });
        i += 1;
    }

    let mut requests = Vec::with_capacity(count);
    let mut j = 0;
    while j < count
        invariant
            j == requests.len() <= count == meta.len() == locals.len(),
            base + count - 1 <= MAX_TICKET_SEQ,
            forall|k: int|
                0 <= k < j ==> {
                    let r: rdma::TransferRequest = #[trigger] requests[k];
                    &&& r.wr_id == base + k && r.read && !r.fence
                    &&& r.locals@ == locals[k]@ && r.remote == meta[k].remote
                },
        decreases count - j,
    {
        requests.push(
            rdma::TransferRequest {
                locals: &locals[j],
                remote: meta[j].remote.descriptor_copy(),
                wr_id: base + j as u64,
                read: true,
                fence: false,
            },
        );
        j += 1;
    }
    let ghost descriptors = requests@;
    let ghost start_buffers = buffers;
    proof {
        lemma_core_timelines_wf(core);
        assert forall|j: int| 0 <= j < count implies {
            let r = #[trigger] descriptors[j];
            &&& r.wr_id == base + j
            &&& buffers.contains_key(j as usize)
            &&& buffers[j as usize].admitted(
                &rdma.qp,
                rdma.cq,
                r,
                core.ledger_spec()@.submitted,
                core.ledger_spec()@.knows,
                cited,
            )
            &&& r.locals@[0].registration().gpu_device_spec().is_some() == gpu
        } by {
            let p = inputs[j];
            let wr = (base + j) as u64;
            assert(!core.timelines_spec().issued(rdma.timeline_id, wr));
            lemma_post_pull_entry_facts(rdma, core, wr, p.handle, p.dst@, p.share@, p.source@);
            assert(rdma.qp.can_post(rdma.cq, wr));
            assert(rdma.qp.transfer_ordered(false, core.ledger_spec()@.submitted, cited)
                == closure_from(core.ledger_spec()@.submitted.map(), cited));
        }
    }
    let ghost unposted_buffers = buffers;
    let raw = {
        let tracked ledger = core.borrow_ledger_mut();
        rdma::post_send_batch_reads(
            &mut rdma.qp,
            &mut rdma.cq,
            Tracked(instance),
            requests.as_slice(),
            base,
            gpu,
            Tracked(&mut buffers),
            Tracked(&mut ledger.submitted),
            Tracked(&ledger.knows),
            Ghost(cited),
        )
    };
    proof {
        assert forall|i: usize| raw.posted.len() <= i < pulls.len() implies #[trigger] buffers[i]
            == start_buffers[i] by {
            assert(unposted_buffers.contains_key(i));
            assert(buffers[i] == unposted_buffers[i]);
        }
        assert forall|i: int| 0 <= i < raw.posted.len() implies posted_pull_submission(
            core,
            #[trigger] raw.posted[i],
            inputs[i],
        ) by {
            reveal(posted_pull_submission);
            let r = descriptors[i];
            let p = inputs[i];
            assert(r.locals@[0].slot_ref() == p.handle.slot@);
            assert(r.remote.slot_ref() == p.source@.slot);
            assert(start_buffers[i as usize].remote == ReadPerm::Shared(p.share@));
        }
        assert forall|i: usize| i < raw.posted.len() implies (#[trigger] buffers[i]).completed_post(
            start_buffers[i],
            gpu,
            instance.id(),
            inputs[i as int].handle.slot@.loc,
            raw.posted[i as int].event@,
        ) by {
            assert(descriptors[i as int].locals@[0].slot_ref() == inputs[i as int].handle.slot@);
        }
    }
    PostedPullBatch {
        frames,
        ops,
        meta,
        raw,
        buffers: Tracked(buffers),
        start_buffers: Ghost(start_buffers),
    }
}

/// Recover the submitted prefix and return the unposted suffix with its original tokens.
fn collect_posted_pulls<'a>(
    rdma: &RdmaBackend<'a>,
    core: &CacheCore,
    target: PullTarget,
    base: u64,
    error: Option<CoreError>,
    meta: Vec<PullBatchMeta>,
    tickets: Vec<Ticket>,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(buffers): Tracked<Map<usize, rdma::BatchReadBuffers>>,
    Ghost(before_core): Ghost<CacheCore>,
    Ghost(before_rdma): Ghost<RdmaBackend<'a>>,
    Ghost(inputs): Ghost<Seq<PreparedPull>>,
    Ghost(start_buffers): Ghost<Map<usize, rdma::BatchReadBuffers>>,
    Ghost(items): Ghost<Seq<rdma::PostedBatchItem>>,
) -> (out: PullBatchPost)
    requires
        meta.len() == inputs.len(),
        tickets.len() == items.len() <= meta.len(),
        error.is_none() ==> tickets.len() == meta.len(),
        core.well_formed(),
        rdma.well_formed(core),
        before_core.well_formed(),
        before_rdma.well_formed(&before_core),
        core.frames_spec() == before_core.frames_spec(),
        core.node_id_spec() == before_core.node_id_spec(),
        core.instance_id() == before_core.instance_id(),
        core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        core.ledger_spec()@.knows == before_core.ledger_spec()@.knows,
        core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
        core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
        before_core.ledger_spec()@.submitted.map().submap_of(core.ledger_spec()@.submitted.map()),
        rdma.effective_gpu_mr_spec() == before_rdma.effective_gpu_mr_spec(),
        instance.id() == core.instance_id(),
        forall|i: int|
            0 <= i < meta.len() ==> {
                let m = #[trigger] meta[i];
                &&& m.handle == inputs[i].handle && m.source == inputs[i].source
                &&& m.remote == inputs[i].remote
            },
        forall|i: int|
            0 <= i < meta.len() ==> (#[trigger] inputs[i]).valid(
                &before_rdma,
                &before_core,
                target,
            ),
        forall|i: usize| i < meta.len() ==> #[trigger] start_buffers.contains_key(i),
        forall|i: usize| i < meta.len() ==> #[trigger] buffers.contains_key(i),
        forall|i: usize|
            tickets.len() <= i < meta.len() ==> #[trigger] buffers[i] == start_buffers[i],
        forall|i: usize|
            i < tickets.len() ==> (#[trigger] buffers[i]).completed_post(
                start_buffers[i],
                target is GpuGdr,
                instance.id(),
                inputs[i as int].handle.slot@.loc,
                items[i as int].event@,
            ),
        forall|i: int|
            0 <= i < meta.len() ==> {
                let b = #[trigger] start_buffers[i as usize];
                b == (rdma::BatchReadBuffers {
                    remote: ReadPerm::Shared(inputs[i].share@),
                    destination: rdma::BatchReadDestination::Owned(inputs[i].dst@),
                })
            },
        forall|i: int|
            0 <= i < tickets.len() ==> posted_pull_recorded(
                rdma,
                core,
                &before_core,
                base + i,
                #[trigger] tickets[i],
                items[i],
                inputs[i],
            ),
    ensures
        out.posted.len() == tickets.len(),
        out.error == error,
        out.posted.len() + out.unposted.len() == inputs.len(),
        out.error.is_none() ==> out.unposted.len() == 0,
        posted_pull_batch_matches(out.posted@, inputs, rdma, core, target, base),
        out.unposted@ == inputs.skip(out.posted.len() as int),
{
    hide(CacheCore::well_formed);
    hide(RdmaBackend::well_formed);
    hide(InFlightRdmaPull::valid);
    hide(complete_receipts);
    hide(replica_valid);
    hide(GuardedReplicaCopy::retained);
    reveal(posted_pull_batch_matches);
    let count = meta.len();
    let n = tickets.len();
    let gpu = matches!(target, PullTarget::GpuGdr);
    let tracked mut buffers = buffers;
    let mut posted: Vec<InFlightRdmaPull> = Vec::with_capacity(n);
    let mut unposted = Vec::with_capacity(count - n);
    let mut remaining = meta.into_iter();
    let ghost all_meta = vstd::std_specs::iter::IteratorSpec::remaining(&remaining);
    let mut k = 0;
    while k < count
        invariant
            k <= count == inputs.len() == all_meta.len(),
            n == items.len() == tickets.len() <= count,
            posted.len() == if k < n {
                k
            } else {
                n
            },
            unposted@ == inputs.subrange(posted.len() as int, k as int),
            vstd::std_specs::iter::IteratorSpec::remaining(&remaining) == all_meta.skip(k as int),
            forall|i: int|
                0 <= i < count ==> {
                    let m = #[trigger] all_meta[i];
                    &&& m.handle == inputs[i].handle && m.source == inputs[i].source
                    &&& m.remote == inputs[i].remote
                },
            forall|i: int|
                0 <= i < count ==> (#[trigger] inputs[i]).valid(&before_rdma, &before_core, target),
            forall|i: usize| i < count ==> #[trigger] start_buffers.contains_key(i),
            forall|i: usize| k <= i < count ==> #[trigger] buffers.contains_key(i),
            forall|i: usize| n <= i < count && k <= i ==> #[trigger] buffers[i] == start_buffers[i],
            forall|i: usize|
                k <= i < n ==> {
                    let b = #[trigger] buffers[i];
                    &&& b.completed_post(
                        start_buffers[i],
                        gpu,
                        instance.id(),
                        inputs[i as int].handle.slot@.loc,
                        items[i as int].event@,
                    )
                },
            forall|i: int|
                0 <= i < count ==> {
                    let b = #[trigger] start_buffers[i as usize];
                    b == (rdma::BatchReadBuffers {
                        remote: ReadPerm::Shared(inputs[i].share@),
                        destination: rdma::BatchReadDestination::Owned(inputs[i].dst@),
                    })
                },
            forall|i: int|
                0 <= i < posted.len() ==> {
                    let p = #[trigger] posted[i];
                    &&& p.valid(rdma, core) && p.target == target && p.handle == inputs[i].handle
                    &&& p.copy@.source == inputs[i].source@ && p.share_cap@.share()
                        == inputs[i].share@.share()
                    &&& p.share_cap@.loc() == inputs[i].share@.loc()
                    &&& p.share_cap@.frontier() == (Frontier {
                        last_writer: inputs[i].share@.frontier().last_writer,
                        readers: inputs[i].share@.frontier().readers.insert(p.writer@),
                    })
                    &&& !p.visibility_flushed
                    &&& p.wr_id == base + i
                    &&& rdma.pending().contains_key(p.wr_id) && rdma.pending()[p.wr_id] == p.writer@
                },
            core.well_formed(),
            rdma.well_formed(core),
            before_core.well_formed(),
            before_rdma.well_formed(&before_core),
            core.frames_spec() == before_core.frames_spec(),
            core.node_id_spec() == before_core.node_id_spec(),
            core.instance_id() == before_core.instance_id(),
            core.next_version_seq_spec() == before_core.next_version_seq_spec(),
            core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
            core.ledger_spec()@.knows == before_core.ledger_spec()@.knows,
            core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
            core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
            before_core.ledger_spec()@.submitted.map().submap_of(
                core.ledger_spec()@.submitted.map(),
            ),
            rdma.effective_gpu_mr_spec() == before_rdma.effective_gpu_mr_spec(),
            forall|i: int|
                0 <= i < tickets.len() ==> posted_pull_recorded(
                    rdma,
                    core,
                    &before_core,
                    base + i,
                    #[trigger] tickets[i],
                    items[i],
                    inputs[i],
                ),
            gpu == (target is GpuGdr),
            instance.id() == core.instance_id(),
        decreases count - k,
    {
        let p = remaining.next().unwrap();
        let ghost before_buffers = buffers;
        assert(buffers.contains_key(k));
        let tracked b = buffers.tracked_remove(k);
        proof {
            assert(start_buffers[k as int as usize].remote == ReadPerm::Shared(
                inputs[k as int].share@,
            ));
            assert(b.remote is Shared);
        }
        let tracked share = b.remote.into_shared();
        if k < n {
            let ghost item = items[k as int];
            let tracked dst_token = match b.destination {
                rdma::BatchReadDestination::Owned(t) => PullSlotToken::Host(t),
                rdma::BatchReadDestination::PendingGdr(g) => PullSlotToken::GpuGdr(g),
            };
            let ticket = tickets[k];
            let pull = InFlightRdmaPull {
                target,
                handle: p.handle,
                ticket,
                wr_id: ticket.seq(),
                writer: Ghost(item.event@),
                copy: Ghost(
                    GuardedReplicaCopy {
                        before: before_core.ledger_spec()@,
                        source: p.source@,
                        source_slot: p.source@.slot,
                        share: share.share(),
                        writer: item.event@,
                        destination: p.handle.slot@,
                    },
                ),
                share_cap: Tracked(share),
                dst_token: Tracked(dst_token),
                visibility_flushed: false,
            };
            proof {
                reveal(posted_pull_recorded);
                reveal(posted_pull_submission);
                assert(posted_pull_recorded(
                    rdma,
                    core,
                    &before_core,
                    base + k,
                    ticket,
                    item,
                    inputs[k as int],
                ));
                lemma_core_wf_facts(&before_core);
                lemma_initializing_frame_slot_valid(&before_core, p.handle.frame);
                if gpu {
                    before_rdma.lemma_effective_slabs_wf(&before_core);
                    let slab = rdma.effective_gpu_mr_spec().unwrap();
                    lemma_slab_frame_slot(
                        &before_core,
                        slab.tier,
                        slab.first_frame,
                        slab.num_frames,
                        slab.mr.space(),
                        p.handle.frame,
                    );
                    assert(p.handle.slot@.loc.space == slab.mr.space());
                }
                assert(b.completed_post(
                    start_buffers[k],
                    gpu,
                    instance.id(),
                    p.handle.slot@.loc,
                    item.event@,
                ));
                assert(share.share() == inputs[k as int].share@.share());
                assert(item.submitted@.value().read_guards == IMap::empty().insert(
                    p.source@.slot.loc,
                    share.share(),
                ));
                lemma_post_pull_finish_valid(
                    &before_core,
                    core,
                    rdma,
                    &pull,
                    p.source@,
                    item.submitted@,
                    before_core.citations(),
                );
            }
            let ghost before_posted = posted@;
            posted.push(pull);
            proof {
                assert forall|i: int| 0 <= i < k implies (#[trigger] posted[i])
                    == before_posted[i] by {}
            }
        } else {
            let tracked dst = match b.destination {
                rdma::BatchReadDestination::Owned(t) => t,
                _ => {
                    assert(false);
                    proof_from_false()
                },
            };
            unposted.push(
                PreparedPull {
                    handle: p.handle,
                    source: p.source,
                    remote: p.remote,
                    dst: Tracked(dst),
                    share: Tracked(share),
                },
            );
        }
        assert(unposted@ =~= inputs.subrange(posted.len() as int, k + 1));
        proof {
            assert forall|i: usize| k < i < count implies #[trigger] buffers[i]
                == before_buffers[i] by {}
        }
        k += 1;
    }
    assert(unposted@ =~= inputs.skip(n as int));
    PullBatchPost { posted, unposted, error }
}

impl InFlightRdmaPull {
    pub open spec fn batch_visible(&self) -> bool {
        self.target is Host || self.visibility_flushed
    }

    pub open spec fn same_submission(&self, before: Self) -> bool {
        &&& self.target == before.target && self.handle == before.handle
        &&& self.ticket == before.ticket && self.wr_id == before.wr_id
        &&& self.writer == before.writer && self.copy == before.copy
        &&& self.share_cap == before.share_cap
    }

    pub open(crate) spec fn completion_observed(&self, core: &CacheCore) -> bool {
        core.ledger_spec()@.knows.set().contains((core.node(), self.writer@))
    }
}

#[verifier::opaque]
pub open spec fn observed_pull_batch_matches(
    after: Seq<InFlightRdmaPull>,
    before: Seq<InFlightRdmaPull>,
    rdma: &RdmaBackend<'_>,
    core: &CacheCore,
    visible: bool,
) -> bool {
    forall|i: int|
        0 <= i < before.len() ==> {
            let p = #[trigger] after[i];
            &&& p.valid(rdma, core) && p.same_submission(before[i])
            &&& (visible ==> p.batch_visible())
        }
}

pub proof fn lemma_observed_pull_batch_entry(
    after: Seq<InFlightRdmaPull>,
    before: Seq<InFlightRdmaPull>,
    rdma: &RdmaBackend<'_>,
    core: &CacheCore,
    visible: bool,
    i: int,
)
    requires
        observed_pull_batch_matches(after, before, rdma, core, visible),
        0 <= i < before.len(),
    ensures
        after[i].valid(rdma, core),
        after[i].same_submission(before[i]),
        visible ==> after[i].batch_visible(),
{
    hide(InFlightRdmaPull::valid);
    reveal(observed_pull_batch_matches);
}

/// Make every completed destination visible before handing the pulls to the
/// publication step. A failed GPU flush returns all pulls still sealed for retry.
#[verifier::rlimit(60)]
pub fn finish_pull_read_batch(
    rdma: &RdmaBackend,
    core: &CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    target: PullTarget,
    pulls: Vec<InFlightRdmaPull>,
) -> (out: (Vec<InFlightRdmaPull>, Option<CoreError>))
    requires
        core.well_formed(),
        rdma.well_formed(core),
        instance.id() == core.instance_id(),
        forall|i: int|
            0 <= i < pulls.len() ==> {
                let p = #[trigger] pulls[i];
                &&& p.valid(rdma, core) && p.target == target
                &&& p.completion_observed(core)
                &&& target is GpuGdr ==> !p.batch_visible()
            },
        forall|i: int, j: int|
            0 <= i < j < pulls.len() ==> (#[trigger] pulls[i]).handle.slot@.loc != (
            #[trigger] pulls[j]).handle.slot@.loc,
    ensures
        out.0.len() == pulls.len(),
        observed_pull_batch_matches(out.0@, pulls@, rdma, core, out.1.is_none()),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    reveal(observed_pull_batch_matches);
    if matches!(target, PullTarget::Host) || pulls.is_empty() {
        return (pulls, None);
    }
    proof {
        rdma.lemma_effective_slabs_wf(core);
        lemma_core_wf_facts(core);
        assert(pulls[0].valid(rdma, core));
        assert(pulls[0].target is GpuGdr && !pulls[0].batch_visible());
        assert(rdma.effective_gpu_mr_spec() is Some);

    }
    let slab = rdma.effective_gpu_mr().unwrap();
    let count = pulls.len();
    let ghost original = pulls@;
    let tracked mut sealed = rdma::PendingGdrSlots::empty(instance.id());
    let mut iter = pulls.into_iter();
    let mut shells: Vec<InFlightRdmaPull> = Vec::with_capacity(count);
    while shells.len() < count
        invariant
            shells.len() <= count == original.len(),
            target is GpuGdr,
            instance.id() == core.instance_id(),
            slab.well_formed(core, rdma.window_pool.pd()),
            slab.mr.gpu_device_spec() is Some,
            core.ledger_spec()@.knows.instance_id() == instance.id(),
            vstd::std_specs::iter::IteratorSpec::remaining(&iter) == original.skip(
                shells.len() as int,
            ),
            forall|i: int|
                0 <= i < count ==> {
                    let p = #[trigger] original[i];
                    &&& p.valid(rdma, core) && p.target == target
                    &&& p.completion_observed(core) && !p.batch_visible()
                    &&& p.handle.slot@.loc.space == slab.mr.space()
                },
            forall|i: int, j: int|
                0 <= i < j < count ==> (#[trigger] original[i]).handle.slot@.loc != (
                #[trigger] original[j]).handle.slot@.loc,
            sealed.well_formed(),
            sealed.instance_id() == instance.id(),
            forall|l: SlotLoc| #[trigger]
                sealed.dom().contains(l) <==> exists|i: int|
                    0 <= i < shells.len() && (#[trigger] original[i]).handle.slot@.loc == l,
            forall|i: int|
                0 <= i < shells.len() ==> {
                    let p = #[trigger] shells[i];
                    let loc = p.handle.slot@.loc;
                    &&& p.same_submission(original[i])
                    &&& sealed.dom().contains(loc)
                    &&& sealed.writers()[loc] == p.writer@
                    &&& sealed.frontiers()[loc] == (Frontier {
                        last_writer: Some(p.writer@),
                        readers: ISet::empty(),
                    })
                },
        decreases count - shells.len(),
    {
        let i = shells.len();
        let p = iter.next().unwrap();
        let tracked slot = match p.dst_token.get() {
            PullSlotToken::GpuGdr(g) => g,
            _ => {
                assert(false);
                proof_from_false()
            },
        };
        proof {
            assert(p == original[i as int]);
            assert(p.valid(rdma, core));
            assert(slot.well_formed() && slot.instance_id() == instance.id());
            if sealed.dom().contains(slot.loc()) {
                let j = choose|j: int|
                    0 <= j < i && (#[trigger] original[j]).handle.slot@.loc == slot.loc();
                assert(original[j].handle.slot@.loc != p.handle.slot@.loc);
            }
            sealed.insert(slot);
        }
        let ghost before_shells = shells@;
        shells.push(InFlightRdmaPull { dst_token: Tracked(PullSlotToken::BatchPending), ..p });
        proof {
            assert forall|l: SlotLoc| #[trigger]
                sealed.dom().contains(l) <==> exists|j: int|
                    0 <= j < shells.len() && (#[trigger] original[j]).handle.slot@.loc == l by {
                if l == p.handle.slot@.loc {
                    assert(original[i as int].handle.slot@.loc == l);
                }
            }
            assert forall|j: int| 0 <= j < i implies (#[trigger] shells[j])
                == before_shells[j] by {}
        }
    }
    proof {
        assert forall|l: SlotLoc| #[trigger] sealed.dom().contains(l) implies l.node
            == slab.mr.node() && l.space == slab.mr.space()
            && core.ledger_spec()@.knows.set().contains((slab.mr.node(), sealed.writers()[l])) by {
            let j = choose|j: int| 0 <= j < count && (#[trigger] original[j]).handle.slot@.loc == l;
            assert(shells[j].same_submission(original[j]));
        }
    }
    let (Tracked(mut tokens), Tracked(mut sealed), error) = match slab.mr.complete_gdr_slots(
        Tracked(instance),
        Tracked(sealed),
        Tracked(&core.borrow_ledger().knows),
    ) {
        Ok(t) => (t, Tracked(rdma::PendingGdrSlots::empty(instance.id())), None),
        Err((e, back)) => (
            Tracked(Map::<SlotLoc, SlotToken>::tracked_empty()),
            back,
            Some(CoreError::Backend { kind: BACKEND_CUDA, code: e.0 }),
        ),
    };
    let ghost shells_view = shells@;
    let mut rest = shells.into_iter();
    let mut out: Vec<InFlightRdmaPull> = Vec::with_capacity(count);
    while out.len() < count
        invariant
            out.len() <= count == original.len() == shells_view.len(),
            vstd::std_specs::iter::IteratorSpec::remaining(&rest) == shells_view.skip(
                out.len() as int,
            ),
            forall|i: int|
                0 <= i < count ==> (#[trigger] original[i]).valid(rdma, core)
                    && original[i].target is GpuGdr && !original[i].batch_visible(),
            forall|i: int, j: int|
                0 <= i < j < count ==> (#[trigger] original[i]).handle.slot@.loc != (
                #[trigger] original[j]).handle.slot@.loc,
            forall|j: int|
                out.len() <= j < count ==> {
                    let p = #[trigger] shells_view[j];
                    let loc = p.handle.slot@.loc;
                    &&& p.same_submission(original[j])
                    &&& if error.is_none() {
                        tokens.contains_key(loc) && slot_owned(
                            tokens[loc],
                            instance.id(),
                            loc,
                            Frontier { last_writer: Some(p.writer@), readers: ISet::empty() },
                        )
                    } else {
                        sealed.dom().contains(loc) && sealed.writers()[loc] == p.writer@
                            && sealed.frontiers()[loc] == (Frontier {
                            last_writer: Some(p.writer@),
                            readers: ISet::empty(),
                        })
                    }
                },
            sealed.well_formed(),
            sealed.instance_id() == instance.id(),
            instance.id() == core.instance_id(),
            forall|j: int|
                0 <= j < out.len() ==> {
                    let p = #[trigger] out[j];
                    &&& p.valid(rdma, core) && p.same_submission(original[j])
                    &&& error.is_none() ==> p.batch_visible()
                },
        decreases count - out.len(),
    {
        let i = out.len();
        let p = rest.next().unwrap();
        let ghost loc = p.handle.slot@.loc;
        let visible = error.is_none();
        let tracked dst_token = if visible {
            PullSlotToken::GpuVisible(tokens.tracked_remove(loc))
        } else {
            PullSlotToken::GpuGdr(sealed.remove(loc))
        };
        out.push(
            InFlightRdmaPull { dst_token: Tracked(dst_token), visibility_flushed: visible, ..p },
        );
    }
    (out, error)
}

/// Complete an in-flight RDMA READ pull after `RdmaBackend::poll_cq` reports
/// `PeerCompletion::Data(op, status)` with `op.sequence == pull.wr_id`.
///
/// For `PullTarget::GpuGdr`, unconditionally executes `mr.complete_gdr_slot`
/// before unsealing the destination `SlotToken`.
/// On `Ok(())`, calls `instance.observe_share_bound` -> `core.record_revoked_map`
/// -> `pull.copy@.certify` -> `core.finish_write` and returns `PullOutcome::Published`.
/// On `Err(code)`, calls `core.abort_write` and returns `PullOutcome::Aborted`.
pub(crate) fn finish_pull_read<'a>(
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
        old(core).ledger_spec()@.knows.set().contains((old(core).node(), pull.writer@)),
        status.is_ok() ==> {
            &&& old(core).citations().contains(pull.writer@)
            &&& event_succeeded(old(core).ledger_spec()@, old(core).node(), pull.writer@)
            &&& (!old(core).frames_spec().versions_spec()@.contains_key(pull.handle.version) || old(
                core,
            ).frames_spec().versions_spec()@[pull.handle.version] == pull.copy@.source.bytes)
            &&& pull.handle.version.well_formed()
            &&& pull.handle.version != VersionId::NONE
        },
    ensures
        final(core).well_formed(),
        rdma.well_formed(final(core)),
        final(core).node_id_spec() == old(core).node_id_spec(),
        final(core).instance_id() == old(core).instance_id(),
        final(core).tiers_spec() == old(core).tiers_spec(),
        final(core).timelines_spec() == old(core).timelines_spec(),
        final(core).frames_spec().same_layout(&old(core).frames_spec()),
        final(core).frames_spec().preserves_fenced_leases(&old(core).frames_spec()),
        final(core).version_registry_spec() == old(core).version_registry_spec(),
        peer_core_step_except(old(core), final(core), rdma.timeline_id, Some(pull.handle.frame)),
        old(core).ledger_spec()@.submitted.map().submap_of(
            final(core).ledger_spec()@.submitted.map(),
        ),
        old(core).citations().subset_of(final(core).citations()),
        match r {
            Err((_, ret_pull)) => *final(core) == *old(core) && ret_pull.valid(rdma, final(core)),
            Ok(PullOutcome::Published { frame, version, share_cap }) => {
                &&& status.is_ok()
                &&& frame == pull.handle.frame
                &&& version == pull.handle.version
                &&& final(core).frames_spec().meta(frame).state == STATE_PUBLISHED
                &&& final(core).frames_spec().frame(frame).bound_version == Some(version)
                &&& final(core).frames_spec().versions_spec()@ == old(
                    core,
                ).frames_spec().versions_spec()@.insert(version, pull.copy@.source.bytes)
                &&& final(core).frames_spec().versions_spec()@.contains_key(version)
                &&& final(core).frames_spec().versions_spec()@[version] == pull.copy@.source.bytes
                &&& share_cap@.well_formed()
                &&& share_cap@.is_fenced()
                &&& share_cap@.instance_id() == pull.share_cap@.instance_id()
                &&& share_cap@.loc() == pull.share_cap@.loc()
                &&& share_cap@.share() == pull.share_cap@.share()
                &&& share_cap@.frontier() == pull.share_cap@.frontier()
            },
            Ok(PullOutcome::Aborted { status: err_code, share_cap }) => {
                &&& status == Err(err_code)
                &&& final(core).frames_spec().same_directory(&old(core).frames_spec())
                &&& share_cap@ == pull.share_cap@
            },
        },
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    let InFlightRdmaPull {
        target,
        handle,
        ticket,
        wr_id,
        writer,
        copy,
        share_cap,
        dst_token,
        visibility_flushed,
    } = pull;
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
        PullTarget::GpuGdr if visibility_flushed => {
            let tracked t = match dst_token.get() {
                PullSlotToken::GpuVisible(t) => t,
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
                Tracked(&core.borrow_ledger().knows),
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
                            CoreError::Backend { kind: BACKEND_CUDA, code: e.0 },
                            InFlightRdmaPull {
                                target,
                                handle,
                                ticket,
                                wr_id,
                                writer,
                                copy,
                                share_cap,
                                dst_token: Tracked(PullSlotToken::GpuGdr(pending_back.get())),
                                visibility_flushed: false,
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
                    &core.borrow_ledger().submitted,
                );
            }
            core.record_revoked_map(Tracked(revoked_map));
            let ghost cert;
            proof {
                lemma_core_wf_facts(core);
                lemma_guarded_copy_revoked_growth(
                    copy@,
                    before_rev.ledger_spec()@,
                    core.ledger_spec()@,
                    core.node(),
                );
                lemma_pull_dst_latest(
                    copy@.writer,
                    copy@.destination.loc,
                    &core.borrow_ledger().submitted,
                    &mut tok,
                    instance,
                );
                lemma_lift_share_bound(
                    core.ledger_spec()@.submitted,
                    revoked_map,
                    core.ledger_spec()@.revoked,
                    copy@.source_slot.loc,
                    copy@.share,
                    copy@.source.writer,
                );
                cert = copy@.certify(core.ledger_spec()@, core.node());
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
                    &core.timelines_spec(),
                    rdma.timeline_id,
                    &before_rev.ledger_spec()@,
                    &core.ledger_spec()@,
                    rdma::CqSnapshot::of(rdma.cq),
                    core.instance_id(),
                    rdma.next_control_wr_id,
                );
                lemma_completed_pulls_after_step(
                    rdma.completed_pulls,
                    &before_rev.timelines_spec(),
                    &core.timelines_spec(),
                    rdma.timeline_id,
                    rdma.cq.observer(),
                    &before_rev.ledger_spec()@,
                    &core.ledger_spec()@,
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
                reveal(core_transition);
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
                    &before_abort.timelines_spec(),
                    &core.timelines_spec(),
                    rdma.timeline_id,
                    rdma.cq.observer(),
                    &before_abort.ledger_spec()@,
                    &core.ledger_spec()@,
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
                    assert(core.frames_spec().frame(f).slot == before_abort.frames_spec().frame(
                        f,
                    ).slot);
                    assert(core.frames_spec().published(f) == before_abort.frames_spec().published(
                        f,
                    ));
                    assert(core.frames_spec().frame(f).fenced_shares
                        == before_abort.frames_spec().frame(f).fenced_shares);
                    assert(core.frames_spec().meta(f).active_fenced_leases
                        == before_abort.frames_spec().meta(f).active_fenced_leases);
                }
                reveal(peer_core_step_except);
                reveal(core_transition);
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
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.instance_id() == before_core.instance_id(),
        after_core.tiers_spec() == before_core.tiers_spec(),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
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
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.instance_id() == before_core.instance_id(),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        touched_frame.0 < before_core.frames_spec().metas_spec().len(),
        before_core.frames_spec().meta(touched_frame).state == STATE_INITIALIZING,
        forall|g: FrameId|
            g.0 < before_core.frames_spec().metas_spec().len() && g != touched_frame ==> {
                &&& (#[trigger] after_core.frames_spec().frame(g))
                    == before_core.frames_spec().frame(g)
                &&& after_core.frames_spec().meta(g).active_fenced_leases
                    == before_core.frames_spec().meta(g).active_fenced_leases
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
            assert(after_core.frames_spec().frame(f).slot == before_core.frames_spec().frame(
                f,
            ).slot) by {
                reveal(FrameTable::same_layout);
            }
        } else {
            assert(f != touched_frame) by {
                if f == touched_frame {
                    assert(before_core.frames_spec().valid_frame(f, before_core.kernel_view()));
                    assert(before_core.frames_spec().meta(f).active_fenced_leases > 0);
                }
            }
            assert(after_core.frames_spec().frame(f) == before_core.frames_spec().frame(f));
            assert(after_core.frames_spec().meta(f).active_fenced_leases
                == before_core.frames_spec().meta(f).active_fenced_leases);
        }
    }
}

} // verus!
