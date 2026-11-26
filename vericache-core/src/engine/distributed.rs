//! Verified composable 4-tier N-peer distributed engine (`DistributedEngine<'a, T, D>`),
//! unifying `CacheCore`, `HostBackend`, `DeviceAttachment<'a, D>` (`None`, `Borrowed`, or
//! `Owned` with async `pending_device_bindings`), optional `NvmeBackend` (`io_uring`
//! `O_DIRECT`), and `peers: Vec<PeerSession<'a, T>>` with shared or per-peer MRs,
//! full-duplex single-QP CQ routing, cooperative-first MW revocation, and Option-A
//! QP error/reset/reconnect recovery.
use super::nvme_storage::nvme_tier;
#[cfg(verus_only)]
use super::nvme_storage::{
    lemma_preserve_nvme, nvme_compatible, nvme_pending_untouched, nvme_tier_spec,
    nvme_timeline_distinct, nvme_untouched, nvme_well_formed,
};
use super::{is_caller_device_timeline, GpuLease, HostLease, NvmeStorage};
#[cfg(verus_only)]
use crate::backends::device::device_puts_untouched;
#[cfg(verus_only)]
use crate::backends::host::lemma_core_transition_preserves_host;
use crate::backends::host::HostBackend;
#[cfg(all(feature = "nvme", verus_only))]
use crate::backends::nvme::NvmeBackend;
#[cfg(feature = "nvme")]
use crate::backends::nvme::NvmeProgress;
use crate::backends::rdma::peer::{
    FetchBatchPulls, FetchOutcome, PeerMessage, PeerProgress, PeerSession, PendingInboundGrant,
    ServeOutcome,
};
use crate::backends::rdma::protocol::{ControlPayload, ReadGrantMeta, ReadReleaseMeta};
use crate::backends::rdma::pull::PullTarget;
use crate::backends::rdma::rdma_error;
#[cfg(verus_only)]
use crate::backends::rdma::{
    leases_free_of_share, lemma_core_transition_preserves_peer_session, lemma_foreign_shares_trans,
    lemma_peer_core_step_trans_retire_new_init, lemma_peer_leases_disjoint_sym,
    lemma_peer_step_footprint_with_pins, lemma_peer_step_preserves_sibling_session, peer_core_step,
    peer_core_step_except, peer_foreign_shares_preserved, peer_leases_disjoint,
    peer_step_with_pins,
};
use crate::backends::{DeviceAttachment, DeviceBackend, DeviceSlice, DeviceStream};
#[cfg(verus_only)]
use crate::frame::*;
use crate::kernel::*;
use crate::provenance::ValueCertificate;
#[cfg(verus_only)]
use crate::provenance::{complete_receipts, replica_valid, revoked_submap, source_slice_valid};
use crate::spec::*;
#[cfg(feature = "nvme")]
use crate::timeline::BITMAP_CAPACITY;
#[cfg(verus_only)]
use foundation::tracked::{knowledge_of, Frontier};
use transfer::context::*;
use transfer::domain::TransferEvent;
#[cfg(verus_only)]
use transfer::domain::{EventSource, NodeId};
#[cfg(all(feature = "nvme", verus_only))]
use transfer::nvme::DIRECT_IO_ALIGNMENT;
use transfer::rdma;
#[cfg(verus_only)]
use transfer::transport::GrantKind;
use transfer::transport::{ByteTransport, ChannelError, GrantTag, Statement};
#[cfg(verus_only)]
use transfer_core::memory::RdmaRegisterableSlab;
#[cfg(verus_only)]
use vericache_kernel::compose::*;
use vericache_kernel::spi::*;
use vstd::prelude::*;

#[cfg(feature = "cuda")]
pub use super::CudaAttachment;
pub use super::DistributedEngine;

verus! {

/// Exclusive reservation for a direct write into an initializing Host DRAM frame.
#[must_use = "commit or abort the reservation on its issuing engine after draining all writes"]
pub struct HostWriteReservation {
    pub(crate) core_id: Option<u64>,
    pub(crate) handle: WriteHandle,
    pub(crate) token: Tracked<SlotToken>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(crate) version_name: Tracked<VersionUnused>,
    pub(crate) host_addr: usize,
    pub(crate) frame_size: usize,
}

impl HostWriteReservation {
    pub open(crate) spec fn handle_spec(&self) -> WriteHandle {
        self.handle
    }

    pub open(crate) spec fn host_addr_spec(&self) -> usize {
        self.host_addr
    }

    pub open(crate) spec fn frame_size_spec(&self) -> usize {
        self.frame_size
    }

    #[inline]
    pub fn frame(&self) -> (r: FrameId)
        ensures
            r == self.handle_spec().frame,
    {
        self.handle.frame
    }

    /// Address of the reserved frame, pinned when the host slab is registered with a device.
    #[inline]
    pub fn host_addr(&self) -> (r: usize)
        ensures
            r == self.host_addr_spec(),
    {
        self.host_addr
    }

    #[inline]
    pub fn frame_size(&self) -> (r: usize)
        ensures
            r == self.frame_size_spec(),
    {
        self.frame_size
    }

    pub open(crate) spec fn issued_by<'a, T: ByteTransport, D: DeviceBackend<'a>>(
        &self,
        engine: &DistributedEngine<'a, T, D>,
    ) -> bool {
        self.core_id == engine.core.runtime_id()
    }

    pub open(crate) spec fn valid_for<'a, T: ByteTransport, D: DeviceBackend<'a>>(
        &self,
        engine: &DistributedEngine<'a, T, D>,
    ) -> bool {
        let c = engine.core;
        let f = self.handle.frame;
        let v = self.handle.version;
        &&& self.issued_by(engine)
        &&& engine.host.owns(f)
        &&& self.handle.tier == engine.host.tier
        &&& f.0 < c.frames_spec().metas_spec().len()
        &&& c.frames_spec().meta(f).state == STATE_INITIALIZING
        &&& !c.frames_spec().meta(f).queued()
        &&& c.frames_spec().frame(f).token is None
        &&& c.frames_spec().frame(f).slot == self.handle.slot@
        &&& c.frames_spec().frame(f).tier == self.handle.tier
        &&& c.frames_spec().frame(f).bound_key == Some(self.handle.key)
        &&& c.frames_spec().frame(f).bound_version == Some(v)
        &&& v.origin_node == c.node_id_spec() && v.seq < c.next_version_seq_spec()
        &&& self.version_name@.instance_id() == cluster_version_instance_id()
        &&& self.version_name@.element() == (v.origin_node as NodeId, v.seq)
        &&& slot_is_owned(self.token@, c.instance_id(), self.handle.slot@.loc)
        &&& write_admitted(slot_frontier(self.token@), c.citations())
        &&& self.host_addr == engine.host.buffer.address() + (f.0 - engine.host.first_frame.0)
            * engine.host.buffer.slot_len()
        &&& self.frame_size == c.frames_spec().tier_frame_size(self.handle.tier)
    }
}

/// Pending `READ_GRANT` received from a remote donor, ready to execute a
/// local Host or GPUDirect RDMA pull via `DistributedEngine::pull_with_grant_from`
pub struct PendingReadGrant {
    pub(crate) core_id: Option<u64>,
    pub(crate) peer_timeline: TimelineId,
    pub(crate) connection_epoch: u64,
    pub(crate) tag: GrantTag,
    pub(crate) meta: ReadGrantMeta,
    pub(crate) cap: rdma::RemoteCapability,
    pub(crate) share: Tracked<ShareCap>,
    pub(crate) statement: Tracked<Statement>,
    pub(crate) cert: Ghost<ValueCertificate>,
    pub(crate) revoked: Tracked<RevokedMap>,
    pub(crate) version_binding: Tracked<Option<VersionBinding>>,
}

impl PendingReadGrant {
    pub fn tag(&self) -> (tag: GrantTag)
        ensures
            tag == self.tag_spec(),
    {
        self.tag
    }

    pub fn meta(&self) -> (meta: ReadGrantMeta)
        ensures
            meta == self.meta_spec(),
    {
        self.meta
    }

    pub open(crate) spec fn tag_spec(&self) -> GrantTag {
        self.tag
    }

    pub open(crate) spec fn meta_spec(&self) -> ReadGrantMeta {
        self.meta
    }

    pub(crate) fn matches_connection<T: ByteTransport>(&self, peer: &PeerSession<'_, T>) -> bool {
        self.core_id == peer.rdma.core_id && self.peer_timeline == peer.rdma.timeline_id
            && self.connection_epoch == peer.reconnect_epoch
    }

    pub(crate) fn into_pending(self) -> (grant: PendingInboundGrant)
        ensures
            grant.tag == self.tag,
            grant.meta == self.meta,
            grant.cap == self.cap,
            grant.share == self.share,
            grant.statement == self.statement,
            grant.cert == self.cert,
            grant.revoked == self.revoked,
            grant.version_binding == self.version_binding,
    {
        PendingInboundGrant {
            tag: self.tag,
            meta: self.meta,
            cap: self.cap,
            share: self.share,
            statement: self.statement,
            cert: self.cert,
            revoked: self.revoked,
            version_binding: self.version_binding,
        }
    }

    pub open(crate) spec fn valid_for_peer<'a, T: ByteTransport, D: DeviceBackend<'a>>(
        &self,
        engine: &DistributedEngine<'a, T, D>,
        peer_idx: int,
    ) -> bool {
        &&& 0 <= peer_idx < engine.peers@.len()
        &&& engine.peers@[peer_idx].channel.received().contains_key(self.tag)
        &&& engine.peers@[peer_idx].channel.received()[self.tag] == (GrantKind::ReadShare {
            slot: self.cap.slot_ref(),
            share_id: self.share@.share(),
        })
        &&& self.statement@.valid(engine.core.instance_id(), engine.core.node())
        &&& self.statement@.knows.set() == knowledge_of(
            engine.core.node(),
            self.statement@.closure(),
        )
        &&& self.cap.well_formed()
        &&& self.cap.slot_ref().loc.node == engine.peers@[peer_idx].rdma.peer_node
        &&& self.cap.access() == rdma::RemoteAccess::ReadOnly
        &&& self.cap.share() == Some(self.share@.share())
        &&& self.cap.slot_ref() == self.cert@.slot
        &&& self.cert@.slot.is_full()
        &&& self.cert@.bytes.len() == self.meta.frame_size
        &&& self.share@.is_fenced()
        &&& self.share@.instance_id() == engine.core.instance_id()
        &&& self.share@.loc() == self.cap.slot_ref().loc
        &&& self.share@.frontier().last_writer == Some(self.cert@.writer)
        &&& transfer::transport::covers(self.share@.frontier(), self.statement@.closure())
        &&& self.revoked@.instance_id() == engine.core.instance_id()
        &&& crate::provenance::complete_receipts_map(self.statement@.submitted)
        &&& crate::backends::rdma::protocol::statement_replica_valid(
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

/// High-level event returned by `DistributedEngine::poll_peer_event_from`.
pub enum DistributedEvent {
    Control(ControlPayload),
    ReadGrant(PendingReadGrant),
    /// Peer cooperatively released a grant; if the grant was still active,
    /// its remote lease has been unpinned and its Type-2 MW invalidated.
    ReadReleased { tag: GrantTag, meta: ReadReleaseMeta, was_active: bool },
    Unexpected,
}

pub open(crate) spec fn peer_session_compatible<'a, T: ByteTransport, D: DeviceBackend<'a>>(
    core: &CacheCore,
    host: &HostBackend,
    device: &DeviceAttachment<'a, D>,
    nvme: &NvmeStorage,
    p: &PeerSession<'a, T>,
) -> bool {
    &&& p.well_formed(core)
    &&& p.rdma.timeline_id != host.timeline
    &&& p.rdma.peer_node != core.node()
    &&& p.rdma.qp.peer() == p.rdma.peer_node
    &&& p.rdma.qp.pd_id() == p.rdma.window_pool.pd()
    &&& p.rdma.qp.source().well_formed()
    &&& p.rdma.qp.source().source().is_rdma_qp()
    &&& p.rdma.qp.source().node() != p.rdma.peer_node
    &&& (device.spec_ref() matches Some(c) ==> c.timeline_id_spec() != p.rdma.timeline_id)
    &&& nvme_timeline_distinct(nvme, p.rdma.timeline_id)
}

pub open(crate) spec fn peers_pairwise_disjoint<'a, T: ByteTransport>(
    peers: Seq<PeerSession<'a, T>>,
) -> bool {
    forall|p: int, q: int|
        0 <= p < q < peers.len() ==> {
            let a = #[trigger] peers[p];
            let b = #[trigger] peers[q];
            &&& a.rdma.timeline_id != b.rdma.timeline_id
            &&& a.rdma.peer_node != b.rdma.peer_node
            &&& peer_leases_disjoint(a.rdma.active_leases@, b.rdma.active_leases@)
        }
}

impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    pub open(crate) spec fn pending_device_bindings_spec(&self) -> Tracked<
        Map<VersionId, VersionBinding>,
    > {
        self.pending_device_bindings
    }

    pub open(crate) spec fn host_ctx_spec(&self) -> SyncContext {
        self.host_ctx
    }

    pub open(crate) spec fn core_spec(&self) -> CacheCore {
        self.core
    }

    pub open(crate) spec fn peers_spec(&self) -> Vec<PeerSession<'a, T>> {
        self.peers
    }

    pub open(crate) spec fn device_spec(&self) -> DeviceAttachment<'a, D> {
        self.device
    }

    pub open(crate) spec fn nvme_spec(&self) -> NvmeStorage {
        self.nvme
    }

    pub open(crate) spec fn host_spec(&self) -> &'a HostBackend {
        self.host
    }

    pub open(crate) spec fn device_attachment_valid(
        core: &CacheCore,
        host: &HostBackend,
        attachment: &DeviceAttachment<'a, D>,
        bindings: Map<VersionId, VersionBinding>,
    ) -> bool {
        &&& match *attachment {
            DeviceAttachment::None => true,
            DeviceAttachment::Borrowed(c) => c.active_put_count_spec() == 0,
            DeviceAttachment::Owned(c) => c.pending_put_bindings_covered(bindings),
        }
        &&& attachment.spec_ref() matches Some(c) ==> {
            &&& c.well_formed(core)
            &&& c.timeline_id_spec() != host.timeline
            &&& c.pinned_host_layout_spec() matches Some((tier, first, count)) ==> tier == host.tier
                && first == host.first_frame && count == host.num_frames
        }
    }

    pub open(crate) spec fn well_formed(&self) -> bool {
        &&& self.core.well_formed()
        &&& self.core.all_versions_bound()
        &&& self.host.well_formed(&self.core)
        &&& self.host_ctx.ready(self.core.instance_id(), self.core.node())
        &&& source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger_spec()@)
        &&& self.host_ctx.source@.source() != self.host.ctx.source@.source()
        &&& Self::device_attachment_valid(
            &self.core,
            self.host,
            &self.device,
            self.pending_device_bindings@,
        )
        &&& nvme_compatible(&self.nvme, &self.core, self.host, self.device.spec_ref())
        &&& forall|i: int|
            #![trigger self.peers@[i]]
            0 <= i < self.peers@.len() ==> peer_session_compatible(
                &self.core,
                self.host,
                &self.device,
                &self.nvme,
                &self.peers@[i],
            )
        &&& peers_pairwise_disjoint(self.peers@)
    }

    /// Assemble an N-peer `DistributedEngine` supporting shared MRs (`attach_shared_host_slab` /
    /// `attach_shared_gpu_slab`) and `DeviceAttachment::{None, Borrowed, Owned}`.
    pub fn new_multi_peer(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: SyncContext,
        device: DeviceAttachment<'a, D>,
        nvme: NvmeStorage,
        peers: Vec<PeerSession<'a, T>>,
    ) -> (r: BackendResult<Self>)
        requires
            core.well_formed(),
            core.all_versions_bound(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger_spec()@),
            host_ctx.source@.source() != host.ctx_spec().source@.source(),
            device.spec_ref() matches Some(c) ==> {
                &&& c.well_formed(&core)
                &&& c.active_put_count_spec() == 0
                &&& c.timeline_id_spec() != host.timeline_spec()
                &&& c.pinned_host_layout_spec() matches Some((tier, first, count)) ==> tier
                    == host.tier_spec() && first == host.first_frame_spec() && count
                    == host.num_frames_spec()
            },
            nvme_compatible(&nvme, &core, host, device.spec_ref()),
            forall|i: int|
                #![trigger peers@[i]]
                0 <= i < peers@.len() ==> peer_session_compatible(
                    &core,
                    host,
                    &device,
                    &nvme,
                    &peers@[i],
                ),
            peers_pairwise_disjoint(peers@),
        ensures
            r matches Ok(engine) ==> {
                &&& engine.well_formed()
                &&& engine.core_spec() == core
                &&& engine.host_spec() == host
                &&& engine.device_spec() == device
                &&& engine.nvme_spec() == nvme
                &&& engine.peers_spec() == peers
            },
    {
        let tracked bindings = Map::tracked_empty();
        proof {
            if let DeviceAttachment::Owned(ref c) = device {
                c.lemma_pending_put_model();
                c.lemma_pending_put_bindings(bindings);
            }
        }
        let engine = Self {
            core,
            host,
            host_ctx,
            device,
            pending_device_bindings: Tracked(bindings),
            nvme,
            peers,
            caller_device_streams: Vec::new(),
        };
        if !engine.valid_configuration() {
            return Err(CoreError::InvalidArgument);
        }
        Ok(engine)
    }

    /// Assemble an N-peer `DistributedEngine` taking ownership of a device backend
    /// for async `submit_put_gpu_async` + `poll_gpu_completions`.
    pub fn new_with_owned_cuda(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: SyncContext,
        device: D,
        nvme: NvmeStorage,
        peers: Vec<PeerSession<'a, T>>,
    ) -> (r: BackendResult<Self>)
        requires
            core.well_formed(),
            core.all_versions_bound(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger_spec()@),
            host_ctx.source@.source() != host.ctx_spec().source@.source(),
            device.well_formed(&core),
            device.active_put_count_spec() == 0,
            device.timeline_id_spec() != host.timeline_spec(),
            device.pinned_host_layout_spec() matches Some((tier, first, count)) ==> tier
                == host.tier_spec() && first == host.first_frame_spec() && count
                == host.num_frames_spec(),
            nvme_compatible(&nvme, &core, host, Some(device)),
            forall|i: int|
                #![trigger peers@[i]]
                0 <= i < peers@.len() ==> peer_session_compatible(
                    &core,
                    host,
                    &DeviceAttachment::Owned(device),
                    &nvme,
                    &peers@[i],
                ),
            peers_pairwise_disjoint(peers@),
        ensures
            r matches Ok(engine) ==> {
                &&& engine.well_formed()
                &&& engine.core_spec() == core
                &&& engine.host_spec() == host
                &&& engine.device_spec() == DeviceAttachment::Owned(device)
                &&& engine.nvme_spec() == nvme
                &&& engine.peers_spec() == peers
            },
    {
        Self::new_multi_peer(core, host, host_ctx, DeviceAttachment::Owned(device), nvme, peers)
    }

    /// Attach an additional freshly-connected `PeerSession<'a, T>` (`active_leases@.len() == 0`)
    /// with distinct `timeline_id` and `peer_node`.
    pub fn add_peer(&mut self, peer: PeerSession<'a, T>) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            peer_session_compatible(
                &old(self).core_spec(),
                old(self).host_spec(),
                &old(self).device_spec(),
                &old(self).nvme_spec(),
                &peer,
            ),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            r.is_ok() ==> final(self).peers_spec()@ == old(self).peers_spec()@.push(peer),
            r.is_err() ==> *final(self) == *old(self),
    {
        if !self.matches_peer(&peer) {
            return Err(CoreError::InvalidArgument);
        }
        if !peer.rdma.active_leases.is_empty() {
            return Err(CoreError::WouldConflict);
        }
        let mut i = 0;
        while i < self.peers.len()
            invariant
                *self == *old(self),
                self.well_formed(),
                i <= self.peers@.len(),
                forall|j: int|
                    0 <= j < i ==> {
                        let existing = #[trigger] self.peers@[j];
                        &&& existing.rdma.timeline_id != peer.rdma.timeline_id
                        &&& existing.rdma.peer_node != peer.rdma.peer_node
                    },
            decreases self.peers@.len() - i,
        {
            let existing = &self.peers[i];
            if existing.rdma.timeline_id == peer.rdma.timeline_id || existing.rdma.peer_node
                == peer.rdma.peer_node {
                return Err(CoreError::WouldConflict);
            }
            i += 1;
        }
        let ghost before_peers = self.peers@;
        self.peers.push(peer);
        proof {
            assert forall|p: int, q: int| 0 <= p < q < self.peers@.len() implies {
                let a = #[trigger] self.peers@[p];
                let b = #[trigger] self.peers@[q];
                &&& a.rdma.timeline_id != b.rdma.timeline_id
                &&& a.rdma.peer_node != b.rdma.peer_node
                &&& peer_leases_disjoint(a.rdma.active_leases@, b.rdma.active_leases@)
            } by {
                if q < before_peers.len() {
                    assert(self.peers@[p] == before_peers[p]);
                    assert(self.peers@[q] == before_peers[q]);
                } else {
                    assert(self.peers@[p] == before_peers[p]);
                    assert(self.peers@[q].rdma.active_leases@.len() == 0);
                }
            }
        }
        Ok(())
    }

    /// Detach a peer after its remote leases have been drained.
    /// Peers after the removed index shift down by one. Timeline IDs stay registered.
    pub fn remove_peer(&mut self, peer_idx: usize) -> (r: Result<PeerSession<'a, T>, CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            peers_pairwise_disjoint(final(self).peers_spec()@),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).host_ctx_spec() == old(self).host_ctx_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).pending_device_bindings_spec() == old(self).pending_device_bindings_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            match r {
                Ok(peer) => {
                    &&& peer_idx < old(self).peers_spec()@.len()
                    &&& peer == old(self).peers_spec()@[peer_idx as int]
                    &&& peer.rdma_spec().active_leases_spec()@.len() == 0
                    &&& final(self).peers_spec()@ == old(self).peers_spec()@.remove(peer_idx as int)
                    &&& peer_session_compatible(
                        &final(self).core_spec(),
                        final(self).host_spec(),
                        &final(self).device_spec(),
                        &final(self).nvme_spec(),
                        &peer,
                    )
                },
                Err(CoreError::InvalidArgument) => peer_idx >= old(self).peers_spec()@.len()
                    && *final(self) == *old(self),
                Err(CoreError::WouldConflict) => peer_idx < old(self).peers_spec()@.len()
                    && *final(self) == *old(self),
                Err(_) => false,
            },
    {
        hide(CacheCore::well_formed);
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        let peer = &self.peers[peer_idx];
        if !peer.rdma.active_leases.is_empty() || peer.rdma.cq.in_flight_count() != 0
            || !peer.pending_inbound_grants.is_empty() {
            return Err(CoreError::WouldConflict);
        }
        let ghost before_peers = self.peers@;
        let removed = self.peers.remove(peer_idx);
        proof {
            assert forall|i: int|
                #![trigger self.peers@[i]]
                0 <= i < self.peers@.len() implies peer_session_compatible(
                &self.core,
                self.host,
                &self.device,
                &self.nvme,
                &self.peers@[i],
            ) by {
                let prior = if i < peer_idx {
                    i
                } else {
                    i + 1
                };
                assert(self.peers@[i] == before_peers[prior]);
            }
            assert forall|p: int, q: int| 0 <= p < q < self.peers@.len() implies {
                let a = #[trigger] self.peers@[p];
                let b = #[trigger] self.peers@[q];
                &&& a.rdma.timeline_id != b.rdma.timeline_id
                &&& a.rdma.peer_node != b.rdma.peer_node
                &&& peer_leases_disjoint(a.rdma.active_leases@, b.rdma.active_leases@)
            } by {
                let old_p = if p < peer_idx {
                    p
                } else {
                    p + 1
                };
                let old_q = if q < peer_idx {
                    q
                } else {
                    q + 1
                };
                assert(0 <= old_p < old_q < before_peers.len());
                assert(self.peers@[p] == before_peers[old_p]);
                assert(self.peers@[q] == before_peers[old_q]);
            }
        }
        Ok(removed)
    }

    pub fn peer_count(&self) -> (n: usize)
        ensures
            n == self.peers_spec()@.len(),
    {
        self.peers.len()
    }

    pub fn find_peer_index(&self, peer_node: u64) -> (r: Option<usize>)
        ensures
            match r {
                Some(idx) => (idx as int) < self.peers_spec()@.len()
                    && self.peers_spec()@[idx as int].rdma_spec().peer_node_spec()
                    == peer_node as NodeId,
                None => forall|i: int|
                    0 <= i < self.peers_spec()@.len() ==> (
                    #[trigger] self.peers_spec()@[i]).rdma_spec().peer_node_spec()
                        != peer_node as NodeId,
            },
    {
        let mut i: usize = 0;
        while i < self.peers.len()
            invariant
                i <= self.peers@.len(),
                forall|j: int|
                    0 <= j < i ==> (#[trigger] self.peers@[j]).rdma.peer_node
                        != peer_node as NodeId,
            decreases self.peers@.len() - i,
        {
            if self.peers[i].rdma.peer_node == peer_node {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    pub fn host_tier(&self) -> (t: TierId)
        ensures
            t == self.host_spec().tier_spec(),
    {
        self.host.tier_id()
    }

    pub fn gpu_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.device_spec().spec_ref() {
                Some(c) => Some(c.device_tier_spec()),
                None => None,
            },
    {
        self.device_tier()
    }

    pub fn device_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.device_spec().spec_ref() {
                Some(c) => Some(c.device_tier_spec()),
                None => None,
            },
    {
        self.device.device_tier_id()
    }

    pub fn nvme_tier(&self) -> (t: Option<TierId>)
        ensures
            t == nvme_tier_spec(&self.nvme_spec()),
    {
        nvme_tier(&self.nvme)
    }

    pub fn gpu_frame_slice(&self, frame: FrameId) -> (s: Option<D::Slice<'_>>)
        requires
            self.well_formed(),
    {
        self.device_frame_slice(frame)
    }

    pub fn device_frame_slice(&self, frame: FrameId) -> (s: Option<D::Slice<'_>>)
        requires
            self.well_formed(),
    {
        match self.device.as_ref() {
            Some(c) => {
                proof {
                    c.lemma_device_layout(&self.core);
                }
                c.device_frame_slice(frame)
            },
            None => None,
        }
    }

    pub fn lookup_tier(&self, key: ObjectKey, tier: TierId) -> (r: Option<(FrameId, VersionId)>)
        requires
            self.well_formed(),
        ensures
            r matches Some((f, version)) ==> {
                &&& tier.valid(self.core_spec().tiers_spec().len())
                &&& f.0 < self.core_spec().frames_spec().metas_spec().len()
                &&& self.core_spec().frames_spec().meta(f).state == STATE_PUBLISHED
                &&& self.core_spec().frames_spec().frame(f).tier == tier
                &&& self.core_spec().frames_spec().frame(f).bound_key == Some(key)
                &&& self.core_spec().frames_spec().frame(f).bound_version == Some(version)
            },
    {
        self.core.lookup_tier(key, tier)
    }

    pub fn lookup_best(&self, key: ObjectKey, priority: &[TierId]) -> (r: Option<
        (FrameId, TierId, VersionId),
    >)
        requires
            self.well_formed(),
        ensures
            r matches Some((f, tier, version)) ==> {
                &&& tier.valid(self.core_spec().tiers_spec().len())
                &&& f.0 < self.core_spec().frames_spec().metas_spec().len()
                &&& self.core_spec().frames_spec().meta(f).state == STATE_PUBLISHED
                &&& self.core_spec().frames_spec().frame(f).tier == tier
                &&& self.core_spec().frames_spec().frame(f).bound_key == Some(key)
                &&& self.core_spec().frames_spec().frame(f).bound_version == Some(version)
            },
    {
        self.core.lookup_best(key, priority)
    }

    pub(super) proof fn lemma_preserve_siblings(
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
        host: &HostBackend,
        host_ctx: &SyncContext,
        device: &DeviceAttachment<'a, D>,
        nvme: &NvmeStorage,
        peers: Seq<PeerSession<'a, T>>,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            core_transition(before, after, fp),
            host.well_formed(before),
            !fp.touched_timelines.contains(host.timeline),
            fp.excludes_source(host.ctx.source@.source()),
            host_ctx.ready(before.instance_id(), before.node()),
            source_fresh_for_ledger(host_ctx.source@, before.ledger_spec()@),
            fp.excludes_source(host_ctx.source@.source()),
            fp.touched_fenced_shares.is_empty(),
            forall|i: int|
                0 <= i < peers.len() ==> {
                    &&& (#[trigger] peers[i]).well_formed(before)
                    &&& !fp.touched_timelines.contains(peers[i].rdma.timeline_id)
                },
            device.spec_ref() matches Some(c) ==> {
                &&& c.well_formed(before)
                &&& !fp.touched_timelines.contains(c.timeline_id_spec())
                &&& fp.excludes_source(c.event_source_spec())
                &&& device_puts_untouched(c.pending_puts_spec(), before, fp)
            },
            nvme_well_formed(nvme, before),
            nvme_untouched(nvme, before, fp),
        ensures
            host.well_formed(after),
            host_ctx.ready(after.instance_id(), after.node()),
            source_fresh_for_ledger(host_ctx.source@, after.ledger_spec()@),
            forall|i: int| 0 <= i < peers.len() ==> (#[trigger] peers[i]).well_formed(after),
            device.spec_ref() matches Some(c) ==> c.well_formed(after),
            nvme_well_formed(nvme, after),
    {
        hide(CacheCore::well_formed);
        lemma_core_transition_preserves_host(before, after, fp, host);
        lemma_core_transition_preserves_source(before, after, fp, host_ctx.source@);
        assert forall|i: int| 0 <= i < peers.len() implies (#[trigger] peers[i]).well_formed(
            after,
        ) by {
            lemma_core_transition_preserves_peer_session(before, after, fp, &peers[i]);
        }
        if let Some(ref c) = device.spec_ref() {
            c.preserve_across_core_transition(before, after, fp);
        }
        lemma_preserve_nvme(before, after, fp, nvme);
    }

    /// Re-establish `self.host`, `self.host_ctx`, `self.device` (both `Borrowed` and
    /// `Owned` with parked async puts), `self.nvme`, and all sibling `self.peers[j]`
    /// (`j != peer_idx`) after `self.peers[peer_idx]` steps with `peer_core_step`
    /// and `peer_foreign_shares_preserved`.
    pub(super) fn restore_after_peer_step(
        &mut self,
        peer_idx: usize,
        Ghost(before): Ghost<CacheCore>,
        Ghost(before_peers): Ghost<Seq<PeerSession<'a, T>>>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            before.well_formed(),
            old(self).core.well_formed(),
            old(self).core.all_versions_bound(),
            instance.id() == old(self).core.instance_id(),
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@.len() == before_peers.len(),
            forall|j: int|
                0 <= j < before_peers.len() && j != peer_idx as int ==> (#[trigger] old(
                    self,
                ).peers@[j]) == before_peers[j],
            forall|j: int|
                #![trigger before_peers[j]]
                0 <= j < before_peers.len() ==> peer_session_compatible(
                    &before,
                    old(self).host,
                    &old(self).device,
                    &old(self).nvme,
                    &before_peers[j],
                ),
            peers_pairwise_disjoint(before_peers),
            old(self).peers@[peer_idx as int].well_formed(&old(self).core),
            old(self).peers@[peer_idx as int].rdma.timeline_id
                == before_peers[peer_idx as int].rdma.timeline_id,
            old(self).peers@[peer_idx as int].rdma.qp.peer() == old(
                self,
            ).peers@[peer_idx as int].rdma.peer_node,
            old(self).peers@[peer_idx as int].rdma.qp.pd_id() == old(
                self,
            ).peers@[peer_idx as int].rdma.window_pool.pd(),
            old(self).peers@[peer_idx as int].rdma.qp.source().well_formed(),
            old(self).peers@[peer_idx as int].rdma.qp.source().source().is_rdma_qp(),
            peer_core_step(
                &before,
                &old(self).core,
                old(self).peers@[peer_idx as int].rdma.timeline_id,
            ),
            peer_foreign_shares_preserved(
                &before,
                &old(self).core,
                before_peers[peer_idx as int].rdma.active_leases@,
                old(self).peers@[peer_idx as int].rdma.active_leases@,
            ),
            old(self).host.well_formed(&before),
            old(self).host_ctx.ready(old(self).core.instance_id(), old(self).core.node()),
            old(self).host_ctx.source@.source() != old(self).host.ctx.source@.source(),
            Self::device_attachment_valid(
                &before,
                old(self).host,
                &old(self).device,
                old(self).pending_device_bindings@,
            ),
            nvme_compatible(&old(self).nvme, &before, old(self).host, old(self).device.spec_ref()),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).device == old(self).device,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id_spec() == old(self).core.node_id_spec(),
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers_spec() == old(self).core.tiers_spec(),
            final(self).core.spec_spec() == old(self).core.spec_spec(),
            final(self).core.frames_spec() == old(self).core.frames_spec(),
            final(self).core.timelines_spec() == old(self).core.timelines_spec(),
            final(self).core.ledger_spec() == old(self).core.ledger_spec(),
    {
        hide(CacheCore::well_formed);
        let ghost rdma_tl = self.peers@[peer_idx as int].rdma.timeline_id;
        let ghost before_leases = before_peers[peer_idx as int].rdma.active_leases@;
        let ghost after_leases = self.peers@[peer_idx as int].rdma.active_leases@;
        let ghost fp = peer_step_with_pins(rdma_tl, None, before_leases);
        proof {
            reveal(peer_core_step_except);
            reveal(core_transition);
            lemma_core_wf_facts(&self.core);
            lemma_core_wf_facts(&before);
            assert(before_peers[peer_idx as int].well_formed(&before));
            assert(self.core.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec());
            assert(self.peers@[peer_idx as int].rdma.peer_node
                == before_peers[peer_idx as int].rdma.peer_node);
            self.host.ctx.source.borrow().lemma_all_future_unsubmitted(
                instance,
                &self.core.borrow_ledger().submitted,
            );
            self.host_ctx.source.borrow().lemma_all_future_unsubmitted(
                instance,
                &self.core.borrow_ledger().submitted,
            );
            lemma_peer_step_footprint_with_pins(
                &before,
                &self.core,
                rdma_tl,
                None,
                before_leases,
                after_leases,
            );
            lemma_core_transition_preserves_host(&before, &self.core, fp, self.host);
            assert(peer_session_compatible(
                &before,
                self.host,
                &self.device,
                &self.nvme,
                &before_peers[peer_idx as int],
            ));
            lemma_preserve_nvme(&before, &self.core, fp, &self.nvme);
            assert forall|j: int| 0 <= j < self.peers@.len() && j != peer_idx as int implies {
                &&& (#[trigger] self.peers@[j]).well_formed(&self.core)
                &&& peer_leases_disjoint(self.peers@[j].rdma.active_leases@, after_leases)
                &&& peer_leases_disjoint(after_leases, self.peers@[j].rdma.active_leases@)
            } by {
                assert(self.peers@[j] == before_peers[j]);
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.device,
                    &self.nvme,
                    &before_peers[j],
                ));
                if j < peer_idx as int {
                    assert(before_peers[j].rdma.timeline_id != rdma_tl);
                    assert(peer_leases_disjoint(
                        before_peers[j].rdma.active_leases@,
                        before_leases,
                    ));
                } else {
                    assert(rdma_tl != before_peers[j].rdma.timeline_id);
                    assert(peer_leases_disjoint(
                        before_leases,
                        before_peers[j].rdma.active_leases@,
                    ));
                    lemma_peer_leases_disjoint_sym(
                        before_leases,
                        before_peers[j].rdma.active_leases@,
                    );
                }
                lemma_peer_step_preserves_sibling_session(
                    &before,
                    &self.core,
                    rdma_tl,
                    before_leases,
                    after_leases,
                    &self.peers@[j],
                );
            }
            assert forall|p: int, q: int| 0 <= p < q < self.peers@.len() implies {
                let a = #[trigger] self.peers@[p];
                let b = #[trigger] self.peers@[q];
                &&& a.rdma.timeline_id != b.rdma.timeline_id
                &&& a.rdma.peer_node != b.rdma.peer_node
                &&& peer_leases_disjoint(a.rdma.active_leases@, b.rdma.active_leases@)
            } by {
                if p == peer_idx as int {
                    assert(self.peers@[q] == before_peers[q]);
                    assert(self.peers@[q].well_formed(&self.core));
                } else if q == peer_idx as int {
                    assert(self.peers@[p] == before_peers[p]);
                    assert(self.peers@[p].well_formed(&self.core));
                } else {
                    assert(self.peers@[p] == before_peers[p]);
                    assert(self.peers@[q] == before_peers[q]);
                }
            }
            assert forall|i: int|
                #![trigger self.peers@[i]]
                0 <= i < self.peers@.len() implies peer_session_compatible(
                &self.core,
                self.host,
                &self.device,
                &self.nvme,
                &self.peers@[i],
            ) by {
                if i == peer_idx as int {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.device,
                        &self.nvme,
                        &before_peers[peer_idx as int],
                    ));
                } else {
                    assert(self.peers@[i] == before_peers[i]);
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.device,
                        &self.nvme,
                        &before_peers[i],
                    ));
                    assert(self.peers@[i].well_formed(&self.core));
                }
            }
        }
        match &self.device {
            DeviceAttachment::None => {},
            DeviceAttachment::Borrowed(c) => {
                proof {
                    c.lemma_device_layout(&before);
                }
                c.stream().lemma_source_fresh(
                    Tracked(instance),
                    Tracked(&self.core.borrow_ledger().submitted),
                );
                proof {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.device,
                        &self.nvme,
                        &before_peers[peer_idx as int],
                    ));
                    c.preserve_idle_across_core_transition(&before, &self.core, fp);
                }
            },
            DeviceAttachment::Owned(c) => {
                proof {
                    c.lemma_device_layout(&before);
                }
                c.stream().lemma_source_fresh(
                    Tracked(instance),
                    Tracked(&self.core.borrow_ledger().submitted),
                );
                proof {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.device,
                        &self.nvme,
                        &before_peers[peer_idx as int],
                    ));
                    reveal(peer_core_step_except);
                    reveal(core_transition);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                    assert(self.core.frames_spec().consistent(self.core.kernel_view())) by {
                        reveal(CacheCore::well_formed);
                    }
                    lemma_core_wf_facts(&before);
                    assert forall|f: FrameId|
                        f.0 < before.frames_spec().metas_spec().len() && (
                        #[trigger] before.frames_spec().meta(f)).state
                            == STATE_INITIALIZING implies self.core.frames_spec().ghosts_spec()@.contains_key(
                    f.0 as nat) && self.core.frames_spec().ghosts_spec()@[f.0 as nat]
                        == before.frames_spec().frame(f) by {
                        assert(f.0 < self.core.frames_spec().metas_spec().len());
                        assert(self.core.frames_spec().frame(f) == before.frames_spec().frame(f));
                    }
                    let ghost pre_core = self.core;
                    let tracked (frame_ghosts, ledger) = self.core.borrow_frame_ghosts_and_ledger();
                    c.prove_pending_destinations_preserved(&before, frame_ghosts, ledger, instance);
                    assert(self.core == pre_core);
                    c.preserve_across_core_transition_with_bindings(
                        &before,
                        &self.core,
                        self.core.borrow_version_registry(),
                        self.pending_device_bindings.borrow(),
                        fp,
                    );
                }
            },
        }
    }

    /// Reserve a ready Host DRAM frame for an external full-frame write.
    /// The address is valid until this reservation is committed or aborted.
    pub fn begin_host_write(&mut self, key: ObjectKey) -> (r: Result<
        HostWriteReservation,
        CoreError,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(self).core_spec().frames_spec().preserves_initializing(
                &old(self).core_spec().frames_spec(),
                None,
            ),
            r matches Ok(res) ==> res.valid_for(final(self)) && res.handle_spec().key == key,
            r is Err ==> *final(self) == *old(self),
    {
        hide(CacheCore::well_formed);
        proof {
            lemma_core_wf_facts(&self.core);
        }
        if self.core.timelines().next_sequences()[self.host.timeline as usize] > MAX_TICKET_SEQ {
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before = self.core;
        let (handle, token) = self.core.begin_write(self.host.tier, key)?;
        let tracked version_name;
        proof {
            // Take the binding authority now so later writes may commit in either order.
            reveal(CacheCore::well_formed);
            version_name = self.core.reserve_version_name(handle.version);
        }
        proof {
            lemma_core_wf_facts(&self.core);
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            lemma_slab_owns_tier_frame(
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
                handle.frame,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        let slice = self.host.frame_slice(handle.frame);
        Ok(
            HostWriteReservation {
                core_id: self.core.runtime_id(),
                handle,
                token,
                version_name: Tracked(version_name),
                host_addr: slice.addr(),
                frame_size: slice.len(),
            },
        )
    }

    /// Publish the bytes written directly to the reserved frame and bind the new version.
    /// The caller must fill the entire frame, synchronize all writes to it, and relinquish
    /// the address before committing. A sequence exhaustion error returns the frame to
    /// the ready queue.
    pub fn commit_host_write(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        reservation: HostWriteReservation,
    ) -> (r: Result<(VersionId, FrameId), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            reservation.issued_by(old(self)) ==> reservation.valid_for(old(self)),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().frames_spec().preserves_initializing(
                &old(self).core_spec().frames_spec(),
                Some(reservation.handle_spec().frame),
            ),
            r matches Ok((version, frame)) ==> {
                let cert = final(self).core_spec().frames_spec().frame(frame).cert.unwrap();
                &&& frame == reservation.handle_spec().frame
                &&& version == reservation.handle_spec().version
                &&& final(self).core_spec().frames_spec().meta(frame).state == STATE_PUBLISHED
                &&& final(self).core_spec().frames_spec().frame(frame).cert is Some
                &&& final(self).core_spec().has_version_binding(version)
                &&& final(self).core_spec().spec_spec()@.versions == old(
                    self,
                ).core_spec().spec_spec()@.versions.insert(version, cert.bytes)
                &&& final(self).core_spec().spec_spec()@.catalog == old(
                    self,
                ).core_spec().spec_spec()@.catalog.insert(reservation.handle_spec().key, version)
            },
            r is Err ==> final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        if reservation.core_id != self.core.runtime_id() {
            return Err(CoreError::InvalidArgument);
        }
        proof {
            lemma_core_wf_facts(&self.core);
        }
        if self.core.timelines().next_sequences()[self.host.timeline as usize] > MAX_TICKET_SEQ {
            self.abort_host_write(reservation);
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before = *self;
        let handle = reservation.handle;
        let Tracked(mut tok) = reservation.token;
        let frame = handle.frame;
        let version = handle.version;
        proof {
            Self::lemma_reserved_version_unbound(
                &before,
                version,
                self.core.borrow_version_registry().tracked_borrow(),
                self.pending_device_bindings.borrow(),
                reservation.version_name.borrow(),
            );
        }
        let (ticket, cert) = self.host.write_and_certify_host_slot_with(
            &mut self.host_ctx,
            &mut self.core,
            Tracked(instance),
            frame,
            handle.slot,
            Tracked(&mut tok),
            None,
        );
        let ghost after_write = self.core;
        self.core.finish_write(handle, Tracked(tok), ticket, cert);
        proof {
            lemma_core_wf_facts(&self.core);
            assert(self.core.version_registry_spec()@->Some_0.instance.id()
                == cluster_version_instance_id()) by {
                reveal(CacheCore::well_formed);
            }
            assert(self.core.frames_spec().versions_spec()@.contains_key(version));
        }
        let tracked binding = self.core.borrow_version_registry().tracked_borrow().instance.bind(
            (version.origin_node as NodeId, version.seq),
            cert@.bytes,
            reservation.version_name.get(),
        );
        self.core.register_version_binding(version, Tracked(binding));
        proof {
            let source = before.host_ctx.source@.source();
            let fp = CoreFootprint::finish_write(self.host.timeline, source, frame, version);
            lemma_core_wf_facts(&self.core);
            lemma_core_frames_separated(&before.core);
            lemma_slab_bound_same_layout(
                &after_write,
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
            );
            assert(core_transition(&before.core, &self.core, fp)) by {
                broadcast use {
                    vstd::map::group_map_lemmas,
                    vstd::imap::group_imap_lemmas,
                    vstd::iset::group_iset_lemmas,
                };

                reveal(core_transition);
                assert(self.core.instance_id() == before.core.instance_id());
                assert(self.core.frames_spec().same_layout(&before.core.frames_spec()));
                assert(self.core.frames_spec().preserves_fenced_leases(&before.core.frames_spec()));
                assert(self.core.frames_spec().preserves_initializing(
                    &before.core.frames_spec(),
                    Some(frame),
                ));
                assert(fenced_shares_preserved(&before.core, &self.core, fp.touched_fenced_shares));
                assert forall|t: TimelineId|
                    t < before.core.timelines_spec().count_spec() && !fp.touched_timelines.contains(
                        t,
                    ) implies (#[trigger] self.core.timelines_spec().next_seq_spec()[t as int])
                    == before.core.timelines_spec().next_seq_spec()[t as int]
                    && self.core.timelines_spec().confirmed_seq_spec()[t as int]
                    == before.core.timelines_spec().confirmed_seq_spec()[t as int] by {
                    assert(t != self.host.timeline);
                    assert(after_write.timelines_spec().next_seq_spec()[t as int]
                        == before.core.timelines_spec().next_seq_spec()[t as int]);
                }
                assert forall|t: TimelineId, s: u64|
                    t < before.core.timelines_spec().count_spec() && t != self.host.timeline && 1
                        <= s
                        < before.core.timelines_spec().next_seq_spec()[t as int] implies #[trigger] self.core.timelines_spec().events_spec()@[(
                t, s)] == before.core.timelines_spec().events_spec()@[(t, s)] by {
                    assert(after_write.timelines_spec().next_seq_spec()[t as int]
                        == before.core.timelines_spec().next_seq_spec()[t as int]);
                }
                assert(self.core.frames_spec().versions_spec()@
                    == before.core.frames_spec().versions_spec()@.insert(version, cert@.bytes));
                assert(before.core.frames_spec().versions_spec()@.submap_of(
                    self.core.frames_spec().versions_spec()@,
                ));
                assert forall|v: VersionId| #[trigger]
                    self.core.frames_spec().versions_spec()@.contains_key(v)
                        && !before.core.frames_spec().versions_spec()@.contains_key(
                        v,
                    ) implies fp.versions.allows(&before.core, v) by {
                    assert(v == version);
                }
                assert(revoked_submap(
                    before.core.ledger_spec()@.revoked,
                    self.core.ledger_spec()@.revoked,
                ));
                assert forall|f: FrameId|
                    f != frame && f.0
                        < before.core.frames_spec().metas_spec().len() implies cert@.slot.loc != (
                #[trigger] before.core.frames_spec().frame(f)).slot.loc by {
                    if f.0 < frame.0 {
                        assert(before.core.frames_spec().frame(f).slot.loc
                            != before.core.frames_spec().frame(frame).slot.loc);
                    } else {
                        assert(before.core.frames_spec().frame(frame).slot.loc
                            != before.core.frames_spec().frame(f).slot.loc);
                    }
                }
                assert forall|e: TransferEvent| #[trigger]
                    self.core.ledger_spec()@.submitted.dom().contains(e)
                        && !before.core.ledger_spec()@.submitted.dom().contains(e) implies e.source
                    == source && forall|f: FrameId|
                    f != frame && f.0 < before.core.frames_spec().metas_spec().len() && (
                    #[trigger] before.core.frames_spec().meta(f)).state == STATE_INITIALIZING
                        ==> !self.core.ledger_spec()@.submitted.map()[e].access.writes.contains(
                        before.core.frames_spec().frame(f).slot.loc,
                    ) by {
                    assert(e == cert@.writer);
                }
            }
            Self::lemma_host_transition_preserves_siblings(&before, &self.core, fp);
            assert(self.core.all_versions_bound()) by {
                assert forall|v: VersionId| #[trigger]
                    self.core.frames_spec().versions_spec()@.contains_key(
                        v,
                    ) implies self.core.has_version_binding(v) by {
                    if v != version {
                        assert(before.core.frames_spec().versions_spec()@.contains_key(v));
                    }
                }
            }
        }
        Ok((version, frame))
    }

    /// Return an unpublished host frame to the ready queue. The caller must
    /// drain any outstanding writes to the frame before aborting.
    pub fn abort_host_write(&mut self, reservation: HostWriteReservation)
        requires
            old(self).well_formed(),
            reservation.issued_by(old(self)) ==> reservation.valid_for(old(self)),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(self).core_spec().frames_spec().preserves_initializing(
                &old(self).core_spec().frames_spec(),
                Some(reservation.handle_spec().frame),
            ),
            reservation.issued_by(old(self)) ==> {
                let meta = final(self).core_spec().frames_spec().meta(
                    reservation.handle_spec().frame,
                );
                &&& meta.state == STATE_EMPTY
                &&& meta.queued()
            },
    {
        hide(CacheCore::well_formed);
        if reservation.core_id != self.core.runtime_id() {
            return;
        }
        let ghost before = *self;
        let ghost fp = CoreFootprint::finish_write(
            self.host.timeline,
            before.host_ctx.source@.source(),
            reservation.handle.frame,
            reservation.handle.version,
        );
        proof {
            Self::lemma_reserved_version_unbound(
                &before,
                reservation.handle.version,
                self.core.borrow_version_registry().tracked_borrow(),
                self.pending_device_bindings.borrow(),
                reservation.version_name.borrow(),
            );
        }
        self.core.abort_write(reservation.handle, reservation.token);
        proof {
            lemma_core_wf_facts(&self.core);
            lemma_slab_bound_same_layout(
                &before.core,
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
            );
            assert(core_transition(&before.core, &self.core, fp)) by {
                reveal(core_transition);
                reveal(FrameTable::same_directory);
            }
            Self::lemma_host_transition_preserves_siblings(&before, &self.core, fp);
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
    }

    proof fn lemma_reserved_version_unbound(
        before: &Self,
        version: VersionId,
        tracked registry: &VersionRegistry,
        tracked pending: &Map<VersionId, VersionBinding>,
        tracked name: &VersionUnused,
    )
        requires
            before.well_formed(),
            before.core.version_registry_spec()@ == Some(*registry),
            before.pending_device_bindings@ == *pending,
            name.instance_id() == cluster_version_instance_id(),
            name.element() == (version.origin_node as NodeId, version.seq),
        ensures
            !before.core.frames_spec().versions_spec()@.contains_key(version),
            before.device.spec_ref() matches Some(c) ==> (forall|i: int|
                0 <= i < c.pending_puts_spec().len() ==> !(
                #[trigger] c.pending_puts_spec()[i]).publishes_version(version)),
    {
        assert(registry.instance.id() == cluster_version_instance_id());
        if before.core.frames_spec().versions_spec()@.contains_key(version) {
            registry.instance.unused_key_differs_from_binding(
                name,
                registry.bindings.tracked_borrow(version),
            );
        }
        if pending.contains_key(version) && pending[version].instance_id() == registry.instance.id()
            && pending[version].key() == name.element() {
            registry.instance.unused_key_differs_from_binding(
                name,
                pending.tracked_borrow(version),
            );
        }
        if let Some(ref c) = before.device.spec_ref() {
            c.lemma_pending_put_model();
            c.lemma_pending_put_bindings(*pending);
            let puts = c.pending_puts_spec();
            assert forall|i: int| 0 <= i < puts.len() implies !(
            #[trigger] puts[i]).publishes_version(version) by {
                if puts[i].publishes_version(version) {
                    assert(before.device is Owned);
                    c.lemma_pending_put_valid(&before.core, i);
                    assert(puts[i].binding_covered(*pending));
                }
            }
        }
    }

    proof fn lemma_host_transition_preserves_siblings(
        before: &Self,
        after: &CacheCore,
        fp: CoreFootprint,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            core_transition(&before.core, after, fp),
            fp.touched_timelines == optional_set(Some(before.host.timeline)),
            fp.allowed_sources == Some(optional_set(Some(before.host_ctx.source@.source()))),
            fp.touched_fenced_shares.is_empty(),
            before.device.spec_ref() matches Some(c) ==> device_puts_untouched(
                c.pending_puts_spec(),
                &before.core,
                fp,
            ),
            nvme_pending_untouched(&before.nvme, &before.core, fp),
        ensures
            forall|i: int|
                0 <= i < before.peers@.len() ==> (#[trigger] before.peers@[i]).well_formed(after),
            before.device.spec_ref() matches Some(c) ==> c.well_formed(after),
            nvme_well_formed(&before.nvme, after),
    {
        hide(CacheCore::well_formed);
        assert forall|i: int| 0 <= i < before.peers@.len() implies (
        #[trigger] before.peers@[i]).well_formed(after) by {
            lemma_core_transition_preserves_peer_session(
                &before.core,
                after,
                fp,
                &before.peers@[i],
            );
        }
        if let Some(ref c) = before.device.spec_ref() {
            c.lemma_device_layout(&before.core);
            c.preserve_across_core_transition(&before.core, after, fp);
        }
        lemma_preserve_nvme(&before.core, after, fp, &before.nvme);
    }

    /// Synchronously publish `data` into Host DRAM and bind its cluster-wide
    /// `VersionBinding` so remote peers can fetch it with version-value agreement.
    pub fn put_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        data: &[u8],
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r matches Ok((ver, frame)) ==> {
                &&& final(self).host_spec().owns(frame)
                &&& final(self).core_spec().spec_spec()@.versions == old(
                    self,
                ).core_spec().spec_spec()@.versions.insert(ver, data@)
                &&& final(self).core_spec().spec_spec()@.catalog == old(
                    self,
                ).core_spec().spec_spec()@.catalog.insert(key, ver)
                &&& final(self).core_spec().frames_spec().meta(frame).state == STATE_PUBLISHED
            },
    {
        hide(CacheCore::well_formed);
        let ghost before_engine = *self;
        let ghost before = before_engine.core;
        let res = self.host.put_host_with(
            &mut self.host_ctx,
            &mut self.core,
            Tracked(instance),
            key,
            data,
        );
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            Self::lemma_host_transition_preserves_siblings(
                &before_engine,
                &self.core,
                CoreFootprint::preserve_all(
                    Some(before_engine.host.timeline),
                    Some(before_engine.host_ctx.source@.source()),
                ),
            );
        }
        match res {
            Err(e) => {
                proof {
                    assert(self.core.frames_spec().versions_spec()@
                        == before.frames_spec().versions_spec()@) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(self.core.all_versions_bound());
                }
                Err(e)
            },
            Ok((ver, frame)) => {
                let ghost before_bind = self.core;
                proof {
                    assert(before.version_registry_spec()@->Some_0.source.min_seq()
                        <= before.next_version_seq_spec() as nat) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(self.core.frames_spec().versions_spec()@.contains_key(ver)) by {
                        reveal(CacheCore::well_formed);
                    }
                }
                self.core.bind_local_version(ver);
                proof {
                    lemma_core_wf_facts(&before_bind);
                    lemma_core_wf_facts(&self.core);
                    assert(core_transition(
                        &before_bind,
                        &self.core,
                        CoreFootprint::preserve_all(None, None),
                    )) by {
                        reveal(core_transition);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                    }
                    Self::lemma_preserve_siblings(
                        &before_bind,
                        &self.core,
                        CoreFootprint::preserve_all(None, None),
                        self.host,
                        &self.host_ctx,
                        &self.device,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.all_versions_bound()) by {
                        assert forall|v: VersionId| #[trigger]
                            self.core.frames_spec().versions_spec()@.contains_key(
                                v,
                            ) implies self.core.has_version_binding(v) by {
                            if v != ver {
                                assert(before.frames_spec().versions_spec()@.contains_key(v)) by {
                                    reveal(CacheCore::well_formed);
                                }
                                assert(before.has_version_binding(v));
                                assert(before_bind.has_version_binding(v));
                            }
                        }
                    }
                }
                Ok((ver, frame))
            },
        }
    }

    /// Synchronously publish a GPU-VRAM frame from caller's device slice (`src`)
    /// using caller's idle `src_stream` / `src_timeline`, and bind its cluster-wide
    /// `VersionBinding`. Works with both `DeviceAttachment::Borrowed` and `Owned`.
    pub fn put_gpu_sync(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        src_stream: &mut D::Stream,
        src_timeline: TimelineId,
        key: ObjectKey,
        src: &D::Slice<'_>,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(self).well_formed(),
            old(self).device_spec().spec_ref() matches Some(c) && {
                &&& c.caller_stream_valid(*old(src_stream), &old(self).core_spec(), src_timeline)
                &&& src_timeline != old(self).host_spec().timeline_spec()
                &&& src_timeline != c.timeline_id_spec()
                &&& forall|i: int|
                    0 <= i < old(self).peers_spec()@.len() ==> src_timeline != (#[trigger] old(
                        self,
                    ).peers_spec()@[i]).rdma_spec().timeline_id_spec()
                &&& nvme_timeline_distinct(&old(self).nvme_spec(), src_timeline)
                &&& old(self).core_spec().timelines_spec().confirmed_seq_spec()[src_timeline as int]
                    + 1 == old(
                    self,
                ).core_spec().timelines_spec().next_seq_spec()[src_timeline as int]
                &&& old(src_stream).device_spec() == c.device_spec()
                &&& src.slot_ref().loc.space.is_gpu() && src.slot_ref().loc.space.gpu_device()
                    != c.device_spec() ==> (old(src_perm).frontier().last_writer matches Some(w)
                    ==> !(w.source.is_rdma_qp()))
            },
            instance.id() == old(self).core_spec().instance_id(),
            src.well_formed(),
            src.instance_id() == old(self).core_spec().instance_id(),
            src.slot_ref().loc.node == old(self).core_spec().node(),
            old(self).device_spec().spec_ref() matches Some(c) ==> src.slot_ref().sub_len == old(
                self,
            ).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
            old(src_perm).valid_cooperative(
                old(self).core_spec().instance_id(),
                src.slot_ref().loc,
            ),
            read_admitted(old(src_perm).frontier(), old(self).core_spec().citations()),
            replica_valid(
                old(self).core_spec().ledger_spec()@,
                old(self).core_spec().node(),
                src_cert,
            ),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(self).core_spec().frames_spec().metas_spec().len() && (#[trigger] old(
                    self,
                ).core_spec().frames_spec().meta(f)).state == STATE_EMPTY ==> src.slot_ref().loc
                    != old(self).core_spec().frames_spec().frame(f).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            final(self).device_spec().spec_ref() matches Some(c) ==> c.caller_stream_valid(
                *final(src_stream),
                &final(self).core_spec(),
                src_timeline,
            ),
            r matches Ok((ver, frame)) ==> {
                &&& final(self).device_spec().spec_ref() matches Some(c) && c.owns_device(frame)
                &&& final(self).core_spec().timelines_spec().confirmed_seq_spec()[src_timeline as int]
                    + 1
                    == final(self).core_spec().timelines_spec().next_seq_spec()[src_timeline as int]
                &&& final(self).core_spec().spec_spec()@.versions == old(
                    self,
                ).core_spec().spec_spec()@.versions.insert(ver, src_cert.bytes)
                &&& final(self).core_spec().spec_spec()@.catalog == old(
                    self,
                ).core_spec().spec_spec()@.catalog.insert(key, ver)
                &&& final(self).core_spec().frames_spec().meta(frame).state == STATE_PUBLISHED
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        if src_timeline == device.timeline_id() || !is_caller_device_timeline::<D>(
            &self.core,
            src_timeline,
            src_stream,
        ) || self.core.tier_frame_size(device.device_tier_id()) != Some(src.len()) {
            return Err(CoreError::InvalidArgument);
        }
        if !self.core.timeline_is_idle(src_timeline) {
            return Err(CoreError::WouldConflict);
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            src_timeline,
            src_stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost src_source = src_stream.source().source();
        proof {
            device.lemma_caller_stream_valid(*src_stream, &before, src_timeline);
            src_stream.lemma_source_kind();
        }
        let res = device.put_device_sync(
            &mut self.core,
            Tracked(instance),
            src_stream,
            src_timeline,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        );
        match res {
            Err(e) => {
                proof {
                    Self::lemma_preserve_siblings(
                        &before,
                        &self.core,
                        CoreFootprint::preserve_all(Some(src_timeline), Some(src_source)),
                        self.host,
                        &self.host_ctx,
                        &self.device,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.all_versions_bound());
                }
                Err(e)
            },
            Ok((ver, frame)) => {
                let ghost before_bind = self.core;
                proof {
                    assert(before.version_registry_spec()@->Some_0.source.min_seq()
                        <= before.next_version_seq_spec() as nat) by {
                        reveal(CacheCore::well_formed);
                    }
                }
                self.core.bind_local_version(ver);
                proof {
                    lemma_core_wf_facts(&before_bind);
                    lemma_core_wf_facts(&self.core);
                    let binding_fp = CoreFootprint::preserve_all(None, None);
                    assert(core_transition(&before_bind, &self.core, binding_fp)) by {
                        reveal(core_transition);
                        reveal(FrameTable::same_layout);
                    }
                    device.preserve_caller_stream(
                        &before_bind,
                        &self.core,
                        binding_fp,
                        *src_stream,
                        src_timeline,
                    );
                    assert(core_transition(
                        &before_bind,
                        &self.core,
                        CoreFootprint::preserve_all(Some(src_timeline), Some(src_source)),
                    )) by {
                        reveal(core_transition);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                    }
                    lemma_core_transition_trans_same(
                        &before,
                        &before_bind,
                        &self.core,
                        CoreFootprint::preserve_all(Some(src_timeline), Some(src_source)),
                    );
                    Self::lemma_preserve_siblings(
                        &before,
                        &self.core,
                        CoreFootprint::preserve_all(Some(src_timeline), Some(src_source)),
                        self.host,
                        &self.host_ctx,
                        &self.device,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.all_versions_bound()) by {
                        assert forall|v: VersionId| #[trigger]
                            self.core.frames_spec().versions_spec()@.contains_key(
                                v,
                            ) implies self.core.has_version_binding(v) by {
                            if v != ver {
                                assert(before.frames_spec().versions_spec()@.contains_key(v));
                                assert(before.has_version_binding(v));
                                assert(before_bind.has_version_binding(v));
                            }
                        }
                    }
                }
                Ok((ver, frame))
            },
        }
    }

    /// Asynchronously enqueue a GPU-VRAM or pinned-host write on `DeviceAttachment::Owned`,
    /// pre-binding its `VersionBinding` into `self.pending_device_bindings` so concurrent
    /// RDMA / NVMe / Host steps remain composable while the GPU copy is in flight.
    pub fn submit_put_gpu_async(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &D::Slice<'_>,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            old(self).device_spec() is Owned,
            instance.id() == old(self).core_spec().instance_id(),
            src.well_formed(),
            src.instance_id() == old(self).core_spec().instance_id(),
            src.slot_ref().loc.node == old(self).core_spec().node(),
            old(self).device_spec().spec_ref() matches Some(c) ==> {
                &&& src.slot_ref().sub_len == old(self).core_spec().frames_spec().tier_frame_size(
                    c.device_tier_spec(),
                )
                &&& src.slot_ref().loc.space.is_gpu() && src.slot_ref().loc.space.gpu_device()
                    != c.device_spec() ==> (old(src_perm).frontier().last_writer matches Some(w)
                    ==> !(w.source.is_rdma_qp()))
            },
            old(src_perm).valid_cooperative(
                old(self).core_spec().instance_id(),
                src.slot_ref().loc,
            ),
            read_admitted(old(src_perm).frontier(), old(self).core_spec().citations()),
            replica_valid(
                old(self).core_spec().ledger_spec()@,
                old(self).core_spec().node(),
                src_cert,
            ),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(self).core_spec().frames_spec().metas_spec().len() && (#[trigger] old(
                    self,
                ).core_spec().frames_spec().meta(f)).state == STATE_EMPTY ==> src.slot_ref().loc
                    != old(self).core_spec().frames_spec().frame(f).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).device_spec() is Owned,
            final(self).device_spec().same_config(&old(self).device_spec()),
            final(self).host_spec() == old(self).host_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            r matches Ok((ver, frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& final(self).device_spec().spec_ref() matches Some(c) && ticket.timeline_id()
                    == c.timeline_id_spec() && ((dst_tier == c.device_tier_spec() && c.owns_device(
                    frame,
                )) || c.owns_pinned_host(frame))
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            if let DeviceAttachment::Owned(ref device) = self.device {
                device.lemma_same_config(device);
            }
        }
        if dst_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let prev_att = self.device.take();
        let DeviceAttachment::Owned(mut device) = prev_att else {
            self.device = prev_att;
            return Err(CoreError::UnsupportedTier);
        };
        if self.core.tier_frame_size(device.device_tier_id()) != Some(src.len()) {
            self.device = DeviceAttachment::Owned(device);
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost before_device = device;
        let ghost before_bindings = self.pending_device_bindings@;
        let ghost device_tl = device.timeline_id_spec();
        let ghost fp = CoreFootprint::preserve_all(
            Some(device_tl),
            Some(device.event_source_spec()),
        );
        let res = device.put_device(
            &mut self.core,
            Tracked(instance),
            dst_tier,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        );
        if let Ok((ver, _, _, _)) = res {
            let ghost after_put = self.core;
            proof {
                lemma_core_wf_facts(&after_put);
                assert(ver.seq < before.version_seq_limit_spec()) by {
                    reveal(core_transition);
                }
                assert(before.version_registry_spec()@->Some_0.source.min_seq()
                    <= before.next_version_seq_spec() as nat) by {
                    reveal(CacheCore::well_formed);
                }
            }
            let Tracked(binding) = self.core.prebind_local_version(ver, Ghost(src_cert.bytes));
            proof {
                self.pending_device_bindings.borrow_mut().tracked_insert(ver, binding);
                let binding_fp = CoreFootprint::preserve_all(None, None);
                lemma_core_wf_facts(&after_put);
                lemma_core_wf_facts(&self.core);
                assert(core_transition(&after_put, &self.core, binding_fp)) by {
                    reveal(core_transition);
                    reveal(FrameTable::same_layout);
                }
                device.preserve_across_core_transition(&after_put, &self.core, binding_fp);
                assert(core_transition(&before, &self.core, fp)) by {
                    reveal(core_transition);
                }
            }
        }
        proof {
            before_device.lemma_pending_put_bindings(before_bindings);
            device.lemma_pending_put_bindings(self.pending_device_bindings@);
            let before_puts = before_device.pending_puts_spec();
            let puts = device.pending_puts_spec();
            assert forall|i: int| 0 <= i < puts.len() implies (#[trigger] puts[i]).binding_covered(
                self.pending_device_bindings@,
            ) by {
                if i < before_puts.len() {
                    assert(puts[i] == before_puts[i]);
                    assert(before_puts[i].binding_covered(before_bindings));
                    if let Ok((ver, _, _, _)) = res {
                        before_device.lemma_pending_put_valid(&before, i);
                        assert(!puts[i].publishes_version(ver));
                    }
                } else {
                    assert(i == before_puts.len());
                    assert(puts[i] == puts.last());
                }
            }
            before_device.lemma_device_layout(&before);
            device.lemma_same_config(&before_device);
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != device_tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &old(self).device,
                    &self.nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                fp,
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert(self.core.all_versions_bound());
        }
        self.device = DeviceAttachment::Owned(device);
        res
    }

    /// Non-blockingly poll ready device timeline events and publish/abort completed
    /// parked GPU writes on `DeviceAttachment::Owned`, registering their pre-bound
    /// `VersionBinding`s into `self.core.version_registry_spec()`.
    pub fn poll_gpu_completions(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<DevicePollReport, (CoreError, DevicePollReport)>)
        requires
            old(self).well_formed(),
            old(self).device_spec() is Owned,
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).device_spec() is Owned,
            final(self).device_spec().same_config(&old(self).device_spec()),
            final(self).host_spec() == old(self).host_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
    {
        hide(CacheCore::well_formed);
        let prev_att = self.device.take();
        let DeviceAttachment::Owned(mut device) = prev_att else {
            self.device = prev_att;
            return Err((CoreError::UnsupportedTier, DevicePollReport::empty()));
        };
        let ghost before = self.core;
        let ghost before_device = device;
        let ghost device_tl = device.timeline_id_spec();
        let ghost poll_fp = device.poll_footprint();
        let res = device.poll_completions_with_bindings(
            &mut self.core,
            Tracked(instance),
            Tracked(self.pending_device_bindings.borrow_mut()),
        );
        proof {
            before_device.lemma_pending_put_model();
            device.lemma_same_config(&before_device);
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != device_tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &old(self).device,
                    &self.nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                poll_fp,
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
        }
        self.device = DeviceAttachment::Owned(device);
        res
    }

    /// Synchronously read a published Host-DRAM object into `dst`.
    pub fn get_into_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst: &mut [u8],
    ) -> (r: BackendResult<Option<VersionId>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            r matches Ok(Some(ver)) ==> final(self).core_spec().spec_spec()@.catalog.contains_key(
                key,
            ) && final(self).core_spec().spec_spec()@.catalog[key] == ver,
    {
        hide(CacheCore::well_formed);
        let Some((frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Ok(None);
        };
        let ghost before = self.core;
        let res = self.host.read_frame_into_with(
            &mut self.host_ctx,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst,
        );
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).well_formed(&self.core) by {
                lemma_core_transition_preserves_peer_session(
                    &before,
                    &self.core,
                    CoreFootprint::preserve_all(None, Some(old(self).host_ctx.source@.source())),
                    &self.peers@[i],
                );
            }
            if let Some(ref c) = self.device.spec_ref() {
                c.lemma_device_layout(&before);
                c.preserve_across_core_transition(
                    &before,
                    &self.core,
                    CoreFootprint::preserve_all(None, Some(old(self).host_ctx.source@.source())),
                );
            }
            lemma_preserve_nvme(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, Some(old(self).host_ctx.source@.source())),
                &self.nvme,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        match res {
            Ok(ver) => Ok(Some(ver)),
            Err(e) => Err(e),
        }
    }

    /// Asynchronously read a published GPU-VRAM object into caller's device slice.
    pub fn get_into_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst_stream: &mut D::Stream,
        dst_timeline: TimelineId,
        dst: &D::Slice<'_>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<Option<(VersionId, Ticket, Ghost<TransferEvent>)>>)
        requires
            old(self).well_formed(),
            old(self).device_spec().spec_ref() matches Some(c) && {
                &&& c.caller_stream_valid(*old(dst_stream), &old(self).core_spec(), dst_timeline)
                &&& dst_timeline != old(self).host_spec().timeline_spec()
                &&& dst_timeline != c.timeline_id_spec()
                &&& forall|i: int|
                    0 <= i < old(self).peers_spec()@.len() ==> dst_timeline != (#[trigger] old(
                        self,
                    ).peers_spec()@[i]).rdma_spec().timeline_id_spec()
                &&& nvme_timeline_distinct(&old(self).nvme_spec(), dst_timeline)
                &&& dst.slot_ref().loc.space.is_gpu() ==> dst.slot_ref().loc.space.gpu_device()
                    == c.device_spec()
            },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core_spec().instance_id(),
            dst.slot_ref().loc.node == old(self).core_spec().node(),
            old(self).device_spec().spec_ref() matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
            slot_is_owned(*old(dst_tok), old(self).core_spec().instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core_spec().citations()),
            old(self).core_spec().disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core_spec().frames_spec().metas_spec().len() ==> dst.slot_ref().loc
                    != (#[trigger] old(self).core_spec().frames_spec().frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).device_spec().spec_ref() matches Some(c) ==> c.caller_stream_valid(
                *final(dst_stream),
                &final(self).core_spec(),
                dst_timeline,
            ),
            r matches Ok(Some((version, ticket, ev))) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == dst_timeline
                &&& final(self).core_spec().spec_spec()@.versions.contains_key(version)
                &&& slot_owned(
                    *final(dst_tok),
                    old(self).core_spec().instance_id(),
                    dst.slot_ref().loc,
                    Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        if dst_timeline == device.timeline_id() || !is_caller_device_timeline::<D>(
            &self.core,
            dst_timeline,
            dst_stream,
        ) || self.core.tier_frame_size(device.device_tier_id()) != Some(dst.len()) {
            return Err(CoreError::InvalidArgument);
        }
        let Some((frame, _ver)) = self.core.lookup_tier(key, device.device_tier_id()) else {
            return Ok(None);
        };
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            dst_timeline,
            dst_stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        proof {
            device.lemma_caller_stream_valid(*dst_stream, &before, dst_timeline);
            dst_stream.lemma_source_kind();
        }
        let res = device.get_into_device(
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst_stream,
            dst_timeline,
            dst,
            Tracked(dst_tok),
        );
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(dst_timeline), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound());
        }
        match res {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    /// Evict a published frame (`Published -> Draining`), preserving sibling
    /// host, device, NVMe, and RDMA invariants.
    pub fn evict_frame(&mut self, frame: FrameId) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r.is_ok() ==> final(self).core_spec().frames_spec().meta(frame).state == STATE_DRAINING,
    {
        hide(CacheCore::well_formed);
        self.core.check_published_frame(frame)?;
        let ghost before = self.core;
        let res = self.core.evict_frame(frame);
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert(core_transition(&before, &self.core, CoreFootprint::preserve_all(None, None)))
                by {
                reveal(core_transition);
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(CacheCore::well_formed);
            }
        }
        res
    }

    /// Reclaim up to `budget` retired draining frames on `tier`.
    pub fn reclaim_draining(&mut self, tier: TierId, budget: usize) -> (r: BackendResult<usize>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            r matches Ok(reclaimed) ==> reclaimed <= budget,
    {
        hide(CacheCore::well_formed);
        if tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let reclaimed = self.core.reclaim_draining(tier, budget);
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        Ok(reclaimed)
    }

    /// Reclaim retired draining frames across every registered tier.
    pub fn reclaim_all_draining(&mut self) -> (total: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let total = self.core.reclaim_all_draining();
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        total
    }

    // =========================================================================
    // Multi-Peer RDMA / Full-Duplex / Cooperative Revocation / QP Recovery
    // =========================================================================
    pub fn send_control_to(
        &mut self,
        peer_idx: usize,
        payload: ControlPayload,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        match self.peers[peer_idx].send_control(&self.core, payload, Tracked(instance)) {
            Ok(value) => Ok(value),
            Err(ChannelError::InvalidFrame) => Err(CoreError::InvalidArgument),
            Err(_) => Err(rdma_error(rdma::RdmaError(-1))),
        }
    }

    pub fn send_fetch_request_to(
        &mut self,
        peer_idx: usize,
        key: ObjectKey,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        if self.peers[peer_idx].next_request_id == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        match self.peers[peer_idx].send_fetch_request(&self.core, key, Tracked(instance)) {
            Ok(value) => Ok(value),
            Err(ChannelError::InvalidFrame) => Err(CoreError::InvalidArgument),
            Err(_) => Err(rdma_error(rdma::RdmaError(-1))),
        }
    }

    pub fn send_fetch_batch_to(
        &mut self,
        peer_idx: usize,
        keys: &[ObjectKey],
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec(),
            r matches Ok(base) ==> base == old(
                self,
            ).peers_spec()@[peer_idx as int].next_request_id_spec() && base + keys.len()
                <= u64::MAX,
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        match self.peers[peer_idx].send_fetch_batch(&self.core, keys, Tracked(instance)) {
            Ok(value) => Ok(value),
            Err(ChannelError::InvalidFrame) => Err(CoreError::InvalidArgument),
            Err(_) => Err(rdma_error(rdma::RdmaError(-1))),
        }
    }

    pub fn serve_fetch_request_for_peer(
        &mut self,
        peer_idx: usize,
        tier: TierId,
        request_id: u64,
        key: ObjectKey,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).peers_spec()@[peer_idx as int].next_grant_tag_spec() < u64::MAX,
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if self.peers[peer_idx].next_grant_tag == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].serve_fetch_request(
            &mut self.core,
            tier,
            request_id,
            key,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    pub fn poll_peer_event_from(
        &mut self,
        peer_idx: usize,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<DistributedEvent, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            r matches Ok(DistributedEvent::ReadGrant(ref g)) ==> g.valid_for_peer(
                final(self),
                peer_idx as int,
            ),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;

        let ghost before_peers = self.peers@;
        let msg = match self.peers[peer_idx].recv_message(&self.core, Tracked(instance)) {
            Ok(m) => m,
            Err(_) => return Err(rdma_error(rdma::RdmaError(-1))),
        };
        match msg {
            PeerMessage::Control { payload, .. } => Ok(DistributedEvent::Control(payload)),
            PeerMessage::ReadGrant {
                tag,
                meta,
                cap,
                share,
                statement,
                cert,
                revoked,
                version_binding,
            } => Ok(
                DistributedEvent::ReadGrant(
                    PendingReadGrant {
                        core_id: self.peers[peer_idx].rdma.core_id,
                        peer_timeline: self.peers[peer_idx].rdma.timeline_id,
                        connection_epoch: self.peers[peer_idx].reconnect_epoch,
                        tag,
                        meta,
                        cap,
                        share,
                        statement,
                        cert,
                        revoked,
                        version_binding,
                    },
                ),
            ),
            PeerMessage::ReadRelease { tag, meta, share, statement } => {
                let ghost before = self.core;
                let res = self.peers[peer_idx].handle_read_release(
                    &mut self.core,
                    tag,
                    share,
                    statement,
                    Tracked(instance),
                );
                self.restore_after_peer_step(
                    peer_idx,
                    Ghost(before),
                    Ghost(before_peers),
                    Tracked(instance),
                );
                match res {
                    Ok(()) => Ok(DistributedEvent::ReadReleased { tag, meta, was_active: true }),
                    Err(CoreError::InvalidArgument) => {
                        Ok(DistributedEvent::ReadReleased { tag, meta, was_active: false })
                    },
                    Err(e) => Err(e),
                }
            },
            PeerMessage::Unexpected => Ok(DistributedEvent::Unexpected),
        }
    }

    pub fn pull_with_grant_from(
        &mut self,
        peer_idx: usize,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        grant: PendingReadGrant,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(self).well_formed(),
            grant.valid_for_peer(old(self), peer_idx as int),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).core_spec().frames_spec().tier_frame_size(dst_tier) as u64
                <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core_spec().instance_id(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_host_mr_spec() matches Some(
                    s,
                ) && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_gpu_mr_spec() matches Some(s)
                    && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        self.check_pull_target(peer_idx, dst_tier, target)?;
        if !grant.matches_connection(&self.peers[peer_idx]) || grant.meta.key != key {
            return Err(CoreError::InvalidArgument);
        }
        if self.peers[peer_idx].rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        if grant.meta.frame_size != self.core.registered_tier_frame_size(dst_tier) as u64 {
            return Err(CoreError::InvalidArgument);
        }
        let PendingReadGrant {
            tag,
            meta,
            cap,
            share,
            statement,
            cert,
            revoked,
            version_binding,
            ..
        } = grant;
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let ghost orig_share_frontier = share@.frontier();
        let pull = match self.peers[peer_idx].fetch_post_pull(
            &mut self.core,
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
        ) {
            Ok(p) => p,
            Err(e) => {
                self.restore_after_peer_step(
                    peer_idx,
                    Ghost(before),
                    Ghost(before_peers),
                    Tracked(instance),
                );
                return Err(e);
            },
        };
        let ghost after_post = self.core;
        let ghost after_post_leases = self.peers@[peer_idx as int].rdma.active_leases@;
        let ghost pull_frame = pull.handle.frame;
        let tracked vb = version_binding.get().tracked_unwrap();
        let res = self.peers[peer_idx].fetch_complete_pull(
            &mut self.core,
            tag,
            meta.request_id,
            pull,
            Tracked(vb),
            Ghost(orig_share_frontier),
            Tracked(instance),
        );
        proof {
            lemma_peer_core_step_trans_retire_new_init(
                &before,
                &after_post,
                &self.core,
                self.peers@[peer_idx as int].rdma.timeline_id,
                pull_frame,
            );
            lemma_foreign_shares_trans(
                &before,
                &after_post,
                &self.core,
                before_peers[peer_idx as int].rdma.active_leases@,
                after_post_leases,
                self.peers@[peer_idx as int].rdma.active_leases@,
            );
        }
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Pull received grants from one peer in one doorbell. Use
    /// `send_fetch_batch_to` to request them and correlate replies by request ID.
    pub fn fetch_batch_from_peer(
        &mut self,
        peer_idx: usize,
        dst_tier: TierId,
        target: PullTarget,
        grants: Vec<PendingReadGrant>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (res: Result<FetchBatchPulls, CoreError>)
        requires
            old(self).well_formed(),
            peer_idx < old(self).peers_spec()@.len(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).peers_spec()@[peer_idx as int].pull_tier_valid(dst_tier, target),
            old(self).core_spec().frames_spec().tier_frame_size(dst_tier) as u64
                <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core_spec().instance_id(),
            forall|i: int|
                0 <= i < grants.len() ==> (#[trigger] grants[i]).valid_for_peer(
                    old(self),
                    peer_idx as int,
                ),
            forall|i: int, j: int|
                0 <= i < j < grants.len() ==> (#[trigger] grants[i]).tag_spec() != (
                #[trigger] grants[j]).tag_spec(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            res matches Ok(r) ==> r.results.len() + r.pending.len() == grants.len(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        self.check_pull_target(peer_idx, dst_tier, target)?;

        if self.peers[peer_idx].rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let count = grants.len();
        let ghost original = grants@;
        let mut input = grants.into_iter();
        let mut pending: Vec<PendingInboundGrant> = Vec::with_capacity(count);
        while pending.len() < count
            invariant
                *self == *old(self),
                self.well_formed(),
                peer_idx < self.peers@.len(),
                self.peers@[peer_idx as int].rdma.qp.ready(),
                self.peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
                pending.len() <= count == original.len(),
                vstd::std_specs::iter::IteratorSpec::remaining(&input) == original.skip(
                    pending.len() as int,
                ),
                forall|i: int|
                    0 <= i < pending.len() ==> {
                        let g = #[trigger] pending[i];
                        &&& g.tag == original[i].tag
                        &&& g.valid(
                            &self.core,
                            &self.peers@[peer_idx as int].channel,
                            self.peers@[peer_idx as int].rdma.peer_node,
                        )
                    },
                forall|i: int|
                    0 <= i < count ==> (#[trigger] original[i]).valid_for_peer(
                        self,
                        peer_idx as int,
                    ),
            decreases count - pending.len(),
        {
            let grant = input.next().unwrap();
            if !grant.matches_connection(&self.peers[peer_idx]) {
                return Err(CoreError::InvalidArgument);
            }
            pending.push(grant.into_pending());
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let r = self.peers[peer_idx].fetch_batch_from_grants(
            &mut self.core,
            dst_tier,
            target,
            pending,
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        Ok(r)
    }

    pub fn fetch_from_peer(
        &mut self,
        peer_idx: usize,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).peers_spec()@[peer_idx as int].next_request_id_spec() < u64::MAX,
            old(self).core_spec().frames_spec().tier_frame_size(dst_tier) as u64
                <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core_spec().instance_id(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_host_mr_spec() matches Some(
                    s,
                ) && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_gpu_mr_spec() matches Some(s)
                    && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        self.check_pull_target(peer_idx, dst_tier, target)?;
        if self.peers[peer_idx].next_request_id == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        if self.peers[peer_idx].rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].fetch_remote_frame(
            &mut self.core,
            key,
            dst_tier,
            target,
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Full-duplex symmetric remote fetch from `peer_idx`: while awaiting the requested
    /// grant and RDMA READ completion, actively serves inbound `FetchRequest`s,
    /// advances pending outbound binds, and acknowledges/replenishes MWs.
    pub fn fetch_from_peer_duplex(
        &mut self,
        peer_idx: usize,
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
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).peers_spec()@[peer_idx as int].next_request_id_spec() < u64::MAX,
            old(self).core_spec().frames_spec().tier_frame_size(dst_tier) as u64
                <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            pd matches Some(p) ==> p.id() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().pd() && p.node() == old(
                self,
            ).core_spec().node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_host_mr_spec() matches Some(
                    s,
                ) && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_gpu_mr_spec() matches Some(s)
                    && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        self.check_pull_target(peer_idx, dst_tier, target)?;
        if self.peers[peer_idx].next_request_id == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        if let Some(pd) = pd {
            self.check_peer_pd(peer_idx, pd)?;
        }
        if self.peers[peer_idx].rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].fetch_remote_frame_duplex(
            &mut self.core,
            serve_tier,
            pd,
            tokens,
            key,
            dst_tier,
            target,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    pub fn poll_peer_progress(
        &mut self,
        peer_idx: usize,
        serve_tier: TierId,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            pd matches Some(p) ==> p.id() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().pd() && p.node() == old(
                self,
            ).core_spec().node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(pd) = pd {
            self.check_peer_pd(peer_idx, pd)?;
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].poll_progress(
            &mut self.core,
            serve_tier,
            pd,
            tokens,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Round-robin non-blocking progress sweep across all attached `PeerSession`s.
    /// A supplied PD enables replenishment using each session's own PD.
    pub fn poll_all_peers_progress(
        &mut self,
        serve_tier: TierId,
        replenish: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            forall|i: int|
                0 <= i < old(self).peers_spec()@.len() ==> {
                    let p = #[trigger] old(self).peers_spec()@[i];
                    &&& p.rdma_spec().qp_spec().ready()
                    &&& p.rdma_spec().qp_spec().state() is ReadyToSend
                },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            forall|i: int|
                0 <= i < final(self).peers_spec()@.len() ==> (
                #[trigger] final(self).peers_spec()@[i]).rdma_spec().qp_spec() == old(
                    self,
                ).peers_spec()@[i].rdma_spec().qp_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        if serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        self.check_all_peers_ready()?;

        let mut total = PeerProgress::empty();
        let mut idx: usize = 0;
        while idx < self.peers.len()
            invariant
                idx <= self.peers@.len(),
                self.well_formed(),
                self.host == old(self).host,
                self.device == old(self).device,
                self.nvme == old(self).nvme,
                self.peers@.len() == old(self).peers@.len(),
                self.core.node_id_spec() == old(self).core.node_id_spec(),
                self.core.instance_id() == old(self).core.instance_id(),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& p.rdma.qp == old(self).peers@[i].rdma.qp
                        &&& p.rdma.qp.ready()
                        &&& p.rdma.qp.state() is ReadyToSend
                    },
                instance.id() == self.core.instance_id(),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.node() == old(shares).node(),
                shares.id() == self.core.instance_id(),
                shares.node() == self.core.node(),
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source().is_rdma_qp() && tokens@[k]@.node()
                        == self.core.node(),
            decreases self.peers@.len() - idx,
        {
            let pd = if replenish.is_some() {
                Some(self.peers[idx].rdma.qp.protection_domain())
            } else {
                None
            };
            let p = self.poll_peer_progress(
                idx,
                serve_tier,
                pd,
                tokens,
                Tracked(shares),
                Tracked(instance),
            )?;
            if total.cq_processed <= usize::MAX - p.cq_processed {
                total.cq_processed += p.cq_processed;
            }
            if total.grants_completed <= usize::MAX - p.grants_completed {
                total.grants_completed += p.grants_completed;
            }
            if total.windows_replenished <= usize::MAX - p.windows_replenished {
                total.windows_replenished += p.windows_replenished;
            }
            idx += 1;
        }
        Ok(total)
    }

    /// Cooperative-first grant revocation on `peer_idx`: returns `Ok(None)` when
    /// the peer releases within `spin_budget` (zero MW quarantine), or `Ok(Some(mw_slot))`
    /// when falling back to unilateral fenced `LOCAL_INV` + `ibv_dealloc_mw`.
    pub fn revoke_grant_cooperative_first(
        &mut self,
        peer_idx: usize,
        grant_tag: GrantTag,
        spin_budget: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Option<u32>, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;

        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].revoke_lease_cooperative_first(
            &mut self.core,
            &mut self.host_ctx,
            grant_tag,
            spin_budget,
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Drain and cooperatively-first revoke all grants referencing `frame` across
    /// every peer in `self.peers`, then transition `frame` to `STATE_DRAINING` if
    /// still `STATE_PUBLISHED`.
    pub fn drain_and_evict_frame_across_peers(
        &mut self,
        frame: FrameId,
        spin_budget: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(self).well_formed(),
            forall|i: int|
                0 <= i < old(self).peers_spec()@.len() ==> {
                    let p = #[trigger] old(self).peers_spec()@[i];
                    &&& p.rdma_spec().qp_spec().ready()
                    &&& p.rdma_spec().qp_spec().state() is ReadyToSend
                },
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            forall|i: int|
                0 <= i < final(self).peers_spec()@.len() ==> (
                #[trigger] final(self).peers_spec()@[i]).rdma_spec().qp_spec() == old(
                    self,
                ).peers_spec()@[i].rdma_spec().qp_spec(),
    {
        hide(CacheCore::well_formed);
        self.core.check_published_frame(frame)?;
        self.check_all_peers_ready()?;

        let mut total_evicted: usize = 0;
        let mut idx: usize = 0;
        while idx < self.peers.len()
            invariant
                idx <= self.peers@.len(),
                self.well_formed(),
                self.host == old(self).host,
                self.device == old(self).device,
                self.nvme == old(self).nvme,
                self.peers@.len() == old(self).peers@.len(),
                self.core.instance_id() == old(self).core.instance_id(),
                instance.id() == self.core.instance_id(),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& p.rdma.qp == old(self).peers@[i].rdma.qp
                        &&& p.rdma.qp.ready()
                        &&& p.rdma.qp.state() is ReadyToSend
                    },
            decreases self.peers@.len() - idx,
        {
            let ghost before = self.core;
            let ghost before_peers = self.peers@;
            let res = self.peers[idx].drain_and_evict_frame_leases(
                &mut self.core,
                &mut self.host_ctx,
                frame,
                spin_budget,
                Tracked(instance),
            );
            self.restore_after_peer_step(
                idx,
                Ghost(before),
                Ghost(before_peers),
                Tracked(instance),
            );
            let n = res?;
            if total_evicted <= usize::MAX - n {
                total_evicted += n;
            }
            idx += 1;
        }
        let _ = self.evict_frame(frame);
        Ok(total_evicted)
    }

    pub fn acknowledge_peer_quarantined_window(&mut self, peer_idx: usize, mw_slot: u32) -> (r:
        BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        self.peers[peer_idx].rdma.acknowledge_quarantined_window(&self.core, mw_slot)
    }

    pub fn replenish_peer_window(
        &mut self,
        peer_idx: usize,
        mw_slot: u32,
        pd: &'a rdma::ProtectionDomain<'a>,
        Tracked(name): Tracked<EventSourceToken>,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            (mw_slot as int) < old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().slots_spec().len(),
            old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().slots_spec()[mw_slot as int].state_spec()
                == crate::backends::rdma::MwSlotState::Vacant,
            pd.id() == old(self).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().pd(),
            pd.node() == old(self).core_spec().node(),
            name.well_formed(),
            name.source().is_rdma_qp(),
            name.node() == pd.node(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        self.check_peer_pd(peer_idx, pd)?;
        let slots = &self.peers[peer_idx].rdma.window_pool.slots;
        if mw_slot as usize >= slots.len() {
            return Err(CoreError::InvalidArgument);
        }
        if !matches!(slots[mw_slot as usize].state, crate::backends::rdma::MwSlotState::Vacant) {
            return Err(CoreError::WouldConflict);
        }
        self.peers[peer_idx].rdma.replenish_window(&self.core, mw_slot, pd, Tracked(name))
    }

    pub fn enter_peer_qp_error(&mut self, peer_idx: usize) -> (r: BackendResult<
        Tracked<rdma::QpErrorWitness>,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().cq_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().cq_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            r.is_ok() ==> final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state()
                == rdma::QueuePairState::Error,
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        self.peers[peer_idx].rdma.enter_qp_error(&self.core)
    }

    pub fn reset_and_reconnect_peer(
        &mut self,
        peer_idx: usize,
        psn: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend
                || old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is Error,
            psn <= 0xffffff,
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            r.is_ok() ==> final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready()
                && final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        if psn > 0xffffff {
            return Err(CoreError::InvalidArgument);
        }
        if self.peers[peer_idx].reconnect_epoch == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        if self.peers[peer_idx].rdma.cq.in_flight_count() != 0 {
            return Err(CoreError::WouldConflict);
        }
        proof {
            assert(self.peers@[peer_idx as int].rdma.cq.in_flight().is_empty());
        }
        let local_endpoint = self.peers[peer_idx].rdma.qp.reset_endpoint(psn);
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].reset_and_reconnect(
            &mut self.core,
            &mut self.host_ctx,
            &local_endpoint,
            psn,
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Cleanly shut down and pop every attached `PeerSession`, recovering all active
    /// remote leases across all 8 states and releasing all MWs and MRs.
    pub fn shutdown_all_peers(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<(), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            r.is_ok() ==> final(self).peers_spec()@.len() == 0,
    {
        hide(CacheCore::well_formed);
        while self.peers.len() > 0
            invariant
                self.well_formed(),
                self.host == old(self).host,
                self.device == old(self).device,
                self.nvme == old(self).nvme,
                self.core.instance_id() == old(self).core.instance_id(),
                instance.id() == self.core.instance_id(),
            decreases self.peers@.len(),
        {
            let ghost pre_pop_peers = self.peers@;
            let mut peer = self.peers.pop().unwrap();
            let ghost last_idx = (pre_pop_peers.len() - 1) as int;
            let ghost before = self.core;
            let ghost before_leases = peer.rdma.active_leases@;
            let ghost rdma_tl = peer.rdma.timeline_id;
            let res = peer.shutdown(&mut self.core, &mut self.host_ctx, Tracked(instance));
            let ghost after_leases = peer.rdma.active_leases@;
            let ghost fp = peer_step_with_pins(rdma_tl, None, before_leases);
            proof {
                reveal(peer_core_step_except);
                reveal(core_transition);
                lemma_core_wf_facts(&self.core);
                lemma_core_wf_facts(&before);
                self.host.ctx.source.borrow().lemma_all_future_unsubmitted(
                    instance,
                    &self.core.borrow_ledger().submitted,
                );
                self.host_ctx.source.borrow().lemma_all_future_unsubmitted(
                    instance,
                    &self.core.borrow_ledger().submitted,
                );
                lemma_peer_step_footprint_with_pins(
                    &before,
                    &self.core,
                    rdma_tl,
                    None,
                    before_leases,
                    after_leases,
                );
                lemma_core_transition_preserves_host(&before, &self.core, fp, self.host);
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.device,
                    &self.nvme,
                    &pre_pop_peers[last_idx],
                ));
                lemma_preserve_nvme(&before, &self.core, fp, &self.nvme);
                assert forall|j: int| 0 <= j < self.peers@.len() implies {
                    &&& (#[trigger] self.peers@[j]).well_formed(&self.core)
                    &&& peer_session_compatible(
                        &self.core,
                        self.host,
                        &self.device,
                        &self.nvme,
                        &self.peers@[j],
                    )
                } by {
                    assert(self.peers@[j] == pre_pop_peers[j]);
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.device,
                        &self.nvme,
                        &pre_pop_peers[j],
                    ));
                    assert(pre_pop_peers[j].rdma.timeline_id != rdma_tl);
                    assert(peer_leases_disjoint(
                        pre_pop_peers[j].rdma.active_leases@,
                        before_leases,
                    ));
                    lemma_peer_step_preserves_sibling_session(
                        &before,
                        &self.core,
                        rdma_tl,
                        before_leases,
                        after_leases,
                        &self.peers@[j],
                    );
                }
                assert forall|p: int, q: int| 0 <= p < q < self.peers@.len() implies {
                    let a = #[trigger] self.peers@[p];
                    let b = #[trigger] self.peers@[q];
                    &&& a.rdma.timeline_id != b.rdma.timeline_id
                    &&& a.rdma.peer_node != b.rdma.peer_node
                    &&& peer_leases_disjoint(a.rdma.active_leases@, b.rdma.active_leases@)
                } by {
                    assert(self.peers@[p] == pre_pop_peers[p]);
                    assert(self.peers@[q] == pre_pop_peers[q]);
                }
            }
            match &self.device {
                DeviceAttachment::None => {},
                DeviceAttachment::Borrowed(c) => {
                    proof {
                        c.lemma_device_layout(&before);
                    }
                    c.stream().lemma_source_fresh(
                        Tracked(instance),
                        Tracked(&self.core.borrow_ledger().submitted),
                    );
                    proof {
                        assert(peer_session_compatible(
                            &before,
                            self.host,
                            &self.device,
                            &self.nvme,
                            &pre_pop_peers[last_idx],
                        ));
                        c.preserve_idle_across_core_transition(&before, &self.core, fp);
                    }
                },
                DeviceAttachment::Owned(c) => {
                    proof {
                        c.lemma_device_layout(&before);
                    }
                    c.stream().lemma_source_fresh(
                        Tracked(instance),
                        Tracked(&self.core.borrow_ledger().submitted),
                    );
                    proof {
                        assert(peer_session_compatible(
                            &before,
                            self.host,
                            &self.device,
                            &self.nvme,
                            &pre_pop_peers[last_idx],
                        ));
                        reveal(peer_core_step_except);
                        reveal(core_transition);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                        assert(self.core.frames_spec().consistent(self.core.kernel_view())) by {
                            reveal(CacheCore::well_formed);
                        }
                        lemma_core_wf_facts(&before);
                        assert forall|f: FrameId|
                            f.0 < before.frames_spec().metas_spec().len() && (
                            #[trigger] before.frames_spec().meta(f)).state
                                == STATE_INITIALIZING implies self.core.frames_spec().ghosts_spec()@.contains_key(
                        f.0 as nat) && self.core.frames_spec().ghosts_spec()@[f.0 as nat]
                            == before.frames_spec().frame(f) by {
                            assert(f.0 < self.core.frames_spec().metas_spec().len());
                            assert(self.core.frames_spec().frame(f) == before.frames_spec().frame(
                                f,
                            ));
                        }
                        let ghost pre_core = self.core;
                        let tracked (frame_ghosts, ledger) =
                            self.core.borrow_frame_ghosts_and_ledger();
                        c.prove_pending_destinations_preserved(
                            &before,
                            frame_ghosts,
                            ledger,
                            instance,
                        );
                        assert(self.core == pre_core);
                        c.preserve_across_core_transition_with_bindings(
                            &before,
                            &self.core,
                            self.core.borrow_version_registry(),
                            self.pending_device_bindings.borrow(),
                            fp,
                        );
                    }
                },
            }
            res?;
        }
        Ok(())
    }

    pub fn revoke_peer_lease_unilateral(
        &mut self,
        peer_idx: usize,
        grant_tag: GrantTag,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u32, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;

        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].revoke_lease_unilateral(
            &mut self.core,
            &mut self.host_ctx,
            grant_tag,
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }
}

#[cfg(feature = "nvme")]
impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    pub fn spill_host_to_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Err((CoreError::Miss, None));
        };
        let (dst_frame, ticket) = match self.stage_dram_to_nvme(
            Tracked(instance),
            Tracked(shares),
            src_frame,
        ) {
            Ok(pair) => pair,
            Err(e) => return Err((e, None)),
        };
        let submit_res = self.submit_staged_nvme();
        let rem_staged = match &self.nvme {
            Some(n) => n.staged_len(),
            None => 0,
        };
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
    }

    /// Submit the currently staged NVMe operations.
    pub fn submit_staged_nvme(&mut self) -> (r: BackendResult<usize>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            match old(self).nvme_spec() {
                Some(n) => final(self).nvme_spec() matches Some(new) && new.timeline_spec()
                    == n.timeline_spec(),
                None => final(self).nvme_spec() is None,
            },
    {
        match &mut self.nvme {
            Some(nvme) => nvme.submit(Ghost(&self.core)),
            None => Ok(0),
        }
    }

    pub fn stage_dram_to_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        if src_frame.0 as usize >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(mut nvme) = self.nvme.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let ghost src_ev = nvme.ring.source().source();
        let staged = nvme.stage_dram_to_nvme(
            self.host,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            src_frame,
        );
        proof {
            let ghost none_nvme: Option<NvmeBackend> = None;
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.device,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            if let Some(ref device) = self.device.spec_ref() {
                device.lemma_device_layout(&before);
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &self.device,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames_spec().versions_spec()@
                == before.frames_spec().versions_spec()@) by {
                reveal(CacheCore::well_formed);
            }
            assert(self.core.all_versions_bound());
        }
        self.nvme = Some(nvme);
        staged
    }

    pub fn warm_host_from_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        let Some(nvme_tier) = self.nvme_tier() else {
            return Err((CoreError::UnsupportedTier, None));
        };
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, nvme_tier) else {
            return Err((CoreError::Miss, None));
        };
        let (dst_frame, ticket) = match self.stage_nvme_to_dram(
            Tracked(instance),
            Tracked(shares),
            src_frame,
        ) {
            Ok(pair) => pair,
            Err(e) => return Err((e, None)),
        };
        let submit_res = self.submit_staged_nvme();
        let rem_staged = match &self.nvme {
            Some(n) => n.staged_len(),
            None => 0,
        };
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
    }

    pub fn stage_nvme_to_dram(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        if src_frame.0 as usize >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(mut nvme) = self.nvme.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let ghost src_ev = nvme.ring.source().source();
        let staged = nvme.stage_nvme_to_dram(
            self.host,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            src_frame,
        );
        proof {
            let ghost none_nvme: Option<NvmeBackend> = None;
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.device,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            if let Some(ref device) = self.device.spec_ref() {
                device.lemma_device_layout(&before);
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &self.device,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames_spec().versions_spec()@
                == before.frames_spec().versions_spec()@) by {
                reveal(CacheCore::well_formed);
            }
            assert(self.core.all_versions_bound());
        }
        self.nvme = Some(nvme);
        staged
    }

    pub fn poll_nvme_completions(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        budget: usize,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r matches Ok(p) ==> p.completed + p.failed <= budget,
    {
        hide(CacheCore::well_formed);
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, NvmeProgress::empty()));
        };
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let ghost poll_fp = nvme.poll_footprint();
        let res = nvme.poll_completions_up_to(&mut self.core, Tracked(instance), budget);
        proof {
            let ghost none_nvme: Option<NvmeBackend> = None;
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.device,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                poll_fp,
                self.host,
                &self.host_ctx,
                &self.device,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames_spec().versions_spec()@
                == before.frames_spec().versions_spec()@) by {
                reveal(core_transition);
            }
            assert(self.core.all_versions_bound());
        }
        self.nvme = Some(nvme);
        res
    }

    pub fn wait_nvme_completion(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
    {
        hide(CacheCore::well_formed);
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, NvmeProgress::empty()));
        };
        if nvme.staged_len() > 0 {
            if let Err(e) = nvme.submit(Ghost(&self.core)) {
                self.nvme = Some(nvme);
                return Err((e, NvmeProgress::empty()));
            }
        }
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let ghost poll_fp = nvme.poll_footprint();
        let res = nvme.wait_one_completion(
            &mut self.core,
            Tracked(instance),
            NvmeProgress::empty(),
        );
        proof {
            let ghost none_nvme: Option<NvmeBackend> = None;
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.device,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                poll_fp,
                self.host,
                &self.host_ctx,
                &self.device,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames_spec().versions_spec()@
                == before.frames_spec().versions_spec()@) by {
                reveal(core_transition);
            }
            assert(self.core.all_versions_bound());
        }
        self.nvme = Some(nvme);
        res
    }

    pub fn wait_nvme_ticket(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        ticket: Ticket,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r.is_ok() ==> final(self).core_spec().timelines_spec().retired(ticket),
    {
        hide(CacheCore::well_formed);
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, NvmeProgress::empty()));
        };
        let ghost tl = nvme.timeline;
        let mut progress = NvmeProgress::empty();
        let mut steps: usize = 0;
        proof {
            lemma_core_wf_facts(&self.core);
        }
        if ticket.is_none() || ticket.timeline() != nvme.timeline
            || !self.core.timelines().contains_ticket(ticket) {
            self.nvme = Some(nvme);
            return Err((CoreError::InvalidArgument, progress));
        }
        while steps < BITMAP_CAPACITY as usize && (nvme.in_flight_len() > 0 || nvme.staged_len()
            > 0) && !self.core.timelines().is_retired(ticket)
            invariant
                steps <= BITMAP_CAPACITY as usize,
                progress.completed + progress.failed == steps,
                self.core.well_formed(),
                self.core.all_versions_bound(),
                self.core.timelines_spec().bounds(),
                self.host.well_formed(&self.core),
                self.host_ctx.ready(self.core.instance_id(), self.core.node()),
                source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger_spec()@),
                self.host_ctx.source@.source() != self.host.ctx.source@.source(),
                self.host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0,
                self.peers == old(self).peers,
                peers_pairwise_disjoint(self.peers@),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& peer_session_compatible(&self.core, self.host, &self.device, &None, &p)
                        &&& tl != p.rdma.timeline_id
                    },
                nvme.well_formed(&self.core),
                nvme.timeline == tl,
                nvme.timeline != self.host.timeline,
                self.device == old(self).device,
                self.pending_device_bindings == old(self).pending_device_bindings,
                Self::device_attachment_valid(
                    &self.core,
                    self.host,
                    &self.device,
                    self.pending_device_bindings@,
                ),
                self.device.spec_ref() matches Some(c) ==> nvme.timeline != c.timeline_id_spec(),
                self.nvme is None,
                self.host == old(self).host,
                self.core.node_id_spec() == old(self).core.node_id_spec(),
                self.core.instance_id() == old(self).core.instance_id(),
                self.core.tiers_spec() == old(self).core.tiers_spec(),
                instance.id() == self.core.instance_id(),
                self.core.timelines_spec().valid_ticket(ticket),
                old(self).core.spec_spec()@.versions_monotonic(self.core.spec_spec()@),
            decreases (BITMAP_CAPACITY as usize) - steps,
        {
            if nvme.staged_len() > 0 && !self.core.timelines().is_retired(ticket) {
                match nvme.submit(Ghost(&self.core)) {
                    Ok(n) => {
                        if n == 0 && nvme.in_flight_len() == 0 {
                            self.nvme = Some(nvme);
                            return Err((CoreError::WouldConflict, progress));
                        }
                    },
                    Err(e) => {
                        self.nvme = Some(nvme);
                        return Err((e, progress));
                    },
                }
            }
            if nvme.in_flight_len() == 0 {
                self.nvme = Some(nvme);
                return Err((CoreError::WouldConflict, progress));
            }
            let ghost before_step = self.core;
            let ghost poll_fp = nvme.poll_footprint();
            let step_res = nvme.wait_one_completion(&mut self.core, Tracked(instance), progress);
            proof {
                let ghost none_nvme: Option<NvmeBackend> = None;
                Self::lemma_preserve_siblings(
                    &before_step,
                    &self.core,
                    poll_fp,
                    self.host,
                    &self.host_ctx,
                    &self.device,
                    &none_nvme,
                    self.peers@,
                );
                lemma_versions_monotonic_trans(
                    old(self).core.spec_spec()@,
                    before_step.spec_spec()@,
                    self.core.spec_spec()@,
                );
                assert(self.core.frames_spec().versions_spec()@
                    == before_step.frames_spec().versions_spec()@) by {
                    reveal(core_transition);
                }
                assert(self.core.all_versions_bound());
                lemma_core_wf_facts(&self.core);
            }
            match step_res {
                Ok(p) => {
                    progress = p;
                    steps += 1;
                },
                Err((e, p)) => {
                    self.nvme = Some(nvme);
                    return Err((e, p));
                },
            }
        }
        self.nvme = Some(nvme);
        if self.core.timelines().is_retired(ticket) {
            Ok(progress)
        } else {
            Err((CoreError::WouldConflict, progress))
        }
    }

    pub fn spill_host_to_nvme_sync(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: BackendResult<FrameId>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        let (dst_frame, ticket) = match self.spill_host_to_nvme(
            Tracked(instance),
            Tracked(shares),
            key,
        ) {
            Ok(pair) => pair,
            Err((e, _)) => return Err(e),
        };
        match self.wait_nvme_ticket(Tracked(instance), ticket) {
            Ok(_) => Ok(dst_frame),
            Err((e, _)) => Err(e),
        }
    }

    pub fn warm_host_from_nvme_sync(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: BackendResult<FrameId>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        let (dst_frame, ticket) = match self.warm_host_from_nvme(
            Tracked(instance),
            Tracked(shares),
            key,
        ) {
            Ok(pair) => pair,
            Err((e, _)) => return Err(e),
        };
        match self.wait_nvme_ticket(Tracked(instance), ticket) {
            Ok(_) => Ok(dst_frame),
            Err((e, _)) => Err(e),
        }
    }
}

impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    // =========================================================================
    // Verified Facade Mechanism Primitives (Cross-Tier Hops, Leases, Multi-Tier Serving)
    // =========================================================================
    /// Synchronously copy a published GPU-VRAM or Pinned-Host frame into the
    /// opposite tier (`GPU <-> Pinned Host`) using caller's idle `src_stream` /
    /// `src_timeline`. Works with both `DeviceAttachment::Borrowed` and `Owned`
    /// while preserving existing `VersionId` and cluster-wide `VersionBinding`.
    pub fn copy_hop_cuda_sync(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_stream: &mut D::Stream,
        src_timeline: TimelineId,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(self).well_formed(),
            old(self).device_spec().spec_ref() matches Some(c) && {
                &&& c.caller_stream_valid(*old(src_stream), &old(self).core_spec(), src_timeline)
                &&& src_timeline != old(self).host_spec().timeline_spec()
                &&& src_timeline != c.timeline_id_spec()
                &&& forall|i: int|
                    0 <= i < old(self).peers_spec()@.len() ==> src_timeline != (#[trigger] old(
                        self,
                    ).peers_spec()@[i]).rdma_spec().timeline_id_spec()
                &&& nvme_timeline_distinct(&old(self).nvme_spec(), src_timeline)
                &&& old(self).core_spec().timelines_spec().confirmed_seq_spec()[src_timeline as int]
                    + 1 == old(
                    self,
                ).core_spec().timelines_spec().next_seq_spec()[src_timeline as int]
                &&& old(src_stream).device_spec() == c.device_spec()
            },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).device_spec().spec_ref() matches Some(c) ==> c.caller_stream_valid(
                *final(src_stream),
                &final(self).core_spec(),
                src_timeline,
            ),
            r matches Ok((ver, dst_frame)) ==> {
                &&& final(self).device_spec().spec_ref() matches Some(c) && ((dst_tier
                    == c.device_tier_spec() && c.owns_device(dst_frame)) || c.owns_pinned_host(
                    dst_frame,
                ))
                &&& final(self).core_spec().timelines_spec().confirmed_seq_spec()[src_timeline as int]
                    + 1
                    == final(self).core_spec().timelines_spec().next_seq_spec()[src_timeline as int]
                &&& final(self).core_spec().frames_spec().meta(dst_frame).state == STATE_PUBLISHED
                &&& final(self).core_spec().frames_spec().frame(dst_frame).tier == dst_tier
                &&& final(self).core_spec().spec_spec()@.versions.contains_key(ver)
            },
    {
        hide(CacheCore::well_formed);
        if src_frame.0 as usize >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        if dst_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        if src_timeline == device.timeline_id() || !is_caller_device_timeline::<D>(
            &self.core,
            src_timeline,
            src_stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        if !self.core.timeline_is_idle(src_timeline) {
            return Err(CoreError::WouldConflict);
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            src_timeline,
            src_stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost src_source = src_stream.source().source();
        proof {
            device.lemma_caller_stream_valid(*src_stream, &before, src_timeline);
            src_stream.lemma_source_kind();
        }
        let res = device.copy_hop_sync(
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            src_stream,
            src_timeline,
            src_frame,
            dst_tier,
        );
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(src_timeline), Some(src_source)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound());
        }
        res
    }

    /// Synchronously read a specific published Host-DRAM frame into `dst`.
    pub fn read_host_frame(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst: &mut [u8],
    ) -> (r: BackendResult<VersionId>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            r matches Ok(ver) ==> final(self).core_spec().spec_spec()@.versions.contains_key(ver),
    {
        hide(CacheCore::well_formed);
        if frame.0 as usize >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let res = self.host.read_frame_into_with(
            &mut self.host_ctx,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst,
        );
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).well_formed(&self.core) by {
                lemma_core_transition_preserves_peer_session(
                    &before,
                    &self.core,
                    CoreFootprint::preserve_all(None, Some(old(self).host_ctx.source@.source())),
                    &self.peers@[i],
                );
            }
            if let Some(ref c) = self.device.spec_ref() {
                c.lemma_device_layout(&before);
                c.preserve_across_core_transition(
                    &before,
                    &self.core,
                    CoreFootprint::preserve_all(None, Some(old(self).host_ctx.source@.source())),
                );
            }
            lemma_preserve_nvme(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, Some(old(self).host_ctx.source@.source())),
                &self.nvme,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        res
    }

    /// Asynchronously read a specific published GPU-VRAM or Pinned-Host frame
    /// into caller's device slice on `dst_stream` (1-hop D2D or H2D Ordered Handoff).
    pub fn get_into_gpu_frame(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut D::Stream,
        dst_timeline: TimelineId,
        dst: &D::Slice<'_>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)>)
        requires
            old(self).well_formed(),
            old(self).device_spec().spec_ref() matches Some(c) && {
                &&& c.caller_stream_valid(*old(dst_stream), &old(self).core_spec(), dst_timeline)
                &&& dst_timeline != old(self).host_spec().timeline_spec()
                &&& dst_timeline != c.timeline_id_spec()
                &&& forall|i: int|
                    0 <= i < old(self).peers_spec()@.len() ==> dst_timeline != (#[trigger] old(
                        self,
                    ).peers_spec()@[i]).rdma_spec().timeline_id_spec()
                &&& nvme_timeline_distinct(&old(self).nvme_spec(), dst_timeline)
                &&& dst.slot_ref().loc.space.is_gpu() ==> dst.slot_ref().loc.space.gpu_device()
                    == c.device_spec()
            },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core_spec().instance_id(),
            dst.slot_ref().loc.node == old(self).core_spec().node(),
            old(self).device_spec().spec_ref() matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
            slot_is_owned(*old(dst_tok), old(self).core_spec().instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core_spec().citations()),
            old(self).core_spec().disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core_spec().frames_spec().metas_spec().len() ==> dst.slot_ref().loc
                    != (#[trigger] old(self).core_spec().frames_spec().frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).device_spec().spec_ref() matches Some(c) ==> c.caller_stream_valid(
                *final(dst_stream),
                &final(self).core_spec(),
                dst_timeline,
            ),
            r matches Ok((version, ticket, ev)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == dst_timeline
                &&& final(self).core_spec().spec_spec()@.versions.contains_key(version)
                &&& slot_owned(
                    *final(dst_tok),
                    old(self).core_spec().instance_id(),
                    dst.slot_ref().loc,
                    Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                )
            },
    {
        hide(CacheCore::well_formed);
        if frame.0 as usize >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        if dst_timeline == device.timeline_id() || !is_caller_device_timeline::<D>(
            &self.core,
            dst_timeline,
            dst_stream,
        ) || self.core.tier_frame_size(device.device_tier_id()) != Some(dst.len()) {
            return Err(CoreError::InvalidArgument);
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            dst_timeline,
            dst_stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        proof {
            device.lemma_caller_stream_valid(*dst_stream, &before, dst_timeline);
            dst_stream.lemma_source_kind();
        }
        let res = device.get_into_device(
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst_stream,
            dst_timeline,
            dst,
            Tracked(dst_tok),
        );
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(dst_timeline), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound());
        }
        res
    }

    /// Acquire a zero-copy local read lease on a published Host-DRAM frame.
    pub fn get_lease_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: BackendResult<Option<HostLease<'a>>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(Some(lease)) ==> {
                &&& lease.slice.well_formed()
                &&& lease.slice.instance_id() == final(self).core_spec().instance_id()
                &&& lease.slice.slot_ref() == final(self).core_spec().frames_spec().frame(
                    lease.handle.frame_spec(),
                ).slot
                &&& final(self).core_spec().active_lease_matches(lease.handle)
                &&& !final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.internal
                &&& lease.cap@.is_cooperative()
                &&& lease.cap@.instance_id() == final(self).core_spec().instance_id()
                &&& lease.cap@.loc() == lease.slice.slot_ref().loc
                &&& lease.cap@.share()
                    == final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.share@
                &&& read_admitted(lease.cap@.frontier(), final(self).core_spec().citations())
                &&& lease.cap@.frontier().readers.is_empty()
                &&& final(self).core_spec().frames_spec().returned_reads(
                    lease.handle.frame_spec(),
                    lease.cap@,
                    Ticket::NONE,
                    final(self).core_spec().kernel_view(),
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some((frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Ok(None);
        };
        let ghost before = self.core;
        let (handle, version, _cert, cap) = self.core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        )?;
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
            lemma_slab_owns_tier_frame(
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
                frame,
            );
            lemma_slab_frame_slot(
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
                frame,
            );
        }
        let slice = self.host.frame_slice(frame);
        Ok(Some(HostLease { handle, version, slice, cap }))
    }

    /// Acquire a zero-copy local read lease on a published GPU-VRAM frame.
    pub fn get_lease_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: BackendResult<Option<GpuLease>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(Some(lease)) ==> {
                &&& lease.handle.frame_spec() == lease.frame
                &&& final(self).device_spec().spec_ref() matches Some(c) && c.owns_device(
                    lease.frame,
                )
                &&& final(self).core_spec().active_lease_matches(lease.handle)
                &&& !final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.internal
                &&& lease.cap@.is_cooperative()
                &&& lease.cap@.instance_id() == final(self).core_spec().instance_id()
                &&& lease.cap@.loc() == final(self).core_spec().frames_spec().frame(
                    lease.frame,
                ).slot.loc
                &&& lease.cap@.share()
                    == final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.share@
                &&& read_admitted(lease.cap@.frontier(), final(self).core_spec().citations())
                &&& lease.cap@.frontier().readers.is_empty()
                &&& final(self).core_spec().frames_spec().returned_reads(
                    lease.frame,
                    lease.cap@,
                    Ticket::NONE,
                    final(self).core_spec().kernel_view(),
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        let Some((frame, _ver)) = self.core.lookup_tier(key, device.device_tier_id()) else {
            return Ok(None);
        };
        let ghost before = self.core;
        let (handle, version, _cert, cap) = self.core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        )?;
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
            device.lemma_device_layout(&self.core);
            lemma_slab_owns_tier_frame(
                &self.core,
                device.device_tier_spec(),
                device.device_first_frame_spec(),
                device.device_num_frames_spec(),
                device.buffer_spec().reg_space(),
                frame,
            );
            device.lemma_frame_ownership(frame);
        }
        let sync_point = device.frame_sync_point(frame);
        Ok(Some(GpuLease { handle, version, frame, sync_point, cap }))
    }

    /// Release a caller-owned (`!internal`) local read lease, recording its
    /// optional trailing `read_ticket` for Observed Reclamation.
    pub fn release_lease(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        handle: ReadLeaseHandle,
        Tracked(cap): Tracked<ShareCap>,
        read_ticket: Ticket,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(self).core_spec().active_lease_matches(handle) ==> {
                let share = old(
                    self,
                ).core_spec().active_leases_spec()[handle.slot_spec() as int]->Some_0.share@;
                &&& cap.share() == share
                &&& old(self).core_spec().frames_spec().returned_reads(
                    handle.frame_spec(),
                    cap,
                    read_ticket,
                    old(self).core_spec().kernel_view(),
                )
            },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        if handle.core_id() != self.core.runtime_id() || handle.frame().0 as usize
            >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        proof {
            lemma_core_wf_facts(&self.core);
        }
        if !self.core.timelines().contains_ticket(read_ticket) {
            return Err(CoreError::InvalidArgument);
        }
        let idx = handle.slot() as usize;
        if idx >= self.core.local_leases().len() {
            return Err(CoreError::InvalidArgument);
        }
        let is_internal = match &self.core.local_leases()[idx] {
            Some(lease) => lease.internal,
            None => false,
        };
        if is_internal {
            return Err(CoreError::WouldConflict);
        }
        let ghost before = self.core;
        let res = self.core.release_local_lease(
            handle,
            Tracked(cap),
            read_ticket,
            Tracked(instance),
        );
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        res
    }

    /// Donor-side handler for `FetchRequest` checking `primary_tier` first and
    /// optionally falling back to `fallback_tier` (e.g. GPU HBM -> Host DRAM).
    pub fn serve_fetch_request_multi_for_peer(
        &mut self,
        peer_idx: usize,
        primary_tier: TierId,
        fallback_tier: Option<TierId>,
        request_id: u64,
        key: ObjectKey,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).peers_spec()@[peer_idx as int].next_grant_tag_spec() < u64::MAX,
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if primary_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(tier) = fallback_tier {
            if tier.as_index() >= self.core.tier_count() {
                return Err(CoreError::InvalidArgument);
            }
        }
        if self.peers[peer_idx].next_grant_tag == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].serve_fetch_request_multi(
            &mut self.core,
            primary_tier,
            fallback_tier,
            request_id,
            key,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Full-duplex remote fetch from `peer_idx` that serves inbound `FetchRequest`s
    /// from `primary_tier` with optional `fallback_tier` (e.g. GPU HBM -> Host DRAM).
    pub fn fetch_from_peer_duplex_multi(
        &mut self,
        peer_idx: usize,
        primary_serve_tier: TierId,
        fallback_serve_tier: Option<TierId>,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            old(self).peers_spec()@[peer_idx as int].next_request_id_spec() < u64::MAX,
            old(self).core_spec().frames_spec().tier_frame_size(dst_tier) as u64
                <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            pd matches Some(p) ==> p.id() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().pd() && p.node() == old(
                self,
            ).core_spec().node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_host_mr_spec() matches Some(
                    s,
                ) && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers_spec()@[peer_idx as int].rdma_spec().effective_gpu_mr_spec() matches Some(s)
                    && s.tier_spec() == dst_tier && s.mr_spec().access().local_write
                    && s.mr_spec().gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if primary_serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(tier) = fallback_serve_tier {
            if tier.as_index() >= self.core.tier_count() {
                return Err(CoreError::InvalidArgument);
            }
        }
        self.check_pull_target(peer_idx, dst_tier, target)?;
        if self.peers[peer_idx].next_request_id == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        if let Some(pd) = pd {
            self.check_peer_pd(peer_idx, pd)?;
        }
        if self.peers[peer_idx].rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].fetch_remote_frame_duplex_multi(
            &mut self.core,
            primary_serve_tier,
            fallback_serve_tier,
            pd,
            tokens,
            key,
            dst_tier,
            target,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Drain one peer's queue of fetch requests for caller admission.
    pub fn take_inbound_fetches(&mut self, peer_idx: usize) -> (r: Result<
        Vec<(u64, ObjectKey)>,
        CoreError,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            peer_idx < old(self).peers_spec()@.len()
                ==> final(self).peers_spec()@[peer_idx as int].rdma_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec(),
            peer_idx >= old(self).peers_spec()@.len() ==> r == Err(CoreError::InvalidArgument)
                && *final(self) == *old(self),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            r matches Ok(requests) ==> {
                &&& peer_idx < old(self).peers_spec()@.len()
                &&& requests@ == old(self).peers_spec()@[peer_idx as int].inbound_fetches()
                &&& final(self).peers_spec()@[peer_idx as int].inbound_fetches().len() == 0
            },
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        Ok(self.peers[peer_idx].take_inbound_fetches(&self.core))
    }

    /// Available Type-2 MW bind headroom for one peer.
    pub fn peer_bind_budget(&self, peer_idx: usize) -> (r: Result<usize, CoreError>)
        requires
            self.well_formed(),
        ensures
            peer_idx >= self.peers_spec()@.len() ==> r == Err(CoreError::InvalidArgument),
            r matches Ok(n) ==> peer_idx < self.peers_spec()@.len() && n
                <= self.peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().free_spec()@.len(),
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        Ok(self.peers[peer_idx].bind_budget(&self.core))
    }

    /// Send a miss for a fetch the caller has decided to reject.
    pub fn reject_fetch_request(
        &mut self,
        peer_idx: usize,
        request_id: u64,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            peer_idx < old(self).peers_spec()@.len()
                ==> final(self).peers_spec()@[peer_idx as int].rdma_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec(),
            peer_idx >= old(self).peers_spec()@.len() ==> r == Err(CoreError::InvalidArgument)
                && *final(self) == *old(self),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        self.peers[peer_idx].reject_fetch_request(&self.core, request_id, Tracked(instance))
    }

    /// Post a bind for a caller-selected published frame. The pump sends its grant when ready.
    pub fn grant_fetch_from_frame(
        &mut self,
        peer_idx: usize,
        request_id: u64,
        key: ObjectKey,
        frame: FrameId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            peer_idx < old(self).peers_spec()@.len() ==> {
                let peer = old(self).peers_spec()@[peer_idx as int];
                &&& peer.rdma_spec().qp_spec().ready()
                &&& peer.rdma_spec().qp_spec().state() is ReadyToSend
            },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            peer_idx < old(self).peers_spec()@.len()
                ==> final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            peer_idx >= old(self).peers_spec()@.len() ==> r == Err(CoreError::InvalidArgument)
                && *final(self) == *old(self),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        self.core.check_published_frame(frame)?;
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].grant_fetch_from_frame(
            &mut self.core,
            request_id,
            key,
            frame,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Admit one fetch notice and post the selected frames in one bind doorbell.
    pub fn grant_fetch_batch(
        &mut self,
        peer_idx: usize,
        serve_tier: TierId,
        entries: &[(u64, ObjectKey)],
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, CoreError>)
        requires
            old(self).well_formed(),
            peer_idx < old(self).peers_spec()@.len() ==> {
                let peer = old(self).peers_spec()@[peer_idx as int];
                &&& peer.rdma_spec().qp_spec().ready()
                &&& peer.rdma_spec().qp_spec().state() is ReadyToSend
            },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            peer_idx < old(self).peers_spec()@.len()
                ==> final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            peer_idx >= old(self).peers_spec()@.len() ==> r == Err(CoreError::InvalidArgument)
                && *final(self) == *old(self),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].grant_fetch_batch(
            &mut self.core,
            serve_tier,
            entries,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Pump one peer's CQ and protocol frames. Fetch requests remain queued for admission.
    pub fn pump_peer_nonblocking(
        &mut self,
        peer_idx: usize,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            peer_idx < old(self).peers_spec()@.len() ==> {
                let peer = old(self).peers_spec()@[peer_idx as int];
                &&& peer.rdma_spec().qp_spec().ready()
                &&& peer.rdma_spec().qp_spec().state() is ReadyToSend
                &&& (pd matches Some(p) ==> p.id() == peer.rdma_spec().window_pool_spec().pd()
                    && p.node() == old(self).core_spec().node())
            },
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            peer_idx < old(self).peers_spec()@.len()
                ==> final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            peer_idx >= old(self).peers_spec()@.len() ==> r == Err(CoreError::InvalidArgument)
                && *final(self) == *old(self),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if let Some(pd) = pd {
            self.check_peer_pd(peer_idx, pd)?;
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].pump_nonblocking(
            &mut self.core,
            pd,
            tokens,
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Sweep each peer once, leaving fetch admission to the caller.
    /// A supplied PD enables replenishment using each session's own PD.
    pub fn pump_all_peers_nonblocking(
        &mut self,
        replenish: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            forall|i: int|
                0 <= i < old(self).peers_spec()@.len() ==> {
                    let p = #[trigger] old(self).peers_spec()@[i];
                    &&& p.rdma_spec().qp_spec().ready()
                    &&& p.rdma_spec().qp_spec().state() is ReadyToSend
                },
            instance.id() == old(self).core_spec().instance_id(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            forall|i: int|
                0 <= i < final(self).peers_spec()@.len() ==> (
                #[trigger] final(self).peers_spec()@[i]).rdma_spec().qp_spec() == old(
                    self,
                ).peers_spec()@[i].rdma_spec().qp_spec(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        self.check_all_peers_ready()?;

        let mut total = PeerProgress::empty();
        let mut idx: usize = 0;
        while idx < self.peers.len()
            invariant
                idx <= self.peers@.len(),
                self.well_formed(),
                self.host == old(self).host,
                self.device == old(self).device,
                self.nvme == old(self).nvme,
                self.peers@.len() == old(self).peers@.len(),
                self.core.node_id_spec() == old(self).core.node_id_spec(),
                self.core.instance_id() == old(self).core.instance_id(),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& p.rdma.qp == old(self).peers@[i].rdma.qp
                        &&& p.rdma.qp.ready()
                        &&& p.rdma.qp.state() is ReadyToSend
                    },
                instance.id() == self.core.instance_id(),
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source().is_rdma_qp() && tokens@[k]@.node()
                        == self.core.node(),
            decreases self.peers@.len() - idx,
        {
            let pd = if replenish.is_some() {
                Some(self.peers[idx].rdma.qp.protection_domain())
            } else {
                None
            };
            let progress = self.pump_peer_nonblocking(idx, pd, tokens, Tracked(instance))?;
            total.accumulate(progress);
            idx += 1;
        }
        Ok(total)
    }

    /// Non-blocking progress sweep on `peer_idx` serving inbound `FetchRequest`s
    /// from `primary_serve_tier` with optional `fallback_serve_tier`.
    pub fn poll_peer_progress_multi(
        &mut self,
        peer_idx: usize,
        primary_serve_tier: TierId,
        fallback_serve_tier: Option<TierId>,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().ready(),
            old(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            pd matches Some(p) ==> p.id() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().window_pool_spec().pd() && p.node() == old(
                self,
            ).core_spec().node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).peers_spec()@[peer_idx as int].rdma_spec().qp_spec() == old(
                self,
            ).peers_spec()@[peer_idx as int].rdma_spec().qp_spec(),
            forall|j: int|
                0 <= j < old(self).peers_spec()@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers_spec()@[j]) == old(self).peers_spec()@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        self.check_peer_ready(peer_idx)?;
        if primary_serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(tier) = fallback_serve_tier {
            if tier.as_index() >= self.core.tier_count() {
                return Err(CoreError::InvalidArgument);
            }
        }
        if let Some(pd) = pd {
            self.check_peer_pd(peer_idx, pd)?;
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[peer_idx].poll_progress_multi(
            &mut self.core,
            primary_serve_tier,
            fallback_serve_tier,
            pd,
            tokens,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        res
    }

    /// Round-robin non-blocking progress sweep across all attached `PeerSession`s,
    /// serving inbound `FetchRequest`s from `primary_serve_tier` with optional `fallback_serve_tier`.
    /// A supplied PD enables replenishment using each session's own PD.
    pub fn poll_all_peers_progress_multi(
        &mut self,
        primary_serve_tier: TierId,
        fallback_serve_tier: Option<TierId>,
        replenish: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            forall|i: int|
                0 <= i < old(self).peers_spec()@.len() ==> {
                    let p = #[trigger] old(self).peers_spec()@[i];
                    &&& p.rdma_spec().qp_spec().ready()
                    &&& p.rdma_spec().qp_spec().state() is ReadyToSend
                },
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source().is_rdma_qp() && old(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).peers_spec()@.len() == old(self).peers_spec()@.len(),
            forall|i: int|
                0 <= i < final(self).peers_spec()@.len() ==> (
                #[trigger] final(self).peers_spec()@[i]).rdma_spec().qp_spec() == old(
                    self,
                ).peers_spec()@[i].rdma_spec().qp_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source().is_rdma_qp() && final(tokens)@[k]@.node() == old(
                    self,
                ).core_spec().node(),
    {
        hide(CacheCore::well_formed);
        if primary_serve_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(tier) = fallback_serve_tier {
            if tier.as_index() >= self.core.tier_count() {
                return Err(CoreError::InvalidArgument);
            }
        }
        self.check_all_peers_ready()?;

        let mut total = PeerProgress::empty();
        let mut idx: usize = 0;
        while idx < self.peers.len()
            invariant
                idx <= self.peers@.len(),
                self.well_formed(),
                self.host == old(self).host,
                self.device == old(self).device,
                self.nvme == old(self).nvme,
                self.peers@.len() == old(self).peers@.len(),
                self.core.node_id_spec() == old(self).core.node_id_spec(),
                self.core.instance_id() == old(self).core.instance_id(),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& p.rdma.qp == old(self).peers@[i].rdma.qp
                        &&& p.rdma.qp.ready()
                        &&& p.rdma.qp.state() is ReadyToSend
                    },
                instance.id() == self.core.instance_id(),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.node() == old(shares).node(),
                shares.id() == self.core.instance_id(),
                shares.node() == self.core.node(),
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source().is_rdma_qp() && tokens@[k]@.node()
                        == self.core.node(),
            decreases self.peers@.len() - idx,
        {
            let pd = if replenish.is_some() {
                Some(self.peers[idx].rdma.qp.protection_domain())
            } else {
                None
            };
            let p = self.poll_peer_progress_multi(
                idx,
                primary_serve_tier,
                fallback_serve_tier,
                pd,
                tokens,
                Tracked(shares),
                Tracked(instance),
            )?;
            if total.cq_processed <= usize::MAX - p.cq_processed {
                total.cq_processed += p.cq_processed;
            }
            if total.grants_completed <= usize::MAX - p.grants_completed {
                total.grants_completed += p.grants_completed;
            }
            if total.windows_replenished <= usize::MAX - p.windows_replenished {
                total.windows_replenished += p.windows_replenished;
            }
            idx += 1;
        }
        Ok(total)
    }
}

} // verus!
