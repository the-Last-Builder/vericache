//! Verified composable 4-tier N-peer distributed engine (`DistributedEngine<'a, T>`),
//! unifying `CacheCore`, `HostBackend`, `CudaAttachment<'a>` (`None`, `Borrowed`, or
//! `Owned` with async `pending_gpu_bindings`), optional `NvmeBackend` (`io_uring`
//! `O_DIRECT`), and `peers: Vec<PeerSession<'a, T>>` with shared or per-peer MRs,
//! full-duplex single-QP CQ routing, cooperative-first MW revocation, and Option-A
//! QP error/reset/reconnect recovery.
#[cfg(verus_only)]
use crate::backends::compose::*;
#[cfg(verus_only)]
use crate::backends::cuda::{
    caller_stream_valid, cuda_puts_bindings_covered, lemma_cuda_puts_bindings_empty,
    lemma_cuda_puts_bindings_push_abort, lemma_cuda_puts_bindings_push_publish, PendingGpuAction,
};
use crate::backends::cuda::{CudaBackend, CudaPollReport};
use crate::backends::host::HostBackend;
use crate::backends::nvme::{NvmeBackend, NvmeProgress};
use crate::backends::rdma::peer::{
    FetchOutcome, PeerMessage, PeerProgress, PeerSession, ServeOutcome,
};
use crate::backends::rdma::protocol::{ControlPayload, ReadGrantMeta, ReadReleaseMeta};
use crate::backends::rdma::pull::PullTarget;
use crate::backends::*;
#[cfg(verus_only)]
use crate::frame::*;
use crate::kernel::*;
use crate::provenance::ValueCertificate;
#[cfg(verus_only)]
use crate::provenance::{complete_receipts, replica_valid, source_slice_valid};
use crate::spec::*;
use crate::timeline::BITMAP_CAPACITY;
#[cfg(verus_only)]
use foundation::tracked::{knowledge_of, Frontier};
use transfer::context::*;
use transfer::cuda::{CudaSlice, CudaStream};
use transfer::domain::TransferEvent;
#[cfg(verus_only)]
use transfer::domain::{EventSource, NodeId};
use transfer::host::HostContext;
#[cfg(verus_only)]
use transfer::nvme::DIRECT_IO_ALIGNMENT;
use transfer::rdma;
#[cfg(verus_only)]
use transfer::transport::GrantKind;
use transfer::transport::{ByteTransport, ChannelError, GrantTag, Statement};
use vstd::prelude::*;

verus! {

/// GPU backend attachment mode for `DistributedEngine<'a, T>`:
/// - `None`: Host/NVMe/RDMA-only engine without a local GPU tier.
/// - `Borrowed(&'a CudaBackend<'a>)`: shared immutable borrow (compatible with
///   concurrent GPUDirect RDMA MRs borrowing `&'a c.gpu_buffer` when idle).
/// - `Owned(CudaBackend<'a>)`: owned mutable backend supporting non-blocking
///   `submit_put_gpu_async` + `poll_gpu_completions` alongside shared RDMA MRs.
pub enum CudaAttachment<'a> {
    None,
    Borrowed(&'a CudaBackend<'a>),
    Owned(CudaBackend<'a>),
}

impl<'a> CudaAttachment<'a> {
    pub open spec fn spec_ref(&self) -> Option<CudaBackend<'a>> {
        match *self {
            CudaAttachment::None => None,
            CudaAttachment::Borrowed(c) => Some(*c),
            CudaAttachment::Owned(c) => Some(c),
        }
    }

    pub open spec fn same_config(&self, other: &Self) -> bool {
        match (*self, *other) {
            (CudaAttachment::None, CudaAttachment::None) => true,
            (CudaAttachment::Borrowed(a), CudaAttachment::Borrowed(b)) => a == b,
            (CudaAttachment::Owned(a), CudaAttachment::Owned(b)) => {
                &&& a.device == b.device
                &&& a.gpu_tier == b.gpu_tier
                &&& a.gpu_first_frame == b.gpu_first_frame
                &&& a.gpu_num_frames == b.gpu_num_frames
                &&& a.same_pinned_layout(&b)
                &&& a.timeline.timeline_id == b.timeline.timeline_id
            },
            _ => false,
        }
    }

    pub fn from_option_borrowed(opt: Option<&'a CudaBackend<'a>>) -> (r: Self)
        ensures
            match opt {
                None => r is None && r.spec_ref() is None,
                Some(c) => r == CudaAttachment::Borrowed(c) && r.spec_ref() == Some(*c),
            },
    {
        match opt {
            None => CudaAttachment::None,
            Some(c) => CudaAttachment::Borrowed(c),
        }
    }

    pub fn as_ref(&self) -> (r: Option<&CudaBackend<'a>>)
        ensures
            match (self, r) {
                (CudaAttachment::None, None) => self.spec_ref() is None,
                (CudaAttachment::Borrowed(c), Some(rc)) => rc == *c && self.spec_ref() == Some(*rc),
                (CudaAttachment::Owned(c), Some(rc)) => rc == c && self.spec_ref() == Some(*rc),
                _ => false,
            },
    {
        match self {
            CudaAttachment::None => None,
            CudaAttachment::Borrowed(c) => Some(*c),
            CudaAttachment::Owned(c) => Some(c),
        }
    }

    pub fn take(&mut self) -> (r: Self)
        ensures
            r == *old(self),
            *final(self) is None,
    {
        let mut out = CudaAttachment::None;
        core::mem::swap(self, &mut out);
        out
    }
}

/// Pending `READ_GRANT` received from a remote donor, ready to execute a
/// local Host or GPUDirect RDMA pull via `DistributedEngine::pull_with_grant_from`
/// (or `pull_with_grant` for peer `0`).
pub struct PendingReadGrant {
    pub tag: GrantTag,
    pub meta: ReadGrantMeta,
    pub cap: rdma::RemoteCapability,
    pub share: Tracked<ShareCap>,
    pub statement: Tracked<Statement>,
    pub cert: Ghost<ValueCertificate>,
    pub revoked: Tracked<RevokedMap>,
    pub version_binding: Tracked<Option<VersionBinding>>,
}

impl PendingReadGrant {
    pub open spec fn valid_for_peer<T: ByteTransport>(
        &self,
        engine: &DistributedEngine<'_, T>,
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
        &&& self.share@.is_rdma_bound()
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

    pub open spec fn valid<T: ByteTransport>(&self, engine: &DistributedEngine<'_, T>) -> bool {
        self.valid_for_peer(engine, 0)
    }
}

/// High-level event returned by `DistributedEngine::poll_peer_event` / `poll_peer_event_from`.
pub enum DistributedEvent {
    Control(ControlPayload),
    ReadGrant(PendingReadGrant),
    /// Peer cooperatively released a grant; if the grant was still active,
    /// its remote lease has been unpinned and its Type-2 MW invalidated.
    ReadReleased { tag: GrantTag, meta: ReadReleaseMeta, was_active: bool },
    Unexpected,
}

/// Verified composable 4-tier N-peer distributed KV-cache engine unifying
/// `CacheCore`, `HostBackend`, `CudaAttachment<'a>`, optional `NvmeBackend`,
/// and `peers: Vec<PeerSession<'a, T>>`.
pub struct DistributedEngine<'a, T: ByteTransport> {
    pub core: CacheCore,
    pub host: &'a HostBackend,
    pub host_ctx: HostContext,
    pub cuda: CudaAttachment<'a>,
    pub pending_gpu_bindings: Tracked<Map<VersionId, VersionBinding>>,
    pub nvme: Option<NvmeBackend>,
    pub peers: Vec<PeerSession<'a, T>>,
}

pub open spec fn peer_session_compatible<'a, T: ByteTransport>(
    core: &CacheCore,
    host: &HostBackend,
    cuda: &CudaAttachment<'a>,
    nvme: &Option<NvmeBackend>,
    p: &PeerSession<'a, T>,
) -> bool {
    &&& p.well_formed(core)
    &&& p.rdma.timeline_id != host.timeline
    &&& p.rdma.peer_node != core.node()
    &&& p.rdma.qp.peer() == p.rdma.peer_node
    &&& p.rdma.qp.pd_id() == p.rdma.window_pool.pd()
    &&& p.rdma.qp.source().well_formed()
    &&& p.rdma.qp.source().source() is RdmaQueuePair
    &&& p.rdma.qp.source().node() != p.rdma.peer_node
    &&& (cuda.spec_ref() matches Some(c) ==> c.timeline.timeline_id != p.rdma.timeline_id)
    &&& (*nvme matches Some(n) ==> n.timeline != p.rdma.timeline_id)
}

pub open spec fn peers_pairwise_disjoint<'a, T: ByteTransport>(
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

impl<'a, T: ByteTransport> DistributedEngine<'a, T> {
    pub open spec fn well_formed(&self) -> bool {
        &&& self.core.well_formed()
        &&& self.core.all_versions_bound()
        &&& self.host.well_formed(&self.core)
        &&& self.host_ctx.ready(self.core.instance_id(), self.core.node())
        &&& source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger@)
        &&& self.host_ctx.source@.source() != self.host.ctx.source@.source()
        &&& match self.cuda {
            CudaAttachment::None => true,
            CudaAttachment::Borrowed(c) => {
                &&& c.well_formed(&self.core)
                &&& c.active_gpu_puts@.len() == 0
                &&& c.timeline.timeline_id != self.host.timeline
                &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier && h.first_frame
                    == self.host.first_frame && h.num_frames == self.host.num_frames
            },
            CudaAttachment::Owned(c) => {
                &&& c.well_formed(&self.core)
                &&& cuda_puts_bindings_covered(c.active_gpu_puts@, self.pending_gpu_bindings@)
                &&& c.timeline.timeline_id != self.host.timeline
                &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier && h.first_frame
                    == self.host.first_frame && h.num_frames == self.host.num_frames
            },
        }
        &&& match self.nvme {
            None => true,
            Some(n) => {
                &&& n.well_formed(&self.core)
                &&& self.host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                &&& n.timeline != self.host.timeline
                &&& (self.cuda.spec_ref() matches Some(c) ==> n.timeline != c.timeline.timeline_id)
            },
        }
        &&& forall|i: int|
            #![trigger self.peers@[i]]
            0 <= i < self.peers@.len() ==> peer_session_compatible(
                &self.core,
                self.host,
                &self.cuda,
                &self.nvme,
                &self.peers@[i],
            )
        &&& peers_pairwise_disjoint(self.peers@)
    }

    /// Assemble a single-peer `DistributedEngine` from an already-initialized `CacheCore`,
    /// shared `HostBackend`, dedicated `HostContext`, optional borrowed `CudaBackend`,
    /// and `PeerSession<'a, T>` (with `nvme: None`).
    pub fn new(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: HostContext,
        cuda: Option<&'a CudaBackend<'a>>,
        peer: PeerSession<'a, T>,
    ) -> (engine: Self)
        requires
            core.well_formed(),
            core.all_versions_bound(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger@),
            host_ctx.source@.source() != host.ctx.source@.source(),
            peer.well_formed(&core),
            peer.rdma.timeline_id != host.timeline,
            peer.rdma.peer_node != core.node(),
            peer.rdma.qp.peer() == peer.rdma.peer_node,
            peer.rdma.qp.pd_id() == peer.rdma.window_pool.pd(),
            peer.rdma.qp.source().well_formed(),
            peer.rdma.qp.source().source() is RdmaQueuePair,
            peer.rdma.qp.source().node() != peer.rdma.peer_node,
            cuda matches Some(c) ==> {
                &&& c.well_formed(&core)
                &&& c.active_gpu_puts@.len() == 0
                &&& c.timeline.timeline_id != host.timeline
                &&& c.timeline.timeline_id != peer.rdma.timeline_id
                &&& c.pinned_host matches Some(h) ==> h.tier == host.tier && h.first_frame
                    == host.first_frame && h.num_frames == host.num_frames
            },
        ensures
            engine.well_formed(),
            engine.core == core,
            engine.host == host,
            engine.cuda.spec_ref() == match cuda {
                None => None,
                Some(c) => Some(*c),
            },
            engine.nvme is None,
            engine.peers@.len() == 1,
            engine.peers@[0] == peer,
    {
        let cuda_att = CudaAttachment::from_option_borrowed(cuda);
        let mut peers: Vec<PeerSession<'a, T>> = Vec::with_capacity(1);
        peers.push(peer);
        Self {
            core,
            host,
            host_ctx,
            cuda: cuda_att,
            pending_gpu_bindings: Tracked(Map::tracked_empty()),
            nvme: None,
            peers,
        }
    }

    /// Assemble a 4-tier single-peer `DistributedEngine` spanning GPU HBM, Pinned Host DRAM,
    /// optional Local NVMe (`io_uring` `O_DIRECT`), and Peer RDMA / GDR.
    pub fn new_with_nvme(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: HostContext,
        cuda: Option<&'a CudaBackend<'a>>,
        peer: PeerSession<'a, T>,
        nvme: Option<NvmeBackend>,
    ) -> (engine: Self)
        requires
            core.well_formed(),
            core.all_versions_bound(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger@),
            host_ctx.source@.source() != host.ctx.source@.source(),
            peer.well_formed(&core),
            peer.rdma.timeline_id != host.timeline,
            peer.rdma.peer_node != core.node(),
            peer.rdma.qp.peer() == peer.rdma.peer_node,
            peer.rdma.qp.pd_id() == peer.rdma.window_pool.pd(),
            peer.rdma.qp.source().well_formed(),
            peer.rdma.qp.source().source() is RdmaQueuePair,
            peer.rdma.qp.source().node() != peer.rdma.peer_node,
            cuda matches Some(c) ==> {
                &&& c.well_formed(&core)
                &&& c.active_gpu_puts@.len() == 0
                &&& c.timeline.timeline_id != host.timeline
                &&& c.timeline.timeline_id != peer.rdma.timeline_id
                &&& c.pinned_host matches Some(h) ==> h.tier == host.tier && h.first_frame
                    == host.first_frame && h.num_frames == host.num_frames
            },
            nvme matches Some(n) ==> {
                &&& n.well_formed(&core)
                &&& host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                &&& n.timeline != host.timeline
                &&& n.timeline != peer.rdma.timeline_id
                &&& (cuda matches Some(c) ==> n.timeline != c.timeline.timeline_id)
            },
        ensures
            engine.well_formed(),
            engine.core == core,
            engine.host == host,
            engine.cuda.spec_ref() == match cuda {
                None => None,
                Some(c) => Some(*c),
            },
            engine.nvme == nvme,
            engine.peers@.len() == 1,
            engine.peers@[0] == peer,
    {
        let cuda_att = CudaAttachment::from_option_borrowed(cuda);
        let mut peers: Vec<PeerSession<'a, T>> = Vec::with_capacity(1);
        peers.push(peer);
        Self {
            core,
            host,
            host_ctx,
            cuda: cuda_att,
            pending_gpu_bindings: Tracked(Map::tracked_empty()),
            nvme,
            peers,
        }
    }

    /// Assemble an N-peer `DistributedEngine` supporting shared MRs (`attach_shared_host_slab` /
    /// `attach_shared_gpu_slab`) and `CudaAttachment::{None, Borrowed, Owned}`.
    pub fn new_multi_peer(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: HostContext,
        cuda: CudaAttachment<'a>,
        nvme: Option<NvmeBackend>,
        peers: Vec<PeerSession<'a, T>>,
    ) -> (engine: Self)
        requires
            core.well_formed(),
            core.all_versions_bound(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger@),
            host_ctx.source@.source() != host.ctx.source@.source(),
            cuda.spec_ref() matches Some(c) ==> {
                &&& c.well_formed(&core)
                &&& c.active_gpu_puts@.len() == 0
                &&& c.timeline.timeline_id != host.timeline
                &&& c.pinned_host matches Some(h) ==> h.tier == host.tier && h.first_frame
                    == host.first_frame && h.num_frames == host.num_frames
            },
            nvme matches Some(n) ==> {
                &&& n.well_formed(&core)
                &&& host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                &&& n.timeline != host.timeline
                &&& (cuda.spec_ref() matches Some(c) ==> n.timeline != c.timeline.timeline_id)
            },
            forall|i: int|
                #![trigger peers@[i]]
                0 <= i < peers@.len() ==> peer_session_compatible(
                    &core,
                    host,
                    &cuda,
                    &nvme,
                    &peers@[i],
                ),
            peers_pairwise_disjoint(peers@),
        ensures
            engine.well_formed(),
            engine.core == core,
            engine.host == host,
            engine.cuda == cuda,
            engine.nvme == nvme,
            engine.peers == peers,
    {
        let tracked bindings = Map::tracked_empty();
        proof {
            if let CudaAttachment::Owned(ref c) = cuda {
                lemma_cuda_puts_bindings_empty(c.active_gpu_puts@, bindings);
            }
        }
        Self { core, host, host_ctx, cuda, pending_gpu_bindings: Tracked(bindings), nvme, peers }
    }

    /// Assemble an N-peer `DistributedEngine` taking ownership of `CudaBackend<'a>`
    /// for async `submit_put_gpu_async` + `poll_gpu_completions`.
    pub fn new_with_owned_cuda(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: HostContext,
        cuda: CudaBackend<'a>,
        nvme: Option<NvmeBackend>,
        peers: Vec<PeerSession<'a, T>>,
    ) -> (engine: Self)
        requires
            core.well_formed(),
            core.all_versions_bound(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger@),
            host_ctx.source@.source() != host.ctx.source@.source(),
            cuda.well_formed(&core),
            cuda.active_gpu_puts@.len() == 0,
            cuda.timeline.timeline_id != host.timeline,
            cuda.pinned_host matches Some(h) ==> h.tier == host.tier && h.first_frame
                == host.first_frame && h.num_frames == host.num_frames,
            nvme matches Some(n) ==> {
                &&& n.well_formed(&core)
                &&& host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                &&& n.timeline != host.timeline
                &&& n.timeline != cuda.timeline.timeline_id
            },
            forall|i: int|
                #![trigger peers@[i]]
                0 <= i < peers@.len() ==> peer_session_compatible(
                    &core,
                    host,
                    &CudaAttachment::Owned(cuda),
                    &nvme,
                    &peers@[i],
                ),
            peers_pairwise_disjoint(peers@),
        ensures
            engine.well_formed(),
            engine.core == core,
            engine.host == host,
            engine.cuda == CudaAttachment::Owned(cuda),
            engine.nvme == nvme,
            engine.peers == peers,
    {
        Self::new_multi_peer(core, host, host_ctx, CudaAttachment::Owned(cuda), nvme, peers)
    }

    /// Attach an additional freshly-connected `PeerSession<'a, T>` (`active_leases@.len() == 0`)
    /// with distinct `timeline_id` and `peer_node`.
    pub fn add_peer(&mut self, peer: PeerSession<'a, T>)
        requires
            old(self).well_formed(),
            peer_session_compatible(
                &old(self).core,
                old(self).host,
                &old(self).cuda,
                &old(self).nvme,
                &peer,
            ),
            peer.rdma.active_leases@.len() == 0,
            forall|i: int|
                0 <= i < old(self).peers@.len() ==> {
                    let existing = #[trigger] old(self).peers@[i];
                    &&& existing.rdma.timeline_id != peer.rdma.timeline_id
                    &&& existing.rdma.peer_node != peer.rdma.peer_node
                },
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@ == old(self).peers@.push(peer),
    {
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
    }

    pub fn peer_count(&self) -> (n: usize)
        ensures
            n == self.peers@.len(),
    {
        self.peers.len()
    }

    pub fn find_peer_index(&self, peer_node: u64) -> (r: Option<usize>)
        ensures
            match r {
                Some(idx) => (idx as int) < self.peers@.len()
                    && self.peers@[idx as int].rdma.peer_node == peer_node as NodeId,
                None => forall|i: int|
                    0 <= i < self.peers@.len() ==> (#[trigger] self.peers@[i]).rdma.peer_node
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
            t == self.host.tier,
    {
        self.host.tier_id()
    }

    pub fn gpu_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.cuda.spec_ref() {
                Some(c) => Some(c.gpu_tier),
                None => None,
            },
    {
        match self.cuda.as_ref() {
            Some(c) => Some(c.gpu_tier_id()),
            None => None,
        }
    }

    pub fn nvme_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.nvme {
                Some(n) => Some(n.tier),
                None => None,
            },
    {
        match &self.nvme {
            Some(n) => Some(n.tier_id()),
            None => None,
        }
    }

    pub fn gpu_frame_slice(&self, frame: FrameId) -> (s: Option<CudaSlice<'_>>)
        requires
            self.well_formed(),
    {
        match self.cuda.as_ref() {
            Some(c) => {
                if c.owns_gpu_frame(frame) {
                    Some(c.gpu_frame_slice(frame))
                } else {
                    None
                }
            },
            None => None,
        }
    }

    pub fn lookup_tier(&self, key: ObjectKey, tier: TierId) -> (r: Option<(FrameId, VersionId)>)
        requires
            self.well_formed(),
        ensures
            r matches Some((f, version)) ==> {
                &&& tier.valid(self.core.tiers.len())
                &&& f.0 < self.core.frames.metas.len()
                &&& self.core.frames.meta(f).state == STATE_PUBLISHED
                &&& self.core.frames.frame(f).tier == tier
                &&& self.core.frames.frame(f).bound_key == Some(key)
                &&& self.core.frames.frame(f).bound_version == Some(version)
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
                &&& tier.valid(self.core.tiers.len())
                &&& f.0 < self.core.frames.metas.len()
                &&& self.core.frames.meta(f).state == STATE_PUBLISHED
                &&& self.core.frames.frame(f).tier == tier
                &&& self.core.frames.frame(f).bound_key == Some(key)
                &&& self.core.frames.frame(f).bound_version == Some(version)
            },
    {
        self.core.lookup_best(key, priority)
    }

    proof fn lemma_preserve_siblings(
        before: &CacheCore,
        after: &CacheCore,
        touched_timeline: Option<u8>,
        allowed_source: Option<EventSource>,
        mode: FrameTouchMode,
        host: &HostBackend,
        host_ctx: &HostContext,
        cuda: &CudaAttachment<'a>,
        nvme: &Option<NvmeBackend>,
        peers: Seq<PeerSession<'a, T>>,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            core_transition(before, after, touched_timeline, allowed_source, mode),
            host.well_formed(before),
            touched_timeline != Some(host.timeline),
            allowed_source != Some(host.ctx.source@.source()),
            host_ctx.ready(before.instance_id(), before.node()),
            source_fresh_for_ledger(host_ctx.source@, before.ledger@),
            allowed_source != Some(host_ctx.source@.source()),
            forall|i: int|
                0 <= i < peers.len() ==> {
                    &&& (#[trigger] peers[i]).well_formed(before)
                    &&& touched_timeline != Some(peers[i].rdma.timeline_id)
                },
            cuda.spec_ref() matches Some(c) ==> {
                &&& c.well_formed(before)
                &&& touched_timeline != Some(c.timeline.timeline_id)
                &&& allowed_source != Some(c.timeline.stream.source().source())
                &&& (mode is PreserveAllInit || mode is PreserveParkedInit)
            },
            nvme matches Some(n) ==> {
                &&& n.well_formed(before)
                &&& touched_timeline != Some(n.timeline)
                &&& allowed_source != Some(n.ring.source().source())
                &&& (mode is PreserveAllInit || mode is PreserveUnparkedInit)
            },
        ensures
            host.well_formed(after),
            host_ctx.ready(after.instance_id(), after.node()),
            source_fresh_for_ledger(host_ctx.source@, after.ledger@),
            forall|i: int| 0 <= i < peers.len() ==> (#[trigger] peers[i]).well_formed(after),
            cuda.spec_ref() matches Some(c) ==> c.well_formed(after),
            nvme matches Some(n) ==> n.well_formed(after),
    {
        hide(CacheCore::well_formed);
        lemma_core_transition_preserves_host(
            before,
            after,
            touched_timeline,
            allowed_source,
            mode,
            host,
        );
        lemma_core_transition_preserves_source(
            before,
            after,
            touched_timeline,
            allowed_source,
            mode,
            host_ctx.source@,
        );
        assert forall|i: int| 0 <= i < peers.len() implies (#[trigger] peers[i]).well_formed(
            after,
        ) by {
            lemma_core_transition_preserves_peer_session(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                &peers[i],
            );
        }
        match cuda {
            CudaAttachment::None => {},
            CudaAttachment::Borrowed(c) => {
                lemma_core_transition_preserves_cuda(
                    before,
                    after,
                    touched_timeline,
                    allowed_source,
                    mode,
                    *c,
                );
            },
            CudaAttachment::Owned(c) => {
                lemma_core_transition_preserves_cuda(
                    before,
                    after,
                    touched_timeline,
                    allowed_source,
                    mode,
                    c,
                );
            },
        }
        if let Some(ref n) = *nvme {
            lemma_core_transition_preserves_nvme(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                n,
            );
        }
    }

    /// Re-establish `self.host`, `self.host_ctx`, `self.cuda` (both `Borrowed` and
    /// `Owned` with parked async puts), `self.nvme`, and all sibling `self.peers[j]`
    /// (`j != peer_idx`) after `self.peers[peer_idx]` steps with `peer_core_step`
    /// and `peer_foreign_shares_preserved`.
    fn restore_after_peer_step(
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
                    &old(self).cuda,
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
            old(self).peers@[peer_idx as int].rdma.qp.source().source() is RdmaQueuePair,
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
            match old(self).cuda {
                CudaAttachment::None => true,
                CudaAttachment::Borrowed(c) => {
                    &&& c.well_formed(&before)
                    &&& c.active_gpu_puts@.len() == 0
                    &&& c.timeline.timeline_id != old(self).host.timeline
                    &&& c.pinned_host matches Some(h) ==> h.tier == old(self).host.tier
                        && h.first_frame == old(self).host.first_frame && h.num_frames == old(
                        self,
                    ).host.num_frames
                },
                CudaAttachment::Owned(c) => {
                    &&& c.well_formed(&before)
                    &&& cuda_puts_bindings_covered(
                        c.active_gpu_puts@,
                        old(self).pending_gpu_bindings@,
                    )
                    &&& c.timeline.timeline_id != old(self).host.timeline
                    &&& c.pinned_host matches Some(h) ==> h.tier == old(self).host.tier
                        && h.first_frame == old(self).host.first_frame && h.num_frames == old(
                        self,
                    ).host.num_frames
                },
            },
            match old(self).nvme {
                None => true,
                Some(n) => {
                    &&& n.well_formed(&before)
                    &&& old(self).host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                    &&& n.timeline != old(self).host.timeline
                    &&& (old(self).cuda.spec_ref() matches Some(c) ==> n.timeline
                        != c.timeline.timeline_id)
                },
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            final(self).core.spec == old(self).core.spec,
            final(self).core.frames == old(self).core.frames,
            final(self).core.timelines == old(self).core.timelines,
            final(self).core.ledger == old(self).core.ledger,
    {
        hide(CacheCore::well_formed);
        let ghost rdma_tl = self.peers@[peer_idx as int].rdma.timeline_id;
        let ghost before_leases = before_peers[peer_idx as int].rdma.active_leases@;
        let ghost after_leases = self.peers@[peer_idx as int].rdma.active_leases@;
        proof {
            reveal(peer_core_step_except);
            lemma_core_wf_facts(&self.core);
            lemma_core_wf_facts(&before);
            assert(before_peers[peer_idx as int].well_formed(&before));
            assert(self.core.timelines.kinds == before.timelines.kinds);
            assert(self.peers@[peer_idx as int].rdma.peer_node
                == before_peers[peer_idx as int].rdma.peer_node);
            self.host.ctx.source.borrow().lemma_all_future_unsubmitted(
                instance,
                &self.core.ledger.borrow().submitted,
            );
            self.host_ctx.source.borrow().lemma_all_future_unsubmitted(
                instance,
                &self.core.ledger.borrow().submitted,
            );
            lemma_peer_core_step_preserves_host(&before, &self.core, rdma_tl, self.host);
            if let Some(ref n) = self.nvme {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.cuda,
                    &self.nvme,
                    &before_peers[peer_idx as int],
                ));
                lemma_peer_core_step_preserves_nvme(&before, &self.core, rdma_tl, n);
            }
            assert forall|j: int| 0 <= j < self.peers@.len() && j != peer_idx as int implies {
                &&& (#[trigger] self.peers@[j]).well_formed(&self.core)
                &&& peer_leases_disjoint(self.peers@[j].rdma.active_leases@, after_leases)
                &&& peer_leases_disjoint(after_leases, self.peers@[j].rdma.active_leases@)
            } by {
                assert(self.peers@[j] == before_peers[j]);
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.cuda,
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
                &self.cuda,
                &self.nvme,
                &self.peers@[i],
            ) by {
                if i == peer_idx as int {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.cuda,
                        &self.nvme,
                        &before_peers[peer_idx as int],
                    ));
                } else {
                    assert(self.peers@[i] == before_peers[i]);
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.cuda,
                        &self.nvme,
                        &before_peers[i],
                    ));
                    assert(self.peers@[i].well_formed(&self.core));
                }
            }
        }
        match &self.cuda {
            CudaAttachment::None => {},
            CudaAttachment::Borrowed(c) => {
                c.timeline.stream.lemma_source_fresh(
                    Tracked(instance),
                    Tracked(&self.core.ledger.borrow().submitted),
                );
                proof {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.cuda,
                        &self.nvme,
                        &before_peers[peer_idx as int],
                    ));
                    lemma_peer_core_step_preserves_idle_cuda(&before, &self.core, rdma_tl, *c);
                }
            },
            CudaAttachment::Owned(c) => {
                let ghost puts = c.active_gpu_puts@;
                let ghost ctl = c.timeline.timeline_id;
                c.timeline.stream.lemma_source_fresh(
                    Tracked(instance),
                    Tracked(&self.core.ledger.borrow().submitted),
                );
                proof {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.cuda,
                        &self.nvme,
                        &before_peers[peer_idx as int],
                    ));
                    reveal(peer_core_step_except);
                    reveal(FrameTable::same_layout);
                    reveal(FrameTable::preserves_initializing);
                    assert(self.core.frames.consistent(self.core.kernel_view())) by {
                        reveal(CacheCore::well_formed);
                    }
                    lemma_core_wf_facts(&before);
                    assert forall|f: FrameId|
                        f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                            == STATE_INITIALIZING implies self.core.frames.ghosts@.contains_key(
                        f.0 as nat,
                    ) && self.core.frames.ghosts@[f.0 as nat] == before.frames.frame(f) by {
                        assert(f.0 < self.core.frames.metas.len());
                        assert(self.core.frames.frame(f) == before.frames.frame(f));
                    }
                    let ghost pre_core = self.core;
                    lemma_prove_cuda_puts_dest_preserved(
                        &before,
                        self.core.frames.ghosts.borrow_mut(),
                        self.core.ledger.borrow(),
                        instance,
                        puts,
                        ctl,
                        puts.len(),
                    );
                    assert(self.core == pre_core);
                    lemma_peer_core_step_preserves_cuda(
                        &before,
                        &self.core,
                        self.core.version_registry.borrow(),
                        self.pending_gpu_bindings.borrow(),
                        rdma_tl,
                        c,
                    );
                }
            },
        }
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
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r matches Ok((ver, frame)) ==> {
                &&& final(self).host.owns(frame)
                &&& final(self).core.spec@.versions == old(self).core.spec@.versions.insert(
                    ver,
                    data@,
                )
                &&& final(self).core.spec@.catalog == old(self).core.spec@.catalog.insert(key, ver)
                &&& final(self).core.frames.meta(frame).state == STATE_PUBLISHED
            },
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
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
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).well_formed(&self.core) by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.cuda,
                    &self.nvme,
                    &self.peers@[i],
                ));
                lemma_core_transition_preserves_peer_session(
                    &before,
                    &self.core,
                    Some(self.host.timeline),
                    Some(old(self).host_ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                    &self.peers@[i],
                );
            }
            match &self.cuda {
                CudaAttachment::None => {},
                CudaAttachment::Borrowed(c) => {
                    lemma_core_transition_preserves_cuda(
                        &before,
                        &self.core,
                        Some(self.host.timeline),
                        Some(old(self).host_ctx.source@.source()),
                        FrameTouchMode::PreserveAllInit,
                        *c,
                    );
                },
                CudaAttachment::Owned(c) => {
                    lemma_core_transition_preserves_cuda(
                        &before,
                        &self.core,
                        Some(self.host.timeline),
                        Some(old(self).host_ctx.source@.source()),
                        FrameTouchMode::PreserveAllInit,
                        c,
                    );
                },
            }
            if let Some(ref n) = self.nvme {
                lemma_core_transition_preserves_nvme(
                    &before,
                    &self.core,
                    Some(self.host.timeline),
                    Some(old(self).host_ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                    n,
                );
            }
        }
        match res {
            Err(e) => {
                proof {
                    assert(self.core.frames.versions@ == before.frames.versions@) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(self.core.all_versions_bound());
                }
                Err(e)
            },
            Ok((ver, frame)) => {
                let ghost before_bind = self.core;
                proof {
                    assert(before.version_registry@->Some_0.source.min_seq()
                        <= before.next_version_seq as nat) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(self.core.frames.versions@.contains_key(ver)) by {
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
                        None,
                        None,
                        FrameTouchMode::PreserveAllInit,
                    )) by {
                        reveal(core_transition);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                    }
                    Self::lemma_preserve_siblings(
                        &before_bind,
                        &self.core,
                        None,
                        None,
                        FrameTouchMode::PreserveAllInit,
                        self.host,
                        &self.host_ctx,
                        &self.cuda,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.all_versions_bound()) by {
                        assert forall|v: VersionId| #[trigger]
                            self.core.frames.versions@.contains_key(
                                v,
                            ) implies self.core.has_version_binding(v) by {
                            if v != ver {
                                assert(before.frames.versions@.contains_key(v)) by {
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

    /// Synchronously publish a GPU-VRAM frame from caller's `CudaSlice` (`src`)
    /// using caller's idle `src_stream` / `src_timeline`, and bind its cluster-wide
    /// `VersionBinding`. Works with both `CudaAttachment::Borrowed` and `Owned`.
    pub fn put_gpu_sync(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        src_stream: &mut CudaStream,
        src_timeline: u8,
        key: ObjectKey,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(self).well_formed(),
            old(self).cuda.spec_ref() matches Some(c) && {
                &&& caller_stream_valid(
                    *old(src_stream),
                    &old(self).core,
                    src_timeline,
                    c.timeline.stream,
                )
                &&& src_timeline != old(self).host.timeline
                &&& src_timeline != c.timeline.timeline_id
                &&& forall|i: int|
                    0 <= i < old(self).peers@.len() ==> src_timeline != (#[trigger] old(
                        self,
                    ).peers@[i]).rdma.timeline_id
                &&& (old(self).nvme matches Some(n) ==> src_timeline != n.timeline)
                &&& old(self).core.timelines.confirmed_seq[src_timeline as int] + 1 == old(
                    self,
                ).core.timelines.next_seq[src_timeline as int]
                &&& old(src_stream).device() == c.device
                &&& src.slot_ref().loc.space is GpuAlloc
                    && src.slot_ref().loc.space->GpuAlloc_device != c.device ==> (old(
                    src_perm,
                ).frontier().last_writer matches Some(w) ==> !(w.source is RdmaQueuePair))
            },
            instance.id() == old(self).core.instance_id(),
            src.well_formed(),
            src.instance_id() == old(self).core.instance_id(),
            src.slot_ref().loc.node == old(self).core.node(),
            old(self).cuda.spec_ref() matches Some(c) ==> src.slot_ref().sub_len == old(
                self,
            ).core.frames.tier_frame_size(c.gpu_tier),
            old(src_perm).valid_local(old(self).core.instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(self).core.citations()),
            replica_valid(old(self).core.ledger@, old(self).core.node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(self).core.frames.metas.len() && (#[trigger] old(self).core.frames.meta(
                    f,
                )).state == STATE_EMPTY ==> src.slot_ref().loc != old(self).core.frames.frame(
                    f,
                ).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            final(self).cuda.spec_ref() matches Some(c) ==> caller_stream_valid(
                *final(src_stream),
                &final(self).core,
                src_timeline,
                c.timeline.stream,
            ),
            r matches Ok((ver, frame)) ==> {
                &&& final(self).cuda.spec_ref() matches Some(c) && c.owns_gpu(frame)
                &&& final(self).core.timelines.confirmed_seq[src_timeline as int] + 1
                    == final(self).core.timelines.next_seq[src_timeline as int]
                &&& final(self).core.spec@.versions == old(self).core.spec@.versions.insert(
                    ver,
                    src_cert.bytes,
                )
                &&& final(self).core.spec@.catalog == old(self).core.spec@.catalog.insert(key, ver)
                &&& final(self).core.frames.meta(frame).state == STATE_PUBLISHED
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost src_source = src_stream.source().source();
        let res = cuda.put_gpu_sync(
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
                        Some(src_timeline),
                        Some(src_source),
                        FrameTouchMode::PreserveAllInit,
                        self.host,
                        &self.host_ctx,
                        &self.cuda,
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
                    assert(before.version_registry@->Some_0.source.min_seq()
                        <= before.next_version_seq as nat) by {
                        reveal(CacheCore::well_formed);
                    }
                }
                self.core.bind_local_version(ver);
                proof {
                    assert(core_transition(
                        &before,
                        &self.core,
                        Some(src_timeline),
                        Some(src_source),
                        FrameTouchMode::PreserveAllInit,
                    )) by {
                        assert(core_transition(
                            &before,
                            &before_bind,
                            Some(src_timeline),
                            Some(src_source),
                            FrameTouchMode::PreserveAllInit,
                        ));
                        assert(self.core.frames == before_bind.frames);
                        assert(self.core.timelines == before_bind.timelines);
                        assert(self.core.ledger == before_bind.ledger);
                        assert(self.core.active_leases == before_bind.active_leases);
                        reveal(core_transition);
                    }
                    Self::lemma_preserve_siblings(
                        &before,
                        &self.core,
                        Some(src_timeline),
                        Some(src_source),
                        FrameTouchMode::PreserveAllInit,
                        self.host,
                        &self.host_ctx,
                        &self.cuda,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.all_versions_bound()) by {
                        assert forall|v: VersionId| #[trigger]
                            self.core.frames.versions@.contains_key(
                                v,
                            ) implies self.core.has_version_binding(v) by {
                            if v != ver {
                                assert(before.frames.versions@.contains_key(v));
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

    /// Asynchronously enqueue a GPU-VRAM or pinned-host write on `CudaAttachment::Owned`,
    /// pre-binding its `VersionBinding` into `self.pending_gpu_bindings` so concurrent
    /// RDMA / NVMe / Host steps remain composable while the GPU copy is in flight.
    pub fn submit_put_gpu_async(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            old(self).cuda is Owned,
            instance.id() == old(self).core.instance_id(),
            src.well_formed(),
            src.instance_id() == old(self).core.instance_id(),
            src.slot_ref().loc.node == old(self).core.node(),
            old(self).cuda.spec_ref() matches Some(c) ==> {
                &&& src.slot_ref().sub_len == old(self).core.frames.tier_frame_size(c.gpu_tier)
                &&& src.slot_ref().loc.space is GpuAlloc
                    && src.slot_ref().loc.space->GpuAlloc_device != c.device ==> (old(
                    src_perm,
                ).frontier().last_writer matches Some(w) ==> !(w.source is RdmaQueuePair))
            },
            old(src_perm).valid_local(old(self).core.instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(self).core.citations()),
            replica_valid(old(self).core.ledger@, old(self).core.node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(self).core.frames.metas.len() && (#[trigger] old(self).core.frames.meta(
                    f,
                )).state == STATE_EMPTY ==> src.slot_ref().loc != old(self).core.frames.frame(
                    f,
                ).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).cuda is Owned,
            final(self).cuda.same_config(&old(self).cuda),
            final(self).host == old(self).host,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            final(self).core.spec == old(self).core.spec,
            r matches Ok((ver, frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& final(self).cuda.spec_ref() matches Some(c) && ticket.timeline_id()
                    == c.timeline.timeline_id && ((dst_tier == c.gpu_tier && c.owns_gpu(frame))
                    || c.owns_pinned_host(frame))
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let prev_att = self.cuda.take();
        let CudaAttachment::Owned(mut cuda) = prev_att else {
            self.cuda = prev_att;
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost before_puts = cuda.active_gpu_puts@;
        let ghost before_bindings = self.pending_gpu_bindings@;
        let ghost cuda_tl = cuda.timeline.timeline_id;
        let ghost cuda_src = cuda.timeline.stream.source().source();
        let res = cuda.put_gpu(
            &mut self.core,
            Tracked(instance),
            dst_tier,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        );
        match res {
            Err(e) => {
                proof {
                    if cuda.active_gpu_puts@.len() > before_puts.len() {
                        lemma_cuda_puts_bindings_push_abort(
                            before_puts,
                            cuda.active_gpu_puts@,
                            cuda.active_gpu_puts@.last(),
                            before_bindings,
                        );
                    }
                    let ghost none_cuda: CudaAttachment<'a> = CudaAttachment::None;
                    assert forall|i: int| 0 <= i < self.peers@.len() implies (
                    #[trigger] self.peers@[i]).rdma.timeline_id != cuda_tl by {
                        assert(peer_session_compatible(
                            &before,
                            self.host,
                            &CudaAttachment::Owned(cuda),
                            &self.nvme,
                            &self.peers@[i],
                        ) || old(self).peers@[i].rdma.timeline_id != cuda_tl);
                    }
                    Self::lemma_preserve_siblings(
                        &before,
                        &self.core,
                        Some(cuda_tl),
                        Some(cuda_src),
                        FrameTouchMode::PreserveAllInit,
                        self.host,
                        &self.host_ctx,
                        &none_cuda,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.frames.versions@ == before.frames.versions@) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(self.core.all_versions_bound());
                }
                self.cuda = CudaAttachment::Owned(cuda);
                Err(e)
            },
            Ok((ver, frame, ticket, sync_pt)) => {
                let ghost after_put = self.core;
                proof {
                    assert(before.version_registry@->Some_0.source.min_seq()
                        <= before.next_version_seq as nat) by {
                        reveal(CacheCore::well_formed);
                    }
                    assert(self.core.has_version_registry());
                }
                let Tracked(vb) = self.core.prebind_local_version(ver, Ghost(src_cert.bytes));
                proof {
                    self.pending_gpu_bindings.borrow_mut().tracked_insert(ver, vb);
                    lemma_cuda_puts_bindings_push_publish(
                        before_puts,
                        cuda.active_gpu_puts@,
                        cuda.active_gpu_puts@.last(),
                        before_bindings,
                        self.pending_gpu_bindings@,
                        ver,
                        vb,
                        &after_put,
                        cuda_tl,
                    );
                    assert(cuda.well_formed(&self.core)) by {
                        reveal(slab_bound_to_core);
                        assert(self.core.frames == after_put.frames);
                        assert(self.core.timelines == after_put.timelines);
                        assert(self.core.ledger == after_put.ledger);
                        assert(self.core.tiers == after_put.tiers);
                        assert(self.core.node_id == after_put.node_id);
                        assert(self.core.next_version_seq == after_put.next_version_seq);
                        assert(self.core.version_seq_limit == after_put.version_seq_limit);
                        assert(self.core.instance_id() == after_put.instance_id()) by {
                            reveal(CacheCore::well_formed);
                        }
                        reveal(crate::backends::cuda::cuda_puts_valid);
                        assert forall|k: int| 0 <= k < cuda.active_gpu_puts@.len() implies (
                        #[trigger] cuda.active_gpu_puts@[k]).valid(&self.core, cuda_tl) by {
                            assert(cuda.active_gpu_puts@[k].valid(&after_put, cuda_tl));
                        }
                    }
                    assert(core_transition(
                        &before,
                        &self.core,
                        Some(cuda_tl),
                        Some(cuda_src),
                        FrameTouchMode::PreserveAllInit,
                    )) by {
                        assert(core_transition(
                            &before,
                            &after_put,
                            Some(cuda_tl),
                            Some(cuda_src),
                            FrameTouchMode::PreserveAllInit,
                        ));
                        reveal(core_transition);
                    }
                    let ghost none_cuda: CudaAttachment<'a> = CudaAttachment::None;
                    assert forall|i: int| 0 <= i < self.peers@.len() implies (
                    #[trigger] self.peers@[i]).rdma.timeline_id != cuda_tl by {
                        assert(peer_session_compatible(
                            &before,
                            self.host,
                            &old(self).cuda,
                            &self.nvme,
                            &self.peers@[i],
                        ));
                    }
                    Self::lemma_preserve_siblings(
                        &before,
                        &self.core,
                        Some(cuda_tl),
                        Some(cuda_src),
                        FrameTouchMode::PreserveAllInit,
                        self.host,
                        &self.host_ctx,
                        &none_cuda,
                        &self.nvme,
                        self.peers@,
                    );
                    assert(self.core.frames.versions@ == before.frames.versions@) by {
                        assert(after_put.frames.versions@ == before.frames.versions@) by {
                            reveal(CacheCore::well_formed);
                        }
                    }
                    assert(self.core.all_versions_bound());
                    lemma_core_wf_facts(&self.core);
                }
                self.cuda = CudaAttachment::Owned(cuda);
                Ok((ver, frame, ticket, sync_pt))
            },
        }
    }

    /// Non-blockingly poll ready CUDA timeline events and publish/abort completed
    /// parked GPU writes on `CudaAttachment::Owned`, registering their pre-bound
    /// `VersionBinding`s into `self.core.version_registry`.
    pub fn poll_gpu_completions(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<CudaPollReport, (CoreError, CudaPollReport)>)
        requires
            old(self).well_formed(),
            old(self).cuda is Owned,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).cuda is Owned,
            final(self).cuda.same_config(&old(self).cuda),
            final(self).host == old(self).host,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
    {
        hide(CacheCore::well_formed);
        let prev_att = self.cuda.take();
        let CudaAttachment::Owned(mut cuda) = prev_att else {
            self.cuda = prev_att;
            return Err(
                (
                    CoreError::UnsupportedTier,
                    CudaPollReport { events_completed: 0, writes_published: 0, writes_aborted: 0 },
                ),
            );
        };
        let ghost before = self.core;
        let ghost cuda_tl = cuda.timeline.timeline_id;
        let res = cuda.poll_completions_with_bindings(
            &mut self.core,
            Tracked(instance),
            Tracked(self.pending_gpu_bindings.borrow_mut()),
        );
        proof {
            let ghost none_cuda: CudaAttachment<'a> = CudaAttachment::None;
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != cuda_tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &old(self).cuda,
                    &self.nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(cuda_tl),
                None,
                FrameTouchMode::PreserveUnparkedInit,
                self.host,
                &self.host_ctx,
                &none_cuda,
                &self.nvme,
                self.peers@,
            );
        }
        self.cuda = CudaAttachment::Owned(cuda);
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
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            r matches Ok(Some(ver)) ==> final(self).core.spec@.catalog.contains_key(key)
                && final(self).core.spec@.catalog[key] == ver,
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
                    None,
                    Some(old(self).host_ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                    &self.peers@[i],
                );
            }
            match &self.cuda {
                CudaAttachment::None => {},
                CudaAttachment::Borrowed(c) => {
                    lemma_core_transition_preserves_cuda(
                        &before,
                        &self.core,
                        None,
                        Some(old(self).host_ctx.source@.source()),
                        FrameTouchMode::PreserveAllInit,
                        *c,
                    );
                },
                CudaAttachment::Owned(c) => {
                    lemma_core_transition_preserves_cuda(
                        &before,
                        &self.core,
                        None,
                        Some(old(self).host_ctx.source@.source()),
                        FrameTouchMode::PreserveAllInit,
                        c,
                    );
                },
            }
            if let Some(ref n) = self.nvme {
                lemma_core_transition_preserves_nvme(
                    &before,
                    &self.core,
                    None,
                    Some(old(self).host_ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                    n,
                );
            }
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        match res {
            Ok(ver) => Ok(Some(ver)),
            Err(e) => Err(e),
        }
    }

    /// Asynchronously read a published GPU-VRAM object into caller's `CudaSlice`.
    pub fn get_into_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dst: &CudaSlice,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<Option<(VersionId, Ticket, Ghost<TransferEvent>)>>)
        requires
            old(self).well_formed(),
            old(self).cuda.spec_ref() matches Some(c) && {
                &&& caller_stream_valid(
                    *old(dst_stream),
                    &old(self).core,
                    dst_timeline,
                    c.timeline.stream,
                )
                &&& dst_timeline != old(self).host.timeline
                &&& dst_timeline != c.timeline.timeline_id
                &&& forall|i: int|
                    0 <= i < old(self).peers@.len() ==> dst_timeline != (#[trigger] old(
                        self,
                    ).peers@[i]).rdma.timeline_id
                &&& (old(self).nvme matches Some(n) ==> dst_timeline != n.timeline)
                &&& dst.slot_ref().loc.space is GpuAlloc
                    ==> dst.slot_ref().loc.space->GpuAlloc_device == c.device
            },
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core.instance_id(),
            dst.slot_ref().loc.node == old(self).core.node(),
            old(self).cuda.spec_ref() matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core.frames.tier_frame_size(c.gpu_tier),
            slot_is_owned(*old(dst_tok), old(self).core.instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core.citations()),
            old(self).core.disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core.frames.metas.len() ==> dst.slot_ref().loc != (#[trigger] old(
                    self,
                ).core.frames.frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).cuda.spec_ref() matches Some(c) ==> caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
                c.timeline.stream,
            ),
            r matches Ok(Some((version, ticket, ev))) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == dst_timeline
                &&& final(self).core.spec@.versions.contains_key(version)
                &&& slot_owned(
                    *final(dst_tok),
                    old(self).core.instance_id(),
                    dst.slot_ref().loc,
                    Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        let Some((frame, _ver)) = self.core.lookup_tier(key, cuda.gpu_tier) else {
            return Ok(None);
        };
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        let res = cuda.get_into_gpu(
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
                Some(dst_timeline),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
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
    /// Host/CUDA/NVMe/RDMA invariants.
    pub fn evict_frame(&mut self, frame: FrameId) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r.is_ok() ==> final(self).core.frames.meta(frame).state == STATE_DRAINING,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let res = self.core.evict_frame(frame);
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert(core_transition(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
            )) by {
                reveal(core_transition);
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
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
    pub fn reclaim_draining(&mut self, tier: TierId, budget: usize) -> (reclaimed: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.spec == old(self).core.spec,
            reclaimed <= budget,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let reclaimed = self.core.reclaim_draining(tier, budget);
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
                self.peers@,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        reclaimed
    }

    /// Reclaim retired draining frames across every registered tier.
    pub fn reclaim_all_draining(&mut self) -> (total: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers == old(self).peers,
            final(self).core.spec == old(self).core.spec,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let total = self.core.reclaim_all_draining();
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
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
    ) -> (r: Result<(), ChannelError<T::Error>>)
        requires
            old(self).well_formed(),
            (peer_idx as int) < old(self).peers@.len(),
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[peer_idx as int].rdma == old(self).peers@[peer_idx as int].rdma,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
        self.peers[peer_idx].send_control(&self.core, payload, Tracked(instance))
    }

    pub fn send_fetch_request_to(
        &mut self,
        peer_idx: usize,
        key: ObjectKey,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, ChannelError<T::Error>>)
        requires
            old(self).well_formed(),
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].next_request_id < u64::MAX,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[peer_idx as int].rdma == old(self).peers@[peer_idx as int].rdma,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
        self.peers[peer_idx].send_fetch_request(&self.core, key, Tracked(instance))
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            old(self).peers@[peer_idx as int].next_grant_tag < u64::MAX,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).peers@[peer_idx as int].rdma.qp == old(
                self,
            ).peers@[peer_idx as int].rdma.qp,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).peers@[peer_idx as int].rdma.qp == old(
                self,
            ).peers@[peer_idx as int].rdma.qp,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
            r matches Ok(DistributedEvent::ReadGrant(ref g)) ==> g.valid_for_peer(
                final(self),
                peer_idx as int,
            ),
    {
        hide(CacheCore::well_formed);
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
            (peer_idx as int) < old(self).peers@.len(),
            grant.valid_for_peer(old(self), peer_idx as int),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            old(self).core.frames.tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers@[peer_idx as int].rdma.effective_host_mr_spec() matches Some(s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers@[peer_idx as int].rdma.effective_gpu_mr_spec() matches Some(s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[peer_idx as int].rdma.qp.ready(),
            final(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
        hide(CacheCore::well_formed);
        if self.peers[peer_idx].rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        if grant.meta.frame_size != self.core.tier_frame_size(dst_tier) as u64 {
            return Err(CoreError::InvalidArgument);
        }
        let PendingReadGrant { tag, meta, cap, share, statement, cert, revoked, version_binding } =
            grant;
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            old(self).peers@[peer_idx as int].next_request_id < u64::MAX,
            old(self).core.frames.tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers@[peer_idx as int].rdma.effective_host_mr_spec() matches Some(s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers@[peer_idx as int].rdma.effective_gpu_mr_spec() matches Some(s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[peer_idx as int].rdma.qp.ready(),
            final(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
        hide(CacheCore::well_formed);
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            old(self).peers@[peer_idx as int].next_request_id < u64::MAX,
            old(self).core.frames.tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            pd matches Some(p) ==> p.id() == old(self).peers@[peer_idx as int].rdma.window_pool.pd()
                && p.node() == old(self).core.node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source() is RdmaQueuePair && old(tokens)@[k]@.node() == old(
                    self,
                ).core.node(),
            match target {
                PullTarget::Host => old(
                    self,
                ).peers@[peer_idx as int].rdma.effective_host_mr_spec() matches Some(s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(
                    self,
                ).peers@[peer_idx as int].rdma.effective_gpu_mr_spec() matches Some(s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[peer_idx as int].rdma.qp.ready(),
            final(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source() is RdmaQueuePair && final(tokens)@[k]@.node()
                    == old(self).core.node(),
    {
        hide(CacheCore::well_formed);
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            pd matches Some(p) ==> p.id() == old(self).peers@[peer_idx as int].rdma.window_pool.pd()
                && p.node() == old(self).core.node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source() is RdmaQueuePair && old(tokens)@[k]@.node() == old(
                    self,
                ).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).peers@[peer_idx as int].rdma.qp == old(
                self,
            ).peers@[peer_idx as int].rdma.qp,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source() is RdmaQueuePair && final(tokens)@[k]@.node()
                    == old(self).core.node(),
    {
        hide(CacheCore::well_formed);
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
    pub fn poll_all_peers_progress(
        &mut self,
        serve_tier: TierId,
        pd: Option<&'a rdma::ProtectionDomain<'a>>,
        tokens: &mut Vec<Tracked<EventSourceToken>>,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<PeerProgress, CoreError>)
        requires
            old(self).well_formed(),
            forall|i: int|
                0 <= i < old(self).peers@.len() ==> {
                    let p = #[trigger] old(self).peers@[i];
                    &&& p.rdma.qp.ready()
                    &&& p.rdma.qp.state() is ReadyToSend
                    &&& (pd matches Some(pd_ref) ==> pd_ref.id() == p.rdma.window_pool.pd())
                },
            pd matches Some(pd_ref) ==> pd_ref.node() == old(self).core.node(),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            forall|k: int|
                0 <= k < old(tokens)@.len() ==> (#[trigger] old(tokens)@[k])@.well_formed() && old(
                    tokens,
                )@[k]@.source() is RdmaQueuePair && old(tokens)@[k]@.node() == old(
                    self,
                ).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            forall|i: int|
                0 <= i < final(self).peers@.len() ==> (#[trigger] final(self).peers@[i]).rdma.qp
                    == old(self).peers@[i].rdma.qp,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            forall|k: int|
                0 <= k < final(tokens)@.len() ==> (#[trigger] final(tokens)@[k])@.well_formed()
                    && final(tokens)@[k]@.source() is RdmaQueuePair && final(tokens)@[k]@.node()
                    == old(self).core.node(),
    {
        hide(CacheCore::well_formed);
        let mut total = PeerProgress::empty();
        let mut idx: usize = 0;
        while idx < self.peers.len()
            invariant
                idx <= self.peers@.len(),
                self.well_formed(),
                self.host == old(self).host,
                self.cuda == old(self).cuda,
                self.nvme == old(self).nvme,
                self.peers@.len() == old(self).peers@.len(),
                self.core.node_id == old(self).core.node_id,
                self.core.instance_id() == old(self).core.instance_id(),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& p.rdma.qp == old(self).peers@[i].rdma.qp
                        &&& p.rdma.qp.ready()
                        &&& p.rdma.qp.state() is ReadyToSend
                    },
                forall|i: int|
                    idx <= i < self.peers@.len() ==> (pd matches Some(pd_ref) ==> pd_ref.id() == (
                    #[trigger] self.peers@[i]).rdma.window_pool.pd()),
                pd matches Some(pd_ref) ==> pd_ref.node() == self.core.node(),
                instance.id() == self.core.instance_id(),
                shares.well_formed(),
                shares.id() == old(shares).id(),
                shares.node() == old(shares).node(),
                shares.id() == self.core.instance_id(),
                shares.node() == self.core.node(),
                forall|k: int|
                    0 <= k < tokens@.len() ==> (#[trigger] tokens@[k])@.well_formed()
                        && tokens@[k]@.source() is RdmaQueuePair && tokens@[k]@.node()
                        == self.core.node(),
            decreases self.peers@.len() - idx,
        {
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.ready(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).peers@[peer_idx as int].rdma.qp == old(
                self,
            ).peers@[peer_idx as int].rdma.qp,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
        hide(CacheCore::well_formed);
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
                0 <= i < old(self).peers@.len() ==> {
                    let p = #[trigger] old(self).peers@[i];
                    &&& p.rdma.qp.ready()
                    &&& p.rdma.qp.state() is ReadyToSend
                },
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            forall|i: int|
                0 <= i < final(self).peers@.len() ==> (#[trigger] final(self).peers@[i]).rdma.qp
                    == old(self).peers@[i].rdma.qp,
    {
        hide(CacheCore::well_formed);
        let mut total_evicted: usize = 0;
        let mut idx: usize = 0;
        while idx < self.peers.len()
            invariant
                idx <= self.peers@.len(),
                self.well_formed(),
                self.host == old(self).host,
                self.cuda == old(self).cuda,
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
            (peer_idx as int) < old(self).peers@.len(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).peers@[peer_idx as int].rdma.qp == old(
                self,
            ).peers@[peer_idx as int].rdma.qp,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
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
            (peer_idx as int) < old(self).peers@.len(),
            (mw_slot as int) < old(self).peers@[peer_idx as int].rdma.window_pool.slots.len(),
            old(self).peers@[peer_idx as int].rdma.window_pool.slots[mw_slot as int].state
                == crate::backends::rdma::MwSlotState::Vacant,
            pd.id() == old(self).peers@[peer_idx as int].rdma.window_pool.pd(),
            pd.node() == old(self).core.node(),
            name.well_formed(),
            name.source() is RdmaQueuePair,
            name.node() == pd.node(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).peers@[peer_idx as int].rdma.qp == old(
                self,
            ).peers@[peer_idx as int].rdma.qp,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
    {
        self.peers[peer_idx].rdma.replenish_window(&self.core, mw_slot, pd, Tracked(name))
    }

    pub fn enter_peer_qp_error(&mut self, peer_idx: usize) -> (r: BackendResult<
        Tracked<rdma::QpErrorWitness>,
    >)
        requires
            old(self).well_formed(),
            (peer_idx as int) < old(self).peers@.len(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[peer_idx as int].rdma.cq == old(
                self,
            ).peers@[peer_idx as int].rdma.cq,
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
            r.is_ok() ==> final(self).peers@[peer_idx as int].rdma.qp.state()
                == rdma::QueuePairState::Error,
    {
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
            (peer_idx as int) < old(self).peers@.len(),
            old(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend || old(
                self,
            ).peers@[peer_idx as int].rdma.qp.state() is Error,
            psn <= 0xffffff,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            forall|j: int|
                0 <= j < old(self).peers@.len() && j != peer_idx as int ==> (
                #[trigger] final(self).peers@[j]) == old(self).peers@[j],
            r.is_ok() ==> final(self).peers@[peer_idx as int].rdma.qp.ready()
                && final(self).peers@[peer_idx as int].rdma.qp.state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
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
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            r.is_ok() ==> final(self).peers@.len() == 0,
    {
        hide(CacheCore::well_formed);
        while self.peers.len() > 0
            invariant
                self.well_formed(),
                self.host == old(self).host,
                self.cuda == old(self).cuda,
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
            proof {
                reveal(peer_core_step_except);
                lemma_core_wf_facts(&self.core);
                lemma_core_wf_facts(&before);
                self.host.ctx.source.borrow().lemma_all_future_unsubmitted(
                    instance,
                    &self.core.ledger.borrow().submitted,
                );
                self.host_ctx.source.borrow().lemma_all_future_unsubmitted(
                    instance,
                    &self.core.ledger.borrow().submitted,
                );
                lemma_peer_core_step_preserves_host(&before, &self.core, rdma_tl, self.host);
                if let Some(ref n) = self.nvme {
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.cuda,
                        &self.nvme,
                        &pre_pop_peers[last_idx],
                    ));
                    lemma_peer_core_step_preserves_nvme(&before, &self.core, rdma_tl, n);
                }
                assert forall|j: int| 0 <= j < self.peers@.len() implies {
                    &&& (#[trigger] self.peers@[j]).well_formed(&self.core)
                    &&& peer_session_compatible(
                        &self.core,
                        self.host,
                        &self.cuda,
                        &self.nvme,
                        &self.peers@[j],
                    )
                } by {
                    assert(self.peers@[j] == pre_pop_peers[j]);
                    assert(peer_session_compatible(
                        &before,
                        self.host,
                        &self.cuda,
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
            match &self.cuda {
                CudaAttachment::None => {},
                CudaAttachment::Borrowed(c) => {
                    c.timeline.stream.lemma_source_fresh(
                        Tracked(instance),
                        Tracked(&self.core.ledger.borrow().submitted),
                    );
                    proof {
                        assert(peer_session_compatible(
                            &before,
                            self.host,
                            &self.cuda,
                            &self.nvme,
                            &pre_pop_peers[last_idx],
                        ));
                        lemma_peer_core_step_preserves_idle_cuda(&before, &self.core, rdma_tl, *c);
                    }
                },
                CudaAttachment::Owned(c) => {
                    let ghost puts = c.active_gpu_puts@;
                    let ghost ctl = c.timeline.timeline_id;
                    c.timeline.stream.lemma_source_fresh(
                        Tracked(instance),
                        Tracked(&self.core.ledger.borrow().submitted),
                    );
                    proof {
                        assert(peer_session_compatible(
                            &before,
                            self.host,
                            &self.cuda,
                            &self.nvme,
                            &pre_pop_peers[last_idx],
                        ));
                        reveal(peer_core_step_except);
                        reveal(FrameTable::same_layout);
                        reveal(FrameTable::preserves_initializing);
                        assert(self.core.frames.consistent(self.core.kernel_view())) by {
                            reveal(CacheCore::well_formed);
                        }
                        lemma_core_wf_facts(&before);
                        assert forall|f: FrameId|
                            f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(
                                f,
                            )).state
                                == STATE_INITIALIZING implies self.core.frames.ghosts@.contains_key(
                            f.0 as nat,
                        ) && self.core.frames.ghosts@[f.0 as nat] == before.frames.frame(f) by {
                            assert(f.0 < self.core.frames.metas.len());
                            assert(self.core.frames.frame(f) == before.frames.frame(f));
                        }
                        let ghost pre_core = self.core;
                        lemma_prove_cuda_puts_dest_preserved(
                            &before,
                            self.core.frames.ghosts.borrow_mut(),
                            self.core.ledger.borrow(),
                            instance,
                            puts,
                            ctl,
                            puts.len(),
                        );
                        assert(self.core == pre_core);
                        lemma_peer_core_step_preserves_cuda(
                            &before,
                            &self.core,
                            self.core.version_registry.borrow(),
                            self.pending_gpu_bindings.borrow(),
                            rdma_tl,
                            c,
                        );
                    }
                },
            }
            res?;
        }
        Ok(())
    }

    // =========================================================================
    // Single-Peer Convenience Wrappers (Peer 0)
    // =========================================================================
    pub fn send_control(
        &mut self,
        payload: ControlPayload,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), ChannelError<T::Error>>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma == old(self).peers@[0].rdma,
    {
        self.send_control_to(0, payload, Tracked(instance))
    }

    pub fn send_fetch_request(
        &mut self,
        key: ObjectKey,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, ChannelError<T::Error>>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            old(self).peers@[0].next_request_id < u64::MAX,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma == old(self).peers@[0].rdma,
    {
        self.send_fetch_request_to(0, key, Tracked(instance))
    }

    pub fn serve_fetch_request(
        &mut self,
        tier: TierId,
        request_id: u64,
        key: ObjectKey,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<ServeOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            old(self).peers@[0].rdma.qp.ready(),
            old(self).peers@[0].rdma.qp.state() is ReadyToSend,
            old(self).peers@[0].next_grant_tag < u64::MAX,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp == old(self).peers@[0].rdma.qp,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        self.serve_fetch_request_for_peer(
            0,
            tier,
            request_id,
            key,
            Tracked(shares),
            Tracked(instance),
        )
    }

    pub fn poll_peer_event(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        DistributedEvent,
        CoreError,
    >)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            old(self).peers@[0].rdma.qp.ready(),
            old(self).peers@[0].rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp == old(self).peers@[0].rdma.qp,
            r matches Ok(DistributedEvent::ReadGrant(ref g)) ==> g.valid(final(self)),
    {
        self.poll_peer_event_from(0, Tracked(instance))
    }

    pub fn pull_with_grant(
        &mut self,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        grant: PendingReadGrant,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            grant.valid(old(self)),
            old(self).peers@[0].rdma.qp.ready(),
            old(self).peers@[0].rdma.qp.state() is ReadyToSend,
            old(self).core.frames.tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            match target {
                PullTarget::Host => old(self).peers@[0].rdma.effective_host_mr_spec() matches Some(
                    s,
                ) && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).peers@[0].rdma.effective_gpu_mr_spec() matches Some(
                    s,
                ) && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp.ready(),
            final(self).peers@[0].rdma.qp.state() is ReadyToSend,
    {
        self.pull_with_grant_from(0, key, dst_tier, target, grant, Tracked(instance))
    }

    pub fn fetch_remote_frame(
        &mut self,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            old(self).peers@[0].rdma.qp.ready(),
            old(self).peers@[0].rdma.qp.state() is ReadyToSend,
            old(self).peers@[0].next_request_id < u64::MAX,
            old(self).core.frames.tier_frame_size(dst_tier) as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            match target {
                PullTarget::Host => old(self).peers@[0].rdma.effective_host_mr_spec() matches Some(
                    s,
                ) && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).peers@[0].rdma.effective_gpu_mr_spec() matches Some(
                    s,
                ) && s.tier == dst_tier && s.mr.access().local_write
                    && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp.ready(),
            final(self).peers@[0].rdma.qp.state() is ReadyToSend,
    {
        self.fetch_from_peer(0, key, dst_tier, target, Tracked(instance))
    }

    pub fn revoke_lease_unilateral(
        &mut self,
        grant_tag: GrantTag,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u32, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            old(self).peers@[0].rdma.qp.ready(),
            old(self).peers@[0].rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp == old(self).peers@[0].rdma.qp,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        let res = self.peers[0].revoke_lease_unilateral(
            &mut self.core,
            &mut self.host_ctx,
            grant_tag,
            Tracked(instance),
        );
        self.restore_after_peer_step(0, Ghost(before), Ghost(before_peers), Tracked(instance));
        res
    }

    pub fn acknowledge_quarantined_window(&mut self, mw_slot: u32) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp == old(self).peers@[0].rdma.qp,
    {
        self.acknowledge_peer_quarantined_window(0, mw_slot)
    }

    pub fn replenish_window(
        &mut self,
        mw_slot: u32,
        pd: &'a rdma::ProtectionDomain<'a>,
        Tracked(name): Tracked<EventSourceToken>,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            (mw_slot as int) < old(self).peers@[0].rdma.window_pool.slots.len(),
            old(self).peers@[0].rdma.window_pool.slots[mw_slot as int].state
                == crate::backends::rdma::MwSlotState::Vacant,
            pd.id() == old(self).peers@[0].rdma.window_pool.pd(),
            pd.node() == old(self).core.node(),
            name.well_formed(),
            name.source() is RdmaQueuePair,
            name.node() == pd.node(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.qp == old(self).peers@[0].rdma.qp,
    {
        self.replenish_peer_window(0, mw_slot, pd, Tracked(name))
    }

    pub fn enter_qp_error(&mut self) -> (r: BackendResult<Tracked<rdma::QpErrorWitness>>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            final(self).peers@[0].rdma.cq == old(self).peers@[0].rdma.cq,
            r.is_ok() ==> final(self).peers@[0].rdma.qp.state() == rdma::QueuePairState::Error,
    {
        self.enter_peer_qp_error(0)
    }

    pub fn reset_and_reconnect(
        &mut self,
        psn: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            old(self).peers@.len() >= 1,
            old(self).peers@[0].rdma.qp.state() is ReadyToSend || old(
                self,
            ).peers@[0].rdma.qp.state() is Error,
            psn <= 0xffffff,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peers@.len() == old(self).peers@.len(),
            r.is_ok() ==> final(self).peers@[0].rdma.qp.ready()
                && final(self).peers@[0].rdma.qp.state() is ReadyToSend,
    {
        self.reset_and_reconnect_peer(0, psn, Tracked(instance))
    }

    // =========================================================================
    // 4-Tier Local NVMe (`io_uring` `O_DIRECT`) Lifecycle & Data-Path Methods
    // =========================================================================
    pub fn spill_host_to_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (final(self).nvme matches Some(
                n,
            ) && ticket.timeline_id() == n.timeline) && final(self).core.timelines.valid_ticket(
                ticket,
            ),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme matches Some(n) && ticket.timeline_id() == n.timeline)
                && final(self).core.timelines.valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Err((CoreError::Miss, None));
        };
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, None));
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
                    &self.cuda,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames.versions@ == before.frames.versions@) by {
                reveal(CacheCore::well_formed);
            }
            assert(self.core.all_versions_bound());
        }
        let (dst_frame, ticket) = match staged {
            Ok(pair) => pair,
            Err(e) => {
                self.nvme = Some(nvme);
                return Err((e, None));
            },
        };
        let submit_res = nvme.submit(Ghost(&self.core));
        let rem_staged = nvme.staged_len();
        self.nvme = Some(nvme);
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
    }

    pub fn warm_host_from_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (final(self).nvme matches Some(
                n,
            ) && ticket.timeline_id() == n.timeline) && final(self).core.timelines.valid_ticket(
                ticket,
            ),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme matches Some(n) && ticket.timeline_id() == n.timeline)
                && final(self).core.timelines.valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        let Some(nvme_tier) = self.nvme_tier() else {
            return Err((CoreError::UnsupportedTier, None));
        };
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, nvme_tier) else {
            return Err((CoreError::Miss, None));
        };
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, None));
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
                    &self.cuda,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames.versions@ == before.frames.versions@) by {
                reveal(CacheCore::well_formed);
            }
            assert(self.core.all_versions_bound());
        }
        let (dst_frame, ticket) = match staged {
            Ok(pair) => pair,
            Err(e) => {
                self.nvme = Some(nvme);
                return Err((e, None));
            },
        };
        let submit_res = nvme.submit(Ghost(&self.core));
        let rem_staged = nvme.staged_len();
        self.nvme = Some(nvme);
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
    }

    pub fn poll_nvme_completions(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        budget: usize,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r matches Ok(p) ==> p.completed + p.failed <= budget,
    {
        hide(CacheCore::well_formed);
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, NvmeProgress::empty()));
        };
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let res = nvme.poll_completions_up_to(&mut self.core, Tracked(instance), budget);
        proof {
            let ghost none_nvme: Option<NvmeBackend> = None;
            assert forall|i: int| 0 <= i < self.peers@.len() implies (
            #[trigger] self.peers@[i]).rdma.timeline_id != tl by {
                assert(peer_session_compatible(
                    &before,
                    self.host,
                    &self.cuda,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                None,
                FrameTouchMode::PreserveParkedInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames.versions@ == before.frames.versions@) by {
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
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
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
                    &self.cuda,
                    &old(self).nvme,
                    &self.peers@[i],
                ));
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                None,
                FrameTouchMode::PreserveParkedInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &none_nvme,
                self.peers@,
            );
            assert(self.core.frames.versions@ == before.frames.versions@) by {
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
            instance.id() == old(self).core.instance_id(),
            old(self).core.timelines.valid_ticket(ticket),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r.is_ok() ==> final(self).core.timelines.retired(ticket),
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
        if ticket.timeline() != nvme.timeline {
            self.nvme = Some(nvme);
            return Err((CoreError::InvalidArgument, progress));
        }
        while steps < BITMAP_CAPACITY as usize && (nvme.in_flight_len() > 0 || nvme.staged_len()
            > 0) && !self.core.timelines.is_retired(ticket)
            invariant
                steps <= BITMAP_CAPACITY as usize,
                progress.completed + progress.failed == steps,
                self.core.well_formed(),
                self.core.all_versions_bound(),
                self.core.timelines.bounds(),
                self.host.well_formed(&self.core),
                self.host_ctx.ready(self.core.instance_id(), self.core.node()),
                source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger@),
                self.host_ctx.source@.source() != self.host.ctx.source@.source(),
                self.host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0,
                self.peers == old(self).peers,
                peers_pairwise_disjoint(self.peers@),
                forall|i: int|
                    0 <= i < self.peers@.len() ==> {
                        let p = #[trigger] self.peers@[i];
                        &&& peer_session_compatible(&self.core, self.host, &self.cuda, &None, &p)
                        &&& tl != p.rdma.timeline_id
                    },
                nvme.well_formed(&self.core),
                nvme.timeline == tl,
                nvme.timeline != self.host.timeline,
                self.cuda == old(self).cuda,
                self.pending_gpu_bindings == old(self).pending_gpu_bindings,
                match self.cuda {
                    CudaAttachment::None => true,
                    CudaAttachment::Borrowed(c) => {
                        &&& c.well_formed(&self.core)
                        &&& c.active_gpu_puts@.len() == 0
                        &&& c.timeline.timeline_id != self.host.timeline
                        &&& nvme.timeline != c.timeline.timeline_id
                        &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier
                            && h.first_frame == self.host.first_frame && h.num_frames
                            == self.host.num_frames
                    },
                    CudaAttachment::Owned(c) => {
                        &&& c.well_formed(&self.core)
                        &&& cuda_puts_bindings_covered(
                            c.active_gpu_puts@,
                            self.pending_gpu_bindings@,
                        )
                        &&& c.timeline.timeline_id != self.host.timeline
                        &&& nvme.timeline != c.timeline.timeline_id
                        &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier
                            && h.first_frame == self.host.first_frame && h.num_frames
                            == self.host.num_frames
                    },
                },
                self.nvme is None,
                self.host == old(self).host,
                self.core.node_id == old(self).core.node_id,
                self.core.instance_id() == old(self).core.instance_id(),
                self.core.tiers == old(self).core.tiers,
                instance.id() == self.core.instance_id(),
                self.core.timelines.valid_ticket(ticket),
                old(self).core.spec@.versions_monotonic(self.core.spec@),
            decreases (BITMAP_CAPACITY as usize) - steps,
        {
            if nvme.staged_len() > 0 && !self.core.timelines.is_retired(ticket) {
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
            let step_res = nvme.wait_one_completion(&mut self.core, Tracked(instance), progress);
            proof {
                let ghost none_nvme: Option<NvmeBackend> = None;
                Self::lemma_preserve_siblings(
                    &before_step,
                    &self.core,
                    Some(tl),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                    self.host,
                    &self.host_ctx,
                    &self.cuda,
                    &none_nvme,
                    self.peers@,
                );
                lemma_versions_monotonic_trans(
                    old(self).core.spec@,
                    before_step.spec@,
                    self.core.spec@,
                );
                assert(self.core.frames.versions@ == before_step.frames.versions@) by {
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
        if self.core.timelines.is_retired(ticket) {
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
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
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
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peers == old(self).peers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
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

} // verus!
