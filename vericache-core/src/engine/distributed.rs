//! Verified composable 4-tier distributed engine (`DistributedEngine<'a, T>`),
//! unifying `CacheCore`, `HostBackend`, optional `CudaBackend`, optional
//! `NvmeBackend` (`io_uring` `O_DIRECT`), `RdmaBackend`, and `PeerSession<'a, T>`
//! for GPU HBM + Pinned Host DRAM + Local NVMe + 2-node Host/GPUDirect RDMA
//! serving, pulling, unilateral revocation, and Option-A QP recovery.
#[cfg(verus_only)]
use crate::backends::compose::*;
#[cfg(verus_only)]
use crate::backends::cuda::caller_stream_valid;
use crate::backends::cuda::CudaBackend;
use crate::backends::host::HostBackend;
use crate::backends::nvme::{NvmeBackend, NvmeProgress};
use crate::backends::rdma::peer::{FetchOutcome, PeerMessage, PeerSession, ServeOutcome};
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
use transfer::cuda::{self, CudaSlice, CudaStream};
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

/// Pending `READ_GRANT` received from a remote donor, ready to execute a
/// local Host or GPUDirect RDMA pull via `DistributedEngine::pull_with_grant`.
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
    pub open spec fn valid<T: ByteTransport>(&self, engine: &DistributedEngine<'_, T>) -> bool {
        &&& engine.peer.channel.received().contains_key(self.tag)
        &&& engine.peer.channel.received()[self.tag] == (GrantKind::ReadShare {
            slot: self.cap.slot_ref(),
            share_id: self.share@.share(),
        })
        &&& self.statement@.valid(engine.core.instance_id(), engine.core.node())
        &&& self.statement@.knows.set() == knowledge_of(
            engine.core.node(),
            self.statement@.closure(),
        )
        &&& self.cap.well_formed()
        &&& self.cap.slot_ref().loc.node == engine.peer.rdma.peer_node
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
}

/// High-level event returned by `DistributedEngine::poll_peer_event`.
pub enum DistributedEvent {
    Control(ControlPayload),
    ReadGrant(PendingReadGrant),
    /// Peer cooperatively released a grant; if the grant was still active,
    /// its remote lease has been unpinned and its Type-2 MW invalidated.
    ReadReleased { tag: GrantTag, meta: ReadReleaseMeta, was_active: bool },
    Unexpected,
}

/// Verified composable 4-tier distributed KV-cache engine unifying `CacheCore`,
/// `HostBackend`, optional `CudaBackend`, optional `NvmeBackend`, `RdmaBackend`,
/// and `PeerSession`.
pub struct DistributedEngine<'a, T: ByteTransport> {
    pub core: CacheCore,
    pub host: &'a HostBackend,
    pub host_ctx: HostContext,
    pub cuda: Option<&'a CudaBackend<'a>>,
    pub nvme: Option<NvmeBackend>,
    pub peer: PeerSession<'a, T>,
}

impl<'a, T: ByteTransport> DistributedEngine<'a, T> {
    pub open spec fn well_formed(&self) -> bool {
        &&& self.core.well_formed()
        &&& self.core.all_versions_bound()
        &&& self.host.well_formed(&self.core)
        &&& self.host_ctx.ready(self.core.instance_id(), self.core.node())
        &&& source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger@)
        &&& self.host_ctx.source@.source() != self.host.ctx.source@.source()
        &&& self.peer.well_formed(&self.core)
        &&& self.peer.rdma.timeline_id != self.host.timeline
        &&& self.peer.rdma.peer_node != self.core.node()
        &&& self.peer.rdma.qp.peer() == self.peer.rdma.peer_node
        &&& self.peer.rdma.qp.pd_id() == self.peer.rdma.window_pool.pd()
        &&& self.peer.rdma.qp.source().well_formed()
        &&& self.peer.rdma.qp.source().source() is RdmaQueuePair
        &&& self.peer.rdma.qp.source().node() != self.peer.rdma.peer_node
        &&& match self.cuda {
            None => true,
            Some(c) => {
                &&& c.well_formed(&self.core)
                &&& c.active_gpu_puts@.len() == 0
                &&& c.timeline.timeline_id != self.host.timeline
                &&& c.timeline.timeline_id != self.peer.rdma.timeline_id
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
                &&& n.timeline != self.peer.rdma.timeline_id
                &&& match self.cuda {
                    None => true,
                    Some(c) => n.timeline != c.timeline.timeline_id,
                }
            },
        }
    }

    /// Assemble a `DistributedEngine` from an already-initialized `CacheCore`,
    /// shared `HostBackend`, dedicated `HostContext`, optional shared
    /// `CudaBackend`, and `PeerSession<'a, T>` (with `nvme: None`).
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
            engine.cuda == cuda,
            engine.nvme is None,
            engine.peer == peer,
    {
        Self { core, host, host_ctx, cuda, nvme: None, peer }
    }

    /// Assemble a 4-tier `DistributedEngine` spanning GPU HBM, Pinned Host DRAM,
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
            engine.cuda == cuda,
            engine.nvme == nvme,
            engine.peer == peer,
    {
        Self { core, host, host_ctx, cuda, nvme, peer }
    }

    pub fn host_tier(&self) -> (t: TierId)
        ensures
            t == self.host.tier,
    {
        self.host.tier_id()
    }

    pub fn gpu_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.cuda {
                Some(c) => Some(c.gpu_tier),
                None => None,
            },
    {
        match self.cuda {
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

    pub fn gpu_frame_slice(&self, frame: FrameId) -> (s: Option<CudaSlice<'a>>)
        requires
            self.well_formed(),
    {
        match self.cuda {
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
        cuda: &Option<&'a CudaBackend<'a>>,
        nvme: &Option<NvmeBackend>,
        peer: &PeerSession<'a, T>,
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
            peer.well_formed(before),
            touched_timeline != Some(peer.rdma.timeline_id),
            cuda matches Some(c) ==> {
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
            peer.well_formed(after),
            cuda matches Some(c) ==> c.well_formed(after),
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
        lemma_core_transition_preserves_peer_session(
            before,
            after,
            touched_timeline,
            allowed_source,
            mode,
            peer,
        );
        if let Some(c) = *cuda {
            lemma_core_transition_preserves_cuda(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                c,
            );
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

    /// Re-establish sibling `host`, `host_ctx`, `nvme`, and idle `cuda`
    /// invariants after any `PeerSession` / `RdmaBackend` mutation summarized
    /// by `peer_core_step`.
    fn restore_local_after_peer_step(
        &self,
        Ghost(before): Ghost<CacheCore>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            before.well_formed(),
            self.core.well_formed(),
            peer_core_step(&before, &self.core, self.peer.rdma.timeline_id),
            instance.id() == self.core.instance_id(),
            self.host.well_formed(&before),
            self.host.timeline != self.peer.rdma.timeline_id,
            self.host_ctx.ready(self.core.instance_id(), self.core.node()),
            self.cuda matches Some(c) ==> {
                &&& c.well_formed(&before)
                &&& c.active_gpu_puts@.len() == 0
                &&& c.timeline.timeline_id != self.peer.rdma.timeline_id
            },
            self.nvme matches Some(n) ==> {
                &&& n.well_formed(&before)
                &&& n.timeline != self.peer.rdma.timeline_id
            },
        ensures
            self.host.well_formed(&self.core),
            source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger@),
            self.cuda matches Some(c) ==> c.well_formed(&self.core),
            self.nvme matches Some(n) ==> n.well_formed(&self.core),
    {
        hide(CacheCore::well_formed);
        proof {
            reveal(peer_core_step_except);
            lemma_core_wf_facts(&self.core);
            self.host.ctx.source.borrow().lemma_all_future_unsubmitted(
                instance,
                &self.core.ledger.borrow().submitted,
            );
            self.host_ctx.source.borrow().lemma_all_future_unsubmitted(
                instance,
                &self.core.ledger.borrow().submitted,
            );
            lemma_peer_core_step_preserves_host(
                &before,
                &self.core,
                self.peer.rdma.timeline_id,
                self.host,
            );
            if let Some(ref n) = self.nvme {
                lemma_peer_core_step_preserves_nvme(
                    &before,
                    &self.core,
                    self.peer.rdma.timeline_id,
                    n,
                );
            }
        }
        if let Some(c) = self.cuda {
            c.timeline.stream.lemma_source_fresh(
                Tracked(instance),
                Tracked(&self.core.ledger.borrow().submitted),
            );
            proof {
                lemma_peer_core_step_preserves_idle_cuda(
                    &before,
                    &self.core,
                    self.peer.rdma.timeline_id,
                    c,
                );
            }
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
            final(self).peer == old(self).peer,
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
            lemma_core_transition_preserves_peer_session(
                &before,
                &self.core,
                Some(self.host.timeline),
                Some(old(self).host_ctx.source@.source()),
                FrameTouchMode::PreserveAllInit,
                &self.peer,
            );
            if let Some(c) = self.cuda {
                lemma_core_transition_preserves_cuda(
                    &before,
                    &self.core,
                    Some(self.host.timeline),
                    Some(old(self).host_ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                    c,
                );
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
                        reveal(FrameTable::preserves_remote_leases);
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
                        &self.peer,
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

    /// Synchronously publish a GPU-VRAM frame from caller's `CudaSlice` (`src`,
    /// e.g. a pinned `CudaHostBuffer` slice) using caller's idle `src_stream` /
    /// `src_timeline`, and bind its cluster-wide `VersionBinding`.
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
            old(self).cuda matches Some(c) && {
                &&& caller_stream_valid(
                    *old(src_stream),
                    &old(self).core,
                    src_timeline,
                    c.timeline.stream,
                )
                &&& src_timeline != old(self).host.timeline
                &&& src_timeline != c.timeline.timeline_id
                &&& src_timeline != old(self).peer.rdma.timeline_id
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
            src.slot_ref().sub_len == old(self).core.frames.frame_size,
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
            final(self).peer == old(self).peer,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            final(self).cuda matches Some(c) ==> caller_stream_valid(
                *final(src_stream),
                &final(self).core,
                src_timeline,
                c.timeline.stream,
            ),
            r matches Ok((ver, frame)) ==> {
                &&& final(self).cuda matches Some(c) && c.owns_gpu(frame)
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
        let Some(cuda) = self.cuda else {
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
                        &self.peer,
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
                        &self.peer,
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
            final(self).peer == old(self).peer,
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
            lemma_core_transition_preserves_peer_session(
                &before,
                &self.core,
                None,
                Some(old(self).host_ctx.source@.source()),
                FrameTouchMode::PreserveAllInit,
                &self.peer,
            );
            if let Some(c) = self.cuda {
                lemma_core_transition_preserves_cuda(
                    &before,
                    &self.core,
                    None,
                    Some(old(self).host_ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                    c,
                );
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

    /// Asynchronously read a published GPU-VRAM object into caller's `CudaSlice`
    /// (`dst`, e.g. GPU buffer or pinned `CudaHostBuffer` slice for D2H readout).
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
            old(self).cuda matches Some(c) && {
                &&& caller_stream_valid(
                    *old(dst_stream),
                    &old(self).core,
                    dst_timeline,
                    c.timeline.stream,
                )
                &&& dst_timeline != old(self).host.timeline
                &&& dst_timeline != c.timeline.timeline_id
                &&& dst_timeline != old(self).peer.rdma.timeline_id
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
            dst.slot_ref().sub_len == old(self).core.frames.frame_size,
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
            final(self).peer == old(self).peer,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).cuda matches Some(c) ==> caller_stream_valid(
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
        let Some(cuda) = self.cuda else {
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
            lemma_core_transition_preserves_host(
                &before,
                &self.core,
                Some(dst_timeline),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
            );
            lemma_core_transition_preserves_source(
                &before,
                &self.core,
                Some(dst_timeline),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host_ctx.source@,
            );
            lemma_core_transition_preserves_peer_session(
                &before,
                &self.core,
                Some(dst_timeline),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                &self.peer,
            );
            if let Some(ref n) = self.nvme {
                lemma_core_transition_preserves_nvme(
                    &before,
                    &self.core,
                    Some(dst_timeline),
                    Some(src_ev),
                    FrameTouchMode::PreserveAllInit,
                    n,
                );
            }
            assert(self.core.all_versions_bound());
        }
        match res {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        }
    }

    /// Evict a published frame (`Published -> Draining`), preserving sibling
    /// Host/CUDA/RDMA invariants.
    pub fn evict_frame(&mut self, frame: FrameId) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer == old(self).peer,
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
                &self.peer,
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
            final(self).peer == old(self).peer,
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
                &self.peer,
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
            final(self).peer == old(self).peer,
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
                &self.peer,
            );
            assert(self.core.all_versions_bound()) by {
                reveal(FrameTable::same_directory);
            }
        }
        total
    }

    /// Send a `ControlPayload` notice to the remote peer.
    pub fn send_control(
        &mut self,
        payload: ControlPayload,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), ChannelError<T::Error>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma == old(self).peer.rdma,
    {
        self.peer.send_control(&self.core, payload, Tracked(instance))
    }

    /// Initiator: send a `ControlPayload::FetchRequest` for `key` and return
    /// the allocated `request_id`.
    pub fn send_fetch_request(
        &mut self,
        key: ObjectKey,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u64, ChannelError<T::Error>>)
        requires
            old(self).well_formed(),
            old(self).peer.next_request_id < u64::MAX,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma == old(self).peer.rdma,
    {
        self.peer.send_fetch_request(&self.core, key, Tracked(instance))
    }

    /// Donor: serve a `FetchRequest { request_id, key }` from `tier` by binding
    /// a Type-2 MW and sending `READ_GRANT` (or `FetchMiss`).
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
            old(self).peer.rdma.qp.ready(),
            old(self).peer.rdma.qp.state() is ReadyToSend,
            old(self).peer.next_grant_tag < u64::MAX,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.qp == old(self).peer.rdma.qp,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let res = self.peer.serve_fetch_request(
            &mut self.core,
            tier,
            request_id,
            key,
            Tracked(shares),
            Tracked(instance),
        );
        self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
        res
    }

    /// Receive and dispatch the next peer message (`Control`, `ReadGrant`, or
    /// cooperative `ReadRelease` handled immediately via `handle_read_release`).
    pub fn poll_peer_event(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        DistributedEvent,
        CoreError,
    >)
        requires
            old(self).well_formed(),
            old(self).peer.rdma.qp.ready(),
            old(self).peer.rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.qp == old(self).peer.rdma.qp,
            r matches Ok(DistributedEvent::ReadGrant(ref g)) ==> g.valid(final(self)),
    {
        hide(CacheCore::well_formed);
        let msg = match self.peer.recv_message(&self.core, Tracked(instance)) {
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
                let res = self.peer.handle_read_release(
                    &mut self.core,
                    tag,
                    share,
                    statement,
                    Tracked(instance),
                );
                self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
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

    /// Initiator: execute an RDMA READ pull (`PullTarget::Host` or
    /// `PullTarget::GpuGdr` with mandatory post-CQE `complete_gdr_slot` BAR1
    /// flush) for a previously received `PendingReadGrant` and send `ReadRelease`.
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
            grant.valid(old(self)),
            old(self).peer.rdma.qp.ready(),
            old(self).peer.rdma.qp.state() is ReadyToSend,
            old(self).core.frames.frame_size as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            match target {
                PullTarget::Host => old(self).peer.rdma.host_mr matches Some(ref s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).peer.rdma.gpu_mr matches Some(ref s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.qp.ready(),
            final(self).peer.rdma.qp.state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
        if self.peer.rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        if grant.meta.frame_size != self.core.frames.frame_size as u64 {
            return Err(CoreError::InvalidArgument);
        }
        let PendingReadGrant { tag, meta, cap, share, statement, cert, revoked, version_binding } =
            grant;
        let ghost before = self.core;
        let ghost orig_share_frontier = share@.frontier();
        let pull = match self.peer.fetch_post_pull(
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
                self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
                return Err(e);
            },
        };
        let ghost after_post = self.core;
        let ghost pull_frame = pull.handle.frame;
        let tracked vb = version_binding.get().tracked_unwrap();
        let res = self.peer.fetch_complete_pull(
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
                self.peer.rdma.timeline_id,
                pull_frame,
            );
        }
        self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
        res
    }

    /// Initiator: one-shot remote fetch (`FetchRequest` -> `ReadGrant` ->
    /// `RDMA_READ` into `PullTarget::Host` or `PullTarget::GpuGdr` ->
    /// `complete_gdr_slot` BAR1 flush -> `finish_write` -> `ReadRelease`).
    pub fn fetch_remote_frame(
        &mut self,
        key: ObjectKey,
        dst_tier: TierId,
        target: PullTarget,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<FetchOutcome, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peer.rdma.qp.ready(),
            old(self).peer.rdma.qp.state() is ReadyToSend,
            old(self).peer.next_request_id < u64::MAX,
            old(self).core.frames.frame_size as u64 <= rdma::MAX_RDMA_MSG_LEN,
            instance.id() == old(self).core.instance_id(),
            match target {
                PullTarget::Host => old(self).peer.rdma.host_mr matches Some(ref s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec().is_none(),
                PullTarget::GpuGdr => old(self).peer.rdma.gpu_mr matches Some(ref s) && s.tier
                    == dst_tier && s.mr.access().local_write && s.mr.gpu_device_spec() is Some,
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.qp.ready(),
            final(self).peer.rdma.qp.state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
        if self.peer.rdma.qp.effective_max_rd_atomic_exec() == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let res = self.peer.fetch_remote_frame(
            &mut self.core,
            key,
            dst_tier,
            target,
            Tracked(instance),
        );
        self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
        res
    }

    /// Donor: Option-A IBTA-strict unilateral MW revocation (`LOCAL_INV` +
    /// `ibv_dealloc_mw` -> `Quarantined`) and `RevokeNotice` notification.
    pub fn revoke_lease_unilateral(
        &mut self,
        grant_tag: GrantTag,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<u32, CoreError>)
        requires
            old(self).well_formed(),
            old(self).peer.rdma.qp.ready(),
            old(self).peer.rdma.qp.state() is ReadyToSend,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.qp == old(self).peer.rdma.qp,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let res = self.peer.revoke_lease_unilateral(
            &mut self.core,
            &mut self.host_ctx,
            grant_tag,
            Tracked(instance),
        );
        self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
        res
    }

    /// Donor: transition a `Quarantined` MW slot to `Vacant` after peer ack.
    pub fn acknowledge_quarantined_window(&mut self, mw_slot: u32) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.qp == old(self).peer.rdma.qp,
    {
        self.peer.rdma.acknowledge_quarantined_window(&self.core, mw_slot)
    }

    /// Donor: replenish a `Vacant` MW pool slot with a freshly allocated Type-2 MW.
    pub fn replenish_window(
        &mut self,
        mw_slot: u32,
        pd: &'a rdma::ProtectionDomain<'a>,
        Tracked(name): Tracked<EventSourceToken>,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            (mw_slot as int) < old(self).peer.rdma.window_pool.slots.len(),
            old(self).peer.rdma.window_pool.slots[mw_slot as int].state
                == crate::backends::rdma::MwSlotState::Vacant,
            pd.id() == old(self).peer.rdma.window_pool.pd(),
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
            final(self).peer.rdma.qp == old(self).peer.rdma.qp,
    {
        self.peer.rdma.replenish_window(&self.core, mw_slot, pd, Tracked(name))
    }

    /// Option-A recovery step 1: transition the QP into `IBV_QPS_ERR`.
    pub fn enter_qp_error(&mut self) -> (r: BackendResult<Tracked<rdma::QpErrorWitness>>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core == old(self).core,
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            final(self).peer.rdma.cq == old(self).peer.rdma.cq,
            r.is_ok() ==> final(self).peer.rdma.qp.state() == rdma::QueuePairState::Error,
    {
        self.peer.rdma.enter_qp_error(&self.core)
    }

    /// Option-A recovery step 2: reset the QP (`IBV_QPS_RESET`), unquarantine
    /// windows, recover any active leases, exchange fresh `QueuePairEndpoint`s
    /// with `psn`, and reconnect to `ReadyToSend`.
    pub fn reset_and_reconnect(
        &mut self,
        psn: u32,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            old(self).peer.rdma.qp.state() is ReadyToSend || old(
                self,
            ).peer.rdma.qp.state() is Error,
            psn <= 0xffffff,
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).nvme == old(self).nvme,
            r.is_ok() ==> final(self).peer.rdma.qp.ready()
                && final(self).peer.rdma.qp.state() is ReadyToSend,
    {
        hide(CacheCore::well_formed);
        if self.peer.rdma.cq.in_flight_count() != 0 {
            return Err(CoreError::WouldConflict);
        }
        proof {
            assert(self.peer.rdma.cq.in_flight().is_empty());
        }
        let local_endpoint = self.peer.rdma.qp.reset_endpoint(psn);
        let ghost before = self.core;
        let res = self.peer.reset_and_reconnect(
            &mut self.core,
            &mut self.host_ctx,
            &local_endpoint,
            psn,
            Tracked(instance),
        );
        self.restore_local_after_peer_step(Ghost(before), Tracked(instance));
        res
    }

    // =========================================================================
    // 4-Tier Local NVMe (`io_uring` `O_DIRECT`) Lifecycle & Data-Path Methods
    // =========================================================================
    /// Stage and submit an async Host-DRAM -> Local-NVMe tier-down copy hop.
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
            final(self).peer == old(self).peer,
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
                &self.peer,
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

    /// Stage and submit an async Local-NVMe -> Host-DRAM warm-up copy hop.
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
            final(self).peer == old(self).peer,
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
                &self.peer,
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

    /// Non-blocking drain of up to `budget` ready NVMe `io_uring` completions
    /// (supports `QD >= 8` batched replay), preserving M8 non-dropping progress.
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
            final(self).peer == old(self).peer,
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
                &self.peer,
            );
            assert(self.core.frames.versions@ == before.frames.versions@) by {
                reveal(core_transition);
            }
            assert(self.core.all_versions_bound());
        }
        self.nvme = Some(nvme);
        res
    }

    /// Block until one in-flight NVMe `io_uring` operation completes, submitting
    /// any staged SQEs first if needed and preserving M8 non-dropping progress.
    pub fn wait_nvme_completion(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).peer == old(self).peer,
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
                &self.peer,
            );
            assert(self.core.frames.versions@ == before.frames.versions@) by {
                reveal(core_transition);
            }
            assert(self.core.all_versions_bound());
        }
        self.nvme = Some(nvme);
        res
    }

    /// Synchronously wait for an NVMe `io_uring` `ticket` to retire, preserving
    /// M8 non-dropping `NvmeProgress` on error.
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
            final(self).peer == old(self).peer,
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
                self.peer == old(self).peer,
                self.peer.well_formed(&self.core),
                self.peer.rdma.timeline_id != self.host.timeline,
                self.peer.rdma.peer_node != self.core.node(),
                self.peer.rdma.qp.peer() == self.peer.rdma.peer_node,
                self.peer.rdma.qp.pd_id() == self.peer.rdma.window_pool.pd(),
                self.peer.rdma.qp.source().well_formed(),
                self.peer.rdma.qp.source().source() is RdmaQueuePair,
                self.peer.rdma.qp.source().node() != self.peer.rdma.peer_node,
                nvme.well_formed(&self.core),
                nvme.timeline == tl,
                nvme.timeline != self.host.timeline,
                nvme.timeline != self.peer.rdma.timeline_id,
                self.cuda == old(self).cuda,
                self.cuda matches Some(c) ==> {
                    &&& c.well_formed(&self.core)
                    &&& c.active_gpu_puts@.len() == 0
                    &&& c.timeline.timeline_id != self.host.timeline
                    &&& c.timeline.timeline_id != self.peer.rdma.timeline_id
                    &&& nvme.timeline != c.timeline.timeline_id
                    &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier && h.first_frame
                        == self.host.first_frame && h.num_frames == self.host.num_frames
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
                    &self.peer,
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

    /// Synchronously spill `key` from Host DRAM to Local NVMe (`io_uring` `O_DIRECT`).
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
            final(self).peer == old(self).peer,
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

    /// Synchronously warm `key` from Local NVMe (`io_uring` `O_DIRECT`) into Host DRAM.
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
            final(self).peer == old(self).peer,
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
