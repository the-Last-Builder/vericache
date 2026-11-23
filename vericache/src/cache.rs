//! Location-transparent [`VeriCache`] facade combining:
//! - Unified polymorphic `get(key, target)` / `put(key, source)` + explicit
//!   `get_into_host`, `get_into_gpu`, `put_host`, `put_gpu`, `lease_host`, `lease_gpu`;
//! - Unverified, pluggable [`EvictionPolicy`], [`TieringPolicy`], and [`PeerRoutingPolicy`];
//! - Verified enforcement via [`DistributedEngine`] (`CacheCore` lifecycle, active lease
//!   protection, cooperative-first + Option-A fenced RDMA MW revocation, `VersionBinding`).

#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::{EventSourceToken, ReadPerm, ShareCap, SlotToken};
use transfer::cuda::{self, CudaHostBuffer, CudaSlice, CudaStream};
use transfer::host::HostContext;
use transfer::rdma::ProtectionDomain;
use transfer::transport::ByteTransport;
use vericache_core::backends::rdma::peer::FetchOutcome;
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::engine::DistributedEngine;
use vericache_core::frame::{STATE_EMPTY, STATE_PUBLISHED};
use vericache_core::kernel::ReadLeaseHandle;
use vericache_core::spec::{CoreError, FrameId, ObjectKey, Ticket, TierId, TierKind, VersionId};
use vericache_core::timeline::TimelineKind;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Tracked, verus};

use crate::lease::{GpuLeaseGuard, HostLeaseGuard};
use crate::policy::{
    DataDomain, EvictionPolicy, HitLocation, PeerMeta, PeerRoutingPolicy, PolicySet, TieringPolicy,
};
use crate::resources::{
    GpuSlot, GpuStream, NodeStorage, PeerLink, ShardCaps, StorageConfig, alloc_gpu_slot_internal,
    download_gpu_slot_internal, upload_gpu_slot_internal,
};
use crate::target::{ReadTarget, WriteSource};
use crate::{CacheError, CacheStats, GpuReadHit, PollSummary, ReadHit};

struct PendingSpill {
    key: ObjectKey,
    dram_frame: FrameId,
    nvme_frame: FrameId,
    ticket: Ticket,
}

verus! {

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn call_put_gpu_sync<'a, T: ByteTransport>(
    engine: &mut DistributedEngine<'a, T>,
    caps: &ShardCaps,
    stream: &mut GpuStream,
    key: ObjectKey,
    slot: &mut GpuSlot,
) -> Result<(VersionId, FrameId), CacheError> {
    let cert = slot.cert.ok_or_else(|| {
        CacheError::Init(
            "GpuSlot has no provenance; populate it via write_gpu_slot before put_gpu".into(),
        )
    })?;
    let src_slice = slot.buffer.slot_slice(0);
    let tracked mut tmp_tok = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(slot.token.borrow_mut(), &mut tmp_tok);
    }
    let tracked mut perm = ReadPerm::Owned(tmp_tok);
    let res = engine.put_gpu_sync(
        Tracked(caps.instance.borrow()),
        &mut stream.stream,
        stream.timeline,
        key,
        &src_slice,
        Tracked(&mut perm),
        cert,
    );
    proof {
        *slot.token.borrow_mut() = perm.into_owned();
    }
    res.map_err(CacheError::Core)
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn call_get_into_gpu_frame<'a, T: ByteTransport>(
    engine: &mut DistributedEngine<'a, T>,
    caps: &mut ShardCaps,
    frame: FrameId,
    stream: &mut GpuStream,
    slot: &mut GpuSlot,
) -> Result<(VersionId, Ticket), CacheError> {
    let dst_slice = slot.buffer.slot_slice(0);
    let (ver, ticket, _ev) = engine
        .get_into_gpu_frame(
            Tracked(caps.instance.borrow()),
            Tracked(caps.shares.borrow_mut()),
            frame,
            &mut stream.stream,
            stream.timeline,
            &dst_slice,
            Tracked(slot.token.borrow_mut()),
        )
        .map_err(CacheError::Core)?;
    Ok((ver, ticket))
}

} // verus!

verus! {

/// Location-transparent distributed KV cache facade spanning GPU HBM, Pinned Host DRAM,
/// Local NVMe (`io_uring`), and `0..N` RDMA / GPUDirect RDMA peers.
pub struct VeriCache<'a, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    node_id: u64,
    frame_size: usize,
    engine: Box<DistributedEngine<'a, T>>,
    caps: ShardCaps,
    pd: Option<&'a ProtectionDomain<'a>>,
    peer_metas: Vec<PeerMeta>,
    policy: PolicySet<E, P, R>,
    internal_gpu_stream: Option<GpuStream>,
    gpu_staging: Option<(CudaHostBuffer, Tracked<SlotToken>, HostContext)>,
    rdma_replenish_tokens: Vec<Tracked<EventSourceToken>>,
    pending_spills: Vec<PendingSpill>,
    probe_buf: Vec<usize>,
    cooperative_revoke_spin_budget: u32,
    stats: CacheStats,
}

impl<'a, T, E, P, R> VeriCache<'a, T, E, P, R>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        node_id: u64,
        frame_size: usize,
        engine: Box<DistributedEngine<'a, T>>,
        caps: ShardCaps,
        pd: Option<&'a ProtectionDomain<'a>>,
        peer_metas: Vec<PeerMeta>,
        policy: PolicySet<E, P, R>,
        internal_gpu_stream: Option<GpuStream>,
        gpu_staging: Option<(CudaHostBuffer, Tracked<SlotToken>, HostContext)>,
    ) -> Self {
        Self {
            node_id,
            frame_size,
            engine,
            caps,
            pd,
            peer_metas,
            policy,
            internal_gpu_stream,
            gpu_staging,
            rdma_replenish_tokens: Vec::new(),
            pending_spills: Vec::new(),
            probe_buf: Vec::new(),
            cooperative_revoke_spin_budget: 256,
            stats: CacheStats::default(),
        }
    }

    /// One-shot scoped helper that hides the two-step zero-`unsafe` storage lifetime
    /// (`NodeStorage` -> `AttachedStorage` -> `VeriCache`) completely from callers.
    pub fn with_session<F, Out>(
        config: StorageConfig,
        peers: Vec<PeerLink<T>>,
        policy: PolicySet<E, P, R>,
        f: F,
    ) -> Result<Out, CacheError>
    where
        F: FnOnce(&mut VeriCache<'_, T, E, P, R>) -> Result<Out, CacheError>,
    {
        let mut storage = NodeStorage::new(config)?;
        let mut attached = storage.attach_tiers()?;
        let mut cache = attached.open_cache(peers, policy)?;
        let out = f(&mut cache)?;
        cache.shutdown()?;
        Ok(out)
    }

    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn policy(&self) -> &PolicySet<E, P, R> {
        &self.policy
    }

    pub fn policy_mut(&mut self) -> &mut PolicySet<E, P, R> {
        &mut self.policy
    }

    pub fn peer_metas(&self) -> &[PeerMeta] {
        &self.peer_metas
    }

    pub fn set_cooperative_revoke_spin_budget(&mut self, budget: u32) {
        self.cooperative_revoke_spin_budget = budget;
    }

    /// Allocate a caller-owned [`GpuSlot`] matching this cache's GPU device and frame size.
    pub fn alloc_gpu_slot(&mut self) -> Result<GpuSlot, CacheError> {
        let dev = self
            .internal_gpu_stream
            .as_ref()
            .map(|s| s.device_id)
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        alloc_gpu_slot_internal(&mut self.caps, dev, self.frame_size)
    }

    /// Create a caller-owned [`GpuStream`] registered as a timeline on `CacheCore`.
    pub fn create_gpu_stream(&mut self) -> Result<GpuStream, CacheError> {
        let dev = self
            .internal_gpu_stream
            .as_ref()
            .map(|s| s.device_id)
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        let timeline = self
            .engine
            .core
            .register_timeline(TimelineKind::CudaStream { device: dev })
            .map_err(CacheError::Core)?;
        let stream = CudaStream::new(dev, self.caps.mint_cuda())
            .map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
        Ok(GpuStream {
            stream,
            timeline,
            device_id: dev,
        })
    }

    /// Synchronize a caller [`GpuStream`] and mark `ticket` complete on `CacheCore`.
    pub fn synchronize_stream(
        &mut self,
        stream: &mut GpuStream,
        ticket: Ticket,
    ) -> Result<(), CacheError> {
        let knows = cuda::stream_synchronize(&stream.stream, Tracked(self.caps.instance.borrow()))
            .map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
        if !ticket.is_none() {
            self.engine
                .core
                .mark_timeline_complete(stream.timeline, ticket.seq(), knows);
        }
        Ok(())
    }

    /// Upload host bytes into a caller [`GpuSlot`] via pinned staging DMA (minting
    /// verified provenance so the slot can subsequently be published with `put`).
    pub fn write_gpu_slot(
        &mut self,
        slot: &mut GpuSlot,
        stream: &mut GpuStream,
        data: &[u8],
    ) -> Result<(), CacheError> {
        let host_tl = self.engine.host.timeline_id();
        let (staging, staging_tok, staging_ctx) = self
            .gpu_staging
            .as_mut()
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        upload_gpu_slot_internal(
            &mut self.engine.core,
            host_tl,
            &mut self.caps,
            staging,
            staging_tok,
            staging_ctx,
            stream,
            slot,
            data,
        )
    }

    /// Download a caller [`GpuSlot`] into host bytes via pinned staging DMA.
    pub fn read_gpu_slot(
        &mut self,
        slot: &mut GpuSlot,
        stream: &mut GpuStream,
        dst: &mut [u8],
    ) -> Result<(), CacheError> {
        let host_tl = self.engine.host.timeline_id();
        let (staging, staging_tok, staging_ctx) = self
            .gpu_staging
            .as_mut()
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        download_gpu_slot_internal(
            &mut self.engine.core,
            host_tl,
            &mut self.caps,
            staging,
            staging_tok,
            staging_ctx,
            stream,
            slot,
            dst,
        )
    }

    // =========================================================================
    // Unified Polymorphic Facade API (Style C)
    // =========================================================================

    /// Location-transparent read into any [`ReadTarget`] (`&mut [u8]`, `&mut Vec<u8>`,
    /// or [`crate::GpuDst`]). Automatically resolves across Local GPU HBM, Local Pinned
    /// Host DRAM, Local NVMe, and all connected RDMA / GPUDirect RDMA peers.
    pub fn get<Dst: ReadTarget>(
        &mut self,
        key: ObjectKey,
        dst: Dst,
    ) -> Result<Option<Dst::Output>, CacheError> {
        dst.load(self, key)
    }

    /// Location-transparent write from any [`WriteSource`] (`&[u8]` or [`crate::GpuSrc`]).
    /// Capacity reclamation, cascading demotion, and RDMA lease fencing are enforced
    /// automatically by the verified core.
    pub fn put<Src: WriteSource>(
        &mut self,
        key: ObjectKey,
        src: Src,
    ) -> Result<Src::Output, CacheError> {
        src.store(self, key)
    }

    // =========================================================================
    // Explicit Domain Methods (Style A) + Zero-Copy RAII Leases
    // =========================================================================

    /// Publish host bytes into Host DRAM (evicting/cascading via policy if full).
    pub fn put_host(&mut self, key: ObjectKey, data: &[u8]) -> Result<VersionId, CacheError> {
        if data.len() != self.frame_size {
            return Err(CacheError::Core(CoreError::InvalidArgument));
        }
        self.reap_completed_spills();
        self.wait_pending_spill_for_key(key)?;
        self.ensure_tier_space(TierKind::HostDram)?;
        let (ver, frame) = self
            .engine
            .put_host(Tracked(self.caps.instance.borrow()), key, data)
            .map_err(CacheError::Core)?;
        self.policy.eviction.on_insert(TierKind::HostDram, key, frame);
        self.stats.puts += 1;
        self.kick_async_spills()?;
        Ok(ver)
    }

    /// Publish a caller [`GpuSlot`] into GPU HBM (evicting/cascading via policy if full).
    pub fn put_gpu(
        &mut self,
        key: ObjectKey,
        stream: &mut GpuStream,
        slot: &mut GpuSlot,
    ) -> Result<VersionId, CacheError> {
        if self.engine.gpu_tier().is_none() {
            return Err(CacheError::Core(CoreError::UnsupportedTier));
        }
        if slot.frame_size != self.frame_size {
            return Err(CacheError::Core(CoreError::InvalidArgument));
        }
        self.reap_completed_spills();
        self.ensure_tier_space(TierKind::GpuVram)?;
        let (ver, frame) = call_put_gpu_sync(&mut self.engine, &self.caps, stream, key, slot)?;
        self.policy.eviction.on_insert(TierKind::GpuVram, key, frame);
        self.stats.puts += 1;
        Ok(ver)
    }

    /// Location-transparent synchronous read into a host buffer `dst`:
    /// `Local DRAM -> Local GPU HBM (1-hop D2H) -> Local NVMe (io_uring) -> Remote Peers (RDMA)`.
    pub fn get_into_host(
        &mut self,
        key: ObjectKey,
        dst: &mut [u8],
    ) -> Result<Option<ReadHit>, CacheError> {
        if dst.len() != self.frame_size {
            return Err(CacheError::Core(CoreError::InvalidArgument));
        }
        self.reap_completed_spills();
        let _ = self.poll_peers_nonblocking();

        let host_tier = self.engine.host_tier();

        // 1. Local Host DRAM (1-hop direct read)
        if let Some((frame, _)) = self.engine.lookup_tier(key, host_tier) {
            let ver = self.read_host_frame_raw(frame, dst)?;
            self.policy
                .eviction
                .on_access(TierKind::HostDram, key, frame);
            self.stats.dram_hits += 1;
            return Ok(Some(ReadHit {
                version: ver,
                location: HitLocation::LocalDram,
            }));
        }

        // 2. Local GPU HBM (1-hop verified D2H copy_hop_cuda_sync -> Host DRAM -> dst)
        if let Some(gpu_tier) = self.engine.gpu_tier() {
            if let Some((gpu_frame, _)) = self.engine.lookup_tier(key, gpu_tier) {
                self.ensure_tier_space(TierKind::HostDram)?;
                let (_ver, dram_frame) = self.copy_hop_internal(gpu_frame, host_tier)?;
                self.policy
                    .eviction
                    .on_access(TierKind::GpuVram, key, gpu_frame);
                self.policy
                    .eviction
                    .on_insert(TierKind::HostDram, key, dram_frame);
                let ver = self.read_host_frame_raw(dram_frame, dst)?;
                self.stats.gpu_hits += 1;
                return Ok(Some(ReadHit {
                    version: ver,
                    location: HitLocation::LocalGpu,
                }));
            }
        }

        // 3. Local NVMe (io_uring O_DIRECT warm -> Host DRAM -> dst)
        self.wait_pending_spill_for_key(key)?;
        if let Some(nvme_tier) = self.engine.nvme_tier() {
            if let Some((nvme_frame, _)) = self.engine.lookup_tier(key, nvme_tier) {
                self.ensure_tier_space(TierKind::HostDram)?;
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.policy
                    .eviction
                    .on_access(TierKind::LocalNvme, key, nvme_frame);
                self.policy
                    .eviction
                    .on_insert(TierKind::HostDram, key, dram_frame);
                let ver = self.read_host_frame_raw(dram_frame, dst)?;
                self.stats.nvme_hits += 1;
                return Ok(Some(ReadHit {
                    version: ver,
                    location: HitLocation::LocalNvme,
                }));
            }
        }

        // 4. Remote Peers (full-duplex RDMA pull into Host DRAM -> dst)
        if let Some((dram_frame, node_id)) = self.fetch_from_peers(key, DataDomain::Host)? {
            let ver = self.read_host_frame_raw(dram_frame, dst)?;
            self.stats.remote_hits += 1;
            return Ok(Some(ReadHit {
                version: ver,
                location: HitLocation::RemotePeer {
                    node_id,
                    gdr: false,
                },
            }));
        }

        self.stats.misses += 1;
        Ok(None)
    }

    /// Location-transparent asynchronous GPU read into `slot` on `stream`:
    /// `Local GPU HBM (1-hop D2D) -> Local Pinned DRAM (1-hop H2D / promote) ->
    /// Local NVMe (io_uring warm -> H2D) -> Remote Peers (GPUDirect RDMA or Host RDMA)`.
    pub fn get_into_gpu(
        &mut self,
        key: ObjectKey,
        stream: &mut GpuStream,
        slot: &mut GpuSlot,
    ) -> Result<Option<GpuReadHit>, CacheError> {
        let gpu_tier = self
            .engine
            .gpu_tier()
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        if slot.frame_size != self.frame_size {
            return Err(CacheError::Core(CoreError::InvalidArgument));
        }
        self.reap_completed_spills();
        let _ = self.poll_peers_nonblocking();

        let host_tier = self.engine.host_tier();

        // 1. Local GPU HBM (1-hop D2D Ordered Handoff)
        if let Some((gpu_frame, _)) = self.engine.lookup_tier(key, gpu_tier) {
            let (ver, ticket) =
                call_get_into_gpu_frame(&mut self.engine, &mut self.caps, gpu_frame, stream, slot)?;
            self.policy
                .eviction
                .on_access(TierKind::GpuVram, key, gpu_frame);
            self.stats.gpu_hits += 1;
            return Ok(Some(GpuReadHit {
                version: ver,
                location: HitLocation::LocalGpu,
                ticket,
            }));
        }

        // 2. Local Pinned Host DRAM (optional promote to HBM or direct 1-hop H2D)
        if let Some((dram_frame, _)) = self.engine.lookup_tier(key, host_tier) {
            self.policy
                .eviction
                .on_access(TierKind::HostDram, key, dram_frame);
            let read_frame = if self
                .policy
                .tiering
                .promote_on_hit(HitLocation::LocalDram, DataDomain::Gpu)
                == Some(TierKind::GpuVram)
            {
                match self.try_promote_dram_to_gpu(key, dram_frame) {
                    Ok(gf) => gf,
                    Err(_) => dram_frame,
                }
            } else {
                dram_frame
            };
            let (ver, ticket) =
                call_get_into_gpu_frame(&mut self.engine, &mut self.caps, read_frame, stream, slot)?;
            self.stats.dram_hits += 1;
            return Ok(Some(GpuReadHit {
                version: ver,
                location: HitLocation::LocalDram,
                ticket,
            }));
        }

        // 3. Local NVMe (io_uring warm -> Pinned DRAM -> optional promote -> GPU stream)
        self.wait_pending_spill_for_key(key)?;
        if let Some(nvme_tier) = self.engine.nvme_tier() {
            if let Some((nvme_frame, _)) = self.engine.lookup_tier(key, nvme_tier) {
                self.ensure_tier_space(TierKind::HostDram)?;
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.policy
                    .eviction
                    .on_access(TierKind::LocalNvme, key, nvme_frame);
                self.policy
                    .eviction
                    .on_insert(TierKind::HostDram, key, dram_frame);
                let read_frame = if self
                    .policy
                    .tiering
                    .promote_on_hit(HitLocation::LocalNvme, DataDomain::Gpu)
                    == Some(TierKind::GpuVram)
                {
                    match self.try_promote_dram_to_gpu(key, dram_frame) {
                        Ok(gf) => gf,
                        Err(_) => dram_frame,
                    }
                } else {
                    dram_frame
                };
                let (ver, ticket) = call_get_into_gpu_frame(
                    &mut self.engine,
                    &mut self.caps,
                    read_frame,
                    stream,
                    slot,
                )?;
                self.stats.nvme_hits += 1;
                return Ok(Some(GpuReadHit {
                    version: ver,
                    location: HitLocation::LocalNvme,
                    ticket,
                }));
            }
        }

        // 4. Remote Peers (GPUDirect RDMA direct into HBM or Host RDMA -> H2D)
        if let Some((frame, node_id, used_gdr)) = self.fetch_from_peers_for_gpu(key)? {
            let (ver, ticket) =
                call_get_into_gpu_frame(&mut self.engine, &mut self.caps, frame, stream, slot)?;
            self.stats.remote_hits += 1;
            return Ok(Some(GpuReadHit {
                version: ver,
                location: HitLocation::RemotePeer {
                    node_id,
                    gdr: used_gdr,
                },
                ticket,
            }));
        }

        self.stats.misses += 1;
        Ok(None)
    }

    /// Location-transparent zero-copy RAII lease on a Host DRAM frame.
    /// Automatically warms from Local GPU / Local NVMe / Remote Peers if not yet in DRAM.
    pub fn lease_host(
        &mut self,
        key: ObjectKey,
    ) -> Result<Option<HostLeaseGuard<'_, 'a, T, E, P, R>>, CacheError> {
        self.reap_completed_spills();
        let _ = self.poll_peers_nonblocking();

        let host_tier = self.engine.host_tier();
        let location = if let Some((frame, _)) = self.engine.lookup_tier(key, host_tier) {
            self.policy
                .eviction
                .on_access(TierKind::HostDram, key, frame);
            self.stats.dram_hits += 1;
            HitLocation::LocalDram
        } else if let Some(gpu_frame) = self
            .engine
            .gpu_tier()
            .and_then(|gt| self.engine.lookup_tier(key, gt))
            .map(|(f, _)| f)
        {
            self.ensure_tier_space(TierKind::HostDram)?;
            let (_ver, dram_frame) = self.copy_hop_internal(gpu_frame, host_tier)?;
            self.policy
                .eviction
                .on_access(TierKind::GpuVram, key, gpu_frame);
            self.policy
                .eviction
                .on_insert(TierKind::HostDram, key, dram_frame);
            self.stats.gpu_hits += 1;
            HitLocation::LocalGpu
        } else {
            self.wait_pending_spill_for_key(key)?;
            if let Some(nvme_frame) = self
                .engine
                .nvme_tier()
                .and_then(|nt| self.engine.lookup_tier(key, nt))
                .map(|(f, _)| f)
            {
                self.ensure_tier_space(TierKind::HostDram)?;
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.policy
                    .eviction
                    .on_access(TierKind::LocalNvme, key, nvme_frame);
                self.policy
                    .eviction
                    .on_insert(TierKind::HostDram, key, dram_frame);
                self.stats.nvme_hits += 1;
                HitLocation::LocalNvme
            } else if let Some((_frame, node_id)) = self.fetch_from_peers(key, DataDomain::Host)? {
                self.stats.remote_hits += 1;
                HitLocation::RemotePeer {
                    node_id,
                    gdr: false,
                }
            } else {
                self.stats.misses += 1;
                return Ok(None);
            }
        };

        let lease = self
            .engine
            .get_lease_host(
                Tracked(self.caps.instance.borrow()),
                Tracked(self.caps.shares.borrow_mut()),
                key,
            )
            .map_err(CacheError::Core)?;
        Ok(lease.map(|l| HostLeaseGuard::new(self, l, location)))
    }

    /// Location-transparent zero-copy RAII lease on a GPU HBM frame.
    /// Automatically promotes from Pinned DRAM / Local NVMe or pulls via GPUDirect RDMA
    /// from Remote Peers, avoiding any extra D2D copy when the caller can consume the
    /// cached HBM frame directly.
    pub fn lease_gpu(
        &mut self,
        key: ObjectKey,
    ) -> Result<Option<GpuLeaseGuard<'_, 'a, T, E, P, R>>, CacheError> {
        let gpu_tier = self
            .engine
            .gpu_tier()
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        self.reap_completed_spills();
        let _ = self.poll_peers_nonblocking();

        let host_tier = self.engine.host_tier();
        let location = if let Some((gpu_frame, _)) = self.engine.lookup_tier(key, gpu_tier) {
            self.policy
                .eviction
                .on_access(TierKind::GpuVram, key, gpu_frame);
            self.stats.gpu_hits += 1;
            HitLocation::LocalGpu
        } else if let Some((dram_frame, _)) = self.engine.lookup_tier(key, host_tier) {
            self.policy
                .eviction
                .on_access(TierKind::HostDram, key, dram_frame);
            let _ = self.try_promote_dram_to_gpu(key, dram_frame)?;
            self.stats.dram_hits += 1;
            HitLocation::LocalDram
        } else {
            self.wait_pending_spill_for_key(key)?;
            if let Some(nvme_frame) = self
                .engine
                .nvme_tier()
                .and_then(|nt| self.engine.lookup_tier(key, nt))
                .map(|(f, _)| f)
            {
                self.ensure_tier_space(TierKind::HostDram)?;
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.policy
                    .eviction
                    .on_access(TierKind::LocalNvme, key, nvme_frame);
                self.policy
                    .eviction
                    .on_insert(TierKind::HostDram, key, dram_frame);
                let _ = self.try_promote_dram_to_gpu(key, dram_frame)?;
                self.stats.nvme_hits += 1;
                HitLocation::LocalNvme
            } else if let Some((frame, node_id, used_gdr)) = self.fetch_from_peers_for_gpu(key)? {
                if !used_gdr {
                    let _ = self.try_promote_dram_to_gpu(key, frame)?;
                }
                self.stats.remote_hits += 1;
                HitLocation::RemotePeer {
                    node_id,
                    gdr: used_gdr,
                }
            } else {
                self.stats.misses += 1;
                return Ok(None);
            }
        };

        let lease = self
            .engine
            .get_lease_gpu(
                Tracked(self.caps.instance.borrow()),
                Tracked(self.caps.shares.borrow_mut()),
                key,
            )
            .map_err(CacheError::Core)?;
        Ok(lease.map(|l| GpuLeaseGuard::new(self, l, location)))
    }

    /// Explicitly evict `key` from a specific local tier (with optional cascade demotion
    /// disabled — purely removes that tier's replica through verified peer/lease fencing).
    pub fn evict_from_tier(&mut self, key: ObjectKey, tier: TierKind) -> Result<bool, CacheError> {
        let Some(tid) = self.tier_id_for(tier) else {
            return Ok(false);
        };
        let Some((frame, _)) = self.engine.lookup_tier(key, tid) else {
            return Ok(false);
        };
        let _ = self.engine.drain_and_evict_frame_across_peers(
            frame,
            self.cooperative_revoke_spin_budget,
            Tracked(self.caps.instance.borrow()),
        );
        let _ = self.engine.reclaim_draining(tid, 16);
        self.policy.eviction.on_remove(tier, key, frame);
        Ok(true)
    }

    /// Manually spill a DRAM key to NVMe and evict its DRAM replica (useful for
    /// deterministic tier placement or testing).
    pub fn spill_key_to_nvme(&mut self, key: ObjectKey) -> Result<bool, CacheError> {
        let Some(nvme_tier) = self.engine.nvme_tier() else {
            return Err(CacheError::Core(CoreError::UnsupportedTier));
        };
        let host_tier = self.engine.host_tier();
        let Some((dram_frame, _)) = self.engine.lookup_tier(key, host_tier) else {
            return Ok(false);
        };
        self.ensure_tier_space(TierKind::LocalNvme)?;
        let nvme_frame = self
            .engine
            .spill_host_to_nvme_sync(
                Tracked(self.caps.instance.borrow()),
                Tracked(self.caps.shares.borrow_mut()),
                key,
            )
            .map_err(CacheError::Core)?;
        self.policy
            .eviction
            .on_insert(TierKind::LocalNvme, key, nvme_frame);
        let _ = self.engine.drain_and_evict_frame_across_peers(
            dram_frame,
            self.cooperative_revoke_spin_budget,
            Tracked(self.caps.instance.borrow()),
        );
        let _ = self.engine.reclaim_draining(host_tier, 16);
        self.policy
            .eviction
            .on_remove(TierKind::HostDram, key, dram_frame);
        let _ = nvme_tier;
        Ok(true)
    }

    /// Non-blocking progress sweep across peer RDMA requests, async NVMe spills,
    /// and draining-frame reclamation.
    pub fn poll(&mut self) -> Result<PollSummary, CacheError> {
        let mut summary = PollSummary::default();
        summary.nvme_spills_retired = self.reap_completed_spills();
        if !self.peer_metas.is_empty() {
            let p = self.poll_peers_nonblocking()?;
            summary.peer_cq_processed = p.cq_processed;
            summary.peer_grants_completed = p.grants_completed;
            summary.peer_windows_replenished = p.windows_replenished;
        }
        summary.reclaimed_frames = self.engine.reclaim_all_draining();
        Ok(summary)
    }

    /// Drain in-flight NVMe spills and cleanly shut down all RDMA peer sessions.
    pub fn shutdown(&mut self) -> Result<(), CacheError> {
        while !self.pending_spills.is_empty() {
            let _ = self.poll_peers_nonblocking();
            if self.reap_completed_spills() == 0 {
                let _ = self
                    .engine
                    .wait_nvme_completion(Tracked(self.caps.instance.borrow()));
                let _ = self.reap_completed_spills();
            }
        }
        if !self.peer_metas.is_empty() {
            self.engine
                .shutdown_all_peers(Tracked(self.caps.instance.borrow()))
                .map_err(CacheError::Core)?;
        }
        Ok(())
    }

    /// Inspect the verified `DistributedEngine` immutably (for diagnostics or tests).
    pub fn engine(&self) -> &DistributedEngine<'a, T> {
        &self.engine
    }

    /// Lightweight metadata-only lookup across local tiers (`LocalGpu -> LocalDram -> LocalNvme`)
    /// without triggering any data movement (designed for vLLM Scheduler prefix matching).
    pub fn contains_local(&self, key: ObjectKey) -> Option<HitLocation> {
        if let Some(gpu_tier) = self.engine.gpu_tier() {
            if self.engine.lookup_tier(key, gpu_tier).is_some() {
                return Some(HitLocation::LocalGpu);
            }
        }
        if self.engine.lookup_tier(key, self.engine.host_tier()).is_some() {
            return Some(HitLocation::LocalDram);
        }
        if let Some(nvme_tier) = self.engine.nvme_tier() {
            if self.engine.lookup_tier(key, nvme_tier).is_some() {
                return Some(HitLocation::LocalNvme);
            }
        }
        None
    }

    /// Batch metadata-only lookup across local tiers for a slice of keys.
    pub fn contains_batch_local(&self, keys: &[ObjectKey]) -> Vec<Option<HitLocation>> {
        let mut out = Vec::with_capacity(keys.len());
        for &k in keys {
            out.push(self.contains_local(k));
        }
        out
    }

    /// Publish a [`GpuSlot`] that was populated directly on the GPU by an external
    /// CUDA producer (e.g., PyTorch/vLLM D2D copy into `slot.device_addr()`).
    pub fn put_gpu_external(
        &mut self,
        key: ObjectKey,
        stream: &mut GpuStream,
        slot: &mut GpuSlot,
    ) -> Result<VersionId, CacheError> {
        crate::resources::mark_gpu_slot_external_write(slot);
        self.put_gpu(key, stream, slot)
    }

    // =========================================================================
    // Internal Helpers (Verified Enforcement + Multi-Tier Routing)
    // =========================================================================

    pub(crate) fn read_host_frame_raw(
        &mut self,
        frame: FrameId,
        dst: &mut [u8],
    ) -> Result<VersionId, CacheError> {
        self.engine
            .read_host_frame(
                Tracked(self.caps.instance.borrow()),
                Tracked(self.caps.shares.borrow_mut()),
                frame,
                dst,
            )
            .map_err(CacheError::Core)
    }

    pub(crate) fn release_local_lease_raw(
        &mut self,
        handle: ReadLeaseHandle,
        cap: Tracked<ShareCap>,
        ticket: Ticket,
    ) -> Result<(), CacheError> {
        self.engine
            .release_lease(Tracked(self.caps.instance.borrow()), handle, cap, ticket)
            .map_err(CacheError::Core)
    }

    pub(crate) fn engine_gpu_frame_slice(&self, frame: FrameId) -> Option<CudaSlice<'_>> {
        self.engine.gpu_frame_slice(frame)
    }

    fn tier_id_for(&self, kind: TierKind) -> Option<TierId> {
        match kind {
            TierKind::GpuVram => self.engine.gpu_tier(),
            TierKind::HostDram => Some(self.engine.host_tier()),
            TierKind::LocalNvme => self.engine.nvme_tier(),
            _ => None,
        }
    }

    fn serve_tier_pair(&self) -> (TierId, Option<TierId>) {
        let host_tier = self.engine.host_tier();
        match self.engine.gpu_tier() {
            Some(gpu_tier) => (gpu_tier, Some(host_tier)),
            None => (host_tier, None),
        }
    }

    fn ensure_replenish_tokens(&mut self, min_count: usize) {
        while self.rdma_replenish_tokens.len() < min_count {
            self.rdma_replenish_tokens.push(self.caps.mint_rdma());
        }
    }

    fn poll_peers_nonblocking(
        &mut self,
    ) -> Result<vericache_core::backends::rdma::peer::PeerProgress, CacheError> {
        if self.peer_metas.is_empty() {
            return Ok(vericache_core::backends::rdma::peer::PeerProgress::empty());
        }
        let (primary, fallback) = self.serve_tier_pair();
        self.ensure_replenish_tokens(4);
        self.engine
            .poll_all_peers_progress_multi(
                primary,
                fallback,
                self.pd,
                &mut self.rdma_replenish_tokens,
                Tracked(self.caps.shares.borrow_mut()),
                Tracked(self.caps.instance.borrow()),
            )
            .map_err(CacheError::Core)
    }

    fn copy_hop_internal(
        &mut self,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> Result<(VersionId, FrameId), CacheError> {
        let mut stream = self
            .internal_gpu_stream
            .take()
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        let res = self.engine.copy_hop_cuda_sync(
            Tracked(self.caps.instance.borrow()),
            Tracked(self.caps.shares.borrow_mut()),
            &mut stream.stream,
            stream.timeline,
            src_frame,
            dst_tier,
        );
        self.internal_gpu_stream = Some(stream);
        res.map_err(CacheError::Core)
    }

    fn try_promote_dram_to_gpu(
        &mut self,
        key: ObjectKey,
        dram_frame: FrameId,
    ) -> Result<FrameId, CacheError> {
        let gpu_tier = self
            .engine
            .gpu_tier()
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        self.ensure_tier_space(TierKind::GpuVram)?;
        let (_ver, gpu_frame) = self.copy_hop_internal(dram_frame, gpu_tier)?;
        self.policy
            .eviction
            .on_insert(TierKind::GpuVram, key, gpu_frame);
        Ok(gpu_frame)
    }

    fn fetch_from_peers(
        &mut self,
        key: ObjectKey,
        consumer: DataDomain,
    ) -> Result<Option<(FrameId, u64)>, CacheError> {
        if self.peer_metas.is_empty() {
            return Ok(None);
        }
        let mut order = core::mem::take(&mut self.probe_buf);
        self.policy
            .routing
            .probe_order(key, &self.peer_metas, &mut order);
        let host_tier = self.engine.host_tier();
        let (primary, fallback) = self.serve_tier_pair();

        let mut hit = None;
        for &peer_idx in &order {
            if peer_idx >= self.peer_metas.len() {
                self.stats.policy_victims_rejected += 1;
                continue;
            }
            let meta = self.peer_metas[peer_idx];
            if !meta.supports_host_rdma {
                continue;
            }
            self.ensure_tier_space(TierKind::HostDram)?;
            self.ensure_replenish_tokens(4);
            let outcome = self
                .engine
                .fetch_from_peer_duplex_multi(
                    peer_idx,
                    primary,
                    fallback,
                    self.pd,
                    &mut self.rdma_replenish_tokens,
                    key,
                    host_tier,
                    PullTarget::Host,
                    Tracked(self.caps.shares.borrow_mut()),
                    Tracked(self.caps.instance.borrow()),
                )
                .map_err(CacheError::Core)?;
            match outcome {
                FetchOutcome::Hit { frame, .. } => {
                    self.policy.routing.on_remote_hit(key, peer_idx);
                    self.policy
                        .eviction
                        .on_insert(TierKind::HostDram, key, frame);
                    hit = Some((frame, meta.node_id));
                    break;
                }
                _ => {
                    self.policy.routing.on_remote_miss(key, peer_idx);
                }
            }
        }
        let _ = consumer;
        self.probe_buf = order;
        Ok(hit)
    }

    fn fetch_from_peers_for_gpu(
        &mut self,
        key: ObjectKey,
    ) -> Result<Option<(FrameId, u64, bool)>, CacheError> {
        if self.peer_metas.is_empty() {
            return Ok(None);
        }
        let mut order = core::mem::take(&mut self.probe_buf);
        self.policy
            .routing
            .probe_order(key, &self.peer_metas, &mut order);
        let host_tier = self.engine.host_tier();
        let gpu_tier = self.engine.gpu_tier();
        let (primary, fallback) = self.serve_tier_pair();

        let mut hit = None;
        for &peer_idx in &order {
            if peer_idx >= self.peer_metas.len() {
                self.stats.policy_victims_rejected += 1;
                continue;
            }
            let meta = self.peer_metas[peer_idx];
            let use_gdr =
                gpu_tier.is_some() && meta.supports_gdr && self.policy.routing.prefer_gdr(DataDomain::Gpu, &meta);
            let (dst_tier, dst_kind, pull_target) = if use_gdr {
                (gpu_tier.unwrap(), TierKind::GpuVram, PullTarget::GpuGdr)
            } else {
                (host_tier, TierKind::HostDram, PullTarget::Host)
            };
            self.ensure_tier_space(dst_kind)?;
            self.ensure_replenish_tokens(4);
            let outcome = self
                .engine
                .fetch_from_peer_duplex_multi(
                    peer_idx,
                    primary,
                    fallback,
                    self.pd,
                    &mut self.rdma_replenish_tokens,
                    key,
                    dst_tier,
                    pull_target,
                    Tracked(self.caps.shares.borrow_mut()),
                    Tracked(self.caps.instance.borrow()),
                )
                .map_err(CacheError::Core)?;
            match outcome {
                FetchOutcome::Hit { frame, .. } => {
                    self.policy.routing.on_remote_hit(key, peer_idx);
                    self.policy.eviction.on_insert(dst_kind, key, frame);
                    hit = Some((frame, meta.node_id, use_gdr));
                    break;
                }
                _ => {
                    self.policy.routing.on_remote_miss(key, peer_idx);
                }
            }
        }
        self.probe_buf = order;
        Ok(hit)
    }

    fn reap_completed_spills(&mut self) -> usize {
        if self.pending_spills.is_empty() {
            return 0;
        }
        let _ = self
            .engine
            .poll_nvme_completions(Tracked(self.caps.instance.borrow()), 32);
        let host_tier = self.engine.host_tier();
        let mut retired = 0usize;
        let mut idx = 0usize;
        while idx < self.pending_spills.len() {
            if self
                .engine
                .core
                .timelines
                .is_retired(self.pending_spills[idx].ticket)
            {
                let sp = self.pending_spills.swap_remove(idx);
                let _ = self.engine.drain_and_evict_frame_across_peers(
                    sp.dram_frame,
                    self.cooperative_revoke_spin_budget,
                    Tracked(self.caps.instance.borrow()),
                );
                let _ = self.engine.reclaim_draining(host_tier, 16);
                self.policy
                    .eviction
                    .on_insert(TierKind::LocalNvme, sp.key, sp.nvme_frame);
                retired += 1;
            } else {
                idx += 1;
            }
        }
        retired
    }

    fn wait_pending_spill_for_key(&mut self, key: ObjectKey) -> Result<(), CacheError> {
        while self.pending_spills.iter().any(|p| p.key == key) {
            let _ = self.poll_peers_nonblocking();
            if self.reap_completed_spills() == 0 {
                let _ = self
                    .engine
                    .wait_nvme_completion(Tracked(self.caps.instance.borrow()));
                let _ = self.reap_completed_spills();
            }
        }
        Ok(())
    }

    fn kick_async_spills(&mut self) -> Result<(), CacheError> {
        let Some(wm) = self.policy.tiering.dram_spill_watermark() else {
            return Ok(());
        };
        if self.engine.nvme_tier().is_none() {
            return Ok(());
        }
        let host_tier = self.engine.host_tier();
        self.reap_completed_spills();
        while self.engine.core.frames.ready_queue[host_tier.as_index()].len()
            <= wm.high_water_free_frames
            && self.pending_spills.len() < wm.max_in_flight
        {
            let Some((dk, df)) = self.policy.eviction.select_victim(TierKind::HostDram) else {
                break;
            };
            if !self.frame_valid_published_on_tier(df, host_tier) {
                self.stats.policy_victims_rejected += 1;
                continue;
            }
            if self.ensure_tier_space(TierKind::LocalNvme).is_err() {
                break;
            }
            if self.engine.lookup_tier(dk, host_tier).is_some() {
                match self.engine.spill_host_to_nvme(
                    Tracked(self.caps.instance.borrow()),
                    Tracked(self.caps.shares.borrow_mut()),
                    dk,
                ) {
                    Ok((nf, ticket)) => {
                        self.pending_spills.push(PendingSpill {
                            key: dk,
                            dram_frame: df,
                            nvme_frame: nf,
                            ticket,
                        });
                    }
                    Err(_) => break,
                }
            }
        }
        Ok(())
    }

    fn frame_valid_published_on_tier(&self, frame: FrameId, expected_tier: TierId) -> bool {
        let idx = frame.0 as usize;
        if idx >= self.engine.core.frames.metas.len() {
            return false;
        }
        if self.engine.core.frames.metas[idx].state != STATE_PUBLISHED {
            return false;
        }
        self.engine.core.frames.bindings[idx].1 == expected_tier
    }

    /// Ensure at least one free frame is available in `tier_kind`'s `ready_queue`.
    ///
    /// **Verified Enforcement**:
    /// - Out-of-bounds, wrong-tier, or non-published `FrameId`s suggested by an
    ///   unverified [`EvictionPolicy`] are safely rejected without corrupting state.
    /// - Actively remote-leased frames are fenced via `drain_and_evict_frame_across_peers`
    ///   (cooperative release -> Option-A fenced `LOCAL_INV + ibv_dealloc_mw`) prior to
    ///   reclamation.
    /// - Actively locally-leased frames (`HostLeaseGuard` / `GpuLeaseGuard`) are never
    ///   reclaimed prematurely by `CacheCore::reclaim_draining`.
    fn ensure_tier_space(&mut self, tier_kind: TierKind) -> Result<(), CacheError> {
        let tier_id = self
            .tier_id_for(tier_kind)
            .ok_or(CacheError::Core(CoreError::UnsupportedTier))?;
        let _ = self.engine.reclaim_draining(tier_id, 16);
        if tier_kind == TierKind::HostDram {
            let _ = self.reap_completed_spills();
        }
        if !self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
            return Ok(());
        }

        let max_attempts = self.engine.core.frames.metas.len().saturating_mul(2) + 16;
        let mut attempts = 0usize;
        while self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
            if attempts >= max_attempts {
                return Err(CacheError::Core(CoreError::CapacityExhausted));
            }
            attempts += 1;

            if tier_kind == TierKind::HostDram && !self.pending_spills.is_empty() {
                let _ = self
                    .engine
                    .wait_nvme_completion(Tracked(self.caps.instance.borrow()));
                let _ = self.reap_completed_spills();
                let _ = self.engine.reclaim_draining(tier_id, 16);
                if !self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
                    return Ok(());
                }
            }

            let Some((vkey, vframe)) = self.policy.eviction.select_victim(tier_kind) else {
                let _ = self.poll_peers_nonblocking();
                let _ = self.engine.reclaim_draining(tier_id, 16);
                if !self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
                    return Ok(());
                }
                return Err(CacheError::Core(CoreError::CapacityExhausted));
            };

            // Verified Enforcement Check 1: validate policy hint against CacheCore truth.
            if !self.frame_valid_published_on_tier(vframe, tier_id) {
                self.stats.policy_victims_rejected += 1;
                continue;
            }

            // Optional cascading demotion to colder tier before evicting victim.
            if let Some(demote_to) = self.policy.tiering.demote_on_evict(tier_kind, vkey) {
                match (tier_kind, demote_to) {
                    (TierKind::GpuVram, TierKind::HostDram) => {
                        let host_tier = self.engine.host_tier();
                        if self.engine.lookup_tier(vkey, host_tier).is_none()
                            && self.ensure_tier_space(TierKind::HostDram).is_ok()
                        {
                            if let Ok((_v, new_df)) = self.copy_hop_internal(vframe, host_tier) {
                                self.policy
                                    .eviction
                                    .on_insert(TierKind::HostDram, vkey, new_df);
                            }
                        }
                    }
                    (TierKind::HostDram, TierKind::LocalNvme) => {
                        if let Some(nvme_tier) = self.engine.nvme_tier() {
                            if self.engine.lookup_tier(vkey, nvme_tier).is_none()
                                && self.ensure_tier_space(TierKind::LocalNvme).is_ok()
                            {
                                if let Ok(new_nf) = self.engine.spill_host_to_nvme_sync(
                                    Tracked(self.caps.instance.borrow()),
                                    Tracked(self.caps.shares.borrow_mut()),
                                    vkey,
                                ) {
                                    self.policy
                                        .eviction
                                        .on_insert(TierKind::LocalNvme, vkey, new_nf);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Verified Enforcement Check 2: fence active remote RDMA leases and transition
            // to Draining, then reclaim only when CacheCore proves zero active leases remain.
            let _ = self.engine.drain_and_evict_frame_across_peers(
                vframe,
                self.cooperative_revoke_spin_budget,
                Tracked(self.caps.instance.borrow()),
            );
            let _ = self.poll_peers_nonblocking();
            let _ = self.engine.reclaim_draining(tier_id, 16);

            if self.engine.core.frames.metas[vframe.0 as usize].state != STATE_EMPTY {
                // Victim is still held by an active local lease or unresolved fence.
                // CacheCore strictly prevents premature reclamation.
                self.stats.policy_victims_pinned += 1;
            }
        }
        Ok(())
    }
}

} // verus!
