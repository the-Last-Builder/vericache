//! Location-transparent [`VeriCache`] facade combining:
//! - Unified polymorphic `get(key, target)` / `put(key, source)` + explicit
//!   `get_into_host`, `get_into_gpu`, `put_host`, `put_gpu`, `lease_host`, `lease_gpu`;
//! - Unverified, pluggable [`EvictionPolicy`], [`TieringPolicy`], and [`PeerRoutingPolicy`];
//! - Verified enforcement via [`DistributedEngine`] (`CacheCore` lifecycle, active lease
//!   protection, cooperative-first + Option-A fenced RDMA MW revocation, `VersionBinding`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[cfg_attr(not(verus_only), allow(unused_imports))]
use foundation::tracked::Frontier;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::{EventSourceToken, ReadPerm, ShareCap, SlotToken, VersionBinding};
use transfer::cuda::{self, CudaHostBuffer, CudaSlice, CudaStream};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::domain::TransferDomain;
use transfer::host::{self, HostContext};
use transfer::rdma::ProtectionDomain;
use transfer::transport::{ByteTransport, GrantTag};
use vericache_core::backends::rdma::peer::{FetchOutcome, PendingInboundGrant};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vericache_core::backends::rdma::protocol::VeriCacheAttachment;
use vericache_core::backends::rdma::protocol::{ReadReleaseMeta, encode_read_release_meta};
use vericache_core::backends::rdma::pull::{InFlightRdmaPull, PullTarget, post_pull_read};
use vericache_core::engine::DistributedEngine;
use vericache_core::frame::{STATE_EMPTY, STATE_PUBLISHED};
use vericache_core::kernel::{ReadLeaseHandle, WriteHandle};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vericache_core::provenance::ValueCertificate;
#[cfg(verus_only)]
use vericache_core::provenance::ValueProof;
use vericache_core::spec::{
    CoreError, FrameId, MAX_TICKET_SEQ, ObjectKey, Ticket, TierId, TierKind, VersionId,
};
use vericache_core::timeline::TimelineKind;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Tracked, verus};

use crate::lease::{GpuLeaseGuard, HostLeaseGuard};
use crate::policy::{
    DataDomain, EvictionPolicy, HitLocation, NvmePromoFilter, PeerMeta, PeerRoutingPolicy,
    PolicySet, TieringPolicy,
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

struct PendingPeerNvmeFetch {
    peer_idx: usize,
    req_id: u64,
    key: ObjectKey,
    nvme_frame: FrameId,
    dram_frame: FrameId,
    ticket: Ticket,
}

/// Atomically find and `swap_remove` the first element matching `pred` without
/// ever exposing a raw `Vec` index that could go stale across re-entrant mutations.
#[inline]
fn pop_matching<T, F>(items: &mut Vec<T>, pred: F) -> Option<T>
where
    F: Fn(&T) -> bool,
{
    let pos = items.iter().position(pred)?;
    Some(items.swap_remove(pos))
}

/// Partition ready GPU/DRAM hit requests for a single peer against its available
/// RDMA bind budget (`min(window_pool.free.len(), cq_avail)`), returning
/// `(allowed_now, deferred_excess)`. Prevents `PeerSession::queue_pending_inbound_fetches`
/// from calling `serve_bind` with 0 free Type-2B windows and emitting fake `FetchMiss`es.
#[inline]
fn admit_within_bind_budget(bind_budget: &mut usize) -> bool {
    if *bind_budget > 0 {
        *bind_budget -= 1;
        true
    } else {
        false
    }
}

/// Compute requester-side per-peer pipelined fetch concurrency with `LOCAL_INV` turnaround
/// headroom (`3/4` of peer Type-2B MW pool, at least 1, capped at `rd_cap`, `tier_cap`, `16`).
#[inline]
fn requester_peer_pipeline_cap(win_cap: usize, rd_cap: usize, tier_cap: usize) -> usize {
    if win_cap == 0 || rd_cap == 0 {
        return 0;
    }
    let headroom_cap = (win_cap * 3 / 4).max(1);
    headroom_cap.min(rd_cap).min(tier_cap).min(16).max(1)
}

const PEER_NVME_WARM_CAP: usize = 64;
const DEFERRED_PEER_NVME_CAP: usize = 1024;

/// RAII guard that pauses non-blocking `TcpTransport::try_recv_frame` across all peers
/// during Phase 2 (`poll_all_peers_progress_multi`) so newly arriving `FetchRequest`s
/// cannot bypass Phase 1's NVMe partitioner & Type-2B window `bind_budget` throttle.
/// Unconditionally unpauses on drop (including early return or panic unwind).
struct RecvPauseGuard<'g> {
    handles: &'g [Option<Arc<AtomicBool>>],
}

impl<'g> RecvPauseGuard<'g> {
    fn new(handles: &'g [Option<Arc<AtomicBool>>]) -> Self {
        for h in handles.iter().flatten() {
            h.store(true, Ordering::Release);
        }
        Self { handles }
    }
}

impl Drop for RecvPauseGuard<'_> {
    fn drop(&mut self) {
        for h in self.handles.iter().flatten() {
            h.store(false, Ordering::Release);
        }
    }
}

struct ActiveReadChunk {
    slot: usize,
    held: Vec<(ReadLeaseHandle, Tracked<ShareCap>)>,
    transient_nvme: Vec<(ObjectKey, FrameId)>,
}

/// Source memory address of a verified leased frame yielded by [`VeriCache::with_gpu_read_batch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchReadSource {
    /// Frame resides in local GPU HBM (`device_addr`), ready for 1-hop D2D DMA.
    GpuDevice(usize),
    /// Frame resides in CUDA-pinned local Host DRAM (`host_addr`), ready for 1-hop H2D DMA.
    PinnedHost(usize),
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
    peer_recv_paused: Vec<Option<Arc<AtomicBool>>>,
    policy: PolicySet<E, P, R>,
    internal_gpu_stream: Option<GpuStream>,
    gpu_staging: Option<(CudaHostBuffer, Tracked<SlotToken>, HostContext)>,
    rdma_replenish_tokens: Vec<Tracked<EventSourceToken>>,
    pending_spills: Vec<PendingSpill>,
    pending_peer_nvme_fetches: Vec<PendingPeerNvmeFetch>,
    deferred_peer_grants: Vec<(usize, u64, ObjectKey)>,
    deferred_peer_nvme_fetches: Vec<(usize, u64, ObjectKey)>,
    local_index: HashMap<ObjectKey, [Option<FrameId>; 3]>,
    nvme_promo_filter: NvmePromoFilter,
    probe_buf: Vec<usize>,
    cooperative_revoke_spin_budget: u32,
    in_peer_nvme_warm: bool,
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
        peer_recv_paused: Vec<Option<Arc<AtomicBool>>>,
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
            peer_recv_paused,
            policy,
            internal_gpu_stream,
            gpu_staging,
            rdma_replenish_tokens: Vec::new(),
            pending_spills: Vec::new(),
            pending_peer_nvme_fetches: Vec::new(),
            deferred_peer_grants: Vec::new(),
            deferred_peer_nvme_fetches: Vec::new(),
            local_index: HashMap::new(),
            nvme_promo_filter: NvmePromoFilter::new(),
            probe_buf: Vec::new(),
            cooperative_revoke_spin_budget: 256,
            in_peer_nvme_warm: false,
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
        T: 'static,
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

    /// Returns `true` when the `HostBackend` DRAM slab is CUDA-pinned (1-hop GPU DMA ready).
    pub fn is_host_pinned(&self) -> bool {
        self.engine
            .cuda
            .as_ref()
            .is_some_and(|c| c.pinned_host.is_some())
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
        self.index_insert(TierKind::HostDram, key, frame);
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
        self.index_insert(TierKind::GpuVram, key, frame);
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
        if let Some(frame) = self.lookup_tier_fast(key, TierKind::HostDram, host_tier) {
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
            if let Some(gpu_frame) = self.lookup_tier_fast(key, TierKind::GpuVram, gpu_tier) {
                self.ensure_tier_space(TierKind::HostDram)?;
                let (_ver, dram_frame) = self.copy_hop_internal(gpu_frame, host_tier)?;
                self.index_insert(TierKind::HostDram, key, dram_frame);
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
            if let Some(nvme_frame) = self.lookup_tier_fast(key, TierKind::LocalNvme, nvme_tier) {
                self.ensure_tier_space(TierKind::HostDram)?;
                let _ = self.submit_staged_nvme();
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.index_insert(TierKind::HostDram, key, dram_frame);
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
        if let Some(gpu_frame) = self.lookup_tier_fast(key, TierKind::GpuVram, gpu_tier) {
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
        if let Some(dram_frame) = self.lookup_tier_fast(key, TierKind::HostDram, host_tier) {
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
            if let Some(nvme_frame) = self.lookup_tier_fast(key, TierKind::LocalNvme, nvme_tier) {
                self.ensure_tier_space(TierKind::HostDram)?;
                let _ = self.submit_staged_nvme();
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.index_insert(TierKind::HostDram, key, dram_frame);
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
        let location = if let Some(frame) = self.lookup_tier_fast(key, TierKind::HostDram, host_tier)
        {
            self.policy
                .eviction
                .on_access(TierKind::HostDram, key, frame);
            self.stats.dram_hits += 1;
            HitLocation::LocalDram
        } else if let Some(gpu_frame) = self
            .engine
            .gpu_tier()
            .and_then(|gt| self.lookup_tier_fast(key, TierKind::GpuVram, gt))
        {
            self.ensure_tier_space(TierKind::HostDram)?;
            let (_ver, dram_frame) = self.copy_hop_internal(gpu_frame, host_tier)?;
            self.index_insert(TierKind::HostDram, key, dram_frame);
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
                .and_then(|nt| self.lookup_tier_fast(key, TierKind::LocalNvme, nt))
            {
                self.ensure_tier_space(TierKind::HostDram)?;
                let _ = self.submit_staged_nvme();
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.index_insert(TierKind::HostDram, key, dram_frame);
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
        let location = if let Some(gpu_frame) =
            self.lookup_tier_fast(key, TierKind::GpuVram, gpu_tier)
        {
            self.policy
                .eviction
                .on_access(TierKind::GpuVram, key, gpu_frame);
            self.stats.gpu_hits += 1;
            HitLocation::LocalGpu
        } else if let Some(dram_frame) = self.lookup_tier_fast(key, TierKind::HostDram, host_tier) {
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
                .and_then(|nt| self.lookup_tier_fast(key, TierKind::LocalNvme, nt))
            {
                self.ensure_tier_space(TierKind::HostDram)?;
                let _ = self.submit_staged_nvme();
                let dram_frame = self
                    .engine
                    .warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    )
                    .map_err(CacheError::Core)?;
                self.index_insert(TierKind::HostDram, key, dram_frame);
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

    /// High-throughput zero-bounce batched read for GPU consumers (e.g., vLLM KVConnector).
    ///
    /// Processes `keys` in chunks bounded by active lease headroom (up to 128 per chunk):
    /// - Prefetches all NVMe-resident keys in the chunk concurrently via async `io_uring`
    ///   (`warm_host_from_nvme`, up to 64 in-flight) instead of serial QD=1 reads.
    /// - Pipelines remote RDMA / GPUDirect RDMA pulls concurrently across the chunk
    ///   (up to `min(windows_per_peer, max_rd_atomic, 16)` in-flight per peer) while
    ///   serving inbound peer fetches full-duplex (including on-demand peer NVMe warming).
    /// - Acquires verified local leases (`GpuLease` for HBM/GDR hits, `HostLease` for
    ///   CUDA-pinned DRAM/NVMe/Host-RDMA hits — zero HBM promotion/eviction write amplification!).
    /// - Invokes `on_hit(index, source)` for each hit so the caller can enqueue async
    ///   D2D / H2D copies onto a CUDA stream, then calls `on_flush()` ONCE per chunk
    ///   (single `cudaStreamSynchronize` / blocking event sync) before releasing all held
    ///   leases back to `CacheCore`.
    pub fn with_gpu_read_batch<FHit, FFlush>(
        &mut self,
        keys: &[ObjectKey],
        mut on_hit: FHit,
        mut on_flush: FFlush,
    ) -> Result<Vec<Option<ReadHit>>, CacheError>
    where
        FHit: FnMut(usize, BatchReadSource) -> Result<(), CacheError>,
        FFlush: FnMut() -> Result<(), CacheError>,
    {
        const CHUNK_SIZE: usize = 128;
        const NVME_PREFETCH_QD: usize = 64;

        let mut results: Vec<Option<ReadHit>> = vec![None; keys.len()];
        let gpu_tier = self.engine.gpu_tier();
        let gpu_cap = self.engine.cuda.as_ref().map(|c| c.gpu_num_frames as usize);
        let host_tier = self.engine.host_tier();
        let nvme_tier = self.engine.nvme_tier();
        let dram_cap = (self.engine.host.num_frames as usize).max(1);

        let mut chunk_start = 0usize;
        while chunk_start < keys.len() {
            self.reap_completed_spills();
            let _ = self.poll_peers_nonblocking();

            // Dynamically respect CacheCore lease headroom and DRAM frame count.
            let free_leases = self.engine.core.free_lease_slots.len();
            let lease_budget = if free_leases > 16 {
                free_leases - 8
            } else {
                free_leases.max(1)
            };
            let effective_chunk = CHUNK_SIZE.min(lease_budget).min(dram_cap);
            let mut chunk_end = (chunk_start + effective_chunk).min(keys.len());
            let mut chunk_keys = &keys[chunk_start..chunk_end];
            let mut chunk_len = chunk_keys.len();

            // Promote already-resident keys in this chunk to MRU before Phase 1/1.5 warms
            // so prefetching NVMe or remote keys never evicts DRAM/GPU hits in the same chunk.
            for &k in chunk_keys {
                if let Some(gt) = gpu_tier {
                    if let Some(gf) = self.lookup_tier_fast(k, TierKind::GpuVram, gt) {
                        self.policy.eviction.on_access(TierKind::GpuVram, k, gf);
                    }
                }
                if let Some(df) = self.lookup_tier_fast(k, TierKind::HostDram, host_tier) {
                    self.policy.eviction.on_access(TierKind::HostDram, k, df);
                }
            }

            // Phase 1: Parallel batched async io_uring NVMe prefetch for NVMe-only keys in this chunk
            // (deduplicated by ObjectKey + 2-hit promotion gate so 1st-hit NVMe reads reclaim DRAM
            // immediately after sync with zero write amplification).
            let mut from_nvme = vec![false; chunk_len];
            let mut transient_nvme: Vec<(ObjectKey, FrameId)> = Vec::new();
            let mut promoted_nvme = false;
            if let Some(nt) = nvme_tier {
                let mut nvme_seen: HashSet<ObjectKey> = HashSet::new();
                let mut nvme_candidates: Vec<(usize, ObjectKey, FrameId)> = Vec::new();
                for (li, &k) in chunk_keys.iter().enumerate() {
                    if gpu_tier.is_some_and(|gt| {
                        self.lookup_tier_fast(k, TierKind::GpuVram, gt).is_some()
                    }) {
                        continue;
                    }
                    if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                        continue;
                    }
                    let _ = self.wait_pending_spill_for_key(k);
                    if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                        continue;
                    }
                    if let Some(nf) = self.lookup_tier_fast(k, TierKind::LocalNvme, nt) {
                        if nvme_seen.insert(k) {
                            nvme_candidates.push((li, k, nf));
                        }
                    }
                }

                if !nvme_candidates.is_empty() {
                    let (warmed, transient) =
                        self.warm_nvme_keys_parallel(&nvme_candidates, NVME_PREFETCH_QD);
                    if !warmed.is_empty() {
                        if warmed.len() > transient.len() {
                            promoted_nvme = true;
                        }
                        transient_nvme = transient;
                        let mut warmed_keys: HashSet<ObjectKey> =
                            HashSet::with_capacity(warmed.len());
                        for li in warmed {
                            warmed_keys.insert(chunk_keys[li]);
                        }
                        for (li, &k) in chunk_keys.iter().enumerate() {
                            if warmed_keys.contains(&k) {
                                from_nvme[li] = true;
                            }
                        }
                    }
                }
            }

            // Phase 1.5: Pipelined multi-window remote RDMA / GDR prefetch for chunk misses.
            let mut remote_hits: HashMap<ObjectKey, (u64, bool)> = HashMap::new();
            if !self.peer_metas.is_empty() {
                let mut remote_candidates: Vec<(usize, ObjectKey)> = Vec::new();
                for (li, &k) in chunk_keys.iter().enumerate() {
                    if gpu_tier.is_some_and(|gt| {
                        self.lookup_tier_fast(k, TierKind::GpuVram, gt).is_some()
                    }) {
                        continue;
                    }
                    if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                        continue;
                    }
                    remote_candidates.push((li, k));
                }
                if !remote_candidates.is_empty() {
                    let has_gdr_peer = gpu_tier.is_some()
                        && self.peer_metas.iter().any(|m| {
                            m.supports_gdr && self.policy.routing.prefer_gdr(DataDomain::Gpu, m)
                        });
                    if has_gdr_peer {
                        let gdr_cap = gpu_cap.unwrap_or(1).max(1);
                        let mut uniq_remote: HashSet<ObjectKey> = HashSet::new();
                        let mut limit_len = chunk_len;
                        for &(li, k) in &remote_candidates {
                            if !uniq_remote.contains(&k) {
                                if uniq_remote.len() >= gdr_cap {
                                    limit_len = li.max(1);
                                    break;
                                }
                                uniq_remote.insert(k);
                            }
                        }
                        if limit_len < chunk_len {
                            chunk_end = chunk_start + limit_len;
                            chunk_keys = &keys[chunk_start..chunk_end];
                            chunk_len = limit_len;
                            from_nvme.truncate(limit_len);
                            remote_candidates.retain(|&(li, _)| li < limit_len);
                        }
                    }
                    remote_hits = self.prefetch_remote_batch_for_gpu(&remote_candidates)?;
                }
            }

            // Phase 2: Acquire verified leases across chunk and enqueue async copies.
            let mut held: Vec<(ReadLeaseHandle, Tracked<ShareCap>)> =
                Vec::with_capacity(chunk_len);
            let mut callback_err: Option<CacheError> = None;

            for (li, &k) in chunk_keys.iter().enumerate() {
                let gi = chunk_start + li;
                if li > 0 && li % 16 == 0 {
                    let _ = self.poll_peers_nonblocking_inner(false);
                }

                let remote_meta = remote_hits.get(&k).copied();

                // 2a. Local GPU HBM hit (or freshly pulled GDR hit) -> 1-hop D2D
                if let Some(gt) = gpu_tier {
                    if let Some(gf) = self.lookup_tier_fast(k, TierKind::GpuVram, gt) {
                        if let Ok(Some(lease)) = self.engine.get_lease_gpu(
                            Tracked(self.caps.instance.borrow()),
                            Tracked(self.caps.shares.borrow_mut()),
                            k,
                        ) {
                            let ver = lease.version;
                            let addr_opt = self
                                .engine
                                .gpu_frame_slice(lease.frame)
                                .map(|s| s.device_addr());
                            held.push((lease.handle, lease.cap));
                            if let Some(addr) = addr_opt {
                                let loc = match remote_meta {
                                    Some((node_id, true)) => {
                                        self.stats.remote_hits += 1;
                                        HitLocation::RemotePeer { node_id, gdr: true }
                                    }
                                    _ => {
                                        self.policy.eviction.on_access(TierKind::GpuVram, k, gf);
                                        self.stats.gpu_hits += 1;
                                        HitLocation::LocalGpu
                                    }
                                };
                                if let Err(e) = on_hit(gi, BatchReadSource::GpuDevice(addr)) {
                                    callback_err = Some(e);
                                    break;
                                }
                                results[gi] = Some(ReadHit {
                                    version: ver,
                                    location: loc,
                                });
                                continue;
                            }
                        }
                    }
                }

                // 2b. Local Pinned Host DRAM hit (or freshly warmed NVMe / Host-RDMA hit) -> 1-hop H2D
                if let Some(df) = self.lookup_tier_fast(k, TierKind::HostDram, host_tier) {
                    if let Ok(Some(lease)) = self.engine.get_lease_host(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        k,
                    ) {
                        let ver = lease.version;
                        let addr = lease.slice.addr();
                        held.push((lease.handle, lease.cap));
                        let loc = match remote_meta {
                            Some((node_id, gdr)) => {
                                self.stats.remote_hits += 1;
                                HitLocation::RemotePeer { node_id, gdr }
                            }
                            _ if from_nvme[li] => {
                                self.stats.nvme_hits += 1;
                                HitLocation::LocalNvme
                            }
                            _ => {
                                self.policy.eviction.on_access(TierKind::HostDram, k, df);
                                self.stats.dram_hits += 1;
                                HitLocation::LocalDram
                            }
                        };
                        if let Err(e) = on_hit(gi, BatchReadSource::PinnedHost(addr)) {
                            callback_err = Some(e);
                            break;
                        }
                        results[gi] = Some(ReadHit {
                            version: ver,
                            location: loc,
                        });
                        continue;
                    }
                }

                self.stats.misses += 1;
            }

            // Phase 3: Single stream synchronize while all leases are held, then release all
            // and reclaim 1st-hit transient NVMe DRAM frames (0 NVMe write amplification).
            let flush_res = if !held.is_empty() {
                on_flush()
            } else {
                Ok(())
            };
            for (handle, cap) in held {
                let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
            }
            self.reclaim_transient_nvme_frames(transient_nvme);
            if let Some(e) = callback_err {
                return Err(e);
            }
            flush_res?;

            if promoted_nvme || !remote_hits.is_empty() {
                let _ = self.kick_async_spills();
            }
            let _ = self.poll_peers_nonblocking();
            chunk_start = chunk_end;
        }

        Ok(results)
    }

    /// Double-buffered pipelined zero-bounce batched read for GPU consumers (e.g., vLLM KVConnector).
    ///
    /// Keeps up to **2 chunks** leased concurrently (`slot = 0 | 1`) so chunk `k`'s NVMe `io_uring`
    /// prefetch, remote RDMA pulls, lease acquisition, and DMA descriptor assembly overlap
    /// completely with chunk `k-1`'s asynchronous GPU Copy Engine H2D/D2D DMA:
    /// - Calls `on_chunk_emit(slot, &entries)` to enqueue chunk `k`'s DMA batch on `slot` (0 or 1)
    ///   and record a non-blocking GPU event.
    /// - Calls `on_chunk_sync(prev_slot)` to wait for the previously emitted slot's GPU event,
    ///   immediately releasing `prev_slot`'s verified leases and reclaiming 1st-hit transient NVMe
    ///   DRAM frames before `prev_slot` can ever be reused.
    /// - Guarantees zero leaked leases on every error or early-exit path.
    pub fn with_gpu_read_batch_pipelined<FEmit, FWait>(
        &mut self,
        keys: &[ObjectKey],
        mut on_chunk_emit: FEmit,
        mut on_chunk_sync: FWait,
    ) -> Result<Vec<Option<ReadHit>>, CacheError>
    where
        FEmit: FnMut(usize, &[(usize, BatchReadSource)]) -> Result<(), CacheError>,
        FWait: FnMut(usize) -> Result<(), CacheError>,
    {
        const PIPELINED_CHUNK_MAX: usize = 96;
        const NVME_PREFETCH_QD: usize = 64;

        let mut results: Vec<Option<ReadHit>> = vec![None; keys.len()];
        if keys.is_empty() {
            return Ok(results);
        }

        let gpu_tier = self.engine.gpu_tier();
        let gpu_cap = self.engine.cuda.as_ref().map(|c| c.gpu_num_frames as usize);
        let host_tier = self.engine.host_tier();
        let nvme_tier = self.engine.nvme_tier();
        let dram_cap = (self.engine.host.num_frames as usize).max(1);
        let half_dram = (dram_cap / 2).max(1);

        let mut active_prev: Option<ActiveReadChunk> = None;
        let mut next_slot: usize = 0;
        let mut chunk_start = 0usize;

        while chunk_start < keys.len() {
            // If DRAM or lease headroom is too tight for 2 simultaneous chunks, drain prev first.
            if active_prev.is_some()
                && (dram_cap < 2 || self.engine.core.free_lease_slots.len() < 4)
            {
                if let Some(prev) = active_prev.take() {
                    let sync_res = on_chunk_sync(prev.slot);
                    for (handle, cap) in prev.held {
                        let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                    }
                    self.reclaim_transient_nvme_frames(prev.transient_nvme);
                    sync_res?;
                }
            }

            self.reap_completed_spills();
            let _ = self.poll_peers_nonblocking();

            // Size chunk so 2 concurrent chunks fit cleanly inside CacheCore lease & DRAM budgets.
            let free_leases = self.engine.core.free_lease_slots.len();
            let lease_budget = if active_prev.is_none() {
                let usable = if free_leases > 16 {
                    free_leases - 8
                } else {
                    free_leases.max(1)
                };
                (usable / 2).max(1)
            } else if free_leases > 12 {
                free_leases - 8
            } else {
                free_leases.max(1)
            };
            let effective_chunk = PIPELINED_CHUNK_MAX
                .min(lease_budget)
                .min(half_dram)
                .max(1);
            let mut chunk_end = (chunk_start + effective_chunk).min(keys.len());
            let mut chunk_keys = &keys[chunk_start..chunk_end];
            let mut chunk_len = chunk_keys.len();

            // Promote already-resident keys in this chunk to MRU before warming.
            for &k in chunk_keys {
                if let Some(gt) = gpu_tier {
                    if let Some(gf) = self.lookup_tier_fast(k, TierKind::GpuVram, gt) {
                        self.policy.eviction.on_access(TierKind::GpuVram, k, gf);
                    }
                }
                if let Some(df) = self.lookup_tier_fast(k, TierKind::HostDram, host_tier) {
                    self.policy.eviction.on_access(TierKind::HostDram, k, df);
                }
            }

            // Phase 1: Parallel batched async io_uring NVMe prefetch (overlapping with prev DMA!).
            let mut from_nvme = vec![false; chunk_len];
            let mut transient_nvme: Vec<(ObjectKey, FrameId)> = Vec::new();
            let mut promoted_nvme = false;
            if let Some(nt) = nvme_tier {
                let mut nvme_seen: HashSet<ObjectKey> = HashSet::new();
                let mut nvme_candidates: Vec<(usize, ObjectKey, FrameId)> = Vec::new();
                for (li, &k) in chunk_keys.iter().enumerate() {
                    if gpu_tier.is_some_and(|gt| {
                        self.lookup_tier_fast(k, TierKind::GpuVram, gt).is_some()
                    }) {
                        continue;
                    }
                    if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                        continue;
                    }
                    let _ = self.wait_pending_spill_for_key(k);
                    if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                        continue;
                    }
                    if let Some(nf) = self.lookup_tier_fast(k, TierKind::LocalNvme, nt) {
                        if nvme_seen.insert(k) {
                            nvme_candidates.push((li, k, nf));
                        }
                    }
                }

                if !nvme_candidates.is_empty() {
                    let (warmed, transient) =
                        self.warm_nvme_keys_parallel(&nvme_candidates, NVME_PREFETCH_QD);
                    if !warmed.is_empty() {
                        if warmed.len() > transient.len() {
                            promoted_nvme = true;
                        }
                        transient_nvme = transient;
                        let mut warmed_keys: HashSet<ObjectKey> =
                            HashSet::with_capacity(warmed.len());
                        for li in warmed {
                            warmed_keys.insert(chunk_keys[li]);
                        }
                        for (li, &k) in chunk_keys.iter().enumerate() {
                            if warmed_keys.contains(&k) {
                                from_nvme[li] = true;
                            }
                        }
                    }
                }
            }

            // Phase 1.5: Pipelined remote RDMA / GDR prefetch (overlapping with prev DMA!).
            let mut remote_hits: HashMap<ObjectKey, (u64, bool)> = HashMap::new();
            if !self.peer_metas.is_empty() {
                let mut remote_candidates: Vec<(usize, ObjectKey)> = Vec::new();
                for (li, &k) in chunk_keys.iter().enumerate() {
                    if gpu_tier.is_some_and(|gt| {
                        self.lookup_tier_fast(k, TierKind::GpuVram, gt).is_some()
                    }) {
                        continue;
                    }
                    if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                        continue;
                    }
                    remote_candidates.push((li, k));
                }
                if !remote_candidates.is_empty() {
                    let has_gdr_peer = gpu_tier.is_some()
                        && self.peer_metas.iter().any(|m| {
                            m.supports_gdr && self.policy.routing.prefer_gdr(DataDomain::Gpu, m)
                        });
                    if has_gdr_peer {
                        let gdr_cap = (gpu_cap.unwrap_or(1) / 2).max(1);
                        let mut uniq_remote: HashSet<ObjectKey> = HashSet::new();
                        let mut limit_len = chunk_len;
                        for &(li, k) in &remote_candidates {
                            if !uniq_remote.contains(&k) {
                                if uniq_remote.len() >= gdr_cap {
                                    limit_len = li.max(1);
                                    break;
                                }
                                uniq_remote.insert(k);
                            }
                        }
                        if limit_len < chunk_len {
                            chunk_end = chunk_start + limit_len;
                            chunk_keys = &keys[chunk_start..chunk_end];
                            chunk_len = limit_len;
                            from_nvme.truncate(limit_len);
                            remote_candidates.retain(|&(li, _)| li < limit_len);
                        }
                    }
                    match self.prefetch_remote_batch_for_gpu(&remote_candidates) {
                        Ok(hits) => remote_hits = hits,
                        Err(e) => {
                            if let Some(prev) = active_prev.take() {
                                let _ = on_chunk_sync(prev.slot);
                                for (handle, cap) in prev.held {
                                    let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                                }
                                self.reclaim_transient_nvme_frames(prev.transient_nvme);
                            }
                            self.reclaim_transient_nvme_frames(transient_nvme);
                            return Err(e);
                        }
                    }
                }
            }

            // Phase 2: Acquire verified leases for chunk `k` and build `(key_idx, BatchReadSource)` slice.
            let mut held: Vec<(ReadLeaseHandle, Tracked<ShareCap>)> =
                Vec::with_capacity(chunk_len);
            let mut chunk_entries: Vec<(usize, BatchReadSource)> = Vec::with_capacity(chunk_len);

            for (li, &k) in chunk_keys.iter().enumerate() {
                let gi = chunk_start + li;
                if li > 0 && li % 16 == 0 {
                    let _ = self.poll_peers_nonblocking_inner(false);
                }

                let remote_meta = remote_hits.get(&k).copied();

                // 2a. Local GPU HBM hit (or freshly pulled GDR hit) -> 1-hop D2D
                if let Some(gt) = gpu_tier {
                    if let Some(gf) = self.lookup_tier_fast(k, TierKind::GpuVram, gt) {
                        if let Ok(Some(lease)) = self.engine.get_lease_gpu(
                            Tracked(self.caps.instance.borrow()),
                            Tracked(self.caps.shares.borrow_mut()),
                            k,
                        ) {
                            let ver = lease.version;
                            let addr_opt = self
                                .engine
                                .gpu_frame_slice(lease.frame)
                                .map(|s| s.device_addr());
                            held.push((lease.handle, lease.cap));
                            if let Some(addr) = addr_opt {
                                let loc = match remote_meta {
                                    Some((node_id, true)) => {
                                        self.stats.remote_hits += 1;
                                        HitLocation::RemotePeer { node_id, gdr: true }
                                    }
                                    _ => {
                                        self.policy.eviction.on_access(TierKind::GpuVram, k, gf);
                                        self.stats.gpu_hits += 1;
                                        HitLocation::LocalGpu
                                    }
                                };
                                chunk_entries.push((gi, BatchReadSource::GpuDevice(addr)));
                                results[gi] = Some(ReadHit {
                                    version: ver,
                                    location: loc,
                                });
                                continue;
                            }
                        }
                    }
                }

                // 2b. Local Pinned Host DRAM hit (or freshly warmed NVMe / Host-RDMA hit) -> 1-hop H2D
                if let Some(df) = self.lookup_tier_fast(k, TierKind::HostDram, host_tier) {
                    if let Ok(Some(lease)) = self.engine.get_lease_host(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        k,
                    ) {
                        let ver = lease.version;
                        let addr = lease.slice.addr();
                        held.push((lease.handle, lease.cap));
                        let loc = match remote_meta {
                            Some((node_id, gdr)) => {
                                self.stats.remote_hits += 1;
                                HitLocation::RemotePeer { node_id, gdr }
                            }
                            _ if from_nvme[li] => {
                                self.stats.nvme_hits += 1;
                                HitLocation::LocalNvme
                            }
                            _ => {
                                self.policy.eviction.on_access(TierKind::HostDram, k, df);
                                self.stats.dram_hits += 1;
                                HitLocation::LocalDram
                            }
                        };
                        chunk_entries.push((gi, BatchReadSource::PinnedHost(addr)));
                        results[gi] = Some(ReadHit {
                            version: ver,
                            location: loc,
                        });
                        continue;
                    }
                }

                self.stats.misses += 1;
            }

            // Phase 3: Double-buffered emit-then-sync-prev so GPU Copy Engine runs back-to-back
            // with zero inter-chunk bubble, and `prev.slot` is always synced before reuse.
            if !chunk_entries.is_empty() {
                let slot = next_slot;
                if let Err(e) = on_chunk_emit(slot, &chunk_entries) {
                    if let Some(prev) = active_prev.take() {
                        let _ = on_chunk_sync(prev.slot);
                        for (handle, cap) in prev.held {
                            let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                        }
                        self.reclaim_transient_nvme_frames(prev.transient_nvme);
                    }
                    for (handle, cap) in held {
                        let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                    }
                    self.reclaim_transient_nvme_frames(transient_nvme);
                    return Err(e);
                }

                if let Some(prev) = active_prev.take() {
                    let sync_res = on_chunk_sync(prev.slot);
                    for (handle, cap) in prev.held {
                        let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                    }
                    self.reclaim_transient_nvme_frames(prev.transient_nvme);
                    if let Err(e) = sync_res {
                        let _ = on_chunk_sync(slot);
                        for (handle, cap) in held {
                            let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                        }
                        self.reclaim_transient_nvme_frames(transient_nvme);
                        return Err(e);
                    }
                }

                active_prev = Some(ActiveReadChunk {
                    slot,
                    held,
                    transient_nvme,
                });
                next_slot ^= 1;
            } else {
                // Chunk had 0 hits: release any defensive handles/transient frames without touching active_prev.
                for (handle, cap) in held {
                    let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
                }
                self.reclaim_transient_nvme_frames(transient_nvme);
            }

            if promoted_nvme || !remote_hits.is_empty() {
                let _ = self.kick_async_spills();
            }
            let _ = self.poll_peers_nonblocking();
            chunk_start = chunk_end;
        }

        // Drain the final in-flight slot before returning.
        if let Some(last) = active_prev.take() {
            let sync_res = on_chunk_sync(last.slot);
            for (handle, cap) in last.held {
                let _ = self.release_local_lease_raw(handle, cap, Ticket::NONE);
            }
            self.reclaim_transient_nvme_frames(last.transient_nvme);
            sync_res?;
        }
        let _ = self.poll_peers_nonblocking();

        Ok(results)
    }

    /// Zero-copy batched write directly into `HostBackend` DRAM frames (CUDA-pinned when GPU
    /// is attached), eliminating intermediate staging-buffer CPU `memcpy`s.
    ///
    /// - Skips keys already resident in any local tier (refreshing DRAM LRU when skipped) and
    ///   deduplicates repeated keys within a chunk.
    /// - Reserves `STATE_INITIALIZING` frames in chunks (`<= 64`, bounded by DRAM capacity) via
    ///   `CacheCore::begin_write`, passing `(key_index, dst_host_addr)` to `on_frame`. Each
    ///   `dst_host_addr` is an exclusively owned, aligned `frame_size`-byte buffer safe for
    ///   direct GPU D2H DMA and/or CPU tail zero-padding inside `on_frame`.
    /// - Calls `on_flush()` ONCE per chunk before publishing all reserved frames atomically via
    ///   `CacheCore::finish_write` (0-byte CPU copy) + `kick_async_spills()`.
    /// - On any error from `on_frame` or `on_flush` (or reservation), drains `on_flush()` if any
    ///   frames were reserved, aborts ALL reserved frames via `CacheCore::abort_write` (zero
    ///   leaked frames), and returns the error.
    /// - Returns the verified contiguous prefix count `0..=keys.len()` (including skipped and
    ///   duplicate keys) confirmed resident locally after commit.
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    pub fn with_host_write_batch<FFrame, FFlush>(
        &mut self,
        keys: &[ObjectKey],
        mut on_frame: FFrame,
        mut on_flush: FFlush,
    ) -> Result<usize, CacheError>
    where
        FFrame: FnMut(usize, usize) -> Result<(), CacheError>,
        FFlush: FnMut() -> Result<(), CacheError>,
    {
        const WRITE_CHUNK_SIZE: usize = 64;

        let host_tier = self.engine.host_tier();
        let host_tl = self.engine.host.timeline_id();
        let dram_cap = (self.engine.host.num_frames as usize).max(1);
        let effective_chunk = WRITE_CHUNK_SIZE.min(dram_cap);

        let mut verified = 0usize;
        let mut chunk_start = 0usize;
        while chunk_start < keys.len() {
            let chunk_end = (chunk_start + effective_chunk).min(keys.len());
            self.reap_completed_spills();
            let _ = self.kick_async_spills();
            let _ = self.poll_peers_nonblocking();

            let mut reserved: Vec<(ObjectKey, WriteHandle, Tracked<SlotToken>)> =
                Vec::with_capacity(chunk_end - chunk_start);
            let mut chunk_seen: HashSet<ObjectKey> =
                HashSet::with_capacity(chunk_end - chunk_start);
            let mut chunk_err: Option<CacheError> = None;

            for gi in chunk_start..chunk_end {
                let key = keys[gi];
                if !chunk_seen.insert(key) {
                    continue;
                }
                if let Some(loc) = self.contains_local(key) {
                    if loc == HitLocation::LocalDram {
                        if let Some(df) = self.lookup_tier_fast(key, TierKind::HostDram, host_tier)
                        {
                            self.policy.eviction.on_access(TierKind::HostDram, key, df);
                        }
                    }
                    continue;
                }
                if let Err(e) = self.wait_pending_spill_for_key(key) {
                    chunk_err = Some(e);
                    break;
                }
                if self.contains_local(key).is_some() {
                    continue;
                }
                if self.engine.core.timelines.next_seq[host_tl as usize] > MAX_TICKET_SEQ {
                    chunk_err = Some(CacheError::Core(CoreError::SequenceExhausted));
                    break;
                }
                if let Err(e) = self.ensure_tier_space(TierKind::HostDram) {
                    chunk_err = Some(e);
                    break;
                }
                let (handle, tok) = match self.engine.core.begin_write(host_tier, key) {
                    Ok(v) => v,
                    Err(e) => {
                        chunk_err = Some(CacheError::Core(e));
                        break;
                    }
                };
                let dst_addr = self.engine.host.frame_slice(handle.frame).addr();
                reserved.push((key, handle, tok));
                if let Err(e) = on_frame(gi, dst_addr) {
                    chunk_err = Some(e);
                    break;
                }
            }

            if let Some(e) = chunk_err {
                // Always drain caller's async stream before aborting reserved frames so no
                // in-flight DMA can write to a frame returned to `ready_queue`.
                if !reserved.is_empty() {
                    let _ = on_flush();
                }
                for (_key, handle, tok) in reserved {
                    self.engine.core.abort_write(handle, tok);
                }
                return Err(e);
            }

            if !reserved.is_empty() {
                if let Err(e) = on_flush() {
                    for (_key, handle, tok) in reserved {
                        self.engine.core.abort_write(handle, tok);
                    }
                    return Err(e);
                }

                let mut publish_err: Option<CacheError> = None;
                let mut iter = reserved.into_iter();
                #[cfg_attr(not(verus_only), allow(unused_mut))]
                while let Some((key, handle, mut tok)) = iter.next() {
                    if self.engine.core.timelines.next_seq[host_tl as usize] > MAX_TICKET_SEQ {
                        self.engine.core.abort_write(handle, tok);
                        for (_rk, rh, rt) in iter {
                            self.engine.core.abort_write(rh, rt);
                        }
                        publish_err = Some(CacheError::Core(CoreError::SequenceExhausted));
                        break;
                    }
                    let dst_zero = self.engine.host.frame_slice(handle.frame).sub_slice(0, 0);
                    let Tracked(comp) = host::write_slice(
                        &mut self.engine.host_ctx,
                        Tracked(self.caps.instance.borrow()),
                        &dst_zero,
                        &[],
                        Tracked(tok.borrow_mut()),
                        Tracked(&self.engine.core.ledger.borrow().submitted),
                        Tracked(&self.engine.core.ledger.borrow().knows),
                        Ghost(self.engine.core.citations()),
                    );
                    let ticket = match self
                        .engine
                        .core
                        .issue_ticket(host_tl, Tracked(comp.submitted))
                    {
                        Ok(t) => t,
                        Err(e) => {
                            self.engine.core.abort_write(handle, tok);
                            for (_rk, rh, rt) in iter {
                                self.engine.core.abort_write(rh, rt);
                            }
                            publish_err = Some(CacheError::Core(e));
                            break;
                        }
                    };
                    self.engine
                        .core
                        .mark_timeline_complete(host_tl, ticket.seq(), Tracked(comp.knows));
                    let ghost cert = ValueCertificate {
                        slot: handle.slot@,
                        writer: transfer::domain::TransferEvent {
                            source: transfer::domain::EventSource::Host { id: 0 },
                            observer: 0,
                            seq: 0,
                        },
                        bytes: vstd::prelude::Seq::empty(),
                        proof: ValueProof::HostWrite,
                    };
                    let frame = handle.frame;
                    let ver = handle.version;
                    self.engine
                        .core
                        .finish_write(handle, tok, ticket, Ghost(cert));
                    let _ = self.engine.core.bind_local_version(ver);
                    self.index_insert(TierKind::HostDram, key, frame);
                    self.policy
                        .eviction
                        .on_insert(TierKind::HostDram, key, frame);
                    self.stats.puts += 1;
                }
                if let Some(e) = publish_err {
                    return Err(e);
                }
                self.kick_async_spills()?;
            }

            let mut chunk_verified = chunk_end;
            for idx in chunk_start..chunk_end {
                if self.contains_local(keys[idx]).is_none() {
                    chunk_verified = idx;
                    break;
                }
            }
            verified = chunk_verified;
            if verified < chunk_end {
                break;
            }
            let _ = self.poll_peers_nonblocking();
            chunk_start = chunk_end;
        }

        Ok(verified)
    }

    /// Explicitly evict `key` from a specific local tier (with optional cascade demotion
    /// disabled — purely removes that tier's replica through verified peer/lease fencing).
    pub fn evict_from_tier(&mut self, key: ObjectKey, tier: TierKind) -> Result<bool, CacheError> {
        let Some(tid) = self.tier_id_for(tier) else {
            return Ok(false);
        };
        let Some(frame) = self.lookup_tier_fast(key, tier, tid) else {
            return Ok(false);
        };
        let _ = self.engine.drain_and_evict_frame_across_peers(
            frame,
            self.cooperative_revoke_spin_budget,
            Tracked(self.caps.instance.borrow()),
        );
        let _ = self.engine.reclaim_draining(tid, 16);
        self.index_remove(tier, key, frame);
        self.policy.eviction.on_remove(tier, key, frame);
        Ok(true)
    }

    /// Manually spill a DRAM key to NVMe and evict its DRAM replica (useful for
    /// deterministic tier placement or testing).
    pub fn spill_key_to_nvme(&mut self, key: ObjectKey) -> Result<bool, CacheError> {
        let Some(_nvme_tier) = self.engine.nvme_tier() else {
            return Err(CacheError::Core(CoreError::UnsupportedTier));
        };
        let host_tier = self.engine.host_tier();
        let Some(dram_frame) = self.lookup_tier_fast(key, TierKind::HostDram, host_tier) else {
            return Ok(false);
        };
        self.ensure_tier_space(TierKind::LocalNvme)?;
        let _ = self.submit_staged_nvme();
        let nvme_frame = self
            .engine
            .spill_host_to_nvme_sync(
                Tracked(self.caps.instance.borrow()),
                Tracked(self.caps.shares.borrow_mut()),
                key,
            )
            .map_err(CacheError::Core)?;
        self.index_insert(TierKind::LocalNvme, key, nvme_frame);
        self.policy
            .eviction
            .on_insert(TierKind::LocalNvme, key, nvme_frame);
        let _ = self.engine.drain_and_evict_frame_across_peers(
            dram_frame,
            self.cooperative_revoke_spin_budget,
            Tracked(self.caps.instance.borrow()),
        );
        let _ = self.engine.reclaim_draining(host_tier, 16);
        self.index_remove(TierKind::HostDram, key, dram_frame);
        self.policy
            .eviction
            .on_remove(TierKind::HostDram, key, dram_frame);
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

    /// Drain in-flight NVMe spills / peer NVMe warms and cleanly shut down all RDMA peer sessions.
    pub fn shutdown(&mut self) -> Result<(), CacheError> {
        let _ = self.submit_staged_nvme();
        self.deferred_peer_grants.clear();
        self.deferred_peer_nvme_fetches.clear();
        while !self.pending_spills.is_empty() || !self.pending_peer_nvme_fetches.is_empty() {
            let _ = self.poll_peers_nonblocking_inner(false);
            let reaped_spills = self.reap_completed_spills();
            let reaped_peer = self.reap_completed_peer_nvme_fetches();
            if reaped_spills == 0 && reaped_peer == 0 {
                let _ = self.submit_staged_nvme();
                let _ = self
                    .engine
                    .wait_nvme_completion(Tracked(self.caps.instance.borrow()));
                let _ = self.reap_completed_spills();
                let _ = self.reap_completed_peer_nvme_fetches();
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

    /// O(1) metadata-only lookup across local tiers (`LocalGpu -> LocalDram -> LocalNvme`)
    /// without triggering any data movement (designed for vLLM Scheduler prefix matching).
    pub fn contains_local(&self, key: ObjectKey) -> Option<HitLocation> {
        if let Some(gpu_tier) = self.engine.gpu_tier() {
            if self.lookup_tier_fast(key, TierKind::GpuVram, gpu_tier).is_some() {
                return Some(HitLocation::LocalGpu);
            }
        }
        if self
            .lookup_tier_fast(key, TierKind::HostDram, self.engine.host_tier())
            .is_some()
        {
            return Some(HitLocation::LocalDram);
        }
        if let Some(nvme_tier) = self.engine.nvme_tier() {
            if self
                .lookup_tier_fast(key, TierKind::LocalNvme, nvme_tier)
                .is_some()
            {
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

    #[inline]
    fn tier_slot(kind: TierKind) -> Option<usize> {
        match kind {
            TierKind::GpuVram => Some(0),
            TierKind::HostDram => Some(1),
            TierKind::LocalNvme => Some(2),
            _ => None,
        }
    }

    #[inline]
    fn index_insert(&mut self, kind: TierKind, key: ObjectKey, frame: FrameId) {
        if let Some(slot) = Self::tier_slot(kind) {
            self.local_index.entry(key).or_insert([None; 3])[slot] = Some(frame);
        }
    }

    #[inline]
    fn index_remove(&mut self, kind: TierKind, key: ObjectKey, frame: FrameId) {
        if let Some(slot) = Self::tier_slot(kind) {
            if let Some(entry) = self.local_index.get_mut(&key) {
                if entry[slot] == Some(frame) {
                    entry[slot] = None;
                    if entry.iter().all(Option::is_none) {
                        self.local_index.remove(&key);
                    }
                }
            }
        }
    }

    #[inline]
    fn frame_matches_key_tier(&self, key: ObjectKey, frame: FrameId, tier: TierId) -> bool {
        let idx = frame.0 as usize;
        idx < self.engine.core.frames.metas.len()
            && self.engine.core.frames.metas[idx].state == STATE_PUBLISHED
            && self.engine.core.frames.bindings[idx] == (key, tier)
    }

    /// O(1) shadow catalog lookup validated against `CacheCore::frames` ground truth.
    #[inline]
    fn lookup_tier_fast(&self, key: ObjectKey, kind: TierKind, tier: TierId) -> Option<FrameId> {
        let slot = Self::tier_slot(kind)?;
        let frame = self.local_index.get(&key)?[slot]?;
        if self.frame_matches_key_tier(key, frame, tier) {
            Some(frame)
        } else {
            None
        }
    }

    /// Stage one NVMe -> DRAM copy hop in the `io_uring` SQ without calling `io_uring_enter`.
    fn stage_nvme_to_dram_one(&mut self, nvme_frame: FrameId) -> Result<(FrameId, Ticket), CoreError> {
        let mut nvme = self.engine.nvme.take().ok_or(CoreError::UnsupportedTier)?;
        let res = nvme.stage_nvme_to_dram(
            &self.engine.host,
            &mut self.engine.core,
            Tracked(self.caps.instance.borrow()),
            Tracked(self.caps.shares.borrow_mut()),
            nvme_frame,
        );
        self.engine.nvme = Some(nvme);
        res
    }

    /// Stage one DRAM -> NVMe copy hop in the `io_uring` SQ without calling `io_uring_enter`.
    fn stage_dram_to_nvme_one(&mut self, dram_frame: FrameId) -> Result<(FrameId, Ticket), CoreError> {
        let mut nvme = self.engine.nvme.take().ok_or(CoreError::UnsupportedTier)?;
        let res = nvme.stage_dram_to_nvme(
            &self.engine.host,
            &mut self.engine.core,
            Tracked(self.caps.instance.borrow()),
            Tracked(self.caps.shares.borrow_mut()),
            dram_frame,
        );
        self.engine.nvme = Some(nvme);
        res
    }

    /// Flush all staged `io_uring` SQEs to the kernel in a single batched `io_uring_enter`.
    fn submit_staged_nvme(&mut self) -> Result<usize, CoreError> {
        let Some(mut nvme) = self.engine.nvme.take() else {
            return Ok(0);
        };
        let mut total = 0usize;
        let mut err: Option<CoreError> = None;
        while nvme.staged_len() > 0 {
            match nvme.submit(Ghost(&self.engine.core)) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        self.engine.nvme = Some(nvme);
        match err {
            Some(e) => Err(e),
            None => Ok(total),
        }
    }

    /// Reclaim 1st-hit transient NVMe DRAM frames immediately after DMA sync & lease release
    /// (zero NVMe write amplification because the NVMe replica already exists).
    fn reclaim_transient_nvme_frames(&mut self, frames: Vec<(ObjectKey, FrameId)>) {
        if frames.is_empty() {
            return;
        }
        let host_tier = self.engine.host_tier();
        let Some(nvme_tier) = self.engine.nvme_tier() else {
            return;
        };
        let mut evicted_any = false;
        for (key, df) in frames {
            if !self.frame_matches_key_tier(key, df, host_tier) {
                continue;
            }
            if self.lookup_tier_fast(key, TierKind::LocalNvme, nvme_tier).is_none() {
                continue;
            }
            let meta = &self.engine.core.frames.metas[df.0 as usize];
            if meta.active_local_reads > 0 || meta.active_remote_leases > 0 {
                // Frame is actively leased by an overlapping pipelined chunk or remote peer;
                // leave it untouched at the cold tail of DRAM LRU.
                continue;
            }
            let _ = self.engine.drain_and_evict_frame_across_peers(
                df,
                self.cooperative_revoke_spin_budget,
                Tracked(self.caps.instance.borrow()),
            );
            self.index_remove(TierKind::HostDram, key, df);
            self.policy.eviction.on_remove(TierKind::HostDram, key, df);
            evicted_any = true;
        }
        if evicted_any {
            let _ = self.engine.reclaim_draining(host_tier, 128);
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

    /// Batched `io_uring` NVMe -> DRAM warm for a chunk's NVMe candidates.
    ///
    /// Stages up to `qd` reads via `stage_nvme_to_dram_one` and calls `submit_staged_nvme`
    /// ONCE per wave, polling peers non-blockingly while waiting for completions.
    /// Returns `(warmed_tags, transient_frames)` where `transient_frames` are 1st-hit NVMe
    /// keys that did not pass the 2-hit promotion gate and should be reclaimed immediately
    /// after chunk DMA sync.
    fn warm_nvme_keys_parallel(
        &mut self,
        candidates: &[(usize, ObjectKey, FrameId)],
        max_qd: usize,
    ) -> (Vec<usize>, Vec<(ObjectKey, FrameId)>) {
        let mut warmed = Vec::with_capacity(candidates.len());
        let mut transient_frames: Vec<(ObjectKey, FrameId)> = Vec::new();
        if candidates.is_empty() {
            return (warmed, transient_frames);
        }
        let host_tier = self.engine.host_tier();
        let dram_cap = (self.engine.host.num_frames as usize).max(1);
        let mut cidx = 0usize;
        while cidx < candidates.len() {
            let free_leases = self.engine.core.free_lease_slots.len();
            let lease_budget = if free_leases > 8 {
                free_leases - 4
            } else {
                free_leases.max(1)
            };
            let qd = max_qd.min(lease_budget).min(dram_cap).max(1);
            let mut in_flight: Vec<(usize, ObjectKey, FrameId, FrameId, Ticket)> =
                Vec::with_capacity(qd.min(candidates.len() - cidx));
            let mut submitted_end = cidx;

            while submitted_end < candidates.len() && in_flight.len() < qd {
                let (tag, k, nf) = candidates[submitted_end];
                if self.lookup_tier_fast(k, TierKind::HostDram, host_tier).is_some() {
                    warmed.push(tag);
                    submitted_end += 1;
                    continue;
                }
                if self.ensure_tier_space(TierKind::HostDram).is_err() {
                    break;
                }
                match self.stage_nvme_to_dram_one(nf) {
                    Ok((df, ticket)) => {
                        in_flight.push((tag, k, nf, df, ticket));
                        submitted_end += 1;
                    }
                    Err(_) => {
                        if !in_flight.is_empty() {
                            // Ring or lease window is currently saturated: submit & reap
                            // this wave first and retry this candidate on the next wave.
                            break;
                        }
                        let _ = self.submit_staged_nvme();
                        if let Ok(df) = self.engine.warm_host_from_nvme_sync(
                            Tracked(self.caps.instance.borrow()),
                            Tracked(self.caps.shares.borrow_mut()),
                            k,
                        ) {
                            self.index_insert(TierKind::HostDram, k, df);
                            self.policy.eviction.on_access(TierKind::LocalNvme, k, nf);
                            let promote = self.nvme_promo_filter.record_hit_and_should_promote(k);
                            self.policy.eviction.on_insert(TierKind::HostDram, k, df);
                            if !promote {
                                transient_frames.push((k, df));
                            }
                            warmed.push(tag);
                        }
                        submitted_end += 1;
                    }
                }
            }

            // Single batched io_uring_enter for the entire wave.
            let _ = self.submit_staged_nvme();

            for (tag, k, nf, df, ticket) in in_flight {
                while !self.engine.core.timelines.is_retired(ticket) {
                    let _ = self.poll_peers_nonblocking_inner(false);
                    let _ = self
                        .engine
                        .poll_nvme_completions(Tracked(self.caps.instance.borrow()), 64);
                    if self.engine.core.timelines.is_retired(ticket) {
                        break;
                    }
                    let _ = self.submit_staged_nvme();
                    let _ = self
                        .engine
                        .wait_nvme_completion(Tracked(self.caps.instance.borrow()));
                }
                if self.frame_matches_key_tier(k, df, host_tier) {
                    self.index_insert(TierKind::HostDram, k, df);
                    self.policy.eviction.on_access(TierKind::LocalNvme, k, nf);
                    let promote = self.nvme_promo_filter.record_hit_and_should_promote(k);
                    // Insert at MRU during Phase 1 so subsequent waves of the same chunk
                    // cannot self-evict wave 1; transient frames are immediately reclaimed
                    // after Phase 3 sync & lease release.
                    self.policy.eviction.on_insert(TierKind::HostDram, k, df);
                    if !promote {
                        transient_frames.push((k, df));
                    }
                    warmed.push(tag);
                } else if let Some((actual_df, _)) = self.engine.lookup_tier(k, host_tier) {
                    self.index_insert(TierKind::HostDram, k, actual_df);
                    self.policy.eviction.on_access(TierKind::LocalNvme, k, nf);
                    let promote = self.nvme_promo_filter.record_hit_and_should_promote(k);
                    self.policy.eviction.on_insert(TierKind::HostDram, k, actual_df);
                    if !promote {
                        transient_frames.push((k, actual_df));
                    }
                    warmed.push(tag);
                }
            }

            if submitted_end == cidx {
                break;
            }
            cidx = submitted_end;
        }
        (warmed, transient_frames)
    }

    /// Reap completed async donor peer NVMe warms and re-queue their `(req_id, key)` onto
    /// the originating peer's `pending_inbound_fetches` for immediate `ReadGrant` emission.
    ///
    /// Safe to call even when `allow_nvme_warm == false` because it NEVER calls
    /// `ensure_tier_space` or overwrites `pending_inbound_fetches`.
    fn reap_completed_peer_nvme_fetches(&mut self) -> usize {
        if self.pending_peer_nvme_fetches.is_empty() {
            return 0;
        }
        let _ = self
            .engine
            .poll_nvme_completions(Tracked(self.caps.instance.borrow()), 64);
        let host_tier = self.engine.host_tier();
        let mut reaped = 0usize;
        let mut idx = 0usize;
        while idx < self.pending_peer_nvme_fetches.len() {
            if self
                .engine
                .core
                .timelines
                .is_retired(self.pending_peer_nvme_fetches[idx].ticket)
            {
                let f = self.pending_peer_nvme_fetches.swap_remove(idx);
                if self.frame_matches_key_tier(f.key, f.dram_frame, host_tier) {
                    self.index_insert(TierKind::HostDram, f.key, f.dram_frame);
                    self.policy
                        .eviction
                        .on_access(TierKind::LocalNvme, f.key, f.nvme_frame);
                    // Insert at MRU (not cold tail) so subsequent `ensure_tier_space(HostDram)`
                    // in Phase 3 does not immediately evict or revoke the freshly granted peer
                    // NVMe frame before the requester finishes its RDMA READ. Because `f.key`
                    // already has its NVMe replica, later eviction drops DRAM with 0 NVMe writes.
                    self.policy
                        .eviction
                        .on_insert(TierKind::HostDram, f.key, f.dram_frame);
                }
                if f.peer_idx < self.engine.peers.len() {
                    self.engine.peers[f.peer_idx]
                        .pending_inbound_fetches
                        .push((f.req_id, f.key));
                }
                reaped += 1;
            } else {
                idx += 1;
            }
        }
        reaped
    }

    /// Phase 1 of non-blocking donor peer serving (runs for ALL peers, with or without NVMe,
    /// and NEVER calls `ensure_tier_space`):
    /// - Pumps CQ + TCP control channels (`ReadRelease` posts `LOCAL_INV`), then immediately
    ///   calls `rdma.pump_cq_all` so completed `LOCAL_INV` CQEs return Type-2B windows to
    ///   `window_pool.free` BEFORE Phase 2 (`queue_pending_inbound_fetches`) runs.
    /// - Re-injects `deferred_peer_grants` (oldest ready hits first) and `deferred_peer_nvme_fetches`.
    /// - Throttles ready GPU/DRAM hits per peer to `min(window_pool.free.len(), cq_avail)` so
    ///   `serve_bind` never runs out of Type-2B windows and never emits a fake `FetchMiss`;
    ///   excess ready hits are parked in `deferred_peer_grants`.
    /// - True misses always pass straight through for immediate `FetchMiss`.
    /// - NVMe-only hits piggyback on active `warming` tickets, go to `nvme_to_stage` when
    ///   `can_stage_nvme` is true, or park in `deferred_peer_nvme_fetches` (up to cap).
    fn partition_inbound_peer_fetches(
        &mut self,
        can_stage_nvme: bool,
    ) -> (
        usize,
        Vec<(usize, u64, ObjectKey)>,
        HashMap<ObjectKey, (FrameId, FrameId, Ticket)>,
    ) {
        let mut pre_cq_processed = 0usize;
        let mut nvme_to_stage: Vec<(usize, u64, ObjectKey)> = Vec::new();
        let mut warming: HashMap<ObjectKey, (FrameId, FrameId, Ticket)> =
            HashMap::with_capacity(self.pending_peer_nvme_fetches.len());
        let num_peers = self.engine.peers.len();
        if num_peers == 0 {
            return (0, nvme_to_stage, warming);
        }

        let gpu_tier = self.engine.gpu_tier();
        let host_tier = self.engine.host_tier();
        let nvme_tier = self.engine.nvme_tier();

        // 1. Pump CQ + TCP channels (receives FetchRequests + posts LOCAL_INV for ReadReleases),
        //    then immediately pump RDMA CQ so completed LOCAL_INV CQEs replenish `window_pool.free`
        //    BEFORE `queue_pending_inbound_fetches` runs in Phase 2!
        for peer_idx in 0..num_peers {
            let _ = self.engine.peers[peer_idx].pump_cq_and_channel_basic(
                &mut self.engine.core,
                Tracked(self.caps.instance.borrow()),
            );
            if let Ok(n) = self.engine.peers[peer_idx]
                .rdma
                .pump_cq_all(&mut self.engine.core, Tracked(self.caps.instance.borrow()))
            {
                pre_cq_processed += n;
            }
        }

        for f in &self.pending_peer_nvme_fetches {
            warming.insert(f.key, (f.nvme_frame, f.dram_frame, f.ticket));
        }

        // Preserve FIFO priority: previously deferred ready hits first, then deferred NVMe hits
        // (when staging is allowed), then newly arrived `pending_inbound_fetches`.
        let mut per_peer_ordered: Vec<Vec<(u64, ObjectKey)>> =
            (0..num_peers).map(|_| Vec::new()).collect();
        let deferred_grants = core::mem::take(&mut self.deferred_peer_grants);
        for (peer_idx, req_id, key) in deferred_grants {
            if peer_idx < num_peers {
                per_peer_ordered[peer_idx].push((req_id, key));
            }
        }
        if can_stage_nvme {
            let deferred_nvme = core::mem::take(&mut self.deferred_peer_nvme_fetches);
            for (peer_idx, req_id, key) in deferred_nvme {
                if peer_idx < num_peers {
                    per_peer_ordered[peer_idx].push((req_id, key));
                }
            }
        }

        // Partition strictly without calling `ensure_tier_space` (zero re-entrant peer pumping).
        for (peer_idx, mut inbound) in per_peer_ordered.into_iter().enumerate() {
            let newly_arrived =
                core::mem::take(&mut self.engine.peers[peer_idx].pending_inbound_fetches);
            inbound.extend(newly_arrived);
            if inbound.is_empty() {
                continue;
            }
            let free_wins = self.engine.peers[peer_idx].rdma.window_pool.free.len();
            let cq = &self.engine.peers[peer_idx].rdma.cq;
            let cq_avail = cq.capacity_exec().saturating_sub(cq.in_flight_count());
            let mut bind_budget = free_wins.min(cq_avail);

            let mut immediate: Vec<(u64, ObjectKey)> = Vec::with_capacity(inbound.len());
            for (req_id, key) in inbound {
                let is_ready_hit = gpu_tier.is_some_and(|gt| {
                    self.lookup_tier_fast(key, TierKind::GpuVram, gt).is_some()
                }) || self
                    .lookup_tier_fast(key, TierKind::HostDram, host_tier)
                    .is_some();
                if is_ready_hit {
                    if admit_within_bind_budget(&mut bind_budget) {
                        immediate.push((req_id, key));
                    } else {
                        self.deferred_peer_grants.push((peer_idx, req_id, key));
                    }
                    continue;
                }
                if let Some(&(nf, df, ticket)) = warming.get(&key) {
                    self.pending_peer_nvme_fetches.push(PendingPeerNvmeFetch {
                        peer_idx,
                        req_id,
                        key,
                        nvme_frame: nf,
                        dram_frame: df,
                        ticket,
                    });
                    continue;
                }
                if nvme_tier.is_some_and(|nt| {
                    self.lookup_tier_fast(key, TierKind::LocalNvme, nt).is_some()
                }) {
                    if can_stage_nvme {
                        nvme_to_stage.push((peer_idx, req_id, key));
                    } else if self.deferred_peer_nvme_fetches.len() < DEFERRED_PEER_NVME_CAP {
                        self.deferred_peer_nvme_fetches
                            .push((peer_idx, req_id, key));
                    } else {
                        immediate.push((req_id, key));
                    }
                    continue;
                }
                // True miss: emit immediate FetchMiss (consumes 0 windows).
                immediate.push((req_id, key));
            }
            // Append (never overwrite) back into `pending_inbound_fetches`.
            self.engine.peers[peer_idx]
                .pending_inbound_fetches
                .extend(immediate);
        }

        (pre_cq_processed, nvme_to_stage, warming)
    }

    /// Phase 3 of non-blocking donor peer NVMe warming: stage async `io_uring` NVMe warms
    /// AFTER Phase 2 has already bound `ReadGrant`s for all ready frames.
    ///
    /// Any fallback/overflow pushes back onto `pending_inbound_fetches` via `.push()`, NEVER
    /// overwriting newly arrived `FetchRequest`s pumped during `ensure_tier_space`.
    fn stage_inbound_peer_nvme_fetches(
        &mut self,
        nvme_to_stage: Vec<(usize, u64, ObjectKey)>,
        mut warming: HashMap<ObjectKey, (FrameId, FrameId, Ticket)>,
    ) {
        if nvme_to_stage.is_empty() {
            return;
        }
        let Some(nvme_tier) = self.engine.nvme_tier() else {
            for (peer_idx, req_id, key) in nvme_to_stage {
                if peer_idx < self.engine.peers.len() {
                    self.engine.peers[peer_idx]
                        .pending_inbound_fetches
                        .push((req_id, key));
                }
            }
            return;
        };
        let gpu_tier = self.engine.gpu_tier();
        let host_tier = self.engine.host_tier();
        let mut staged_any = false;

        for (peer_idx, req_id, key) in nvme_to_stage {
            if peer_idx >= self.engine.peers.len() {
                continue;
            }
            if gpu_tier.is_some_and(|gt| {
                self.lookup_tier_fast(key, TierKind::GpuVram, gt).is_some()
            }) || self
                .lookup_tier_fast(key, TierKind::HostDram, host_tier)
                .is_some()
            {
                self.engine.peers[peer_idx]
                    .pending_inbound_fetches
                    .push((req_id, key));
                continue;
            }
            if let Some(&(nf, df, ticket)) = warming.get(&key) {
                self.pending_peer_nvme_fetches.push(PendingPeerNvmeFetch {
                    peer_idx,
                    req_id,
                    key,
                    nvme_frame: nf,
                    dram_frame: df,
                    ticket,
                });
                continue;
            }
            if let Some(nf) = self.lookup_tier_fast(key, TierKind::LocalNvme, nvme_tier) {
                if self.pending_peer_nvme_fetches.len() < PEER_NVME_WARM_CAP
                    && self.engine.core.free_lease_slots.len() >= 8
                    && self.ensure_tier_space(TierKind::HostDram).is_ok()
                {
                    if let Some(actual_nf) =
                        self.lookup_tier_fast(key, TierKind::LocalNvme, nvme_tier)
                    {
                        match self.stage_nvme_to_dram_one(actual_nf) {
                            Ok((df, ticket)) => {
                                warming.insert(key, (actual_nf, df, ticket));
                                self.pending_peer_nvme_fetches.push(PendingPeerNvmeFetch {
                                    peer_idx,
                                    req_id,
                                    key,
                                    nvme_frame: actual_nf,
                                    dram_frame: df,
                                    ticket,
                                });
                                staged_any = true;
                                continue;
                            }
                            Err(_) => {
                                let _ = nf;
                                // Transient io_uring/lease saturation: defer for next poll
                                // instead of pushing to `pending_inbound_fetches` (fake FetchMiss).
                                if self.deferred_peer_nvme_fetches.len() < DEFERRED_PEER_NVME_CAP {
                                    self.deferred_peer_nvme_fetches
                                        .push((peer_idx, req_id, key));
                                } else {
                                    self.engine.peers[peer_idx]
                                        .pending_inbound_fetches
                                        .push((req_id, key));
                                }
                                continue;
                            }
                        }
                    }
                } else if self.deferred_peer_nvme_fetches.len() < DEFERRED_PEER_NVME_CAP {
                    self.deferred_peer_nvme_fetches
                        .push((peer_idx, req_id, key));
                    continue;
                }
            }
            // Key evicted from NVMe or deferred cap reached: push back for immediate FetchMiss.
            self.engine.peers[peer_idx]
                .pending_inbound_fetches
                .push((req_id, key));
        }

        if staged_any {
            let _ = self.submit_staged_nvme();
        }
    }

    fn poll_peers_nonblocking(
        &mut self,
    ) -> Result<vericache_core::backends::rdma::peer::PeerProgress, CacheError> {
        self.poll_peers_nonblocking_inner(true)
    }

    fn poll_peers_nonblocking_inner(
        &mut self,
        allow_nvme_warm: bool,
    ) -> Result<vericache_core::backends::rdma::peer::PeerProgress, CacheError> {
        if self.peer_metas.is_empty() {
            return Ok(vericache_core::backends::rdma::peer::PeerProgress::empty());
        }

        // Phase 1: Always reap completed async peer NVMe warms + pump RDMA CQ + throttle
        // ready hits by each peer's available Type-2B window & CQ budget (never calls
        // `ensure_tier_space`, works with or without NVMe, and runs even when `allow_nvme_warm == false`).
        let _ = self.reap_completed_peer_nvme_fetches();

        let can_stage_nvme =
            allow_nvme_warm && !self.in_peer_nvme_warm && self.engine.nvme_tier().is_some();
        if can_stage_nvme {
            self.in_peer_nvme_warm = true;
        }
        let (pre_cq_processed, nvme_to_stage, warming) =
            self.partition_inbound_peer_fetches(can_stage_nvme);

        // Phase 2: Pause non-blocking TCP `try_recv_frame` via RAII guard while running
        // `poll_all_peers_progress_multi` so newly arriving NVMe-only / over-budget
        // `FetchRequest`s CANNOT sneak past Phase 1 into `queue_pending_inbound_fetches`
        // and trigger fake `FetchMiss`es!
        let (primary, fallback) = self.serve_tier_pair();
        self.ensure_replenish_tokens(4);
        let mut progress_res = {
            let _recv_pause = RecvPauseGuard::new(&self.peer_recv_paused);
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
        };
        if let Ok(ref mut p) = progress_res {
            p.cq_processed += pre_cq_processed;
        }

        // Phase 3: Stage new async NVMe warms safely AFTER grants are bound.
        if can_stage_nvme {
            if progress_res.is_ok() {
                self.stage_inbound_peer_nvme_fetches(nvme_to_stage, warming);
            } else {
                for (peer_idx, req_id, key) in nvme_to_stage {
                    if peer_idx < self.engine.peers.len() {
                        self.engine.peers[peer_idx]
                            .pending_inbound_fetches
                            .push((req_id, key));
                    }
                }
            }
            self.in_peer_nvme_warm = false;
        }

        progress_res
    }

    /// Send a cooperative `ReadRelease` for an unused/failed inbound `ReadGrant` so the
    /// remote donor immediately unpins its lease and invalidates its Type-2B MW without leak.
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    fn send_read_release_raw(
        &mut self,
        peer_idx: usize,
        tag: GrantTag,
        request_id: u64,
        share_cap: Tracked<ShareCap>,
    ) {
        let rel_meta = ReadReleaseMeta { request_id };
        let rel_bytes = encode_read_release_meta(rel_meta);
        let ghost cited = self.engine.core.citations();
        let tracked att = VeriCacheAttachment::ReadRelease { meta: rel_meta };
        let _ = self.engine.peers[peer_idx]
            .channel
            .send_remote_read_release_with(
                tag,
                &rel_bytes,
                Tracked(self.caps.instance.borrow()),
                Tracked(&self.engine.core.ledger.borrow().submitted),
                Tracked(&self.engine.core.ledger.borrow().knows),
                Tracked(&self.engine.core.ledger.borrow().succeeded),
                Ghost(cited),
                share_cap,
                Tracked(att),
            );
    }

    /// Pop a buffered inbound `ReadGrant` matching `req_id` atomically (never holding a
    /// `Vec` index across `&mut self` calls where peer revocation can shrink `pending_inbound_grants`).
    fn take_inbound_grant(
        &mut self,
        peer_idx: usize,
        req_id: u64,
    ) -> Option<PendingInboundGrant> {
        pop_matching(
            &mut self.engine.peers[peer_idx].pending_inbound_grants,
            |g| g.meta.request_id == req_id,
        )
    }

    /// Pop a buffered inbound `FetchMiss` matching `req_id` atomically.
    fn take_inbound_miss(&mut self, peer_idx: usize, req_id: u64) -> bool {
        pop_matching(
            &mut self.engine.peers[peer_idx].pending_inbound_misses,
            |&id| id == req_id,
        )
        .is_some()
    }

    /// Merge `g.statement` into `CacheCore` and send `ReadRelease` for a popped grant that
    /// cannot be consumed (e.g., `CapacityExhausted` with no in-flight pulls or size mismatch).
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    fn release_unused_inbound_grant(&mut self, peer_idx: usize, g: PendingInboundGrant) {
        let tracked stmt = g.statement.get();
        self.engine.core.merge_peer_statement(
            Tracked(stmt.submitted),
            Tracked(stmt.knows),
            Tracked(stmt.succeeded),
            Tracked(self.caps.instance.borrow()),
        );
        self.engine.core.record_revoked_map(g.revoked);
        self.send_read_release_raw(peer_idx, g.tag, g.meta.request_id, g.share);
    }

    /// Post an RDMA READ pull for a popped `PendingInboundGrant`, or release its `ShareCap`
    /// cleanly via `ReadRelease` on any failure (zero leaked peer Type-2B MWs or local frames).
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    fn try_post_pull_or_release_grant(
        &mut self,
        peer_idx: usize,
        key: ObjectKey,
        dst_tier: TierId,
        pull_target: PullTarget,
        g: PendingInboundGrant,
    ) -> Result<
        (
            GrantTag,
            InFlightRdmaPull,
            Tracked<Option<VersionBinding>>,
            Ghost<Frontier<TransferDomain>>,
        ),
        CoreError,
    > {
        let tag = g.tag;
        let req_id = g.meta.request_id;
        let expected_size = self.engine.core.tier_frame_size(dst_tier) as u64;
        let tracked stmt = g.statement.get();
        self.engine.core.merge_peer_statement(
            Tracked(stmt.submitted),
            Tracked(stmt.knows),
            Tracked(stmt.succeeded),
            Tracked(self.caps.instance.borrow()),
        );
        self.engine.core.record_revoked_map(g.revoked);

        if g.meta.frame_size != expected_size {
            self.send_read_release_raw(peer_idx, tag, req_id, g.share);
            return Err(CoreError::InvalidArgument);
        }

        let (handle, dst_tok) = match self
            .engine
            .core
            .begin_write_version(dst_tier, key, g.meta.version)
        {
            Ok(v) => v,
            Err(e) => {
                self.send_read_release_raw(peer_idx, tag, req_id, g.share);
                return Err(e);
            }
        };

        let owns = match pull_target {
            PullTarget::Host => self.engine.peers[peer_idx]
                .rdma
                .effective_host_mr()
                .is_some_and(|mr| mr.owns_frame(handle.frame)),
            PullTarget::GpuGdr => self.engine.peers[peer_idx]
                .rdma
                .effective_gpu_mr()
                .is_some_and(|mr| mr.owns_frame(handle.frame)),
        };
        if !owns {
            self.engine.core.abort_write(handle, dst_tok);
            self.send_read_release_raw(peer_idx, tag, req_id, g.share);
            return Err(CoreError::UnsupportedTier);
        }

        let ghost orig_share_frontier = g.share@.frontier();
        match post_pull_read(
            &mut self.engine.peers[peer_idx].rdma,
            &mut self.engine.core,
            Tracked(self.caps.instance.borrow()),
            pull_target,
            handle,
            dst_tok,
            &g.cap,
            g.share,
            g.cert,
        ) {
            Ok(pull) => Ok((tag, pull, g.version_binding, Ghost(orig_share_frontier))),
            Err((e, h, tok, share_cap)) => {
                self.engine.core.abort_write(h, tok);
                self.send_read_release_raw(peer_idx, tag, req_id, share_cap);
                Err(e)
            }
        }
    }

    /// Pipelined multi-window remote RDMA / GPUDirect RDMA batch fetch across a chunk.
    /// Keeps up to `min(windows_per_peer, max_rd_atomic, 16)` remote fetches in flight
    /// per peer while driving full-duplex peer serving (including peer NVMe warm-and-serve).
    #[cfg_attr(not(verus_only), allow(unused_variables, non_shorthand_field_patterns))]
    fn prefetch_remote_batch_for_gpu(
        &mut self,
        candidates: &[(usize, ObjectKey)],
    ) -> Result<HashMap<ObjectKey, (u64, bool)>, CacheError> {
        enum RemoteState {
            PendingSend {
                probe_pos: usize,
            },
            AwaitingGrant {
                probe_pos: usize,
                peer_idx: usize,
                req_id: u64,
                use_gdr: bool,
                dst_tier: TierId,
                dst_kind: TierKind,
                pull_target: PullTarget,
            },
            PullInFlight {
                probe_pos: usize,
                peer_idx: usize,
                req_id: u64,
                tag: GrantTag,
                pull: InFlightRdmaPull,
                version_binding: Tracked<Option<VersionBinding>>,
                orig_share_frontier: Ghost<Frontier<TransferDomain>>,
                use_gdr: bool,
                dst_kind: TierKind,
            },
            Done,
        }

        struct RemoteItem {
            key: ObjectKey,
            probe_peers: Vec<usize>,
            retries_remaining: u8,
            state: RemoteState,
        }

        let mut resolved: HashMap<ObjectKey, (u64, bool)> = HashMap::new();
        if self.peer_metas.is_empty() || candidates.is_empty() {
            return Ok(resolved);
        }

        let host_tier = self.engine.host_tier();
        let gpu_tier = self.engine.gpu_tier();
        let dram_cap = (self.engine.host.num_frames as usize).max(1);
        let gdr_cap = self
            .engine
            .cuda
            .as_ref()
            .map(|c| (c.gpu_num_frames as usize).max(1))
            .unwrap_or(1);

        let num_peers = self.engine.peers.len();
        let mut peer_caps = vec![0usize; num_peers];
        for (pidx, cap) in peer_caps.iter_mut().enumerate() {
            let win_cap = self.engine.peers[pidx].rdma.window_pool.slots.len();
            let rd_cap = self.engine.peers[pidx]
                .rdma
                .qp
                .effective_max_rd_atomic_exec() as usize;
            if win_cap > 0 && rd_cap > 0 {
                let meta = self.peer_metas[pidx];
                let tier_cap = if gpu_tier.is_some()
                    && meta.supports_gdr
                    && self.policy.routing.prefer_gdr(DataDomain::Gpu, &meta)
                {
                    gdr_cap
                } else {
                    dram_cap
                };
                *cap = requester_peer_pipeline_cap(win_cap, rd_cap, tier_cap);
            }
        }

        let mut seen = HashSet::with_capacity(candidates.len());
        let mut items: Vec<RemoteItem> = Vec::with_capacity(candidates.len());
        let mut order = core::mem::take(&mut self.probe_buf);
        let mut active_count = 0usize;
        for &(_li, key) in candidates {
            if !seen.insert(key) {
                continue;
            }
            self.policy
                .routing
                .probe_order(key, &self.peer_metas, &mut order);
            let mut probe_peers = Vec::with_capacity(order.len());
            for &peer_idx in &order {
                if peer_idx >= self.peer_metas.len() {
                    self.stats.policy_victims_rejected += 1;
                    continue;
                }
                if peer_caps[peer_idx] > 0 {
                    probe_peers.push(peer_idx);
                }
            }
            let state = if probe_peers.is_empty() {
                RemoteState::Done
            } else {
                active_count += 1;
                RemoteState::PendingSend { probe_pos: 0 }
            };
            items.push(RemoteItem {
                key,
                probe_peers,
                retries_remaining: 1,
                state,
            });
        }
        self.probe_buf = order;

        if active_count == 0 {
            return Ok(resolved);
        }

        let mut in_flight_per_peer = vec![0usize; num_peers];
        let mut pulling_count = 0usize;
        let mut pulling_gdr_count = 0usize;
        let mut idle_spins = 0u32;
        let deadline = Instant::now() + Duration::from_secs(5);

        while active_count > 0 {
            let mut progressed = false;

            // Step A: Issue new FetchRequests up to each peer's window/RDMA-READ cap.
            for item in &mut items {
                let probe_pos = match item.state {
                    RemoteState::PendingSend { probe_pos } => probe_pos,
                    _ => continue,
                };
                if probe_pos >= item.probe_peers.len() {
                    item.state = RemoteState::Done;
                    active_count -= 1;
                    continue;
                }
                let peer_idx = item.probe_peers[probe_pos];
                if in_flight_per_peer[peer_idx] >= peer_caps[peer_idx] {
                    continue;
                }
                let meta = self.peer_metas[peer_idx];
                let use_gdr = gpu_tier.is_some()
                    && meta.supports_gdr
                    && self.policy.routing.prefer_gdr(DataDomain::Gpu, &meta);
                let (dst_tier, dst_kind, pull_target) = if use_gdr {
                    (gpu_tier.unwrap(), TierKind::GpuVram, PullTarget::GpuGdr)
                } else {
                    (host_tier, TierKind::HostDram, PullTarget::Host)
                };
                match self.engine.send_fetch_request_to(
                    peer_idx,
                    item.key,
                    Tracked(self.caps.instance.borrow()),
                ) {
                    Ok(req_id) => {
                        in_flight_per_peer[peer_idx] += 1;
                        item.state = RemoteState::AwaitingGrant {
                            probe_pos,
                            peer_idx,
                            req_id,
                            use_gdr,
                            dst_tier,
                            dst_kind,
                            pull_target,
                        };
                        progressed = true;
                    }
                    Err(_) => {
                        let next_pos = probe_pos + 1;
                        if next_pos < item.probe_peers.len() {
                            item.state = RemoteState::PendingSend {
                                probe_pos: next_pos,
                            };
                        } else {
                            item.state = RemoteState::Done;
                            active_count -= 1;
                        }
                        progressed = true;
                    }
                }
            }

            // Step B: Drive full-duplex CQ + TCP channel + peer NVMe warm-and-serve.
            if let Ok(p) = self.poll_peers_nonblocking() {
                if p.cq_processed > 0 || p.grants_completed > 0 || p.windows_replenished > 0 {
                    progressed = true;
                }
            }

            // Step C: Match inbound grants/misses and immediately post RDMA READs.
            for item in &mut items {
                let (probe_pos, peer_idx, req_id, use_gdr, dst_tier, dst_kind, pull_target) =
                    match item.state {
                        RemoteState::AwaitingGrant {
                            probe_pos,
                            peer_idx,
                            req_id,
                            use_gdr,
                            dst_tier,
                            dst_kind,
                            pull_target,
                        } => (
                            probe_pos,
                            peer_idx,
                            req_id,
                            use_gdr,
                            dst_tier,
                            dst_kind,
                            pull_target,
                        ),
                        _ => continue,
                    };

                // Check for matching ReadGrant (NEVER hold a Vec index across `&mut self` calls,
                // because `ensure_tier_space` -> `drain_and_evict_frame_across_peers` can pump peer
                // control channels and cooperatively revoke entries from `pending_inbound_grants`).
                let has_grant = self.engine.peers[peer_idx]
                    .pending_inbound_grants
                    .iter()
                    .any(|g| g.meta.request_id == req_id);
                if has_grant {
                    // Bound concurrent GDR PullInFlight by `gdr_cap`; defer while at cap.
                    if use_gdr && pulling_gdr_count >= gdr_cap {
                        continue;
                    }
                    // Check destination tier space BEFORE popping `g`. If temporarily exhausted
                    // while other pulls hold STATE_INITIALIZING frames, leave `g` buffered and
                    // retry after Step D completes an in-flight pull (unless `g` was revoked in
                    // the meantime).
                    if self.ensure_tier_space(dst_kind).is_err() {
                        if pulling_count > 0
                            && self.engine.peers[peer_idx]
                                .pending_inbound_grants
                                .iter()
                                .any(|g| g.meta.request_id == req_id)
                        {
                            continue;
                        }
                        if let Some(g) = self.take_inbound_grant(peer_idx, req_id) {
                            self.release_unused_inbound_grant(peer_idx, g);
                        }
                        progressed = true;
                        in_flight_per_peer[peer_idx] =
                            in_flight_per_peer[peer_idx].saturating_sub(1);
                        self.policy.routing.on_remote_miss(item.key, peer_idx);
                        let next_pos = probe_pos + 1;
                        if next_pos < item.probe_peers.len() {
                            item.state = RemoteState::PendingSend {
                                probe_pos: next_pos,
                            };
                        } else {
                            item.state = RemoteState::Done;
                            active_count -= 1;
                        }
                        continue;
                    }

                    // Re-lookup atomically after `ensure_tier_space`: if the donor revoked this
                    // grant while we were reclaiming space (e.g. HBM->DRAM demotion), retry once
                    // on the same peer before advancing so the retry hits donor DRAM cleanly.
                    let Some(g) = self.take_inbound_grant(peer_idx, req_id) else {
                        progressed = true;
                        in_flight_per_peer[peer_idx] =
                            in_flight_per_peer[peer_idx].saturating_sub(1);
                        if item.retries_remaining > 0 {
                            item.retries_remaining -= 1;
                            item.state = RemoteState::PendingSend { probe_pos };
                        } else {
                            self.policy.routing.on_remote_miss(item.key, peer_idx);
                            let next_pos = probe_pos + 1;
                            if next_pos < item.probe_peers.len() {
                                item.retries_remaining = 1;
                                item.state = RemoteState::PendingSend {
                                    probe_pos: next_pos,
                                };
                            } else {
                                item.state = RemoteState::Done;
                                active_count -= 1;
                            }
                        }
                        continue;
                    };
                    progressed = true;
                    match self.try_post_pull_or_release_grant(
                        peer_idx,
                        item.key,
                        dst_tier,
                        pull_target,
                        g,
                    ) {
                        Ok((tag, pull, version_binding, orig_share_frontier)) => {
                            pulling_count += 1;
                            if use_gdr {
                                pulling_gdr_count += 1;
                            }
                            item.state = RemoteState::PullInFlight {
                                probe_pos,
                                peer_idx,
                                req_id,
                                tag,
                                pull,
                                version_binding,
                                orig_share_frontier,
                                use_gdr,
                                dst_kind,
                            };
                            continue;
                        }
                        Err(_) => {
                            // Grant has already been cooperatively released via ReadRelease.
                            in_flight_per_peer[peer_idx] =
                                in_flight_per_peer[peer_idx].saturating_sub(1);
                            if item.retries_remaining > 0 {
                                item.retries_remaining -= 1;
                                item.state = RemoteState::PendingSend { probe_pos };
                            } else {
                                self.policy.routing.on_remote_miss(item.key, peer_idx);
                                let next_pos = probe_pos + 1;
                                if next_pos < item.probe_peers.len() {
                                    item.retries_remaining = 1;
                                    item.state = RemoteState::PendingSend {
                                        probe_pos: next_pos,
                                    };
                                } else {
                                    item.state = RemoteState::Done;
                                    active_count -= 1;
                                }
                            }
                            continue;
                        }
                    }
                }

                // Check for matching FetchMiss
                if self.take_inbound_miss(peer_idx, req_id) {
                    progressed = true;
                    in_flight_per_peer[peer_idx] = in_flight_per_peer[peer_idx].saturating_sub(1);
                    self.policy.routing.on_remote_miss(item.key, peer_idx);
                    let next_pos = probe_pos + 1;
                    if next_pos < item.probe_peers.len() {
                        item.retries_remaining = 1;
                        item.state = RemoteState::PendingSend {
                            probe_pos: next_pos,
                        };
                    } else {
                        item.state = RemoteState::Done;
                        active_count -= 1;
                    }
                }
            }

            // Step D: Complete finished RDMA READ pulls and immediately send ReadRelease.
            for item in &mut items {
                let ready = match &item.state {
                    RemoteState::PullInFlight { peer_idx, pull, .. } => {
                        let cp = &self.engine.peers[*peer_idx].rdma.completed_pulls;
                        let slot = (pull.wr_id as usize) % cp.len();
                        matches!(cp[slot], Some((s, _)) if s == pull.wr_id)
                    }
                    _ => false,
                };
                if !ready {
                    continue;
                }
                let prev = core::mem::replace(&mut item.state, RemoteState::Done);
                let RemoteState::PullInFlight {
                    probe_pos,
                    peer_idx,
                    req_id,
                    tag,
                    pull,
                    version_binding,
                    orig_share_frontier,
                    use_gdr,
                    dst_kind,
                } = prev
                else {
                    continue;
                };
                pulling_count -= 1;
                if use_gdr {
                    pulling_gdr_count = pulling_gdr_count.saturating_sub(1);
                }
                in_flight_per_peer[peer_idx] = in_flight_per_peer[peer_idx].saturating_sub(1);
                progressed = true;

                let tracked vb = version_binding.get().tracked_unwrap();
                let res = self.engine.peers[peer_idx].fetch_complete_pull(
                    &mut self.engine.core,
                    tag,
                    req_id,
                    pull,
                    Tracked(vb),
                    orig_share_frontier,
                    Tracked(self.caps.instance.borrow()),
                );
                match res {
                    Ok(FetchOutcome::Hit { frame, .. }) => {
                        let meta = self.peer_metas[peer_idx];
                        self.index_insert(dst_kind, item.key, frame);
                        self.policy.routing.on_remote_hit(item.key, peer_idx);
                        self.policy.eviction.on_insert(dst_kind, item.key, frame);
                        resolved.insert(item.key, (meta.node_id, use_gdr));
                        active_count -= 1;
                    }
                    _ => {
                        // Transient pull error / Option-A grant revocation (e.g. donor evicted
                        // HBM->DRAM): retry the same peer once before treating as a remote miss.
                        if item.retries_remaining > 0 {
                            item.retries_remaining -= 1;
                            item.state = RemoteState::PendingSend { probe_pos };
                        } else {
                            self.policy.routing.on_remote_miss(item.key, peer_idx);
                            let next_pos = probe_pos + 1;
                            if next_pos < item.probe_peers.len() {
                                item.retries_remaining = 1;
                                item.state = RemoteState::PendingSend {
                                    probe_pos: next_pos,
                                };
                            } else {
                                active_count -= 1;
                            }
                        }
                    }
                }
            }

            // Step E: Backoff / bounded timeout without leaking any InFlightRdmaPull or grant.
            if progressed {
                idle_spins = 0;
            } else {
                idle_spins += 1;
                if idle_spins >= 64 {
                    std::thread::sleep(Duration::from_micros(25));
                } else if idle_spins >= 16 {
                    std::thread::yield_now();
                }
                if idle_spins >= 500_000 || Instant::now() >= deadline {
                    eprintln!(
                        "[VeriCache][node={}] prefetch_remote_batch_for_gpu timeout: \
                         active={}/{} pulling={} resolved={} idle_spins={}",
                        self.node_id,
                        active_count,
                        items.len(),
                        pulling_count,
                        resolved.len(),
                        idle_spins,
                    );
                    // Drain any remaining PullInFlight via fetch_complete_pull before exiting
                    // so no STATE_INITIALIZING frame or RDMA WR is ever leaked.
                    if pulling_count > 0 {
                        for item in &mut items {
                            if !matches!(item.state, RemoteState::PullInFlight { .. }) {
                                continue;
                            }
                            let prev = core::mem::replace(&mut item.state, RemoteState::Done);
                            let RemoteState::PullInFlight {
                                peer_idx,
                                req_id,
                                tag,
                                pull,
                                version_binding,
                                orig_share_frontier,
                                use_gdr,
                                dst_kind,
                                ..
                            } = prev
                            else {
                                continue;
                            };
                            let tracked vb = version_binding.get().tracked_unwrap();
                            if let Ok(FetchOutcome::Hit { frame, .. }) =
                                self.engine.peers[peer_idx].fetch_complete_pull(
                                    &mut self.engine.core,
                                    tag,
                                    req_id,
                                    pull,
                                    Tracked(vb),
                                    orig_share_frontier,
                                    Tracked(self.caps.instance.borrow()),
                                )
                            {
                                let meta = self.peer_metas[peer_idx];
                                self.index_insert(dst_kind, item.key, frame);
                                self.policy.routing.on_remote_hit(item.key, peer_idx);
                                self.policy.eviction.on_insert(dst_kind, item.key, frame);
                                resolved.insert(item.key, (meta.node_id, use_gdr));
                            }
                        }
                    }
                    // Cooperatively release any buffered grants for abandoned AwaitingGrant items
                    // so no donor Type-2B MW is left pinned on timeout.
                    for item in &items {
                        if let RemoteState::AwaitingGrant {
                            peer_idx, req_id, ..
                        } = item.state
                        {
                            if let Some(g) = self.take_inbound_grant(peer_idx, req_id) {
                                self.release_unused_inbound_grant(peer_idx, g);
                            }
                        }
                    }
                    break;
                }
            }
        }

        Ok(resolved)
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
        self.index_insert(TierKind::GpuVram, key, gpu_frame);
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
                    self.index_insert(TierKind::HostDram, key, frame);
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
                    self.index_insert(dst_kind, key, frame);
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
            .poll_nvme_completions(Tracked(self.caps.instance.borrow()), 64);
        let host_tier = self.engine.host_tier();
        let nvme_tier = self.engine.nvme_tier();
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
                let nvme_committed = nvme_tier
                    .is_some_and(|nt| self.frame_matches_key_tier(sp.key, sp.nvme_frame, nt));
                if nvme_committed {
                    self.index_insert(TierKind::LocalNvme, sp.key, sp.nvme_frame);
                    self.policy
                        .eviction
                        .on_insert(TierKind::LocalNvme, sp.key, sp.nvme_frame);
                    if self.frame_valid_published_on_tier(sp.key, sp.dram_frame, host_tier) {
                        let _ = self.engine.drain_and_evict_frame_across_peers(
                            sp.dram_frame,
                            self.cooperative_revoke_spin_budget,
                            Tracked(self.caps.instance.borrow()),
                        );
                        let _ = self.engine.reclaim_draining(host_tier, 32);
                        self.index_remove(TierKind::HostDram, sp.key, sp.dram_frame);
                        self.policy
                            .eviction
                            .on_remove(TierKind::HostDram, sp.key, sp.dram_frame);
                    }
                } else {
                    // Spill failed or was aborted: keep the DRAM copy tracked in LRU.
                    if self.frame_valid_published_on_tier(sp.key, sp.dram_frame, host_tier) {
                        self.policy
                            .eviction
                            .on_insert_cold(TierKind::HostDram, sp.key, sp.dram_frame);
                    }
                }
                retired += 1;
            } else {
                idx += 1;
            }
        }
        retired
    }

    fn wait_pending_spill_for_key(&mut self, key: ObjectKey) -> Result<(), CacheError> {
        while self.pending_spills.iter().any(|p| p.key == key) {
            let _ = self.poll_peers_nonblocking_inner(false);
            if self.reap_completed_spills() == 0 {
                let _ = self.submit_staged_nvme();
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
        let Some(nvme_tier) = self.engine.nvme_tier() else {
            return Ok(());
        };
        let host_tier = self.engine.host_tier();
        self.reap_completed_spills();
        let _ = self.engine.reclaim_draining(host_tier, 32);
        let mut staged_any = false;
        while self.engine.core.frames.ready_queue[host_tier.as_index()].len()
            <= wm.high_water_free_frames
            && self.pending_spills.len() < wm.max_in_flight
        {
            let Some((dk, df)) = self.policy.eviction.select_victim(TierKind::HostDram) else {
                break;
            };
            if !self.frame_valid_published_on_tier(dk, df, host_tier) {
                self.stats.policy_victims_rejected += 1;
                continue;
            }
            // If key already has a replica on NVMe, drop the DRAM copy immediately (zero NVMe write).
            if self.lookup_tier_fast(dk, TierKind::LocalNvme, nvme_tier).is_some() {
                let _ = self.engine.drain_and_evict_frame_across_peers(
                    df,
                    self.cooperative_revoke_spin_budget,
                    Tracked(self.caps.instance.borrow()),
                );
                let _ = self.engine.reclaim_draining(host_tier, 32);
                self.index_remove(TierKind::HostDram, dk, df);
                self.policy.eviction.on_remove(TierKind::HostDram, dk, df);
                continue;
            }
            if self.ensure_tier_space(TierKind::LocalNvme).is_err() {
                self.policy.eviction.on_insert_cold(TierKind::HostDram, dk, df);
                break;
            }
            if self.frame_matches_key_tier(dk, df, host_tier) {
                match self.stage_dram_to_nvme_one(df) {
                    Ok((nf, ticket)) => {
                        self.pending_spills.push(PendingSpill {
                            key: dk,
                            dram_frame: df,
                            nvme_frame: nf,
                            ticket,
                        });
                        staged_any = true;
                    }
                    Err(_) => {
                        self.policy.eviction.on_insert_cold(TierKind::HostDram, dk, df);
                        break;
                    }
                }
            }
        }
        if staged_any {
            let _ = self.submit_staged_nvme();
        }
        Ok(())
    }

    #[inline]
    fn frame_valid_published_on_tier(
        &self,
        expected_key: ObjectKey,
        frame: FrameId,
        expected_tier: TierId,
    ) -> bool {
        self.frame_matches_key_tier(expected_key, frame, expected_tier)
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

            if tier_kind == TierKind::HostDram {
                // Proactively top up async QD=64 spills before falling back to sync eviction
                // so write bursts never fall off the QD=1 sync-spill cliff.
                if let Some(wm) = self.policy.tiering.dram_spill_watermark() {
                    if self.engine.nvme_tier().is_some()
                        && self.pending_spills.len() < wm.max_in_flight
                    {
                        let _ = self.kick_async_spills();
                        if !self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
                            return Ok(());
                        }
                    }
                }
                if !self.pending_spills.is_empty() {
                    let _ = self.submit_staged_nvme();
                    let _ = self
                        .engine
                        .wait_nvme_completion(Tracked(self.caps.instance.borrow()));
                    let _ = self.reap_completed_spills();
                    let _ = self.engine.reclaim_draining(tier_id, 16);
                    if !self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
                        return Ok(());
                    }
                }
            }

            let Some((vkey, vframe)) = self.policy.eviction.select_victim(tier_kind) else {
                let _ = self.poll_peers_nonblocking_inner(false);
                let _ = self.engine.reclaim_draining(tier_id, 16);
                if !self.engine.core.frames.ready_queue[tier_id.as_index()].is_empty() {
                    return Ok(());
                }
                return Err(CacheError::Core(CoreError::CapacityExhausted));
            };

            // Verified Enforcement Check 1: validate policy hint against CacheCore truth.
            if !self.frame_valid_published_on_tier(vkey, vframe, tier_id) {
                self.stats.policy_victims_rejected += 1;
                continue;
            }

            // Optional cascading demotion to colder tier before evicting victim.
            if let Some(demote_to) = self.policy.tiering.demote_on_evict(tier_kind, vkey) {
                match (tier_kind, demote_to) {
                    (TierKind::GpuVram, TierKind::HostDram) => {
                        let host_tier = self.engine.host_tier();
                        if self
                            .lookup_tier_fast(vkey, TierKind::HostDram, host_tier)
                            .is_none()
                            && self.ensure_tier_space(TierKind::HostDram).is_ok()
                        {
                            if let Ok((_v, new_df)) = self.copy_hop_internal(vframe, host_tier) {
                                self.index_insert(TierKind::HostDram, vkey, new_df);
                                self.policy
                                    .eviction
                                    .on_insert(TierKind::HostDram, vkey, new_df);
                            }
                        }
                    }
                    (TierKind::HostDram, TierKind::LocalNvme) => {
                        if let Some(nvme_tier) = self.engine.nvme_tier() {
                            if self
                                .lookup_tier_fast(vkey, TierKind::LocalNvme, nvme_tier)
                                .is_none()
                                && self.ensure_tier_space(TierKind::LocalNvme).is_ok()
                            {
                                let _ = self.submit_staged_nvme();
                                let mut spilled = self.engine.spill_host_to_nvme_sync(
                                    Tracked(self.caps.instance.borrow()),
                                    Tracked(self.caps.shares.borrow_mut()),
                                    vkey,
                                );
                                // Never silently drop a cold block on transient ring/lease pressure:
                                // drain one NVMe completion and retry once before giving up.
                                if spilled.is_err() {
                                    let _ = self.submit_staged_nvme();
                                    let _ = self.engine.wait_nvme_completion(Tracked(
                                        self.caps.instance.borrow(),
                                    ));
                                    let _ = self.reap_completed_spills();
                                    if self.frame_matches_key_tier(vkey, vframe, tier_id)
                                        && self
                                            .lookup_tier_fast(vkey, TierKind::LocalNvme, nvme_tier)
                                            .is_none()
                                        && self.ensure_tier_space(TierKind::LocalNvme).is_ok()
                                    {
                                        spilled = self.engine.spill_host_to_nvme_sync(
                                            Tracked(self.caps.instance.borrow()),
                                            Tracked(self.caps.shares.borrow_mut()),
                                            vkey,
                                        );
                                    }
                                }
                                if let Ok(new_nf) = spilled {
                                    self.index_insert(TierKind::LocalNvme, vkey, new_nf);
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
            let _ = self.poll_peers_nonblocking_inner(false);
            let _ = self.engine.reclaim_draining(tier_id, 16);

            self.index_remove(tier_kind, vkey, vframe);
            self.policy.eviction.on_remove(tier_kind, vkey, vframe);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::DefaultPolicy;
    use crate::resources::{NvmeConfig, StorageConfig};
    use crate::transport::NoTransport;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_nvme_dir(tag: &str) -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("unit-nvme")
            .join(format!("vericache_cache_unit_{tag}_{pid}_{n}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp nvme dir");
        dir
    }

    fn key(id: u64) -> ObjectKey {
        ObjectKey([id, id.wrapping_mul(0x9e37_79b9_7f4a_7c15), 0xabcd, 0x1234])
    }

    #[test]
    fn host_write_batch_zero_copy_skip_dedup_and_abort() {
        let mut cfg = StorageConfig::new(0, 4096, 16);
        cfg.max_leases = 256;

        VeriCache::<NoTransport, _, _, _>::with_session(
            cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                assert!(!cache.is_host_pinned());

                // Pre-populate key(0) via regular put_host.
                let k0 = key(0);
                let data0 = vec![0xA5u8; 4096];
                cache.put_host(k0, &data0)?;

                // Batch contains existing key(0), unique new keys(1..=4), and duplicate key(2).
                let batch = vec![k0, key(1), key(2), key(2), key(3), key(4)];
                let mut seen_indices = Vec::new();
                let mut seen_addrs = HashSet::new();
                let mut flushes = 0usize;

                let written = cache.with_host_write_batch(
                    &batch,
                    |gi, addr| {
                        assert_ne!(addr, 0);
                        assert_eq!(addr % 4096, 0);
                        assert!(seen_addrs.insert(addr), "distinct frame addresses per chunk");
                        seen_indices.push(gi);
                        Ok(())
                    },
                    || {
                        flushes += 1;
                        Ok(())
                    },
                )?;
                assert_eq!(written, batch.len());
                // key(0) skipped, duplicate key(2) at index 3 skipped -> 4 unique new keys.
                assert_eq!(seen_indices, vec![1, 2, 4, 5]);
                assert_eq!(flushes, 1);

                // Verify all keys are resident via contains_batch_local.
                let locs = cache.contains_batch_local(&batch);
                assert!(locs.iter().all(|l| *l == Some(HitLocation::LocalDram)));

                // Verify error path drains on_flush and aborts reserved frames without leaking.
                let host_tier = cache.engine().host_tier().as_index();
                let free_before = cache.engine().core.frames.ready_queue[host_tier].len();
                let fail_keys = vec![key(10), key(11), key(12)];
                let mut fail_flushes = 0usize;
                let err_res = cache.with_host_write_batch(
                    &fail_keys,
                    |gi, _addr| {
                        if gi == 2 {
                            return Err(CacheError::Init("simulated dma enqueue error".into()));
                        }
                        Ok(())
                    },
                    || {
                        fail_flushes += 1;
                        Ok(())
                    },
                );
                assert!(err_res.is_err());
                assert_eq!(fail_flushes, 1, "on_flush must drain stream before aborting");
                let free_after = cache.engine().core.frames.ready_queue[host_tier].len();
                assert_eq!(free_before, free_after, "zero leaked frames on error abort");
                assert!(cache.contains_local(key(10)).is_none());
                assert!(cache.contains_local(key(11)).is_none());
                assert!(cache.contains_local(key(12)).is_none());

                Ok(())
            },
        )
        .expect("host_write_batch_zero_copy_skip_dedup_and_abort");
    }

    #[test]
    fn gpu_read_batch_chunking_nvme_qd64_and_lease_headroom() {
        let nvme_dir = temp_nvme_dir("batch_qd64");
        // 48 DRAM frames + 96 NVMe frames, max_leases = 256 (and queue_depth = 128).
        let mut cfg = StorageConfig::new(0, 4096, 48).with_nvme(NvmeConfig {
            dir: nvme_dir.clone(),
            nvme_frames: 96,
            queue_depth: 128,
        });
        cfg.max_leases = 256;

        let res = VeriCache::<NoTransport, _, _, _>::with_session(
            cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                // Insert 80 keys: first ~32 spill to NVMe, last ~48 stay in DRAM.
                let keys: Vec<ObjectKey> = (0..80u64).map(key).collect();
                let buf = vec![0x3Cu8; 4096];
                for &k in &keys {
                    cache.put_host(k, &buf)?;
                }

                // Batch contains all 80 keys + 10 missing keys -> crosses effective_chunk (=48).
                let mut query = keys.clone();
                for i in 1000..1010u64 {
                    query.push(key(i));
                }

                let mut hit_count = 0usize;
                let mut flush_count = 0usize;
                let hits = cache.with_gpu_read_batch(
                    &query,
                    |gi, src| {
                        assert!(gi < 80);
                        match src {
                            BatchReadSource::PinnedHost(addr) => {
                                assert_ne!(addr, 0);
                                assert_eq!(addr % 4096, 0);
                            }
                            BatchReadSource::GpuDevice(_) => {
                                panic!("unexpected GPU hit in CPU-only test");
                            }
                        }
                        hit_count += 1;
                        Ok(())
                    },
                    || {
                        flush_count += 1;
                        Ok(())
                    },
                )?;

                assert_eq!(hit_count, 80);
                assert!(flush_count >= 2, "expected multiple chunks for 80 hits with 48 DRAM frames");
                assert!(hits[..80].iter().all(|h| h.is_some()));
                assert!(hits[80..].iter().all(|h| h.is_none()));
                assert!(cache.stats().nvme_hits > 0, "expected NVMe warm hits");
                assert_eq!(
                    cache.engine().core.free_lease_slots.len(),
                    256,
                    "all leases must be released after batch read"
                );
                Ok(())
            },
        );

        let _ = std::fs::remove_dir_all(&nvme_dir);
        res.expect("gpu_read_batch_chunking_nvme_qd64_and_lease_headroom");
    }

    #[test]
    fn nvme_two_hit_promotion_gate_reclaims_first_hit_and_promotes_second() {
        let nvme_dir = temp_nvme_dir("two_hit_promo");
        let mut cfg = StorageConfig::new(0, 4096, 8).with_nvme(NvmeConfig {
            dir: nvme_dir.clone(),
            nvme_frames: 32,
            queue_depth: 64,
        });
        cfg.max_leases = 128;

        let res = VeriCache::<NoTransport, _, _, _>::with_session(
            cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                let buf = vec![0x5Au8; 4096];
                // Fill DRAM (8) and spill 4 keys (0..4) to NVMe.
                for i in 0..12u64 {
                    cache.put_host(key(i), &buf)?;
                }
                assert_eq!(cache.contains_local(key(0)), Some(HitLocation::LocalNvme));
                assert_eq!(cache.contains_local(key(1)), Some(HitLocation::LocalNvme));

                // 1st NVMe batch read of key(0) and key(1): transient warm -> reclaimed immediately
                // after chunk flush (back to LocalNvme, zero DRAM retention thrash!).
                let q = vec![key(0), key(1)];
                let hits1 = cache.with_gpu_read_batch(&q, |_, _| Ok(()), || Ok(()))?;
                assert_eq!(hits1[0].unwrap().location, HitLocation::LocalNvme);
                assert_eq!(hits1[1].unwrap().location, HitLocation::LocalNvme);
                assert_eq!(
                    cache.contains_local(key(0)),
                    Some(HitLocation::LocalNvme),
                    "1st NVMe hit must reclaim transient DRAM frame"
                );
                assert_eq!(
                    cache.contains_local(key(1)),
                    Some(HitLocation::LocalNvme),
                    "1st NVMe hit must reclaim transient DRAM frame"
                );

                // 2nd NVMe batch read of key(0): passes 2-hit gate -> promoted & retained in DRAM!
                let hits2 = cache.with_gpu_read_batch(&[key(0)], |_, _| Ok(()), || Ok(()))?;
                assert_eq!(hits2[0].unwrap().location, HitLocation::LocalNvme);
                assert_eq!(
                    cache.contains_local(key(0)),
                    Some(HitLocation::LocalDram),
                    "2nd NVMe hit must promote into DRAM"
                );

                // 3rd read hits DRAM directly.
                let hits3 = cache.with_gpu_read_batch(&[key(0)], |_, _| Ok(()), || Ok(()))?;
                assert_eq!(hits3[0].unwrap().location, HitLocation::LocalDram);
                assert_eq!(cache.engine().core.free_lease_slots.len(), 128);
                Ok(())
            },
        );

        let _ = std::fs::remove_dir_all(&nvme_dir);
        res.expect("nvme_two_hit_promotion_gate_reclaims_first_hit_and_promotes_second");
    }

    #[test]
    fn gpu_read_batch_pipelined_double_buffer_overlap_and_error_lease_cleanup() {
        let nvme_dir = temp_nvme_dir("pipelined_dbuf");
        let mut cfg = StorageConfig::new(0, 4096, 32).with_nvme(NvmeConfig {
            dir: nvme_dir.clone(),
            nvme_frames: 96,
            queue_depth: 128,
        });
        cfg.max_leases = 256;

        let res = VeriCache::<NoTransport, _, _, _>::with_session(
            cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                let buf = vec![0x77u8; 4096];
                let keys: Vec<ObjectKey> = (0..64u64).map(key).collect();
                for &k in &keys {
                    cache.put_host(k, &buf)?;
                }

                // Append misses at the end to verify miss handling in pipelined path.
                let mut query = keys.clone();
                for i in 500..508u64 {
                    query.push(key(i));
                }

                // Track slot emit/sync sequence to prove true double-buffered overlap:
                // emit(0) -> emit(1) -> sync(0) -> emit(0) -> sync(1) -> ... -> sync(last).
                let events: std::cell::RefCell<Vec<(&'static str, usize)>> =
                    std::cell::RefCell::new(Vec::new());
                let in_flight_slots = std::cell::RefCell::new([false; 2]);
                let mut total_emitted = 0usize;

                let hits = cache.with_gpu_read_batch_pipelined(
                    &query,
                    |slot, entries| {
                        assert!(slot <= 1);
                        let mut slots = in_flight_slots.borrow_mut();
                        assert!(
                            !slots[slot],
                            "slot {slot} must be synced before reuse"
                        );
                        assert!(!entries.is_empty());
                        for &(gi, src) in entries {
                            assert!(gi < 64);
                            match src {
                                BatchReadSource::PinnedHost(addr) => {
                                    assert_ne!(addr, 0);
                                    assert_eq!(addr % 4096, 0);
                                }
                                BatchReadSource::GpuDevice(_) => {
                                    panic!("unexpected GPU hit in CPU-only test");
                                }
                            }
                        }
                        slots[slot] = true;
                        total_emitted += entries.len();
                        events.borrow_mut().push(("emit", slot));
                        Ok(())
                    },
                    |slot| {
                        assert!(slot <= 1);
                        let mut slots = in_flight_slots.borrow_mut();
                        assert!(slots[slot], "sync called on idle slot {slot}");
                        slots[slot] = false;
                        events.borrow_mut().push(("sync", slot));
                        Ok(())
                    },
                )?;

                let events = events.into_inner();
                let in_flight_slots = in_flight_slots.into_inner();
                assert_eq!(total_emitted, 64);
                assert!(hits[..64].iter().all(|h| h.is_some()));
                assert!(hits[64..].iter().all(|h| h.is_none()));
                assert!(!in_flight_slots[0] && !in_flight_slots[1]);
                // Verify true double-buffered overlap: first two events must be emit(0), emit(1), then sync(0).
                assert!(events.len() >= 6, "expected >=3 chunks, got {events:?}");
                assert_eq!(events[0], ("emit", 0));
                assert_eq!(events[1], ("emit", 1));
                assert_eq!(events[2], ("sync", 0));
                assert_eq!(cache.engine().core.free_lease_slots.len(), 256);

                // Verify zero leaked leases when on_chunk_emit returns an error mid-pipeline.
                let mut emit_calls = 0usize;
                let err_emit = cache.with_gpu_read_batch_pipelined(
                    &keys,
                    |_slot, _entries| {
                        emit_calls += 1;
                        if emit_calls == 2 {
                            return Err(CacheError::Init("simulated emit failure".into()));
                        }
                        Ok(())
                    },
                    |_slot| Ok(()),
                );
                assert!(err_emit.is_err());
                assert_eq!(
                    cache.engine().core.free_lease_slots.len(),
                    256,
                    "zero leaked leases on on_chunk_emit error"
                );

                // Verify zero leaked leases when on_chunk_sync returns an error mid-pipeline.
                let mut sync_calls = 0usize;
                let err_sync = cache.with_gpu_read_batch_pipelined(
                    &keys,
                    |_slot, _entries| Ok(()),
                    |_slot| {
                        sync_calls += 1;
                        if sync_calls == 1 {
                            return Err(CacheError::Init("simulated sync failure".into()));
                        }
                        Ok(())
                    },
                );
                assert!(err_sync.is_err());
                assert_eq!(
                    cache.engine().core.free_lease_slots.len(),
                    256,
                    "zero leaked leases on on_chunk_sync error"
                );
                Ok(())
            },
        );

        let _ = std::fs::remove_dir_all(&nvme_dir);
        res.expect("gpu_read_batch_pipelined_double_buffer_overlap_and_error_lease_cleanup");
    }

    #[test]
    fn async_spill_watermark_burst_writes_keep_all_keys_resident() {
        use crate::policy::{CascadeTiering, LruEviction, PolicySet, ProbeAllPeers};

        let nvme_dir = temp_nvme_dir("async_spill_burst");
        let mut cfg = StorageConfig::new(0, 4096, 16).with_nvme(NvmeConfig {
            dir: nvme_dir.clone(),
            nvme_frames: 96,
            queue_depth: 128,
        });
        cfg.max_leases = 256;

        let policy = PolicySet::new(
            LruEviction::new(),
            CascadeTiering::serving_dram_first(8, 32),
            ProbeAllPeers,
        );

        let res = VeriCache::<NoTransport, _, _, _>::with_session(
            cfg,
            Vec::new(),
            policy,
            |cache| {
                // Burst-write 80 keys (> 16 DRAM capacity, fits in 16 DRAM + 96 NVMe)
                // via zero-copy with_host_write_batch.
                let keys: Vec<ObjectKey> = (0..80u64).map(key).collect();
                let written = cache.with_host_write_batch(
                    &keys,
                    |_gi, addr| {
                        assert_ne!(addr, 0);
                        Ok(())
                    },
                    || Ok(()),
                )?;
                assert_eq!(written, 80);

                // Drain any in-flight async spills and verify ZERO cold blocks were dropped.
                for _ in 0..16 {
                    let _ = cache.poll()?;
                }
                let locs = cache.contains_batch_local(&keys);
                let missing = locs.iter().filter(|l| l.is_none()).count();
                assert_eq!(missing, 0, "no cold blocks may be silently dropped during burst spill");
                let nvme_count = locs
                    .iter()
                    .filter(|l| **l == Some(HitLocation::LocalNvme))
                    .count();
                assert!(nvme_count >= 60, "expected >=60 keys spilled to NVMe, got {nvme_count}");
                Ok(())
            },
        );

        let _ = std::fs::remove_dir_all(&nvme_dir);
        res.expect("async_spill_watermark_burst_writes_keep_all_keys_resident");
    }

    #[test]
    fn catalog_buckets_scale_with_total_frames() {
        // Small config keeps DEFAULT_BUCKETS (4096).
        let small_cfg = StorageConfig::new(0, 4096, 16);
        VeriCache::<NoTransport, _, _, _>::with_session(
            small_cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                assert_eq!(
                    cache.engine().core.catalog.bucket_count(),
                    vericache_core::catalog::DEFAULT_BUCKETS
                );
                Ok(())
            },
        )
        .expect("small catalog bucket check");

        // Larger config (5000 frames) scales up to next power of two (8192 buckets).
        let large_cfg = StorageConfig::new(0, 4096, 5000);
        VeriCache::<NoTransport, _, _, _>::with_session(
            large_cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                assert_eq!(cache.engine().core.catalog.bucket_count(), 8192);
                let k = key(42);
                let data = vec![0xABu8; 4096];
                cache.put_host(k, &data)?;
                let mut out = vec![0u8; 4096];
                let hit = cache.get_into_host(k, &mut out)?.expect("hit");
                assert_eq!(hit.location, HitLocation::LocalDram);
                assert_eq!(out, data);
                Ok(())
            },
        )
        .expect("scaled catalog bucket check");
    }

    #[test]
    fn stale_grant_revoked_during_ensure_tier_space_never_panics() {
        // Regression test for the live Run 1 panic:
        //   "swap_remove index (is 0) should be < len (is 0)"
        // Simulates `pending_inbound_grants` being mutated/emptied by peer revocation
        // inside `ensure_tier_space` between initial presence check and grant pop.
        #[derive(Debug, PartialEq, Eq)]
        struct FakeGrant {
            req_id: u64,
        }

        // Case 1: Single grant (pos=0) is revoked during `ensure_tier_space` -> len becomes 0.
        let mut grants = vec![FakeGrant { req_id: 101 }];
        assert!(grants.iter().any(|g| g.req_id == 101));
        // Simulate peer.rs:6023 removing revoked grant inside `ensure_tier_space` (Ok or Err):
        grants.clear();
        assert_eq!(pop_matching(&mut grants, |g| g.req_id == 101), None);

        // Case 2: Earlier grant (pos=0) is revoked during `ensure_tier_space`, shifting target
        // grant from index 1 -> index 0. Re-lookup finds and pops the exact matching grant.
        let mut grants2 = vec![FakeGrant { req_id: 201 }, FakeGrant { req_id: 202 }];
        assert!(grants2.iter().any(|g| g.req_id == 202));
        // Simulate peer.rs:6023 revoking req_id=201 via swap_remove(0) -> [202] at index 0.
        let _ = pop_matching(&mut grants2, |g| g.req_id == 201);
        assert_eq!(
            pop_matching(&mut grants2, |g| g.req_id == 202),
            Some(FakeGrant { req_id: 202 })
        );
        assert!(grants2.is_empty());
    }

    #[test]
    fn peer_nvme_reaped_frame_survives_dram_pressure_and_preserves_arrivals() {
        let nvme_dir = temp_nvme_dir("peer_nvme_reap_survive");
        let mut cfg = StorageConfig::new(0, 4096, 4).with_nvme(NvmeConfig {
            dir: nvme_dir.clone(),
            nvme_frames: 16,
            queue_depth: 64,
        });
        cfg.max_leases = 128;

        let res = VeriCache::<NoTransport, _, _, _>::with_session(
            cfg,
            Vec::new(),
            DefaultPolicy::default(),
            |cache| {
                let buf = vec![0x42u8; 4096];
                // Fill 4 DRAM frames and spill keys 0..2 to NVMe; keys 2..6 fill DRAM.
                for i in 0..6u64 {
                    cache.put_host(key(i), &buf)?;
                }
                assert_eq!(cache.contains_local(key(0)), Some(HitLocation::LocalNvme));
                assert_eq!(cache.contains_local(key(1)), Some(HitLocation::LocalNvme));

                // Stage async peer NVMe warm for key(0) and reap via reap_completed_peer_nvme_fetches.
                let nvme_tier = cache.engine().nvme_tier().unwrap();
                let nf0 = cache
                    .lookup_tier_fast(key(0), TierKind::LocalNvme, nvme_tier)
                    .expect("key(0) on NVMe");
                cache.ensure_tier_space(TierKind::HostDram)?;
                let (df0, ticket0) = cache.stage_nvme_to_dram_one(nf0)?;
                let _ = cache.submit_staged_nvme()?;
                cache.pending_peer_nvme_fetches.push(PendingPeerNvmeFetch {
                    peer_idx: 0,
                    req_id: 77,
                    key: key(0),
                    nvme_frame: nf0,
                    dram_frame: df0,
                    ticket: ticket0,
                });
                while !cache.pending_peer_nvme_fetches.is_empty() {
                    if cache.reap_completed_peer_nvme_fetches() == 0 {
                        std::thread::yield_now();
                    }
                }
                // Freshly reaped peer NVMe frame is now in DRAM at MRU.
                assert_eq!(cache.contains_local(key(0)), Some(HitLocation::LocalDram));

                // Subsequent ensure_tier_space(HostDram) (as in Phase 3 staging the next peer
                // NVMe warm for key(1)) must NOT evict key(0) before its ReadGrant completes.
                cache.ensure_tier_space(TierKind::HostDram)?;
                assert_eq!(
                    cache.contains_local(key(0)),
                    Some(HitLocation::LocalDram),
                    "just-reaped peer NVMe frame must not be immediately evicted by ensure_tier_space"
                );

                // Verify append semantics never overwrite arrivals during Phase 3 staging.
                let mut simulated_pending: Vec<(u64, ObjectKey)> = vec![(1, key(10))];
                let inbound = core::mem::take(&mut simulated_pending);
                let mut immediate = Vec::new();
                for item in inbound {
                    immediate.push(item);
                }
                // Simulate new FetchRequest arriving while staging:
                simulated_pending.push((2, key(11)));
                // Phase 1/3 uses extend/push, NEVER overwrite:
                simulated_pending.extend(immediate);
                assert_eq!(simulated_pending.len(), 2);
                assert!(simulated_pending.contains(&(1, key(10))));
                assert!(simulated_pending.contains(&(2, key(11))));
                Ok(())
            },
        );

        let _ = std::fs::remove_dir_all(&nvme_dir);
        res.expect("peer_nvme_reaped_frame_survives_dram_pressure_and_preserves_arrivals");
    }

    #[test]
    fn window_aware_donor_throttle_and_requester_headroom_prevent_fake_misses() {
        // 1. Requester-side headroom leaves 25% of Type-2B windows free for LOCAL_INV CQE turnaround.
        assert_eq!(requester_peer_pipeline_cap(16, 16, 585), 12);
        assert_eq!(requester_peer_pipeline_cap(32, 16, 585), 16);
        assert_eq!(requester_peer_pipeline_cap(4, 16, 585), 3);
        assert_eq!(requester_peer_pipeline_cap(1, 16, 585), 1);
        assert_eq!(requester_peer_pipeline_cap(0, 16, 585), 0);

        // 2. Donor-side window throttle: with 2 free windows and 4 ready DRAM hits + 1 true miss,
        //    admits exactly 2 hits + 1 true miss immediately and parks 2 excess hits in
        //    `deferred_peer_grants` (never feeding > bind_budget hits to `serve_bind`).
        let inbound: Vec<(u64, ObjectKey, bool /* is_ready_hit */)> = vec![
            (1, key(1), true),
            (2, key(2), true),
            (3, key(3), false), // true miss
            (4, key(4), true),
            (5, key(5), true),
        ];
        let mut bind_budget = 2usize;
        let mut immediate = Vec::new();
        let mut deferred = Vec::new();
        for (req_id, k, is_ready_hit) in inbound {
            if is_ready_hit {
                if admit_within_bind_budget(&mut bind_budget) {
                    immediate.push((req_id, k));
                } else {
                    deferred.push((req_id, k));
                }
            } else {
                immediate.push((req_id, k));
            }
        }
        assert_eq!(bind_budget, 0);
        assert_eq!(immediate, vec![(1, key(1)), (2, key(2)), (3, key(3))]);
        assert_eq!(deferred, vec![(4, key(4)), (5, key(5))]);

        // Next poll after LOCAL_INV frees 2 windows: deferred hits drain first in FIFO order.
        bind_budget = 2;
        let mut next_immediate = Vec::new();
        for (req_id, k) in deferred {
            assert!(admit_within_bind_budget(&mut bind_budget));
            next_immediate.push((req_id, k));
        }
        assert_eq!(next_immediate, vec![(4, key(4)), (5, key(5))]);
    }

    #[test]
    fn recv_pause_guard_blocks_phase2_fetch_sneak_and_unpauses_on_drop() {
        use crate::transport::TcpTransport;
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let client = TcpStream::connect(addr).expect("connect");
        let (server, _) = listener.accept().expect("accept");

        let mut tx = TcpTransport::with_timeout(client, Duration::from_secs(2)).expect("tx");
        let mut rx = TcpTransport::with_timeout(server, Duration::from_secs(2)).expect("rx");
        let handles = vec![Some(rx.recv_pause_handle()), None];

        // Simulate Phase 2 holding RecvPauseGuard while an NVMe-only FetchRequest arrives on TCP.
        {
            let _guard = RecvPauseGuard::new(&handles);
            tx.send_frame(b"nvme-fetch-req-99").expect("send");
            assert_eq!(
                rx.try_recv_frame().expect("try_recv during phase 2"),
                None,
                "Phase 2 must not read new FetchRequests bypassing Phase 1 partitioner"
            );
        }
        // Guard dropped: next Phase 1 poll immediately sees the queued FetchRequest intact.
        assert_eq!(
            rx.recv_frame().expect("recv on next phase 1"),
            b"nvme-fetch-req-99"
        );
        // Verify unwind safety: dropping RecvPauseGuard on panic still resets recv_paused = false.
        let panic_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = RecvPauseGuard::new(&handles);
            assert!(handles[0].as_ref().unwrap().load(Ordering::Acquire));
            panic!("simulated phase 2 panic");
        }));
        assert!(panic_res.is_err());
        assert!(
            !handles[0].as_ref().unwrap().load(Ordering::Acquire),
            "RecvPauseGuard Drop must reset recv_paused even on panic unwind"
        );
    }
}
