//! Zero-`unsafe` two-step storage lifecycle (`NodeStorage` -> `AttachedStorage` -> `VeriCache`)
//! and ergonomic GPU buffer/stream wrappers (`GpuSlot`, `GpuStream`) hiding all Verus
//! `Tracked`/`Ghost` tokens from callers.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::{
    AllocSource, EventSourceAllocator, EventSourceKind, EventSourceToken, NodeContext,
    ShardAllocator, ShardSources, ShareSource, SlotToken, TransferInstance, VersionInstance,
    VersionSource,
};
use transfer::cuda::{self, CudaBuffer, CudaHostBuffer, CudaStream};
use transfer::domain::DeviceId;
use transfer::host::{self, HostContext};
use transfer::rdma::{
    AccessFlags, CompletionQueue, DeviceContext, ProtectionDomain, RdmaError,
    UnconnectedQueuePair, alloc_window, connect_rc, dealloc_window,
};
use transfer::transport::{ByteTransport, Channel};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::backends::rdma::peer::PeerSession;
use vericache_core::backends::rdma::protocol::VeriCacheAttachment;
use vericache_core::engine::{CudaAttachment, DistributedEngine};
use vericache_core::kernel::CacheCore;
use vericache_core::provenance::ValueCertificate;
#[cfg(verus_only)]
use vericache_core::provenance::ValueProof;
use vericache_core::spec::CoreError;
use vericache_core::timeline::TimelineKind;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Tracked, verus};

use crate::CacheError;
use crate::cache::VeriCache;
use crate::policy::{EvictionPolicy, PeerMeta, PeerRoutingPolicy, PolicySet, TieringPolicy};

/// Default per-shard version sequence budget (`2^40` versions per shard).
pub const DEFAULT_SHARD_VERSION_QUOTA: u64 = 1u64 << 40;

static NODE_SHARD_ALLOCATOR: OnceLock<Mutex<(u64, ShardAllocator)>> = OnceLock::new();

fn take_shard_sources(node_id: u64, version_quota: u64) -> Result<ShardSources, CacheError> {
    let lock = NODE_SHARD_ALLOCATOR.get_or_init(|| {
        let ctx = NodeContext::try_bootstrap(node_id)
            .expect("NodeContext::try_bootstrap once per OS process");
        Mutex::new((node_id, ShardAllocator::from_node_context(ctx)))
    });
    let mut guard = lock
        .lock()
        .map_err(|_| CacheError::Init("node ShardAllocator mutex poisoned".into()))?;
    if guard.0 != node_id {
        return Err(CacheError::Init(format!(
            "process already bootstrapped for node_id {}, cannot switch to {}",
            guard.0, node_id
        )));
    }
    guard
        .1
        .split_shard(version_quota)
        .ok_or_else(|| CacheError::Init("version quota exhausted in ShardAllocator".into()))
}

verus! {

/// Per-instance linear capability bundle carved from the process `ShardAllocator`
/// with zero `assume_new`.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub(crate) struct ShardCaps {
    pub(crate) node: u64,
    pub(crate) instance: Tracked<TransferInstance>,
    pub(crate) alloc: Tracked<AllocSource>,
    pub(crate) events: Tracked<EventSourceAllocator>,
    pub(crate) shares: Tracked<ShareSource>,
    pub(crate) version_instance: Tracked<VersionInstance>,
    pub(crate) versions: Option<Tracked<VersionSource>>,
    pub(crate) version_start: u64,
    pub(crate) version_limit: u64,
}

impl ShardCaps {
    pub(crate) fn take(node_id: u64) -> Result<Self, CacheError> {
        let shard = take_shard_sources(node_id, DEFAULT_SHARD_VERSION_QUOTA)?;
        Ok(Self {
            node: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            version_instance: shard.version_instance,
            versions: Some(shard.versions),
            version_start: shard.version_start,
            version_limit: shard.version_limit,
        })
    }

    pub(crate) fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::Host))
    }

    pub(crate) fn mint_cuda(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::CudaStream))
    }

    pub(crate) fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::IoUring))
    }

    pub(crate) fn mint_rdma(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair))
    }

    pub(crate) fn make_versioned_core(
        &mut self,
        frame_size: usize,
        max_leases: usize,
    ) -> Result<CacheCore, CacheError> {
        let ver_src = self
            .versions
            .take()
            .ok_or_else(|| CacheError::Init("shard VersionSource already consumed".into()))?;
        Ok(CacheCore::from_shard_versions(
            self.node as u16,
            frame_size,
            Vec::new(),
            max_leases,
            self.version_start,
            self.version_limit,
            Tracked(self.instance.borrow()),
            Tracked(self.version_instance.borrow()),
            ver_src,
        ))
    }
}

/// Caller-owned GPU buffer slot managed by `VeriCache`, hiding `SlotToken` and
/// `ValueCertificate` provenance from ergonomic callers.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct GpuSlot {
    pub(crate) buffer: CudaBuffer,
    pub(crate) token: Tracked<SlotToken>,
    pub(crate) cert: Option<Ghost<ValueCertificate>>,
    pub(crate) frame_size: usize,
    pub(crate) device_id: DeviceId,
}

impl GpuSlot {
    pub fn len(&self) -> usize {
        self.frame_size
    }

    pub fn is_empty(&self) -> bool {
        self.frame_size == 0
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Raw CUDA device address of this slot (for external CUDA interop such as
    /// PyTorch/vLLM `cudaMemcpyAsync` into/out of a managed `GpuSlot`).
    pub fn device_addr(&self) -> usize {
        self.buffer.slot_slice(0).device_addr()
    }
}

verus! {

/// Record external GPU-producer provenance for a `GpuSlot` populated via an
/// external CUDA stream (e.g., PyTorch/vLLM tensor D2D copy into `slot.device_addr()`).
#[cfg_attr(not(verus_only), allow(unused_variables))]
pub(crate) fn mark_gpu_slot_external_write(slot: &mut GpuSlot) {
    let slice = slot.buffer.slot_slice(0);
    let ghost cert = ValueCertificate {
        slot: slice.slot_ref(),
        writer: transfer::domain::TransferEvent {
            source: transfer::domain::EventSource::Host { id: 0 },
            observer: 0,
            seq: 0,
        },
        bytes: vstd::prelude::Seq::empty(),
        proof: ValueProof::HostWrite,
    };
    slot.cert = Some(Ghost(cert));
}

} // verus!

/// Caller-owned CUDA stream paired with its registered `CacheCore` timeline id.
pub struct GpuStream {
    pub(crate) stream: CudaStream,
    pub(crate) timeline: u8,
    pub(crate) device_id: DeviceId,
}

impl GpuStream {
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn timeline_id(&self) -> u8 {
        self.timeline
    }
}

pub(crate) fn select_type2_rdma_device(
    caps: &mut ShardCaps,
    requested: Option<&str>,
) -> Result<DeviceContext, CacheError> {
    let count = DeviceContext::count()
        .map_err(|e| CacheError::Init(format!("ibv_get_device_list failed: {e:?}")))?;
    for index in 0..count {
        let device = DeviceContext::open(index, caps.node)
            .map_err(|e| CacheError::Init(format!("open verbs device {index} failed: {e:?}")))?;
        if requested.is_some_and(|name| name != device.name()) {
            let _ = device.close();
            continue;
        }
        let type2 = device
            .supports_type_2()
            .map_err(|e| CacheError::Init(format!("query type-2 MW caps failed: {e:?}")))?;
        if type2 {
            let pd = ProtectionDomain::new(&device, caps.mint_rdma())
                .map_err(|e| CacheError::Init(format!("probe PD failed: {e:?}")))?;
            let supported = match alloc_window(&pd, caps.mint_rdma()) {
                Ok(w) => {
                    let _ = dealloc_window(w);
                    true
                }
                Err(RdmaError(38)) | Err(RdmaError(95)) => false,
                Err(e) => {
                    let _ = pd.destroy();
                    let _ = device.close();
                    return Err(CacheError::Init(format!(
                        "probe ibv_alloc_mw(TYPE_2) failed: {e:?}"
                    )));
                }
            };
            let _ = pd.destroy();
            if supported {
                return Ok(device);
            }
        }
        let _ = device.close();
    }
    Err(CacheError::Init(format!(
        "no usable Type-2B RDMA device matching {}",
        requested.unwrap_or("any")
    )))
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
pub(crate) fn alloc_gpu_slot_internal(
    caps: &mut ShardCaps,
    device_id: DeviceId,
    frame_size: usize,
) -> Result<GpuSlot, CacheError> {
    let (buffer, token) = cuda::alloc_device(
        Tracked(caps.alloc.borrow_mut()),
        device_id,
        frame_size,
    )
    .map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
    Ok(GpuSlot {
        buffer,
        token,
        cert: None,
        frame_size,
        device_id,
    })
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
pub(crate) fn upload_gpu_slot_internal(
    core: &mut CacheCore,
    host_tl: u8,
    caps: &mut ShardCaps,
    staging: &CudaHostBuffer,
    staging_tok: &mut Tracked<SlotToken>,
    staging_ctx: &mut HostContext,
    stream: &mut GpuStream,
    slot: &mut GpuSlot,
    data: &[u8],
) -> Result<(), CacheError> {
    if data.len() != slot.frame_size {
        return Err(CacheError::Core(CoreError::InvalidArgument));
    }
    let host_slice = staging.as_host_slice();
    let Tracked(w_comp) = host::write_slice(
        staging_ctx,
        Tracked(caps.instance.borrow()),
        &host_slice,
        data,
        Tracked(staging_tok.borrow_mut()),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    let ghost w_ev = w_comp.submitted.key();
    let t_w = core
        .issue_ticket(host_tl, Tracked(w_comp.submitted))
        .map_err(CacheError::Core)?;
    core.mark_timeline_complete(host_tl, t_w.seq(), Tracked(w_comp.knows));

    let src_cuda = staging.as_cuda_slice();
    let dst_cuda = slot.buffer.slot_slice(0);
    let tracked mut tmp_tok = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(staging_tok.borrow_mut(), &mut tmp_tok);
    }
    let tracked mut src_perm = transfer::context::ReadPerm::Owned(tmp_tok);
    let copy_res = {
        let tracked ledger_ref = core.ledger.borrow_mut();
        cuda::memcpy_async(
            &mut stream.stream,
            Tracked(caps.instance.borrow()),
            &dst_cuda,
            &src_cuda,
            Tracked(&mut src_perm),
            Tracked(slot.token.borrow_mut()),
            Tracked(&mut ledger_ref.submitted),
            Tracked(&ledger_ref.knows),
            Ghost(core.citations()),
        )
    };
    proof {
        *staging_tok.borrow_mut() = src_perm.into_owned();
    }
    let (Ghost(_copy_ev), Tracked(receipt)) =
        copy_res.map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
    let t_copy = core
        .issue_ticket(stream.timeline, Tracked(receipt))
        .map_err(CacheError::Core)?;
    let knows = cuda::stream_synchronize(&stream.stream, Tracked(caps.instance.borrow()))
        .map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
    core.mark_timeline_complete(stream.timeline, t_copy.seq(), knows);
    let ghost slot_cert = ValueCertificate {
        slot: dst_cuda.slot_ref(),
        writer: w_ev,
        bytes: data@,
        proof: ValueProof::HostWrite,
    };
    slot.cert = Some(Ghost(slot_cert));
    Ok(())
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
pub(crate) fn download_gpu_slot_internal(
    core: &mut CacheCore,
    host_tl: u8,
    caps: &mut ShardCaps,
    staging: &CudaHostBuffer,
    staging_tok: &mut Tracked<SlotToken>,
    staging_ctx: &mut HostContext,
    stream: &mut GpuStream,
    slot: &mut GpuSlot,
    dst: &mut [u8],
) -> Result<(), CacheError> {
    if dst.len() != slot.frame_size {
        return Err(CacheError::Core(CoreError::InvalidArgument));
    }
    let src_cuda = slot.buffer.slot_slice(0);
    let dst_cuda = staging.as_cuda_slice();
    let tracked mut tmp_tok = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(slot.token.borrow_mut(), &mut tmp_tok);
    }
    let tracked mut src_perm = transfer::context::ReadPerm::Owned(tmp_tok);
    let copy_res = {
        let tracked ledger_ref = core.ledger.borrow_mut();
        cuda::memcpy_async(
            &mut stream.stream,
            Tracked(caps.instance.borrow()),
            &dst_cuda,
            &src_cuda,
            Tracked(&mut src_perm),
            Tracked(staging_tok.borrow_mut()),
            Tracked(&mut ledger_ref.submitted),
            Tracked(&ledger_ref.knows),
            Ghost(core.citations()),
        )
    };
    proof {
        *slot.token.borrow_mut() = src_perm.into_owned();
    }
    let (_, Tracked(receipt)) =
        copy_res.map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
    let t_copy = core
        .issue_ticket(stream.timeline, Tracked(receipt))
        .map_err(CacheError::Core)?;
    let knows = cuda::stream_synchronize(&stream.stream, Tracked(caps.instance.borrow()))
        .map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
    core.mark_timeline_complete(stream.timeline, t_copy.seq(), knows);

    let host_slice = staging.as_host_slice();
    let tracked mut tmp_stage = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(staging_tok.borrow_mut(), &mut tmp_stage);
    }
    let tracked mut stage_perm = transfer::context::ReadPerm::Owned(tmp_stage);
    let Tracked(r_comp) = host::read_slice(
        staging_ctx,
        Tracked(caps.instance.borrow()),
        &host_slice,
        dst,
        Tracked(&mut stage_perm),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    proof {
        *staging_tok.borrow_mut() = stage_perm.into_owned();
    }
    let t_r = core
        .issue_ticket(host_tl, Tracked(r_comp.submitted))
        .map_err(CacheError::Core)?;
    core.mark_timeline_complete(host_tl, t_r.seq(), Tracked(r_comp.knows));
    Ok(())
}

} // verus!

/// GPU tier configuration for [`StorageConfig`].
#[derive(Clone, Debug)]
pub struct GpuConfig {
    pub device_id: DeviceId,
    pub hbm_frames: u32,
    pub event_ring_size: usize,
    pub pin_host_dram: bool,
}

impl GpuConfig {
    pub fn new(device_id: DeviceId, hbm_frames: u32) -> Self {
        Self {
            device_id,
            hbm_frames,
            event_ring_size: 32,
            pin_host_dram: true,
        }
    }
}

/// Local NVMe (`io_uring` `O_DIRECT`) tier configuration for [`StorageConfig`].
#[derive(Clone, Debug)]
pub struct NvmeConfig {
    pub dir: PathBuf,
    pub nvme_frames: u32,
    pub queue_depth: u32,
}

impl NvmeConfig {
    pub fn new(dir: impl Into<PathBuf>, nvme_frames: u32) -> Self {
        Self {
            dir: dir.into(),
            nvme_frames,
            queue_depth: 64,
        }
    }
}

/// RDMA / GPUDirect RDMA configuration for [`StorageConfig`].
#[derive(Clone, Debug)]
pub struct RdmaConfig {
    pub device_name: Option<String>,
    pub gid_index: u8,
    pub windows_per_peer: usize,
    pub cq_depth: usize,
    pub qp_depth: usize,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            gid_index: 0,
            windows_per_peer: 8,
            cq_depth: 128,
            qp_depth: 64,
        }
    }
}

/// Complete per-node / per-shard hardware tier configuration.
#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub node_id: u64,
    pub frame_size: usize,
    pub dram_frames: u32,
    pub max_leases: usize,
    pub gpu: Option<GpuConfig>,
    pub nvme: Option<NvmeConfig>,
    pub rdma: Option<RdmaConfig>,
}

impl StorageConfig {
    pub fn new(node_id: u64, frame_size: usize, dram_frames: u32) -> Self {
        Self {
            node_id,
            frame_size,
            dram_frames,
            max_leases: 64,
            gpu: None,
            nvme: None,
            rdma: None,
        }
    }

    pub fn with_gpu(mut self, gpu: GpuConfig) -> Self {
        self.gpu = Some(gpu);
        self
    }

    pub fn with_nvme(mut self, nvme: NvmeConfig) -> Self {
        self.nvme = Some(nvme);
        self
    }

    pub fn with_rdma(mut self, rdma: RdmaConfig) -> Self {
        self.rdma = Some(rdma);
        self
    }
}

/// A pre-connected control-plane link to a remote peer node.
pub struct PeerLink<T: ByteTransport> {
    pub peer_node_id: u64,
    pub transport: T,
    pub psn_hint: u32,
}

impl<T: ByteTransport> PeerLink<T> {
    pub fn new(peer_node_id: u64, transport: T) -> Self {
        Self {
            peer_node_id,
            transport,
            psn_hint: 0x1000,
        }
    }
}

/// Step 1 of zero-`unsafe` initialization: owns the base Host DRAM slab and
/// optional RDMA `DeviceContext`.
pub struct NodeStorage {
    pub(crate) config: StorageConfig,
    pub(crate) host: Box<HostBackend>,
    pub(crate) rdma_device: Option<DeviceContext>,
    pub(crate) pending_core: Option<Box<CacheCore>>,
    pub(crate) pending_caps: Option<ShardCaps>,
}

verus! {

impl NodeStorage {
    pub fn new(config: StorageConfig) -> Result<Self, CacheError> {
        let mut caps = ShardCaps::take(config.node_id)?;
        let mut core = Box::new(caps.make_versioned_core(config.frame_size, config.max_leases)?);
        let host = Box::new(
            HostBackend::new(
                &mut core,
                Tracked(caps.instance.borrow()),
                Tracked(caps.alloc.borrow_mut()),
                caps.mint_host(),
                config.dram_frames,
                4096,
            )
            .map_err(CacheError::Core)?,
        );
        let rdma_device = match &config.rdma {
            Some(rcfg) => Some(select_type2_rdma_device(
                &mut caps,
                rcfg.device_name.as_deref(),
            )?),
            None => None,
        };
        Ok(Self {
            config,
            host,
            rdma_device,
            pending_core: Some(core),
            pending_caps: Some(caps),
        })
    }

    pub fn node_id(&self) -> u64 {
        self.config.node_id
    }

    pub fn frame_size(&self) -> usize {
        self.config.frame_size
    }

    /// Step 2 of zero-`unsafe` initialization: attach optional CUDA (pinning the
    /// Host DRAM slab for 1-hop DMA) and RDMA `ProtectionDomain`.
    pub fn attach_tiers(&mut self) -> Result<AttachedStorage<'_>, CacheError> {
        let mut core = self
            .pending_core
            .take()
            .ok_or_else(|| CacheError::Init("NodeStorage::attach_tiers already called".into()))?;
        let mut caps = self
            .pending_caps
            .take()
            .ok_or_else(|| CacheError::Init("NodeStorage::attach_tiers already called".into()))?;

        let cuda = match &self.config.gpu {
            Some(gcfg) => {
                let mut cb = Box::new(
                    CudaBackend::new(
                        &mut core,
                        Tracked(caps.instance.borrow()),
                        Tracked(caps.alloc.borrow_mut()),
                        caps.mint_cuda(),
                        gcfg.device_id,
                        gcfg.hbm_frames,
                        gcfg.event_ring_size,
                        gcfg.pin_host_dram,
                    )
                    .map_err(CacheError::Core)?,
                );
                if gcfg.pin_host_dram {
                    cb.attach_pinned_host(&core, &self.host)
                        .map_err(CacheError::Core)?;
                }
                Some(cb)
            }
            None => None,
        };

        let pd = match &self.rdma_device {
            Some(dev) => Some(
                ProtectionDomain::new(dev, caps.mint_rdma())
                    .map_err(|e| CacheError::Init(format!("ProtectionDomain::new failed: {e:?}")))?,
            ),
            None => None,
        };

        Ok(AttachedStorage {
            config: self.config.clone(),
            host: &self.host,
            cuda,
            rdma_device: self.rdma_device.as_ref(),
            pd,
            pending_core: Some(core),
            pending_caps: Some(caps),
        })
    }
}

/// Step 2 of zero-`unsafe` initialization: borrows `&'s HostBackend` / `&'s DeviceContext`
/// and owns optional `CudaBackend<'s>` and `ProtectionDomain<'s>`.
pub struct AttachedStorage<'s> {
    pub(crate) config: StorageConfig,
    pub(crate) host: &'s HostBackend,
    pub(crate) cuda: Option<Box<CudaBackend<'s>>>,
    pub(crate) rdma_device: Option<&'s DeviceContext>,
    pub(crate) pd: Option<ProtectionDomain<'s>>,
    pub(crate) pending_core: Option<Box<CacheCore>>,
    pub(crate) pending_caps: Option<ShardCaps>,
}

impl<'s> AttachedStorage<'s> {
    /// Step 3: connect optional NVMe and `0..N` RDMA/GDR peers and open a
    /// location-transparent [`VeriCache`] handle governed by `policy`.
    pub fn open_cache<'a, T, E, P, R>(
        &'a mut self,
        peers: Vec<PeerLink<T>>,
        policy: PolicySet<E, P, R>,
    ) -> Result<VeriCache<'a, T, E, P, R>, CacheError>
    where
        's: 'a,
        T: ByteTransport,
        E: EvictionPolicy,
        P: TieringPolicy,
        R: PeerRoutingPolicy,
    {
        let mut core = self
            .pending_core
            .take()
            .ok_or_else(|| CacheError::Init("AttachedStorage::open_cache already called".into()))?;
        let mut caps = self
            .pending_caps
            .take()
            .ok_or_else(|| CacheError::Init("AttachedStorage::open_cache already called".into()))?;

        let nvme = match &self.config.nvme {
            Some(ncfg) => {
                let dir_str = ncfg
                    .dir
                    .to_str()
                    .ok_or_else(|| CacheError::Init("NVMe directory path is not valid UTF-8".into()))?;
                Some(
                    NvmeBackend::new(
                        &mut core,
                        &self.host.ctx,
                        Tracked(caps.instance.borrow()),
                        Tracked(caps.alloc.borrow_mut()),
                        caps.mint_nvme(),
                        dir_str,
                        ncfg.nvme_frames,
                        ncfg.queue_depth,
                    )
                    .map_err(CacheError::Core)?,
                )
            }
            None => None,
        };

        let mut peer_sessions = Vec::with_capacity(peers.len());
        let mut peer_metas = Vec::with_capacity(peers.len());
        if !peers.is_empty() {
            let rcfg = self.config.rdma.as_ref().ok_or_else(|| {
                CacheError::Init("peers supplied without RdmaConfig in StorageConfig".into())
            })?;
            let dev = self
                .rdma_device
                .ok_or_else(|| CacheError::Init("RDMA device not initialized".into()))?;
            let pd = self
                .pd
                .as_ref()
                .ok_or_else(|| CacheError::Init("RDMA ProtectionDomain not initialized".into()))?;

            for (idx, link) in peers.into_iter().enumerate() {
                let cq = CompletionQueue::new(dev, rcfg.cq_depth as i32, caps.mint_rdma())
                    .map_err(|e| CacheError::Init(format!("CompletionQueue::new failed: {e:?}")))?;
                let psn = link
                    .psn_hint
                    .wrapping_add(((self.config.node_id as u32) << 8) + idx as u32);
                let reset = UnconnectedQueuePair::new(
                    pd,
                    &cq,
                    rcfg.qp_depth as u32,
                    1,
                    rcfg.gid_index,
                    psn,
                    caps.mint_rdma(),
                )
                .map_err(|e| CacheError::Init(format!("UnconnectedQueuePair::new failed: {e:?}")))?;
                let mut channel: Channel<T, VeriCacheAttachment> = Channel::new_with_protocol(
                    link.transport,
                    self.config.node_id,
                    link.peer_node_id,
                    Tracked(caps.instance.borrow()),
                );
                let peer_ep = channel
                    .exchange_queue_pair_endpoint(
                        &reset.endpoint(),
                        Tracked(caps.instance.borrow()),
                    )
                    .map_err(|_| {
                        CacheError::Init("exchange_queue_pair_endpoint failed".into())
                    })?;
                let qp = connect_rc(reset, peer_ep, link.peer_node_id)
                    .map_err(|e| CacheError::Init(format!("connect_rc failed: {e:?}")))?;

                let mut windows = Vec::with_capacity(rcfg.windows_per_peer);
                for _ in 0..rcfg.windows_per_peer {
                    windows.push(
                        alloc_window(pd, caps.mint_rdma())
                            .map_err(|e| CacheError::Init(format!("alloc_window failed: {e:?}")))?,
                    );
                }

                let mut rdma = RdmaBackend::new(&mut core, link.peer_node_id, qp, cq, windows)
                    .map_err(|(e, _, _, _)| CacheError::Core(e))?;
                let flags = AccessFlags {
                    local_write: true,
                    mw_bind: true,
                };
                rdma.register_host_slab(
                    &core,
                    pd,
                    &self.host.buffer,
                    self.host.tier,
                    self.host.first_frame,
                    self.host.num_frames,
                    flags,
                )
                .map_err(CacheError::Core)?;

                let mut supports_gdr = false;
                if let Some(ref c) = self.cuda {
                    rdma.register_gpu_slab(
                        &core,
                        pd,
                        &c.gpu_buffer,
                        c.gpu_tier,
                        c.gpu_first_frame,
                        c.gpu_num_frames,
                        flags,
                    )
                    .map_err(CacheError::Core)?;
                    supports_gdr = true;
                }

                let session = PeerSession::new(rdma, channel, &core);
                peer_sessions.push(session);
                peer_metas.push(PeerMeta {
                    index: idx,
                    node_id: link.peer_node_id,
                    supports_gdr,
                    supports_host_rdma: true,
                });
            }
        }

        // Internal CUDA stream + staging buffer when GPU is enabled, so HBM<->DRAM
        // transparent hops and GpuSlot upload/download require zero extra user setup.
        let (internal_gpu_stream, gpu_staging) = match &self.cuda {
            Some(c) => {
                let dev = c.device;
                let tl = core
                    .register_timeline(TimelineKind::CudaStream { device: dev })
                    .map_err(CacheError::Core)?;
                let stream = CudaStream::new(dev, caps.mint_cuda())
                    .map_err(|e| CacheError::Init(format!("internal CudaStream failed: {e:?}")))?;
                let (buf, tok) =
                    cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), self.config.frame_size)
                        .map_err(|e| {
                            CacheError::Init(format!("alloc pinned staging failed: {e:?}"))
                        })?;
                let hctx = HostContext::new(caps.mint_host());
                (
                    Some(GpuStream {
                        stream,
                        timeline: tl,
                        device_id: dev,
                    }),
                    Some((buf, tok, hctx)),
                )
            }
            None => (None, None),
        };

        let eng_ctx = HostContext::new(caps.mint_host());
        let cuda_att = CudaAttachment::from_option_borrowed(self.cuda.as_deref());
        let engine = Box::new(DistributedEngine::new_multi_peer(
            *core,
            self.host,
            eng_ctx,
            cuda_att,
            nvme,
            peer_sessions,
        ));

        Ok(VeriCache::from_parts(
            self.config.node_id,
            self.config.frame_size,
            engine,
            caps,
            self.pd.as_ref(),
            peer_metas,
            policy,
            internal_gpu_stream,
            gpu_staging,
        ))
    }
}

} // verus!
