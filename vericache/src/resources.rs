//! Storage initialization from `NodeStorage` through `AttachedStorage` to `VeriCache`.

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::{
    AllocSource, EventSourceAllocator, EventSourceKind, EventSourceToken, NodeContext,
    ShardAllocator, ShardSources, ShareSource, TransferInstance, VersionInstance,
    VersionSource,
};
use transfer::domain::DeviceId;
use transfer::host::HostContext;
use transfer::rdma::{
    AccessFlags, CompletionQueue, DeviceContext, ProtectionDomain, RdmaError,
    UnconnectedQueuePair, alloc_window, connect_rc, dealloc_window,
};
use transfer::transport::{ByteTransport, Channel};
use vericache_core::backends::DeviceAttachment;
use vericache_core::backends::host::HostBackend;
#[cfg(feature = "nvme")]
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::backends::rdma::peer::PeerSession;
use vericache_core::backends::rdma::protocol::VeriCacheAttachment;
use vericache_core::catalog::Catalog;
use vericache_core::engine::DistributedEngine;
use vericache_core::kernel::CacheCore;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Tracked, verus};

use crate::cache::VeriCache;
#[cfg(feature = "cuda")]
pub use crate::cuda_runtime::{CudaRuntime, GpuSlot, GpuStream};
use crate::device::DeviceRuntime;
use crate::policy::{EvictionPolicy, PeerMeta, PeerRoutingPolicy, PolicySet, TieringPolicy};
use crate::{CacheError, DefaultDeviceRuntime};

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

/// Per-shard capabilities used to initialize backends and allocate caller resources.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct ShardCaps {
    pub(crate) node: u64,
    pub instance: Tracked<TransferInstance>,
    pub alloc: Tracked<AllocSource>,
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

    /// Mint an event source for this shard's runtime or driver.
    pub fn mint_event(&mut self, Ghost(kind): Ghost<EventSourceKind>) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(kind))
    }

    pub fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        self.mint_event(Ghost(EventSourceKind::host()))
    }

    #[cfg(feature = "nvme")]
    pub(crate) fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::io_uring()))
    }

    pub(crate) fn mint_rdma(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::rdma_qp()))
    }

    pub(crate) fn make_versioned_core(
        &mut self,
        frame_size: usize,
        max_leases: usize,
        catalog_buckets: usize,
    ) -> Result<CacheCore, CacheError> {
        let ver_src = self
            .versions
            .take()
            .ok_or_else(|| CacheError::Init("shard VersionSource already consumed".into()))?;
        Ok(CacheCore::from_shard_versions_with_buckets(
            self.node as u16,
            frame_size,
            Vec::new(),
            max_leases,
            self.version_start,
            self.version_limit,
            catalog_buckets,
            Tracked(self.instance.borrow()),
            Tracked(self.version_instance.borrow()),
            ver_src,
        ))
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
pub struct NodeStorage<RT: DeviceRuntime = DefaultDeviceRuntime> {
    pub(crate) config: StorageConfig,
    pub(crate) host: Box<HostBackend>,
    pub(crate) rdma_device: Option<DeviceContext>,
    pub(crate) pending_core: Option<Box<CacheCore>>,
    pub(crate) pending_caps: Option<ShardCaps>,
    runtime: PhantomData<fn() -> RT>,
}

impl NodeStorage<DefaultDeviceRuntime> {
    pub fn new(config: StorageConfig) -> Result<Self, CacheError> {
        Self::new_with_runtime(config)
    }
}

verus! {

impl<RT: DeviceRuntime> NodeStorage<RT> {
    /// Allocate storage for the selected runtime. Device resources attach in `attach_tiers`.
    pub fn new_with_runtime(config: StorageConfig) -> Result<Self, CacheError> {
        #[cfg(not(feature = "nvme"))]
        if config.nvme.is_some() {
            return Err(vericache_core::spec::CoreError::UnsupportedTier.into());
        }
        let mut caps = ShardCaps::take(config.node_id)?;
        let total_frames = (config.dram_frames as usize)
            .saturating_add(config.gpu.as_ref().map_or(0, |g| g.hbm_frames as usize))
            .saturating_add(config.nvme.as_ref().map_or(0, |n| n.nvme_frames as usize));
        let mut core = Box::new(caps.make_versioned_core(
            config.frame_size,
            config.max_leases,
            Catalog::recommended_buckets(total_frames),
        )?);
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
            runtime: PhantomData,
        })
    }

    pub fn node_id(&self) -> u64 {
        self.config.node_id
    }

    pub fn frame_size(&self) -> usize {
        self.config.frame_size
    }

    /// Attach the optional device tier, pinned Host DRAM, and RDMA protection domain.
    pub fn attach_tiers(&mut self) -> Result<AttachedStorage<'_, RT>, CacheError> {
        let mut core = self
            .pending_core
            .take()
            .ok_or_else(|| CacheError::Init("NodeStorage::attach_tiers already called".into()))?;
        let mut caps = self
            .pending_caps
            .take()
            .ok_or_else(|| CacheError::Init("NodeStorage::attach_tiers already called".into()))?;

        let device = match &self.config.gpu {
            Some(gcfg) => {
                let mut backend = Box::new(RT::new_backend(&mut core, &mut caps, gcfg)?);
                if gcfg.pin_host_dram {
                    RT::attach_pinned_host(&mut backend, &core, &self.host)?;
                }
                Some(backend)
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
            device,
            rdma_device: self.rdma_device.as_ref(),
            pd,
            pending_core: Some(core),
            pending_caps: Some(caps),
        })
    }
}

/// Borrows node storage and owns the optional device backend and RDMA protection domain.
pub struct AttachedStorage<'s, RT: DeviceRuntime = DefaultDeviceRuntime> {
    pub(crate) config: StorageConfig,
    pub(crate) host: &'s HostBackend,
    pub(crate) device: Option<Box<RT::Backend<'s>>>,
    pub(crate) rdma_device: Option<&'s DeviceContext>,
    pub(crate) pd: Option<ProtectionDomain<'s>>,
    pub(crate) pending_core: Option<Box<CacheCore>>,
    pub(crate) pending_caps: Option<ShardCaps>,
}

impl<'s, RT: DeviceRuntime> AttachedStorage<'s, RT> {
    /// Step 3: connect optional NVMe and `0..N` RDMA/GDR peers and open a
    /// location-transparent [`VeriCache`] handle governed by `policy`.
    pub fn open_cache<'a, T, E, P, R>(
        &'a mut self,
        peers: Vec<PeerLink<T>>,
        policy: PolicySet<E, P, R>,
    ) -> Result<VeriCache<'a, T, E, P, R, RT>, CacheError>
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

        #[cfg(feature = "nvme")]
        let nvme = match &self.config.nvme {
            Some(ncfg) => {
                let dir_str = ncfg
                    .dir
                    .to_str()
                    .ok_or_else(|| CacheError::Init("NVMe directory path is not valid UTF-8".into()))?;
                Some(
                    NvmeBackend::new(
                        &mut core,
                        self.host.context(),
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
        #[cfg(not(feature = "nvme"))]
        let nvme = None;

        let device = self.device.as_deref().map(RT::borrow_backend);
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
                    self.host.buffer(),
                    self.host.tier_id(),
                    self.host.first_frame(),
                    self.host.num_frames(),
                    flags,
                )
                .map_err(CacheError::Core)?;

                if let Some(backend) = device {
                    RT::register_device_slab(backend, &mut rdma, &core, pd, flags)?;
                }

                let session = PeerSession::new(rdma, channel, &core);
                peer_sessions.push(session);
                peer_metas.push(PeerMeta {
                    index: idx,
                    node_id: link.peer_node_id,
                    supports_gdr: device.is_some(),
                    supports_host_rdma: true,
                });
            }
        }

        let eng_ctx = HostContext::new(caps.mint_host());
        let mut engine = Box::new(DistributedEngine::new_multi_peer(
            *core,
            self.host,
            eng_ctx,
            DeviceAttachment::from_option_borrowed(device),
            nvme,
            peer_sessions,
        ).map_err(CacheError::Core)?);

        // Provision resources for internal cross-tier hops and caller slot transfers.
        let (internal_device_stream, device_staging) = if device.is_some() {
            (
                Some(RT::new_stream(&mut engine, &mut caps)?),
                Some(RT::new_staging(&mut caps, self.config.frame_size)?),
            )
        } else {
            (None, None)
        };

        Ok(VeriCache::from_parts(
            self.config.node_id,
            self.config.frame_size,
            engine,
            caps,
            self.pd.as_ref(),
            peer_metas,
            policy,
            internal_device_stream,
            device_staging,
        ))
    }
}

} // verus!
