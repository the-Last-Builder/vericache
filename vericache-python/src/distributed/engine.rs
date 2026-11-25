//! Multi-shard Distributed VeriCache engine over verified `DistributedEngine`
//! (Host DRAM + optional Local NVMe `io_uring` + cross-instance Type-2B MW RDMA).
//!
//! Architecture:
//! - Each shard runs on a dedicated OS worker thread owning its `HostBackend`,
//!   optional `CudaHostRegistration`, optional `NvmeBackend`, RDMA `DeviceContext`,
//!   `ProtectionDomain`, shared `RegisteredSlab` MR, and `DistributedEngine`.
//! - Peer connections are established and recovered asynchronously inside each shard's
//!   worker thread so vLLM startup never blocks and peer crashes/restarts never deadlock.
//! - Idle shards poll `poll_peer_progress_multi` every ~100 us so inbound remote RDMA
//!   `FetchRequest`s are served concurrently with local Python/PyTorch execution.
//! - DRAM evictions enforce cooperative-first (`SoftRevokeRequest -> ReadRelease -> LOCAL_INV`)
//!   with Option-A unilateral fallback (`ibv_dealloc_mw`) via `DistributedEngine`, recording
//!   exact revocation counts and latencies.

use super::shard_cap::DistShardCaps;
use super::transport::BufferedTcpTransport;
use crate::backend_mode;
use std::collections::{HashMap, VecDeque};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use transfer::context::{EventSourceToken, ShareCap};
use transfer::cuda::{self, CudaHostRegistration};
use transfer::host::HostContext;
use transfer::rdma::{
    AccessFlags, CompletionQueue, DeviceContext, ProtectionDomain, RdmaError,
    UnconnectedQueuePair, alloc_window, connect_rc, dealloc_window,
};
use transfer::transport::Channel;
use vericache::policy::LruList;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, PeerSession};
use vericache_core::backends::rdma::protocol::VeriCacheAttachment;
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::backends::rdma::{RdmaBackend, RegisteredSlab};
use vericache_core::engine::{CudaAttachment, DistributedEngine};
use vericache_core::frame::{STATE_EMPTY, STATE_PUBLISHED};
use vericache_core::kernel::ReadLeaseHandle;
use vericache_core::spec::{CoreError, FrameId, ObjectKey, Ticket, TierId};
use vstd::prelude::{Tracked, verus};

pub const LOC_MISS: i32 = 0;
pub const LOC_LOCAL_DRAM: i32 = 2;
pub const LOC_LOCAL_NVME: i32 = 3;
pub const LOC_REMOTE_RDMA_HOST: i32 = 4;

const IDLE_POLL_US: u64 = 25;
const LOCAL_ONLY_IDLE_POLL_US: u64 = 100_000;
const RECONNECT_RETRY_MS: u64 = 100;
const TCP_IO_TIMEOUT_MS: u64 = 1500;
const DEFAULT_COOP_SPIN_BUDGET: u32 = 512;

/// Serializes `cudaHostRegister` + `ibv_reg_mr` across shard threads to avoid mlx5
/// driver `EAGAIN` (errno 11) bursts when many multi-GiB slabs register simultaneously.
static SLAB_REG_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug)]
pub struct DistPeerSpec {
    pub peer_node_id: u64,
    pub tcp_host: String,
    pub base_port: u16,
    pub is_listener: bool,
    pub psn_hint: u32,
}

#[derive(Clone, Debug)]
pub struct DistEngineConfig {
    pub node_id: u64,
    pub num_shards: usize,
    pub host_frames_per_shard: u32,
    pub nvme_frames_per_shard: u32,
    pub frame_size: usize,
    pub nvme_path: Option<String>,
    pub prefer_cuda_pinned: bool,
    pub rdma_device: Option<String>,
    pub rdma_gid_index: u8,
    pub rdma_windows_per_peer: usize,
    pub peer_fetch_enabled: bool,
    pub peers: Vec<DistPeerSpec>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistLeaseResult {
    pub lease_id: u64,
    pub address: usize,
    pub len: usize,
    pub location: i32,
    /// `0 = hit`, `1 = miss`, `< 0 = error`
    pub status: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistStats {
    pub puts: u64,
    pub gets: u64,
    pub hits: u64,
    pub misses: u64,
    pub dram_hits: u64,
    pub nvme_hits: u64,
    pub remote_hits: u64,
    pub dram_evictions: u64,
    pub nvme_evictions: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub nvme_read_bytes: u64,
    pub nvme_write_bytes: u64,
    pub rdma_fetch_bytes: u64,
    pub rdma_fetches_ok: u64,
    pub rdma_fetches_miss: u64,
    pub rdma_fetches_err: u64,
    pub rdma_grants_served: u64,
    pub mw_revocations_cooperative: u64,
    pub mw_revocations_unilateral: u64,
    pub mw_revocation_latency_us_sum: u64,
    pub mw_revocation_latency_us_max: u64,
    pub peer_disconnects: u64,
    pub peer_reconnects: u64,
    pub peers_connected: u64,
    pub active_leases: u64,
    pub dram_resident_keys: u64,
}

impl DistStats {
    fn accumulate(&mut self, other: &DistStats) {
        self.puts += other.puts;
        self.gets += other.gets;
        self.hits += other.hits;
        self.misses += other.misses;
        self.dram_hits += other.dram_hits;
        self.nvme_hits += other.nvme_hits;
        self.remote_hits += other.remote_hits;
        self.dram_evictions += other.dram_evictions;
        self.nvme_evictions += other.nvme_evictions;
        self.bytes_written += other.bytes_written;
        self.bytes_read += other.bytes_read;
        self.nvme_read_bytes += other.nvme_read_bytes;
        self.nvme_write_bytes += other.nvme_write_bytes;
        self.rdma_fetch_bytes += other.rdma_fetch_bytes;
        self.rdma_fetches_ok += other.rdma_fetches_ok;
        self.rdma_fetches_miss += other.rdma_fetches_miss;
        self.rdma_fetches_err += other.rdma_fetches_err;
        self.rdma_grants_served += other.rdma_grants_served;
        self.mw_revocations_cooperative += other.mw_revocations_cooperative;
        self.mw_revocations_unilateral += other.mw_revocations_unilateral;
        self.mw_revocation_latency_us_sum += other.mw_revocation_latency_us_sum;
        self.mw_revocation_latency_us_max = self
            .mw_revocation_latency_us_max
            .max(other.mw_revocation_latency_us_max);
        self.peer_disconnects += other.peer_disconnects;
        self.peer_reconnects += other.peer_reconnects;
        self.peers_connected += other.peers_connected;
        self.active_leases += other.active_leases;
        self.dram_resident_keys += other.dram_resident_keys;
    }
}

struct ActiveLocalLease {
    handle: ReadLeaseHandle,
    cap: Tracked<ShareCap>,
}

struct PeerSlot {
    spec: DistPeerSpec,
    addr: String,
    listener: Option<TcpListener>,
    connected: bool,
    ever_connected: bool,
    last_attempt: Instant,
    connect_epoch: u32,
}

enum ShardCmd {
    PutBatch {
        /// `(key, src_ptr, len)` — caller blocks until reply, keeping raw pointers valid.
        items: Vec<(ObjectKey, usize, usize)>,
        reply: Sender<Vec<i32>>,
    },
    ContainsBatch {
        keys: Vec<ObjectKey>,
        /// If true, check DRAM only (RDMA-servable); if false, check DRAM or NVMe.
        dram_only: bool,
        reply: Sender<Vec<u8>>,
    },
    AcquireLeaseBatch {
        /// `(key, allow_remote)`
        items: Vec<(ObjectKey, bool)>,
        reply: Sender<Vec<DistLeaseResult>>,
    },
    ReleaseLeases {
        lease_ids: Vec<u64>,
        reply: Sender<()>,
    },
    GetBatch {
        /// `(key, dst_ptr, cap, allow_remote)` — caller blocks until reply.
        items: Vec<(ObjectKey, usize, usize, bool)>,
        reply: Sender<Vec<(usize, i32, i32)>>,
    },
    DrainEvictions {
        max_keys: usize,
        reply: Sender<Vec<ObjectKey>>,
    },
    SetDramBudgetFrames {
        max_frames: usize,
        reply: Sender<usize>,
    },
    SetCoopRevokeSpinBudget {
        spins: u32,
        reply: Sender<()>,
    },
    EvictLocalDramKeys {
        keys: Vec<ObjectKey>,
        reply: Sender<usize>,
    },
    Poll {
        reply: Sender<()>,
    },
    Stats {
        reply: Sender<DistStats>,
    },
    Shutdown {
        reply: Sender<()>,
    },
}

struct ShardHandle {
    tx: Mutex<Sender<ShardCmd>>,
    worker: Option<JoinHandle<()>>,
}

impl ShardHandle {
    fn send(&self, cmd: ShardCmd) -> bool {
        match self.tx.lock() {
            Ok(guard) => guard.send(cmd).is_ok(),
            Err(_) => false,
        }
    }
}

pub struct DistributedVllmEngine {
    shards: Vec<ShardHandle>,
    node_id: u64,
    frame_size: usize,
    has_nvme: bool,
    cuda_pinned: bool,
    peer_fetch_enabled: Arc<AtomicBool>,
    peers_connected_total: Arc<AtomicU64>,
}

#[inline]
pub fn shard_for_key(key: ObjectKey, num_shards: usize) -> usize {
    let mut h = key.0[0] ^ key.0[1].rotate_left(13) ^ key.0[2].rotate_left(29) ^ key.0[3];
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as usize) % num_shards
}

fn select_type2_rdma_device(
    caps: &mut DistShardCaps,
    requested: Option<&str>,
) -> Result<DeviceContext, String> {
    let count =
        DeviceContext::count().map_err(|e| format!("ibv_get_device_list failed: {e:?}"))?;
    for index in 0..count {
        let device = DeviceContext::open(index, caps.node)
            .map_err(|e| format!("open verbs device {index} failed: {e:?}"))?;
        if requested.is_some_and(|name| name != device.name()) {
            let _ = device.close();
            continue;
        }
        let type2 = device
            .supports_type_2()
            .map_err(|e| format!("query type-2 MW caps failed: {e:?}"))?;
        if type2 {
            let pd = ProtectionDomain::new(&device, caps.mint_rdma())
                .map_err(|e| format!("probe PD failed: {e:?}"))?;
            let supported = match alloc_window(&pd, caps.mint_rdma()) {
                Ok(w) => {
                    let _ = dealloc_window(w);
                    true
                }
                Err(RdmaError(38)) | Err(RdmaError(95)) => false,
                Err(e) => {
                    let _ = pd.destroy();
                    let _ = device.close();
                    return Err(format!("probe ibv_alloc_mw(TYPE_2) failed: {e:?}"));
                }
            };
            let _ = pd.destroy();
            if supported {
                return Ok(device);
            }
        }
        let _ = device.close();
    }
    Err(format!(
        "no usable Type-2B RDMA device matching {}",
        requested.unwrap_or("any")
    ))
}

struct ShardRuntime<'a> {
    shard_idx: usize,
    frame_size: usize,
    host_capacity: usize,
    dram_budget_frames: usize,
    nvme_capacity: usize,
    rdma_gid_index: u8,
    rdma_windows_per_peer: usize,
    coop_spin_budget: u32,
    peer_fetch_enabled: Arc<AtomicBool>,
    peers_connected_total: Arc<AtomicU64>,
    caps: DistShardCaps,
    rdma_dev: Option<&'a DeviceContext>,
    pd: Option<&'a ProtectionDomain<'a>>,
    shared_host_mr: Option<&'a RegisteredSlab<'a>>,
    engine: DistributedEngine<'a, BufferedTcpTransport>,
    peer_slots: Vec<PeerSlot>,
    replenish_tokens: Vec<Tracked<EventSourceToken>>,
    dram_lru: LruList<ObjectKey, FrameId>,
    nvme_lru: LruList<ObjectKey, FrameId>,
    leases: HashMap<u64, ActiveLocalLease>,
    lease_seq: u64,
    evicted_dram_keys: VecDeque<ObjectKey>,
    stats: DistStats,
}

verus! {

impl<'a> ShardRuntime<'a> {
    fn ensure_replenish_tokens(&mut self, min_count: usize) {
        while self.replenish_tokens.len() < min_count {
            self.replenish_tokens.push(self.caps.mint_rdma());
        }
    }

    fn next_lease_id(&mut self) -> u64 {
        self.lease_seq = self.lease_seq.wrapping_add(1).max(1);
        ((self.shard_idx as u64) << 48) | (self.lease_seq & 0x0000_ffff_ffff_ffff)
    }

    fn frame_is_published_on(&self, frame: FrameId, tier: TierId) -> bool {
        let idx = frame.0 as usize;
        if idx >= self.engine.core.frames.metas.len() {
            return false;
        }
        self.engine.core.frames.metas[idx].state == STATE_PUBLISHED
            && self.engine.core.frames.bindings[idx].1 == tier
    }

    fn attach_peer_stream(&mut self, slot_idx: usize, stream: TcpStream) -> Result<(), String> {
        let dev = self
            .rdma_dev
            .ok_or_else(|| "RDMA device not initialized".to_string())?;
        let pd = self
            .pd
            .ok_or_else(|| "RDMA ProtectionDomain not initialized".to_string())?;
        let shared_mr = self
            .shared_host_mr
            .ok_or_else(|| "RDMA shared host MR not initialized".to_string())?;

        // If this peer_node_id is already attached (stale half-open connection), tear it down first.
        let peer_node_id = self.peer_slots[slot_idx].spec.peer_node_id;
        if let Some(existing_idx) = self.engine.find_peer_index(peer_node_id) {
            self.disconnect_peer_by_engine_idx(existing_idx);
        }

        let transport = BufferedTcpTransport::new(stream, Duration::from_millis(TCP_IO_TIMEOUT_MS))
            .map_err(|e| format!("BufferedTcpTransport::new failed: {e}"))?;

        let cq = CompletionQueue::new(dev, 128, self.caps.mint_rdma())
            .map_err(|e| format!("CompletionQueue::new failed: {e:?}"))?;

        let epoch = self.peer_slots[slot_idx].connect_epoch;
        self.peer_slots[slot_idx].connect_epoch = epoch.wrapping_add(1);
        let psn = (self.peer_slots[slot_idx]
            .spec
            .psn_hint
            .wrapping_add(((self.caps.node as u32) << 12) ^ ((self.shard_idx as u32) << 6) ^ epoch))
            & 0x00ff_ffff;

        let reset = UnconnectedQueuePair::new(
            pd,
            &cq,
            64,
            1,
            self.rdma_gid_index,
            psn,
            self.caps.mint_rdma(),
        )
        .map_err(|e| format!("UnconnectedQueuePair::new failed: {e:?}"))?;

        let mut channel: Channel<BufferedTcpTransport, VeriCacheAttachment> =
            Channel::new_with_protocol(
                transport,
                self.caps.node,
                peer_node_id,
                Tracked(self.caps.instance.borrow()),
            );

        let peer_ep = channel
            .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked(self.caps.instance.borrow()))
            .map_err(|_| "exchange_queue_pair_endpoint failed".to_string())?;

        let qp = connect_rc(reset, peer_ep, peer_node_id)
            .map_err(|e| format!("connect_rc failed: {e:?}"))?;

        let mut windows = Vec::with_capacity(self.rdma_windows_per_peer);
        for _ in 0..self.rdma_windows_per_peer {
            windows.push(
                alloc_window(pd, self.caps.mint_rdma())
                    .map_err(|e| format!("alloc_window failed: {e:?}"))?,
            );
        }

        let mut rdma = RdmaBackend::new(&mut self.engine.core, peer_node_id, qp, cq, windows)
            .map_err(|(e, _, _, _)| format!("RdmaBackend::new failed: {e:?}"))?;

        rdma.attach_shared_host_slab(&self.engine.core, shared_mr)
            .map_err(|e| format!("attach_shared_host_slab failed: {e:?}"))?;

        let session = PeerSession::new(rdma, channel, &self.engine.core);
        self.engine.add_peer(session);

        let slot = &mut self.peer_slots[slot_idx];
        slot.connected = true;
        if slot.ever_connected {
            self.stats.peer_reconnects += 1;
        } else {
            slot.ever_connected = true;
        }
        self.stats.peers_connected = self.engine.peers.len() as u64;
        self.peers_connected_total.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn disconnect_peer_by_engine_idx(&mut self, peer_idx: usize) {
        if peer_idx >= self.engine.peers.len() {
            return;
        }
        let peer_node_id = self.engine.peers[peer_idx].rdma.peer_node;
        let _ = self.engine.peers[peer_idx].shutdown(
            &mut self.engine.core,
            &mut self.engine.host_ctx,
            Tracked(self.caps.instance.borrow()),
        );
        self.engine.peers.swap_remove(peer_idx);
        let _ = self.engine.reclaim_all_draining();
        for slot in &mut self.peer_slots {
            if slot.spec.peer_node_id == peer_node_id && slot.connected {
                slot.connected = false;
                slot.last_attempt = Instant::now();
                self.stats.peer_disconnects += 1;
                self.peers_connected_total.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.stats.peers_connected = self.engine.peers.len() as u64;
    }

    fn try_connect_peers(&mut self) {
        let now = Instant::now();
        for slot_idx in 0..self.peer_slots.len() {
            if self.peer_slots[slot_idx].spec.is_listener {
                // Even if marked connected, accept a fresh connection from a restarted peer.
                let accepted = self.peer_slots[slot_idx]
                    .listener
                    .as_ref()
                    .and_then(|l| match l.accept() {
                        Ok((s, _)) => Some(s),
                        Err(_) => None,
                    });
                if let Some(stream) = accepted {
                    let _ = self.attach_peer_stream(slot_idx, stream);
                }
            } else if !self.peer_slots[slot_idx].connected {
                if now.duration_since(self.peer_slots[slot_idx].last_attempt)
                    < Duration::from_millis(RECONNECT_RETRY_MS)
                {
                    continue;
                }
                self.peer_slots[slot_idx].last_attempt = now;
                let addr_str = self.peer_slots[slot_idx].addr.clone();
                let sock_addr: Option<SocketAddr> =
                    addr_str.to_socket_addrs().ok().and_then(|mut it| it.next());
                if let Some(sa) = sock_addr {
                    if let Ok(stream) =
                        TcpStream::connect_timeout(&sa, Duration::from_millis(50))
                    {
                        let _ = self.attach_peer_stream(slot_idx, stream);
                    }
                }
            }
        }
    }

    fn poll_peers_safe(&mut self) {
        if self.engine.peers.is_empty() {
            return;
        }
        let host_tier = self.engine.host_tier();
        let mut idx = 0usize;
        while idx < self.engine.peers.len() {
            self.ensure_replenish_tokens(4);
            let res = self.engine.poll_peer_progress_multi(
                idx,
                host_tier,
                None,
                self.pd,
                &mut self.replenish_tokens,
                Tracked(self.caps.shares.borrow_mut()),
                Tracked(self.caps.instance.borrow()),
            );
            let closed = self.engine.peers[idx].channel.is_closed();
            match res {
                Ok(prog) if !closed => {
                    self.stats.rdma_grants_served += prog.grants_completed as u64;
                    idx += 1;
                }
                _ => {
                    self.disconnect_peer_by_engine_idx(idx);
                    // swap_remove brought another peer into `idx`; re-check same index.
                }
            }
        }
        let _ = self.engine.reclaim_all_draining();
    }

    fn revoke_and_evict_dram_frame(&mut self, vkey: ObjectKey, vframe: FrameId) -> bool {
        let host_tier = self.engine.host_tier();
        if !self.frame_is_published_on(vframe, host_tier) {
            return false;
        }

        // Optional cascade demotion to NVMe before removing from DRAM.
        if let Some(nvme_tier) = self.engine.nvme_tier() {
            if self.engine.lookup_tier(vkey, nvme_tier).is_none()
                && self.ensure_nvme_space().is_ok()
            {
                if let Ok(new_nf) = self.engine.spill_host_to_nvme_sync(
                    Tracked(self.caps.instance.borrow()),
                    Tracked(self.caps.shares.borrow_mut()),
                    vkey,
                ) {
                    self.nvme_lru.insert(vkey, new_nf);
                    self.stats.nvme_write_bytes += self.frame_size as u64;
                }
            }
        }

        // Cooperative-first + Option-A unilateral MW revocation across all connected peers.
        let mut p_idx = 0usize;
        while p_idx < self.engine.peers.len() {
            // Cancel any queued outbound binds for `vframe`.
            let mut had_pending = false;
            for pg in &mut self.engine.peers[p_idx].pending_outbound_grants {
                if pg.frame == vframe {
                    pg.send_on_ready = false;
                    had_pending = true;
                }
            }
            if had_pending {
                self.ensure_replenish_tokens(4);
                let _ = self.engine.poll_peer_progress_multi(
                    p_idx,
                    host_tier,
                    None,
                    self.pd,
                    &mut self.replenish_tokens,
                    Tracked(self.caps.shares.borrow_mut()),
                    Tracked(self.caps.instance.borrow()),
                );
            }

            let tags: Vec<u64> = self.engine.peers[p_idx]
                .active_grants
                .iter()
                .filter(|g| g.frame == vframe)
                .map(|g| g.grant_tag)
                .collect();

            let mut disconnected = false;
            for tag in tags {
                let t0 = Instant::now();
                match self.engine.revoke_grant_cooperative_first(
                    p_idx,
                    tag,
                    self.coop_spin_budget,
                    Tracked(self.caps.instance.borrow()),
                ) {
                    Ok(None) => {
                        self.stats.mw_revocations_cooperative += 1;
                    }
                    Ok(Some(mw_slot)) => {
                        self.stats.mw_revocations_unilateral += 1;
                        if let Some(pd) = self.pd {
                            let _ = self
                                .engine
                                .acknowledge_peer_quarantined_window(p_idx, mw_slot);
                            let _ = self.engine.replenish_peer_window(
                                p_idx,
                                mw_slot,
                                pd,
                                self.caps.mint_rdma(),
                            );
                        }
                    }
                    Err(_) => {
                        self.disconnect_peer_by_engine_idx(p_idx);
                        disconnected = true;
                        break;
                    }
                }
                let elapsed_us = t0.elapsed().as_micros() as u64;
                self.stats.mw_revocation_latency_us_sum += elapsed_us;
                self.stats.mw_revocation_latency_us_max =
                    self.stats.mw_revocation_latency_us_max.max(elapsed_us);
            }

            if !disconnected {
                if self.engine.peers[p_idx].channel.is_closed() {
                    self.disconnect_peer_by_engine_idx(p_idx);
                } else {
                    p_idx += 1;
                }
            }
        }

        let _ = self.engine.evict_frame(vframe);
        let _ = self.engine.reclaim_draining(host_tier, 16);
        let reclaimed = self.engine.core.frames.metas[vframe.0 as usize].state == STATE_EMPTY;
        self.evicted_dram_keys.push_back(vkey);
        if reclaimed {
            self.stats.dram_evictions += 1;
        }
        reclaimed
    }

    fn ensure_nvme_space(&mut self) -> Result<(), ()> {
        let Some(nvme_tier) = self.engine.nvme_tier() else {
            return Err(());
        };
        let _ = self.engine.reclaim_draining(nvme_tier, 16);
        if !self.engine.core.frames.ready_queue[nvme_tier.as_index()].is_empty() {
            return Ok(());
        }
        let max_attempts = self.nvme_capacity.saturating_mul(2) + 16;
        for _ in 0..max_attempts {
            if !self.engine.core.frames.ready_queue[nvme_tier.as_index()].is_empty() {
                return Ok(());
            }
            let Some((_vk, vf)) = self.nvme_lru.pop_lru() else {
                let _ = self.engine.reclaim_draining(nvme_tier, 16);
                return if self.engine.core.frames.ready_queue[nvme_tier.as_index()].is_empty() {
                    Err(())
                } else {
                    Ok(())
                };
            };
            if !self.frame_is_published_on(vf, nvme_tier) {
                continue;
            }
            let _ = self.engine.evict_frame(vf);
            let _ = self.engine.reclaim_draining(nvme_tier, 16);
            self.stats.nvme_evictions += 1;
        }
        Err(())
    }

    fn ensure_dram_space(&mut self) -> Result<(), ()> {
        let host_tier = self.engine.host_tier();
        let _ = self.engine.reclaim_draining(host_tier, 16);
        if !self.engine.core.frames.ready_queue[host_tier.as_index()].is_empty()
            && self.dram_lru.len() < self.dram_budget_frames
        {
            return Ok(());
        }
        let max_attempts = self.host_capacity.saturating_mul(2) + 16;
        let mut skipped: Vec<(ObjectKey, FrameId)> = Vec::new();
        for _ in 0..max_attempts {
            if !self.engine.core.frames.ready_queue[host_tier.as_index()].is_empty()
                && self.dram_lru.len() < self.dram_budget_frames
            {
                for (k, f) in skipped {
                    self.dram_lru.insert(k, f);
                }
                return Ok(());
            }
            let Some((vkey, vframe)) = self.dram_lru.pop_lru() else {
                self.poll_peers_safe();
                let _ = self.engine.reclaim_draining(host_tier, 16);
                break;
            };
            if !self.frame_is_published_on(vframe, host_tier) {
                continue;
            }
            // Skip frames held by an active local read lease; CacheCore enforces this too.
            if self.engine.core.frames.metas[vframe.0 as usize].active_local_reads > 0 {
                skipped.push((vkey, vframe));
                continue;
            }
            let _ = self.revoke_and_evict_dram_frame(vkey, vframe);
        }
        for (k, f) in skipped {
            self.dram_lru.insert(k, f);
        }
        if !self.engine.core.frames.ready_queue[host_tier.as_index()].is_empty() {
            Ok(())
        } else {
            Err(())
        }
    }

    fn put_one(&mut self, key: ObjectKey, data: &[u8]) -> i32 {
        if data.len() != self.frame_size {
            return -1;
        }
        let host_tier = self.engine.host_tier();
        if let Some((frame, _)) = self.engine.lookup_tier(key, host_tier) {
            self.dram_lru.insert(key, frame);
            return 0;
        }
        if self.ensure_dram_space().is_err() {
            return -4;
        }
        match self
            .engine
            .put_host(Tracked(self.caps.instance.borrow()), key, data)
        {
            Ok((_ver, frame)) => {
                self.dram_lru.insert(key, frame);
                self.stats.puts += 1;
                self.stats.bytes_written += data.len() as u64;
                0
            }
            Err(_) => -2,
        }
    }

    fn fetch_from_peer_into_dram(&mut self, key: ObjectKey) -> Option<FrameId> {
        if !self.peer_fetch_enabled.load(Ordering::Relaxed) || self.engine.peers.is_empty() {
            return None;
        }
        let host_tier = self.engine.host_tier();
        let mut p_idx = 0usize;
        while p_idx < self.engine.peers.len() {
            if self.ensure_dram_space().is_err() {
                return None;
            }
            self.ensure_replenish_tokens(4);
            let res = self.engine.fetch_from_peer_duplex_multi(
                p_idx,
                host_tier,
                None,
                self.pd,
                &mut self.replenish_tokens,
                key,
                host_tier,
                PullTarget::Host,
                Tracked(self.caps.shares.borrow_mut()),
                Tracked(self.caps.instance.borrow()),
            );
            let closed = self.engine.peers[p_idx].channel.is_closed();
            match res {
                Ok(FetchOutcome::Hit { frame, .. }) => {
                    self.dram_lru.insert(key, frame);
                    self.stats.remote_hits += 1;
                    self.stats.rdma_fetches_ok += 1;
                    self.stats.rdma_fetch_bytes += self.frame_size as u64;
                    if closed {
                        self.disconnect_peer_by_engine_idx(p_idx);
                    }
                    return Some(frame);
                }
                Ok(_) => {
                    self.stats.rdma_fetches_miss += 1;
                    if closed {
                        self.disconnect_peer_by_engine_idx(p_idx);
                    } else {
                        p_idx += 1;
                    }
                }
                Err(_) => {
                    self.stats.rdma_fetches_err += 1;
                    self.disconnect_peer_by_engine_idx(p_idx);
                }
            }
        }
        None
    }

    fn ensure_key_in_dram(&mut self, key: ObjectKey, allow_remote: bool) -> Option<i32> {
        let host_tier = self.engine.host_tier();
        self.stats.gets += 1;

        // 1. Local DRAM
        if let Some((frame, _)) = self.engine.lookup_tier(key, host_tier) {
            self.dram_lru.touch(&key);
            let _ = frame;
            self.stats.hits += 1;
            self.stats.dram_hits += 1;
            return Some(LOC_LOCAL_DRAM);
        }

        // 2. Local NVMe
        if let Some(nvme_tier) = self.engine.nvme_tier() {
            if self.engine.lookup_tier(key, nvme_tier).is_some() {
                if self.ensure_dram_space().is_ok() {
                    if let Ok(dram_frame) = self.engine.warm_host_from_nvme_sync(
                        Tracked(self.caps.instance.borrow()),
                        Tracked(self.caps.shares.borrow_mut()),
                        key,
                    ) {
                        self.nvme_lru.touch(&key);
                        self.dram_lru.insert(key, dram_frame);
                        self.stats.hits += 1;
                        self.stats.nvme_hits += 1;
                        self.stats.nvme_read_bytes += self.frame_size as u64;
                        return Some(LOC_LOCAL_NVME);
                    }
                }
            }
        }

        // 3. Remote Peer over Type-2B MW RDMA (only when allowed by directory & peer_fetch=on)
        if allow_remote {
            if self.fetch_from_peer_into_dram(key).is_some() {
                self.stats.hits += 1;
                return Some(LOC_REMOTE_RDMA_HOST);
            }
        }

        self.stats.misses += 1;
        None
    }

    fn acquire_lease(&mut self, key: ObjectKey, allow_remote: bool) -> DistLeaseResult {
        let Some(location) = self.ensure_key_in_dram(key, allow_remote) else {
            return DistLeaseResult {
                status: 1,
                location: LOC_MISS,
                ..Default::default()
            };
        };
        match self.engine.get_lease_host(
            Tracked(self.caps.instance.borrow()),
            Tracked(self.caps.shares.borrow_mut()),
            key,
        ) {
            Ok(Some(lease)) => {
                let addr = lease.slice.addr();
                let len = lease.slice.len();
                let id = self.next_lease_id();
                self.leases.insert(
                    id,
                    ActiveLocalLease {
                        handle: lease.handle,
                        cap: lease.cap,
                    },
                );
                self.stats.bytes_read += len as u64;
                self.stats.active_leases = self.leases.len() as u64;
                DistLeaseResult {
                    lease_id: id,
                    address: addr,
                    len,
                    location,
                    status: 0,
                }
            }
            Ok(None) => DistLeaseResult {
                status: 1,
                location: LOC_MISS,
                ..Default::default()
            },
            Err(CoreError::CapacityExhausted | CoreError::WouldConflict) => DistLeaseResult {
                status: -4,
                location: LOC_MISS,
                ..Default::default()
            },
            Err(_) => DistLeaseResult {
                status: -2,
                location: LOC_MISS,
                ..Default::default()
            },
        }
    }

    fn release_leases(&mut self, lease_ids: &[u64]) {
        for id in lease_ids {
            if let Some(active) = self.leases.remove(id) {
                let _ = self.engine.release_lease(
                    Tracked(self.caps.instance.borrow()),
                    active.handle,
                    active.cap,
                    Ticket::NONE,
                );
            }
        }
        let host_tier = self.engine.host_tier();
        let _ = self.engine.reclaim_draining(host_tier, 32);
        self.stats.active_leases = self.leases.len() as u64;
    }

    fn get_into_buf(&mut self, key: ObjectKey, dst: &mut [u8], allow_remote: bool) -> (usize, i32, i32) {
        if dst.len() < self.frame_size {
            return (self.frame_size, -3, LOC_MISS);
        }
        let Some(location) = self.ensure_key_in_dram(key, allow_remote) else {
            return (0, 1, LOC_MISS);
        };
        let host_tier = self.engine.host_tier();
        let Some((frame, _)) = self.engine.lookup_tier(key, host_tier) else {
            return (0, 1, LOC_MISS);
        };
        match self.engine.read_host_frame(
            Tracked(self.caps.instance.borrow()),
            Tracked(self.caps.shares.borrow_mut()),
            frame,
            &mut dst[..self.frame_size],
        ) {
            Ok(_) => {
                self.stats.bytes_read += self.frame_size as u64;
                (self.frame_size, 0, location)
            }
            Err(_) => (0, -2, LOC_MISS),
        }
    }

    fn shrink_dram_budget(&mut self, new_budget: usize) -> usize {
        self.dram_budget_frames = new_budget.clamp(1, self.host_capacity);
        let mut evicted = 0usize;
        let host_tier = self.engine.host_tier();
        let mut skipped: Vec<(ObjectKey, FrameId)> = Vec::new();
        while self.dram_lru.len() > self.dram_budget_frames {
            let Some((vkey, vframe)) = self.dram_lru.pop_lru() else {
                break;
            };
            if !self.frame_is_published_on(vframe, host_tier) {
                continue;
            }
            if self.engine.core.frames.metas[vframe.0 as usize].active_local_reads > 0 {
                skipped.push((vkey, vframe));
                continue;
            }
            if self.revoke_and_evict_dram_frame(vkey, vframe) {
                evicted += 1;
            }
        }
        for (k, f) in skipped {
            self.dram_lru.insert(k, f);
        }
        evicted
    }

    fn evict_specific_dram_keys(&mut self, keys: &[ObjectKey]) -> usize {
        let host_tier = self.engine.host_tier();
        let mut count = 0usize;
        for &key in keys {
            if let Some((frame, _)) = self.engine.lookup_tier(key, host_tier) {
                self.dram_lru.remove(&key);
                if self.revoke_and_evict_dram_frame(key, frame) {
                    count += 1;
                }
            }
        }
        count
    }

    fn shutdown(&mut self) {
        // Release any remaining local leases before shutdown.
        let ids: Vec<u64> = self.leases.keys().copied().collect();
        self.release_leases(&ids);
        while !self.engine.peers.is_empty() {
            self.disconnect_peer_by_engine_idx(0);
        }
    }
}

fn shard_worker_main(
    shard_idx: usize,
    cfg: DistEngineConfig,
    peer_fetch_enabled: Arc<AtomicBool>,
    peers_connected_total: Arc<AtomicU64>,
    rx: mpsc::Receiver<ShardCmd>,
    ready_tx: Sender<Result<bool, String>>,
) {
    let mut caps = match DistShardCaps::take(cfg.node_id) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let mut core = match caps.make_core(cfg.frame_size) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let host = match HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        cfg.host_frames_per_shard,
        4096,
    ) {
        Ok(h) => h,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("HostBackend::new failed: {e:?}")));
            return;
        }
    };

    let nvme = match (&cfg.nvme_path, cfg.nvme_frames_per_shard) {
        (Some(base), n) if n > 0 => {
            let shard_dir = PathBuf::from(format!("{base}.shard-{shard_idx}"));
            if let Err(e) = std::fs::create_dir_all(&shard_dir) {
                let _ = ready_tx.send(Err(format!(
                    "create_dir_all({}) failed: {e}",
                    shard_dir.display()
                )));
                return;
            }
            let dir_str = shard_dir.to_string_lossy().into_owned();
            match NvmeBackend::new(
                &mut core,
                &host.ctx,
                Tracked(caps.instance.borrow()),
                Tracked(caps.alloc.borrow_mut()),
                caps.mint_nvme(),
                &dir_str,
                n,
                64,
            ) {
                Ok(nb) => Some(nb),
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("NvmeBackend::new failed: {e:?}")));
                    return;
                }
            }
        }
        _ => None,
    };

    let need_rdma = !cfg.peers.is_empty() || cfg.rdma_device.is_some();
    let rdma_dev = if need_rdma {
        match select_type2_rdma_device(&mut caps, cfg.rdma_device.as_deref()) {
            Ok(d) => Some(d),
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        }
    } else {
        None
    };

    let pd = match &rdma_dev {
        Some(dev) => match ProtectionDomain::new(dev, caps.mint_rdma()) {
            Ok(p) => Some(p),
            Err(e) => {
                let _ = ready_tx.send(Err(format!("ProtectionDomain::new failed: {e:?}")));
                return;
            }
        },
        None => None,
    };

    // Pin host DRAM slab for 1-hop CUDA H2D/D2H BEFORE ibv_reg_mr (same order as vericache::resources).
    // Hold SLAB_REG_LOCK continuously across both registrations to prevent mlx5 EAGAIN (errno 11) bursts.
    let (cuda_reg, shared_host_mr): (Option<CudaHostRegistration<'_>>, Option<RegisteredSlab<'_>>) = {
        let _reg_guard = SLAB_REG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cuda_reg = if cfg.prefer_cuda_pinned {
            cuda::host_register(&host.buffer).ok()
        } else {
            None
        };
        let shared_host_mr = match &pd {
            Some(pd_ref) => {
                let flags = AccessFlags {
                    local_write: true,
                    mw_bind: true,
                };
                let mut last_err = None;
                let mut registered = None;
                for attempt in 0..4 {
                    match RegisteredSlab::register_host(
                        &core,
                        pd_ref,
                        &host.buffer,
                        host.tier,
                        host.first_frame,
                        host.num_frames,
                        flags,
                    ) {
                        Ok(mr) => {
                            registered = Some(mr);
                            break;
                        }
                        Err(e) => {
                            let is_eagain = matches!(e, CoreError::Backend { code: 11, .. });
                            last_err = Some(e);
                            if is_eagain && attempt < 3 {
                                thread::sleep(Duration::from_millis(50 * (attempt + 1) as u64));
                            } else {
                                break;
                            }
                        }
                    }
                }
                match registered {
                    Some(mr) => Some(mr),
                    None => {
                        drop(cuda_reg);
                        let _ = ready_tx.send(Err(format!(
                            "RegisteredSlab::register_host failed (shard {}): {:?}",
                            shard_idx, last_err
                        )));
                        return;
                    }
                }
            }
            None => None,
        };
        (cuda_reg, shared_host_mr)
    };
    let is_cuda_pinned = cuda_reg.is_some();

    // Bind TCP listeners for any listener peer specs BEFORE signaling ready.
    let mut peer_slots = Vec::with_capacity(cfg.peers.len());
    for p in &cfg.peers {
        let port = p.base_port.wrapping_add(shard_idx as u16);
        let addr = format!("{}:{}", p.tcp_host, port);
        let listener = if p.is_listener {
            match TcpListener::bind(&addr) {
                Ok(l) => {
                    if let Err(e) = l.set_nonblocking(true) {
                        let _ = ready_tx
                            .send(Err(format!("listener set_nonblocking({addr}) failed: {e}")));
                        return;
                    }
                    Some(l)
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("bind peer listener {addr} failed: {e}")));
                    return;
                }
            }
        } else {
            None
        };
        peer_slots.push(PeerSlot {
            spec: p.clone(),
            addr,
            listener,
            connected: false,
            ever_connected: false,
            last_attempt: Instant::now() - Duration::from_secs(10),
            connect_epoch: 0,
        });
    }

    let eng_ctx = HostContext::new(caps.mint_host());
    let engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        CudaAttachment::None,
        nvme,
        Vec::new(),
    );

    let mut rt = ShardRuntime {
        shard_idx,
        frame_size: cfg.frame_size,
        host_capacity: cfg.host_frames_per_shard as usize,
        dram_budget_frames: cfg.host_frames_per_shard as usize,
        nvme_capacity: cfg.nvme_frames_per_shard as usize,
        rdma_gid_index: cfg.rdma_gid_index,
        rdma_windows_per_peer: cfg.rdma_windows_per_peer.max(4),
        coop_spin_budget: DEFAULT_COOP_SPIN_BUDGET,
        peer_fetch_enabled,
        peers_connected_total,
        caps,
        rdma_dev: rdma_dev.as_ref(),
        pd: pd.as_ref(),
        shared_host_mr: shared_host_mr.as_ref(),
        engine,
        peer_slots,
        replenish_tokens: Vec::new(),
        dram_lru: LruList::new(),
        nvme_lru: LruList::new(),
        leases: HashMap::new(),
        lease_seq: 0,
        evicted_dram_keys: VecDeque::new(),
        stats: DistStats::default(),
    };

    let _ = ready_tx.send(Ok(is_cuda_pinned));

    // Kick initial non-blocking peer connection immediately.
    rt.try_connect_peers();

    let idle_timeout = if rt.peer_slots.is_empty() {
        Duration::from_micros(LOCAL_ONLY_IDLE_POLL_US)
    } else {
        Duration::from_micros(IDLE_POLL_US)
    };
    loop {
        match rx.recv_timeout(idle_timeout) {
            Ok(cmd) => match cmd {
                ShardCmd::PutBatch { items, reply } => {
                    let mut statuses = Vec::with_capacity(items.len());
                    for (idx, (key, ptr, len)) in items.into_iter().enumerate() {
                        let data = if len == 0 {
                            &[][..]
                        } else {
                            unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }
                        };
                        statuses.push(rt.put_one(key, data));
                        if (idx & 15) == 15 {
                            rt.poll_peers_safe();
                        }
                    }
                    rt.poll_peers_safe();
                    let _ = reply.send(statuses);
                }
                ShardCmd::ContainsBatch {
                    keys,
                    dram_only,
                    reply,
                } => {
                    let host_tier = rt.engine.host_tier();
                    let nvme_tier = if dram_only {
                        None
                    } else {
                        rt.engine.nvme_tier()
                    };
                    let mut out = Vec::with_capacity(keys.len());
                    for key in keys {
                        let present = rt.engine.lookup_tier(key, host_tier).is_some()
                            || nvme_tier
                                .is_some_and(|nt| rt.engine.lookup_tier(key, nt).is_some());
                        out.push(u8::from(present));
                    }
                    let _ = reply.send(out);
                }
                ShardCmd::AcquireLeaseBatch { items, reply } => {
                    rt.poll_peers_safe();
                    let mut out = Vec::with_capacity(items.len());
                    for (key, allow_remote) in items {
                        out.push(rt.acquire_lease(key, allow_remote));
                    }
                    let _ = reply.send(out);
                }
                ShardCmd::ReleaseLeases { lease_ids, reply } => {
                    rt.release_leases(&lease_ids);
                    let _ = reply.send(());
                }
                ShardCmd::GetBatch { items, reply } => {
                    rt.poll_peers_safe();
                    let mut out = Vec::with_capacity(items.len());
                    for (key, dst_ptr, cap, allow_remote) in items {
                        let dst = if cap == 0 {
                            &mut [][..]
                        } else {
                            unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, cap) }
                        };
                        out.push(rt.get_into_buf(key, dst, allow_remote));
                    }
                    let _ = reply.send(out);
                }
                ShardCmd::DrainEvictions { max_keys, reply } => {
                    let take = max_keys.min(rt.evicted_dram_keys.len());
                    let mut out = Vec::with_capacity(take);
                    for _ in 0..take {
                        if let Some(k) = rt.evicted_dram_keys.pop_front() {
                            out.push(k);
                        }
                    }
                    let _ = reply.send(out);
                }
                ShardCmd::SetDramBudgetFrames { max_frames, reply } => {
                    let evicted = rt.shrink_dram_budget(max_frames);
                    let _ = reply.send(evicted);
                }
                ShardCmd::SetCoopRevokeSpinBudget { spins, reply } => {
                    rt.coop_spin_budget = spins;
                    let _ = reply.send(());
                }
                ShardCmd::EvictLocalDramKeys { keys, reply } => {
                    let n = rt.evict_specific_dram_keys(&keys);
                    let _ = reply.send(n);
                }
                ShardCmd::Poll { reply } => {
                    rt.try_connect_peers();
                    rt.poll_peers_safe();
                    let _ = reply.send(());
                }
                ShardCmd::Stats { reply } => {
                    let mut s = rt.stats;
                    s.dram_resident_keys = rt.dram_lru.len() as u64;
                    s.active_leases = rt.leases.len() as u64;
                    s.peers_connected = rt.engine.peers.len() as u64;
                    let _ = reply.send(s);
                }
                ShardCmd::Shutdown { reply } => {
                    rt.shutdown();
                    let _ = reply.send(());
                    break;
                }
            },
            Err(RecvTimeoutError::Timeout) => {
                rt.try_connect_peers();
                rt.poll_peers_safe();
            }
            Err(RecvTimeoutError::Disconnected) => {
                rt.shutdown();
                break;
            }
        }
    }

    drop(rt);
    drop(shared_host_mr);
    drop(pd);
    if let Some(dev) = rdma_dev {
        let _ = dev.close();
    }
    drop(cuda_reg);
    drop(host);
}

} // verus!

impl DistributedVllmEngine {
    pub fn new(cfg: DistEngineConfig) -> Result<Self, String> {
        if cfg.num_shards == 0
            || cfg.num_shards > 64
            || cfg.host_frames_per_shard == 0
            || cfg.frame_size == 0
        {
            return Err("invalid distributed engine parameters".into());
        }
        backend_mode::claim_distributed_vllm().map_err(|e| e.to_string())?;

        let peer_fetch_enabled = Arc::new(AtomicBool::new(cfg.peer_fetch_enabled));
        let peers_connected_total = Arc::new(AtomicU64::new(0));
        let mut shards = Vec::with_capacity(cfg.num_shards);
        let mut all_pinned = cfg.prefer_cuda_pinned;

        let mut ready_rxs = Vec::with_capacity(cfg.num_shards);
        for s in 0..cfg.num_shards {
            let (tx, rx) = mpsc::channel();
            let (ready_tx, ready_rx) = mpsc::channel();
            let scfg = cfg.clone();
            let pf = Arc::clone(&peer_fetch_enabled);
            let pc = Arc::clone(&peers_connected_total);
            let worker = thread::Builder::new()
                .name(format!("vc-dist-n{}-s{s}", cfg.node_id))
                .stack_size(16 * 1024 * 1024)
                .spawn(move || shard_worker_main(s, scfg, pf, pc, rx, ready_tx))
                .map_err(|e| format!("failed to spawn distributed shard thread {s}: {e}"))?;
            shards.push(ShardHandle {
                tx: Mutex::new(tx),
                worker: Some(worker),
            });
            ready_rxs.push(ready_rx);
        }

        let mut init_err = None;
        for rx in ready_rxs {
            match rx.recv() {
                Ok(Ok(pinned)) => all_pinned &= pinned,
                Ok(Err(e)) => {
                    if init_err.is_none() {
                        init_err = Some(e);
                    }
                }
                Err(_) => {
                    if init_err.is_none() {
                        init_err = Some("distributed shard exited before ready".into());
                    }
                }
            }
        }
        if let Some(err) = init_err {
            for sh in &mut shards {
                let (rtx, rrx) = mpsc::channel();
                let _ = sh.send(ShardCmd::Shutdown { reply: rtx });
                let _ = rrx.recv();
                if let Some(w) = sh.worker.take() {
                    let _ = w.join();
                }
            }
            return Err(err);
        }

        let has_nvme = cfg.nvme_path.is_some() && cfg.nvme_frames_per_shard > 0;
        Ok(Self {
            shards,
            node_id: cfg.node_id,
            frame_size: cfg.frame_size,
            has_nvme,
            cuda_pinned: all_pinned,
            peer_fetch_enabled,
            peers_connected_total,
        })
    }

    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn has_nvme(&self) -> bool {
        self.has_nvme
    }

    pub fn is_cuda_pinned(&self) -> bool {
        self.cuda_pinned
    }

    pub fn set_peer_fetch_enabled(&self, enabled: bool) {
        self.peer_fetch_enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn peer_fetch_enabled(&self) -> bool {
        self.peer_fetch_enabled.load(Ordering::Relaxed)
    }

    pub fn peers_connected(&self) -> u64 {
        self.peers_connected_total.load(Ordering::Relaxed)
    }

    pub fn put_batch(&self, items: &[(ObjectKey, usize, usize)]) -> Vec<i32> {
        let n = self.shards.len();
        let mut by_shard: Vec<Vec<(usize, ObjectKey, usize, usize)>> = vec![Vec::new(); n];
        for (orig_idx, &(key, ptr, len)) in items.iter().enumerate() {
            by_shard[shard_for_key(key, n)].push((orig_idx, key, ptr, len));
        }
        let mut pending = Vec::new();
        for (s, bucket) in by_shard.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let orig_idxs: Vec<usize> = bucket.iter().map(|x| x.0).collect();
            let payload: Vec<(ObjectKey, usize, usize)> =
                bucket.into_iter().map(|x| (x.1, x.2, x.3)).collect();
            let (rtx, rrx) = mpsc::channel();
            if self.shards[s]
                .send(ShardCmd::PutBatch {
                    items: payload,
                    reply: rtx,
                })
            {
                pending.push((orig_idxs, rrx));
            }
        }
        let mut out = vec![-2; items.len()];
        for (orig_idxs, rrx) in pending {
            if let Ok(statuses) = rrx.recv() {
                for (idx, st) in orig_idxs.into_iter().zip(statuses) {
                    out[idx] = st;
                }
            }
        }
        out
    }

    pub fn contains_batch(&self, keys: &[ObjectKey], dram_only: bool) -> Vec<u8> {
        let n = self.shards.len();
        let mut by_shard: Vec<Vec<(usize, ObjectKey)>> = vec![Vec::new(); n];
        for (orig_idx, &key) in keys.iter().enumerate() {
            by_shard[shard_for_key(key, n)].push((orig_idx, key));
        }
        let mut pending = Vec::new();
        for (s, bucket) in by_shard.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let orig_idxs: Vec<usize> = bucket.iter().map(|x| x.0).collect();
            let skeys: Vec<ObjectKey> = bucket.into_iter().map(|x| x.1).collect();
            let (rtx, rrx) = mpsc::channel();
            if self.shards[s]
                .send(ShardCmd::ContainsBatch {
                    keys: skeys,
                    dram_only,
                    reply: rtx,
                })
            {
                pending.push((orig_idxs, rrx));
            }
        }
        let mut out = vec![0u8; keys.len()];
        for (orig_idxs, rrx) in pending {
            if let Ok(flags) = rrx.recv() {
                for (idx, f) in orig_idxs.into_iter().zip(flags) {
                    out[idx] = f;
                }
            }
        }
        out
    }

    pub fn acquire_lease_batch(&self, items: &[(ObjectKey, bool)]) -> Vec<DistLeaseResult> {
        let n = self.shards.len();
        let mut by_shard: Vec<Vec<(usize, ObjectKey, bool)>> = vec![Vec::new(); n];
        for (orig_idx, &(key, allow_remote)) in items.iter().enumerate() {
            by_shard[shard_for_key(key, n)].push((orig_idx, key, allow_remote));
        }
        let mut pending = Vec::new();
        for (s, bucket) in by_shard.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let orig_idxs: Vec<usize> = bucket.iter().map(|x| x.0).collect();
            let payload: Vec<(ObjectKey, bool)> = bucket.into_iter().map(|x| (x.1, x.2)).collect();
            let (rtx, rrx) = mpsc::channel();
            if self.shards[s]
                .send(ShardCmd::AcquireLeaseBatch {
                    items: payload,
                    reply: rtx,
                })
            {
                pending.push((orig_idxs, rrx));
            }
        }
        let mut out = vec![
            DistLeaseResult {
                status: -2,
                ..Default::default()
            };
            items.len()
        ];
        for (orig_idxs, rrx) in pending {
            if let Ok(results) = rrx.recv() {
                for (idx, r) in orig_idxs.into_iter().zip(results) {
                    out[idx] = r;
                }
            }
        }
        out
    }

    pub fn release_leases(&self, lease_ids: &[u64]) {
        let n = self.shards.len();
        let mut by_shard: Vec<Vec<u64>> = vec![Vec::new(); n];
        for &id in lease_ids {
            if id == 0 {
                continue;
            }
            let s = (id >> 48) as usize;
            if s < n {
                by_shard[s].push(id);
            }
        }
        let mut pending = Vec::new();
        for (s, ids) in by_shard.into_iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let (rtx, rrx) = mpsc::channel();
            if self.shards[s]
                .send(ShardCmd::ReleaseLeases {
                    lease_ids: ids,
                    reply: rtx,
                })
            {
                pending.push(rrx);
            }
        }
        for rrx in pending {
            let _ = rrx.recv();
        }
    }

    pub fn get_batch(&self, items: &[(ObjectKey, usize, usize, bool)]) -> Vec<(usize, i32, i32)> {
        let n = self.shards.len();
        let mut by_shard: Vec<Vec<(usize, ObjectKey, usize, usize, bool)>> = vec![Vec::new(); n];
        for (orig_idx, &(key, dst_ptr, cap, allow_remote)) in items.iter().enumerate() {
            by_shard[shard_for_key(key, n)].push((orig_idx, key, dst_ptr, cap, allow_remote));
        }
        let mut pending = Vec::new();
        for (s, bucket) in by_shard.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let orig_idxs: Vec<usize> = bucket.iter().map(|x| x.0).collect();
            let payload: Vec<(ObjectKey, usize, usize, bool)> =
                bucket.into_iter().map(|x| (x.1, x.2, x.3, x.4)).collect();
            let (rtx, rrx) = mpsc::channel();
            if self.shards[s]
                .send(ShardCmd::GetBatch {
                    items: payload,
                    reply: rtx,
                })
            {
                pending.push((orig_idxs, rrx));
            }
        }
        let mut out = vec![(0, -2, LOC_MISS); items.len()];
        for (orig_idxs, rrx) in pending {
            if let Ok(results) = rrx.recv() {
                for (idx, r) in orig_idxs.into_iter().zip(results) {
                    out[idx] = r;
                }
            }
        }
        out
    }

    pub fn drain_evictions(&self, max_keys: usize) -> Vec<ObjectKey> {
        let mut out = Vec::new();
        let mut rem = max_keys;
        for sh in &self.shards {
            if rem == 0 {
                break;
            }
            let (rtx, rrx) = mpsc::channel();
            if sh
                .send(ShardCmd::DrainEvictions {
                    max_keys: rem,
                    reply: rtx,
                })
            {
                if let Ok(keys) = rrx.recv() {
                    rem = rem.saturating_sub(keys.len());
                    out.extend(keys);
                }
            }
        }
        out
    }

    pub fn set_dram_budget_frames(&self, max_frames_per_shard: usize) -> usize {
        let mut pending = Vec::with_capacity(self.shards.len());
        for sh in &self.shards {
            let (rtx, rrx) = mpsc::channel();
            if sh
                .send(ShardCmd::SetDramBudgetFrames {
                    max_frames: max_frames_per_shard,
                    reply: rtx,
                })
            {
                pending.push(rrx);
            }
        }
        let mut evicted = 0usize;
        for rrx in pending {
            if let Ok(n) = rrx.recv() {
                evicted += n;
            }
        }
        evicted
    }

    pub fn set_coop_revoke_spin_budget(&self, spins: u32) {
        let mut pending = Vec::with_capacity(self.shards.len());
        for sh in &self.shards {
            let (rtx, rrx) = mpsc::channel();
            if sh
                .send(ShardCmd::SetCoopRevokeSpinBudget { spins, reply: rtx })
            {
                pending.push(rrx);
            }
        }
        for rrx in pending {
            let _ = rrx.recv();
        }
    }

    pub fn evict_local_dram_keys(&self, keys: &[ObjectKey]) -> usize {
        let n = self.shards.len();
        let mut by_shard: Vec<Vec<ObjectKey>> = vec![Vec::new(); n];
        for &key in keys {
            by_shard[shard_for_key(key, n)].push(key);
        }
        let mut pending = Vec::new();
        for (s, skeys) in by_shard.into_iter().enumerate() {
            if skeys.is_empty() {
                continue;
            }
            let (rtx, rrx) = mpsc::channel();
            if self.shards[s]
                .send(ShardCmd::EvictLocalDramKeys {
                    keys: skeys,
                    reply: rtx,
                })
            {
                pending.push(rrx);
            }
        }
        let mut evicted = 0usize;
        for rrx in pending {
            if let Ok(k) = rrx.recv() {
                evicted += k;
            }
        }
        evicted
    }

    pub fn poll(&self) {
        let mut pending = Vec::with_capacity(self.shards.len());
        for sh in &self.shards {
            let (rtx, rrx) = mpsc::channel();
            if sh.send(ShardCmd::Poll { reply: rtx }) {
                pending.push(rrx);
            }
        }
        for rrx in pending {
            let _ = rrx.recv();
        }
    }

    pub fn stats(&self) -> DistStats {
        let mut pending = Vec::with_capacity(self.shards.len());
        for sh in &self.shards {
            let (rtx, rrx) = mpsc::channel();
            if sh.send(ShardCmd::Stats { reply: rtx }) {
                pending.push(rrx);
            }
        }
        let mut total = DistStats::default();
        for rrx in pending {
            if let Ok(s) = rrx.recv() {
                total.accumulate(&s);
            }
        }
        total
    }
}

impl Drop for DistributedVllmEngine {
    fn drop(&mut self) {
        let mut pending = Vec::with_capacity(self.shards.len());
        for sh in &self.shards {
            let (rtx, rrx) = mpsc::channel();
            if sh.send(ShardCmd::Shutdown { reply: rtx }) {
                pending.push(rrx);
            }
        }
        for rrx in pending {
            let _ = rrx.recv();
        }
        for sh in &mut self.shards {
            if let Some(w) = sh.worker.take() {
                let _ = w.join();
            }
        }
    }
}
