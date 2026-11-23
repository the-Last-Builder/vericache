use crate::policy::LruPolicy;
use crate::schema::{
    JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps, read_proc_hwm_mb,
};
use crate::shard::{
    ShardCapabilityBundle, ensure_bench_node_allocator, key_from_hash_id, splitmix64,
};
use crate::suites::rdma_ablation::open_rdma_device;
use crate::suites::trace_replay::{TraceRequest, load_trace};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::*;
use transfer::cuda::{self, CudaStream};
use transfer::domain::DeviceId;
use transfer::host::{self, HostContext};
use transfer::rdma::{
    AccessFlags, CompletionQueue, DeviceContext, ProtectionDomain, UnconnectedQueuePair,
    alloc_window, connect_rc,
};
use transfer::transport::{ByteTransport, Channel};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::backends::rdma::peer::{PeerSession, ServeOutcome};
use vericache_core::backends::rdma::protocol::{ControlPayload, VeriCacheAttachment};
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::engine::{DistributedEngine, DistributedEvent};
use vericache_core::frame::STATE_EMPTY;
use vericache_core::kernel::CacheCore;
#[cfg(verus_only)]
use vericache_core::provenance::{ValueCertificate, ValueProof};
use vericache_core::spec::{CoreError, FrameId, ObjectKey, Ticket, TierId};
use vericache_core::timeline::TimelineKind;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Tracked, verus};

const PEER_DONE_TAG: u64 = 0xDEAD_BEEF_CAFE_0005;
const COORD_CHANNEL_ID: u8 = 255;
const SPILL_PIPELINE_QD: usize = 4;
const SPILL_HIGH_WATER: usize = 4;

struct PendingSpill {
    key: ObjectKey,
    dram_frame: FrameId,
    nvme_frame: FrameId,
    ticket: Ticket,
}

/// Deterministic 2-node 70/30 session-sticky routing + 8-shard affinity.
/// Exact bit-for-bit match with Python baselines (`experiments/baselines/common.py`).
#[inline]
fn route_request(
    req_idx: usize,
    hash_ids: &[u64],
    shards_per_node: usize,
    sticky_pct: u64,
) -> (u64, usize) {
    let pos = if hash_ids.len() > 12 {
        12
    } else {
        hash_ids.len() - 1
    };
    let sk = hash_ids[pos];
    let h = splitmix64(sk);
    let home_node = h & 1;
    let shard_id = ((h >> 1) as usize) % shards_per_node;
    let drift_hash = splitmix64((req_idx as u64) ^ sk.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let node_id = if (drift_hash % 100) < sticky_pct {
        home_node
    } else {
        1 - home_node
    };
    (node_id, shard_id)
}

/// TCP ByteTransport with non-blocking peek helper for deadlock-free bidirectional peer serving.
struct ScaleupTcpTransport(TcpStream);

impl ScaleupTcpTransport {
    fn new(stream: TcpStream) -> Self {
        stream.set_nodelay(true).expect("set_nodelay");
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .expect("set_read_timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(300)))
            .expect("set_write_timeout");
        Self(stream)
    }

    fn clone_for_peek(&self) -> TcpStream {
        self.0.try_clone().expect("try_clone tcp stream for peek")
    }
}

/// Non-blocking check whether at least 1 byte is queued on the TCP peer socket.
/// Safe on a single shard thread because `O_NONBLOCK` is always restored to `false` before returning.
fn tcp_has_pending_data(sock: &TcpStream) -> bool {
    let mut buf = [0u8; 1];
    let _ = sock.set_nonblocking(true);
    let ready = match sock.peek(&mut buf) {
        Ok(n) => n > 0,
        Err(_) => false,
    };
    let _ = sock.set_nonblocking(false);
    ready
}

impl ByteTransport for ScaleupTcpTransport {
    type Error = io::Error;

    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        self.0.write_all(&n.to_be_bytes())?;
        self.0.write_all(frame)
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        let mut prefix = [0u8; 4];
        self.0.read_exact(&mut prefix)?;
        let n = u32::from_be_bytes(prefix) as usize;
        if n > 32 * 1024 * 1024 {
            return Err(io::Error::other("frame exceeds limit"));
        }
        let mut frame = vec![0u8; n];
        self.0.read_exact(&mut frame)?;
        Ok(frame)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct ShardStats {
    reqs: u64,
    hbm_hits: u64,
    dram_hits: u64,
    nvme_hits: u64,
    remote_hits: u64,
    cold_misses: u64,
    init_time_s: f64,
    hist_buckets: Vec<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeReport {
    node_id: u64,
    init_time_s: f64,
    elapsed_s: f64,
    cpu_pct: f64,
    rss_mb: f64,
    vram_mb: f64,
    reqs: u64,
    hbm_hits: u64,
    dram_hits: u64,
    nvme_hits: u64,
    remote_hits: u64,
    cold_misses: u64,
    hist_buckets: Vec<(u64, u64)>,
}

struct ScaleupConfig {
    frame_size: usize,
    shards_per_node: usize,
    hbm_frames_per_shard: usize,
    dram_frames_per_shard: usize,
    nvme_frames_per_shard: usize,
    hbm_gb_per_node: f64,
    dram_gb_per_node: f64,
    nvme_gb_per_node: f64,
    sticky_pct: u64,
}

impl ScaleupConfig {
    fn from_job(job: &JobSpec) -> Self {
        let frame_size = job.chunk_bytes;
        let shards = job.shards_per_node.unwrap_or(8).max(1);
        let hbm_gb = job.hbm_gb_per_node.unwrap_or(50.0);
        let dram_gb = job.dram_gb_per_node.unwrap_or(200.0);
        let nvme_gb = job.nvme_gb_per_node.unwrap_or(50.0);
        let sticky_pct = ((job.sticky_ratio.unwrap_or(0.70) * 100.0).round() as u64).clamp(0, 100);

        let gb_to_frames = |gb: f64| -> usize {
            let total_bytes = (gb * (1024.0 * 1024.0 * 1024.0)).round() as u64;
            let total_frames = (total_bytes / (frame_size as u64)) as usize;
            (total_frames / shards).max(4)
        };

        Self {
            frame_size,
            shards_per_node: shards,
            hbm_frames_per_shard: gb_to_frames(hbm_gb),
            dram_frames_per_shard: gb_to_frames(dram_gb),
            nvme_frames_per_shard: gb_to_frames(nvme_gb),
            hbm_gb_per_node: hbm_gb,
            dram_gb_per_node: dram_gb,
            nvme_gb_per_node: nvme_gb,
            sticky_pct,
        }
    }
}

verus! {

#[allow(clippy::too_many_arguments)]
fn connect_shard_peer_rdma<'a>(
    caps: &mut ShardCapabilityBundle,
    shard_id: usize,
    gid_index: u8,
    core: &mut CacheCore,
    pd: &'a ProtectionDomain<'a>,
    device: &'a DeviceContext,
    transport: ScaleupTcpTransport,
    host: &'a HostBackend,
    cuda: &'a CudaBackend<'a>,
) -> PeerSession<'a, ScaleupTcpTransport> {
    let node_id = caps.node;
    let peer_id = 1 - node_id;
    let cq = CompletionQueue::new(device, 512, caps.mint_rdma()).expect("shard CQ");
    let psn = 0x1000 + ((node_id as u32) << 8) + (shard_id as u32);
    let reset = UnconnectedQueuePair::new(
        pd,
        &cq,
        128,
        1,
        gid_index,
        psn,
        caps.mint_rdma(),
    )
    .expect("shard UnconnectedQueuePair");
    let mut channel: Channel<ScaleupTcpTransport, VeriCacheAttachment> =
        Channel::new_with_protocol(
            transport,
            node_id,
            peer_id,
            Tracked(caps.instance.borrow()),
        );
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked(caps.instance.borrow()))
        .expect("exchange shard QP endpoint");
    let qp = connect_rc(reset, peer_ep, peer_id).expect("connect_rc shard QP");

    let mut windows = Vec::with_capacity(64);
    for _ in 0..64 {
        windows.push(alloc_window(pd, caps.mint_rdma()).expect("alloc shard Type-2B MW"));
    }
    let mut rdma = RdmaBackend::new(core, peer_id, qp, cq, windows)
        .map_err(|e| e.0)
        .expect("RdmaBackend::new");
    let flags = AccessFlags {
        local_write: true,
        mw_bind: true,
    };
    rdma.register_host_slab(
        core,
        pd,
        &host.buffer,
        host.tier,
        host.first_frame,
        host.num_frames,
        flags,
    )
    .expect("register_host_slab (25 GiB Pinned DRAM)");
    rdma.register_gpu_slab(
        core,
        pd,
        &cuda.gpu_buffer,
        cuda.gpu_tier,
        cuda.gpu_first_frame,
        cuda.gpu_num_frames,
        flags,
    )
    .expect("register_gpu_slab (6.25 GiB GPU HBM)");
    PeerSession::new(rdma, channel, core)
}

/// Helper: handle one incoming peer event (serving FetchRequest from HBM or DRAM,
/// handling cooperative ReadRelease, or noting PeerDone).
fn handle_peer_event(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    gpu_tier: TierId,
    host_tier: TierId,
    peer_done: &mut bool,
) -> Option<DistributedEvent> {
    let ev = engine
        .poll_peer_event(Tracked(caps.instance.borrow()))
        .expect("poll_peer_event");
    match ev {
        DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key }) => {
            // Serve from HBM first (GPU->GPU GDR), else DRAM (Host->GPU GDR), else FetchMiss.
            let found_tier = if engine.lookup_tier(key, gpu_tier).is_some() {
                Some(gpu_tier)
            } else if engine.lookup_tier(key, host_tier).is_some() {
                Some(host_tier)
            } else {
                None
            };
            match found_tier {
                Some(t) => {
                    let outcome = engine
                        .serve_fetch_request(
                            t,
                            request_id,
                            key,
                            Tracked(caps.shares.borrow_mut()),
                            Tracked(caps.instance.borrow()),
                        )
                        .expect("serve_fetch_request");
                    if let ServeOutcome::Miss = outcome {
                        // Already sent FetchMiss inside serve_fetch_request
                    }
                }
                None => {
                    engine
                        .send_control(
                            ControlPayload::FetchMiss { request_id },
                            Tracked(caps.instance.borrow()),
                        )
                        .expect("send FetchMiss");
                }
            }
            None
        }
        DistributedEvent::ReadReleased { .. } => None,
        DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag })
            if grant_tag == PEER_DONE_TAG =>
        {
            *peer_done = true;
            None
        }
        other => Some(other),
    }
}

/// Drain any queued peer messages without blocking.
fn drain_peer_nonblocking(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    peek_sock: &TcpStream,
    gpu_tier: TierId,
    host_tier: TierId,
    peer_done: &mut bool,
) {
    while tcp_has_pending_data(peek_sock) {
        let _ = handle_peer_event(caps, engine, gpu_tier, host_tier, peer_done);
    }
}

/// Evict and reclaim a frame on `tier`, handling frames already auto-superseded
/// into `STATE_DRAINING` (`Err(CoreError::StaleReplica)`), and only draining peer
/// events if this specific victim still holds an active remote lease.
fn evict_and_reclaim(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    tier: TierId,
    frame: FrameId,
    gpu_tier: TierId,
    host_tier: TierId,
    peer_done: &mut bool,
) {
    match engine.evict_frame(frame) {
        Ok(()) | Err(CoreError::StaleReplica) => {}
        Err(e) => panic!("evict_frame unexpected error: {e:?}"),
    }
    while engine.core.frames.metas[frame.0 as usize].state != STATE_EMPTY {
        let reclaimed = engine.reclaim_draining(tier, 8);
        if reclaimed == 0 && engine.core.frames.metas[frame.0 as usize].state != STATE_EMPTY {
            let _ = handle_peer_event(caps, engine, gpu_tier, host_tier, peer_done);
        }
    }
}

/// Ensure NVMe has at least 1 free frame in `ready_queue` (LRU evict if empty).
fn ensure_nvme_space(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    gpu_tier: TierId,
    host_tier: TierId,
    nvme_tier: TierId,
    peer_done: &mut bool,
) {
    let _ = engine.reclaim_draining(nvme_tier, 8);
    while engine.core.frames.ready_queue[nvme_tier.as_index()].is_empty() {
        if let Some((_nk, nf)) = nvme_lru.pop_lru() {
            evict_and_reclaim(caps, engine, nvme_tier, nf, gpu_tier, host_tier, peer_done);
        } else {
            let _ = handle_peer_event(caps, engine, gpu_tier, host_tier, peer_done);
            let _ = engine.reclaim_draining(nvme_tier, 8);
        }
    }
}

/// Non-blockingly poll NVMe io_uring completions and retire any completed async spills,
/// reclaiming their source DRAM frames back to `ready_queue` and inserting them into `nvme_lru`.
#[allow(clippy::too_many_arguments)]
fn reap_completed_spills(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    pending_spills: &mut Vec<PendingSpill>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    gpu_tier: TierId,
    host_tier: TierId,
    peer_done: &mut bool,
) -> usize {
    if pending_spills.is_empty() {
        return 0;
    }
    let _ = engine.poll_nvme_completions(Tracked(caps.instance.borrow()), 32);
    let mut retired = 0usize;
    let mut idx = 0usize;
    while idx < pending_spills.len() {
        if engine.core.timelines.is_retired(pending_spills[idx].ticket) {
            let sp = pending_spills.swap_remove(idx);
            evict_and_reclaim(
                caps,
                engine,
                host_tier,
                sp.dram_frame,
                gpu_tier,
                host_tier,
                peer_done,
            );
            let _ = nvme_lru.insert(sp.key, sp.nvme_frame);
            retired += 1;
        } else {
            idx += 1;
        }
    }
    retired
}

/// Wait until a specific `key` is no longer in `pending_spills` (rare hit on an in-flight victim).
#[allow(clippy::too_many_arguments)]
fn wait_pending_spill_for_key(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    peek_sock: &TcpStream,
    pending_spills: &mut Vec<PendingSpill>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    gpu_tier: TierId,
    host_tier: TierId,
    key: ObjectKey,
    peer_done: &mut bool,
) {
    while pending_spills.iter().any(|p| p.key == key) {
        drain_peer_nonblocking(caps, engine, peek_sock, gpu_tier, host_tier, peer_done);
        if reap_completed_spills(
            caps,
            engine,
            pending_spills,
            nvme_lru,
            gpu_tier,
            host_tier,
            peer_done,
        ) == 0
        {
            let _ = engine.wait_nvme_completion(Tracked(caps.instance.borrow()));
            let _ = reap_completed_spills(
                caps,
                engine,
                pending_spills,
                nvme_lru,
                gpu_tier,
                host_tier,
                peer_done,
            );
        }
    }
}

/// Drain all remaining in-flight async NVMe spills (called before peer shutdown).
#[allow(clippy::too_many_arguments)]
fn drain_all_pending_spills(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    peek_sock: &TcpStream,
    pending_spills: &mut Vec<PendingSpill>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    gpu_tier: TierId,
    host_tier: TierId,
    peer_done: &mut bool,
) {
    while !pending_spills.is_empty() {
        drain_peer_nonblocking(caps, engine, peek_sock, gpu_tier, host_tier, peer_done);
        if reap_completed_spills(
            caps,
            engine,
            pending_spills,
            nvme_lru,
            gpu_tier,
            host_tier,
            peer_done,
        ) == 0
        {
            let _ = engine.wait_nvme_completion(Tracked(caps.instance.borrow()));
            let _ = reap_completed_spills(
                caps,
                engine,
                pending_spills,
                nvme_lru,
                gpu_tier,
                host_tier,
                peer_done,
            );
        }
    }
}

/// Proactively submit async io_uring NVMe spills up to `SPILL_PIPELINE_QD` whenever DRAM
/// free frames dip to `SPILL_HIGH_WATER`, keeping spills off the request critical path.
#[allow(clippy::too_many_arguments)]
fn kick_async_spills(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    peek_sock: &TcpStream,
    dram_lru: &mut LruPolicy<ObjectKey, FrameId>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    pending_spills: &mut Vec<PendingSpill>,
    gpu_tier: TierId,
    host_tier: TierId,
    nvme_tier: TierId,
    peer_done: &mut bool,
) {
    let _ = reap_completed_spills(
        caps,
        engine,
        pending_spills,
        nvme_lru,
        gpu_tier,
        host_tier,
        peer_done,
    );
    while engine.core.frames.ready_queue[host_tier.as_index()].len() <= SPILL_HIGH_WATER
        && pending_spills.len() < SPILL_PIPELINE_QD
        && dram_lru.len() > 0
    {
        let Some((dk, df)) = dram_lru.pop_lru() else {
            break;
        };
        if pending_spills.iter().any(|p| p.key == dk) {
            wait_pending_spill_for_key(
                caps,
                engine,
                peek_sock,
                pending_spills,
                nvme_lru,
                gpu_tier,
                host_tier,
                dk,
                peer_done,
            );
        }
        if let Some(old_nf) = nvme_lru.remove(&dk) {
            evict_and_reclaim(caps, engine, nvme_tier, old_nf, gpu_tier, host_tier, peer_done);
        }
        ensure_nvme_space(caps, engine, nvme_lru, gpu_tier, host_tier, nvme_tier, peer_done);
        if engine.lookup_tier(dk, host_tier).is_some() {
            let (nf, ticket) = engine
                .spill_host_to_nvme(
                    Tracked(caps.instance.borrow()),
                    Tracked(caps.shares.borrow_mut()),
                    dk,
                )
                .expect("spill_host_to_nvme async");
            pending_spills.push(PendingSpill {
                key: dk,
                dram_frame: df,
                nvme_frame: nf,
                ticket,
            });
        } else {
            evict_and_reclaim(caps, engine, host_tier, df, gpu_tier, host_tier, peer_done);
        }
    }
}

/// Ensure Host DRAM has at least 1 free frame in `ready_queue`
/// (pipelined async io_uring O_DIRECT spills -> NVMe).
#[allow(clippy::too_many_arguments)]
fn ensure_dram_space(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    peek_sock: &TcpStream,
    dram_lru: &mut LruPolicy<ObjectKey, FrameId>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    pending_spills: &mut Vec<PendingSpill>,
    gpu_tier: TierId,
    host_tier: TierId,
    nvme_tier: TierId,
    peer_done: &mut bool,
) {
    let _ = engine.reclaim_draining(host_tier, 8);
    kick_async_spills(
        caps,
        engine,
        peek_sock,
        dram_lru,
        nvme_lru,
        pending_spills,
        gpu_tier,
        host_tier,
        nvme_tier,
        peer_done,
    );
    while engine.core.frames.ready_queue[host_tier.as_index()].is_empty() {
        drain_peer_nonblocking(caps, engine, peek_sock, gpu_tier, host_tier, peer_done);
        let _ = reap_completed_spills(
            caps,
            engine,
            pending_spills,
            nvme_lru,
            gpu_tier,
            host_tier,
            peer_done,
        );
        let _ = engine.reclaim_draining(host_tier, 8);
        if !engine.core.frames.ready_queue[host_tier.as_index()].is_empty() {
            break;
        }
        if !pending_spills.is_empty() {
            let _ = engine.wait_nvme_completion(Tracked(caps.instance.borrow()));
            let _ = reap_completed_spills(
                caps,
                engine,
                pending_spills,
                nvme_lru,
                gpu_tier,
                host_tier,
                peer_done,
            );
        } else if dram_lru.len() > 0 {
            kick_async_spills(
                caps,
                engine,
                peek_sock,
                dram_lru,
                nvme_lru,
                pending_spills,
                gpu_tier,
                host_tier,
                nvme_tier,
                peer_done,
            );
        } else {
            let _ = handle_peer_event(caps, engine, gpu_tier, host_tier, peer_done);
            let _ = engine.reclaim_draining(host_tier, 8);
        }
    }
}

/// Ensure GPU HBM has at least 1 free frame in `ready_queue`
/// (cascading LRU victim -> DRAM via D2H + put_host -> pipelined async NVMe).
#[allow(clippy::too_many_arguments)]
fn ensure_hbm_space(
    caps: &mut ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, ScaleupTcpTransport>,
    peek_sock: &TcpStream,
    hbm_lru: &mut LruPolicy<ObjectKey, FrameId>,
    dram_lru: &mut LruPolicy<ObjectKey, FrameId>,
    nvme_lru: &mut LruPolicy<ObjectKey, FrameId>,
    pending_spills: &mut Vec<PendingSpill>,
    gpu_tier: TierId,
    host_tier: TierId,
    nvme_tier: TierId,
    caller_stream: &mut CudaStream,
    caller_tl: u8,
    d2h_stage_buf: &cuda::CudaHostBuffer,
    Tracked(d2h_tok): Tracked<&mut SlotToken>,
    d2h_host_ctx: &mut HostContext,
    d2h_scratch: &mut [u8],
    peer_done: &mut bool,
) {
    let _ = engine.reclaim_draining(gpu_tier, 8);
    if !engine.core.frames.ready_queue[gpu_tier.as_index()].is_empty() {
        return;
    }
    let Some((vk, vf)) = hbm_lru.pop_lru() else {
        while engine.core.frames.ready_queue[gpu_tier.as_index()].is_empty() {
            let _ = handle_peer_event(caps, engine, gpu_tier, host_tier, peer_done);
            let _ = engine.reclaim_draining(gpu_tier, 8);
        }
        return;
    };
    // D2H readout of victim HBM frame into pinned staging buffer before reclaiming HBM frame.
    let d2h_cuda = d2h_stage_buf.as_cuda_slice();
    if let Some((_ver, t_d2h, _ev)) = engine
        .get_into_gpu(
            Tracked(caps.instance.borrow()),
            Tracked(caps.shares.borrow_mut()),
            vk,
            caller_stream,
            caller_tl,
            &d2h_cuda,
            Tracked(d2h_tok),
        )
        .expect("get_into_gpu D2H demote")
    {
        let k_d2h = cuda::stream_synchronize(caller_stream, Tracked(caps.instance.borrow()))
            .expect("sync D2H demote");
        engine
            .core
            .mark_timeline_complete(caller_tl, t_d2h.seq(), k_d2h);
        let d2h_host = d2h_stage_buf.as_host_slice();
        let tracked mut tmp_tok = SlotToken::empty(caps.instance.borrow().id());
        proof {
            vstd::modes::tracked_swap(d2h_tok, &mut tmp_tok);
        }
        let tracked mut d2h_perm = ReadPerm::Owned(tmp_tok);
        let Tracked(r_comp) = host::read_slice(
            d2h_host_ctx,
            Tracked(caps.instance.borrow()),
            &d2h_host,
            d2h_scratch,
            Tracked(&mut d2h_perm),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        let t_r = engine
            .core
            .issue_ticket(engine.host.timeline_id(), Tracked(r_comp.submitted))
            .expect("issue D2H host read ticket");
        engine.core.mark_timeline_complete(
            engine.host.timeline_id(),
            t_r.seq(),
            Tracked(r_comp.knows),
        );
        proof {
            *d2h_tok = d2h_perm.into_owned();
        }
    }
    evict_and_reclaim(caps, engine, gpu_tier, vf, gpu_tier, host_tier, peer_done);

    // Demote victim into L2 DRAM (evicting any stale copy and cascading to NVMe if DRAM is low).
    if pending_spills.iter().any(|p| p.key == vk) {
        wait_pending_spill_for_key(
            caps,
            engine,
            peek_sock,
            pending_spills,
            nvme_lru,
            gpu_tier,
            host_tier,
            vk,
            peer_done,
        );
    }
    if let Some(old_df) = dram_lru.remove(&vk) {
        evict_and_reclaim(caps, engine, host_tier, old_df, gpu_tier, host_tier, peer_done);
    }
    ensure_dram_space(
        caps,
        engine,
        peek_sock,
        dram_lru,
        nvme_lru,
        pending_spills,
        gpu_tier,
        host_tier,
        nvme_tier,
        peer_done,
    );
    let (_v_host, new_df) = engine
        .put_host(Tracked(caps.instance.borrow()), vk, d2h_scratch)
        .expect("put_host demote HBM->DRAM");
    let _ = dram_lru.insert(vk, new_df);
    kick_async_spills(
        caps,
        engine,
        peek_sock,
        dram_lru,
        nvme_lru,
        pending_spills,
        gpu_tier,
        host_tier,
        nvme_tier,
        peer_done,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_shard_worker(
    node_id: u64,
    shard_id: usize,
    gpu_id: DeviceId,
    cfg: &ScaleupConfig,
    job: &JobSpec,
    transport: ScaleupTcpTransport,
    reqs: Vec<Vec<u64>>,
    init_done_barrier: Arc<Barrier>,
    start_go_barrier: Arc<Barrier>,
) -> ShardStats {
    let frame_size = cfg.frame_size;
    let init_t0 = Instant::now();

    // 1. Share-nothing per-shard capability bundle carved from node's ShardAllocator.
    let mut caps = ShardCapabilityBundle::take(node_id);
    let device = open_rdma_device(node_id, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("shard PD");
    let mut core = caps.make_versioned_core(frame_size, 128);

    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        cfg.dram_frames_per_shard as u32,
        4096,
    )
    .expect("HostBackend::new (DRAM shard slab)");

    let mut cuda_b = CudaBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_cuda(),
        gpu_id,
        cfg.hbm_frames_per_shard as u32,
        32,
        true,
    )
    .expect("CudaBackend::new (HBM shard slab)");

    // Pin the HostBackend DRAM slab for 1-hop DMA H2D promotions!
    cuda_b
        .attach_pinned_host(&core, &host)
        .expect("attach_pinned_host");

    let nvme = NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_nvme(),
        job.nvme_dir_for_node(node_id),
        cfg.nvme_frames_per_shard as u32,
        64,
    )
    .expect("NvmeBackend::new (O_DIRECT io_uring shard slab)");

    let caller_tl = core
        .register_timeline(TimelineKind::CudaStream { device: gpu_id })
        .expect("register caller_tl");
    let mut caller_stream =
        CudaStream::new(gpu_id, caps.mint_cuda()).expect("caller CudaStream::new");

    let peek_sock = transport.clone_for_peek();
    let peer = connect_shard_peer_rdma(
        &mut caps,
        shard_id,
        job.gid_index,
        &mut core,
        &pd,
        &device,
        transport,
        &host,
        &cuda_b,
    );
    let eng_ctx = HostContext::new(caps.mint_host());
    let mut engine =
        DistributedEngine::new_with_nvme(core, &host, eng_ctx, Some(&cuda_b), peer, Some(nvme));

    let gpu_tier = engine.gpu_tier().unwrap();
    let host_tier = engine.host_tier();
    let nvme_tier = engine.nvme_tier().unwrap();

    // Reusable per-shard staging buffers (compute HBM slot, D2H demote pinned buf, cold writeback pinned buf).
    let (compute_slot_buf, Tracked(mut compute_tok)) =
        cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), gpu_id, frame_size)
            .expect("alloc compute slot");
    let compute_slot_slice = compute_slot_buf.slot_slice(0);

    let (d2h_stage_buf, Tracked(mut d2h_tok)) =
        cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), frame_size).expect("alloc d2h stage");
    let mut d2h_host_ctx = HostContext::new(caps.mint_host());
    let mut d2h_scratch = vec![0u8; frame_size];

    let (cold_stage_buf, Tracked(mut cold_tok)) =
        cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), frame_size).expect("alloc cold stage");
    let mut cold_ctx = HostContext::new(caps.mint_host());
    let cold_payload: Vec<u8> = (0..frame_size)
        .map(|i| ((i.wrapping_mul(29) + shard_id + 7) & 0xff) as u8)
        .collect();
    let cold_host_slice = cold_stage_buf.as_host_slice();
    let Tracked(cold_w_comp) = host::write_slice(
        &mut cold_ctx,
        Tracked(caps.instance.borrow()),
        &cold_host_slice,
        &cold_payload,
        Tracked(&mut cold_tok),
        Tracked(&engine.core.ledger.borrow().submitted),
        Tracked(&engine.core.ledger.borrow().knows),
        Ghost(engine.core.citations()),
    );
    let ghost cold_w_ev = cold_w_comp.submitted.key();
    let t_cold_w = engine
        .core
        .issue_ticket(engine.host.timeline_id(), Tracked(cold_w_comp.submitted))
        .expect("issue cold stage ticket");
    engine.core.mark_timeline_complete(
        engine.host.timeline_id(),
        t_cold_w.seq(),
        Tracked(cold_w_comp.knows),
    );
    let cold_cuda_slice = cold_stage_buf.as_cuda_slice();
    let tracked mut cold_perm = ReadPerm::Owned(cold_tok);
    let ghost cold_cert = ValueCertificate {
        slot: cold_cuda_slice.slot_ref(),
        writer: cold_w_ev,
        bytes: cold_payload@,
        proof: ValueProof::HostWrite,
    };

    let init_time_s = init_t0.elapsed().as_secs_f64();

    // Signal init completion, then wait for cross-node GO barrier before starting timed replay.
    init_done_barrier.wait();
    start_go_barrier.wait();

    let mut hbm_lru: LruPolicy<ObjectKey, FrameId> = LruPolicy::new(cfg.hbm_frames_per_shard);
    let mut dram_lru: LruPolicy<ObjectKey, FrameId> = LruPolicy::new(cfg.dram_frames_per_shard);
    let mut nvme_lru: LruPolicy<ObjectKey, FrameId> = LruPolicy::new(cfg.nvme_frames_per_shard);
    let mut pending_spills: Vec<PendingSpill> = Vec::with_capacity(SPILL_PIPELINE_QD);

    let mut stats = ShardStats {
        init_time_s,
        ..Default::default()
    };
    let mut hist = LatencyHist::new();
    let mut peer_done = false;

    for hash_ids in &reqs {
        // Service any incoming peer RDMA FetchRequest / ReadRelease promptly before starting request.
        drain_peer_nonblocking(
            &mut caps,
            &mut engine,
            &peek_sock,
            gpu_tier,
            host_tier,
            &mut peer_done,
        );
        let _ = reap_completed_spills(
            &mut caps,
            &mut engine,
            &mut pending_spills,
            &mut nvme_lru,
            gpu_tier,
            host_tier,
            &mut peer_done,
        );

        let req_t0 = Instant::now();
        let mut prefix_open = true;
        let mut pending_l1_hit_ticket: Option<Ticket> = None;

        for (blk_idx, &hid) in hash_ids.iter().enumerate() {
            if (blk_idx & 7) == 7 {
                drain_peer_nonblocking(
                    &mut caps,
                    &mut engine,
                    &peek_sock,
                    gpu_tier,
                    host_tier,
                    &mut peer_done,
                );
            }
            let key = key_from_hash_id(hid);

            if prefix_open {
                // Tier 1: L1 Local GPU HBM hit -> batched D2D read into compute slot (sync once at prefix end)
                if hbm_lru.contains(&key) {
                    if let Some((_ver, t_hit, _ev)) = engine
                        .get_into_gpu(
                            Tracked(caps.instance.borrow()),
                            Tracked(caps.shares.borrow_mut()),
                            key,
                            &mut caller_stream,
                            caller_tl,
                            &compute_slot_slice,
                            Tracked(&mut compute_tok),
                        )
                        .expect("get_into_gpu L1 HBM hit")
                    {
                        pending_l1_hit_ticket = Some(t_hit);
                        hbm_lru.touch(&key);
                        stats.hbm_hits += 1;
                        continue;
                    }
                }

                // Flush any batched L1 HBM hits before using caller_stream for H2D/D2H transfers.
                if let Some(t_last) = pending_l1_hit_ticket.take() {
                    let k_hit =
                        cuda::stream_synchronize(&caller_stream, Tracked(caps.instance.borrow()))
                            .expect("sync batched L1 HBM hits");
                    engine
                        .core
                        .mark_timeline_complete(caller_tl, t_last.seq(), k_hit);
                }

                // Tier 2: L2 Local Pinned DRAM hit -> 1-hop DMA H2D promote into L1 HBM
                if let Some(host_frame) = dram_lru.touch(&key) {
                    ensure_hbm_space(
                        &mut caps,
                        &mut engine,
                        &peek_sock,
                        &mut hbm_lru,
                        &mut dram_lru,
                        &mut nvme_lru,
                        &mut pending_spills,
                        gpu_tier,
                        host_tier,
                        nvme_tier,
                        &mut caller_stream,
                        caller_tl,
                        &d2h_stage_buf,
                        Tracked(&mut d2h_tok),
                        &mut d2h_host_ctx,
                        &mut d2h_scratch,
                        &mut peer_done,
                    );
                    let (lease_h, _l_ver, Ghost(src_cert), Tracked(share_cap)) = engine
                        .core
                        .acquire_local_lease(
                            host_frame,
                            false,
                            Tracked(caps.shares.borrow_mut()),
                            Tracked(caps.instance.borrow()),
                        )
                        .expect("acquire_local_lease DRAM->HBM");
                    let tracked mut src_perm = ReadPerm::Shared(share_cap);
                    let pinned_src = cuda_b.pinned_host_frame_slice(host_frame);
                    let (_ver, new_hbm_f) = engine
                        .put_gpu_sync(
                            Tracked(caps.instance.borrow()),
                            &mut caller_stream,
                            caller_tl,
                            key,
                            &pinned_src,
                            Tracked(&mut src_perm),
                            Ghost(src_cert),
                        )
                        .expect("put_gpu_sync promote DRAM->HBM");
                    let tracked returned_cap = src_perm.into_shared();
                    engine
                        .core
                        .release_local_lease(
                            lease_h,
                            Tracked(returned_cap),
                            Ticket::NONE,
                            Tracked(caps.instance.borrow()),
                        )
                        .expect("release_local_lease DRAM->HBM");
                    let _ = hbm_lru.insert(key, new_hbm_f);
                    if let Some(df) = dram_lru.remove(&key) {
                        evict_and_reclaim(
                            &mut caps,
                            &mut engine,
                            host_tier,
                            df,
                            gpu_tier,
                            host_tier,
                            &mut peer_done,
                        );
                    }
                    stats.dram_hits += 1;
                    continue;
                }

                // If this key is currently in flight to NVMe, wait for its spill to retire into nvme_lru.
                if pending_spills.iter().any(|p| p.key == key) {
                    wait_pending_spill_for_key(
                        &mut caps,
                        &mut engine,
                        &peek_sock,
                        &mut pending_spills,
                        &mut nvme_lru,
                        gpu_tier,
                        host_tier,
                        key,
                        &mut peer_done,
                    );
                }

                // Tier 3: L3 Local NVMe hit -> io_uring O_DIRECT warm to DRAM -> DMA H2D promote to HBM
                if nvme_lru.contains(&key) {
                    ensure_dram_space(
                        &mut caps,
                        &mut engine,
                        &peek_sock,
                        &mut dram_lru,
                        &mut nvme_lru,
                        &mut pending_spills,
                        gpu_tier,
                        host_tier,
                        nvme_tier,
                        &mut peer_done,
                    );
                    let warmed_df = engine
                        .warm_host_from_nvme_sync(
                            Tracked(caps.instance.borrow()),
                            Tracked(caps.shares.borrow_mut()),
                            key,
                        )
                        .expect("warm_host_from_nvme_sync");
                    let _ = reap_completed_spills(
                        &mut caps,
                        &mut engine,
                        &mut pending_spills,
                        &mut nvme_lru,
                        gpu_tier,
                        host_tier,
                        &mut peer_done,
                    );
                    if let Some(nf) = nvme_lru.remove(&key) {
                        evict_and_reclaim(
                            &mut caps,
                            &mut engine,
                            nvme_tier,
                            nf,
                            gpu_tier,
                            host_tier,
                            &mut peer_done,
                        );
                    }
                    ensure_hbm_space(
                        &mut caps,
                        &mut engine,
                        &peek_sock,
                        &mut hbm_lru,
                        &mut dram_lru,
                        &mut nvme_lru,
                        &mut pending_spills,
                        gpu_tier,
                        host_tier,
                        nvme_tier,
                        &mut caller_stream,
                        caller_tl,
                        &d2h_stage_buf,
                        Tracked(&mut d2h_tok),
                        &mut d2h_host_ctx,
                        &mut d2h_scratch,
                        &mut peer_done,
                    );
                    let (lease_h, _l_ver, Ghost(src_cert), Tracked(share_cap)) = engine
                        .core
                        .acquire_local_lease(
                            warmed_df,
                            false,
                            Tracked(caps.shares.borrow_mut()),
                            Tracked(caps.instance.borrow()),
                        )
                        .expect("acquire_local_lease NVMe->DRAM->HBM");
                    let tracked mut src_perm = ReadPerm::Shared(share_cap);
                    let pinned_src = cuda_b.pinned_host_frame_slice(warmed_df);
                    let (_ver, new_hbm_f) = engine
                        .put_gpu_sync(
                            Tracked(caps.instance.borrow()),
                            &mut caller_stream,
                            caller_tl,
                            key,
                            &pinned_src,
                            Tracked(&mut src_perm),
                            Ghost(src_cert),
                        )
                        .expect("put_gpu_sync promote NVMe->DRAM->HBM");
                    let tracked returned_cap = src_perm.into_shared();
                    engine
                        .core
                        .release_local_lease(
                            lease_h,
                            Tracked(returned_cap),
                            Ticket::NONE,
                            Tracked(caps.instance.borrow()),
                        )
                        .expect("release_local_lease NVMe->DRAM->HBM");
                    let _ = hbm_lru.insert(key, new_hbm_f);
                    evict_and_reclaim(
                        &mut caps,
                        &mut engine,
                        host_tier,
                        warmed_df,
                        gpu_tier,
                        host_tier,
                        &mut peer_done,
                    );
                    stats.nvme_hits += 1;
                    continue;
                }

                // Tier 4: Remote Peer hit -> cross-node GPUDirect RDMA into GPU HBM
                ensure_hbm_space(
                    &mut caps,
                    &mut engine,
                    &peek_sock,
                    &mut hbm_lru,
                    &mut dram_lru,
                    &mut nvme_lru,
                    &mut pending_spills,
                    gpu_tier,
                    host_tier,
                    nvme_tier,
                    &mut caller_stream,
                    caller_tl,
                    &d2h_stage_buf,
                    Tracked(&mut d2h_tok),
                    &mut d2h_host_ctx,
                    &mut d2h_scratch,
                    &mut peer_done,
                );
                let my_req_id = engine
                    .send_fetch_request(key, Tracked(caps.instance.borrow()))
                    .expect("send_fetch_request");
                let mut got_remote = false;
                loop {
                    if let Some(ev) = handle_peer_event(
                        &mut caps,
                        &mut engine,
                        gpu_tier,
                        host_tier,
                        &mut peer_done,
                    ) {
                        match ev {
                            DistributedEvent::ReadGrant(grant)
                                if grant.meta.request_id == my_req_id =>
                            {
                                let outcome = engine
                                    .pull_with_grant(
                                        key,
                                        gpu_tier,
                                        PullTarget::GpuGdr,
                                        grant,
                                        Tracked(caps.instance.borrow()),
                                    )
                                    .expect("pull_with_grant GpuGdr");
                                if let vericache_core::backends::rdma::peer::FetchOutcome::Hit {
                                    frame,
                                    ..
                                } = outcome
                                {
                                    let _ = hbm_lru.insert(key, frame);
                                    got_remote = true;
                                }
                                break;
                            }
                            DistributedEvent::Control(ControlPayload::FetchMiss { request_id })
                                if request_id == my_req_id =>
                            {
                                break;
                            }
                            other => {
                                panic!(
                                    "node {node_id} shard {shard_id}: unexpected peer event while waiting for req {my_req_id}: {:?}",
                                    std::mem::discriminant(&other)
                                );
                            }
                        }
                    }
                }
                if got_remote {
                    stats.remote_hits += 1;
                    continue;
                }
            }

            // Flush any batched L1 HBM hits before cold miss writeback on caller_stream.
            if let Some(t_last) = pending_l1_hit_ticket.take() {
                let k_hit =
                    cuda::stream_synchronize(&caller_stream, Tracked(caps.instance.borrow()))
                        .expect("sync batched L1 HBM hits before miss");
                engine
                    .core
                    .mark_timeline_complete(caller_tl, t_last.seq(), k_hit);
            }

            // Tier 5: Cold Miss -> close prefix match and writeback suffix block into L1 HBM
            prefix_open = false;
            stats.cold_misses += 1;

            if hbm_lru.contains(&key) {
                hbm_lru.touch(&key);
                continue;
            }
            if pending_spills.iter().any(|p| p.key == key) {
                wait_pending_spill_for_key(
                    &mut caps,
                    &mut engine,
                    &peek_sock,
                    &mut pending_spills,
                    &mut nvme_lru,
                    gpu_tier,
                    host_tier,
                    key,
                    &mut peer_done,
                );
            }
            if let Some(df) = dram_lru.remove(&key) {
                evict_and_reclaim(
                    &mut caps,
                    &mut engine,
                    host_tier,
                    df,
                    gpu_tier,
                    host_tier,
                    &mut peer_done,
                );
            }
            if let Some(nf) = nvme_lru.remove(&key) {
                evict_and_reclaim(
                    &mut caps,
                    &mut engine,
                    nvme_tier,
                    nf,
                    gpu_tier,
                    host_tier,
                    &mut peer_done,
                );
            }
            ensure_hbm_space(
                &mut caps,
                &mut engine,
                &peek_sock,
                &mut hbm_lru,
                &mut dram_lru,
                &mut nvme_lru,
                &mut pending_spills,
                gpu_tier,
                host_tier,
                nvme_tier,
                &mut caller_stream,
                caller_tl,
                &d2h_stage_buf,
                Tracked(&mut d2h_tok),
                &mut d2h_host_ctx,
                &mut d2h_scratch,
                &mut peer_done,
            );
            let (_ver, new_hbm_f) = engine
                .put_gpu_sync(
                    Tracked(caps.instance.borrow()),
                    &mut caller_stream,
                    caller_tl,
                    key,
                    &cold_cuda_slice,
                    Tracked(&mut cold_perm),
                    Ghost(cold_cert),
                )
                .expect("put_gpu_sync cold writeback");
            let _ = hbm_lru.insert(key, new_hbm_f);
        }

        // Flush any trailing batched L1 HBM hits at request completion.
        if let Some(t_last) = pending_l1_hit_ticket.take() {
            let k_hit = cuda::stream_synchronize(&caller_stream, Tracked(caps.instance.borrow()))
                .expect("sync trailing L1 HBM hits");
            engine
                .core
                .mark_timeline_complete(caller_tl, t_last.seq(), k_hit);
        }

        let req_ns = req_t0.elapsed().as_nanos() as u64;
        hist.record_ns(req_ns);
        stats.reqs += 1;
    }

    drain_all_pending_spills(
        &mut caps,
        &mut engine,
        &peek_sock,
        &mut pending_spills,
        &mut nvme_lru,
        gpu_tier,
        host_tier,
        &mut peer_done,
    );

    // Notify peer shard we finished all our local requests, then keep serving peer FetchRequests
    // until peer also signals PEER_DONE_TAG and all remote leases are released!
    engine
        .send_control(
            ControlPayload::RevokeAck {
                grant_tag: PEER_DONE_TAG,
            },
            Tracked(caps.instance.borrow()),
        )
        .expect("send PEER_DONE_TAG");
    while !peer_done || engine.peers[0].rdma.active_lease_count() > 0 {
        let _ = handle_peer_event(&mut caps, &mut engine, gpu_tier, host_tier, &mut peer_done);
    }

    let _ = caller_stream.destroy();
    stats.hist_buckets = hist.export_buckets();
    stats
}

} // verus!

fn partition_requests_for_node(
    reqs: &[TraceRequest],
    node_id: u64,
    shards_per_node: usize,
    sticky_pct: u64,
) -> Vec<Vec<Vec<u64>>> {
    let mut per_shard: Vec<Vec<Vec<u64>>> = (0..shards_per_node).map(|_| Vec::new()).collect();
    for (req_idx, req) in reqs.iter().enumerate() {
        let (target_node, shard_id) =
            route_request(req_idx, &req.hash_ids, shards_per_node, sticky_pct);
        if target_node == node_id {
            per_shard[shard_id].push(req.hash_ids.clone());
        }
    }
    per_shard
}

fn run_node(
    node_id: u64,
    gpu_id: DeviceId,
    job: &JobSpec,
    cfg: &ScaleupConfig,
    reqs: &[TraceRequest],
    shard_transports: Vec<ScaleupTcpTransport>,
    coord_transport: &mut ScaleupTcpTransport,
) -> NodeReport {
    // Fail fast on any worker thread panic so a Barrier can never deadlock.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[scaleup_cluster] FATAL PANIC: {info}");
        std::process::abort();
    }));
    // Bootstrap NodeContext -> ShardAllocator once per OS process; each worker thread carves its own disjoint ShardSources.
    ensure_bench_node_allocator(node_id);
    fs::create_dir_all(job.nvme_dir_for_node(node_id)).expect("create nvme_dir");
    let per_shard_reqs =
        partition_requests_for_node(reqs, node_id, cfg.shards_per_node, cfg.sticky_pct);
    let init_done_barrier = Arc::new(Barrier::new(cfg.shards_per_node + 1));
    let start_go_barrier = Arc::new(Barrier::new(cfg.shards_per_node + 1));

    let node_init_t0 = Instant::now();
    let mut handles = Vec::with_capacity(cfg.shards_per_node);
    for (shard_id, (transport, s_reqs)) in shard_transports
        .into_iter()
        .zip(per_shard_reqs.into_iter())
        .enumerate()
    {
        let job_clone = job.clone();
        let cfg_copy = ScaleupConfig::from_job(job);
        let b_done = Arc::clone(&init_done_barrier);
        let b_go = Arc::clone(&start_go_barrier);
        handles.push(thread::spawn(move || {
            run_shard_worker(
                node_id, shard_id, gpu_id, &cfg_copy, &job_clone, transport, s_reqs, b_done, b_go,
            )
        }));
    }

    // Wait for all shards on this node to complete parallel pool allocation & QP setup.
    init_done_barrier.wait();
    let init_time_s = node_init_t0.elapsed().as_secs_f64();

    // Cross-node ready barrier so Node 0 and Node 1 begin timed trace replay simultaneously.
    if node_id == 0 {
        let _ = coord_transport.recv_frame().expect("recv node1 ready");
        coord_transport.send_frame(b"GO").expect("send GO to node1");
    } else {
        coord_transport
            .send_frame(b"READY")
            .expect("send READY to node0");
        let _ = coord_transport.recv_frame().expect("recv GO from node0");
    }

    let sampler = ProcSampler::start();
    let run_t0 = Instant::now();
    start_go_barrier.wait();

    let mut merged_hist = LatencyHist::new();
    let mut total_reqs = 0u64;
    let mut hbm_hits = 0u64;
    let mut dram_hits = 0u64;
    let mut nvme_hits = 0u64;
    let mut remote_hits = 0u64;
    let mut cold_misses = 0u64;
    let mut max_shard_init = init_time_s;

    for h in handles {
        let st = h.join().expect("shard worker panicked");
        total_reqs += st.reqs;
        hbm_hits += st.hbm_hits;
        dram_hits += st.dram_hits;
        nvme_hits += st.nvme_hits;
        remote_hits += st.remote_hits;
        cold_misses += st.cold_misses;
        if st.init_time_s > max_shard_init {
            max_shard_init = st.init_time_s;
        }
        merged_hist.merge_buckets(&st.hist_buckets);
    }

    let elapsed_s = run_t0.elapsed().as_secs_f64();
    let (cpu_pct, rss_now_mb) = sampler.finish();
    let rss_mb = rss_now_mb.max(read_proc_hwm_mb());
    let vram_mb = (cfg.hbm_frames_per_shard * cfg.shards_per_node * cfg.frame_size) as f64
        / (1024.0 * 1024.0);

    NodeReport {
        node_id,
        init_time_s: max_shard_init,
        elapsed_s,
        cpu_pct,
        rss_mb,
        vram_mb,
        reqs: total_reqs,
        hbm_hits,
        dram_hits,
        nvme_hits,
        remote_hits,
        cold_misses,
        hist_buckets: merged_hist.export_buckets(),
    }
}

fn accept_all_node1_channels(
    listener: &TcpListener,
    session: &str,
    shards_per_node: usize,
    child: &mut Child,
) -> (Vec<ScaleupTcpTransport>, ScaleupTcpTransport) {
    listener.set_nonblocking(true).expect("set_nonblocking");
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut shard_slots: Vec<Option<ScaleupTcpTransport>> =
        (0..shards_per_node).map(|_| None).collect();
    let mut coord_slot: Option<ScaleupTcpTransport> = None;
    let expected = shards_per_node + 1;
    let mut accepted = 0usize;

    while accepted < expected {
        match listener.accept() {
            Ok((stream, _)) => {
                let mut t = ScaleupTcpTransport::new(stream);
                let hello = t.recv_frame().expect("recv channel hello");
                assert!(hello.len() == 17, "bad hello len {}", hello.len());
                assert_eq!(&hello[..16], &session.as_bytes()[..16], "session mismatch");
                let ch_id = hello[16];
                t.send_frame(&session.as_bytes()[16..])
                    .expect("send hello ack");
                if ch_id == COORD_CHANNEL_ID {
                    coord_slot = Some(t);
                } else {
                    let s = ch_id as usize;
                    assert!(s < shards_per_node, "invalid shard channel {s}");
                    shard_slots[s] = Some(t);
                }
                accepted += 1;
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(st) = child.try_wait().expect("try_wait") {
                    panic!("node1 child exited early during connect: {st}");
                }
                assert!(Instant::now() < deadline, "node1 connect timed out");
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => panic!("accept failed: {e}"),
        }
    }

    let shards = shard_slots
        .into_iter()
        .map(|o| o.expect("missing shard transport"))
        .collect();
    let coord = coord_slot.expect("missing coord transport");
    (shards, coord)
}

fn connect_channel(addr: &str, session: &str, ch_id: u8) -> ScaleupTcpTransport {
    let stream = TcpStream::connect(addr).expect("connect to node0");
    let mut t = ScaleupTcpTransport::new(stream);
    let mut hello = Vec::with_capacity(17);
    hello.extend_from_slice(&session.as_bytes()[..16]);
    hello.push(ch_id);
    t.send_frame(&hello).expect("send hello");
    let ack = t.recv_frame().expect("recv ack");
    assert_eq!(ack, &session.as_bytes()[16..], "session ack mismatch");
    t
}

pub fn run(job: &JobSpec) -> Vec<MetricRecord> {
    let cfg = ScaleupConfig::from_job(job);
    let trace_path = job
        .trace_path
        .as_deref()
        .expect("s5_scaleup_cluster requires trace_path");
    let reqs = load_trace(trace_path, job.max_requests);

    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback listener");
    let addr = listener.local_addr().unwrap().to_string();
    let mut secret = [0u8; 16];
    fs::File::open("/dev/urandom")
        .expect("urandom")
        .read_exact(&mut secret)
        .expect("read secret");
    let session: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    let job_json = serde_json::to_string(job).expect("serialize job");

    let exe = std::env::current_exe().expect("current_exe");
    let mut node1_child: Child = Command::new(&exe)
        .args([
            "--job-json",
            &job_json,
            "--role",
            "node1",
            "--peer-addr",
            &addr,
            "--session",
            &session,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn node1 child");

    let (shard_transports, mut coord) =
        accept_all_node1_channels(&listener, &session, cfg.shards_per_node, &mut node1_child);

    let r0 = run_node(
        0,
        job.donor_gpu,
        job,
        &cfg,
        &reqs,
        shard_transports,
        &mut coord,
    );

    // Receive Node 1's report over coord channel.
    let r1_bytes = coord.recv_frame().expect("recv node1 report");
    let r1: NodeReport = serde_json::from_slice(&r1_bytes).expect("parse node1 report");

    let output = node1_child.wait_with_output().expect("wait node1 child");
    assert!(
        output.status.success(),
        "node1 child failed with status {}",
        output.status
    );

    // Aggregate 2-node cluster metrics.
    let mut cluster_hist = LatencyHist::new();
    cluster_hist.merge_buckets(&r0.hist_buckets);
    cluster_hist.merge_buckets(&r1.hist_buckets);

    let init_time_s = r0.init_time_s.max(r1.init_time_s);
    let elapsed_s = r0.elapsed_s.max(r1.elapsed_s).max(1e-9);
    let total_reqs = r0.reqs + r1.reqs;
    let hbm_hits = r0.hbm_hits + r1.hbm_hits;
    let dram_hits = r0.dram_hits + r1.dram_hits;
    let nvme_hits = r0.nvme_hits + r1.nvme_hits;
    let remote_hits = r0.remote_hits + r1.remote_hits;
    let cold_misses = r0.cold_misses + r1.cold_misses;
    let total_blocks = hbm_hits + dram_hits + nvme_hits + remote_hits + cold_misses;
    let total_hits = hbm_hits + dram_hits + nvme_hits + remote_hits;

    let pct = |n: u64| -> f64 {
        if total_blocks == 0 {
            0.0
        } else {
            (n as f64) * 100.0 / (total_blocks as f64)
        }
    };

    let total_bytes = total_blocks * (cfg.frame_size as u64);
    let hit_bytes = total_hits * (cfg.frame_size as u64);
    let throughput_gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);
    let goodput_gbps = throughput_gbps;
    let hit_goodput_gbps = bytes_per_sec_to_gbps(hit_bytes, elapsed_s);
    let req_per_s = (total_reqs as f64) / elapsed_s;
    let hit_rate = if total_blocks > 0 {
        (total_hits as f64) / (total_blocks as f64)
    } else {
        0.0
    };

    let cpu_pct = r0.cpu_pct + r1.cpu_pct;
    let rss_mb = r0.rss_mb + r1.rss_mb;
    let vram_mb = r0.vram_mb + r1.vram_mb;
    let total_cluster_frames = 2
        * cfg.shards_per_node
        * (cfg.hbm_frames_per_shard + cfg.dram_frames_per_shard + cfg.nvme_frames_per_shard);

    let notes_obj = serde_json::json!({
        "init_time_s": (init_time_s * 1000.0).round() / 1000.0,
        "elapsed_s": (elapsed_s * 1000.0).round() / 1000.0,
        "req_per_s": (req_per_s * 100.0).round() / 100.0,
        "reqs": total_reqs,
        "total_blocks": total_blocks,
        "hit_goodput_gbps": (hit_goodput_gbps * 100.0).round() / 100.0,
        "hbm_hit_pct": (pct(hbm_hits) * 100.0).round() / 100.0,
        "dram_hit_pct": (pct(dram_hits) * 100.0).round() / 100.0,
        "nvme_hit_pct": (pct(nvme_hits) * 100.0).round() / 100.0,
        "remote_hit_pct": (pct(remote_hits) * 100.0).round() / 100.0,
        "cold_miss_pct": (pct(cold_misses) * 100.0).round() / 100.0,
        "hbm_hits": hbm_hits,
        "dram_hits": dram_hits,
        "nvme_hits": nvme_hits,
        "remote_hits": remote_hits,
        "cold_misses": cold_misses,
        "p999_us": (cluster_hist.p999_us() * 10.0).round() / 10.0,
        "vram_mb": (vram_mb * 10.0).round() / 10.0,
        "hbm_gb_per_node": cfg.hbm_gb_per_node,
        "dram_gb_per_node": cfg.dram_gb_per_node,
        "nvme_gb_per_node": cfg.nvme_gb_per_node,
        "shards_per_node": cfg.shards_per_node,
    });

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.num_blocks = total_cluster_frames;
    rec.concurrency = cfg.shards_per_node * 2;
    rec.throughput_gbps = throughput_gbps;
    rec.goodput_gbps = goodput_gbps;
    rec.p50_us = cluster_hist.p50_us();
    rec.p95_us = cluster_hist.p95_us();
    rec.p99_us = cluster_hist.p99_us();
    rec.hit_rate = hit_rate;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = notes_obj.to_string();
    vec![rec]
}

pub fn run_role(job: &JobSpec, role: &str, peer_addr: Option<&str>, session: Option<&str>) {
    assert_eq!(role, "node1", "s5_scaleup_cluster child role must be node1");
    let addr = peer_addr.expect("peer_addr");
    let sess = session.expect("session");
    let cfg = ScaleupConfig::from_job(job);
    let trace_path = job
        .trace_path
        .as_deref()
        .expect("s5_scaleup_cluster requires trace_path");
    let reqs = load_trace(trace_path, job.max_requests);

    let mut shard_transports = Vec::with_capacity(cfg.shards_per_node);
    for s in 0..cfg.shards_per_node {
        shard_transports.push(connect_channel(addr, sess, s as u8));
    }
    let mut coord = connect_channel(addr, sess, COORD_CHANNEL_ID);

    let report = run_node(
        1,
        job.initiator_gpu,
        job,
        &cfg,
        &reqs,
        shard_transports,
        &mut coord,
    );
    let payload = serde_json::to_vec(&report).expect("serialize node1 report");
    coord.send_frame(&payload).expect("send node1 report");
}
