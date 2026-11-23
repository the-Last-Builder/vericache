use crate::ffi::{
    CudaDevBuf, CudaPinnedBuf, CudaStream, ODirectFile, TransferEngine, cuda_device_sync,
    cuda_set_device,
};
use crate::policy::LruPolicy;
use crate::schema::{
    JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps, read_proc_hwm_mb,
};
use crate::trace::{TraceRequest, load_trace, route_request};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const COORD_CHANNEL_ID: u8 = 255;

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

fn configure_sock(sock: &TcpStream) {
    sock.set_nodelay(true).expect("set_nodelay");
    sock.set_read_timeout(Some(Duration::from_secs(1800)))
        .expect("set_read_timeout");
    sock.set_write_timeout(Some(Duration::from_secs(1800)))
        .expect("set_write_timeout");
}

fn send_frame(sock: &mut TcpStream, data: &[u8]) {
    let n = data.len() as u32;
    sock.write_all(&n.to_be_bytes()).expect("write frame len");
    sock.write_all(data).expect("write frame payload");
}

fn recv_frame(sock: &mut TcpStream) -> Vec<u8> {
    let mut len_b = [0u8; 4];
    sock.read_exact(&mut len_b).expect("read frame len");
    let n = u32::from_be_bytes(len_b) as usize;
    let mut buf = vec![0u8; n];
    sock.read_exact(&mut buf).expect("read frame payload");
    buf
}

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

struct ShardState {
    hbm_lru: LruPolicy<u64, usize>,
    dram_lru: LruPolicy<u64, usize>,
    nvme_lru: LruPolicy<u64, usize>,
    free_hbm: Vec<usize>,
    free_dram: Vec<usize>,
    free_nvme: Vec<usize>,
    pinned_hbm: HashMap<usize, u32>,
    pinned_dram: HashMap<usize, u32>,
}

impl ShardState {
    fn new(hbm_f: usize, dram_f: usize, nvme_f: usize) -> Self {
        Self {
            hbm_lru: LruPolicy::new(hbm_f),
            dram_lru: LruPolicy::new(dram_f),
            nvme_lru: LruPolicy::new(nvme_f),
            free_hbm: (0..hbm_f).collect(),
            free_dram: (0..dram_f).collect(),
            free_nvme: (0..nvme_f).collect(),
            pinned_hbm: HashMap::new(),
            pinned_dram: HashMap::new(),
        }
    }
}

fn ensure_dram_slot(
    shared: &(Mutex<ShardState>, Condvar),
    nvme: &ODirectFile,
    dram_ptr: u64,
    frame_size: usize,
) -> usize {
    let (lock, cvar) = shared;
    let (dk, dslot) = {
        let mut st = lock.lock().unwrap();
        if let Some(s) = st.free_dram.pop() {
            return s;
        }
        let (dk, dslot) = st.dram_lru.pop_lru().expect("dram_lru non-empty");
        while *st.pinned_dram.get(&dslot).unwrap_or(&0) > 0 {
            st = cvar.wait(st).unwrap();
        }
        (dk, dslot)
    };

    let nslot = {
        let mut st = lock.lock().unwrap();
        if let Some(ns) = st.free_nvme.pop() {
            ns
        } else {
            st.nvme_lru.pop_lru().expect("nvme_lru non-empty").1
        }
    };

    nvme.pwrite(
        dram_ptr + (dslot * frame_size) as u64,
        frame_size,
        (nslot * frame_size) as u64,
    );

    {
        let mut st = lock.lock().unwrap();
        if let Some(old_n) = st.nvme_lru.remove(&dk) {
            st.free_nvme.push(old_n);
        }
        st.nvme_lru.insert(dk, nslot);
    }
    dslot
}

#[allow(clippy::too_many_arguments)]
fn ensure_hbm_slot(
    shared: &(Mutex<ShardState>, Condvar),
    nvme: &ODirectFile,
    stream: &CudaStream,
    hbm_ptr: u64,
    dram_ptr: u64,
    frame_size: usize,
) -> usize {
    let (lock, cvar) = shared;
    let (vk, hslot) = {
        let mut st = lock.lock().unwrap();
        if let Some(s) = st.free_hbm.pop() {
            return s;
        }
        let (vk, hslot) = st.hbm_lru.pop_lru().expect("hbm_lru non-empty");
        while *st.pinned_hbm.get(&hslot).unwrap_or(&0) > 0 {
            st = cvar.wait(st).unwrap();
        }
        if let Some(old_d) = st.dram_lru.remove(&vk) {
            while *st.pinned_dram.get(&old_d).unwrap_or(&0) > 0 {
                st = cvar.wait(st).unwrap();
            }
            st.free_dram.push(old_d);
        }
        (vk, hslot)
    };

    let dslot = ensure_dram_slot(shared, nvme, dram_ptr, frame_size);
    stream.memcpy_d2h_async(
        dram_ptr + (dslot * frame_size) as u64,
        hbm_ptr + (hslot * frame_size) as u64,
        frame_size,
    );
    stream.sync();

    {
        let mut st = lock.lock().unwrap();
        st.dram_lru.insert(vk, dslot);
    }
    hslot
}

#[allow(clippy::too_many_arguments)]
fn run_shard_worker(
    node_id: u64,
    shard_id: usize,
    gpu_id: u32,
    cfg: &ScaleupConfig,
    job: &JobSpec,
    mut req_sock: TcpStream,
    mut srv_sock: TcpStream,
    reqs: Vec<Vec<u64>>,
    init_done_barrier: Arc<Barrier>,
    start_go_barrier: Arc<Barrier>,
) -> ShardStats {
    cuda_set_device(gpu_id);
    let init_t0 = Instant::now();

    let frame_size = cfg.frame_size;
    let hbm_f = cfg.hbm_frames_per_shard;
    let dram_f = cfg.dram_frames_per_shard;
    let nvme_f = cfg.nvme_frames_per_shard;

    let te = TransferEngine::new(&job.rdma_dev).unwrap_or_else(|e| {
        panic!("Mooncake S5 node{node_id} shard{shard_id} TE init failed: {e}")
    });
    let local_seg = te.segment_name();

    let hbm_bytes = hbm_f * frame_size;
    let dram_bytes = dram_f * frame_size;
    let nvme_bytes = (nvme_f as u64) * (frame_size as u64);

    let hbm_slab = CudaDevBuf::alloc(hbm_bytes);
    let dram_slab = CudaPinnedBuf::alloc(dram_bytes);
    let hbm_ptr = hbm_slab.addr();
    let dram_ptr = dram_slab.addr();

    te.register_memory(hbm_ptr, hbm_bytes).unwrap_or_else(|rc| {
        panic!(
            "Mooncake S5 node{node_id} shard{shard_id} register HBM ({hbm_bytes} B) failed rc={rc}"
        )
    });
    te.register_memory(dram_ptr, dram_bytes)
        .unwrap_or_else(|rc| {
            panic!(
                "Mooncake S5 node{node_id} shard{shard_id} register DRAM ({dram_bytes} B) failed rc={rc}"
            )
        });

    let nvme_path = format!(
        "{}/mooncake_s5_{}_n{}_s{}.bin",
        job.nvme_dir_for_node(node_id),
        std::process::id(),
        node_id,
        shard_id
    );
    let nvme = ODirectFile::create_unlinked(&nvme_path, nvme_bytes);

    // Exchange TransferEngine segment names over the shard's unidirectional TCP sockets.
    // Both sides have already registered both HBM & DRAM before sending seg_name.
    send_frame(&mut req_sock, local_seg.as_bytes());
    let peer_seg_b = recv_frame(&mut srv_sock);
    let peer_seg = String::from_utf8(peer_seg_b).expect("utf8 peer_seg");
    let peer_seg_id = te.open_segment(&peer_seg).unwrap_or_else(|rc| {
        panic!("Mooncake S5 node{node_id} shard{shard_id} open_segment({peer_seg}) failed rc={rc}")
    });

    let compute_slot = CudaDevBuf::alloc(frame_size);
    let mut cold_src = CudaPinnedBuf::alloc(frame_size);
    cold_src.fill(0x3C ^ (shard_id as u8));
    let stream = CudaStream::new();
    cuda_device_sync();

    let init_time_s = init_t0.elapsed().as_secs_f64();
    let shared = (Mutex::new(ShardState::new(hbm_f, dram_f, nvme_f)), Condvar::new());

    let mut stats = ShardStats {
        init_time_s,
        ..Default::default()
    };

    thread::scope(|scope| {
        let shared_ref = &shared;
        // Background peer responder thread serving cross-node RDMA queries (HBM or DRAM).
        let resp_handle = scope.spawn(move || {
            let (lock, cvar) = shared_ref;
            let mut hdr = [0u8; 9];
            let mut reply = [0u8; 17];
            loop {
                srv_sock.read_exact(&mut hdr).expect("responder read hdr");
                let op = hdr[0];
                let val = u64::from_be_bytes(hdr[1..9].try_into().unwrap());
                match op {
                    1 => {
                        // QUERY
                        let mut st = lock.lock().unwrap();
                        if let Some(slot) = st.hbm_lru.get(&val) {
                            *st.pinned_hbm.entry(slot).or_insert(0) += 1;
                            let addr = hbm_ptr + (slot * frame_size) as u64;
                            reply[0] = 1;
                            reply[1..9].copy_from_slice(&addr.to_be_bytes());
                            reply[9..17].copy_from_slice(&(slot as u64).to_be_bytes());
                        } else if let Some(slot) = st.dram_lru.get(&val) {
                            *st.pinned_dram.entry(slot).or_insert(0) += 1;
                            let addr = dram_ptr + (slot * frame_size) as u64;
                            reply[0] = 2;
                            reply[1..9].copy_from_slice(&addr.to_be_bytes());
                            reply[9..17].copy_from_slice(&(slot as u64).to_be_bytes());
                        } else {
                            reply.fill(0);
                        }
                        drop(st);
                        srv_sock.write_all(&reply).expect("responder write query reply");
                    }
                    2 => {
                        // RELEASE (tier_tag << 56 | slot)
                        let tier_tag = val >> 56;
                        let slot = (val & 0x00FF_FFFF_FFFF_FFFF) as usize;
                        {
                            let mut st = lock.lock().unwrap();
                            if tier_tag == 1 {
                                let cnt = st.pinned_hbm.get(&slot).copied().unwrap_or(1);
                                if cnt <= 1 {
                                    st.pinned_hbm.remove(&slot);
                                } else {
                                    st.pinned_hbm.insert(slot, cnt - 1);
                                }
                            } else {
                                let cnt = st.pinned_dram.get(&slot).copied().unwrap_or(1);
                                if cnt <= 1 {
                                    st.pinned_dram.remove(&slot);
                                } else {
                                    st.pinned_dram.insert(slot, cnt - 1);
                                }
                            }
                            cvar.notify_all();
                        }
                        srv_sock.write_all(b"K").expect("responder write release ack");
                    }
                    3 => {
                        // DONE
                        srv_sock.write_all(b"D").expect("responder write done ack");
                        break;
                    }
                    other => panic!("unexpected responder op={other}"),
                }
            }
        });

        init_done_barrier.wait();
        start_go_barrier.wait();

        let (lock, cvar) = shared_ref;
        let mut hist = LatencyHist::new();
        let mut req_buf = [0u8; 9];
        let mut resp_buf = [0u8; 17];
        let mut ack_buf = [0u8; 1];

        for hash_ids in &reqs {
            let req_t0 = Instant::now();
            let mut prefix_open = true;
            let mut pending_l1_sync = false;

            for &hid in hash_ids {
                if prefix_open {
                    // Tier 1: L1 Local GPU HBM hit -> batched D2D async copy into compute slot
                    let hslot_hit = {
                        let mut st = lock.lock().unwrap();
                        st.hbm_lru.touch(&hid)
                    };
                    if let Some(hslot) = hslot_hit {
                        stream.memcpy_d2d_async(
                            compute_slot.addr(),
                            hbm_ptr + (hslot * frame_size) as u64,
                            frame_size,
                        );
                        pending_l1_sync = true;
                        stats.hbm_hits += 1;
                        continue;
                    }

                    if pending_l1_sync {
                        stream.sync();
                        pending_l1_sync = false;
                    }

                    // Tier 2: L2 Pinned DRAM hit -> 1-hop DMA H2D promote to HBM
                    let dslot_hit = {
                        let mut st = lock.lock().unwrap();
                        if let Some(ds) = st.dram_lru.remove(&hid) {
                            while *st.pinned_dram.get(&ds).unwrap_or(&0) > 0 {
                                st = cvar.wait(st).unwrap();
                            }
                            Some(ds)
                        } else {
                            None
                        }
                    };
                    if let Some(dslot) = dslot_hit {
                        let new_hslot = ensure_hbm_slot(
                            shared_ref, &nvme, &stream, hbm_ptr, dram_ptr, frame_size,
                        );
                        stream.memcpy_h2d_async(
                            hbm_ptr + (new_hslot * frame_size) as u64,
                            dram_ptr + (dslot * frame_size) as u64,
                            frame_size,
                        );
                        stream.sync();
                        {
                            let mut st = lock.lock().unwrap();
                            st.free_dram.push(dslot);
                            st.hbm_lru.insert(hid, new_hslot);
                        }
                        stats.dram_hits += 1;
                        continue;
                    }

                    // Tier 3: L3 Local NVMe hit -> O_DIRECT read -> DMA H2D promote to HBM
                    let nslot_hit = {
                        let mut st = lock.lock().unwrap();
                        st.nvme_lru.remove(&hid)
                    };
                    if let Some(nslot) = nslot_hit {
                        let tmp_dslot = ensure_dram_slot(shared_ref, &nvme, dram_ptr, frame_size);
                        nvme.pread(
                            dram_ptr + (tmp_dslot * frame_size) as u64,
                            frame_size,
                            (nslot * frame_size) as u64,
                        );
                        let new_hslot = ensure_hbm_slot(
                            shared_ref, &nvme, &stream, hbm_ptr, dram_ptr, frame_size,
                        );
                        stream.memcpy_h2d_async(
                            hbm_ptr + (new_hslot * frame_size) as u64,
                            dram_ptr + (tmp_dslot * frame_size) as u64,
                            frame_size,
                        );
                        stream.sync();
                        {
                            let mut st = lock.lock().unwrap();
                            st.free_nvme.push(nslot);
                            st.free_dram.push(tmp_dslot);
                            st.hbm_lru.insert(hid, new_hslot);
                        }
                        stats.nvme_hits += 1;
                        continue;
                    }

                    // Tier 4: Remote Peer hit -> Mooncake native C++ TransferEngine GPUDirect RDMA into GPU HBM
                    req_buf[0] = 1;
                    req_buf[1..9].copy_from_slice(&hid.to_be_bytes());
                    req_sock.write_all(&req_buf).expect("send peer query");
                    req_sock.read_exact(&mut resp_buf).expect("recv peer query reply");
                    let found = resp_buf[0];
                    if found != 0 {
                        let remote_addr = u64::from_be_bytes(resp_buf[1..9].try_into().unwrap());
                        let remote_slot = u64::from_be_bytes(resp_buf[9..17].try_into().unwrap());
                        let new_hslot = ensure_hbm_slot(
                            shared_ref, &nvme, &stream, hbm_ptr, dram_ptr, frame_size,
                        );
                        let local_gpu_addr = hbm_ptr + (new_hslot * frame_size) as u64;
                        let rd_res = te.sync_read(peer_seg_id, local_gpu_addr, remote_addr, frame_size);

                        let rel_val = ((found as u64) << 56) | remote_slot;
                        req_buf[0] = 2;
                        req_buf[1..9].copy_from_slice(&rel_val.to_be_bytes());
                        req_sock.write_all(&req_buf).expect("send peer release");
                        req_sock.read_exact(&mut ack_buf).expect("recv release ack");

                        rd_res.unwrap_or_else(|rc| {
                            panic!(
                                "Mooncake S5 sync_read failed rc={rc} (node={node_id}, shard={shard_id})"
                            )
                        });
                        {
                            let mut st = lock.lock().unwrap();
                            st.hbm_lru.insert(hid, new_hslot);
                        }
                        stats.remote_hits += 1;
                        continue;
                    }
                }

                if pending_l1_sync {
                    stream.sync();
                    pending_l1_sync = false;
                }

                // Tier 5: Cold Miss -> writeback suffix frame into L1 HBM
                prefix_open = false;
                stats.cold_misses += 1;

                let already_in_hbm = {
                    let mut st = lock.lock().unwrap();
                    if st.hbm_lru.contains(&hid) {
                        st.hbm_lru.touch(&hid);
                        true
                    } else {
                        if let Some(old_d) = st.dram_lru.remove(&hid) {
                            while *st.pinned_dram.get(&old_d).unwrap_or(&0) > 0 {
                                st = cvar.wait(st).unwrap();
                            }
                            st.free_dram.push(old_d);
                        }
                        if let Some(old_n) = st.nvme_lru.remove(&hid) {
                            st.free_nvme.push(old_n);
                        }
                        false
                    }
                };
                if already_in_hbm {
                    continue;
                }

                let new_hslot =
                    ensure_hbm_slot(shared_ref, &nvme, &stream, hbm_ptr, dram_ptr, frame_size);
                stream.memcpy_h2d_async(
                    hbm_ptr + (new_hslot * frame_size) as u64,
                    cold_src.addr(),
                    frame_size,
                );
                stream.sync();
                {
                    let mut st = lock.lock().unwrap();
                    st.hbm_lru.insert(hid, new_hslot);
                }
            }

            if pending_l1_sync {
                stream.sync();
            }

            hist.record_ns(req_t0.elapsed().as_nanos().max(1) as u64);
            stats.reqs += 1;
        }

        // Signal peer responder we are done sending queries and wait for peer responder to finish.
        req_buf[0] = 3;
        req_buf[1..9].copy_from_slice(&0u64.to_be_bytes());
        req_sock.write_all(&req_buf).expect("send DONE");
        req_sock.read_exact(&mut ack_buf).expect("recv DONE ack");
        resp_handle.join().expect("responder thread panicked");

        stats.hist_buckets = hist.export_buckets();
    });

    // Unregister slabs before dropping buffers and TransferEngine.
    let _ = te.unregister_memory(hbm_ptr);
    let _ = te.unregister_memory(dram_ptr);
    drop(te);
    drop(nvme);
    drop(stream);
    drop(compute_slot);
    drop(cold_src);
    drop(hbm_slab);
    drop(dram_slab);

    stats
}

fn run_node(
    node_id: u64,
    gpu_id: u32,
    job: &JobSpec,
    cfg: &ScaleupConfig,
    reqs: &[TraceRequest],
    shard_socks: Vec<(TcpStream, TcpStream)>,
    coord: &mut TcpStream,
) -> NodeReport {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[mooncake-bench s5] FATAL PANIC: {info}");
        std::process::abort();
    }));
    fs::create_dir_all(job.nvme_dir_for_node(node_id)).expect("create nvme_dir");

    let per_shard_reqs =
        partition_requests_for_node(reqs, node_id, cfg.shards_per_node, cfg.sticky_pct);
    let init_done_barrier = Arc::new(Barrier::new(cfg.shards_per_node + 1));
    let start_go_barrier = Arc::new(Barrier::new(cfg.shards_per_node + 1));

    let node_init_t0 = Instant::now();
    let mut handles = Vec::with_capacity(cfg.shards_per_node);
    for (shard_id, ((req_sock, srv_sock), s_reqs)) in shard_socks
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
                node_id, shard_id, gpu_id, &cfg_copy, &job_clone, req_sock, srv_sock, s_reqs,
                b_done, b_go,
            )
        }));
    }

    init_done_barrier.wait();
    let init_time_s = node_init_t0.elapsed().as_secs_f64();

    if node_id == 0 {
        let ready = recv_frame(coord);
        assert_eq!(ready, b"READY");
        send_frame(coord, b"GO");
    } else {
        send_frame(coord, b"READY");
        let go = recv_frame(coord);
        assert_eq!(go, b"GO");
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
) -> (Vec<(TcpStream, TcpStream)>, TcpStream) {
    listener.set_nonblocking(true).expect("set_nonblocking");
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut req_slots: Vec<Option<TcpStream>> = (0..shards_per_node).map(|_| None).collect();
    let mut srv_slots: Vec<Option<TcpStream>> = (0..shards_per_node).map(|_| None).collect();
    let mut coord_slot: Option<TcpStream> = None;
    let expected = shards_per_node * 2 + 1;
    let mut accepted = 0usize;

    while accepted < expected {
        match listener.accept() {
            Ok((mut stream, _)) => {
                configure_sock(&stream);
                let hello = recv_frame(&mut stream);
                assert_eq!(hello.len(), 17, "bad hello len {}", hello.len());
                assert_eq!(&hello[..16], &session.as_bytes()[..16], "session mismatch");
                let ch_id = hello[16];
                send_frame(&mut stream, &session.as_bytes()[16..]);
                if ch_id == COORD_CHANNEL_ID {
                    coord_slot = Some(stream);
                } else {
                    let s = (ch_id / 2) as usize;
                    assert!(s < shards_per_node, "invalid shard channel {ch_id}");
                    if ch_id % 2 == 0 {
                        req_slots[s] = Some(stream);
                    } else {
                        srv_slots[s] = Some(stream);
                    }
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

    let shards = req_slots
        .into_iter()
        .zip(srv_slots.into_iter())
        .map(|(r, s)| (r.expect("missing req sock"), s.expect("missing srv sock")))
        .collect();
    let coord = coord_slot.expect("missing coord sock");
    (shards, coord)
}

fn connect_channel(addr: &str, session: &str, ch_id: u8) -> TcpStream {
    let mut stream = TcpStream::connect(addr).expect("connect to node0");
    configure_sock(&stream);
    let mut hello = Vec::with_capacity(17);
    hello.extend_from_slice(&session.as_bytes()[..16]);
    hello.push(ch_id);
    send_frame(&mut stream, &hello);
    let ack = recv_frame(&mut stream);
    assert_eq!(ack, &session.as_bytes()[16..], "session ack mismatch");
    stream
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

    let (shard_socks, mut coord) =
        accept_all_node1_channels(&listener, &session, cfg.shards_per_node, &mut node1_child);

    let r0 = run_node(0, job.donor_gpu, job, &cfg, &reqs, shard_socks, &mut coord);

    let r1_bytes = recv_frame(&mut coord);
    let r1: NodeReport = serde_json::from_slice(&r1_bytes).expect("parse node1 report");

    let output = node1_child.wait_with_output().expect("wait node1 child");
    assert!(
        output.status.success(),
        "node1 child failed with status {}",
        output.status
    );

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
        "driver": "rust_cpp_native",
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

    let mut rec = MetricRecord::from_job("mooncake", job);
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

    let mut shard_socks = Vec::with_capacity(cfg.shards_per_node);
    for s in 0..cfg.shards_per_node {
        // Channel (s*2) is node0's req_sock -> node1's srv_sock!
        let srv = connect_channel(addr, sess, (s * 2) as u8);
        // Channel (s*2 + 1) is node0's srv_sock -> node1's req_sock!
        let req = connect_channel(addr, sess, (s * 2 + 1) as u8);
        shard_socks.push((req, srv));
    }
    let mut coord = connect_channel(addr, sess, COORD_CHANNEL_ID);

    let report = run_node(1, job.initiator_gpu, job, &cfg, &reqs, shard_socks, &mut coord);
    let payload = serde_json::to_vec(&report).expect("serialize node1 report");
    send_frame(&mut coord, &payload);
}
