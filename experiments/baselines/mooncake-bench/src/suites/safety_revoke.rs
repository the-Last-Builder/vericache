use crate::ffi::{CudaPinnedBuf, TransferEngine};
use crate::policy::LruPolicy;
use crate::schema::{JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps};
use crate::trace::{load_trace, should_revoke, stamp_checksummed_frame, verify_checksummed_frame};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
struct S4DonorHello {
    seg_name: String,
    slab_ptr: u64,
    donor_cap: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct S4Grant {
    addr: u64,
    hid: u64,
    evict: bool,
}

fn send_frame(sock: &mut TcpStream, data: &[u8]) {
    let n = data.len() as u32;
    sock.write_all(&n.to_be_bytes()).expect("write len");
    sock.write_all(data).expect("write payload");
}

fn recv_frame(sock: &mut TcpStream) -> Vec<u8> {
    let mut len_b = [0u8; 4];
    sock.read_exact(&mut len_b).expect("read len");
    let n = u32::from_be_bytes(len_b) as usize;
    let mut buf = vec![0u8; n];
    sock.read_exact(&mut buf).expect("read payload");
    buf
}

fn build_s4_workload(job: &JobSpec) -> (Vec<u64>, usize) {
    let trace_path = job
        .trace_path
        .as_deref()
        .unwrap_or("/workspace/target/bench-traces/toolagent_trace.jsonl");
    let max_reqs = job.max_requests.or(Some(128));
    let reqs = load_trace(trace_path, max_reqs);
    let mut unique = HashSet::new();
    let mut seq = Vec::new();
    for r in &reqs {
        for &h in &r.hash_ids {
            unique.insert(h);
            seq.push(h);
        }
    }
    let ratio = job.capacity_ratio.unwrap_or(0.15);
    let cap_frames = job
        .capacity_frames
        .unwrap_or_else(|| ((unique.len() as f64 * ratio).round() as usize).clamp(16, 512));
    let total_ops = job.iters.max(200);
    let ops = (0..total_ops).map(|i| seq[i % seq.len()]).collect();
    (ops, cap_frames)
}

pub fn run(job: &JobSpec) -> Vec<MetricRecord> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
    let addr = listener.local_addr().unwrap().to_string();
    let job_json = serde_json::to_string(job).expect("serialize job");
    let exe = std::env::current_exe().expect("current_exe");

    let donor_child: Child = Command::new(&exe)
        .args([
            "--job-json",
            &job_json,
            "--role",
            "donor",
            "--peer-addr",
            &addr,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn s4 donor child");

    let (mut sock, _) = listener.accept().expect("accept donor");
    sock.set_nodelay(true).expect("nodelay");
    sock.set_read_timeout(Some(Duration::from_secs(120))).ok();

    let hello_bytes = recv_frame(&mut sock);
    let hello: S4DonorHello = serde_json::from_slice(&hello_bytes).expect("parse S4DonorHello");

    let frame_size = job.chunk_bytes;
    let revoke_rate = job.revoke_rate.unwrap_or(0.01);
    let (ops, donor_cap) = build_s4_workload(job);
    assert_eq!(donor_cap, hello.donor_cap);

    let te = TransferEngine::new(&job.rdma_dev).expect("s4 initiator TransferEngine");
    let local_buf = CudaPinnedBuf::alloc(frame_size);
    let local_ptr = local_buf.addr();
    te.register_memory(local_ptr, frame_size)
        .expect("s4 initiator register_memory");
    let seg_id = te
        .open_segment(&hello.seg_name)
        .expect("s4 initiator open_segment");

    let mut hist = LatencyHist::new();
    let sampler = ProcSampler::start();
    let t_start = Instant::now();

    let mut corrupted = 0u64;
    let mut valid = 0u64;
    let mut forced_overwrites = 0u64;

    for &expected_hid in &ops {
        let gb = recv_frame(&mut sock);
        let grant: S4Grant = serde_json::from_slice(&gb).expect("parse S4Grant");
        if grant.evict {
            forced_overwrites += 1;
            thread::sleep(Duration::from_micros(50));
        }
        let t0 = Instant::now();
        te.sync_read(seg_id, local_ptr, grant.addr, frame_size)
            .expect("Mooncake S4 RDMA read failed");
        hist.record_ns(t0.elapsed().as_nanos() as u64);

        if verify_checksummed_frame(local_buf.as_u64_slice(), expected_hid) {
            valid += 1;
        } else {
            corrupted += 1;
        }
        send_frame(&mut sock, b"OK");
    }

    let elapsed_s = t_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let _ = te.unregister_memory(local_ptr);

    let output = donor_child.wait_with_output().expect("wait s4 donor");
    assert!(
        output.status.success(),
        "Mooncake S4 donor failed: {}",
        output.status
    );

    let mut rec = MetricRecord::from_job("mooncake", job);
    rec.num_blocks = donor_cap;
    rec.throughput_gbps = bytes_per_sec_to_gbps((ops.len() * frame_size) as u64, elapsed_s);
    rec.goodput_gbps = bytes_per_sec_to_gbps(valid * (frame_size as u64), elapsed_s);
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = (valid as f64) / (ops.len() as f64);
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.corrupted_frames = corrupted;
    rec.recovery_us = 0.0;
    rec.notes = format!(
        "Native C++ TransferEngine Static-MR rkey (no Type-2B MW revoke): forced_evict_overwrites={forced_overwrites}, corrupted_silent_reads={corrupted}/{} (rate={revoke_rate:.3}, donor_cap_15pct={donor_cap})",
        ops.len()
    );
    vec![rec]
}

pub fn run_role(job: &JobSpec, role: &str, peer_addr: Option<&str>, _session: Option<&str>) {
    assert_eq!(role, "donor", "s4 child role must be donor");
    let addr = peer_addr.expect("peer_addr");
    let frame_size = job.chunk_bytes;
    let revoke_rate = job.revoke_rate.unwrap_or(0.01);
    let (ops, donor_cap) = build_s4_workload(job);

    let te = TransferEngine::new(&job.rdma_dev).expect("s4 donor TransferEngine");
    let seg_name = te.segment_name();

    let mut slab = CudaPinnedBuf::alloc(donor_cap * frame_size);
    let slab_ptr = slab.addr();
    te.register_memory(slab_ptr, donor_cap * frame_size)
        .expect("s4 donor register_memory");

    let mut sock = TcpStream::connect(addr).expect("connect to s4 initiator");
    sock.set_nodelay(true).expect("nodelay");
    let hello = S4DonorHello {
        seg_name,
        slab_ptr,
        donor_cap,
    };
    send_frame(
        &mut sock,
        &serde_json::to_vec(&hello).expect("serialize hello"),
    );

    let u64_per_frame = frame_size / 8;
    let mut lru: LruPolicy<u64, usize> = LruPolicy::new(donor_cap);
    let mut free_slots: Vec<usize> = (0..donor_cap).collect();
    let mut version_counter = 1u64;

    for (op_idx, &hid) in ops.iter().enumerate() {
        let slot = if let Some(s) = lru.touch(&hid) {
            s
        } else {
            let s = if let Some(fs) = free_slots.pop() {
                fs
            } else {
                lru.pop_lru().expect("lru non-empty").1
            };
            let slice = &mut slab.as_u64_slice_mut()[s * u64_per_frame..(s + 1) * u64_per_frame];
            stamp_checksummed_frame(slice, hid, version_counter);
            version_counter += 1;
            lru.insert(hid, s);
            s
        };

        let remote_addr = slab_ptr + (slot * frame_size) as u64;
        let will_evict = should_revoke(op_idx, revoke_rate);
        let grant = S4Grant {
            addr: remote_addr,
            hid,
            evict: will_evict,
        };
        send_frame(
            &mut sock,
            &serde_json::to_vec(&grant).expect("serialize grant"),
        );

        if will_evict {
            let conflict_hid = hid.wrapping_add(0xDEAD_BEEF);
            lru.remove(&hid);
            let slice =
                &mut slab.as_u64_slice_mut()[slot * u64_per_frame..(slot + 1) * u64_per_frame];
            stamp_checksummed_frame(slice, conflict_hid, version_counter);
            version_counter += 1;
            lru.insert(conflict_hid, slot);
        }

        let ack = recv_frame(&mut sock);
        assert_eq!(ack, b"OK");
    }

    let _ = te.unregister_memory(slab_ptr);
}
