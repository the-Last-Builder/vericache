use crate::ffi::{CudaDevBuf, CudaPinnedBuf, TransferEngine, cuda_device_sync, cuda_set_device};
use crate::schema::{JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
struct DonorInfo {
    seg_name: String,
    ptr: u64,
    total_bytes: usize,
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

fn use_gpu_for_case(case: &str) -> bool {
    matches!(
        case,
        "2b_gdr_gpu"
            | "2c_ablation_raw_mw"
            | "2c_ablation_dynamic_single"
            | "2c_ablation_dynamic_pipelined"
    )
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
        .expect("spawn s2 donor child");

    let (mut sock, _) = listener.accept().expect("accept donor");
    sock.set_nodelay(true).expect("nodelay");
    sock.set_read_timeout(Some(Duration::from_secs(120))).ok();

    let info_bytes = recv_frame(&mut sock);
    let peer: DonorInfo = serde_json::from_slice(&info_bytes).expect("parse DonorInfo");

    let use_gpu = use_gpu_for_case(&job.case);
    let frame_size = job.chunk_bytes;
    let qd = job.concurrency.max(1);
    let warmup = job.warmup_iters;
    let iters = job.iters;
    let total_bytes = frame_size * qd;

    let te = TransferEngine::new(&job.rdma_dev).expect("initiator TransferEngine");

    let (_gpu_buf, _host_buf, local_base) = if use_gpu {
        cuda_set_device(job.initiator_gpu);
        let b = CudaDevBuf::alloc(total_bytes);
        b.fill(0x00);
        cuda_device_sync();
        let addr = b.addr();
        (Some(b), None, addr)
    } else {
        let mut b = CudaPinnedBuf::alloc(total_bytes);
        b.fill(0x00);
        let addr = b.addr();
        (None, Some(b), addr)
    };

    te.register_memory(local_base, total_bytes)
        .expect("initiator register_memory");
    let seg_id = te
        .open_segment(&peer.seg_name)
        .expect("initiator open_segment");

    let local_ptrs: Vec<u64> = (0..qd)
        .map(|i| local_base + (i * frame_size) as u64)
        .collect();
    let remote_ptrs: Vec<u64> = (0..qd)
        .map(|i| peer.ptr + (i * frame_size) as u64)
        .collect();
    let lens: Vec<usize> = vec![frame_size; qd];

    let warmup_batches = warmup.div_ceil(qd);
    let meas_batches = iters.div_ceil(qd);
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler: Option<ProcSampler> = None;
    let mut t_meas_start = Instant::now();
    let mut total_ops = 0u64;

    for b in 0..total_batches {
        if b == warmup_batches {
            sampler = Some(ProcSampler::start());
            t_meas_start = Instant::now();
        }
        let t0 = Instant::now();
        let rc = if qd == 1 {
            te.sync_read(seg_id, local_base, peer.ptr, frame_size)
        } else {
            te.batch_sync_read(seg_id, &local_ptrs, &remote_ptrs, &lens)
        };
        rc.unwrap_or_else(|e| {
            panic!("Mooncake S2 RDMA read failed rc={e} (qd={qd}, len={frame_size})")
        });
        let dt_ns = t0.elapsed().as_nanos() as u64;
        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += qd as u64;
        }
    }

    let elapsed_s = t_meas_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.unwrap().finish();
    let total_xfer = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_xfer, elapsed_s);

    let _ = te.unregister_memory(local_base);
    send_frame(&mut sock, b"DONE");

    let output = donor_child.wait_with_output().expect("wait donor child");
    assert!(
        output.status.success(),
        "Mooncake S2 donor failed: {}",
        output.status
    );

    let mut rec = MetricRecord::from_job("mooncake", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    let api_name = if qd > 1 {
        "batch_submitTransfer(READ)"
    } else {
        "submitTransfer(READ)"
    };
    let tier_desc = if use_gpu {
        format!("GPU{}->GPU{} GDR", job.donor_gpu, job.initiator_gpu)
    } else {
        "Host->Host RDMA".to_string()
    };
    rec.notes = format!(
        "Mooncake 0.3.14 native C++ TransferEngine {api_name} ({tier_desc}, static MR rkey, QD={qd}, {})",
        job.rdma_dev
    );
    vec![rec]
}

pub fn run_role(job: &JobSpec, role: &str, peer_addr: Option<&str>, _session: Option<&str>) {
    assert_eq!(role, "donor", "s2 child role must be donor");
    let addr = peer_addr.expect("peer_addr");
    let use_gpu = use_gpu_for_case(&job.case);
    let frame_size = job.chunk_bytes;
    let qd = job.concurrency.max(1);
    let total_bytes = frame_size * qd;

    let te = TransferEngine::new(&job.rdma_dev).expect("donor TransferEngine");
    let seg_name = te.segment_name();

    let (_gpu_buf, _host_buf, ptr) = if use_gpu {
        cuda_set_device(job.donor_gpu);
        let b = CudaDevBuf::alloc(total_bytes);
        b.fill(0x5A);
        cuda_device_sync();
        let a = b.addr();
        (Some(b), None, a)
    } else {
        let mut b = CudaPinnedBuf::alloc(total_bytes);
        b.fill(0x5A);
        let a = b.addr();
        (None, Some(b), a)
    };

    te.register_memory(ptr, total_bytes)
        .expect("donor register_memory");

    let mut sock = TcpStream::connect(addr).expect("connect to initiator");
    sock.set_nodelay(true).expect("nodelay");
    let info = DonorInfo {
        seg_name,
        ptr,
        total_bytes,
    };
    let payload = serde_json::to_vec(&info).expect("serialize DonorInfo");
    send_frame(&mut sock, &payload);

    let done = recv_frame(&mut sock);
    assert_eq!(done, b"DONE");
    let _ = te.unregister_memory(ptr);
}
