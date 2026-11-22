use crate::policy::LruPolicy;
use crate::schema::{bytes_per_sec_to_gbps, JobSpec, LatencyHist, MetricRecord, ProcSampler};
use crate::shard::{key_from_hash_id, stamp_checksummed_frame, verify_checksummed_frame};
use crate::suites::rdma_ablation::{
    build_distributed_engine, connect_peer, open_rdma_device, TcpTransport,
};
use crate::suites::trace_replay::load_trace;
use std::collections::HashSet;
use std::io::Read;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use transfer::context::NodeContext;
use transfer::rdma::ProtectionDomain;
use transfer::transport::ByteTransport;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, ServeOutcome};
use vericache_core::backends::rdma::protocol::ControlPayload;
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::engine::DistributedEvent;
use vericache_core::kernel::CacheCore;
use vericache_core::spec::{CoreError, FrameId, ObjectKey, BACKEND_RDMA};
use vstd::prelude::Tracked;

const IBV_WC_REM_ACCESS_ERR: i32 = 10;

pub fn run(job: &JobSpec) -> Vec<MetricRecord> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind listener");
    let addr = listener.local_addr().unwrap().to_string();
    let mut secret = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .expect("urandom")
        .read_exact(&mut secret)
        .expect("read secret");
    let session: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    let job_json = serde_json::to_string(job).expect("serialize job");

    let exe = std::env::current_exe().expect("current_exe");
    let mut init_child: Child = Command::new(&exe)
        .args([
            "--job-json",
            &job_json,
            "--role",
            "initiator",
            "--peer-addr",
            &addr,
            "--session",
            &session,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn s4 initiator");

    listener.set_nonblocking(true).expect("nonblocking");
    let deadline = Instant::now() + Duration::from_secs(30);
    let transport = loop {
        match listener.accept() {
            Ok((stream, _)) => {
                let mut t = TcpTransport::new(stream);
                let got = t.recv_frame().expect("handshake recv");
                assert_eq!(got, session.as_bytes()[..16]);
                t.send_frame(&session.as_bytes()[16..])
                    .expect("handshake send");
                break t;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if let Some(st) = init_child.try_wait().expect("try_wait") {
                    panic!("s4 initiator exited early: {st}");
                }
                assert!(Instant::now() < deadline, "s4 accept timeout");
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(e) => panic!("accept: {e}"),
        }
    };

    run_donor(job, transport);

    let output = init_child.wait_with_output().expect("wait s4 initiator");
    assert!(
        output.status.success(),
        "s4 initiator failed: {}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut records = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('{') {
            if let Ok(rec) = serde_json::from_str::<MetricRecord>(trimmed) {
                records.push(rec);
            }
        }
    }
    assert!(!records.is_empty(), "s4 initiator produced no MetricRecord");
    records
}

pub fn run_role(job: &JobSpec, role: &str, peer_addr: Option<&str>, session: Option<&str>) {
    assert_eq!(role, "initiator");
    let transport = connect_peer(peer_addr.expect("peer_addr"), session.expect("session"));
    let rec = run_initiator(job, transport);
    println!("{}", serde_json::to_string(&rec).unwrap());
}

/// Extract a deterministic sequence of hot block `hash_id`s from `toolagent_trace.jsonl`
/// and compute tight 15% donor capacity.
fn build_hotspot_workload(job: &JobSpec) -> (Vec<u64>, usize) {
    let trace_path = job
        .trace_path
        .as_deref()
        .unwrap_or("/workspace/target/bench-traces/toolagent_trace.jsonl");
    let reqs = load_trace(trace_path, job.max_requests.or(Some(256)));
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
        .unwrap_or_else(|| ((unique.len() as f64) * ratio).round() as usize)
        .clamp(16, 512);
    let total_ops = job.iters.max(200);
    let mut ops = Vec::with_capacity(total_ops);
    for i in 0..total_ops {
        ops.push(seq[i % seq.len()]);
    }
    (ops, cap_frames)
}

fn should_revoke(op_idx: usize, revoke_rate: f64) -> bool {
    if revoke_rate <= 0.0 {
        return false;
    }
    let period = (1.0 / revoke_rate).round().max(1.0) as usize;
    (op_idx + 1) % period == 0
}

fn make_core(node_id: u64, frame_size: usize) -> CacheCore {
    let (_inst, _alloc, _ev, _sh, _vi, _vs) =
        NodeContext::bootstrap(node_id).into_sources_with_versions();
    let mut core = CacheCore::new(
        node_id as u16,
        frame_size,
        Vec::new(),
        64,
        Tracked::assume_new(),
    );
    core.enable_version_registry(Tracked::assume_new(), Tracked::assume_new());
    core
}

fn run_donor(job: &JobSpec, transport: TcpTransport) {
    let frame_size = job.chunk_bytes;
    let revoke_rate = job.revoke_rate.unwrap_or(0.01);
    let (ops, donor_cap) = build_hotspot_workload(job);

    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, Tracked::assume_new()).expect("PD");
    let mut core = make_core(0, frame_size);
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        donor_cap as u32,
        4096,
    )
    .expect("donor HostBackend");
    let mut engine = build_distributed_engine(
        0,
        frame_size,
        donor_cap as u32,
        8,
        job.gid_index,
        &device,
        &pd,
        transport,
        &host,
        None,
        core,
    );
    let host_tier = engine.host_tier();
    let mut lru: LruPolicy<ObjectKey, FrameId> = LruPolicy::new(donor_cap);
    let mut buf = vec![0u8; frame_size];
    let mut version_counter: u64 = 1;
    let mut psn_counter: u32 = 0x1000;

    // Helper closure: ensure `key` is resident in donor's tight 15%-capacity LRU slab with checksummed payload.
    let ensure_resident = |engine: &mut vericache_core::engine::DistributedEngine<'_, TcpTransport>,
                           lru: &mut LruPolicy<ObjectKey, FrameId>,
                           buf: &mut [u8],
                           version_counter: &mut u64,
                           hid: u64| {
        let key = key_from_hash_id(hid);
        if lru.contains(&key) {
            lru.touch(&key);
            return;
        }
        if lru.is_full() {
            if let Some((_vk, vf)) = lru.pop_lru() {
                engine.evict_frame(vf).expect("donor evict lru");
                let r = engine.reclaim_draining(host_tier, 8);
                assert!(r >= 1, "donor reclaim failed");
            }
        }
        let ver = *version_counter;
        *version_counter += 1;
        stamp_checksummed_frame(buf, hid, ver);
        let (_v, frame) = engine
            .put_host(Tracked::assume_new(), key, buf)
            .expect("donor put_host");
        let _ = lru.insert(key, frame);
    };

    for (op_idx, &hid) in ops.iter().enumerate() {
        ensure_resident(&mut engine, &mut lru, &mut buf, &mut version_counter, hid);
        let key = key_from_hash_id(hid);
        let req_id = match engine
            .poll_peer_event(Tracked::assume_new())
            .expect("donor poll FetchRequest")
        {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert_eq!(k, key);
                request_id
            }
            _ => panic!("donor expected FetchRequest at op {op_idx}"),
        };

        let grant_tag = match engine
            .serve_fetch_request(
                host_tier,
                req_id,
                key,
                Tracked::assume_new(),
                Tracked::assume_new(),
            )
            .expect("donor serve_fetch_request")
        {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed resident key"),
        };

        if should_revoke(op_idx, revoke_rate) {
            // Option-A unilateral donor revocation: LOCAL_INV + ibv_dealloc_mw -> Quarantined + RevokeNotice.
            let mw_slot = engine
                .revoke_lease_unilateral(grant_tag, Tracked::assume_new())
                .expect("donor revoke_lease_unilateral");

            // Immediately evict, reclaim, and overwrite that frame with a conflicting hot key to
            // prove fenced hardware revocation prevents ANY torn/stale read!
            if let Some(evicted_f) = lru.remove(&key) {
                engine.evict_frame(evicted_f).expect("evict revoked frame");
                let r = engine.reclaim_draining(host_tier, 8);
                assert_eq!(r, 1, "revoked frame must reclaim immediately after LOCAL_INV");
                let conflict_hid = hid.wrapping_add(0xDEAD_BEEF);
                let ver = version_counter;
                version_counter += 1;
                stamp_checksummed_frame(&mut buf, conflict_hid, ver);
                let conflict_key = key_from_hash_id(conflict_hid);
                let (_cv, cf) = engine
                    .put_host(Tracked::assume_new(), conflict_key, &buf)
                    .expect("overwrite reclaimed donor frame");
                let _ = lru.insert(conflict_key, cf);
            }

            // Drain initiator's post-abort ReadRelease.
            match engine
                .poll_peer_event(Tracked::assume_new())
                .expect("donor poll post-abort ReadRelease")
            {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert_eq!(tag, grant_tag);
                    assert!(!was_active);
                }
                _ => panic!("donor expected post-abort ReadRelease"),
            }

            // Option-A QP recovery: Error -> Reset -> Reconnect -> Replenish MW.
            let _err_w = engine.enter_qp_error().expect("donor enter_qp_error");
            psn_counter = (psn_counter + 7) & 0x00ff_ffff;
            engine
                .reset_and_reconnect(psn_counter, Tracked::assume_new())
                .expect("donor reset_and_reconnect");
            engine
                .replenish_window(mw_slot, &pd, Tracked::assume_new())
                .expect("donor replenish_window");

            // Re-populate `hid` and serve the clean retry fetch!
            ensure_resident(&mut engine, &mut lru, &mut buf, &mut version_counter, hid);
            let retry_req_id = match engine
                .poll_peer_event(Tracked::assume_new())
                .expect("donor poll retry FetchRequest")
            {
                DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                    assert_eq!(k, key);
                    request_id
                }
                _ => panic!("donor expected retry FetchRequest"),
            };
            let retry_tag = match engine
                .serve_fetch_request(
                    host_tier,
                    retry_req_id,
                    key,
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                )
                .expect("donor serve retry")
            {
                ServeOutcome::Granted { grant_tag, .. } => grant_tag,
                ServeOutcome::Miss => panic!("donor missed retry key"),
            };
            match engine
                .poll_peer_event(Tracked::assume_new())
                .expect("donor poll retry ReadRelease")
            {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert_eq!(tag, retry_tag);
                    assert!(was_active);
                }
                _ => panic!("donor expected retry ReadRelease"),
            }
        } else {
            // Normal cooperative release.
            match engine
                .poll_peer_event(Tracked::assume_new())
                .expect("donor poll ReadRelease")
            {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert_eq!(tag, grant_tag);
                    assert!(was_active);
                }
                _ => panic!("donor expected cooperative ReadRelease"),
            }
        }
    }
}

fn run_initiator(job: &JobSpec, transport: TcpTransport) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let revoke_rate = job.revoke_rate.unwrap_or(0.01);
    let (ops, donor_cap) = build_hotspot_workload(job);

    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, Tracked::assume_new()).expect("PD");
    let mut core = make_core(1, frame_size);
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        8,
        4096,
    )
    .expect("initiator HostBackend");
    let mut engine = build_distributed_engine(
        1,
        frame_size,
        8,
        8,
        job.gid_index,
        &device,
        &pd,
        transport,
        &host,
        None,
        core,
    );
    let host_tier = engine.host_tier();

    let mut read_buf = vec![0u8; frame_size];
    let mut op_hist = LatencyHist::new();
    let mut recov_hist = LatencyHist::new();
    let mut corrupted_frames: u64 = 0;
    let mut revoke_count: u64 = 0;
    let mut rem_access_errs: u64 = 0;
    let mut valid_fetches: u64 = 0;
    let mut total_wire_attempts: u64 = 0;
    let mut psn_counter: u32 = 0x2000;

    let sampler = ProcSampler::start();
    let run_start = Instant::now();

    for (op_idx, &hid) in ops.iter().enumerate() {
        let key = key_from_hash_id(hid);
        let t_op = Instant::now();

        if should_revoke(op_idx, revoke_rate) {
            revoke_count += 1;
            total_wire_attempts += 1;
            let _req_id = engine
                .send_fetch_request(key, Tracked::assume_new())
                .expect("send_fetch_request");
            let grant = match engine
                .poll_peer_event(Tracked::assume_new())
                .expect("poll ReadGrant")
            {
                DistributedEvent::ReadGrant(g) => g,
                _ => panic!("expected ReadGrant"),
            };
            match engine
                .poll_peer_event(Tracked::assume_new())
                .expect("poll RevokeNotice")
            {
                DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                    assert_eq!(grant_tag, grant.tag);
                }
                _ => panic!("expected RevokeNotice"),
            }

            // Attempt RDMA_READ against unilaterally revoked MW -> hardware IBV_WC_REM_ACCESS_ERR + abort_write!
            let t_recov = Instant::now();
            let pull_res = engine.pull_with_grant(
                key,
                host_tier,
                PullTarget::Host,
                grant,
                Tracked::assume_new(),
            );
            match pull_res {
                Err(CoreError::Backend {
                    kind: BACKEND_RDMA,
                    code,
                }) => {
                    assert_eq!(
                        code, IBV_WC_REM_ACCESS_ERR,
                        "expected IBV_WC_REM_ACCESS_ERR (10), got {code}"
                    );
                    rem_access_errs += 1;
                }
                other => panic!("expected IBV_WC_REM_ACCESS_ERR on revoked MW, got {other:?}"),
            }
            // Frame MUST NOT be published after abort_write.
            assert!(
                engine.lookup_tier(key, host_tier).is_none(),
                "aborted write must never be published"
            );

            // Option-A QP recovery: Error -> Reset -> Reconnect.
            let _err_w = engine.enter_qp_error().expect("init enter_qp_error");
            psn_counter = (psn_counter + 7) & 0x00ff_ffff;
            engine
                .reset_and_reconnect(psn_counter, Tracked::assume_new())
                .expect("init reset_and_reconnect");
            let recov_ns = t_recov.elapsed().as_nanos() as u64;
            recov_hist.record_ns(recov_ns);

            // Clean retry fetch over reconnected QP + full checksum verification.
            total_wire_attempts += 1;
            let FetchOutcome::Hit { frame, .. } = engine
                .fetch_remote_frame(key, host_tier, PullTarget::Host, Tracked::assume_new())
                .expect("retry fetch_remote_frame")
            else {
                panic!("expected retry Hit");
            };
            let got = engine
                .get_into_host(
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    key,
                    &mut read_buf,
                )
                .expect("get_into_host retry");
            assert!(got.is_some());
            if verify_checksummed_frame(&read_buf, Some(hid)).is_err() {
                corrupted_frames += 1;
            } else {
                valid_fetches += 1;
            }
            engine.evict_frame(frame).expect("evict retry frame");
            assert_eq!(engine.reclaim_draining(host_tier, 8), 1);
        } else {
            total_wire_attempts += 1;
            let FetchOutcome::Hit { frame, .. } = engine
                .fetch_remote_frame(key, host_tier, PullTarget::Host, Tracked::assume_new())
                .expect("fetch_remote_frame")
            else {
                panic!("expected Hit");
            };
            let got = engine
                .get_into_host(
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    key,
                    &mut read_buf,
                )
                .expect("get_into_host");
            assert!(got.is_some());
            if verify_checksummed_frame(&read_buf, Some(hid)).is_err() {
                corrupted_frames += 1;
            } else {
                valid_fetches += 1;
            }
            engine.evict_frame(frame).expect("evict frame");
            assert_eq!(engine.reclaim_draining(host_tier, 8), 1);
        }

        let op_ns = t_op.elapsed().as_nanos() as u64;
        op_hist.record_ns(op_ns);
    }

    let elapsed_s = run_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let good_bytes = valid_fetches * (frame_size as u64);
    let wire_bytes = total_wire_attempts * (frame_size as u64);

    assert_eq!(
        corrupted_frames, 0,
        "VeriCache Option-A revocation MUST guarantee corrupted_frames == 0"
    );

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.num_blocks = donor_cap;
    rec.throughput_gbps = bytes_per_sec_to_gbps(wire_bytes, elapsed_s);
    rec.goodput_gbps = bytes_per_sec_to_gbps(good_bytes, elapsed_s);
    rec.p50_us = op_hist.p50_us();
    rec.p95_us = op_hist.p95_us();
    rec.p99_us = op_hist.p99_us();
    rec.hit_rate = (valid_fetches as f64) / (ops.len() as f64);
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.corrupted_frames = corrupted_frames;
    rec.recovery_us = recov_hist.mean_us();
    rec.notes = format!(
        "Option-A unilateral revoke (rate={:.3}, revokes={}, IBV_WC_REM_ACCESS_ERR={}, recov_p50_us={:.1}, recov_p99_us={:.1}, donor_cap_15pct={})",
        revoke_rate,
        revoke_count,
        rem_access_errs,
        recov_hist.p50_us(),
        recov_hist.p99_us(),
        donor_cap
    );
    rec
}
