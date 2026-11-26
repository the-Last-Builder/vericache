use crate::policy::LruPolicy;
use crate::schema::{JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps};
use crate::shard::{
    ShardCapabilityBundle, key_from_hash_id, stamp_checksummed_frame, verify_checksummed_frame,
};
use crate::suites::rdma_ablation::{
    TcpTransport, build_distributed_engine, connect_peer, open_rdma_device,
};
use crate::suites::trace_replay::load_trace;
use std::collections::HashSet;
use std::io::Read;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use transfer::rdma::ProtectionDomain;
use transfer::transport::ByteTransport;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, ServeOutcome};
use vericache_core::backends::rdma::protocol::ControlPayload;
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::engine::{DistributedEngine, DistributedEvent};
use vericache_core::spec::{BACKEND_RDMA, CoreError, FrameId, ObjectKey, TierId};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Tracked, verus};

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

fn build_revoke_schedule(total_ops: usize, revoke_rate: f64) -> Vec<bool> {
    if revoke_rate <= 0.0 {
        return vec![false; total_ops];
    }
    let period = (1.0 / revoke_rate).round().max(1.0) as usize;
    (0..total_ops).map(|idx| (idx + 1) % period == 0).collect()
}

verus! {

fn ensure_donor_resident(
    caps: &ShardCapabilityBundle,
    engine: &mut DistributedEngine<'_, TcpTransport>,
    host_tier: TierId,
    lru: &mut LruPolicy<ObjectKey, FrameId>,
    buf: &mut [u8],
    version_counter: &mut u64,
    hid: u64,
) {
    let key = key_from_hash_id(hid);
    if lru.contains(&key) {
        lru.touch(&key);
        return;
    }
    if lru.is_full() {
        if let Some((_vk, vf)) = lru.pop_lru() {
            engine.evict_frame(vf).expect("donor evict lru");
            let r = engine.reclaim_draining(host_tier, 8).unwrap();
            assert!(r >= 1, "donor reclaim failed");
        }
    }
    let ver = *version_counter;
    *version_counter += 1;
    stamp_checksummed_frame(buf, hid, ver);
    let (_v, frame) = engine
        .put_host(Tracked(caps.instance.borrow()), key, buf)
        .expect("donor put_host");
    let _ = lru.insert(key, frame);
}

fn run_donor_inner(
    job: &JobSpec,
    transport: TcpTransport,
    ops: &[u64],
    revoke_flags: &[bool],
    donor_cap: usize,
) {
    let frame_size = job.chunk_bytes;
    let mut caps = ShardCapabilityBundle::take(0);
    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut core = caps.make_versioned_core(frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        donor_cap as u32,
        4096,
    )
    .expect("donor HostBackend");
    let mut engine = build_distributed_engine(
        &mut caps,
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

    let mut op_idx = 0usize;
    while op_idx < ops.len() {
        let hid = ops[op_idx];
        ensure_donor_resident(
            &caps,
            &mut engine,
            host_tier,
            &mut lru,
            &mut buf,
            &mut version_counter,
            hid,
        );
        let key = key_from_hash_id(hid);
        let req_id = match engine
            .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
            .expect("donor poll FetchRequest")
        {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest"),
        };

        let grant_tag = match engine
            .serve_fetch_request_for_peer(0, host_tier,
                req_id,
                key,
                Tracked(caps.shares.borrow_mut()),
                Tracked(caps.instance.borrow()),
            )
            .expect("donor serve_fetch_request")
        {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed resident key"),
        };

        if revoke_flags[op_idx] {
            // Option-A unilateral donor revocation: LOCAL_INV + ibv_dealloc_mw -> Quarantined + RevokeNotice.
            let mw_slot = engine
                .revoke_peer_lease_unilateral(0, grant_tag, Tracked(caps.instance.borrow()))
                .expect("donor revoke_lease_unilateral");

            // Immediately evict, reclaim, and overwrite that frame with a conflicting hot key to
            // prove fenced hardware revocation prevents ANY torn/stale read!
            if let Some(evicted_f) = lru.remove(&key) {
                engine.evict_frame(evicted_f).expect("evict revoked frame");
                let r = engine.reclaim_draining(host_tier, 8).unwrap();
                assert!(r == 1, "revoked frame must reclaim immediately after LOCAL_INV");
                let conflict_hid = hid.wrapping_add(0xDEAD_BEEF);
                let ver = version_counter;
                version_counter += 1;
                stamp_checksummed_frame(&mut buf, conflict_hid, ver);
                let conflict_key = key_from_hash_id(conflict_hid);
                let (_cv, cf) = engine
                    .put_host(Tracked(caps.instance.borrow()), conflict_key, &buf)
                    .expect("overwrite reclaimed donor frame");
                let _ = lru.insert(conflict_key, cf);
            }

            // Drain initiator's post-abort ReadRelease.
            match engine
                .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
                .expect("donor poll post-abort ReadRelease")
            {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert!(tag == grant_tag);
                    assert!(!was_active);
                },
                _ => panic!("donor expected post-abort ReadRelease"),
            }

            // Option-A QP recovery: Error -> Reset -> Reconnect -> Replenish MW.
            let _err_w = engine.enter_peer_qp_error(0).expect("donor enter_qp_error");
            psn_counter = (psn_counter + 7) & 0x00ff_ffff;
            engine
                .reset_and_reconnect_peer(0, psn_counter, Tracked(caps.instance.borrow()))
                .expect("donor reset_and_reconnect");
            engine
                .replenish_peer_window(0, mw_slot, &pd, caps.mint_rdma())
                .expect("donor replenish_window");

            // Re-populate `hid` and serve the clean retry fetch!
            ensure_donor_resident(
                &caps,
                &mut engine,
                host_tier,
                &mut lru,
                &mut buf,
                &mut version_counter,
                hid,
            );
            let retry_req_id = match engine
                .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
                .expect("donor poll retry FetchRequest")
            {
                DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                    assert!(k == key);
                    request_id
                },
                _ => panic!("donor expected retry FetchRequest"),
            };
            let retry_tag = match engine
                .serve_fetch_request_for_peer(0, host_tier,
                    retry_req_id,
                    key,
                    Tracked(caps.shares.borrow_mut()),
                    Tracked(caps.instance.borrow()),
                )
                .expect("donor serve retry")
            {
                ServeOutcome::Granted { grant_tag, .. } => grant_tag,
                ServeOutcome::Miss => panic!("donor missed retry key"),
            };
            match engine
                .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
                .expect("donor poll retry ReadRelease")
            {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert!(tag == retry_tag);
                    assert!(was_active);
                },
                _ => panic!("donor expected retry ReadRelease"),
            }
        } else {
            // Normal cooperative release.
            match engine
                .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
                .expect("donor poll ReadRelease")
            {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert!(tag == grant_tag);
                    assert!(was_active);
                },
                _ => panic!("donor expected cooperative ReadRelease"),
            }
        }
        op_idx += 1;
    }
}

struct InitiatorRunStats {
    elapsed_s: f64,
    sampler: ProcSampler,
    op_hist: LatencyHist,
    recov_hist: LatencyHist,
    corrupted_frames: u64,
    revoke_count: u64,
    rem_access_errs: u64,
    valid_fetches: u64,
    total_wire_attempts: u64,
}

fn run_initiator_inner(
    job: &JobSpec,
    transport: TcpTransport,
    ops: &[u64],
    revoke_flags: &[bool],
) -> InitiatorRunStats {
    let frame_size = job.chunk_bytes;
    let mut caps = ShardCapabilityBundle::take(1);
    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut core = caps.make_versioned_core(frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        8,
        4096,
    )
    .expect("initiator HostBackend");
    let mut engine = build_distributed_engine(
        &mut caps,
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

    let mut op_idx = 0usize;
    while op_idx < ops.len() {
        let hid = ops[op_idx];
        let key = key_from_hash_id(hid);
        let t_op = Instant::now();

        if revoke_flags[op_idx] {
            revoke_count += 1;
            total_wire_attempts += 1;
            let _req_id = engine
                .send_fetch_request_to(0, key, Tracked(caps.instance.borrow()))
                .expect("send_fetch_request");
            let grant = match engine
                .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
                .expect("poll ReadGrant")
            {
                DistributedEvent::ReadGrant(g) => g,
                _ => panic!("expected ReadGrant"),
            };
            match engine
                .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
                .expect("poll RevokeNotice")
            {
                DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                    assert!(grant_tag == grant.tag());
                },
                _ => panic!("expected RevokeNotice"),
            }

            // Attempt RDMA_READ against unilaterally revoked MW -> hardware IBV_WC_REM_ACCESS_ERR + abort_write!
            let t_recov = Instant::now();
            let pull_res = engine.pull_with_grant_from(0, key,
                host_tier,
                PullTarget::Host,
                grant,
                Tracked(caps.instance.borrow()),
            );
            match pull_res {
                Err(CoreError::Backend {
                    kind: BACKEND_RDMA,
                    code,
                }) => {
                    assert!(
                        code == IBV_WC_REM_ACCESS_ERR,
                        "expected IBV_WC_REM_ACCESS_ERR (10)"
                    );
                    rem_access_errs += 1;
                },
                _ => panic!("expected IBV_WC_REM_ACCESS_ERR on revoked MW"),
            }
            // Frame MUST NOT be published after abort_write.
            assert!(
                engine.lookup_tier(key, host_tier).is_none(),
                "aborted write must never be published"
            );

            // Option-A QP recovery: Error -> Reset -> Reconnect.
            let _err_w = engine.enter_peer_qp_error(0).expect("init enter_qp_error");
            psn_counter = (psn_counter + 7) & 0x00ff_ffff;
            engine
                .reset_and_reconnect_peer(0, psn_counter, Tracked(caps.instance.borrow()))
                .expect("init reset_and_reconnect");
            let recov_ns = t_recov.elapsed().as_nanos() as u64;
            recov_hist.record_ns(recov_ns);

            // Clean retry fetch over reconnected QP + full checksum verification.
            total_wire_attempts += 1;
            let FetchOutcome::Hit { frame, .. } = engine
                .fetch_from_peer(0, key,
                    host_tier,
                    PullTarget::Host,
                    Tracked(caps.instance.borrow()),
                )
                .expect("retry fetch_remote_frame")
            else {
                panic!("expected retry Hit");
            };
            let got = engine
                .get_into_host(
                    Tracked(caps.instance.borrow()),
                    Tracked(caps.shares.borrow_mut()),
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
            assert!(engine.reclaim_draining(host_tier, 8).unwrap() == 1);
        } else {
            total_wire_attempts += 1;
            let FetchOutcome::Hit { frame, .. } = engine
                .fetch_from_peer(0, key,
                    host_tier,
                    PullTarget::Host,
                    Tracked(caps.instance.borrow()),
                )
                .expect("fetch_remote_frame")
            else {
                panic!("expected Hit");
            };
            let got = engine
                .get_into_host(
                    Tracked(caps.instance.borrow()),
                    Tracked(caps.shares.borrow_mut()),
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
            assert!(engine.reclaim_draining(host_tier, 8).unwrap() == 1);
        }

        let op_ns = t_op.elapsed().as_nanos() as u64;
        op_hist.record_ns(op_ns);
        op_idx += 1;
    }

    InitiatorRunStats {
        elapsed_s: run_start.elapsed().as_secs_f64(),
        sampler,
        op_hist,
        recov_hist,
        corrupted_frames,
        revoke_count,
        rem_access_errs,
        valid_fetches,
        total_wire_attempts,
    }
}

} // verus!

fn run_donor(job: &JobSpec, transport: TcpTransport) {
    let revoke_rate = job.revoke_rate.unwrap_or(0.01);
    let (ops, donor_cap) = build_hotspot_workload(job);
    let revoke_flags = build_revoke_schedule(ops.len(), revoke_rate);
    run_donor_inner(job, transport, &ops, &revoke_flags, donor_cap);
}

fn run_initiator(job: &JobSpec, transport: TcpTransport) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let revoke_rate = job.revoke_rate.unwrap_or(0.01);
    let (ops, donor_cap) = build_hotspot_workload(job);
    let revoke_flags = build_revoke_schedule(ops.len(), revoke_rate);
    let st = run_initiator_inner(job, transport, &ops, &revoke_flags);

    let (cpu_pct, rss_mb) = st.sampler.finish();
    let good_bytes = st.valid_fetches * (frame_size as u64);
    let wire_bytes = st.total_wire_attempts * (frame_size as u64);

    assert_eq!(
        st.corrupted_frames, 0,
        "VeriCache Option-A revocation MUST guarantee corrupted_frames == 0"
    );

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.num_blocks = donor_cap;
    rec.throughput_gbps = bytes_per_sec_to_gbps(wire_bytes, st.elapsed_s);
    rec.goodput_gbps = bytes_per_sec_to_gbps(good_bytes, st.elapsed_s);
    rec.p50_us = st.op_hist.p50_us();
    rec.p95_us = st.op_hist.p95_us();
    rec.p99_us = st.op_hist.p99_us();
    rec.hit_rate = (st.valid_fetches as f64) / (ops.len() as f64);
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.corrupted_frames = st.corrupted_frames;
    rec.recovery_us = st.recov_hist.mean_us();
    rec.notes = format!(
        "Option-A unilateral revoke (rate={:.3}, revokes={}, IBV_WC_REM_ACCESS_ERR={}, recov_p50_us={:.1}, recov_p99_us={:.1}, donor_cap_15pct={})",
        revoke_rate,
        st.revoke_count,
        st.rem_access_errs,
        st.recov_hist.p50_us(),
        st.recov_hist.p99_us(),
        donor_cap
    );
    rec
}
