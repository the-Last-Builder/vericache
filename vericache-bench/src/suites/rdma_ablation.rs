use crate::schema::{bytes_per_sec_to_gbps, JobSpec, LatencyHist, MetricRecord, ProcSampler};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use transfer::context::*;
use transfer::cuda::{self, CudaStream};
use transfer::host::{self, HostContext};
use transfer::rdma::*;
use transfer::transport::{ByteTransport, Channel, ReceivedFrame, Statement};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, PeerSession, ServeOutcome};
use vericache_core::backends::rdma::protocol::{ControlPayload, VeriCacheAttachment};
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::engine::{DistributedEngine, DistributedEvent};
use vericache_core::kernel::CacheCore;
use vericache_core::spec::ObjectKey;
use vericache_core::timeline::TimelineKind;
use vstd::prelude::{Ghost, Tracked};

pub struct TcpTransport(TcpStream);

impl TcpTransport {
    pub fn new(stream: TcpStream) -> Self {
        stream.set_nodelay(true).expect("set_nodelay");
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .expect("set_read_timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(60)))
            .expect("set_write_timeout");
        Self(stream)
    }
}

impl ByteTransport for TcpTransport {
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
        if n > 16 * 1024 * 1024 {
            return Err(io::Error::other("frame exceeds limit"));
        }
        let mut frame = vec![0u8; n];
        self.0.read_exact(&mut frame)?;
        Ok(frame)
    }
}

pub fn open_rdma_device(node_id: u64, requested: &str) -> DeviceContext {
    let count = DeviceContext::count().expect("enumerate RDMA devices");
    for idx in 0..count {
        let dev = DeviceContext::open(idx, node_id).expect("open RDMA device");
        if dev.name() == requested {
            assert!(
                dev.supports_type_2().expect("query Type-2 MW"),
                "RDMA device {requested} does not support Type-2B MW"
            );
            return dev;
        }
        dev.close().expect("close non-matching device");
    }
    panic!("RDMA device '{requested}' not found among {count} devices");
}

pub fn spawn_two_process_pair(job: &JobSpec) -> Vec<MetricRecord> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback listener");
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
        .expect("spawn initiator child");

    let transport = accept_peer(&listener, &session, &mut init_child);

    // Run donor directly in parent process (already NUMA-pinned).
    run_donor_for_suite(job, transport);

    let output = init_child
        .wait_with_output()
        .expect("wait initiator child");
    assert!(
        output.status.success(),
        "initiator child failed with status {}",
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
    assert!(
        !records.is_empty(),
        "initiator produced no MetricRecord lines; stdout={stdout}"
    );
    records
}

fn accept_peer(listener: &TcpListener, session: &str, child: &mut Child) -> TcpTransport {
    listener.set_nonblocking(true).expect("set_nonblocking");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                let mut t = TcpTransport::new(stream);
                let got = t.recv_frame().expect("handshake recv");
                assert_eq!(got, session.as_bytes()[..16], "session mismatch");
                t.send_frame(&session.as_bytes()[16..])
                    .expect("handshake send");
                return t;
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(st) = child.try_wait().expect("try_wait") {
                    panic!("initiator exited before connect: {st}");
                }
                assert!(Instant::now() < deadline, "initiator connect timed out");
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(e) => panic!("accept failed: {e}"),
        }
    }
}

pub fn connect_peer(addr: &str, session: &str) -> TcpTransport {
    let stream = TcpStream::connect(addr).expect("connect to donor");
    let mut t = TcpTransport::new(stream);
    t.send_frame(&session.as_bytes()[..16])
        .expect("handshake send");
    let got = t.recv_frame().expect("handshake recv");
    assert_eq!(got, session.as_bytes()[16..], "session ack mismatch");
    t
}

pub fn run(job: &JobSpec) -> Vec<MetricRecord> {
    spawn_two_process_pair(job)
}

pub fn run_role(job: &JobSpec, role: &str, peer_addr: Option<&str>, session: Option<&str>) {
    assert_eq!(role, "initiator", "only initiator spawns as child");
    let transport = connect_peer(
        peer_addr.expect("peer_addr"),
        session.expect("session"),
    );
    let rec = run_initiator_for_suite(job, transport);
    println!("{}", serde_json::to_string(&rec).unwrap());
}

fn run_donor_for_suite(job: &JobSpec, transport: TcpTransport) {
    match job.case.as_str() {
        "2a_host_rdma" => {
            if job.mode == "raw_mw" {
                run_raw_mw_donor(job, transport, false);
            } else {
                run_dynamic_donor(job, transport, false);
            }
        }
        "2b_gdr_gpu" => {
            if job.mode == "raw_mw" {
                run_raw_mw_donor(job, transport, true);
            } else {
                run_dynamic_donor(job, transport, true);
            }
        }
        "2c_ablation_raw_mw" => run_raw_mw_donor(job, transport, true),
        "2c_ablation_dynamic_single" | "2c_ablation_dynamic_pipelined" => {
            run_dynamic_donor(job, transport, true)
        }
        "2c_ablation_host_raw_mw" => run_raw_mw_donor(job, transport, false),
        "2c_ablation_host_dynamic_single" | "2c_ablation_host_dynamic_pipelined" => {
            run_dynamic_donor(job, transport, false)
        }
        other => panic!("unknown s2_rdma_ablation case '{other}'"),
    }
}

fn run_initiator_for_suite(job: &JobSpec, transport: TcpTransport) -> MetricRecord {
    match job.case.as_str() {
        "2a_host_rdma" => {
            if job.mode == "raw_mw" {
                run_raw_mw_initiator(job, transport, false)
            } else {
                run_dynamic_initiator(job, transport, false)
            }
        }
        "2b_gdr_gpu" => {
            if job.mode == "raw_mw" {
                run_raw_mw_initiator(job, transport, true)
            } else {
                run_dynamic_initiator(job, transport, true)
            }
        }
        "2c_ablation_raw_mw" => run_raw_mw_initiator(job, transport, true),
        "2c_ablation_dynamic_single" | "2c_ablation_dynamic_pipelined" => {
            run_dynamic_initiator(job, transport, true)
        }
        "2c_ablation_host_raw_mw" => run_raw_mw_initiator(job, transport, false),
        "2c_ablation_host_dynamic_single" | "2c_ablation_host_dynamic_pipelined" => {
            run_dynamic_initiator(job, transport, false)
        }
        other => panic!("unknown s2_rdma_ablation case '{other}'"),
    }
}

fn fill_pattern(len: usize, seed: u8) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for (i, b) in v.iter_mut().enumerate() {
        *b = seed.wrapping_add(((i.wrapping_mul(17) + 3) & 0xff) as u8);
    }
    v
}

fn complete_cq(cq: &mut CompletionQueue) -> Completed {
    loop {
        let mut batch = poll_completion_queue_batch(cq, 1, Tracked::assume_new())
            .expect("poll_completion_queue_batch");
        if let Some(c) = batch.pop() {
            return c;
        }
    }
}

// =========================================================================
// Mode A: `vericache_raw_mw` (pre-bound Type-2B MW + RDMA_READ + GDR BAR1 flush)
// =========================================================================
fn run_raw_mw_donor(job: &JobSpec, transport: TcpTransport, use_gpu: bool) {
    let frame_size = job.chunk_bytes;
    let (_inst, _alloc, _ev, _sh) = NodeContext::bootstrap(0).into_sources();
    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, Tracked::assume_new()).expect("PD");
    let mut cq = CompletionQueue::new(&device, 128, Tracked::assume_new()).expect("CQ");
    let reset = UnconnectedQueuePair::new(
        &pd,
        &cq,
        64,
        1,
        job.gid_index,
        0x420,
        Tracked::assume_new(),
    )
    .expect("UnconnectedQueuePair");
    let mut channel: Channel<TcpTransport> =
        Channel::new(transport, 0, 1, Tracked::assume_new());
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked::assume_new())
        .expect("exchange QP ep");
    let qp = connect_rc(reset, peer_ep, 1).expect("connect_rc");

    let mut host_ctx = HostContext::new(Tracked::assume_new());
    let data = fill_pattern(frame_size, 0x33);
    let flags = AccessFlags {
        local_write: true,
        mw_bind: true,
    };

    let mut host_buf_opt = None;
    let mut gpu_buf_opt = None;

    let mr = if use_gpu {
        let gpu = job.donor_gpu;
        let (dev_buf, _) = cuda::alloc_device(Tracked::assume_new(), gpu, frame_size)
            .expect("alloc donor gpu");
        let (stage_buf, _) =
            cuda::alloc_host(Tracked::assume_new(), frame_size).expect("alloc donor stage");
        let h_slice = stage_buf.as_host_slice();
        let _ = host::write_slice(
            &mut host_ctx,
            Tracked::assume_new(),
            &h_slice,
            &data,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Ghost::assume_new(),
        );
        let mut stream = CudaStream::new(gpu, Tracked::assume_new()).expect("donor stream");
        let _ = cuda::memcpy_async(
            &mut stream,
            Tracked::assume_new(),
            &dev_buf.slot_slice(0),
            &stage_buf.as_cuda_slice(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Ghost::assume_new(),
        )
        .expect("upload donor gpu");
        let _ = cuda::stream_synchronize(&stream, Tracked::assume_new()).expect("sync upload");
        let _ = stream.destroy();
        gpu_buf_opt = Some(dev_buf);
        register_memory(
            &pd,
            RegistrationBuffer::Cuda(gpu_buf_opt.as_ref().unwrap()),
            flags,
        )
        .expect("register donor GPU MR")
    } else {
        let (hbuf, _) = host::alloc_aligned(
            &host_ctx,
            Tracked::assume_new(),
            Tracked::assume_new(),
            frame_size,
            4096,
        )
        .expect("alloc host");
        let _ = host::write_slice(
            &mut host_ctx,
            Tracked::assume_new(),
            &hbuf.as_slice(),
            &data,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Ghost::assume_new(),
        );
        host_buf_opt = Some(hbuf);
        register_memory(
            &pd,
            RegistrationBuffer::Host(host_buf_opt.as_ref().unwrap()),
            flags,
        )
        .expect("register donor Host MR")
    };

    // Pre-bind Type-2B MW once before benchmark loop.
    let mut mw = alloc_window(&pd, Tracked::assume_new()).expect("alloc Type-2B MW");
    let rkey = mw.next_rkey();
    post_bind_window(
        &qp,
        &mut cq,
        &mr.as_slice(),
        &mut mw,
        rkey,
        RemoteAccess::ReadOnly,
        Tracked::assume_new(),
        1,
    )
    .map_err(|e| e.0)
    .expect("post_bind_window");
    let Completed::WindowBound { witness: bound, .. } = complete_cq(&mut cq) else {
        panic!("expected WindowBound CQE");
    };
    mw.finish_bind(bound);

    let cap = mw.remote_capability();
    let empty_stmt: Tracked<Statement> = Tracked::assume_new();
    channel
        .send_remote_read_grant(1, &cap, b"raw_mw", empty_stmt, Tracked::assume_new())
        .map_err(|e| e.0)
        .expect("send pre-bound ReadGrant");

    // Wait for initiator to finish all timed iterations and return release.
    let _rel = channel
        .recv_remote_read_release(Tracked::assume_new())
        .expect("recv ReadRelease");

    post_local_invalidate(&qp, &mut cq, &mut mw, 2).expect("post_local_invalidate");
    let Completed::WindowInvalidated { witness: inv, .. } = complete_cq(&mut cq) else {
        panic!("expected WindowInvalidated CQE");
    };
    mw.finish_invalidate(inv, Tracked::assume_new());
    dealloc_window(mw).map_err(|e| e.0).expect("dealloc_window");
    let _ = deregister_memory(
        &mut host_ctx,
        Tracked::assume_new(),
        mr,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        Ghost::assume_new(),
    );
    let _ = qp.destroy(&cq);
    let _ = cq.destroy();
    let _ = pd.destroy();
    let _ = device.close();
}

fn run_raw_mw_initiator(job: &JobSpec, transport: TcpTransport, use_gpu: bool) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let qd = job.concurrency.max(1).min(16);
    let (_inst, _alloc, _ev, _sh) = NodeContext::bootstrap(1).into_sources();
    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, Tracked::assume_new()).expect("PD");
    let mut cq = CompletionQueue::new(&device, 128, Tracked::assume_new()).expect("CQ");
    let reset = UnconnectedQueuePair::new(
        &pd,
        &cq,
        64,
        1,
        job.gid_index,
        0x421,
        Tracked::assume_new(),
    )
    .expect("UnconnectedQueuePair");
    let mut channel: Channel<TcpTransport> =
        Channel::new(transport, 1, 0, Tracked::assume_new());
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked::assume_new())
        .expect("exchange QP ep");
    let mut qp = connect_rc(reset, peer_ep, 0).expect("connect_rc");

    let mut host_ctx = HostContext::new(Tracked::assume_new());
    let flags = AccessFlags {
        local_write: true,
        mw_bind: false,
    };

    let mut host_slab_opt = None;
    let mut gpu_slab_opt = None;
    let mr = if use_gpu {
        let gpu = job.initiator_gpu;
        let (dev_slab, _) = cuda::alloc_device_slab(Tracked::assume_new(), gpu, qd, frame_size)
            .expect("alloc init gpu slab");
        gpu_slab_opt = Some(dev_slab);
        register_memory(
            &pd,
            RegistrationBuffer::Cuda(gpu_slab_opt.as_ref().unwrap()),
            flags,
        )
        .expect("register init GPU MR")
    } else {
        let (h_slab, _) = host::alloc_slab(
            &host_ctx,
            Tracked::assume_new(),
            Tracked::assume_new(),
            qd,
            frame_size,
            4096,
        )
        .expect("alloc host slab");
        host_slab_opt = Some(h_slab);
        register_memory(
            &pd,
            RegistrationBuffer::Host(host_slab_opt.as_ref().unwrap()),
            flags,
        )
        .expect("register init Host MR")
    };

    let ReceivedFrame::ReadGrant(grant) = channel
        .recv_any(Tracked::assume_new())
        .expect("recv pre-bound ReadGrant")
    else {
        panic!("expected ReadGrant");
    };
    let cap = grant.capability.expect("capability");

    let warmup_batches = (job.warmup_iters + qd - 1) / qd;
    let meas_batches = (job.iters + qd - 1) / qd;
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();
    let mut next_wr_id: u64 = 1;
    let mut total_ops = 0u64;

    for b in 0..total_batches {
        if b == warmup_batches {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        if use_gpu {
            let mut pendings = Vec::with_capacity(qd);
            for slot in 0..qd {
                let wr = next_wr_id;
                next_wr_id += 1;
                let (_, _, pending_gdr) = post_rdma_read_gdr(
                    &mut qp,
                    &mut cq,
                    Tracked::assume_new(),
                    &mr.slot_slice(slot),
                    &cap,
                    wr,
                    false,
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    Ghost::assume_new(),
                )
                .map_err(|e| e.0)
                .expect("post_rdma_read_gdr");
                pendings.push((wr, pending_gdr));
            }
            for (_wr, pending_gdr) in pendings {
                match complete_cq(&mut cq) {
                    Completed::Data { .. } => {}
                    Completed::DataFailed { status, .. } => {
                        panic!("raw_mw GDR RDMA_READ failed status={status}");
                    }
                    _ => panic!("unexpected CQE"),
                }
                // Mandatory GPUDirect RDMA BAR1 visibility flush.
                let _tok = mr
                    .complete_gdr_slot(
                        Tracked::assume_new(),
                        pending_gdr,
                        Tracked::assume_new(),
                    )
                    .map_err(|e| e.0)
                    .expect("complete_gdr_slot BAR1 flush");
            }
        } else {
            for slot in 0..qd {
                let wr = next_wr_id;
                next_wr_id += 1;
                let _ = post_rdma_read(
                    &mut qp,
                    &mut cq,
                    Tracked::assume_new(),
                    &mr.slot_slice(slot),
                    &cap,
                    wr,
                    false,
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    Ghost::assume_new(),
                )
                .expect("post_rdma_read");
            }
            for _ in 0..qd {
                match complete_cq(&mut cq) {
                    Completed::Data { .. } => {}
                    Completed::DataFailed { status, .. } => {
                        panic!("raw_mw Host RDMA_READ failed status={status}");
                    }
                    _ => panic!("unexpected CQE"),
                }
            }
        }
        let dt_ns = t0.elapsed().as_nanos() as u64;
        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += qd as u64;
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);

    channel
        .send_remote_read_release(
            grant.tag,
            b"raw_mw",
            Tracked::assume_new(),
            Tracked::assume_new(),
        )
        .map_err(|e| e.0)
        .expect("send ReadRelease");

    let _ = deregister_memory_slab(
        &mut host_ctx,
        Tracked::assume_new(),
        mr,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        Ghost::assume_new(),
    );
    let _ = qp.destroy(&cq);
    let _ = cq.destroy();
    let _ = pd.destroy();
    let _ = device.close();

    let mut rec = MetricRecord::from_job("vericache_raw_mw", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = if use_gpu {
        format!(
            "Pre-bound Type-2B MW + post_rdma_read_gdr (GPU{}->GPU{}) + complete_gdr_slot BAR1 flush (QD={qd}, {})",
            job.donor_gpu, job.initiator_gpu, job.rdma_dev
        )
    } else {
        format!(
            "Pre-bound Type-2B MW + Host->Host post_rdma_read (QD={qd}, {})",
            job.rdma_dev
        )
    };
    rec
}

// =========================================================================
// Mode B: `vericache_dynamic_lease` (`DistributedEngine` per-lease BIND_MW +
//         RDMA_READ + BAR1 flush + LOCAL_INV, single-flight & pipelined)
// =========================================================================
pub fn build_distributed_engine<'a>(
    node_id: u64,
    _frame_size: usize,
    num_frames: u32,
    num_windows: usize,
    gid_index: u8,
    device: &'a DeviceContext,
    pd: &'a ProtectionDomain<'a>,
    transport: TcpTransport,
    host: &'a HostBackend,
    cuda: Option<&'a CudaBackend<'a>>,
    mut core: CacheCore,
) -> DistributedEngine<'a, TcpTransport> {
    let peer_id = 1 - node_id;
    let cq = CompletionQueue::new(device, 256, Tracked::assume_new()).expect("CQ");
    let reset = UnconnectedQueuePair::new(
        pd,
        &cq,
        64,
        1,
        gid_index,
        0x520 + node_id as u32,
        Tracked::assume_new(),
    )
    .expect("UnconnectedQueuePair");
    let mut channel: Channel<TcpTransport, VeriCacheAttachment> = Channel::new_with_protocol(
        transport,
        node_id,
        peer_id,
        Tracked::assume_new(),
    );
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked::assume_new())
        .expect("exchange QP ep");
    let qp = connect_rc(reset, peer_ep, peer_id).expect("connect_rc");

    let mut windows = Vec::with_capacity(num_windows);
    for _ in 0..num_windows {
        windows.push(alloc_window(pd, Tracked::assume_new()).expect("alloc Type-2B MW"));
    }
    let mut rdma =
        RdmaBackend::new(&mut core, peer_id, qp, cq, windows).map_err(|e| e.0).expect("RdmaBackend::new");
    let flags = AccessFlags {
        local_write: true,
        mw_bind: true,
    };
    rdma.register_host_slab(
        &core,
        pd,
        &host.buffer,
        host.tier,
        host.first_frame,
        host.num_frames,
        flags,
    )
    .expect("register_host_slab");
    if let Some(c) = cuda {
        rdma.register_gpu_slab(
            &core,
            pd,
            &c.gpu_buffer,
            c.gpu_tier,
            c.gpu_first_frame,
            c.gpu_num_frames,
            flags,
        )
        .expect("register_gpu_slab");
    }
    let peer = PeerSession::new(rdma, channel, &core);
    let eng_ctx = HostContext::new(Tracked::assume_new());
    let _ = num_frames;
    DistributedEngine::new(core, host, eng_ctx, cuda, peer)
}

fn make_versioned_core(node_id: u64, frame_size: usize, max_leases: usize) -> CacheCore {
    let (_inst, _alloc, _ev, _sh, _vi, _vs) =
        NodeContext::bootstrap(node_id).into_sources_with_versions();
    let mut core = CacheCore::new(
        node_id as u16,
        frame_size,
        Vec::new(),
        max_leases,
        Tracked::assume_new(),
    );
    core.enable_version_registry(Tracked::assume_new(), Tracked::assume_new());
    core
}

fn run_dynamic_donor(job: &JobSpec, transport: TcpTransport, use_gpu: bool) {
    let frame_size = job.chunk_bytes;
    let pipelined = job.mode == "pipelined";
    let window_w = if pipelined {
        job.concurrency.max(1).min(16)
    } else {
        1
    };
    let num_frames = (window_w + 4) as u32;
    let num_windows = window_w.max(4);

    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, Tracked::assume_new()).expect("PD");
    let mut core = make_versioned_core(0, frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        num_frames,
        4096,
    )
    .expect("HostBackend::new");

    let cuda_opt = if use_gpu {
        let c = CudaBackend::new(
            &mut core,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            job.donor_gpu,
            num_frames,
            32,
            true,
        )
        .expect("CudaBackend::new");
        Some(c)
    } else {
        None
    };

    let mut caller_stream_opt = if use_gpu {
        let tl = core
            .register_timeline(TimelineKind::CudaStream {
                device: job.donor_gpu,
            })
            .expect("caller_tl");
        let st = CudaStream::new(job.donor_gpu, Tracked::assume_new()).expect("donor stream");
        Some((st, tl))
    } else {
        None
    };

    let mut engine = build_distributed_engine(
        0,
        frame_size,
        num_frames,
        num_windows,
        job.gid_index,
        &device,
        &pd,
        transport,
        &host,
        cuda_opt.as_ref(),
        core,
    );

    let serve_tier = if use_gpu {
        engine.gpu_tier().unwrap()
    } else {
        engine.host_tier()
    };

    // Publish `window_w` distinct keys on Donor so pipelined batches can pull distinct frames concurrently.
    let payload = fill_pattern(frame_size, 0x77);
    let mut stage_ctx = HostContext::new(Tracked::assume_new());
    for i in 0..window_w {
        let key = ObjectKey([20, i as u64, frame_size as u64, 0]);
        if use_gpu {
            let (stage_buf, _) =
                cuda::alloc_host(Tracked::assume_new(), frame_size).expect("alloc_host stage");
            let _ = host::write_slice(
                &mut stage_ctx,
                Tracked::assume_new(),
                &stage_buf.as_host_slice(),
                &payload,
                Tracked::assume_new(),
                Tracked::assume_new(),
                Tracked::assume_new(),
                Ghost::assume_new(),
            );
            let (st, tl) = caller_stream_opt.as_mut().unwrap();
            let _ = engine
                .put_gpu_sync(
                    Tracked::assume_new(),
                    st,
                    *tl,
                    key,
                    &stage_buf.as_cuda_slice(),
                    Tracked::assume_new(),
                    Ghost::assume_new(),
                )
                .expect("put_gpu_sync donor");
        } else {
            let _ = engine
                .put_host(Tracked::assume_new(), key, &payload)
                .expect("put_host donor");
        }
    }

    // Serve events until initiator sends final RevokeAck sentinel (grant_tag == u64::MAX).
    loop {
        match engine
            .poll_peer_event(Tracked::assume_new())
            .expect("donor poll_peer_event")
        {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key }) => {
                let res = engine
                    .serve_fetch_request(
                        serve_tier,
                        request_id,
                        key,
                        Tracked::assume_new(),
                        Tracked::assume_new(),
                    )
                    .expect("serve_fetch_request");
                assert!(
                    matches!(res, ServeOutcome::Granted { .. }),
                    "donor missed published key"
                );
            }
            DistributedEvent::ReadReleased { was_active, .. } => {
                assert!(was_active, "expected active lease release");
            }
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag })
                if grant_tag == u64::MAX =>
            {
                break;
            }
            _ => {}
        }
    }

    if let Some((st, _)) = caller_stream_opt {
        let _ = st.destroy();
    }
}

fn run_dynamic_initiator(job: &JobSpec, transport: TcpTransport, use_gpu: bool) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let pipelined = job.mode == "pipelined";
    let window_w = if pipelined {
        job.concurrency.max(1).min(16)
    } else {
        1
    };
    let num_frames = (window_w + 4) as u32;
    let num_windows = window_w.max(4);

    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, Tracked::assume_new()).expect("PD");
    let mut core = make_versioned_core(1, frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        num_frames,
        4096,
    )
    .expect("HostBackend::new");

    let cuda_opt = if use_gpu {
        let c = CudaBackend::new(
            &mut core,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            job.initiator_gpu,
            num_frames,
            32,
            true,
        )
        .expect("CudaBackend::new");
        Some(c)
    } else {
        None
    };

    let mut engine = build_distributed_engine(
        1,
        frame_size,
        num_frames,
        num_windows,
        job.gid_index,
        &device,
        &pd,
        transport,
        &host,
        cuda_opt.as_ref(),
        core,
    );

    let (dst_tier, pull_target) = if use_gpu {
        (engine.gpu_tier().unwrap(), PullTarget::GpuGdr)
    } else {
        (engine.host_tier(), PullTarget::Host)
    };

    let keys: Vec<ObjectKey> = (0..window_w)
        .map(|i| ObjectKey([20, i as u64, frame_size as u64, 0]))
        .collect();

    let warmup_batches = (job.warmup_iters + window_w - 1) / window_w;
    let meas_batches = (job.iters + window_w - 1) / window_w;
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();
    let mut total_ops = 0u64;

    for b in 0..total_batches {
        if b == warmup_batches {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let mut pulled_frames = Vec::with_capacity(window_w);

        if !pipelined || window_w == 1 {
            let outcome = engine
                .fetch_remote_frame(keys[0], dst_tier, pull_target, Tracked::assume_new())
                .expect("fetch_remote_frame");
            let FetchOutcome::Hit { frame, .. } = outcome else {
                panic!("expected Hit");
            };
            pulled_frames.push(frame);
        } else {
            // Pipelined window of W requests:
            // 1) Burst-send W FetchRequests (overlaps TCP + Donor MW binds)
            for &key in &keys {
                let _ = engine
                    .send_fetch_request(key, Tracked::assume_new())
                    .expect("send_fetch_request");
            }
            // 2) Receive all W ReadGrants
            let mut grants = Vec::with_capacity(window_w);
            for _ in 0..window_w {
                match engine
                    .poll_peer_event(Tracked::assume_new())
                    .expect("poll ReadGrant")
                {
                    DistributedEvent::ReadGrant(g) => grants.push(g),
                    _ => panic!("expected ReadGrant in pipeline"),
                }
            }
            // 3) Post all W RDMA READs back-to-back onto the QP before completing CQEs
            let mut inflight = Vec::with_capacity(window_w);
            for (idx, grant) in grants.into_iter().enumerate() {
                let tag = grant.tag;
                let req_id = grant.meta.request_id;
                let pull = engine
                    .peer
                    .fetch_post_pull(
                        &mut engine.core,
                        keys[idx],
                        dst_tier,
                        pull_target,
                        grant.meta,
                        &grant.cap,
                        grant.share,
                        grant.statement,
                        grant.cert,
                        grant.revoked,
                        Tracked::assume_new(),
                    )
                    .expect("fetch_post_pull");
                inflight.push((tag, req_id, pull));
            }
            // 4) Complete all W in-flight RDMA READs (BAR1 flush + finish_write + ReadRelease)
            for (tag, req_id, pull) in inflight {
                let outcome = engine
                    .peer
                    .fetch_complete_pull(
                        &mut engine.core,
                        tag,
                        req_id,
                        pull,
                        Tracked::assume_new(),
                        Ghost::assume_new(),
                        Tracked::assume_new(),
                    )
                    .expect("fetch_complete_pull");
                let FetchOutcome::Hit { frame, .. } = outcome else {
                    panic!("expected Hit in pipelined pull");
                };
                pulled_frames.push(frame);
            }
        }
        let dt_ns = t0.elapsed().as_nanos() as u64;

        // Evict and reclaim local pulled frames so next iteration performs a fresh remote pull.
        for f in pulled_frames {
            engine.evict_frame(f).expect("evict pulled frame");
        }
        let reclaimed = engine.reclaim_draining(dst_tier, window_w + 4);
        assert_eq!(reclaimed, window_w, "reclaim all pulled frames");

        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += window_w as u64;
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);

    // Signal donor to exit cleanly.
    engine
        .send_control(
            ControlPayload::RevokeAck {
                grant_tag: u64::MAX,
            },
            Tracked::assume_new(),
        )
        .expect("send exit sentinel");

    let sys_name = if job.case.starts_with("2c_") {
        "vericache_dynamic_lease"
    } else {
        "vericache"
    };
    let mut rec = MetricRecord::from_job(sys_name, job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = format!(
        "DistributedEngine dynamic Type-2B MW lease (BIND_MW+RDMA_READ+{}+LOCAL_INV, mode={}, W={}, {})",
        if use_gpu { "GDR_BAR1_flush" } else { "Host" },
        job.mode,
        window_w,
        job.rdma_dev
    );
    rec
}
