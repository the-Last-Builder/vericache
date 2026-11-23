use crate::schema::{JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps};
use crate::shard::ShardCapabilityBundle;
#[cfg(verus_only)]
use foundation::tracked::ISet;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::*;
use transfer::cuda::{self, CudaStream};
use transfer::host::{self, HostContext};
use transfer::rdma::*;
use transfer::transport::{ByteTransport, Channel, ReceivedFrame};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, PeerSession, ServeOutcome};
use vericache_core::backends::rdma::protocol::{ControlPayload, VeriCacheAttachment};
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::engine::{DistributedEngine, DistributedEvent};
use vericache_core::kernel::CacheCore;
#[cfg(verus_only)]
use vericache_core::provenance::{ValueCertificate, ValueProof};
use vericache_core::spec::ObjectKey;
use vericache_core::timeline::TimelineKind;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Map, Tracked, verus};

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

    let output = init_child.wait_with_output().expect("wait initiator child");
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
    let transport = connect_peer(peer_addr.expect("peer_addr"), session.expect("session"));
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

verus! {

fn complete_cq(cq: &mut CompletionQueue, Tracked(instance): Tracked<&TransferInstance>) -> Completed {
    loop {
        let mut batch = poll_completion_queue_batch(cq, 1, Tracked(instance))
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
    let mut caps = ShardCapabilityBundle::take(0);
    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut cq = CompletionQueue::new(&device, 128, caps.mint_rdma()).expect("CQ");
    let reset = UnconnectedQueuePair::new(
        &pd,
        &cq,
        64,
        1,
        job.gid_index,
        0x420,
        caps.mint_rdma(),
    )
    .expect("UnconnectedQueuePair");
    let mut channel: Channel<TcpTransport> =
        Channel::new(transport, 0, 1, Tracked(caps.instance.borrow()));
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked(caps.instance.borrow()))
        .expect("exchange QP ep");
    let qp = connect_rc(reset, peer_ep, 1).expect("connect_rc");

    let mut host_ctx = HostContext::new(caps.mint_host());
    let data = fill_pattern(frame_size, 0x33);
    let flags = AccessFlags {
        local_write: true,
        mw_bind: true,
    };

    let mut mw = alloc_window(&pd, caps.mint_rdma()).expect("alloc Type-2B MW");

    if use_gpu {
        let gpu = job.donor_gpu;
        let (dev_buf, Tracked(mut dev_tok)) =
            cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), gpu, frame_size)
                .expect("alloc donor gpu");
        let (stage_buf, Tracked(mut stage_tok)) =
            cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), frame_size)
                .expect("alloc donor stage");
        let h_slice = stage_buf.as_host_slice();
        let tracked mut sub_map = SubmittedMap::empty(caps.instance.borrow().id());
        let tracked empty_knows = Knows::empty(caps.instance.borrow().id());
        let Tracked(h_comp) = host::write_slice(
            &mut host_ctx,
            Tracked(caps.instance.borrow()),
            &h_slice,
            &data,
            Tracked(&mut stage_tok),
            Tracked(&sub_map),
            Tracked(&empty_knows),
            Ghost(ISet::empty()),
        );
        let ghost h_ev = h_comp.submitted.key();
        proof {
            sub_map.insert(h_comp.submitted);
        }
        let mut stream = CudaStream::new(gpu, caps.mint_cuda()).expect("donor stream");
        let tracked mut stage_perm = ReadPerm::Owned(stage_tok);
        let (_, Tracked(copy_sub)) = cuda::memcpy_async(
            &mut stream,
            Tracked(caps.instance.borrow()),
            &dev_buf.slot_slice(0),
            &stage_buf.as_cuda_slice(),
            Tracked(&mut stage_perm),
            Tracked(&mut dev_tok),
            Tracked(&mut sub_map),
            Tracked(&h_comp.knows),
            Ghost(ISet::empty().insert(h_ev)),
        )
        .expect("upload donor gpu");
        let Tracked(k_sync) = cuda::stream_synchronize(&stream, Tracked(caps.instance.borrow()))
            .expect("sync upload");
        let _ = stream.destroy();
        let tracked gpu_writer = Completion {
            submitted: copy_sub,
            knows: k_sync,
        };
        let mr = register_memory(&pd, RegistrationBuffer::Cuda(&dev_buf), flags)
            .expect("register donor GPU MR");
        let (_done, _tok) = read_lifecycle_donor(
            &mut channel,
            &qp,
            &mut cq,
            mr,
            &mut mw,
            &mut host_ctx,
            Tracked(caps.instance.borrow()),
            Tracked(dev_tok),
            Tracked(gpu_writer),
            Tracked(caps.shares.borrow_mut()),
            1,
            b"raw_mw",
        )
        .expect("read_lifecycle_donor gpu");
    } else {
        let (hbuf, Tracked(mut h_tok)) = host::alloc_aligned(
            &host_ctx,
            Tracked(caps.instance.borrow()),
            Tracked(caps.alloc.borrow_mut()),
            frame_size,
            4096,
        )
        .expect("alloc host");
        let tracked empty_sub = SubmittedMap::empty(caps.instance.borrow().id());
        let tracked empty_knows = Knows::empty(caps.instance.borrow().id());
        let Tracked(writer) = host::write_slice(
            &mut host_ctx,
            Tracked(caps.instance.borrow()),
            &hbuf.as_slice(),
            &data,
            Tracked(&mut h_tok),
            Tracked(&empty_sub),
            Tracked(&empty_knows),
            Ghost(ISet::empty()),
        );
        let mr = register_memory(&pd, RegistrationBuffer::Host(&hbuf), flags)
            .expect("register donor Host MR");
        let (_done, _tok) = read_lifecycle_donor(
            &mut channel,
            &qp,
            &mut cq,
            mr,
            &mut mw,
            &mut host_ctx,
            Tracked(caps.instance.borrow()),
            Tracked(h_tok),
            Tracked(writer),
            Tracked(caps.shares.borrow_mut()),
            1,
            b"raw_mw",
        )
        .expect("read_lifecycle_donor host");
    }

    dealloc_window(mw).map_err(|e| e.0).expect("dealloc_window");
    let _ = qp.destroy(&cq);
    let _ = cq.destroy();
    let _ = pd.destroy();
    let _ = device.close();
}

fn run_raw_mw_initiator(job: &JobSpec, transport: TcpTransport, use_gpu: bool) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let qd = job.concurrency.max(1).min(16);
    let mut caps = ShardCapabilityBundle::take(1);
    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut cq = CompletionQueue::new(&device, 128, caps.mint_rdma()).expect("CQ");
    let reset = UnconnectedQueuePair::new(
        &pd,
        &cq,
        64,
        1,
        job.gid_index,
        0x421,
        caps.mint_rdma(),
    )
    .expect("UnconnectedQueuePair");
    let mut channel: Channel<TcpTransport> =
        Channel::new(transport, 1, 0, Tracked(caps.instance.borrow()));
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked(caps.instance.borrow()))
        .expect("exchange QP ep");
    let mut qp = connect_rc(reset, peer_ep, 0).expect("connect_rc");

    let mut host_ctx = HostContext::new(caps.mint_host());
    let flags = AccessFlags {
        local_write: true,
        mw_bind: false,
    };

    let mut host_slab_opt = None;
    let mut gpu_slab_opt = None;
    let tracked mut slot_toks: Map<usize, SlotToken>;
    let mr = if use_gpu {
        let gpu = job.initiator_gpu;
        let (dev_slab, Tracked(toks)) =
            cuda::alloc_device_slab(Tracked(caps.alloc.borrow_mut()), gpu, qd, frame_size)
                .expect("alloc init gpu slab");
        proof {
            slot_toks = toks;
        }
        gpu_slab_opt = Some(dev_slab);
        register_memory(
            &pd,
            RegistrationBuffer::Cuda(gpu_slab_opt.as_ref().unwrap()),
            flags,
        )
        .expect("register init GPU MR")
    } else {
        let (h_slab, Tracked(toks)) = host::alloc_slab(
            &host_ctx,
            Tracked(caps.instance.borrow()),
            Tracked(caps.alloc.borrow_mut()),
            qd,
            frame_size,
            4096,
        )
        .expect("alloc host slab");
        proof {
            slot_toks = toks;
        }
        host_slab_opt = Some(h_slab);
        register_memory(
            &pd,
            RegistrationBuffer::Host(host_slab_opt.as_ref().unwrap()),
            flags,
        )
        .expect("register init Host MR")
    };

    let ReceivedFrame::ReadGrant(grant) = channel
        .recv_any(Tracked(caps.instance.borrow()))
        .expect("recv pre-bound ReadGrant")
    else {
        panic!("expected ReadGrant");
    };
    let cap = grant.capability.expect("capability");
    let Tracked(share_opt) = grant.share;
    let tracked share_cap = share_opt.tracked_unwrap();
    let Tracked(stmt) = grant.statement;
    let tracked mut sub_map = stmt.submitted;
    let tracked mut knows_map = stmt.knows;
    let ghost mut cited = stmt.facts;
    let tracked mut remote_perm = ReadPerm::Shared(share_cap);

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
                let tracked slot_tok = slot_toks.tracked_remove(slot);
                let (_ev, Tracked(sub), pending_gdr) = post_rdma_read_gdr(
                    &mut qp,
                    &mut cq,
                    Tracked(caps.instance.borrow()),
                    &mr.slot_slice(slot),
                    &cap,
                    wr,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(slot_tok),
                    Tracked(&mut sub_map),
                    Tracked(&knows_map),
                    Ghost(cited),
                )
                .map_err(|e| e.0)
                .expect("post_rdma_read_gdr");
                proof {
                    sub_map.insert(sub);
                }
                pendings.push((slot, pending_gdr));
            }
            for (slot, pending_gdr) in pendings {
                match complete_cq(&mut cq, Tracked(caps.instance.borrow())) {
                    Completed::Data {
                        event,
                        submitted,
                        knows,
                        ..
                    } => {
                        proof {
                            sub_map.insert(submitted.get());
                            let inst_id = caps.instance.borrow().id();
                            let tracked mut prev = Knows::empty(inst_id);
                            vstd::modes::tracked_swap(&mut prev, &mut knows_map);
                            let tracked mut entries = prev.into_map();
                            entries.tracked_union_prefer_right(knows.get().into_map());
                            knows_map = Knows::from_map(inst_id, entries);
                            cited = cited.insert(event@);
                        }
                    }
                    Completed::DataFailed { status, .. } => {
                        panic!("raw_mw GDR RDMA_READ failed status={status}");
                    }
                    _ => panic!("unexpected CQE"),
                }
                // Mandatory GPUDirect RDMA BAR1 visibility flush.
                let Tracked(tok_back) = mr
                    .complete_gdr_slot(
                        Tracked(caps.instance.borrow()),
                        pending_gdr,
                        Tracked(&knows_map),
                    )
                    .map_err(|e| e.0)
                    .expect("complete_gdr_slot BAR1 flush");
                proof {
                    slot_toks.tracked_insert(slot, tok_back);
                }
            }
        } else {
            for slot in 0..qd {
                let wr = next_wr_id;
                next_wr_id += 1;
                let tracked mut slot_tok = slot_toks.tracked_remove(slot);
                let (_ev, Tracked(sub)) = post_rdma_read(
                    &mut qp,
                    &mut cq,
                    Tracked(caps.instance.borrow()),
                    &mr.slot_slice(slot),
                    &cap,
                    wr,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(&mut slot_tok),
                    Tracked(&mut sub_map),
                    Tracked(&knows_map),
                    Ghost(cited),
                )
                .expect("post_rdma_read");
                proof {
                    sub_map.insert(sub);
                    slot_toks.tracked_insert(slot, slot_tok);
                }
            }
            for _ in 0..qd {
                match complete_cq(&mut cq, Tracked(caps.instance.borrow())) {
                    Completed::Data {
                        event,
                        submitted,
                        knows,
                        ..
                    } => {
                        proof {
                            sub_map.insert(submitted.get());
                            let inst_id = caps.instance.borrow().id();
                            let tracked mut prev = Knows::empty(inst_id);
                            vstd::modes::tracked_swap(&mut prev, &mut knows_map);
                            let tracked mut entries = prev.into_map();
                            entries.tracked_union_prefer_right(knows.get().into_map());
                            knows_map = Knows::from_map(inst_id, entries);
                            cited = cited.insert(event@);
                        }
                    }
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

    let tracked sh_out = remote_perm.into_shared();
    let tracked empty_succ = SucceededMap::empty(caps.instance.borrow().id());
    channel
        .send_remote_read_release_with(
            grant.tag,
            b"raw_mw",
            Tracked(caps.instance.borrow()),
            Tracked(&sub_map),
            Tracked(&knows_map),
            Tracked(&empty_succ),
            Ghost(cited),
            Tracked(sh_out),
            Tracked(()),
        )
        .map_err(|e| e.0)
        .expect("send ReadRelease");

    let _ = deregister_memory_slab(
        &mut host_ctx,
        Tracked(caps.instance.borrow()),
        mr,
        Tracked(slot_toks),
        Tracked(&sub_map),
        Tracked(&knows_map),
        Ghost(cited),
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
#[allow(clippy::too_many_arguments)]
pub fn build_distributed_engine<'a>(
    caps: &mut ShardCapabilityBundle,
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
    let node_id = caps.node;
    let peer_id = 1 - node_id;
    let cq = CompletionQueue::new(device, 256, caps.mint_rdma()).expect("CQ");
    let reset = UnconnectedQueuePair::new(
        pd,
        &cq,
        64,
        1,
        gid_index,
        0x520 + node_id as u32,
        caps.mint_rdma(),
    )
    .expect("UnconnectedQueuePair");
    let mut channel: Channel<TcpTransport, VeriCacheAttachment> = Channel::new_with_protocol(
        transport,
        node_id,
        peer_id,
        Tracked(caps.instance.borrow()),
    );
    let peer_ep = channel
        .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked(caps.instance.borrow()))
        .expect("exchange QP ep");
    let qp = connect_rc(reset, peer_ep, peer_id).expect("connect_rc");

    let mut windows = Vec::with_capacity(num_windows);
    for _ in 0..num_windows {
        windows.push(alloc_window(pd, caps.mint_rdma()).expect("alloc Type-2B MW"));
    }
    let mut rdma = RdmaBackend::new(&mut core, peer_id, qp, cq, windows)
        .map_err(|e| e.0)
        .expect("RdmaBackend::new");
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
    let eng_ctx = HostContext::new(caps.mint_host());
    let _ = num_frames;
    DistributedEngine::new(core, host, eng_ctx, cuda, peer)
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

    let mut caps = ShardCapabilityBundle::take(0);
    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut core = caps.make_versioned_core(frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        num_frames,
        4096,
    )
    .expect("HostBackend::new");

    let cuda_opt = if use_gpu {
        let c = CudaBackend::new(
            &mut core,
            Tracked(caps.instance.borrow()),
            Tracked(caps.alloc.borrow_mut()),
            caps.mint_cuda(),
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
        let st = CudaStream::new(job.donor_gpu, caps.mint_cuda()).expect("donor stream");
        Some((st, tl))
    } else {
        None
    };

    let mut engine = build_distributed_engine(
        &mut caps,
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
    let mut stage_ctx = HostContext::new(caps.mint_host());
    for i in 0..window_w {
        let key = ObjectKey([20, i as u64, frame_size as u64, 0]);
        if use_gpu {
            let (stage_buf, Tracked(mut stage_tok)) =
                cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), frame_size)
                    .expect("alloc_host stage");
            let Tracked(w_comp) = host::write_slice(
                &mut stage_ctx,
                Tracked(caps.instance.borrow()),
                &stage_buf.as_host_slice(),
                &payload,
                Tracked(&mut stage_tok),
                Tracked(&engine.core.ledger.borrow().submitted),
                Tracked(&engine.core.ledger.borrow().knows),
                Ghost(engine.core.citations()),
            );
            let ghost w_ev = w_comp.submitted.key();
            let t_w = engine
                .core
                .issue_ticket(engine.host.timeline_id(), Tracked(w_comp.submitted))
                .expect("issue stage ticket");
            engine.core.mark_timeline_complete(
                engine.host.timeline_id(),
                t_w.seq(),
                Tracked(w_comp.knows),
            );
            let stage_cuda = stage_buf.as_cuda_slice();
            let tracked mut stage_perm = ReadPerm::Owned(stage_tok);
            let ghost stage_cert = ValueCertificate {
                slot: stage_cuda.slot_ref(),
                writer: w_ev,
                bytes: payload@,
                proof: ValueProof::HostWrite,
            };
            let (st, tl) = caller_stream_opt.as_mut().unwrap();
            let _ = engine
                .put_gpu_sync(
                    Tracked(caps.instance.borrow()),
                    st,
                    *tl,
                    key,
                    &stage_cuda,
                    Tracked(&mut stage_perm),
                    Ghost(stage_cert),
                )
                .expect("put_gpu_sync donor");
        } else {
            let _ = engine
                .put_host(Tracked(caps.instance.borrow()), key, &payload)
                .expect("put_host donor");
        }
    }

    // Serve events until initiator sends final RevokeAck sentinel (grant_tag == u64::MAX).
    loop {
        match engine
            .poll_peer_event(Tracked(caps.instance.borrow()))
            .expect("donor poll_peer_event")
        {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key }) => {
                let res = engine
                    .serve_fetch_request(
                        serve_tier,
                        request_id,
                        key,
                        Tracked(caps.shares.borrow_mut()),
                        Tracked(caps.instance.borrow()),
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

    let mut caps = ShardCapabilityBundle::take(1);
    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut core = caps.make_versioned_core(frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        num_frames,
        4096,
    )
    .expect("HostBackend::new");

    let cuda_opt = if use_gpu {
        let c = CudaBackend::new(
            &mut core,
            Tracked(caps.instance.borrow()),
            Tracked(caps.alloc.borrow_mut()),
            caps.mint_cuda(),
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
        &mut caps,
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
                .fetch_remote_frame(keys[0], dst_tier, pull_target, Tracked(caps.instance.borrow()))
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
                    .send_fetch_request(key, Tracked(caps.instance.borrow()))
                    .expect("send_fetch_request");
            }
            // 2) Receive all W ReadGrants
            let mut grants = Vec::with_capacity(window_w);
            for _ in 0..window_w {
                match engine
                    .poll_peer_event(Tracked(caps.instance.borrow()))
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
                let ghost orig_frontier = grant.share@.frontier();
                let tracked vb = grant.version_binding.get().tracked_unwrap();
                let pull = engine.peers[0]
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
                        Tracked(caps.instance.borrow()),
                    )
                    .expect("fetch_post_pull");
                inflight.push((tag, req_id, pull, Tracked(vb), Ghost(orig_frontier)));
            }
            // 4) Complete all W in-flight RDMA READs (BAR1 flush + finish_write + ReadRelease)
            for (tag, req_id, pull, vb, orig_frontier) in inflight {
                let outcome = engine.peers[0]
                    .fetch_complete_pull(
                        &mut engine.core,
                        tag,
                        req_id,
                        pull,
                        vb,
                        orig_frontier,
                        Tracked(caps.instance.borrow()),
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
            Tracked(caps.instance.borrow()),
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

} // verus!
