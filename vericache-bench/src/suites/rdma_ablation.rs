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

fn read_qp_config(device: &DeviceContext, requested: usize) -> QueuePairConfig {
    let attrs = device.attributes().expect("query verbs device limits");
    let send_depth = attrs.max_qp_wr.min(64);
    let depth = requested.clamp(1, send_depth as usize) as u8;
    QueuePairConfig {
        send_depth,
        max_send_sge: 1,
        max_rd_atomic: depth.min(attrs.max_qp_init_rd_atomic),
        max_dest_rd_atomic: depth.min(attrs.max_qp_rd_atomic),
    }
}

const RAW_CQ_CAPACITY: i32 = 128;

fn connect_raw_qps<'a>(
    job: &JobSpec,
    use_gpu: bool,
    caps: &mut ShardCapabilityBundle,
    pd: &'a ProtectionDomain<'a>,
    device: &DeviceContext,
    cq: &CompletionQueue<'a>,
    channel: &mut Channel<TcpTransport>,
) -> Vec<QueuePair<'a>> {
    let depth = job.concurrency.clamp(1, RAW_CQ_CAPACITY as usize);
    let default_qps = if use_gpu { 2 } else { 1 };
    let qp_count = job.raw_qps.unwrap_or(default_qps).clamp(1, depth);
    let mut qps = Vec::with_capacity(qp_count);
    for index in 0..qp_count {
        let requested = depth / qp_count + usize::from(index < depth % qp_count);
        let reset = UnconnectedQueuePair::new_with_config(
            pd,
            cq,
            read_qp_config(device, requested),
            1,
            job.gid_index,
            0x420 + 2 * index as u32 + caps.node as u32,
            caps.mint_rdma(),
        )
        .expect("UnconnectedQueuePair");
        let endpoint = channel
            .exchange_queue_pair_endpoint(&reset.endpoint(), Tracked(caps.instance.borrow()))
            .expect("exchange QP endpoint");
        let qp = connect_rc(reset, endpoint, 1 - caps.node).expect("connect_rc");
        assert!(
            qp.effective_max_rd_atomic_exec() > 0,
            "QP negotiated no RDMA READ credits"
        );
        qps.push(qp);
    }
    qps
}

enum RawBuffer {
    Gpu(cuda::CudaBuffer),
    Host(host::HostBuffer),
}

impl RawBuffer {
    fn register<'a>(
        &'a self,
        pd: &'a ProtectionDomain<'a>,
        flags: AccessFlags,
    ) -> Result<MemoryRegion<'a>, RdmaError> {
        match self {
            Self::Gpu(b) => register_memory(pd, b, flags),
            Self::Host(b) => register_memory(pd, b, flags),
        }
    }
}

fn init_raw_donor_buffer(
    caps: &mut ShardCapabilityBundle,
    host_ctx: &mut HostContext,
    gpu: Option<u32>,
    ring_bytes: usize,
) -> (RawBuffer, Tracked<(SlotToken, host::Completion)>) {
    let data = fill_pattern(ring_bytes, 0x33);
    if let Some(gpu) = gpu {
        let (dev_buf, Tracked(mut dev_tok)) =
            cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), gpu, ring_bytes)
                .expect("alloc donor gpu");
        let (stage_buf, Tracked(mut stage_tok)) =
            cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), ring_bytes)
                .expect("alloc donor stage");
        let h_slice = stage_buf.as_host_slice();
        let tracked mut sub_map = SubmittedMap::empty(caps.instance.borrow().id());
        let tracked empty_knows = Knows::empty(caps.instance.borrow().id());
        let Tracked(h_comp) = host::write_slice(
            host_ctx,
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
        (RawBuffer::Gpu(dev_buf), Tracked((dev_tok, gpu_writer)))
    } else {
        let (hbuf, Tracked(mut h_tok)) = host::alloc_aligned(
            host_ctx,
            Tracked(caps.instance.borrow()),
            Tracked(caps.alloc.borrow_mut()),
            ring_bytes,
            4096,
        )
        .expect("alloc host");
        let tracked empty_sub = SubmittedMap::empty(caps.instance.borrow().id());
        let tracked empty_knows = Knows::empty(caps.instance.borrow().id());
        let Tracked(writer) = host::write_slice(
            host_ctx,
            Tracked(caps.instance.borrow()),
            &hbuf.as_slice(),
            &data,
            Tracked(&mut h_tok),
            Tracked(&empty_sub),
            Tracked(&empty_knows),
            Ghost(ISet::empty()),
        );
        (RawBuffer::Host(hbuf), Tracked((h_tok, writer)))
    }
}

fn run_raw_mw_donor(job: &JobSpec, transport: TcpTransport, use_gpu: bool) {
    let mut caps = ShardCapabilityBundle::take(0);
    let device = open_rdma_device(0, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut cq = CompletionQueue::new(&device, RAW_CQ_CAPACITY, caps.mint_rdma()).expect("CQ");
    let mut channel = Channel::new(transport, 0, 1, Tracked(caps.instance.borrow()));
    let qps = connect_raw_qps(job, use_gpu, &mut caps, &pd, &device, &cq, &mut channel);
    let mut host_ctx = HostContext::new(caps.mint_host());
    let gpu = if use_gpu { Some(job.donor_gpu) } else { None };
    let mut buffers = Vec::with_capacity(qps.len());
    let tracked mut contents = Map::tracked_empty();
    for (index, qp) in qps.iter().enumerate() {
        let depth = usize::from(qp.effective_max_rd_atomic_exec());
        let bytes = depth.checked_mul(job.chunk_bytes).expect("raw MW ring size");
        let (buffer, Tracked(written)) = init_raw_donor_buffer(&mut caps, &mut host_ctx, gpu, bytes);
        buffers.push(buffer);
        proof { contents.tracked_insert(index, written); }
    }

    // A Type-2B window binds to its responder QP. Each QP grants its own range.
    let mut windows = Vec::with_capacity(qps.len());
    let tracked mut leases = Map::tracked_empty();
    for (index, (qp, buffer)) in qps.iter().zip(&buffers).enumerate() {
        let flags = AccessFlags {
            local_write: true,
            mw_bind: true,
        };
        let mr = buffer.register(&pd, flags).expect("register donor MR");
        let mut mw = alloc_window(&pd, caps.mint_rdma()).expect("allocate Type-2B MW");
        let tracked (tok, writer) = contents.tracked_remove(index);
        let tracked write = writer.submitted;
        let (Tracked(lent), Tracked(prior)) = grant_read_window(
            &mut channel,
            qp,
            &mut cq,
            &mr,
            &mut mw,
            Tracked(caps.instance.borrow()),
            Tracked(tok),
            Tracked(writer),
            Tracked(caps.shares.borrow_mut()),
            index as u64 + 1,
            b"raw_mw",
        )
        .expect("grant raw MW");
        windows.push((mr, mw));
        proof { leases.tracked_insert(index, (lent, prior, write)); }
    }

    for (index, (mr, mut mw)) in windows.into_iter().enumerate() {
        let release = channel
            .recv_remote_read_release(Tracked(caps.instance.borrow()))
            .expect("receive raw MW release");
        assert_eq!(release.tag, index as u64 + 1);
        let Tracked(mut statement) = release.statement;
        let tracked (lent, prior, write) = leases.tracked_remove(index);
        let tracked (Tracked(tok), Tracked(returned)) = caps.instance.borrow().give_back_slot(
            lent,
            release.share.get().tracked_unwrap(),
        );
        let ghost cited = statement.facts.insert(write.key());
        proof {
            if !statement.submitted.dom().contains(write.key()) {
                statement.submitted.insert(write);
            }
            statement.knows.insert(prior);
        }
        post_local_invalidate(&qps[index], &mut cq, &mut mw, 2).expect("invalidate raw MW");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match poll_completion_queue(&mut cq, Tracked(caps.instance.borrow()))
                .expect("poll LOCAL_INV")
            {
                PollStatus::Empty => assert!(Instant::now() < deadline, "LOCAL_INV timed out"),
                PollStatus::Completed(Completed::WindowInvalidated { witness, .. }) => {
                    mw.finish_invalidate(witness, Tracked(Some(returned)));
                    break;
                },
                PollStatus::Completed(_) => panic!("unexpected LOCAL_INV CQE"),
            }
        }
        dealloc_window(mw).map_err(|e| e.0).expect("deallocate raw MW");
        deregister_memory(
            &mut host_ctx,
            Tracked(caps.instance.borrow()),
            mr,
            Tracked(tok),
            Tracked(&statement.submitted),
            Tracked(&statement.knows),
            Ghost(cited),
        )
        .map_err(|e| e.0)
        .expect("deregister donor MR");
    }
    for qp in qps {
        qp.destroy(&cq).expect("destroy QP");
    }
    let _ = cq.destroy();
    let _ = pd.destroy();
    let _ = device.close();
}

struct RawReadRing {
    cap: RemoteCapability,
    tag: u64,
    first_slot: usize,
    free: Vec<usize>,
}

fn run_raw_mw_initiator(job: &JobSpec, transport: TcpTransport, use_gpu: bool) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let mut caps = ShardCapabilityBundle::take(1);
    let device = open_rdma_device(1, &job.rdma_dev);
    let pd = ProtectionDomain::new(&device, caps.mint_rdma()).expect("PD");
    let mut cq = CompletionQueue::new(&device, RAW_CQ_CAPACITY, caps.mint_rdma()).expect("CQ");
    let mut channel = Channel::new(transport, 1, 0, Tracked(caps.instance.borrow()));
    let mut qps = connect_raw_qps(job, use_gpu, &mut caps, &pd, &device, &cq, &mut channel);
    let qp_count = qps.len();
    let qd = qps.iter().map(|qp| usize::from(qp.effective_max_rd_atomic_exec())).sum();

    let mut host_ctx = HostContext::new(caps.mint_host());
    let flags = AccessFlags {
        local_write: true,
        mw_bind: false,
    };

    let (buffer, Tracked(mut slot_toks)) = if use_gpu {
        let gpu = job.initiator_gpu;
        let (dev_slab, Tracked(toks)) =
            cuda::alloc_device_slab(Tracked(caps.alloc.borrow_mut()), gpu, qd, frame_size)
                .expect("alloc init gpu slab");
        (RawBuffer::Gpu(dev_slab), Tracked(toks))
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
        (RawBuffer::Host(h_slab), Tracked(toks))
    };
    let mr = buffer.register(&pd, flags).expect("register initiator MR");

    let mut rings = Vec::with_capacity(qp_count);
    let mut slot_ring = Vec::with_capacity(qd);
    let ghost instance_id = caps.instance.borrow().id();
    let tracked mut sub_map = SubmittedMap::empty(instance_id);
    let tracked mut knows_map = Knows::empty(instance_id);
    let ghost mut cited = ISet::empty();
    let tracked mut shares = Map::tracked_empty();
    let tracked mut owned = Map::tracked_empty();
    let tracked mut pending = PendingGdrSlots::empty(instance_id);
    for (index, qp) in qps.iter().enumerate() {
        let depth = usize::from(qp.effective_max_rd_atomic_exec());
        let ReceivedFrame::ReadGrant(grant) = channel
            .recv_any(Tracked(caps.instance.borrow()))
            .expect("recv pre-bound ReadGrant")
        else {
            panic!("expected ReadGrant");
        };
        assert_eq!(grant.tag, index as u64 + 1);
        let cap = grant.capability.expect("capability");
        assert!(cap.len() >= (depth * frame_size) as u64, "donor ring covers initiator QD");
        proof {
            shares.tracked_insert(cap.slot_ref().loc, grant.share.get().tracked_unwrap());
            let tracked stmt = grant.statement.get();
            let tracked mut prior_sub = SubmittedMap::empty(instance_id);
            vstd::modes::tracked_swap(&mut prior_sub, &mut sub_map);
            let tracked mut subs = prior_sub.into_map();
            subs.tracked_union_prefer_right(stmt.submitted.into_map());
            sub_map = SubmittedMap::from_map(instance_id, subs);
            let tracked mut prior_knows = Knows::empty(instance_id);
            vstd::modes::tracked_swap(&mut prior_knows, &mut knows_map);
            let tracked mut known = prior_knows.into_map();
            known.tracked_union_prefer_right(stmt.knows.into_map());
            knows_map = Knows::from_map(instance_id, known);
            cited = cited.union(stmt.facts);
        }
        let first_slot = slot_ring.len();
        slot_ring.resize(first_slot + depth, index);
        rings.push(RawReadRing {
            cap,
            tag: grant.tag,
            first_slot,
            free: (first_slot..first_slot + depth).collect(),
        });
    }
    proof {
        for slot in 0..qd {
            owned.tracked_insert(mr.slot_loc(slot as nat), slot_toks.tracked_remove(slot));
        }
    }

    let total = job.warmup_iters + job.iters;
    let refill = (qd / 4).max(1);
    let slots: Vec<_> = (0..qd).map(|i| [mr.slot_slice(i)]).collect();
    let mut requests = Vec::with_capacity(qd);
    let mut reaped = Vec::with_capacity(refill);
    let mut submitted_at = vec![Instant::now(); qd];
    let mut posted = 0;
    let mut completed = 0;
    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();

    while completed < total {
        for (qp, ring) in qps.iter_mut().zip(&mut rings) {
            let count = ring.free.len().min(total - posted);
            if count == 0 {
                continue;
            }
            requests.clear();
            let now = Instant::now();
            for _ in 0..count {
                let slot = ring.free.pop().unwrap();
                submitted_at[slot] = now;
                requests.push(TransferRequest {
                    locals: &slots[slot],
                    remote: ring.cap.sub_slice(
                        ((slot - ring.first_slot) * frame_size) as u64,
                        frame_size as u64,
                    ),
                    wr_id: slot as u64,
                    read: true,
                    fence: false,
                });
            }
            let outcome = if use_gpu {
                let (outcome, Tracked(mut sealed)) = post_send_batch_gdr(
                    qp, &mut cq, Tracked(caps.instance.borrow()), &requests,
                    Tracked(&mut owned), Tracked(&mut shares), Tracked(&mut sub_map),
                    Tracked(&knows_map), Ghost(cited),
                );
                proof {
                    for req in requests.iter().take(outcome.posted.len()) {
                        let loc = mr.slot_loc(req.wr_id as nat);
                        pending.insert(sealed.remove(loc));
                    }
                }
                outcome
            } else {
                post_send_batch(
                    qp, &mut cq, Tracked(caps.instance.borrow()), &requests,
                    Tracked(&mut owned), Tracked(&mut shares), Tracked(&mut sub_map),
                    Tracked(&knows_map), Ghost(cited),
                )
            };
            assert_eq!(outcome.posted.len(), count, "raw MW post error: {:?}", outcome.error);
            posted += count;
        }

        reaped.clear();
        let target = refill.min(posted - completed);
        let tracked mut flush = PendingGdrSlots::empty(caps.instance.borrow().id());
        while reaped.len() < target {
            let cqes = poll_completion_queue_batch(
                &mut cq, target - reaped.len(), Tracked(caps.instance.borrow()),
            ).expect("poll READ batch");
            if cqes.is_empty() {
                std::hint::spin_loop();
            }
            for cqe in cqes {
                match cqe {
                    Completed::Data { wr_id, event, submitted, knows, .. } => {
                        reaped.push(wr_id as usize);
                        proof {
                            sub_map.insert(submitted.get());
                            let inst_id = caps.instance.borrow().id();
                            let tracked mut prev = Knows::empty(inst_id);
                            vstd::modes::tracked_swap(&mut prev, &mut knows_map);
                            let tracked mut entries = prev.into_map();
                            entries.tracked_union_prefer_right(knows.get().into_map());
                            knows_map = Knows::from_map(inst_id, entries);
                            cited = cited.insert(event@);
                            if use_gpu {
                                flush.insert(pending.remove(mr.slot_loc(wr_id as nat)));
                            }
                        }
                    }
                    Completed::DataFailed { status, .. } => panic!("raw MW READ CQE status={status}"),
                    _ => panic!("unexpected CQE"),
                }
            }
        }
        if use_gpu {
            let Tracked(tokens) = mr.complete_gdr_slots(
                Tracked(caps.instance.borrow()), Tracked(flush), Tracked(&knows_map),
            ).map_err(|e| e.0).expect("flush reaped GPU slots before reuse");
            proof {
                owned.tracked_union_prefer_right(tokens);
            }
        }

        let now = Instant::now();
        for &slot in &reaped {
            if completed == job.warmup_iters && completed > 0 {
                sampler = ProcSampler::start();
                measure_start = now;
            }
            if completed >= job.warmup_iters {
                hist.record_ns(now.duration_since(submitted_at[slot]).as_nanos() as u64);
            }
            completed += 1;
            rings[slot_ring[slot]].free.push(slot);
        }
    }
    let total_ops = completed.saturating_sub(job.warmup_iters) as u64;

    proof {
        for slot in 0..qd {
            slot_toks.tracked_insert(slot, owned.tracked_remove(mr.slot_loc(slot as nat)));
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);

    let tracked empty_succ = SucceededMap::empty(caps.instance.borrow().id());
    for ring in rings {
        let tracked share = shares.tracked_remove(ring.cap.slot_ref().loc);
        channel
            .send_remote_read_release_with(
                ring.tag,
                b"raw_mw",
                Tracked(caps.instance.borrow()),
                Tracked(&sub_map),
                Tracked(&knows_map),
                Tracked(&empty_succ),
                Ghost(cited),
                Tracked(share),
                Tracked(()),
            )
            .map_err(|e| e.0)
            .expect("send ReadRelease");
    }

    let _ = deregister_memory_slab(
        &mut host_ctx,
        Tracked(caps.instance.borrow()),
        mr,
        Tracked(slot_toks),
        Tracked(&sub_map),
        Tracked(&knows_map),
        Ghost(cited),
    );
    for qp in qps {
        qp.destroy(&cq).expect("destroy QP");
    }
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
            "Pre-bound Type-2B MW + sliding batched READ (GPU{}->GPU{}, QPs={qp_count}, QD={qd}, flush group={refill}, disjoint ranges, {})",
            job.donor_gpu, job.initiator_gpu, job.rdma_dev
        )
    } else {
        format!(
            "Pre-bound Type-2B MW + sliding batched Host->Host READ (QPs={qp_count}, QD={qd}, refill={refill}, {})",
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
    let reset = UnconnectedQueuePair::new_with_config(
        pd,
        &cq,
        read_qp_config(device, num_windows),
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
        host.buffer(),
        host.tier_id(),
        host.first_frame(),
        host.num_frames(),
        flags,
    )
    .expect("register_host_slab");
    if let Some(c) = cuda {
        rdma.register_gpu_slab(
            &core,
            pd,
            c.gpu_buffer(),
            c.gpu_tier_id(),
            c.gpu_first_frame(),
            c.gpu_num_frames(),
            flags,
        )
        .expect("register_gpu_slab");
    }
    let peer = PeerSession::new(rdma, channel, &core);
    let eng_ctx = HostContext::new(caps.mint_host());
    let _ = num_frames;
    DistributedEngine::new_multi_peer(core, host, eng_ctx, vericache_core::engine::CudaAttachment::from_option_borrowed(cuda), None, vec![peer]).expect("assemble distributed engine")
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
                Tracked(&engine.core().ledger().borrow().submitted),
                Tracked(&engine.core().ledger().borrow().knows),
                Ghost(engine.core().citations()),
            );
            let ghost w_ev = w_comp.submitted.key();
            engine.record_host_completion(Tracked(w_comp));
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
            .poll_peer_event_from(0, Tracked(caps.instance.borrow()))
            .expect("donor poll_peer_event")
        {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key }) => {
                let res = engine
                    .serve_fetch_request_for_peer(0, serve_tier,
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
            DistributedEvent::Control(ControlPayload::FetchBatch { entries }) => {
                let posted = engine.grant_fetch_batch(
                    0, serve_tier, &entries, Tracked(caps.shares.borrow_mut()),
                    Tracked(caps.instance.borrow()),
                ).expect("grant fetch batch");
                assert_eq!(posted, entries.len(), "donor did not bind the full benchmark batch");
                let mut names = Vec::new();
                let deadline = Instant::now() + Duration::from_secs(60);
                while engine.peer_pending_grant_count(0).unwrap() != 0 {
                    assert!(Instant::now() < deadline, "batch bind completion timed out");
                    engine.pump_peer_nonblocking(0, None, &mut names, Tracked(caps.instance.borrow()),
                    ).expect("send batched grants");
                }
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
                .fetch_from_peer(0, keys[0], dst_tier, pull_target, Tracked(caps.instance.borrow()))
                .expect("fetch_remote_frame");
            let FetchOutcome::Hit { frame, .. } = outcome else {
                panic!("expected Hit");
            };
            pulled_frames.push(frame);
        } else {
            let base = engine.send_fetch_batch_to(0, &keys, Tracked(caps.instance.borrow()))
                .expect("send one fetch notice");
            let mut grants = Vec::with_capacity(window_w);
            let mut seen = vec![false; window_w];
            for _ in 0..window_w {
                match engine.poll_peer_event_from(0, Tracked(caps.instance.borrow())).expect("poll ReadGrant") {
                    DistributedEvent::ReadGrant(grant) => {
                        let i = usize::try_from(grant.meta().request_id - base).unwrap();
                        assert!(!seen[i] && keys[i] == grant.meta().key);
                        seen[i] = true;
                        grants.push(grant);
                    },
                    _ => panic!("expected ReadGrant in batch"),
                }
            }
            let outcome = engine.fetch_batch_from_peer(
                0, dst_tier, pull_target, grants, Tracked(caps.instance.borrow()),
            ).expect("pull batched grants");
            assert!(outcome.pending.is_empty(), "batch wait: {:?}", outcome.wait_error);
            for (_, result) in outcome.results {
                let FetchOutcome::Hit { frame, .. } = result.expect("batch READ failed") else {
                    panic!("expected Hit in batched pull");
                };
                pulled_frames.push(frame);
            }

        }
        let dt_ns = t0.elapsed().as_nanos() as u64;

        // Evict and reclaim local pulled frames so next iteration performs a fresh remote pull.
        for f in pulled_frames {
            engine.evict_frame(f).expect("evict pulled frame");
        }
        let reclaimed = engine.reclaim_draining(dst_tier, window_w + 4).unwrap();
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
        .send_control_to(0, ControlPayload::RevokeAck {
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
