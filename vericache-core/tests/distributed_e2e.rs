//! Real-hardware 2-node distributed RDMA & GPUDirect RDMA E2E suite for
//! `DistributedEngine<'a, TcpTransport>`.
//!
//! Covers all 4 Milestone 3 hardware scenarios across isolated donor (`NodeId 0`)
//! and initiator (`NodeId 1`) processes connected over TCP + ConnectX-7 RC RDMA:
//! 1. Host DRAM -> Host DRAM `RDMA_READ` pull + payload verification +
//!    cooperative `ReadRelease` + donor unbind/reclaim.
//! 2. GPU VRAM -> GPU VRAM GPUDirect RDMA pull (`post_rdma_read_gdr` +
//!    mandatory post-CQE `complete_gdr_slot` BAR1 visibility flush) across
//!    2 GPUs + D2H payload verification + cooperative release.
//! 3. Evict/reclaim local pulled replica followed by re-pull of the same
//!    `(Key, VersionId)` verifying `VersionBinding` version-value agreement.
//! 4. Option-A IBTA-strict unilateral MW revocation (`LOCAL_INV` +
//!    `ibv_dealloc_mw` -> `Quarantined`) -> initiator `RDMA_READ` fails with
//!    `IBV_WC_REM_ACCESS_ERR` -> `abort_write` -> QP `Error` -> `Reset` ->
//!    `reconnect_rc` -> clean re-grant and successful re-fetch.
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use transfer::context::*;
use transfer::cuda;
use transfer::domain::DeviceId;
#[cfg(verus_only)]
use transfer::domain::*;
use transfer::host::{self, HostContext};
use transfer::rdma::*;
use transfer::transport::{ByteTransport, Channel};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, PeerSession, ServeOutcome};
use vericache_core::backends::rdma::protocol::{ControlPayload, VeriCacheAttachment};
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::backends::rdma::RdmaBackend;
use vericache_core::engine::{DistributedEngine, DistributedEvent};
use vericache_core::frame::STATE_EMPTY;
use vericache_core::kernel::CacheCore;
#[cfg(verus_only)]
use vericache_core::provenance::*;
use vericache_core::spec::*;
use vericache_core::timeline::TimelineKind;
use vstd::prelude::*;

const FRAME_SIZE: usize = 4096;
const IBV_WC_REM_ACCESS_ERR: i32 = 10;

const ROLE: &str = "VERICACHE_DIST_E2E_ROLE";
const ADDRESS: &str = "VERICACHE_DIST_E2E_ADDRESS";
const SESSION: &str = "VERICACHE_DIST_E2E_SESSION";

struct TcpTransport(TcpStream);

impl TcpTransport {
    fn new(stream: TcpStream) -> Self {
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .unwrap();
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
        if n > 1024 * 1024 {
            return Err(io::Error::other("frame exceeds test limit"));
        }
        let mut frame = vec![0u8; n];
        self.0.read_exact(&mut frame)?;
        Ok(frame)
    }
}

fn require_hw() -> bool {
    std::env::var_os("VERICACHE_REQUIRE_HW").is_some()
}

fn gid_index() -> u8 {
    std::env::var("VERICACHE_RDMA_GID_INDEX")
        .map(|s| s.parse().expect("invalid VERICACHE_RDMA_GID_INDEX"))
        .unwrap_or(0)
}

fn gpu_ordinal(node: u64) -> Option<DeviceId> {
    let count = match cuda::device_count() {
        Ok(n) => n,
        Err(_) => 0,
    };
    let key = if node == 0 {
        "VERICACHE_GDR_DONOR_GPU"
    } else {
        "VERICACHE_GDR_INITIATOR_GPU"
    };
    let desired: u32 = std::env::var(key)
        .map(|s| s.parse().expect("invalid GDR GPU ordinal"))
        .unwrap_or(node as u32);
    if desired < count {
        Some(desired)
    } else {
        None
    }
}

fn pattern(seed: u8) -> Vec<u8> {
    (0..FRAME_SIZE)
        .map(|i| seed.wrapping_add(((i * 13 + 7) % 251) as u8))
        .collect()
}

fn spawn_role(test: &str, role: &str) -> Command {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", test, "--nocapture"]).env(ROLE, role);
    cmd
}

fn start_initiator_peer(test: &str) -> (TcpTransport, Child) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut secret = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut secret)
        .unwrap();
    let session: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    let mut process = spawn_role(test, "initiator")
        .env(SESSION, &session)
        .env(ADDRESS, listener.local_addr().unwrap().to_string())
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        match listener.accept() {
            Ok((s, _)) => {
                let mut t = TcpTransport::new(s);
                assert!(t.recv_frame().unwrap() == session.as_bytes()[..32]);
                t.send_frame(&session.as_bytes()[32..]).unwrap();
                return (t, process);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(status) = process.try_wait().unwrap() {
                    panic!("initiator exited before connect: {status}");
                }
                if Instant::now() >= until {
                    let _ = process.kill();
                    let _ = process.wait();
                    panic!("initiator connect timed out");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("accept: {e}"),
        }
    }
}

fn connect_to_donor() -> TcpTransport {
    let mut t = TcpTransport::new(TcpStream::connect(std::env::var(ADDRESS).unwrap()).unwrap());
    let session = std::env::var(SESSION).unwrap();
    t.send_frame(&session.as_bytes()[..32]).unwrap();
    assert!(t.recv_frame().unwrap() == session.as_bytes()[32..]);
    t
}

verus! {

#[cfg_attr(not(verus_only), allow(dead_code))]
struct NodeEnv {
    id: u64,
    instance: Tracked<TransferInstance>,
    alloc: Tracked<AllocSource>,
    events: Tracked<EventSourceAllocator>,
    shares: Tracked<ShareSource>,
    ver_instance: Tracked<VersionInstance>,
    ver_source: Option<Tracked<VersionSource>>,
}

impl NodeEnv {
    fn new(id: u64) -> Self {
        let (instance, alloc, events, shares, ver_instance, ver_source) = NodeContext::bootstrap(
            id,
        ).into_sources_with_versions();
        Self { id, instance, alloc, events, shares, ver_instance, ver_source: Some(ver_source) }
    }

    fn mint_rdma(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair))
    }

    fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::Host))
    }

    fn mint_cuda(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::CudaStream))
    }

    fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::IoUring))
    }

    fn make_core(&mut self) -> CacheCore {
        let mut core = CacheCore::new(
            self.id as u16,
            FRAME_SIZE,
            Vec::new(),
            8,
            Tracked(self.instance.borrow()),
        );
        let ver_src = self.ver_source.take().unwrap();
        let tracked vi = self.ver_instance.borrow().clone();
        core.enable_version_registry(Tracked(vi), ver_src);
        core
    }
}

fn select_rdma_device(node: &mut NodeEnv, requested: Option<&str>) -> Option<DeviceContext> {
    let count = DeviceContext::count().expect("enumerate verbs devices");
    for index in 0..count {
        let device = DeviceContext::open(index, node.id).expect("open verbs device");
        if requested.is_some_and(|name| name != device.name()) {
            device.close().unwrap();
            continue;
        }
        let type2 = device.supports_type_2().expect("query device capabilities");
        if type2 {
            let pd = ProtectionDomain::new(&device, node.mint_rdma()).expect("allocate probe PD");
            let supported = match alloc_window(&pd, node.mint_rdma()) {
                Ok(window) => {
                    dealloc_window(window).map_err(|e| e.0).expect("destroy probe window");
                    true
                },
                Err(RdmaError(38)) | Err(RdmaError(95)) => false,
                Err(e) => panic!("device {index}: ibv_alloc_mw(TYPE_2) failed: {e:?}"),
            };
            pd.destroy().expect("destroy probe PD");
            if supported {
                return Some(device);
            }
        }
        device.close().unwrap();
    }
    None
}

fn open_rdma_device(node: &mut NodeEnv, prefer_gdr: bool) -> Option<DeviceContext> {
    let requested = std::env::var("VERICACHE_RDMA_DEV").ok().or_else(
        || std::env::var("VERICACHE_RDMA_DEVICE").ok(),
    );
    let found = match requested.as_deref() {
        Some(name) => select_rdma_device(node, Some(name)),
        None if prefer_gdr => {
            select_rdma_device(node, Some("mlx5_4")).or_else(|| select_rdma_device(node, None))
        },
        None => select_rdma_device(node, None),
    };
    if found.is_none() {
        if require_hw() {
            panic!(
                "VERICACHE_REQUIRE_HW set, but no usable Type-2B RDMA device matching {}",
                requested.as_deref().unwrap_or("any")
            );
        }
        eprintln!(
            "SKIP distributed_e2e: no usable Type-2B RDMA device matching {}",
            requested.as_deref().unwrap_or("any")
        );
    }
    found
}

fn connect_peer_rdma<'a>(
    node: &mut NodeEnv,
    core: &mut CacheCore,
    pd: &'a ProtectionDomain<'a>,
    device: &'a DeviceContext,
    transport: TcpTransport,
    host: &'a HostBackend,
    cuda: Option<&'a CudaBackend<'a>>,
) -> PeerSession<'a, TcpTransport> {
    let peer_id = 1 - node.id;
    let cq = CompletionQueue::new(device, 64, node.mint_rdma()).unwrap();
    let reset = UnconnectedQueuePair::new(
        pd,
        &cq,
        32,
        1,
        gid_index(),
        0x420 + node.id as u32,
        node.mint_rdma(),
    ).expect("create RC QP");
    let mut channel: Channel<TcpTransport, VeriCacheAttachment> = Channel::new_with_protocol(
        transport,
        node.id,
        peer_id,
        Tracked(node.instance.borrow()),
    );
    let peer_ep = channel.exchange_queue_pair_endpoint(
        &reset.endpoint(),
        Tracked(node.instance.borrow()),
    ).unwrap();
    let qp = connect_rc(reset, peer_ep, peer_id).expect("connect RC QP");
    let mut windows = Vec::new();
    let mut i = 0usize;
    while i < 4 {
        windows.push(alloc_window(pd, node.mint_rdma()).expect("alloc Type-2 MW"));
        i += 1;
    }
    let mut rdma = RdmaBackend::new(core, peer_id, qp, cq, windows).map_err(|e| e.0).unwrap();
    let flags = AccessFlags { local_write: true, mw_bind: true };
    rdma.register_host_slab(
        core,
        pd,
        &host.buffer,
        host.tier,
        host.first_frame,
        host.num_frames,
        flags,
    ).expect("register host slab MR");
    if let Some(c) = cuda {
        rdma.register_gpu_slab(
            core,
            pd,
            &c.gpu_buffer,
            c.gpu_tier,
            c.gpu_first_frame,
            c.gpu_num_frames,
            flags,
        ).expect("register GPU slab MR for GPUDirect RDMA");
    }
    PeerSession::new(rdma, channel, core)
}

// =========================================================================
// Scenario 1: Host DRAM -> Host DRAM RDMA_READ pull + cooperative release
//             + donor unbind/reclaim
// =========================================================================
fn run_scenario1_host_pull(node: &mut NodeEnv, device: &DeviceContext, transport: TcpTransport) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host_src = node.mint_host();
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        host_src,
        4,
        4096,
    ).unwrap();
    let peer = connect_peer_rdma(node, &mut core, &pd, device, transport, &host, None);
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, None, peer);
    let host_tier = engine.host_tier();
    let key = ObjectKey([11, 22, 33, 44]);
    let expected = pattern(0x2A);

    if node.id == 0 {
        let (ver, f_donor) = engine.put_host(
            Tracked(node.instance.borrow()),
            key,
            &expected,
        ).unwrap();
        let req_id = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest"),
        };
        let grant_tag = match engine.serve_fetch_request(
            host_tier,
            req_id,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed published host key"),
        };
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag && was_active);
            },
            _ => panic!("donor expected cooperative ReadReleased"),
        }
        assert!(engine.peers[0].rdma.active_lease_count() == 0);
        // Donor evicts and reclaims the frame now that the remote lease was cooperatively released.
        engine.evict_frame(f_donor).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        assert!(engine.core.frames.metas[f_donor.0 as usize].state == STATE_EMPTY);
        engine.send_control(
            ControlPayload::RevokeAck { grant_tag: ver.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        let outcome = engine.fetch_remote_frame(
            key,
            host_tier,
            PullTarget::Host,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let FetchOutcome::Hit { frame: _f_init, version: ver_init } = outcome else {
            panic!("initiator expected Hit");
        };
        let mut out = vec![0u8; FRAME_SIZE];
        let got = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut out,
        ).unwrap();
        assert!(got == Some(ver_init));
        assert!(out == expected);
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: donor_seq }) => {
                assert!(donor_seq == ver_init.seq);
            },
            _ => panic!("initiator expected donor completion ack"),
        }
    }
}

// =========================================================================
// Scenario 2: GPU VRAM -> GPU VRAM GPUDirect RDMA pull + mandatory post-CQE
//             complete_gdr_slot BAR1 visibility flush + D2H verification
// =========================================================================
fn run_scenario2_gdr_gpu_pull(
    node: &mut NodeEnv,
    device: &DeviceContext,
    gpu_dev: DeviceId,
    transport: TcpTransport,
) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host_src = node.mint_host();
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        host_src,
        2,
        4096,
    ).unwrap();
    let cuda_src = node.mint_cuda();
    let cuda = CudaBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        cuda_src,
        gpu_dev,
        4,
        16,
        true,
    ).unwrap();
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: gpu_dev }).unwrap();
    let mut caller_stream = cuda::CudaStream::new(gpu_dev, node.mint_cuda()).unwrap();
    let peer = connect_peer_rdma(node, &mut core, &pd, device, transport, &host, Some(&cuda));
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, Some(&cuda), peer);
    let gpu_tier = engine.gpu_tier().unwrap();
    let key = ObjectKey([55, 66, 77, 88]);
    let expected = pattern(0x5C);

    if node.id == 0 {
        // Stage payload into pinned host buffer, then publish into donor GPU VRAM via put_gpu_sync.
        let (stage_buf, Tracked(mut stage_tok)) = cuda::alloc_host(
            Tracked(node.alloc.borrow_mut()),
            FRAME_SIZE,
        ).unwrap();
        let mut stage_ctx = HostContext::new(node.mint_host());
        let stage_host_slice = stage_buf.as_host_slice();
        let Tracked(w_comp) = host::write_slice(
            &mut stage_ctx,
            Tracked(node.instance.borrow()),
            &stage_host_slice,
            &expected,
            Tracked(&mut stage_tok),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        let ghost w_ev = w_comp.submitted.key();
        let t_w = engine.core.issue_ticket(
            engine.host.timeline_id(),
            Tracked(w_comp.submitted),
        ).unwrap();
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
            bytes: expected@,
            proof: ValueProof::HostWrite,
        };
        let (ver, f_gpu) = engine.put_gpu_sync(
            Tracked(node.instance.borrow()),
            &mut caller_stream,
            caller_tl,
            key,
            &stage_cuda,
            Tracked(&mut stage_perm),
            Ghost(stage_cert),
        ).unwrap();

        let req_id = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("GDR donor expected FetchRequest"),
        };
        let grant_tag = match engine.serve_fetch_request(
            gpu_tier,
            req_id,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("GDR donor unexpectedly missed GPU key"),
        };
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag && was_active);
            },
            _ => panic!("GDR donor expected cooperative ReadReleased"),
        }
        assert!(engine.peers[0].rdma.active_lease_count() == 0);
        engine.evict_frame(f_gpu).unwrap();
        assert!(engine.reclaim_draining(gpu_tier, 4) == 1);
        assert!(engine.core.frames.metas[f_gpu.0 as usize].state == STATE_EMPTY);
        engine.send_control(
            ControlPayload::RevokeAck { grant_tag: ver.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        // Pull directly into initiator GPU VRAM via GPUDirect RDMA + complete_gdr_slot BAR1 flush.
        let outcome = engine.fetch_remote_frame(
            key,
            gpu_tier,
            PullTarget::GpuGdr,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let FetchOutcome::Hit { frame: _f_gpu, version: ver_init } = outcome else {
            panic!("GDR initiator expected Hit");
        };
        // Read pulled GPU VRAM frame back to pinned host buffer via get_into_gpu (D2H) and verify.
        let (rb_buf, Tracked(mut rb_tok)) = cuda::alloc_host(
            Tracked(node.alloc.borrow_mut()),
            FRAME_SIZE,
        ).unwrap();
        let rb_cuda = rb_buf.as_cuda_slice();
        let (got_ver, t_d2h, _ev_d2h) = engine.get_into_gpu(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut caller_stream,
            caller_tl,
            &rb_cuda,
            Tracked(&mut rb_tok),
        ).unwrap().unwrap();
        assert!(got_ver == ver_init);
        let Tracked(k_d2h) = cuda::stream_synchronize(
            &caller_stream,
            Tracked(node.instance.borrow()),
        ).unwrap();
        engine.core.mark_timeline_complete(caller_tl, t_d2h.seq(), Tracked(k_d2h));

        let mut rb_ctx = HostContext::new(node.mint_host());
        let rb_host = rb_buf.as_host_slice();
        let tracked mut rb_perm = ReadPerm::Owned(rb_tok);
        let mut out = vec![0u8; FRAME_SIZE];
        let _r_comp = host::read_slice(
            &mut rb_ctx,
            Tracked(node.instance.borrow()),
            &rb_host,
            &mut out,
            Tracked(&mut rb_perm),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        assert!(out == expected);
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: donor_seq }) => {
                assert!(donor_seq == ver_init.seq);
            },
            _ => panic!("GDR initiator expected donor completion ack"),
        }
    }
    let _ = caller_stream.destroy();
}

// =========================================================================
// Scenario 3: Evict/reclaim local pulled replica followed by re-pull of the
//             same (Key, VersionId) verifying VersionBinding agreement
// =========================================================================
fn run_scenario3_evict_repull(node: &mut NodeEnv, device: &DeviceContext, transport: TcpTransport) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host_src = node.mint_host();
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        host_src,
        4,
        4096,
    ).unwrap();
    let peer = connect_peer_rdma(node, &mut core, &pd, device, transport, &host, None);
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, None, peer);
    let host_tier = engine.host_tier();
    let key = ObjectKey([101, 102, 103, 104]);
    let expected = pattern(0x7E);

    if node.id == 0 {
        let (_ver, f_donor) = engine.put_host(
            Tracked(node.instance.borrow()),
            key,
            &expected,
        ).unwrap();
        // Serve two sequential pulls of the same (Key, VersionId).
        let mut round = 0usize;
        while round < 2 {
            let req_id = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
                DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                    assert!(k == key);
                    request_id
                },
                _ => panic!("donor expected FetchRequest"),
            };
            let grant_tag = match engine.serve_fetch_request(
                host_tier,
                req_id,
                key,
                Tracked(node.shares.borrow_mut()),
                Tracked(node.instance.borrow()),
            ).unwrap() {
                ServeOutcome::Granted { grant_tag, .. } => grant_tag,
                ServeOutcome::Miss => panic!("donor unexpectedly missed key"),
            };
            match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert!(tag == grant_tag && was_active);
                },
                _ => panic!("donor expected ReadReleased"),
            }
            round += 1;
        }
        engine.evict_frame(f_donor).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
    } else {
        // Pull #1.
        let FetchOutcome::Hit { frame: f1, version: v1 } = engine.fetch_remote_frame(
            key,
            host_tier,
            PullTarget::Host,
            Tracked(node.instance.borrow()),
        ).unwrap() else {
            panic!("initiator expected Hit on first pull");
        };
        let mut out1 = vec![0u8; FRAME_SIZE];
        let got1 = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut out1,
        ).unwrap();
        assert!(got1 == Some(v1) && out1 == expected);

        // Evict & reclaim the local pulled replica completely.
        engine.evict_frame(f1).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        assert!(engine.lookup_tier(key, host_tier).is_none());
        assert!(engine.core.frames.metas[f1.0 as usize].state == STATE_EMPTY);

        // Pull #2 of the exact same (Key, VersionId) verifies VersionBinding agreement.
        let FetchOutcome::Hit { frame: _f2, version: v2 } = engine.fetch_remote_frame(
            key,
            host_tier,
            PullTarget::Host,
            Tracked(node.instance.borrow()),
        ).unwrap() else {
            panic!("initiator expected Hit on re-pull");
        };
        assert!(v2 == v1);
        let mut out2 = vec![0u8; FRAME_SIZE];
        let got2 = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut out2,
        ).unwrap();
        assert!(got2 == Some(v1) && out2 == expected);
    }
}

// =========================================================================
// Scenario 4: Option-A IBTA-strict unilateral MW revocation ->
//             IBV_WC_REM_ACCESS_ERR -> abort_write -> QP Error -> Reset ->
//             reconnect_rc -> clean re-grant and successful re-fetch
// =========================================================================
fn run_scenario4_unilateral_revoke_and_reconnect(
    node: &mut NodeEnv,
    device: &DeviceContext,
    transport: TcpTransport,
) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host_src = node.mint_host();
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        host_src,
        4,
        4096,
    ).unwrap();
    let peer = connect_peer_rdma(node, &mut core, &pd, device, transport, &host, None);
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, None, peer);
    let host_tier = engine.host_tier();
    let key = ObjectKey([201, 202, 203, 204]);
    let expected = pattern(0xA5);

    if node.id == 0 {
        let (ver, f_donor) = engine.put_host(
            Tracked(node.instance.borrow()),
            key,
            &expected,
        ).unwrap();

        // 1. Serve FetchRequest #1, then immediately perform Option-A unilateral MW revocation.
        let req_id1 = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest #1"),
        };
        let grant_tag1 = match engine.serve_fetch_request(
            host_tier,
            req_id1,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed key"),
        };
        // Unilateral revocation: LOCAL_INV + ibv_dealloc_mw -> Quarantined + RevokeNotice.
        let mw_slot = engine.revoke_lease_unilateral(
            grant_tag1,
            Tracked(node.instance.borrow()),
        ).unwrap();
        assert!(engine.peers[0].rdma.active_lease_count() == 0);

        // 2. Drain initiator's post-abort ReadRelease before QP Error -> Reset -> Reconnect.
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag1 && !was_active);
            },
            _ => panic!("donor expected post-abort ReadReleased"),
        }

        // 3. Option-A QP recovery: Error -> Reset (unquarantines slot -> Vacant) -> Reconnect.
        let _err_w = engine.enter_qp_error().unwrap();
        engine.reset_and_reconnect(0x5678, Tracked(node.instance.borrow())).unwrap();
        engine.replenish_window(mw_slot, &pd, node.mint_rdma()).unwrap();

        // 4. Clean re-grant and cooperative release on the reconnected QP.
        let req_id2 = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest #2 after reconnect"),
        };
        let grant_tag2 = match engine.serve_fetch_request(
            host_tier,
            req_id2,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed key on re-fetch"),
        };
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag2 && was_active);
            },
            _ => panic!("donor expected cooperative ReadReleased #2"),
        }
        engine.evict_frame(f_donor).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        engine.send_control(
            ControlPayload::RevokeAck { grant_tag: ver.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        // 1. Request key and hold the received ReadGrant until donor's RevokeNotice arrives.
        let _req_id1 = engine.send_fetch_request(key, Tracked(node.instance.borrow())).unwrap();
        let grant = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadGrant(g) => g,
            _ => panic!("initiator expected ReadGrant #1"),
        };
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                assert!(grant_tag == grant.tag);
            },
            _ => panic!("initiator expected RevokeNotice"),
        }

        // 2. Attempt RDMA_READ against the revoked MW -> IBV_WC_REM_ACCESS_ERR (10) + abort_write.
        let ready_before = engine.core.frames.ready_queue[host_tier.as_index()].len();
        let pull_err = engine.pull_with_grant(
            key,
            host_tier,
            PullTarget::Host,
            grant,
            Tracked(node.instance.borrow()),
        );
        match pull_err {
            Err(CoreError::Backend { kind: BACKEND_RDMA, code }) => {
                assert!(code != 0);
                if device.name() == "mlx5_0" || device.name().starts_with("mlx5_") {
                    assert!(code == IBV_WC_REM_ACCESS_ERR);
                }
            },
            other => panic!("expected RDMA remote access error after revocation, got {other:?}"),
        }
        // Verify abort_write returned the destination frame cleanly without publishing.
        assert!(engine.lookup_tier(key, host_tier).is_none());
        assert!(engine.core.frames.ready_queue[host_tier.as_index()].len() == ready_before);

        // 3. Option-A QP recovery: Error -> Reset -> Reconnect with fresh PSN.
        let _err_w = engine.enter_qp_error().unwrap();
        engine.reset_and_reconnect(0x1234, Tracked(node.instance.borrow())).unwrap();

        // 4. Clean re-fetch succeeds over the reconnected QP with intact payload.
        let FetchOutcome::Hit { frame: _f_ok, version: ver_ok } = engine.fetch_remote_frame(
            key,
            host_tier,
            PullTarget::Host,
            Tracked(node.instance.borrow()),
        ).unwrap() else {
            panic!("initiator expected Hit on re-fetch after reconnect");
        };
        let mut out = vec![0u8; FRAME_SIZE];
        let got = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut out,
        ).unwrap();
        assert!(got == Some(ver_ok));
        assert!(out == expected);
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: donor_seq }) => {
                assert!(donor_seq == ver_ok.seq);
            },
            _ => panic!("initiator expected donor final ack"),
        }
    }
}

// =========================================================================
// Scenario 5: 4-Tier DistributedEngine (HBM + DRAM + Local NVMe io_uring +
//             2-GPU GPUDirect RDMA) full roundtrip + async NVMe progress
// =========================================================================
fn run_scenario5_four_tier_roundtrip(
    node: &mut NodeEnv,
    device: &DeviceContext,
    gpu_dev: DeviceId,
    transport: TcpTransport,
) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host_src = node.mint_host();
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        host_src,
        4,
        4096,
    ).unwrap();
    let cuda_src = node.mint_cuda();
    let cuda = CudaBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        cuda_src,
        gpu_dev,
        4,
        16,
        true,
    ).unwrap();
    let nvme_src = node.mint_nvme();
    let nvme = NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        nvme_src,
        "/tmp",
        4,
        16,
    ).expect("initialize O_DIRECT io_uring NvmeBackend");
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: gpu_dev }).unwrap();
    let mut caller_stream = cuda::CudaStream::new(gpu_dev, node.mint_cuda()).unwrap();
    let peer = connect_peer_rdma(node, &mut core, &pd, device, transport, &host, Some(&cuda));
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new_with_nvme(
        core,
        &host,
        eng_ctx,
        Some(&cuda),
        peer,
        Some(nvme),
    );
    let host_tier = engine.host_tier();
    let gpu_tier = engine.gpu_tier().unwrap();
    let nvme_tier = engine.nvme_tier().unwrap();
    let key = ObjectKey([41, 42, 43, 44]);
    let key_async = ObjectKey([45, 46, 47, 48]);
    let expected = pattern(0xD4);
    let expected_async = pattern(0x3B);

    if node.id == 0 {
        // 1. Publish payload into Donor H20 GPU HBM via put_gpu_sync.
        let (stage_buf, Tracked(mut stage_tok)) = cuda::alloc_host(
            Tracked(node.alloc.borrow_mut()),
            FRAME_SIZE,
        ).unwrap();
        let mut stage_ctx = HostContext::new(node.mint_host());
        let stage_host_slice = stage_buf.as_host_slice();
        let Tracked(w_comp) = host::write_slice(
            &mut stage_ctx,
            Tracked(node.instance.borrow()),
            &stage_host_slice,
            &expected,
            Tracked(&mut stage_tok),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        let ghost w_ev = w_comp.submitted.key();
        let t_w = engine.core.issue_ticket(
            engine.host.timeline_id(),
            Tracked(w_comp.submitted),
        ).unwrap();
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
            bytes: expected@,
            proof: ValueProof::HostWrite,
        };
        let (ver_gpu1, f_gpu1) = engine.put_gpu_sync(
            Tracked(node.instance.borrow()),
            &mut caller_stream,
            caller_tl,
            key,
            &stage_cuda,
            Tracked(&mut stage_perm),
            Ghost(stage_cert),
        ).unwrap();

        // 2. Copy H20 GPU HBM -> Host DRAM (D2H readout -> put_host) and reclaim GPU frame.
        let (d2h_buf, Tracked(mut d2h_tok)) = cuda::alloc_host(
            Tracked(node.alloc.borrow_mut()),
            FRAME_SIZE,
        ).unwrap();
        let d2h_cuda = d2h_buf.as_cuda_slice();
        let (got_v1, t_d2h1, _ev_d2h1) = engine.get_into_gpu(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut caller_stream,
            caller_tl,
            &d2h_cuda,
            Tracked(&mut d2h_tok),
        ).unwrap().unwrap();
        assert!(got_v1 == ver_gpu1);
        let Tracked(k_d2h1) = cuda::stream_synchronize(
            &caller_stream,
            Tracked(node.instance.borrow()),
        ).unwrap();
        engine.core.mark_timeline_complete(caller_tl, t_d2h1.seq(), Tracked(k_d2h1));

        let mut d2h_ctx = HostContext::new(node.mint_host());
        let d2h_host = d2h_buf.as_host_slice();
        let tracked mut d2h_perm = ReadPerm::Owned(d2h_tok);
        let mut from_gpu = vec![0u8; FRAME_SIZE];
        let _r1 = host::read_slice(
            &mut d2h_ctx,
            Tracked(node.instance.borrow()),
            &d2h_host,
            &mut from_gpu,
            Tracked(&mut d2h_perm),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        assert!(from_gpu == expected);

        engine.evict_frame(f_gpu1).unwrap();
        assert!(engine.reclaim_draining(gpu_tier, 4) == 1);
        assert!(engine.lookup_tier(key, gpu_tier).is_none());

        let (ver_host, f_host1) = engine.put_host(
            Tracked(node.instance.borrow()),
            key,
            &from_gpu,
        ).unwrap();

        // 3. Spill Host DRAM -> Local NVMe (O_DIRECT io_uring) sync + async poll/wait check.
        let f_nvme = engine.spill_host_to_nvme_sync(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
        ).unwrap();
        assert!(engine.lookup_tier(key, nvme_tier) == Some((f_nvme, ver_host)));

        // Exercise async spill_host_to_nvme + wait_nvme_completion + poll_nvme_completions.
        let (_ver_async, f_async_host) = engine.put_host(
            Tracked(node.instance.borrow()),
            key_async,
            &expected_async,
        ).unwrap();
        let (f_async_nvme, _t_async) = engine.spill_host_to_nvme(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key_async,
        ).unwrap();
        let prog_wait = engine.wait_nvme_completion(Tracked(node.instance.borrow())).unwrap();
        assert!(prog_wait.completed == 1 && prog_wait.failed == 0);
        let prog_poll = engine.poll_nvme_completions(Tracked(node.instance.borrow()), 8).unwrap();
        assert!(prog_poll.completed == 0 && prog_poll.failed == 0);
        assert!(engine.lookup_tier(key_async, nvme_tier).is_some());
        let _ = f_async_nvme;
        engine.evict_frame(f_async_host).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);

        // 4. Evict & reclaim Host DRAM copy of `key` so NVMe is the sole replica.
        engine.evict_frame(f_host1).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        assert!(engine.lookup_tier(key, host_tier).is_none());
        assert!(engine.lookup_tier(key, gpu_tier).is_none());
        assert!(engine.core.frames.metas[f_host1.0 as usize].state == STATE_EMPTY);

        // 5. Warm Host DRAM from Local NVMe via io_uring O_DIRECT and verify bytes.
        let f_host2 = engine.warm_host_from_nvme_sync(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
        ).unwrap();
        assert!(engine.lookup_tier(key, host_tier) == Some((f_host2, ver_host)));
        let mut warmed = vec![0u8; FRAME_SIZE];
        let got_warm = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut warmed,
        ).unwrap();
        assert!(got_warm == Some(ver_host));
        assert!(warmed == expected);

        // 6. Promote warmed Host payload into Donor H20 GPU HBM and serve Peer 2-GPU GDR pull.
        let (stage2_buf, Tracked(mut stage2_tok)) = cuda::alloc_host(
            Tracked(node.alloc.borrow_mut()),
            FRAME_SIZE,
        ).unwrap();
        let mut stage2_ctx = HostContext::new(node.mint_host());
        let stage2_host_slice = stage2_buf.as_host_slice();
        let Tracked(w2_comp) = host::write_slice(
            &mut stage2_ctx,
            Tracked(node.instance.borrow()),
            &stage2_host_slice,
            &warmed,
            Tracked(&mut stage2_tok),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        let ghost w2_ev = w2_comp.submitted.key();
        let t_w2 = engine.core.issue_ticket(
            engine.host.timeline_id(),
            Tracked(w2_comp.submitted),
        ).unwrap();
        engine.core.mark_timeline_complete(
            engine.host.timeline_id(),
            t_w2.seq(),
            Tracked(w2_comp.knows),
        );
        let stage2_cuda = stage2_buf.as_cuda_slice();
        let tracked mut stage2_perm = ReadPerm::Owned(stage2_tok);
        let ghost stage2_cert = ValueCertificate {
            slot: stage2_cuda.slot_ref(),
            writer: w2_ev,
            bytes: warmed@,
            proof: ValueProof::HostWrite,
        };
        let (ver_gpu2, f_gpu2) = engine.put_gpu_sync(
            Tracked(node.instance.borrow()),
            &mut caller_stream,
            caller_tl,
            key,
            &stage2_cuda,
            Tracked(&mut stage2_perm),
            Ghost(stage2_cert),
        ).unwrap();

        let req_id = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("4-tier donor expected FetchRequest"),
        };
        let grant_tag = match engine.serve_fetch_request(
            gpu_tier,
            req_id,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("4-tier donor unexpectedly missed GPU key"),
        };
        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag && was_active);
            },
            _ => panic!("4-tier donor expected cooperative ReadReleased"),
        }
        assert!(engine.peers[0].rdma.active_lease_count() == 0);
        engine.evict_frame(f_gpu2).unwrap();
        assert!(engine.reclaim_draining(gpu_tier, 4) == 1);
        assert!(engine.core.frames.metas[f_gpu2.0 as usize].state == STATE_EMPTY);
        engine.send_control(
            ControlPayload::RevokeAck { grant_tag: ver_gpu2.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        // Peer 2-GPU GPUDirect RDMA pull into Initiator H20 GPU HBM + BAR1 flush + D2H verification.
        let outcome = engine.fetch_remote_frame(
            key,
            gpu_tier,
            PullTarget::GpuGdr,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let FetchOutcome::Hit { frame: _f_gpu, version: ver_init } = outcome else {
            panic!("4-tier initiator expected Hit");
        };
        let (rb_buf, Tracked(mut rb_tok)) = cuda::alloc_host(
            Tracked(node.alloc.borrow_mut()),
            FRAME_SIZE,
        ).unwrap();
        let rb_cuda = rb_buf.as_cuda_slice();
        let (got_ver, t_d2h, _ev_d2h) = engine.get_into_gpu(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut caller_stream,
            caller_tl,
            &rb_cuda,
            Tracked(&mut rb_tok),
        ).unwrap().unwrap();
        assert!(got_ver == ver_init);
        let Tracked(k_d2h) = cuda::stream_synchronize(
            &caller_stream,
            Tracked(node.instance.borrow()),
        ).unwrap();
        engine.core.mark_timeline_complete(caller_tl, t_d2h.seq(), Tracked(k_d2h));

        let mut rb_ctx = HostContext::new(node.mint_host());
        let rb_host = rb_buf.as_host_slice();
        let tracked mut rb_perm = ReadPerm::Owned(rb_tok);
        let mut out = vec![0u8; FRAME_SIZE];
        let _r_comp = host::read_slice(
            &mut rb_ctx,
            Tracked(node.instance.borrow()),
            &rb_host,
            &mut out,
            Tracked(&mut rb_perm),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        assert!(out == expected);

        // Also exercise initiator-side Host -> NVMe spill + reclaim + NVMe warm.
        let (v_local, f_init_host) = engine.put_host(
            Tracked(node.instance.borrow()),
            key,
            &out,
        ).unwrap();
        let _f_init_nvme = engine.spill_host_to_nvme_sync(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
        ).unwrap();
        engine.evict_frame(f_init_host).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        let _f_init_warm = engine.warm_host_from_nvme_sync(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
        ).unwrap();
        let mut out_nvme = vec![0u8; FRAME_SIZE];
        let got_nvme = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key,
            &mut out_nvme,
        ).unwrap();
        assert!(got_nvme == Some(v_local));
        assert!(out_nvme == expected);

        match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: donor_seq }) => {
                assert!(donor_seq == ver_init.seq);
            },
            _ => panic!("4-tier initiator expected donor completion ack"),
        }
    }
    let _ = caller_stream.destroy();
}

} // verus!
#[derive(Clone, Copy)]
enum Scenario {
    HostPull,
    GdrGpuPull,
    EvictRepull,
    UnilateralRevokeReconnect,
    FourTierRoundtrip,
}

fn dispatch_scenario(test_name: &str, scenario: Scenario) {
    let prefer_gdr = matches!(scenario, Scenario::GdrGpuPull | Scenario::FourTierRoundtrip);
    match std::env::var(ROLE).as_deref() {
        Ok("donor") => {
            let mut node = NodeEnv::new(0);
            let Some(device) = open_rdma_device(&mut node, prefer_gdr) else {
                return;
            };
            let gpu_opt = if prefer_gdr {
                let d0 = gpu_ordinal(0);
                let d1 = gpu_ordinal(1);
                if d0.is_none() || d1.is_none() {
                    if require_hw() {
                        panic!("VERICACHE_REQUIRE_HW set, but 2 CUDA GPUs unavailable for GDR");
                    }
                    eprintln!("SKIP {test_name}: 2 CUDA GPUs unavailable for GDR");
                    return;
                }
                d0
            } else {
                None
            };
            let (transport, mut child) = start_initiator_peer(test_name);
            match scenario {
                Scenario::HostPull => run_scenario1_host_pull(&mut node, &device, transport),
                Scenario::GdrGpuPull => {
                    run_scenario2_gdr_gpu_pull(&mut node, &device, gpu_opt.unwrap(), transport)
                }
                Scenario::EvictRepull => run_scenario3_evict_repull(&mut node, &device, transport),
                Scenario::UnilateralRevokeReconnect => {
                    run_scenario4_unilateral_revoke_and_reconnect(&mut node, &device, transport)
                }
                Scenario::FourTierRoundtrip => run_scenario5_four_tier_roundtrip(
                    &mut node,
                    &device,
                    gpu_opt.unwrap(),
                    transport,
                ),
            }
            assert!(child.wait().unwrap().success());
        }
        Ok("initiator") => {
            let mut node = NodeEnv::new(1);
            let device = open_rdma_device(&mut node, prefer_gdr).unwrap();
            let transport = connect_to_donor();
            match scenario {
                Scenario::HostPull => run_scenario1_host_pull(&mut node, &device, transport),
                Scenario::GdrGpuPull => {
                    let gpu = gpu_ordinal(1).unwrap();
                    run_scenario2_gdr_gpu_pull(&mut node, &device, gpu, transport)
                }
                Scenario::EvictRepull => run_scenario3_evict_repull(&mut node, &device, transport),
                Scenario::UnilateralRevokeReconnect => {
                    run_scenario4_unilateral_revoke_and_reconnect(&mut node, &device, transport)
                }
                Scenario::FourTierRoundtrip => {
                    let gpu = gpu_ordinal(1).unwrap();
                    run_scenario5_four_tier_roundtrip(&mut node, &device, gpu, transport)
                }
            }
        }
        _ => {
            let status = spawn_role(test_name, "donor").status().unwrap();
            assert!(
                status.success(),
                "{test_name} donor subprocess failed: {status}"
            );
        }
    }
}

#[test]
fn distributed_host_rdma_pull_release_reclaim() {
    dispatch_scenario(
        "distributed_host_rdma_pull_release_reclaim",
        Scenario::HostPull,
    );
}

#[test]
fn distributed_gdr_two_gpu_pull_flush_release() {
    dispatch_scenario(
        "distributed_gdr_two_gpu_pull_flush_release",
        Scenario::GdrGpuPull,
    );
}

#[test]
fn distributed_evict_reclaim_and_repull_version_agreement() {
    dispatch_scenario(
        "distributed_evict_reclaim_and_repull_version_agreement",
        Scenario::EvictRepull,
    );
}

#[test]
fn distributed_option_a_unilateral_revoke_error_reset_reconnect_refetch() {
    dispatch_scenario(
        "distributed_option_a_unilateral_revoke_error_reset_reconnect_refetch",
        Scenario::UnilateralRevokeReconnect,
    );
}

#[test]
fn distributed_four_tier_hbm_dram_nvme_rdma_roundtrip() {
    dispatch_scenario(
        "distributed_four_tier_hbm_dram_nvme_rdma_roundtrip",
        Scenario::FourTierRoundtrip,
    );
}
