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
use transfer::transport::{ByteTransport, Channel, ReceivedFrame};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::peer::{FetchOutcome, PeerSession, ServeOutcome};
use vericache_core::backends::rdma::protocol::{ControlPayload, VeriCacheAttachment};
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::backends::rdma::{RdmaBackend, WindowInvalidation};
use vericache_core::backends::NoDevice;
use vericache_core::engine::{
    CudaAttachment, DistributedEngine, DistributedEvent, LocalEngine, PendingReadGrant,
};
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

type FrameCapture = std::rc::Rc<std::cell::RefCell<Vec<u8>>>;
struct TcpTransport {
    stream: TcpStream,
    capture: Option<FrameCapture>,
    pending: Vec<u8>,
}

struct ProbeFrame(Vec<u8>);

impl ByteTransport for ProbeFrame {
    type Error = io::Error;

    fn send_frame(&mut self, _: &[u8]) -> io::Result<()> {
        Err(io::Error::other("read-only test frame"))
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        Ok(std::mem::take(&mut self.0))
    }
}

impl TcpTransport {
    fn new(stream: TcpStream) -> Self {
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        Self {
            stream,
            capture: None,
            pending: Vec::new(),
        }
    }

    fn read_frame(&mut self) -> io::Result<Vec<u8>> {
        // Keep partial reads across pump calls. Reading one frame at a time also
        // lets tests hand the stream off to another reader after a control exchange.
        let mut bytes = [0u8; 4096];
        loop {
            let needed = if self.pending.len() < 4 {
                4
            } else {
                let len = u32::from_be_bytes(self.pending[..4].try_into().unwrap()) as usize;
                if len > 1024 * 1024 {
                    return Err(io::Error::other("frame exceeds test limit"));
                }
                4 + len
            };
            if self.pending.len() == needed && self.pending.len() >= 4 {
                let frame = self.pending.split_off(4);
                self.pending.clear();
                return Ok(frame);
            }
            let limit = (needed - self.pending.len()).min(bytes.len());
            match self.stream.read(&mut bytes[..limit]) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => self.pending.extend_from_slice(&bytes[..n]),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
    }
}

impl ByteTransport for TcpTransport {
    type Error = io::Error;

    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        if let Some(capture) = &self.capture {
            let mut saved = capture.borrow_mut();
            saved.clear();
            saved.extend_from_slice(frame);
        }
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        self.stream.write_all(&n.to_be_bytes())?;
        self.stream.write_all(frame)
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        self.read_frame()
    }

    fn try_recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        self.stream.set_nonblocking(true)?;
        let result = self.read_frame();
        self.stream.set_nonblocking(false)?;
        match result {
            Ok(frame) => Ok(Some(frame)),
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(err) => Err(err),
        }
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
    if role.starts_with("tenant-") {
        // Debug builds temporarily hold several inline RDMA completion tables.
        cmd.env("RUST_MIN_STACK", (16 * 1024 * 1024).to_string());
    }
    cmd
}

fn start_initiator_peer(test: &str) -> (TcpTransport, Child) {
    start_test_peer(test, "initiator")
}

fn start_test_peer(test: &str, role: &str) -> (TcpTransport, Child) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut secret = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut secret)
        .unwrap();
    let session: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    let mut process = spawn_role(test, role)
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
        Tracked(self.events.borrow_mut().mint(EventSourceKind::rdma_qp()))
    }

    fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::host()))
    }

    fn mint_cuda(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::cuda_stream()))
    }

    fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::io_uring()))
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
    connect_peer_rdma_for(node, 1 - node.id, core, pd, device, transport, host, cuda)
}

fn connect_peer_rdma_for<'a>(
    node: &mut NodeEnv,
    peer_id: u64,
    core: &mut CacheCore,
    pd: &'a ProtectionDomain<'a>,
    device: &'a DeviceContext,
    transport: TcpTransport,
    host: &'a HostBackend,
    cuda: Option<&'a CudaBackend<'a>>,
) -> PeerSession<'a, TcpTransport> {
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
        host.buffer(),
        host.tier_id(),
        host.first_frame(),
        host.num_frames(),
        flags,
    ).expect("register host slab MR");
    if let Some(c) = cuda {
        rdma.register_gpu_slab(
            core,
            pd,
            c.gpu_buffer(),
            c.gpu_tier_id(),
            c.gpu_first_frame(),
            c.gpu_num_frames(),
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
    let mut engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        vericache_core::engine::CudaAttachment::None,
        None,
        vec![peer],
    ).expect("assemble distributed engine");
    let host_tier = engine.host_tier();
    let key = ObjectKey([11, 22, 33, 44]);
    let expected = pattern(0x2A);

    if node.id == 0 {
        let (ver, f_donor) = engine.put_host(
            Tracked(node.instance.borrow()),
            key,
            &expected,
        ).unwrap();
        let req_id = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest"),
        };
        let grant_tag = match engine.serve_fetch_request_for_peer(
            0,
            host_tier,
            req_id,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed published host key"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag && was_active);
            },
            _ => panic!("donor expected cooperative ReadReleased"),
        }
        assert!(engine.peer_active_lease_count(0).unwrap() == 0);
        // Donor evicts and reclaims the frame now that the remote lease was cooperatively released.
        engine.evict_frame(f_donor).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);
        assert!(engine.core().frame_state(f_donor).unwrap() == STATE_EMPTY);
        engine.send_control_to(
            0,
            ControlPayload::RevokeAck { grant_tag: ver.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        let outcome = engine.fetch_from_peer(
            0,
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
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
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
    batched: bool,
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
    let mut engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        vericache_core::engine::CudaAttachment::from_option_borrowed(Some(&cuda)),
        None,
        vec![peer],
    ).expect("assemble distributed engine");
    let gpu_tier = engine.gpu_tier().unwrap();
    let all_keys = [ObjectKey([55, 66, 77, 88]), ObjectKey([55, 66, 77, 89])];
    let keys = &all_keys[..if batched {
        2
    } else {
        1
    }];
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
            bytes: expected@,
            proof: ValueProof::HostWrite,
        };
        let mut published = Vec::new();
        for &key in keys {
            published.push(
                engine.put_gpu_sync(
                    Tracked(node.instance.borrow()),
                    &mut caller_stream,
                    caller_tl,
                    key,
                    &stage_cuda,
                    Tracked(&mut stage_perm),
                    Ghost(stage_cert),
                ).unwrap(),
            );
        }
        let request = engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap();
        match request {
            DistributedEvent::Control(ControlPayload::FetchBatch { entries }) if batched => {
                assert_eq!(entries.len(), keys.len());
                let n = engine.grant_fetch_batch(
                    0,
                    gpu_tier,
                    &entries,
                    Tracked(node.shares.borrow_mut()),
                    Tracked(node.instance.borrow()),
                ).unwrap();
                assert_eq!(n, keys.len());
                let deadline = Instant::now() + Duration::from_secs(5);
                while engine.peer_pending_grant_count(0).unwrap() != 0 {
                    assert!(Instant::now() < deadline, "GDR batch bind timed out");
                    engine.pump_peer_nonblocking(
                        0,
                        None,
                        &mut Vec::new(),
                        Tracked(node.instance.borrow()),
                    ).unwrap();
                }
            },
            DistributedEvent::Control(
                ControlPayload::FetchRequest { request_id, key },
            ) if !batched => {
                assert_eq!(key, keys[0]);
                let grant = engine.serve_fetch_request_for_peer(
                    0,
                    gpu_tier,
                    request_id,
                    key,
                    Tracked(node.shares.borrow_mut()),
                    Tracked(node.instance.borrow()),
                ).unwrap();
                assert!(matches!(grant, ServeOutcome::Granted { .. }));
            },
            _ => panic!("GDR donor expected a fetch"),
        }
        for _ in keys {
            let release = engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap();
            assert!(matches!(release, DistributedEvent::ReadReleased { was_active: true, .. }));
        }
        assert_eq!(engine.peer_active_lease_count(0).unwrap(), 0);
        for &(_, frame) in &published {
            engine.evict_frame(frame).unwrap();
        }
        assert_eq!(engine.reclaim_draining(gpu_tier, 4).unwrap(), keys.len());
        engine.send_control_to(
            0,
            ControlPayload::RevokeAck { grant_tag: published[0].0.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        let mut versions = vec![VersionId::NONE; keys.len()];
        if batched {
            let base = engine.send_fetch_batch_to(
                0,
                keys,
                Tracked(node.instance.borrow()),
            ).unwrap();
            let grants = collect_batch_grants(
                &mut engine,
                base,
                keys,
                Tracked(node.instance.borrow()),
            );
            let finished = engine.fetch_batch_from_peer(
                0,
                gpu_tier,
                PullTarget::GpuGdr,
                grants,
                Tracked(node.instance.borrow()),
            ).unwrap();
            assert!(finished.wait_error.is_none() && finished.pending.is_empty());
            assert_eq!(finished.results.len(), keys.len());
            for (id, outcome) in finished.results {
                let FetchOutcome::Hit { version, .. } = outcome.unwrap() else {
                    panic!("GDR initiator expected batched hit");
                };
                versions[(id - base) as usize] = version;
            }
        } else {
            let outcome = engine.fetch_from_peer(
                0,
                keys[0],
                gpu_tier,
                PullTarget::GpuGdr,
                Tracked(node.instance.borrow()),
            ).unwrap();
            let FetchOutcome::Hit { version, .. } = outcome else {
                panic!("GDR initiator expected hit");
            };
            versions[0] = version;
        }
        for (i, &key) in keys.iter().enumerate() {
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
            assert!(got_ver == versions[i]);
            engine.synchronize_caller_stream(
                &caller_stream,
                caller_tl,
                t_d2h,
                Tracked(node.instance.borrow()),
            ).unwrap();

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
                Tracked(&engine.core().ledger().borrow().submitted),
                Tracked(&engine.core().ledger().borrow().knows),
                Ghost(engine.core().citations()),
            );
            assert!(out == expected);
        }
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: donor_seq }) => {
                assert!(donor_seq == versions[0].seq);
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
    let mut engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        vericache_core::engine::CudaAttachment::None,
        None,
        vec![peer],
    ).expect("assemble distributed engine");
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
            let req_id = match engine.poll_peer_event_from(
                0,
                Tracked(node.instance.borrow()),
            ).unwrap() {
                DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                    assert!(k == key);
                    request_id
                },
                _ => panic!("donor expected FetchRequest"),
            };
            let grant_tag = match engine.serve_fetch_request_for_peer(
                0,
                host_tier,
                req_id,
                key,
                Tracked(node.shares.borrow_mut()),
                Tracked(node.instance.borrow()),
            ).unwrap() {
                ServeOutcome::Granted { grant_tag, .. } => grant_tag,
                ServeOutcome::Miss => panic!("donor unexpectedly missed key"),
            };
            match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
                DistributedEvent::ReadReleased { tag, was_active, .. } => {
                    assert!(tag == grant_tag && was_active);
                },
                _ => panic!("donor expected ReadReleased"),
            }
            round += 1;
        }
        engine.evict_frame(f_donor).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);
    } else {
        // Pull #1.
        let FetchOutcome::Hit { frame: f1, version: v1 } = engine.fetch_from_peer(
            0,
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
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);
        assert!(engine.lookup_tier(key, host_tier).is_none());
        assert!(engine.core().frame_state(f1).unwrap() == STATE_EMPTY);

        // Pull #2 of the exact same (Key, VersionId) verifies VersionBinding agreement.
        let FetchOutcome::Hit { frame: _f2, version: v2 } = engine.fetch_from_peer(
            0,
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
    let mut engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        vericache_core::engine::CudaAttachment::None,
        None,
        vec![peer],
    ).expect("assemble distributed engine");
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
        let req_id1 = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest #1"),
        };
        let grant_tag1 = match engine.serve_fetch_request_for_peer(
            0,
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
        let mw_slot = engine.revoke_peer_lease_unilateral(
            0,
            grant_tag1,
            Tracked(node.instance.borrow()),
        ).unwrap();
        assert!(engine.peer_active_lease_count(0).unwrap() == 0);

        // 2. Drain initiator's post-abort ReadRelease before QP Error -> Reset -> Reconnect.
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag1 && !was_active);
            },
            _ => panic!("donor expected post-abort ReadReleased"),
        }

        // 3. Option-A QP recovery: Error -> Reset (unquarantines slot -> Vacant) -> Reconnect.
        let _err_w = engine.enter_peer_qp_error(0).unwrap();
        engine.reset_and_reconnect_peer(0, 0x5678, Tracked(node.instance.borrow())).unwrap();
        engine.replenish_peer_window(0, mw_slot, &pd, node.mint_rdma()).unwrap();

        // 4. Clean re-grant and cooperative release on the reconnected QP.
        let req_id2 = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("donor expected FetchRequest #2 after reconnect"),
        };
        let grant_tag2 = match engine.serve_fetch_request_for_peer(
            0,
            host_tier,
            req_id2,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor unexpectedly missed key on re-fetch"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag2 && was_active);
            },
            _ => panic!("donor expected cooperative ReadReleased #2"),
        }
        engine.evict_frame(f_donor).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);
        engine.send_control_to(
            0,
            ControlPayload::RevokeAck { grant_tag: ver.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        // 1. Request key and hold the received ReadGrant until donor's RevokeNotice arrives.
        let _req_id1 = engine.send_fetch_request_to(
            0,
            key,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let grant = match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadGrant(g) => g,
            _ => panic!("initiator expected ReadGrant #1"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                assert!(grant_tag == grant.tag());
            },
            _ => panic!("initiator expected RevokeNotice"),
        }

        // 2. Attempt RDMA_READ against the revoked MW -> IBV_WC_REM_ACCESS_ERR (10) + abort_write.
        let ready_before = engine.core().tier_ready_count(host_tier).unwrap();
        let pull_err = engine.pull_with_grant_from(
            0,
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
        assert!(engine.core().tier_ready_count(host_tier).unwrap() == ready_before);

        // 3. Option-A QP recovery: Error -> Reset -> Reconnect with fresh PSN.
        let _err_w = engine.enter_peer_qp_error(0).unwrap();
        engine.reset_and_reconnect_peer(0, 0x1234, Tracked(node.instance.borrow())).unwrap();

        // 4. Clean re-fetch succeeds over the reconnected QP with intact payload.
        let FetchOutcome::Hit { frame: _f_ok, version: ver_ok } = engine.fetch_from_peer(
            0,
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
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
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
        host.context(),
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
    let mut engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        vericache_core::engine::CudaAttachment::from_option_borrowed(Some(&cuda)),
        Some(nvme),
        vec![peer],
    ).expect("assemble distributed engine");
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
        engine.synchronize_caller_stream(
            &caller_stream,
            caller_tl,
            t_d2h1,
            Tracked(node.instance.borrow()),
        ).unwrap();

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
            Tracked(&engine.core().ledger().borrow().submitted),
            Tracked(&engine.core().ledger().borrow().knows),
            Ghost(engine.core().citations()),
        );
        assert!(from_gpu == expected);

        engine.evict_frame(f_gpu1).unwrap();
        assert!(engine.reclaim_draining(gpu_tier, 4).unwrap() == 1);
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
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);

        // 4. Evict & reclaim Host DRAM copy of `key` so NVMe is the sole replica.
        engine.evict_frame(f_host1).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);
        assert!(engine.lookup_tier(key, host_tier).is_none());
        assert!(engine.lookup_tier(key, gpu_tier).is_none());
        assert!(engine.core().frame_state(f_host1).unwrap() == STATE_EMPTY);

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
            Tracked(&engine.core().ledger().borrow().submitted),
            Tracked(&engine.core().ledger().borrow().knows),
            Ghost(engine.core().citations()),
        );
        let ghost w2_ev = w2_comp.submitted.key();
        engine.record_host_completion(Tracked(w2_comp));
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

        let req_id = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("4-tier donor expected FetchRequest"),
        };
        let grant_tag = match engine.serve_fetch_request_for_peer(
            0,
            gpu_tier,
            req_id,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("4-tier donor unexpectedly missed GPU key"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == grant_tag && was_active);
            },
            _ => panic!("4-tier donor expected cooperative ReadReleased"),
        }
        assert!(engine.peer_active_lease_count(0).unwrap() == 0);
        engine.evict_frame(f_gpu2).unwrap();
        assert!(engine.reclaim_draining(gpu_tier, 4).unwrap() == 1);
        assert!(engine.core().frame_state(f_gpu2).unwrap() == STATE_EMPTY);
        engine.send_control_to(
            0,
            ControlPayload::RevokeAck { grant_tag: ver_gpu2.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        // Peer 2-GPU GPUDirect RDMA pull into Initiator H20 GPU HBM + BAR1 flush + D2H verification.
        let outcome = engine.fetch_from_peer(
            0,
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
        engine.synchronize_caller_stream(
            &caller_stream,
            caller_tl,
            t_d2h,
            Tracked(node.instance.borrow()),
        ).unwrap();

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
            Tracked(&engine.core().ledger().borrow().submitted),
            Tracked(&engine.core().ledger().borrow().knows),
            Ghost(engine.core().citations()),
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
        assert!(engine.reclaim_draining(host_tier, 4).unwrap() == 1);
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

        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: donor_seq }) => {
                assert!(donor_seq == ver_init.seq);
            },
            _ => panic!("4-tier initiator expected donor completion ack"),
        }
    }
    let _ = caller_stream.destroy();
}

fn exercise_batched_invalidations(
    backend: RdmaBackend<'_>,
    mr: &transfer::rdma::MemoryRegion<'_>,
    Tracked(instance): Tracked<&TransferInstance>,
) {
    let headroom = backend.window_free_count();
    let (qp, mut cq, mut pool) = backend.into_idle_parts().unwrap();
    let (qp, cq, pool) = (&qp, &mut cq, &mut pool);
    let mut entries = Vec::new();
    // Bind two test windows without publishing their keys.
    for slot in 0..2 {
        let (index, generation) = pool.checkout(slot as u64).unwrap();
        let bind_wr_id = 100 + slot as u64;
        let ret = pool.post_bind(
            index,
            generation,
            qp,
            cq,
            &mr.slot_slice(slot),
            RemoteAccess::ReadWrite,
            Tracked(WindowBindGrant::ReadWrite),
            bind_wr_id,
            false,
            &mut None,
        );
        ret.map_err(|(error, _)| error).unwrap();
        entries.push(WindowInvalidation { slot: index, generation, wr_id: 200 + slot as u64 });
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut bound = 0;
    while bound < entries.len() {
        assert!(Instant::now() < deadline, "test binds timed out");
        let cqes = poll_completion_queue_batch(cq, 2, Tracked(instance)).unwrap();
        for cqe in cqes {
            let Completed::WindowBound { wr_id, witness } = cqe else {
                panic!("expected a successful test bind");
            };
            let entry = entries[(wr_id - 100) as usize];
            pool.complete_bind(entry.slot, entry.generation, witness).unwrap();
            bound += 1;
        }
    }
    let (posted, error) = pool.post_invalidate_batch(qp, cq, &entries);
    assert_eq!(posted, entries.len());
    assert!(error.is_none());
    let mut invalidated = 0;
    while invalidated < entries.len() {
        assert!(Instant::now() < deadline, "test invalidations timed out");
        let cqes = poll_completion_queue_batch(cq, 2, Tracked(instance)).unwrap();
        for cqe in cqes {
            let Completed::WindowInvalidated { wr_id, witness } = cqe else {
                panic!("expected successful LOCAL_INV");
            };
            let entry = entries[(wr_id - 200) as usize];
            pool.finish_invalidate(entry.slot, entry.generation, witness, Tracked(None));
            invalidated += 1;
        }
    }
    assert_eq!(pool.free_count(), headroom);
}

fn tenant_host(node: &mut NodeEnv, core: &mut CacheCore) -> HostBackend {
    let name = node.mint_host();
    HostBackend::new(
        core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        name,
        8,
        4096,
    ).unwrap()
}

fn collect_batch_grants(
    client: &mut DistributedEngine<'_, TcpTransport>,
    base: u64,
    keys: &[ObjectKey],
    Tracked(instance): Tracked<&TransferInstance>,
) -> Vec<PendingReadGrant> {
    let mut grants = Vec::with_capacity(keys.len());
    let mut seen = vec![false; keys.len()];
    for _ in keys {
        let DistributedEvent::ReadGrant(grant) = client.poll_peer_event_from(
            0,
            Tracked(instance),
        ).unwrap() else {
            panic!("expected a read grant");
        };
        let i = (grant.meta().request_id - base) as usize;
        assert!(grant.meta().key == keys[i] && !seen[i]);
        seen[i] = true;
        grants.push(grant);
    }
    grants
}

fn check_tenant_batch(
    client: &mut DistributedEngine<'_, TcpTransport>,
    node: &mut NodeEnv,
    keys: &[ObjectKey],
    seeds: &[u8],
    grants: Vec<PendingReadGrant>,
) {
    let out = client.fetch_batch_from_peer(
        0,
        client.host_tier(),
        PullTarget::Host,
        grants,
        Tracked(node.instance.borrow()),
    ).unwrap();
    assert!(out.wait_error.is_none() && out.pending.is_empty());
    assert_eq!(out.results.len(), keys.len());
    for (_, result) in out.results {
        assert!(matches!(result.unwrap(), FetchOutcome::Hit { .. }));
    }
    for i in 0..keys.len() {
        let mut buffer = vec![0; FRAME_SIZE];
        let version = client.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            keys[i],
            &mut buffer,
        ).unwrap();
        assert!(version.is_some());
        assert_eq!(buffer, pattern(seeds[i]));
    }
}

fn pull_tenant(tenant: u64, device: &DeviceContext, node: &mut NodeEnv) {
    let transport = connect_to_donor();
    let mut probe = TcpTransport::new(transport.stream.try_clone().unwrap());
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host = tenant_host(node, &mut core);
    let session = connect_peer_rdma_for(node, 0, &mut core, &pd, device, transport, &host, None);
    let mut client = DistributedEngine::new_multi_peer(
        core,
        &host,
        HostContext::new(node.mint_host()),
        vericache_core::engine::CudaAttachment::None,
        None,
        vec![session],
    ).expect("assemble distributed engine");
    let keys = tenant_keys(tenant);
    let base = client.send_fetch_batch_to(0, &keys, Tracked(node.instance.borrow())).unwrap();
    let grants = collect_batch_grants(&mut client, base, &keys, Tracked(node.instance.borrow()));
    // Wait until the donor has checked the two live grants for each tenant.
    let ready = client.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap();
    assert!(matches!(ready, DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: 0 })));
    check_tenant_batch(
        &mut client,
        node,
        &keys,
        &[0x30 + tenant as u8, 0x40 + tenant as u8],
        grants,
    );
    if tenant == 1 {
        return;
    }
    // Decode the test's captured grant on a disconnected channel, then send a
    // READ with that live rkey over the tenant's real QP.

    let mut capture: Channel<ProbeFrame, VeriCacheAttachment> = Channel::new_with_protocol(
        ProbeFrame(probe.recv_frame().unwrap()),
        tenant,
        0,
        Tracked(node.instance.borrow()),
    );
    let ReceivedFrame::ReadGrant(live_a) = capture.recv_any(
        Tracked(node.instance.borrow()),
    ).unwrap() else {
        panic!("expected live PD_A grant");
    };
    let cap = live_a.capability.unwrap();
    let tracked mut a_perm = ReadPerm::Shared(live_a.share.get().tracked_unwrap());
    let ctx = HostContext::new(node.mint_host());
    let (buffer, Tracked(mut probe_tok)) = host::alloc_aligned(
        &ctx,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        FRAME_SIZE,
        4096,
    ).unwrap();
    let mr = register_memory(
        &pd,
        &buffer,
        AccessFlags { local_write: true, mw_bind: false },
    ).unwrap();
    let (mut qp, mut cq, _pool) = client.remove_peer(
        0,
    ).unwrap().into_rdma().into_idle_parts().unwrap();
    let tracked mut submitted = SubmittedMap::empty(node.instance.borrow().id());
    let tracked knows = Knows::empty(node.instance.borrow().id());
    let wr_id = 0x1234;
    let _ = post_rdma_read(
        &mut qp,
        &mut cq,
        Tracked(node.instance.borrow()),
        &mr.as_slice(),
        &cap,
        wr_id,
        false,
        Tracked(&mut a_perm),
        Tracked(&mut probe_tok),
        Tracked(&mut submitted),
        Tracked(&knows),
        Ghost(ISet::empty()),
    ).expect("unauthorized READ must reach the HCA");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match poll_completion_queue(&mut cq, Tracked(node.instance.borrow())).unwrap() {
            PollStatus::Empty => assert!(Instant::now() < deadline, "no cross-PD CQE"),
            PollStatus::Completed(Completed::DataFailed { wr_id: actual, status, .. }) => {
                assert_eq!(actual, wr_id);
                assert_eq!(status, IBV_WC_REM_ACCESS_ERR);
                println!("cross-PD read CQE status={status}");
                break;
            },
            _ => panic!("PD_B must reject PD_A's live rkey"),
        }
    }
    probe.send_frame(b"probe rejected").unwrap();
}

fn tenant_keys(tenant: u64) -> [ObjectKey; 2] {
    [ObjectKey([30, tenant, 0, 0]), ObjectKey([30, tenant, 1, 0])]
}

#[test]
fn distributed_per_peer_pd_isolation_and_batched_pull() {
    const TEST: &str = "distributed_per_peer_pd_isolation_and_batched_pull";
    let role = std::env::var(ROLE);
    let tenant = match role.as_deref() {
        Ok("tenant-a") => 1,
        Ok("tenant-b") => 2,
        Ok("tenant-donor") => 0,
        _ => {
            assert!(spawn_role(TEST, "tenant-donor").status().unwrap().success());
            return;
        },
    };
    let mut node = NodeEnv::new(tenant);
    let Some(device) = open_rdma_device(&mut node, false) else {
        return;
    };
    if tenant != 0 {
        pull_tenant(tenant, &device, &mut node);
        return;
    }
    let pd_a = ProtectionDomain::new(&device, node.mint_rdma()).unwrap();
    let pd_b = ProtectionDomain::new(&device, node.mint_rdma()).unwrap();
    let mut core = node.make_core();
    let host = tenant_host(&mut node, &mut core);
    let (mut tcp_a, mut child_a) = start_test_peer(TEST, "tenant-a");
    let (tcp_b, mut child_b) = start_test_peer(TEST, "tenant-b");
    let mut probe = TcpTransport::new(tcp_b.stream.try_clone().unwrap());
    let captured = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    tcp_a.capture = Some(captured.clone());
    let server_a = connect_peer_rdma_for(
        &mut node,
        1,
        &mut core,
        &pd_a,
        &device,
        tcp_a,
        &host,
        None,
    );
    let server_b = connect_peer_rdma_for(
        &mut node,
        2,
        &mut core,
        &pd_b,
        &device,
        tcp_b,
        &host,
        None,
    );
    assert!(std::ptr::eq(server_a.rdma().protection_domain(), &pd_a));
    assert!(std::ptr::eq(server_b.rdma().protection_domain(), &pd_b));
    assert!(!std::ptr::eq(server_a.rdma().protection_domain(), server_b.rdma().protection_domain()));
    let mut donor = DistributedEngine::new_multi_peer(
        core,
        &host,
        HostContext::new(node.mint_host()),
        CudaAttachment::None,
        None,
        vec![server_a, server_b],
    ).expect("assemble distributed engine");
    for client in 1..=2 {
        let keys = tenant_keys(client);
        donor.put_host(
            Tracked(node.instance.borrow()),
            keys[0],
            &pattern(0x30 + client as u8),
        ).unwrap();
        donor.put_host(
            Tracked(node.instance.borrow()),
            keys[1],
            &pattern(0x40 + client as u8),
        ).unwrap();
    }
    for i in 0..2 {
        let DistributedEvent::Control(ControlPayload::FetchBatch { entries }) =
            donor.poll_peer_event_from(i, Tracked(node.instance.borrow())).unwrap() else {
            panic!("donor expected FetchBatch");
        };
        assert_eq!(entries.len(), 2);
        let keys = tenant_keys((i + 1) as u64);
        assert!(entries.iter().all(|(_, key)| keys.contains(key)));
        let posted = donor.grant_fetch_batch(
            i,
            donor.host_tier(),
            &entries,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap();
        assert_eq!(posted, 2);
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    for i in 0..2 {
        while donor.peer_pending_grant_count(i).unwrap() != 0 {
            assert!(Instant::now() < deadline, "bind batch timed out");
            donor.pump_peer_nonblocking(
                i,
                None,
                &mut Vec::new(),
                Tracked(node.instance.borrow()),
            ).unwrap();
        }
        assert_eq!(donor.peer_active_lease_count(i).unwrap(), 2);
    }

    donor.send_control_to(
        1,
        ControlPayload::RevokeAck { grant_tag: 0 },
        Tracked(node.instance.borrow()),
    ).unwrap();
    for _ in 0..2 {
        let release = donor.poll_peer_event_from(1, Tracked(node.instance.borrow())).unwrap();
        assert!(matches!(release, DistributedEvent::ReadReleased { was_active: true, .. }));
    }
    assert_eq!(donor.peer_active_lease_count(0).unwrap(), 2);
    probe.send_frame(&captured.borrow()).unwrap();
    assert_eq!(probe.recv_frame().unwrap(), b"probe rejected");

    // The same still-bound rkey succeeds over its authorized QP.
    donor.send_control_to(
        0,
        ControlPayload::RevokeAck { grant_tag: 0 },
        Tracked(node.instance.borrow()),
    ).unwrap();
    for _ in 0..2 {
        let release = donor.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap();
        assert!(matches!(release, DistributedEvent::ReadReleased { was_active: true, .. }));
    }
    assert_eq!(donor.peer_active_lease_count(0).unwrap(), 0);
    assert_eq!(donor.peer_active_lease_count(1).unwrap(), 0);
    let backend = donor.remove_peer(0).unwrap().into_rdma();
    let mr = register_memory(
        &pd_a,
        host.buffer(),
        AccessFlags { local_write: true, mw_bind: true },
    ).unwrap();
    exercise_batched_invalidations(backend, &mr, Tracked(node.instance.borrow()));
    assert!(child_a.wait().unwrap().success());
    assert!(child_b.wait().unwrap().success());
}

#[test]
fn engine_rejects_out_of_bounds_ids_without_changing_cache() {
    let mut node = NodeEnv::new(3);
    let mut core = node.make_core();
    let host = tenant_host(&mut node, &mut core);
    let ctx = HostContext::new(node.mint_host());
    let mut engine: DistributedEngine<'_, TcpTransport> = DistributedEngine::new_multi_peer(
        core,
        &host,
        ctx,
        CudaAttachment::None,
        None,
        Vec::new(),
    ).unwrap();

    let bad_frame = FrameId(u32::MAX);
    let bad_tier = TierId(u16::MAX);
    let bad_peer = usize::MAX;
    let bad_request = engine.send_fetch_request_to(
        bad_peer,
        ObjectKey([0;4]),
        Tracked(node.instance.borrow()),
    );
    assert!(matches!(bad_request, Err(CoreError::InvalidArgument)));
    assert!(matches!(engine.take_inbound_grant(bad_peer, 1), Err(CoreError::InvalidArgument)));
    assert_eq!(engine.evict_frame(bad_frame), Err(CoreError::InvalidArgument));
    assert_eq!(engine.evict_frame(FrameId(0)), Err(CoreError::WouldConflict));
    assert_eq!(engine.reclaim_draining(bad_tier, 1), Err(CoreError::InvalidArgument));
    assert_eq!(engine.core().frame_state(bad_frame), None);
    assert_eq!(engine.core().tier_ready_count(bad_tier), None);
    assert_eq!(engine.peer_node(bad_peer), None);

    let key = ObjectKey([41, 0, 0, 1]);
    let (version, frame) = engine.put_host(
        Tracked(node.instance.borrow()),
        key,
        &pattern(0x41),
    ).unwrap();
    assert_eq!(engine.lookup_tier(key, engine.host_tier()), Some((frame, version)));
    assert_eq!(engine.core().frame_bound_key(frame), Some(key));

    let mut other_core = CacheCore::new(
        node.id as u16,
        FRAME_SIZE,
        Vec::new(),
        8,
        Tracked(node.instance.borrow()),
    );
    let _other_host = tenant_host(&mut node, &mut other_core);
    let foreign_ctx = HostContext::new(node.mint_host());
    let mismatched = LocalEngine::<NoDevice>::new(other_core, &host, foreign_ctx, None, None);
    assert!(matches!(mismatched, Err(CoreError::InvalidArgument)));

}

} // verus!
#[derive(Clone, Copy)]
enum Scenario {
    HostPull,
    GdrGpuPull,
    GdrGpuPullBatch,
    EvictRepull,
    UnilateralRevokeReconnect,
    FourTierRoundtrip,
}

fn dispatch_scenario(test_name: &str, scenario: Scenario) {
    let prefer_gdr = matches!(
        scenario,
        Scenario::GdrGpuPull | Scenario::GdrGpuPullBatch | Scenario::FourTierRoundtrip
    );
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
                Scenario::GdrGpuPull | Scenario::GdrGpuPullBatch => run_scenario2_gdr_gpu_pull(
                    &mut node,
                    &device,
                    gpu_opt.unwrap(),
                    transport,
                    matches!(scenario, Scenario::GdrGpuPullBatch),
                ),
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
                Scenario::GdrGpuPull | Scenario::GdrGpuPullBatch => {
                    let gpu = gpu_ordinal(1).unwrap();
                    run_scenario2_gdr_gpu_pull(
                        &mut node,
                        &device,
                        gpu,
                        transport,
                        matches!(scenario, Scenario::GdrGpuPullBatch),
                    )
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
fn distributed_gdr_two_gpu_batched_pull_flush_release() {
    dispatch_scenario(
        "distributed_gdr_two_gpu_batched_pull_flush_release",
        Scenario::GdrGpuPullBatch,
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
