//! Production-readiness stress and multi-peer integration suite for
//! `ShardAllocator`, `LocalEngine`, multi-size `CacheCore` tiers, >64 overflow
//! local read leases, owned async `CudaAttachment::Owned`, bidirectional
//! full-duplex `fetch_from_peer_duplex`, 3-node shared-MR `DistributedEngine`
//! mesh (`RegisteredSlab` + `drain_and_evict_frame_across_peers`), and
//! cooperative-first grant revocation (`revoke_grant_cooperative_first`).
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::sync::{Mutex, OnceLock};
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
use vericache_core::backends::rdma::peer::{FetchOutcome, PeerSession, ServeOutcome};
use vericache_core::backends::rdma::protocol::{ControlPayload, VeriCacheAttachment};
use vericache_core::backends::rdma::pull::PullTarget;
use vericache_core::backends::rdma::{RdmaBackend, RegisteredSlab};
use vericache_core::engine::{CudaAttachment, DistributedEngine, DistributedEvent, LocalEngine};
use vericache_core::frame::{STATE_DRAINING, STATE_EMPTY};
use vericache_core::kernel::CacheCore;
#[cfg(verus_only)]
use vericache_core::provenance::*;
use vericache_core::spec::*;
use vericache_core::timeline::TimelineKind;
use vstd::prelude::*;

const FRAME_4K: usize = 4096;
const FRAME_64K: usize = 65536;
const IBV_WC_REM_ACCESS_ERR: i32 = 10;

const ROLE: &str = "VERICACHE_PROD_STRESS_ROLE";
const ADDRESS: &str = "VERICACHE_PROD_STRESS_ADDRESS";
const SESSION: &str = "VERICACHE_PROD_STRESS_SESSION";

static LOCAL_ALLOCATOR: OnceLock<Mutex<ShardAllocator>> = OnceLock::new();

fn take_local_shard(quota: u64) -> ShardSources {
    let lock = LOCAL_ALLOCATOR.get_or_init(|| {
        Mutex::new(ShardAllocator::from_node_context(NodeContext::bootstrap(
            10,
        )))
    });
    let mut guard = lock.lock().unwrap();
    guard.split_shard(quota).expect("shard quota available")
}

struct TcpTransport {
    stream: TcpStream,
    rx_buf: Vec<u8>,
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
            rx_buf: Vec::new(),
        }
    }

    fn pop_buffered_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        if self.rx_buf.len() < 4 {
            return Ok(None);
        }
        let prefix: [u8; 4] = [
            self.rx_buf[0],
            self.rx_buf[1],
            self.rx_buf[2],
            self.rx_buf[3],
        ];
        let n = u32::from_be_bytes(prefix) as usize;
        if n > 1024 * 1024 {
            return Err(io::Error::other("frame exceeds test limit"));
        }
        let total = 4 + n;
        if self.rx_buf.len() < total {
            return Ok(None);
        }
        let frame = self.rx_buf[4..total].to_vec();
        self.rx_buf.drain(0..total);
        Ok(Some(frame))
    }
}

impl ByteTransport for TcpTransport {
    type Error = io::Error;

    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        self.stream.write_all(&n.to_be_bytes())?;
        self.stream.write_all(frame)
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        loop {
            if let Some(frame) = self.pop_buffered_frame()? {
                return Ok(frame);
            }
            let mut chunk = [0u8; 4096];
            match self.stream.read(&mut chunk) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "peer closed")),
                Ok(n) => self.rx_buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
    }

    fn try_recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        if let Some(frame) = self.pop_buffered_frame()? {
            return Ok(Some(frame));
        }
        self.stream.set_nonblocking(true)?;
        let mut scratch = [0u8; 4096];
        let mut read_err: Option<io::Error> = None;
        loop {
            match self.stream.read(&mut scratch) {
                Ok(0) => break,
                Ok(n) => self.rx_buf.extend_from_slice(&scratch[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    read_err = Some(e);
                    break;
                }
            }
        }
        let restore_res = self.stream.set_nonblocking(false);
        if let Some(e) = read_err {
            return Err(e);
        }
        restore_res?;
        self.pop_buffered_frame()
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

fn pattern_sized(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add(((i * 17 + 11) % 251) as u8))
        .collect()
}

fn spawn_role(test: &str, role: &str) -> Command {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", test, "--nocapture"]).env(ROLE, role);
    cmd
}

fn random_session_hex() -> String {
    let mut secret = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut secret)
        .unwrap();
    secret.iter().map(|b| format!("{b:02x}")).collect()
}

fn start_peer_process(test: &str, role: &str) -> (TcpTransport, Child) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let session = random_session_hex();
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
                    panic!("{role} exited before connect: {status}");
                }
                if Instant::now() >= until {
                    let _ = process.kill();
                    let _ = process.wait();
                    panic!("{role} connect timed out");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("accept: {e}"),
        }
    }
}

fn connect_to_coordinator() -> TcpTransport {
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
    ver_start: u64,
    ver_limit: u64,
}

impl NodeEnv {
    fn new(id: u64) -> Self {
        let shard = ShardAllocator::from_node_context(NodeContext::bootstrap(id)).split_shard(
            65536,
        ).unwrap();
        Self::from_shard(shard)
    }

    fn from_shard(shard: ShardSources) -> Self {
        Self {
            id: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            ver_instance: shard.version_instance,
            ver_source: Some(shard.versions),
            ver_start: shard.version_start,
            ver_limit: shard.version_limit,
        }
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

    fn make_core_with_leases(&mut self, max_leases: usize) -> CacheCore {
        let mut core = CacheCore::new_with_version_range(
            self.id as u16,
            FRAME_4K,
            Vec::new(),
            max_leases,
            self.ver_start,
            self.ver_limit,
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

fn open_rdma_device(node: &mut NodeEnv, test_name: &str) -> Option<DeviceContext> {
    let requested = std::env::var("VERICACHE_RDMA_DEV").ok().or_else(
        || std::env::var("VERICACHE_RDMA_DEVICE").ok(),
    );
    let found = match requested.as_deref() {
        Some(name) => select_rdma_device(node, Some(name)),
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
            "SKIP {test_name}: no usable Type-2B RDMA device matching {}",
            requested.as_deref().unwrap_or("any")
        );
    }
    found
}

fn connect_peer_dedicated_mr<'a>(
    node: &mut NodeEnv,
    peer_id: u64,
    core: &mut CacheCore,
    pd: &'a ProtectionDomain<'a>,
    device: &'a DeviceContext,
    transport: TcpTransport,
    host: &'a HostBackend,
    psn: u32,
) -> PeerSession<'a, TcpTransport> {
    let cq = CompletionQueue::new(device, 64, node.mint_rdma()).unwrap();
    let reset = UnconnectedQueuePair::new(
        pd,
        &cq,
        32,
        1,
        gid_index(),
        psn,
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
    PeerSession::new(rdma, channel, core)
}

fn connect_peer_shared_mr<'a>(
    node: &mut NodeEnv,
    peer_id: u64,
    core: &mut CacheCore,
    pd: &'a ProtectionDomain<'a>,
    device: &'a DeviceContext,
    transport: TcpTransport,
    shared_slab: &'a RegisteredSlab<'a>,
    psn: u32,
) -> PeerSession<'a, TcpTransport> {
    let cq = CompletionQueue::new(device, 64, node.mint_rdma()).unwrap();
    let reset = UnconnectedQueuePair::new(
        pd,
        &cq,
        32,
        1,
        gid_index(),
        psn,
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
    rdma.attach_shared_host_slab(core, shared_slab).expect("attach shared host slab MR");
    PeerSession::new(rdma, channel, core)
}

// =========================================================================
// Stress 1: 4-Thread Disjoint ShardAllocator + Concurrent LocalEngine
// =========================================================================
fn run_shard_worker(
    #[cfg_attr(not(verus_only), allow(unused_mut))]
    mut shard: ShardSources,
    worker_idx: usize,
) -> Vec<VersionId> {
    let mut core = CacheCore::new_with_version_range(
        shard.node as u16,
        FRAME_4K,
        Vec::new(),
        16,
        shard.version_start,
        shard.version_limit,
        Tracked(shard.instance.borrow()),
    );
    let tracked vi = shard.version_instance.borrow().clone();
    core.enable_version_registry(Tracked(vi), shard.versions);

    let host_src = Tracked(shard.events.borrow_mut().mint(EventSourceKind::Host));
    let host = HostBackend::new(
        &mut core,
        Tracked(shard.instance.borrow()),
        Tracked(shard.alloc.borrow_mut()),
        host_src,
        4,
        4096,
    ).unwrap();
    let eng_ctx = HostContext::new(Tracked(shard.events.borrow_mut().mint(EventSourceKind::Host)));
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, None);
    let host_tier = engine.host.tier;

    let mut minted = Vec::new();
    let mut round = 0usize;
    while round < 8 {
        let seed = ((worker_idx * 31 + round * 7 + 3) & 0xff) as u8;
        let payload = pattern_sized(seed, FRAME_4K);
        let key = ObjectKey([worker_idx as u64, round as u64, 0x51, 0x99]);
        let (ver, frame) = engine.put_host(
            Tracked(shard.instance.borrow()),
            key,
            &payload,
        ).unwrap();
        assert!(ver.origin_node == shard.node as u16);
        assert!(ver.seq >= shard.version_start && ver.seq < shard.version_limit);
        minted.push(ver);

        let lease_a = engine.get_lease_host(
            Tracked(shard.instance.borrow()),
            Tracked(shard.shares.borrow_mut()),
            key,
        ).unwrap().unwrap();
        let lease_b = engine.get_lease_host(
            Tracked(shard.instance.borrow()),
            Tracked(shard.shares.borrow_mut()),
            key,
        ).unwrap().unwrap();
        assert!(lease_a.version == ver && lease_b.version == ver);

        let mut read_ctx = HostContext::new(
            Tracked(shard.events.borrow_mut().mint(EventSourceKind::Host)),
        );
        let tracked mut perm_a = ReadPerm::Shared(lease_a.cap.get());
        let mut out = vec![0u8; FRAME_4K];
        let _rc = host::read_slice(
            &mut read_ctx,
            Tracked(shard.instance.borrow()),
            &lease_a.slice,
            &mut out,
            Tracked(&mut perm_a),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        assert!(out == payload);
        let tracked cap_a = match perm_a {
            ReadPerm::Shared(c) => c,
            ReadPerm::Owned(_) => proof_from_false(),
        };

        engine.release_lease(
            Tracked(shard.instance.borrow()),
            lease_a.handle,
            Tracked(cap_a),
            Ticket::NONE,
        ).unwrap();
        engine.release_lease(
            Tracked(shard.instance.borrow()),
            lease_b.handle,
            lease_b.cap,
            Ticket::NONE,
        ).unwrap();

        engine.evict_frame(frame).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        assert!(engine.core.frames.metas[frame.0 as usize].state == STATE_EMPTY);
        round += 1;
    }
    minted
}

// =========================================================================
// Stress 2: Multi-Size Tiers (4 KiB + 64 KiB), >64 Overflow Leases, and
//           Owned Async CUDA (`submit_put_gpu_async` + `poll_gpu_completions`)
// =========================================================================
fn run_multi_size_tiers_and_overflow_leases(shard: ShardSources) {
    let mut node = NodeEnv::from_shard(shard);
    let mut core = node.make_core_with_leases(96);

    let src_4k = node.mint_host();
    let tier_4k = HostBackend::new_with_frame_size(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        src_4k,
        4,
        4096,
        FRAME_4K,
    ).unwrap();
    let src_64k = node.mint_host();
    let tier_64k = HostBackend::new_with_frame_size(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        src_64k,
        4,
        4096,
        FRAME_64K,
    ).unwrap();
    assert!(tier_4k.tier != tier_64k.tier);

    let mut ctx_4k = HostContext::new(node.mint_host());
    let mut ctx_64k = HostContext::new(node.mint_host());

    let key_4k = ObjectKey([1, 4, 0, 0]);
    let key_64k = ObjectKey([1, 64, 0, 0]);
    let expected_4k = pattern_sized(0x31, FRAME_4K);
    let expected_64k = pattern_sized(0x7C, FRAME_64K);

    let (ver_4k, f_4k) = tier_4k.put_host_with(
        &mut ctx_4k,
        &mut core,
        Tracked(node.instance.borrow()),
        key_4k,
        &expected_4k,
    ).unwrap();
    let (ver_64k, f_64k) = tier_64k.put_host_with(
        &mut ctx_64k,
        &mut core,
        Tracked(node.instance.borrow()),
        key_64k,
        &expected_64k,
    ).unwrap();

    let mut out_4k = vec![0u8; FRAME_4K];
    let got_4k = tier_4k.read_frame_into_with(
        &mut ctx_4k,
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.shares.borrow_mut()),
        f_4k,
        &mut out_4k,
    ).unwrap();
    assert!(got_4k == ver_4k && out_4k == expected_4k);

    let mut out_64k = vec![0u8; FRAME_64K];
    let got_64k = tier_64k.read_frame_into_with(
        &mut ctx_64k,
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.shares.borrow_mut()),
        f_64k,
        &mut out_64k,
    ).unwrap();
    assert!(got_64k == ver_64k && out_64k == expected_64k);

    // Acquire 72 concurrent local read leases on `f_64k` (spilling past 64-bit inline_bitmap).
    let mut held: Vec<(vericache_core::kernel::ReadLeaseHandle, Tracked<ShareCap>)> = Vec::new();
    let mut i = 0usize;
    while i < 72 {
        let (h, v, _cert, cap) = core.acquire_local_lease(
            f_64k,
            false,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap();
        assert!(v == ver_64k);
        held.push((h, cap));
        i += 1;
    }
    assert!(core.frames.metas[f_64k.0 as usize].active_local_reads == 72);

    // Evict transitions to Draining, but cannot reclaim while 72 leases are active.
    core.evict_frame(f_64k).unwrap();
    assert!(core.frames.metas[f_64k.0 as usize].state == STATE_DRAINING);
    assert!(core.reclaim_draining(tier_64k.tier, 4) == 0);

    // Release out of order: odd indices (popped from back) vs front to exercise overflow_seqs.
    let mut evens = Vec::new();
    let mut odds = Vec::new();
    let mut idx = 0usize;
    while let Some(pair) = held.pop() {
        if idx % 2 == 0 {
            evens.push(pair);
        } else {
            odds.push(pair);
        }
        idx += 1;
    }
    while let Some((h, cap)) = odds.pop() {
        core.release_local_lease(h, cap, Ticket::NONE, Tracked(node.instance.borrow())).unwrap();
    }
    assert!(core.reclaim_draining(tier_64k.tier, 4) == 0);
    while let Some((h, cap)) = evens.pop() {
        core.release_local_lease(h, cap, Ticket::NONE, Tracked(node.instance.borrow())).unwrap();
    }
    assert!(core.frames.metas[f_64k.0 as usize].active_local_reads == 0);
    assert!(core.reclaim_draining(tier_64k.tier, 4) == 1);
    assert!(core.frames.metas[f_64k.0 as usize].state == STATE_EMPTY);
}

fn run_owned_async_cuda_roundtrip(shard: ShardSources, gpu_dev: DeviceId) {
    let mut node = NodeEnv::from_shard(shard);
    let mut core = node.make_core_with_leases(16);
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        node.mint_host(),
        4,
        4096,
    ).unwrap();
    let cuda = CudaBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        node.mint_cuda(),
        gpu_dev,
        4,
        16,
        false,
    ).unwrap();
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: gpu_dev }).unwrap();
    let mut caller_stream = cuda::CudaStream::new(gpu_dev, node.mint_cuda()).unwrap();

    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine: DistributedEngine<'_, TcpTransport> = DistributedEngine::new_with_owned_cuda(
        core,
        &host,
        eng_ctx,
        cuda,
        None,
        Vec::new(),
    );
    let gpu_tier = engine.gpu_tier().unwrap();
    let key = ObjectKey([9, 9, 9, 1]);
    let expected = pattern_sized(0x5A, FRAME_4K);

    let (stage_buf, Tracked(mut stage_tok)) = cuda::alloc_host(
        Tracked(node.alloc.borrow_mut()),
        FRAME_4K,
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
    engine.core.mark_timeline_complete(engine.host.timeline_id(), t_w.seq(), Tracked(w_comp.knows));
    let stage_cuda = stage_buf.as_cuda_slice();
    let tracked mut stage_perm = ReadPerm::Owned(stage_tok);
    let ghost stage_cert = ValueCertificate {
        slot: stage_cuda.slot_ref(),
        writer: w_ev,
        bytes: expected@,
        proof: ValueProof::HostWrite,
    };

    let (ver_async, f_async, _ticket, _sp) = engine.submit_put_gpu_async(
        Tracked(node.instance.borrow()),
        gpu_tier,
        key,
        &stage_cuda,
        Tracked(&mut stage_perm),
        Ghost(stage_cert),
    ).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut published = 0usize;
    while published == 0 {
        let rep = engine.poll_gpu_completions(Tracked(node.instance.borrow())).unwrap();
        published += rep.writes_published;
        assert!(rep.writes_aborted == 0);
        if published == 0 {
            assert!(Instant::now() < deadline, "poll_gpu_completions timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    assert!(published == 1);
    assert!(engine.lookup_tier(key, gpu_tier) == Some((f_async, ver_async)));

    let (rb_buf, Tracked(mut rb_tok)) = cuda::alloc_host(
        Tracked(node.alloc.borrow_mut()),
        FRAME_4K,
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
    assert!(got_ver == ver_async);
    let Tracked(k_d2h) = cuda::stream_synchronize(
        &caller_stream,
        Tracked(node.instance.borrow()),
    ).unwrap();
    engine.core.mark_timeline_complete(caller_tl, t_d2h.seq(), Tracked(k_d2h));

    let mut rb_ctx = HostContext::new(node.mint_host());
    let rb_host = rb_buf.as_host_slice();
    let tracked mut rb_perm = ReadPerm::Owned(rb_tok);
    let mut out = vec![0u8; FRAME_4K];
    let _rc = host::read_slice(
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
    let _ = caller_stream.destroy();
}

// =========================================================================
// Stress 3: Bidirectional Full-Duplex Fetch Over Blocking TCP + RC RDMA
// =========================================================================
fn run_stress3_bidirectional_duplex(
    node: &mut NodeEnv,
    device: &DeviceContext,
    transport: TcpTransport,
) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core_with_leases(16);
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        node.mint_host(),
        6,
        4096,
    ).unwrap();
    let peer_id = 1 - node.id;
    let peer = connect_peer_dedicated_mr(
        node,
        peer_id,
        &mut core,
        &pd,
        device,
        transport,
        &host,
        0x300 + node.id as u32,
    );
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, None, peer);
    let host_tier = engine.host_tier();

    let key_0 = ObjectKey([30, 0, 1, 1]);
    let key_1 = ObjectKey([30, 1, 1, 1]);
    let payload_0 = pattern_sized(0xA1, FRAME_4K);
    let payload_1 = pattern_sized(0xB2, FRAME_4K);

    let (own_key, own_bytes, want_key, want_bytes) = if node.id == 0 {
        (key_0, &payload_0, key_1, &payload_1)
    } else {
        (key_1, &payload_1, key_0, &payload_0)
    };

    let (_own_ver, f_own) = engine.put_host(
        Tracked(node.instance.borrow()),
        own_key,
        own_bytes,
    ).unwrap();

    // Ready barrier so both keys are published before simultaneous duplex fetches start.
    engine.send_control(
        ControlPayload::RevokeAck { grant_tag: 0xAA00 + node.id },
        Tracked(node.instance.borrow()),
    ).unwrap();
    match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
        DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag }) => {
            assert!(grant_tag == 0xAA00 + peer_id);
        },
        _ => panic!("expected ready barrier ack"),
    }

    let init_grant_tag = engine.peers[0].next_grant_tag;
    let mut rdma_tokens = Vec::new();
    let mut k = 0usize;
    while k < 4 {
        rdma_tokens.push(node.mint_rdma());
        k += 1;
    }

    let outcome = engine.fetch_from_peer_duplex(
        0,
        host_tier,
        Some(&pd),
        &mut rdma_tokens,
        want_key,
        host_tier,
        PullTarget::Host,
        Tracked(node.shares.borrow_mut()),
        Tracked(node.instance.borrow()),
    ).unwrap();
    let FetchOutcome::Hit { frame: _f_pulled, version: pulled_ver } = outcome else {
        panic!("bidirectional duplex fetch expected Hit");
    };

    let mut out = vec![0u8; FRAME_4K];
    let got = engine.get_into_host(
        Tracked(node.instance.borrow()),
        Tracked(node.shares.borrow_mut()),
        want_key,
        &mut out,
    ).unwrap();
    assert!(got == Some(pulled_ver));
    assert!(&out == want_bytes);

    // Keep servicing peer progress until our outbound grant has been served, released, and unbound.
    let deadline = Instant::now() + Duration::from_secs(15);
    while !(engine.peers[0].next_grant_tag > init_grant_tag
        && engine.peers[0].pending_outbound_grants.is_empty()
        && engine.peers[0].active_grants.is_empty() && engine.peers[0].rdma.active_lease_count()
        == 0) {
        assert!(Instant::now() < deadline, "duplex drain timed out");
        let _ = engine.poll_peer_progress(
            0,
            host_tier,
            Some(&pd),
            &mut rdma_tokens,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }

    engine.evict_frame(f_own).unwrap();
    assert!(engine.reclaim_draining(host_tier, 4) == 1);
    engine.shutdown_all_peers(Tracked(node.instance.borrow())).unwrap();
    assert!(engine.peer_count() == 0);
}

// =========================================================================
// Stress 4: 3-Node Shared-MR Mesh (`RegisteredSlab::register_host` +
//           `attach_shared_host_slab` + `drain_and_evict_frame_across_peers`)
// =========================================================================
fn run_stress4_coordinator_node0(
    node: &mut NodeEnv,
    device: &DeviceContext,
    t_peer1: TcpTransport,
    t_peer2: TcpTransport,
) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core_with_leases(16);
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        node.mint_host(),
        6,
        4096,
    ).unwrap();
    let flags = AccessFlags { local_write: true, mw_bind: true };
    let shared_slab = RegisteredSlab::register_host(
        &core,
        &pd,
        &host.buffer,
        host.tier,
        host.first_frame,
        host.num_frames,
        flags,
    ).expect("register single shared host slab MR");

    let peer1 = connect_peer_shared_mr(
        node,
        1,
        &mut core,
        &pd,
        device,
        t_peer1,
        &shared_slab,
        0x410,
    );
    let peer2 = connect_peer_shared_mr(
        node,
        2,
        &mut core,
        &pd,
        device,
        t_peer2,
        &shared_slab,
        0x420,
    );
    let eng_ctx = HostContext::new(node.mint_host());
    let mut peers = Vec::new();
    peers.push(peer1);
    let mut engine = DistributedEngine::new_multi_peer(
        core,
        &host,
        eng_ctx,
        CudaAttachment::None,
        None,
        peers,
    );
    engine.add_peer(peer2);
    assert!(engine.peer_count() == 2);
    assert!(engine.find_peer_index(1) == Some(0));
    assert!(engine.find_peer_index(2) == Some(1));
    assert!(engine.find_peer_index(99).is_none());

    let host_tier = engine.host_tier();
    let key = ObjectKey([44, 0, 0, 1]);
    let expected = pattern_sized(0x6D, FRAME_4K);
    let (ver, f_donor) = engine.put_host(Tracked(node.instance.borrow()), key, &expected).unwrap();

    // Serve the same frame concurrently to both Peer 0 (Node 1) and Peer 1 (Node 2) via shared MR.
    let mut idx = 0usize;
    while idx < 2 {
        let req_id = match engine.poll_peer_event_from(
            idx,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key);
                request_id
            },
            _ => panic!("coordinator expected FetchRequest from peer index {idx}"),
        };
        match engine.serve_fetch_request_for_peer(
            idx,
            host_tier,
            req_id,
            key,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { .. } => {},
            ServeOutcome::Miss => panic!("coordinator unexpectedly missed key for peer {idx}"),
        }
        idx += 1;
    }
    assert!(engine.core.frames.metas[f_donor.0 as usize].active_remote_leases == 2);

    // Drain and cooperatively-first revoke across both peers, transitioning `f_donor` to Draining.
    let evicted_grants = engine.drain_and_evict_frame_across_peers(
        f_donor,
        1_000_000,
        Tracked(node.instance.borrow()),
    ).unwrap();
    assert!(evicted_grants == 2);
    assert!(engine.core.frames.metas[f_donor.0 as usize].state == STATE_DRAINING);
    assert!(engine.core.frames.metas[f_donor.0 as usize].active_remote_leases == 0);
    assert!(engine.peers[0].rdma.active_lease_count() == 0);
    assert!(engine.peers[1].rdma.active_lease_count() == 0);
    assert!(engine.reclaim_draining(host_tier, 4) == 1);
    assert!(engine.core.frames.metas[f_donor.0 as usize].state == STATE_EMPTY);

    // Notify both peers that cross-peer reclaim succeeded, then run shutdown_all_peers.
    engine.send_control_to(
        0,
        ControlPayload::RevokeAck { grant_tag: ver.seq },
        Tracked(node.instance.borrow()),
    ).unwrap();
    engine.send_control_to(
        1,
        ControlPayload::RevokeAck { grant_tag: ver.seq },
        Tracked(node.instance.borrow()),
    ).unwrap();
    engine.shutdown_all_peers(Tracked(node.instance.borrow())).unwrap();
    assert!(engine.peer_count() == 0);
}

fn run_stress4_leaf_peer(node: &mut NodeEnv, device: &DeviceContext, transport: TcpTransport) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core_with_leases(16);
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        node.mint_host(),
        4,
        4096,
    ).unwrap();
    let peer = connect_peer_dedicated_mr(
        node,
        0,
        &mut core,
        &pd,
        device,
        transport,
        &host,
        0x400 + node.id as u32,
    );
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, None, peer);
    let host_tier = engine.host_tier();
    let key = ObjectKey([44, 0, 0, 1]);
    let expected = pattern_sized(0x6D, FRAME_4K);

    let _req_id = engine.send_fetch_request(key, Tracked(node.instance.borrow())).unwrap();
    let grant = match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
        DistributedEvent::ReadGrant(g) => g,
        _ => panic!("leaf peer expected ReadGrant"),
    };

    // Hold the grant open until coordinator initiates cooperative-first cross-peer drain.
    match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
        DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
            assert!(grant_tag == grant.tag);
        },
        _ => panic!("leaf peer expected cooperative RevokeNotice"),
    }

    // Pull payload over RDMA and send cooperative ReadRelease promptly.
    let FetchOutcome::Hit { frame: _pulled_frame, version: ver_got } = engine.pull_with_grant(
        key,
        host_tier,
        PullTarget::Host,
        grant,
        Tracked(node.instance.borrow()),
    ).unwrap() else {
        panic!("leaf peer expected Hit");
    };
    let mut out = vec![0u8; FRAME_4K];
    let got = engine.get_into_host(
        Tracked(node.instance.borrow()),
        Tracked(node.shares.borrow_mut()),
        key,
        &mut out,
    ).unwrap();
    assert!(got == Some(ver_got));
    assert!(out == expected);

    match engine.poll_peer_event(Tracked(node.instance.borrow())).unwrap() {
        DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: seq }) => {
            assert!(seq == ver_got.seq);
        },
        _ => panic!("leaf peer expected final coordinator ack"),
    }
    engine.shutdown_all_peers(Tracked(node.instance.borrow())).unwrap();
    assert!(engine.peer_count() == 0);
}

// =========================================================================
// Stress 5: Cooperative-First Grant Revocation (Fast Path + Timeout Fallback
//           -> Unilateral MW Quarantine -> QP Reset/Reconnect/Replenish)
// =========================================================================
fn run_stress5_cooperative_first_revocation(
    node: &mut NodeEnv,
    device: &DeviceContext,
    transport: TcpTransport,
) {
    let pd = ProtectionDomain::new(device, node.mint_rdma()).unwrap();
    let mut core = node.make_core_with_leases(16);
    let host = HostBackend::new(
        &mut core,
        Tracked(node.instance.borrow()),
        Tracked(node.alloc.borrow_mut()),
        node.mint_host(),
        6,
        4096,
    ).unwrap();
    let peer_id = 1 - node.id;
    let peer = connect_peer_dedicated_mr(
        node,
        peer_id,
        &mut core,
        &pd,
        device,
        transport,
        &host,
        0x500 + node.id as u32,
    );
    let eng_ctx = HostContext::new(node.mint_host());
    let mut engine = DistributedEngine::new(core, &host, eng_ctx, None, peer);
    let host_tier = engine.host_tier();

    let key_fast = ObjectKey([55, 1, 0, 1]);
    let key_slow = ObjectKey([55, 2, 0, 2]);
    let expected_fast = pattern_sized(0x41, FRAME_4K);
    let expected_slow = pattern_sized(0x83, FRAME_4K);

    if node.id == 0 {
        let (_ver_fast, f_fast) = engine.put_host(
            Tracked(node.instance.borrow()),
            key_fast,
            &expected_fast,
        ).unwrap();
        let (ver_slow, f_slow) = engine.put_host(
            Tracked(node.instance.borrow()),
            key_slow,
            &expected_slow,
        ).unwrap();

        // --- Phase A: Cooperative Fast Path (Ok(None), zero MW quarantine) ---
        let req_fast = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key_fast);
                request_id
            },
            _ => panic!("donor expected FetchRequest for key_fast"),
        };
        let tag_fast = match engine.serve_fetch_request_for_peer(
            0,
            host_tier,
            req_fast,
            key_fast,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor missed key_fast"),
        };
        let fast_revoke = engine.revoke_grant_cooperative_first(
            0,
            tag_fast,
            1_000_000,
            Tracked(node.instance.borrow()),
        ).unwrap();
        assert!(fast_revoke.is_none());
        assert!(engine.peers[0].rdma.active_lease_count() == 0);
        engine.evict_frame(f_fast).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);

        // --- Phase B: Timeout Fallback (Ok(Some(mw_slot))) + QP Recovery ---
        let req_slow = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key_slow);
                request_id
            },
            _ => panic!("donor expected FetchRequest for key_slow"),
        };
        let tag_slow = match engine.serve_fetch_request_for_peer(
            0,
            host_tier,
            req_slow,
            key_slow,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor missed key_slow"),
        };
        let slow_revoke = engine.revoke_grant_cooperative_first(
            0,
            tag_slow,
            8,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let mw_slot = slow_revoke.expect("timeout fallback must quarantine MW slot");
        assert!(engine.peers[0].rdma.active_lease_count() == 0);

        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == tag_slow && !was_active);
            },
            _ => panic!("donor expected post-abort ReadReleased for tag_slow"),
        }

        let _err_w = engine.enter_peer_qp_error(0).unwrap();
        engine.reset_and_reconnect_peer(0, 0x6789, Tracked(node.instance.borrow())).unwrap();
        engine.replenish_peer_window(0, mw_slot, &pd, node.mint_rdma()).unwrap();

        // Serve clean post-reconnect fetch of `key_slow`.
        let req_retry = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::Control(ControlPayload::FetchRequest { request_id, key: k }) => {
                assert!(k == key_slow);
                request_id
            },
            _ => panic!("donor expected post-reconnect FetchRequest"),
        };
        let tag_retry = match engine.serve_fetch_request_for_peer(
            0,
            host_tier,
            req_retry,
            key_slow,
            Tracked(node.shares.borrow_mut()),
            Tracked(node.instance.borrow()),
        ).unwrap() {
            ServeOutcome::Granted { grant_tag, .. } => grant_tag,
            ServeOutcome::Miss => panic!("donor missed key_slow after reconnect"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::ReadReleased { tag, was_active, .. } => {
                assert!(tag == tag_retry && was_active);
            },
            _ => panic!("donor expected cooperative ReadReleased after reconnect"),
        }
        engine.evict_frame(f_slow).unwrap();
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        engine.send_control_to(
            0,
            ControlPayload::RevokeAck { grant_tag: ver_slow.seq },
            Tracked(node.instance.borrow()),
        ).unwrap();
    } else {
        // --- Phase A: Promptly honor SoftRevokeNotice ---
        let _r1 = engine.send_fetch_request_to(
            0,
            key_fast,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let grant_fast = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::ReadGrant(g) => g,
            _ => panic!("initiator expected ReadGrant for key_fast"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                assert!(grant_tag == grant_fast.tag);
            },
            _ => panic!("initiator expected cooperative RevokeNotice for key_fast"),
        }
        let FetchOutcome::Hit { frame: _f_fast, version: v_fast } = engine.pull_with_grant_from(
            0,
            key_fast,
            host_tier,
            PullTarget::Host,
            grant_fast,
            Tracked(node.instance.borrow()),
        ).unwrap() else {
            panic!("initiator expected Hit for key_fast");
        };
        let mut out_fast = vec![0u8; FRAME_4K];
        let got_fast = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key_fast,
            &mut out_fast,
        ).unwrap();
        assert!(got_fast == Some(v_fast) && out_fast == expected_fast);

        // --- Phase B: Withhold release until unilateral RevokeNotice arrives ---
        let _r2 = engine.send_fetch_request_to(
            0,
            key_slow,
            Tracked(node.instance.borrow()),
        ).unwrap();
        let grant_slow = match engine.poll_peer_event_from(
            0,
            Tracked(node.instance.borrow()),
        ).unwrap() {
            DistributedEvent::ReadGrant(g) => g,
            _ => panic!("initiator expected ReadGrant for key_slow"),
        };
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                assert!(grant_tag == grant_slow.tag);
            },
            _ => panic!("initiator expected cooperative advisory RevokeNotice for key_slow"),
        }
        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeNotice { grant_tag }) => {
                assert!(grant_tag == grant_slow.tag);
            },
            _ => panic!("initiator expected unilateral RevokeNotice fallback"),
        }

        let pull_err = engine.pull_with_grant_from(
            0,
            key_slow,
            host_tier,
            PullTarget::Host,
            grant_slow,
            Tracked(node.instance.borrow()),
        );
        match pull_err {
            Err(CoreError::Backend { kind: BACKEND_RDMA, code }) => {
                assert!(code != 0);
                if device.name().starts_with("mlx5_") {
                    assert!(code == IBV_WC_REM_ACCESS_ERR);
                }
            },
            other => panic!("expected RDMA remote access error after fallback, got {other:?}"),
        }

        let _err_w = engine.enter_peer_qp_error(0).unwrap();
        engine.reset_and_reconnect_peer(0, 0x2345, Tracked(node.instance.borrow())).unwrap();

        let FetchOutcome::Hit { frame: _f_ok, version: v_ok } = engine.fetch_from_peer(
            0,
            key_slow,
            host_tier,
            PullTarget::Host,
            Tracked(node.instance.borrow()),
        ).unwrap() else {
            panic!("initiator expected Hit after reconnect");
        };
        let mut out_slow = vec![0u8; FRAME_4K];
        let got_slow = engine.get_into_host(
            Tracked(node.instance.borrow()),
            Tracked(node.shares.borrow_mut()),
            key_slow,
            &mut out_slow,
        ).unwrap();
        assert!(got_slow == Some(v_ok) && out_slow == expected_slow);

        match engine.poll_peer_event_from(0, Tracked(node.instance.borrow())).unwrap() {
            DistributedEvent::Control(ControlPayload::RevokeAck { grant_tag: seq }) => {
                assert!(seq == v_ok.seq);
            },
            _ => panic!("initiator expected donor completion ack"),
        }
    }
    engine.shutdown_all_peers(Tracked(node.instance.borrow())).unwrap();
    assert!(engine.peer_count() == 0);
}

} // verus!
#[test]
fn stress_1_shard_allocator_4_thread_local_engine() {
    let parent = take_local_shard(16384);
    let base = parent.version_start;
    let mut alloc = ShardAllocator::from_shard(parent);
    let s0 = alloc.split_shard(4096).unwrap();
    let s1 = alloc.split_shard(4096).unwrap();
    let s2 = alloc.split_shard(4096).unwrap();
    let s3 = alloc.split_shard(4096).unwrap();
    assert_eq!((s0.version_start, s0.version_limit), (base, base + 4096));
    assert_eq!(
        (s1.version_start, s1.version_limit),
        (base + 4096, base + 8192)
    );
    assert_eq!(
        (s2.version_start, s2.version_limit),
        (base + 8192, base + 12288)
    );
    assert_eq!(
        (s3.version_start, s3.version_limit),
        (base + 12288, base + 16384)
    );

    let shards = [s0, s1, s2, s3];
    let handles: Vec<_> = shards
        .into_iter()
        .enumerate()
        .map(|(idx, shard)| std::thread::spawn(move || run_shard_worker(shard, idx)))
        .collect();

    let mut all_versions = Vec::new();
    for h in handles {
        let minted = h.join().expect("shard worker panicked");
        assert_eq!(minted.len(), 8);
        for v in minted {
            assert!(
                !all_versions.contains(&v),
                "duplicate VersionId minted across shards"
            );
            all_versions.push(v);
        }
    }
    assert_eq!(all_versions.len(), 32);
}

#[test]
fn stress_2_multi_size_tiers_overflow_leases_and_owned_async_cuda() {
    run_multi_size_tiers_and_overflow_leases(take_local_shard(4096));

    let gpu_count = cuda::device_count().unwrap_or(0);
    if gpu_count == 0 {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but no CUDA GPU available for stress_2");
        }
        eprintln!("SKIP GPU part of stress_2: no CUDA GPU available");
        return;
    }
    run_owned_async_cuda_roundtrip(take_local_shard(4096), 0);
}

#[test]
fn stress_3_bidirectional_duplex_fetch_over_blocking_tcp() {
    let test_name = "stress_3_bidirectional_duplex_fetch_over_blocking_tcp";
    match std::env::var(ROLE).as_deref() {
        Ok("node0") => {
            let mut node = NodeEnv::new(0);
            let Some(device) = open_rdma_device(&mut node, test_name) else {
                return;
            };
            let (transport, mut child) = start_peer_process(test_name, "node1");
            run_stress3_bidirectional_duplex(&mut node, &device, transport);
            assert!(child.wait().unwrap().success());
        }
        Ok("node1") => {
            let mut node = NodeEnv::new(1);
            let device = open_rdma_device(&mut node, test_name).unwrap();
            let transport = connect_to_coordinator();
            run_stress3_bidirectional_duplex(&mut node, &device, transport);
        }
        _ => {
            let status = spawn_role(test_name, "node0").status().unwrap();
            assert!(status.success(), "{test_name} node0 failed: {status}");
        }
    }
}

#[test]
fn stress_4_three_node_shared_mr_mesh_and_cross_peer_evict() {
    let test_name = "stress_4_three_node_shared_mr_mesh_and_cross_peer_evict";
    match std::env::var(ROLE).as_deref() {
        Ok("coord") => {
            let mut node = NodeEnv::new(0);
            let Some(device) = open_rdma_device(&mut node, test_name) else {
                return;
            };
            let (t1, mut child1) = start_peer_process(test_name, "peer1");
            let (t2, mut child2) = start_peer_process(test_name, "peer2");
            run_stress4_coordinator_node0(&mut node, &device, t1, t2);
            assert!(child1.wait().unwrap().success());
            assert!(child2.wait().unwrap().success());
        }
        Ok("peer1") => {
            let mut node = NodeEnv::new(1);
            let device = open_rdma_device(&mut node, test_name).unwrap();
            let transport = connect_to_coordinator();
            run_stress4_leaf_peer(&mut node, &device, transport);
        }
        Ok("peer2") => {
            let mut node = NodeEnv::new(2);
            let device = open_rdma_device(&mut node, test_name).unwrap();
            let transport = connect_to_coordinator();
            run_stress4_leaf_peer(&mut node, &device, transport);
        }
        _ => {
            let status = spawn_role(test_name, "coord").status().unwrap();
            assert!(status.success(), "{test_name} coord failed: {status}");
        }
    }
}

#[test]
fn stress_5_cooperative_first_revocation_fast_and_timeout_fallback() {
    let test_name = "stress_5_cooperative_first_revocation_fast_and_timeout_fallback";
    match std::env::var(ROLE).as_deref() {
        Ok("donor") => {
            let mut node = NodeEnv::new(0);
            let Some(device) = open_rdma_device(&mut node, test_name) else {
                return;
            };
            let (transport, mut child) = start_peer_process(test_name, "initiator");
            run_stress5_cooperative_first_revocation(&mut node, &device, transport);
            assert!(child.wait().unwrap().success());
        }
        Ok("initiator") => {
            let mut node = NodeEnv::new(1);
            let device = open_rdma_device(&mut node, test_name).unwrap();
            let transport = connect_to_coordinator();
            run_stress5_cooperative_first_revocation(&mut node, &device, transport);
        }
        _ => {
            let status = spawn_role(test_name, "donor").status().unwrap();
            assert!(status.success(), "{test_name} donor failed: {status}");
        }
    }
}
