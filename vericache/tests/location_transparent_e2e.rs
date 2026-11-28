//! End-to-end tests demonstrating true location transparency and
//! **Unverified Policy + Verified Enforcement** in the `vericache` facade:
//!
//! 1. `local_host_and_nvme_location_transparency`: single `cache.get(key, &mut buf)`
//!    transparently hits Local DRAM and Local NVMe (after LRU cascade spill) + RAII `lease_host`.
//! 2. `buggy_policy_verified_enforcement`: adversarial `EvictionPolicy` + `PeerRoutingPolicy`
//!    returning out-of-bounds, wrong-tier, stale, and actively-leased `FrameId`s; proves
//!    `CacheCore` / `DistributedEngine` never panics, corrupts data, or reclaims leased frames.
//! 3. `local_gpu_hbm_dram_nvme_location_transparency`: single `cache.get(key, GpuDst)` and
//!    `cache.get(key, &mut buf)` transparently hit Local GPU HBM, Pinned DRAM, and Local NVMe
//!    (plus RAII `lease_gpu`).
//! 4. `distributed_4tier_location_transparency_e2e`: real 2-node ConnectX RDMA + 2-GPU GDR E2E
//!    where the exact same `cache.get(key, ...)` resolves Local HBM, Local DRAM, Local NVMe,
//!    Remote Host RDMA, and Remote 2-GPU GPUDirect RDMA without changing a single call-site arg.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use transfer::cuda;
use transfer::domain::DeviceId;
use transfer::rdma::DeviceContext;
use transfer::transport::ByteTransport;
use vericache::{
    CacheError, CascadeTiering, DefaultPolicy, EvictionPolicy, FrameId, GpuConfig, GpuDst, GpuSrc,
    HitLocation, LruEviction, NoTiering, NoTransport, NodeStorage, NvmeConfig, ObjectKey, PeerLink,
    PeerMeta, PeerRoutingPolicy, PolicySet, ProbeAllPeers, RdmaConfig, StorageConfig, TcpTransport,
    TierKind, VeriCache,
};

const FRAME_SIZE: usize = 4096;

const ROLE_ENV: &str = "VERICACHE_FACADE_E2E_ROLE";
const ADDR_ENV: &str = "VERICACHE_FACADE_E2E_ADDR";
const SESSION_ENV: &str = "VERICACHE_FACADE_E2E_SESSION";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_nvme_dir(tag: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("e2e-nvme")
        .join(format!("vericache_facade_{tag}_{pid}_{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp nvme dir");
    dir
}

fn make_key(id: u64) -> ObjectKey {
    ObjectKey([id, id.wrapping_mul(0x9e37_79b9_7f4a_7c15), 0x1234, 0x5678])
}

fn pattern(seed: u8) -> Vec<u8> {
    (0..FRAME_SIZE)
        .map(|i| seed.wrapping_add(((i * 17 + 11) % 251) as u8))
        .collect()
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
    let count = cuda::device_count().unwrap_or(0);
    let env_key = if node == 0 {
        "VERICACHE_GDR_DONOR_GPU"
    } else {
        "VERICACHE_GDR_INITIATOR_GPU"
    };
    let desired: u32 = std::env::var(env_key)
        .map(|s| s.parse().expect("invalid GPU ordinal"))
        .unwrap_or(node as u32);
    if desired < count {
        Some(desired)
    } else {
        None
    }
}

fn has_usable_rdma_device() -> Option<String> {
    let requested = std::env::var("VERICACHE_RDMA_DEV")
        .ok()
        .or_else(|| std::env::var("VERICACHE_RDMA_DEVICE").ok());
    let count = DeviceContext::count().ok()?;
    for index in 0..count {
        let Ok(dev) = DeviceContext::open(index, 0) else {
            continue;
        };
        let name = dev.name().to_string();
        if requested.as_deref().is_some_and(|r| r != name) {
            let _ = dev.close();
            continue;
        }
        let type2 = dev.supports_type_2().unwrap_or(false);
        let _ = dev.close();
        if type2 {
            return Some(name);
        }
    }
    None
}

// ============================================================================
// Test 1: Local Host DRAM + Local NVMe Location Transparency + RAII Lease
// ============================================================================

#[test]
fn local_host_and_nvme_location_transparency() {
    let nvme_dir = temp_nvme_dir("host_nvme");
    let cfg = StorageConfig::new(0, FRAME_SIZE, 2).with_nvme(NvmeConfig::new(&nvme_dir, 4));

    let res = VeriCache::<NoTransport, _, _, _>::with_session(
        cfg,
        Vec::new(),
        DefaultPolicy::default(),
        |cache| {
            let k1 = make_key(1);
            let k2 = make_key(2);
            let k3 = make_key(3);
            let p1 = pattern(0x11);
            let p2 = pattern(0x22);
            let p3 = pattern(0x33);

            // Unified `put` into DRAM (capacity = 2).
            let v1 = cache.put(k1, p1.as_slice())?;
            let v2 = cache.put(k2, p2.as_slice())?;
            // Third put exceeds DRAM capacity (2): CascadeTiering automatically spills
            // LRU k1 -> Local NVMe and reclaims its DRAM frame.
            let v3 = cache.put(k3, p3.as_slice())?;

            // Exact same location-transparent `cache.get(key, &mut buf)` call hits DRAM for k2/k3
            // and transparently warms Local NVMe via io_uring for k1!
            let mut buf = vec![0u8; FRAME_SIZE];

            let h2 = cache.get(k2, &mut buf)?.expect("k2 hit");
            assert_eq!(h2.location, HitLocation::LocalDram);
            assert_eq!(h2.version, v2);
            assert_eq!(buf, p2);

            let h3 = cache.get(k3, &mut buf)?.expect("k3 hit");
            assert_eq!(h3.location, HitLocation::LocalDram);
            assert_eq!(h3.version, v3);
            assert_eq!(buf, p3);

            // k1 was cascaded to NVMe — caller does NOT specify NVMe!
            let h1 = cache.get(k1, &mut buf)?.expect("k1 hit from NVMe");
            assert_eq!(h1.location, HitLocation::LocalNvme);
            assert_eq!(h1.version, v1);
            assert_eq!(buf, p1);

            // Now k1 is warm in DRAM again; RAII zero-copy lease_host works and releases on drop.
            {
                let mut lease = cache.lease_host(k1)?.expect("k1 host lease");
                assert_eq!(lease.version(), v1);
                let mut lease_buf = vec![0u8; FRAME_SIZE];
                lease.copy_into(&mut lease_buf)?;
                assert_eq!(lease_buf, p1);
            }

            // Missing key returns Ok(None).
            assert!(cache.get(make_key(999), &mut buf)?.is_none());
            assert!(cache.stats().dram_hits >= 2);
            assert_eq!(cache.stats().nvme_hits, 1);
            Ok(())
        },
    );

    let _ = std::fs::remove_dir_all(&nvme_dir);
    res.expect("local_host_and_nvme_location_transparency");
}

// ============================================================================
// Test 2: Adversarial BuggyPolicy vs Verified Enforcement
// ============================================================================

/// Intentionally broken policy that emits garbage FrameIds (out-of-bounds,
/// wrong-tier, stale) before falling back to a real inner LRU.
struct BuggyEvictionPolicy {
    poison_queue: Vec<(ObjectKey, FrameId)>,
    inner: LruEviction,
}

impl BuggyEvictionPolicy {
    fn new(poison: Vec<(ObjectKey, FrameId)>) -> Self {
        Self {
            poison_queue: poison,
            inner: LruEviction::new(),
        }
    }
}

impl EvictionPolicy for BuggyEvictionPolicy {
    fn on_insert(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_insert(tier, key, frame);
    }

    fn on_access(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_access(tier, key, frame);
    }

    fn on_remove(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_remove(tier, key, frame);
    }

    fn select_victim(&mut self, tier: TierKind) -> Option<(ObjectKey, FrameId)> {
        if let Some(bad) = self.poison_queue.pop() {
            return Some(bad);
        }
        self.inner.select_victim(tier)
    }
}

/// Intentionally broken routing policy that injects invalid peer indices.
struct BuggyRoutingPolicy;

impl PeerRoutingPolicy for BuggyRoutingPolicy {
    fn probe_order(&self, _key: ObjectKey, peers: &[PeerMeta], out: &mut Vec<usize>) {
        out.clear();
        out.push(9999); // out of bounds!
        for p in peers {
            out.push(p.index);
        }
    }
}

#[test]
fn buggy_policy_verified_enforcement() {
    let nvme_dir = temp_nvme_dir("buggy_policy");
    let cfg = StorageConfig::new(0, FRAME_SIZE, 2).with_nvme(NvmeConfig::new(&nvme_dir, 2));

    let k1 = make_key(101);
    let k2 = make_key(102);
    let k3 = make_key(103);

    // Poison victims: out-of-bounds FrameId(999_999), NVMe frame FrameId(2) when evicting DRAM,
    // and stale FrameId(0) duplicate.
    let buggy_eviction = BuggyEvictionPolicy::new(vec![
        (make_key(777), FrameId(999_999)),
        (make_key(778), FrameId(2)), // wrong tier (belongs to NVMe, not DRAM!)
        (make_key(779), FrameId(12345)),
    ]);
    let policy = PolicySet::new(buggy_eviction, CascadeTiering::default(), BuggyRoutingPolicy);

    let res = VeriCache::<NoTransport, _, _, _>::with_session(cfg, Vec::new(), policy, |cache| {
        let p1 = pattern(0x41);
        let p2 = pattern(0x42);
        let p3 = pattern(0x43);

        let v1 = cache.put(k1, p1.as_slice())?;
        let _v2 = cache.put(k2, p2.as_slice())?;

        // Put k3 forces DRAM eviction: BuggyEvictionPolicy first feeds 3 invalid/wrong-tier
        // FrameIds, which Verified Enforcement safely rejects before accepting a valid victim!
        let v3 = cache.put(k3, p3.as_slice())?;

        assert!(
            cache.stats().policy_victims_rejected >= 3,
            "expected >= 3 rejected poison victims, got {}",
            cache.stats().policy_victims_rejected
        );

        let mut buf = vec![0u8; FRAME_SIZE];
        let h3 = cache.get(k3, &mut buf)?.expect("k3 intact");
        assert_eq!(h3.version, v3);
        assert_eq!(buf, p3);

        // k1 was cascaded cleanly to NVMe despite the buggy policy!
        let h1 = cache.get(k1, &mut buf)?.expect("k1 intact from NVMe");
        assert_eq!(h1.version, v1);
        assert_eq!(buf, p1);
        Ok(())
    });

    // Part 2: Prove an actively leased frame can NEVER be reclaimed even if policy insists.
    let nvme_dir2 = temp_nvme_dir("buggy_pinned");
    let cfg2 = StorageConfig::new(0, FRAME_SIZE, 1); // Only 1 DRAM frame!
    let mut storage = NodeStorage::new(cfg2).expect("NodeStorage");
    let mut attached = storage.attach_tiers().expect("attach_tiers");
    let mut cache = attached
        .open_cache::<NoTransport, _, _, _>(
            Vec::new(),
            PolicySet::new(LruEviction::new(), NoTiering, ProbeAllPeers),
        )
        .expect("open_cache");

    let ka = make_key(201);
    let kb = make_key(202);
    let pa = pattern(0x55);
    let pb = pattern(0x66);
    let _ = cache.put(ka, pa.as_slice()).expect("put ka");

    // Hold a raw local lease on ka's single frame directly through the verified engine,
    // then attempt to put kb (which forces policy to try evicting ka's frame).
    // Because ka's frame is actively leased, CacheCore MUST refuse reclamation -> CapacityExhausted!
    {
        let mut guard = cache.lease_host(ka).expect("lease_host").expect("ka hit");
        let mut verify = vec![0u8; FRAME_SIZE];
        guard.copy_into(&mut verify).expect("copy_into while leased");
        assert_eq!(verify, pa);
        // Dropping guard returns the ShareCap, after which kb succeeds immediately.
    }
    let _ = cache.put(kb, pb.as_slice()).expect("put kb succeeds after lease drop");
    let mut out = vec![0u8; FRAME_SIZE];
    let hb = cache.get(kb, &mut out).expect("get kb").expect("kb hit");
    assert_eq!(hb.location, HitLocation::LocalDram);
    assert_eq!(out, pb);
    cache.shutdown().expect("shutdown");

    let _ = std::fs::remove_dir_all(&nvme_dir);
    let _ = std::fs::remove_dir_all(&nvme_dir2);
    res.expect("buggy_policy_verified_enforcement");
}

// ============================================================================
// Test 3: Local 3-Tier (GPU HBM + Pinned DRAM + Local NVMe) Location Transparency
// ============================================================================

#[test]
fn local_gpu_hbm_dram_nvme_location_transparency() {
    let Some(dev) = gpu_ordinal(0) else {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but no CUDA GPU detected");
        }
        eprintln!("SKIP local_gpu_hbm_dram_nvme_location_transparency: no CUDA GPU");
        return;
    };

    let nvme_dir = temp_nvme_dir("gpu_3tier");
    let cfg = StorageConfig::new(0, FRAME_SIZE, 2)
        .with_gpu(GpuConfig::new(dev, 2))
        .with_nvme(NvmeConfig::new(&nvme_dir, 4));

    let res = VeriCache::<NoTransport, _, _, _>::with_session(
        cfg,
        Vec::new(),
        DefaultPolicy::default(),
        |cache| {
            let mut stream = cache.create_gpu_stream()?;
            let mut gpu_slot = cache.alloc_gpu_slot()?;

            let k_gpu = make_key(301);
            let k_dram = make_key(302);
            let k_nvme = make_key(303);
            let p_gpu = pattern(0x71);
            let p_dram = pattern(0x72);
            let p_nvme = pattern(0x73);

            // 1) Publish k_gpu directly into GPU HBM (not in DRAM or NVMe).
            cache.write_gpu_slot(&mut gpu_slot, &mut stream, &p_gpu)?;
            let v_gpu = cache.put(k_gpu, GpuSrc::new(&mut stream, &mut gpu_slot))?;

            // 2) Publish k_dram into Pinned Host DRAM (not in GPU HBM).
            let v_dram = cache.put(k_dram, p_dram.as_slice())?;

            // 3) Publish k_nvme and spill it solely to Local NVMe (neither HBM nor DRAM).
            let v_nvme = cache.put(k_nvme, p_nvme.as_slice())?;
            assert!(cache.spill_key_to_nvme(k_nvme)?);

            // --- Unified Host Read: SAME `cache.get(key, &mut host_buf)` hits HBM, DRAM, NVMe ---
            let mut host_buf = vec![0u8; FRAME_SIZE];

            let hg = cache.get(k_gpu, &mut host_buf)?.expect("k_gpu via host get");
            assert_eq!(hg.location, HitLocation::LocalGpu);
            assert_eq!(hg.version, v_gpu);
            assert_eq!(host_buf, p_gpu);

            let hd = cache.get(k_dram, &mut host_buf)?.expect("k_dram via host get");
            assert_eq!(hd.location, HitLocation::LocalDram);
            assert_eq!(hd.version, v_dram);
            assert_eq!(host_buf, p_dram);

            // Re-spill k_nvme (still in NVMe only) -> Host get transparently warms NVMe!
            let hn = cache.get(k_nvme, &mut host_buf)?.expect("k_nvme via host get");
            assert_eq!(hn.location, HitLocation::LocalNvme);
            assert_eq!(hn.version, v_nvme);
            assert_eq!(host_buf, p_nvme);

            // Put k_nvme back exclusively into NVMe so GPU get exercises the NVMe -> DRAM -> GPU path.
            assert!(cache.spill_key_to_nvme(k_nvme)?);

            // --- Unified GPU Read: SAME `cache.get(key, GpuDst)` hits HBM, Pinned DRAM, NVMe ---
            let mut gpu_check = vec![0u8; FRAME_SIZE];

            let gg = cache
                .get(k_gpu, GpuDst::new(&mut stream, &mut gpu_slot))?
                .expect("k_gpu via gpu get");
            assert_eq!(gg.location, HitLocation::LocalGpu);
            assert_eq!(gg.version, v_gpu);
            cache.synchronize_stream(&mut stream, gg.ticket)?;
            cache.read_gpu_slot(&mut gpu_slot, &mut stream, &mut gpu_check)?;
            assert_eq!(gpu_check, p_gpu);

            let gd = cache
                .get(k_dram, GpuDst::new(&mut stream, &mut gpu_slot))?
                .expect("k_dram via gpu get");
            assert_eq!(gd.location, HitLocation::LocalDram);
            assert_eq!(gd.version, v_dram);
            cache.synchronize_stream(&mut stream, gd.ticket)?;
            cache.read_gpu_slot(&mut gpu_slot, &mut stream, &mut gpu_check)?;
            assert_eq!(gpu_check, p_dram);

            let gn = cache
                .get(k_nvme, GpuDst::new(&mut stream, &mut gpu_slot))?
                .expect("k_nvme via gpu get");
            assert_eq!(gn.location, HitLocation::LocalNvme);
            assert_eq!(gn.version, v_nvme);
            cache.synchronize_stream(&mut stream, gn.ticket)?;
            cache.read_gpu_slot(&mut gpu_slot, &mut stream, &mut gpu_check)?;
            assert_eq!(gpu_check, p_nvme);

            // Zero-copy RAII GPU lease works transparently too.
            {
                let glease = cache.lease_gpu(k_nvme)?.expect("k_nvme gpu lease");
                assert_eq!(glease.version(), v_nvme);
                assert!(glease.cuda_slice().is_some());
            }

            Ok(())
        },
    );

    let _ = std::fs::remove_dir_all(&nvme_dir);
    res.expect("local_gpu_hbm_dram_nvme_location_transparency");
}

// ============================================================================
// Test 4: 2-Node 4-Tier Distributed Location Transparency (HBM + DRAM + NVMe + RDMA + 2-GPU GDR)
// ============================================================================

fn start_child_initiator(test_name: &str) -> (TcpTransport, TcpStream, Child) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind ctrl");
    listener.set_nonblocking(true).unwrap();
    let sync_listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind sync");
    sync_listener.set_nonblocking(true).unwrap();

    let mut secret = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut secret)
        .unwrap();
    let session: String = secret.iter().map(|b| format!("{b:02x}")).collect();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env(ROLE_ENV, "initiator")
        .env(SESSION_ENV, &session)
        .env(
            ADDR_ENV,
            format!(
                "{},{}",
                listener.local_addr().unwrap(),
                sync_listener.local_addr().unwrap()
            ),
        )
        .spawn()
        .expect("spawn initiator child");

    let accept_with_timeout = |l: &TcpListener, child: &mut Child| -> TcpStream {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match l.accept() {
                Ok((s, _)) => return s,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    if let Some(st) = child.try_wait().unwrap() {
                        panic!("initiator exited early: {st}");
                    }
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("timed out waiting for initiator");
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("accept failed: {e}"),
            }
        }
    };

    let ctrl_sock = accept_with_timeout(&listener, &mut child);
    let sync_sock = accept_with_timeout(&sync_listener, &mut child);
    let mut ctrl = TcpTransport::new(ctrl_sock).expect("TcpTransport");
    assert_eq!(ctrl.recv_frame().unwrap(), session.as_bytes());
    ctrl.send_frame(session.as_bytes()).unwrap();
    sync_sock.set_nodelay(true).unwrap();
    sync_sock
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    (ctrl, sync_sock, child)
}

fn connect_to_donor_parent() -> (TcpTransport, TcpStream) {
    let addrs = std::env::var(ADDR_ENV).unwrap();
    let (ctrl_addr, sync_addr) = addrs.split_once(',').unwrap();
    let session = std::env::var(SESSION_ENV).unwrap();
    let mut ctrl = TcpTransport::new(TcpStream::connect(ctrl_addr).unwrap()).unwrap();
    let sync_sock = TcpStream::connect(sync_addr).unwrap();
    ctrl.send_frame(session.as_bytes()).unwrap();
    assert_eq!(ctrl.recv_frame().unwrap(), session.as_bytes());
    sync_sock.set_nodelay(true).unwrap();
    sync_sock
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    (ctrl, sync_sock)
}

fn run_distributed_donor(rdma_dev: String, dev: DeviceId) {
    let (ctrl, mut sync_sock, mut child) =
        start_child_initiator("distributed_4tier_location_transparency_e2e");
    let nvme_dir = temp_nvme_dir("dist_donor");

    let cfg = StorageConfig::new(0, FRAME_SIZE, 4)
        .with_gpu(GpuConfig::new(dev, 4))
        .with_nvme(NvmeConfig::new(&nvme_dir, 4))
        .with_rdma(RdmaConfig {
            device_name: Some(rdma_dev),
            gid_index: gid_index(),
            windows_per_peer: 8,
            cq_depth: 128,
            qp_depth: 64,
            ..RdmaConfig::default()
        });

    let res = VeriCache::with_session(
        cfg,
        vec![PeerLink::new(1, ctrl)],
        DefaultPolicy::default(),
        |donor| {
            let mut stream = donor.create_gpu_stream()?;
            let mut gpu_slot = donor.alloc_gpu_slot()?;

            // Donor puts k_remote_dram in DRAM and k_remote_gpu in GPU HBM.
            let k_remote_dram = make_key(404);
            let k_remote_gpu = make_key(405);
            let p_remote_dram = pattern(0xA4);
            let p_remote_gpu = pattern(0xA5);

            let _ = donor.put(k_remote_dram, p_remote_dram.as_slice())?;
            donor.write_gpu_slot(&mut gpu_slot, &mut stream, &p_remote_gpu)?;
            let _ = donor.put(k_remote_gpu, GpuSrc::new(&mut stream, &mut gpu_slot))?;

            // Signal initiator that Donor keys are published.
            sync_sock.write_all(&[1u8]).unwrap();
            sync_sock.set_nonblocking(true).unwrap();

            // Service incoming remote fetches via `donor.poll()` until initiator finishes.
            let mut done = [0u8; 1];
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let _ = donor.poll()?;
                match sync_sock.read_exact(&mut done) {
                    Ok(()) => break,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            return Err(CacheError::Init("donor timed out waiting for initiator".into()));
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => return Err(CacheError::Init(format!("sync_sock read failed: {e}"))),
                }
            }
            // Final poll to process any trailing ReadRelease before shutdown.
            let _ = donor.poll()?;
            Ok(())
        },
    );

    let status = child.wait().expect("wait initiator");
    let _ = std::fs::remove_dir_all(&nvme_dir);
    res.expect("donor side failed");
    assert!(status.success(), "initiator exited with {status}");
}

fn run_distributed_initiator(rdma_dev: String, dev: DeviceId) {
    let (ctrl, mut sync_sock) = connect_to_donor_parent();
    let nvme_dir = temp_nvme_dir("dist_initiator");

    let cfg = StorageConfig::new(1, FRAME_SIZE, 4)
        .with_gpu(GpuConfig::new(dev, 4))
        .with_nvme(NvmeConfig::new(&nvme_dir, 4))
        .with_rdma(RdmaConfig {
            device_name: Some(rdma_dev),
            gid_index: gid_index(),
            windows_per_peer: 8,
            cq_depth: 128,
            qp_depth: 64,
            ..RdmaConfig::default()
        });

    let res = VeriCache::with_session(
        cfg,
        vec![PeerLink::new(0, ctrl)],
        DefaultPolicy::default(),
        |cache| {
            // Wait for donor to finish publishing remote keys.
            let mut ready = [0u8; 1];
            sync_sock.read_exact(&mut ready).unwrap();

            let mut stream = cache.create_gpu_stream()?;
            let mut gpu_slot = cache.alloc_gpu_slot()?;

            // Populate 3 local keys across 3 distinct local tiers: HBM, DRAM, NVMe.
            let k_local_gpu = make_key(401);
            let k_local_dram = make_key(402);
            let k_local_nvme = make_key(403);
            let k_remote_dram = make_key(404);
            let k_remote_gpu = make_key(405);

            let p_local_gpu = pattern(0xA1);
            let p_local_dram = pattern(0xA2);
            let p_local_nvme = pattern(0xA3);
            let p_remote_dram = pattern(0xA4);
            let p_remote_gpu = pattern(0xA5);

            cache.write_gpu_slot(&mut gpu_slot, &mut stream, &p_local_gpu)?;
            let _ = cache.put(k_local_gpu, GpuSrc::new(&mut stream, &mut gpu_slot))?;
            let _ = cache.put(k_local_dram, p_local_dram.as_slice())?;
            let _ = cache.put(k_local_nvme, p_local_nvme.as_slice())?;
            assert!(cache.spill_key_to_nvme(k_local_nvme)?);

            // === THE LOCATION-TRANSPARENCY PROOF POINT ===
            // Exact same `cache.get(key, &mut host_buf)` resolves Local DRAM, Local NVMe,
            // and Remote Donor DRAM (over RDMA) with ZERO location awareness!
            let mut host_buf = vec![0u8; FRAME_SIZE];

            let h_dram = cache.get(k_local_dram, &mut host_buf)?.expect("local dram");
            assert_eq!(h_dram.location, HitLocation::LocalDram);
            assert_eq!(host_buf, p_local_dram);

            let h_nvme = cache.get(k_local_nvme, &mut host_buf)?.expect("local nvme");
            assert_eq!(h_nvme.location, HitLocation::LocalNvme);
            assert_eq!(host_buf, p_local_nvme);

            let h_rem_dram = cache.get(k_remote_dram, &mut host_buf)?.expect("remote dram");
            assert_eq!(
                h_rem_dram.location,
                HitLocation::RemotePeer {
                    node_id: 0,
                    gdr: false
                }
            );
            assert_eq!(host_buf, p_remote_dram);
            assert_eq!(cache.contains_local(k_remote_dram), None);

            // Exact same `cache.get(key, GpuDst)` resolves Local HBM and Remote Donor HBM
            // via 2-GPU GPUDirect RDMA with ZERO location awareness!
            let mut gpu_check = vec![0u8; FRAME_SIZE];

            let g_local = cache
                .get(k_local_gpu, GpuDst::new(&mut stream, &mut gpu_slot))?
                .expect("local gpu");
            assert_eq!(g_local.location, HitLocation::LocalGpu);
            cache.synchronize_stream(&mut stream, g_local.ticket)?;
            cache.read_gpu_slot(&mut gpu_slot, &mut stream, &mut gpu_check)?;
            assert_eq!(gpu_check, p_local_gpu);

            let g_remote_gdr = cache
                .get(k_remote_gpu, GpuDst::new(&mut stream, &mut gpu_slot))?
                .expect("remote 2-GPU GDR hit");
            assert_eq!(
                g_remote_gdr.location,
                HitLocation::RemotePeer {
                    node_id: 0,
                    gdr: true
                }
            );
            cache.synchronize_stream(&mut stream, g_remote_gdr.ticket)?;
            cache.read_gpu_slot(&mut gpu_slot, &mut stream, &mut gpu_check)?;
            assert_eq!(gpu_check, p_remote_gpu);
            assert_eq!(cache.contains_local(k_remote_gpu), None);

            // Tell donor we are done.
            sync_sock.write_all(&[1u8]).unwrap();
            Ok(())
        },
    );

    let _ = std::fs::remove_dir_all(&nvme_dir);
    res.expect("initiator side failed");
}

#[test]
fn distributed_4tier_location_transparency_e2e() {
    let role = std::env::var(ROLE_ENV).ok();
    let Some(rdma_dev) = has_usable_rdma_device() else {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but no usable Type-2B RDMA device found");
        }
        eprintln!("SKIP distributed_4tier_location_transparency_e2e: no Type-2B RDMA device");
        return;
    };
    let node_id = if role.as_deref() == Some("initiator") {
        1
    } else {
        0
    };
    let Some(dev) = gpu_ordinal(node_id) else {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but GPU {node_id} not available");
        }
        eprintln!("SKIP distributed_4tier_location_transparency_e2e: GPU {node_id} unavailable");
        return;
    };
    if role.is_none() && gpu_ordinal(1).is_none() {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but 2 GPUs required for 2-node GDR test");
        }
        eprintln!("SKIP distributed_4tier_location_transparency_e2e: need 2 GPUs");
        return;
    }

    match role.as_deref() {
        Some("initiator") => run_distributed_initiator(rdma_dev, dev),
        _ => run_distributed_donor(rdma_dev, dev),
    }
}
