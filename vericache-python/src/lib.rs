//! C ABI facade over `vericache::VeriCache` running inside a dedicated owning
//! worker thread.
//!
//! Architecture:
//! - Each `VeriCacheHandle` owns a channel to a dedicated Rust worker thread that
//!   runs `VeriCache::with_session(...)` on its own stack (preserving zero-`unsafe`
//!   lifetimes inside `vericache` while exposing a `'static` handle to Python/vLLM).
//! - Whenever no Python command is queued, the worker thread automatically calls
//!   `cache.poll()` every ~500 us so inbound remote RDMA/GDR `FetchRequest`s are
//!   served concurrently with Python/PyTorch forward passes without blocking on GIL.

mod backend_mode;
mod cuda_copy;
mod local;

pub use local::*;

use core::ffi::{CStr, c_char};
use std::cell::RefCell;
use std::ffi::CString;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use vericache::{
    CacheError, CacheStats, DefaultPolicy, GpuConfig, HitLocation, NvmeConfig, ObjectKey, PeerLink,
    RdmaConfig, StorageConfig, TcpTransport, TierKind, VeriCache,
};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(msg: impl Into<String>) {
    let s = msg.into().replace('\0', " ");
    let c = CString::new(s).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = c;
    });
}

fn loc_to_i32(loc: HitLocation) -> i32 {
    match loc {
        HitLocation::LocalGpu => 1,
        HitLocation::LocalDram => 2,
        HitLocation::LocalNvme => 3,
        HitLocation::RemotePeer { gdr: false, .. } => 4,
        HitLocation::RemotePeer { gdr: true, .. } => 5,
    }
}

fn loc_peer_node(loc: HitLocation) -> u64 {
    match loc {
        HitLocation::RemotePeer { node_id, .. } => node_id,
        _ => 0,
    }
}

/// C-compatible peer TCP link specification.
#[repr(C)]
pub struct VeriCachePeerSpecC {
    pub peer_node_id: u64,
    /// Null-terminated `host:port` TCP address.
    pub tcp_addr: *const c_char,
    /// `1` to bind & accept as TCP server; `0` to connect as TCP client.
    pub is_listener: u8,
    pub psn_hint: u32,
}

/// C-compatible storage configuration passed to `vericache_create`.
#[repr(C)]
pub struct VeriCacheConfigC {
    pub node_id: u64,
    pub frame_size: usize,
    pub dram_frames: u32,
    /// CUDA device ordinal, or `-1` to disable GPU tier.
    pub gpu_device_id: i32,
    pub hbm_frames: u32,
    /// Null-terminated directory path for NVMe `io_uring` slab, or null to disable NVMe.
    pub nvme_dir: *const c_char,
    pub nvme_frames: u32,
    /// Null-terminated verbs device name (e.g. `"mlx5_0"`), or null for auto/disabled.
    pub rdma_device: *const c_char,
    pub rdma_gid_index: u8,
    pub rdma_windows_per_peer: usize,
    pub peers: *const VeriCachePeerSpecC,
    pub num_peers: usize,
}

/// C-compatible hit metadata returned by `vericache_get_*`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VeriCacheHitInfoC {
    pub found: u8,
    /// `1=LocalGpu, 2=LocalDram, 3=LocalNvme, 4=RemoteRdmaHost, 5=RemoteGdrGpu`.
    pub location: i32,
    pub peer_node_id: u64,
    pub version_origin: u16,
    pub version_seq: u64,
}

/// C-compatible counters snapshot returned by `vericache_stats`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VeriCacheStatsC {
    pub puts: u64,
    pub gpu_hits: u64,
    pub dram_hits: u64,
    pub nvme_hits: u64,
    pub remote_hits: u64,
    pub misses: u64,
    pub policy_victims_rejected: u64,
    pub policy_victims_pinned: u64,
}

impl From<CacheStats> for VeriCacheStatsC {
    fn from(s: CacheStats) -> Self {
        Self {
            puts: s.puts,
            gpu_hits: s.gpu_hits,
            dram_hits: s.dram_hits,
            nvme_hits: s.nvme_hits,
            remote_hits: s.remote_hits,
            misses: s.misses,
            policy_victims_rejected: s.policy_victims_rejected,
            policy_victims_pinned: s.policy_victims_pinned,
        }
    }
}

struct OwnedPeerSpec {
    peer_node_id: u64,
    tcp_addr: String,
    is_listener: bool,
    psn_hint: u32,
}

struct OwnedConfig {
    storage: StorageConfig,
    peers: Vec<OwnedPeerSpec>,
}

enum Command {
    PutHost {
        key: ObjectKey,
        data: Vec<u8>,
        reply: Sender<Result<(u16, u64), String>>,
    },
    GetHost {
        key: ObjectKey,
        len: usize,
        reply: Sender<Result<(VeriCacheHitInfoC, Vec<u8>), String>>,
    },
    PutGpuPtr {
        key: ObjectKey,
        src_dev_ptr: usize,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<(u16, u64), String>>,
    },
    GetGpuPtr {
        key: ObjectKey,
        dst_dev_ptr: usize,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<VeriCacheHitInfoC, String>>,
    },
    PutGpuBatch {
        keys: Vec<ObjectKey>,
        src_dev_ptrs: Vec<usize>,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<usize, String>>,
    },
    GetGpuBatch {
        keys: Vec<ObjectKey>,
        dst_dev_ptrs: Vec<usize>,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<Vec<VeriCacheHitInfoC>, String>>,
    },
    ContainsBatchLocal {
        keys: Vec<ObjectKey>,
        reply: Sender<Vec<i32>>,
    },
    SpillToNvme {
        key: ObjectKey,
        reply: Sender<Result<bool, String>>,
    },
    EvictTier {
        key: ObjectKey,
        tier: TierKind,
        reply: Sender<Result<bool, String>>,
    },
    Poll {
        reply: Sender<Result<(), String>>,
    },
    Stats {
        reply: Sender<VeriCacheStatsC>,
    },
    Shutdown {
        reply: Sender<Result<(), String>>,
    },
}

/// Opaque handle owning a channel to the dedicated `VeriCache` worker thread.
pub struct VeriCacheHandle {
    tx: Sender<Command>,
    worker: Option<JoinHandle<()>>,
    frame_size: usize,
}

fn connect_peer_tcp(spec: &OwnedPeerSpec) -> Result<PeerLink<TcpTransport>, CacheError> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let stream = if spec.is_listener {
        let listener = TcpListener::bind(&spec.tcp_addr).map_err(|e| {
            CacheError::Init(format!("bind peer listener {} failed: {e}", spec.tcp_addr))
        })?;
        listener.set_nonblocking(true).map_err(|e| {
            CacheError::Init(format!("set_nonblocking on {} failed: {e}", spec.tcp_addr))
        })?;
        loop {
            match listener.accept() {
                Ok((s, _)) => break s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(CacheError::Init(format!(
                            "timed out accepting peer {} on {}",
                            spec.peer_node_id, spec.tcp_addr
                        )));
                    }
                    thread::sleep(Duration::from_millis(5));
                }
                Err(e) => {
                    return Err(CacheError::Init(format!(
                        "accept peer {} on {} failed: {e}",
                        spec.peer_node_id, spec.tcp_addr
                    )));
                }
            }
        }
    } else {
        loop {
            match TcpStream::connect(&spec.tcp_addr) {
                Ok(s) => break s,
                Err(_) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(CacheError::Init(format!(
                        "connect to peer {} at {} failed: {e}",
                        spec.peer_node_id, spec.tcp_addr
                    )));
                }
            }
        }
    };
    let transport = TcpTransport::new(stream)
        .map_err(|e| CacheError::Init(format!("TcpTransport::new failed: {e}")))?;
    Ok(PeerLink {
        peer_node_id: spec.peer_node_id,
        transport,
        psn_hint: spec.psn_hint,
    })
}

fn worker_loop(
    cfg: OwnedConfig,
    rx: mpsc::Receiver<Command>,
    ready_tx: Sender<Result<(), String>>,
) {
    let mut peer_links = Vec::with_capacity(cfg.peers.len());
    for p in &cfg.peers {
        match connect_peer_tcp(p) {
            Ok(link) => peer_links.push(link),
            Err(e) => {
                let _ = ready_tx.send(Err(e.to_string()));
                return;
            }
        }
    }

    let mut ready_tx = Some(ready_tx);
    let session_res = VeriCache::with_session(
        cfg.storage,
        peer_links,
        DefaultPolicy::default(),
        |cache| {
            let mut gpu_io = if cache.engine().gpu_tier().is_some() {
                let stream = cache.create_gpu_stream()?;
                let slot = cache.alloc_gpu_slot()?;
                Some((stream, slot))
            } else {
                None
            };

            if let Some(tx) = ready_tx.take() {
                let _ = tx.send(Ok(()));
            }

            let idle_poll = Duration::from_micros(500);
            loop {
                match rx.recv_timeout(idle_poll) {
                    Ok(cmd) => match cmd {
                        Command::PutHost { key, data, reply } => {
                            let res = cache
                                .put_host(key, &data)
                                .map(|v| (v.origin_node, v.seq))
                                .map_err(|e| e.to_string());
                            let _ = reply.send(res);
                        }
                        Command::GetHost { key, len, reply } => {
                            let mut buf = vec![0u8; len];
                            let res = match cache.get_into_host(key, &mut buf) {
                                Ok(Some(hit)) => Ok((
                                    VeriCacheHitInfoC {
                                        found: 1,
                                        location: loc_to_i32(hit.location),
                                        peer_node_id: loc_peer_node(hit.location),
                                        version_origin: hit.version.origin_node,
                                        version_seq: hit.version.seq,
                                    },
                                    buf,
                                )),
                                Ok(None) => Ok((VeriCacheHitInfoC::default(), Vec::new())),
                                Err(e) => Err(e.to_string()),
                            };
                            let _ = reply.send(res);
                        }
                        Command::PutGpuPtr {
                            key,
                            src_dev_ptr,
                            len,
                            cuda_stream,
                            reply,
                        } => {
                            let res = match gpu_io.as_mut() {
                                None => Err("GPU tier is not enabled".into()),
                                Some((stream, slot)) => {
                                    if len != slot.len() {
                                        Err(format!(
                                            "length mismatch: got {len}, expected {}",
                                            slot.len()
                                        ))
                                    } else {
                                        match cuda_copy::memcpy_d2d_sync(
                                            slot.device_addr(),
                                            src_dev_ptr,
                                            len,
                                            cuda_stream,
                                        ) {
                                            Err(code) => Err(format!(
                                                "cudaMemcpyAsync(D2D) failed with code {code}"
                                            )),
                                            Ok(()) => cache
                                                .put_gpu_external(key, stream, slot)
                                                .map(|v| (v.origin_node, v.seq))
                                                .map_err(|e| e.to_string()),
                                        }
                                    }
                                }
                            };
                            let _ = reply.send(res);
                        }
                        Command::GetGpuPtr {
                            key,
                            dst_dev_ptr,
                            len,
                            cuda_stream,
                            reply,
                        } => {
                            let res = if len != cache.frame_size() {
                                Err(format!(
                                    "length mismatch: got {len}, expected {}",
                                    cache.frame_size()
                                ))
                            } else {
                                match cache.lease_gpu(key) {
                                    Ok(Some(lease)) => {
                                        let loc = lease.location();
                                        let ver = lease.version();
                                        match lease.device_addr() {
                                            Some(src_addr) => match cuda_copy::memcpy_d2d_sync(
                                                dst_dev_ptr,
                                                src_addr,
                                                len,
                                                cuda_stream,
                                            ) {
                                                Ok(()) => Ok(VeriCacheHitInfoC {
                                                    found: 1,
                                                    location: loc_to_i32(loc),
                                                    peer_node_id: loc_peer_node(loc),
                                                    version_origin: ver.origin_node,
                                                    version_seq: ver.seq,
                                                }),
                                                Err(code) => Err(format!(
                                                    "cudaMemcpyAsync(D2D) failed with code {code}"
                                                )),
                                            },
                                            None => Err("leased GPU frame has no slice".into()),
                                        }
                                    }
                                    Ok(None) => Ok(VeriCacheHitInfoC::default()),
                                    Err(e) => Err(e.to_string()),
                                }
                            };
                            let _ = reply.send(res);
                        }
                        Command::PutGpuBatch {
                            keys,
                            src_dev_ptrs,
                            len,
                            cuda_stream,
                            reply,
                        } => {
                            let res = match gpu_io.as_mut() {
                                None => Err("GPU tier is not enabled".into()),
                                Some((stream, slot)) => {
                                    if len != slot.len() {
                                        Err(format!(
                                            "length mismatch: got {len}, expected {}",
                                            slot.len()
                                        ))
                                    } else {
                                        let mut count = 0usize;
                                        let mut err = None;
                                        for (key, src_ptr) in
                                            keys.into_iter().zip(src_dev_ptrs.into_iter())
                                        {
                                            if let Err(code) = cuda_copy::memcpy_d2d_sync(
                                                slot.device_addr(),
                                                src_ptr,
                                                len,
                                                cuda_stream,
                                            ) {
                                                err = Some(format!(
                                                    "cudaMemcpyAsync(D2D) failed with code {code}"
                                                ));
                                                break;
                                            }
                                            if let Err(e) =
                                                cache.put_gpu_external(key, stream, slot)
                                            {
                                                err = Some(e.to_string());
                                                break;
                                            }
                                            count += 1;
                                        }
                                        match err {
                                            Some(e) => Err(e),
                                            None => Ok(count),
                                        }
                                    }
                                }
                            };
                            let _ = reply.send(res);
                        }
                        Command::GetGpuBatch {
                            keys,
                            dst_dev_ptrs,
                            len,
                            cuda_stream,
                            reply,
                        } => {
                            let res = if len != cache.frame_size() {
                                Err(format!(
                                    "length mismatch: got {len}, expected {}",
                                    cache.frame_size()
                                ))
                            } else {
                                let mut hits = Vec::with_capacity(keys.len());
                                let mut err = None;
                                for (key, dst_ptr) in
                                    keys.into_iter().zip(dst_dev_ptrs.into_iter())
                                {
                                    match cache.lease_gpu(key) {
                                        Ok(Some(lease)) => {
                                            let loc = lease.location();
                                            let ver = lease.version();
                                            let Some(src_addr) = lease.device_addr() else {
                                                err = Some("leased GPU frame has no slice".into());
                                                break;
                                            };
                                            if let Err(code) = cuda_copy::memcpy_d2d_sync(
                                                dst_ptr,
                                                src_addr,
                                                len,
                                                cuda_stream,
                                            ) {
                                                err = Some(format!(
                                                    "cudaMemcpyAsync(D2D) failed with code {code}"
                                                ));
                                                break;
                                            }
                                            hits.push(VeriCacheHitInfoC {
                                                found: 1,
                                                location: loc_to_i32(loc),
                                                peer_node_id: loc_peer_node(loc),
                                                version_origin: ver.origin_node,
                                                version_seq: ver.seq,
                                            });
                                        }
                                        Ok(None) => hits.push(VeriCacheHitInfoC::default()),
                                        Err(e) => {
                                            err = Some(e.to_string());
                                            break;
                                        }
                                    }
                                }
                                match err {
                                    Some(e) => Err(e),
                                    None => Ok(hits),
                                }
                            };
                            let _ = reply.send(res);
                        }
                        Command::ContainsBatchLocal { keys, reply } => {
                            let locs = cache
                                .contains_batch_local(&keys)
                                .into_iter()
                                .map(|opt| opt.map(loc_to_i32).unwrap_or(0))
                                .collect();
                            let _ = reply.send(locs);
                        }
                        Command::SpillToNvme { key, reply } => {
                            let _ = reply
                                .send(cache.spill_key_to_nvme(key).map_err(|e| e.to_string()));
                        }
                        Command::EvictTier { key, tier, reply } => {
                            let _ = reply
                                .send(cache.evict_from_tier(key, tier).map_err(|e| e.to_string()));
                        }
                        Command::Poll { reply } => {
                            let _ = reply.send(cache.poll().map(|_| ()).map_err(|e| e.to_string()));
                        }
                        Command::Stats { reply } => {
                            let _ = reply.send((*cache.stats()).into());
                        }
                        Command::Shutdown { reply } => {
                            let _ = cache.poll();
                            let _ = reply.send(Ok(()));
                            break;
                        }
                    },
                    Err(RecvTimeoutError::Timeout) => {
                        let _ = cache.poll();
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        let _ = cache.poll();
                        break;
                    }
                }
            }
            Ok(())
        },
    );

    if let Some(tx) = ready_tx.take() {
        let msg = match session_res {
            Ok(()) => "worker exited before ready".into(),
            Err(e) => e.to_string(),
        };
        let _ = tx.send(Err(msg));
    }
}

unsafe fn read_key(key4: *const u64) -> Option<ObjectKey> {
    if key4.is_null() {
        return None;
    }
    let s = unsafe { core::slice::from_raw_parts(key4, 4) };
    Some(ObjectKey([s[0], s[1], s[2], s[3]]))
}

unsafe fn cstr_to_string(ptr: *const c_char) -> Result<Option<String>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| format!("invalid UTF-8 string: {e}"))?;
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s.to_owned()))
    }
}

/// Return pointer to thread-local null-terminated error string from the last failed call.
#[unsafe(no_mangle)]
pub extern "C" fn vericache_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Create a new `VeriCache` instance backed by a dedicated owning worker thread.
/// Returns null on error (inspect `vericache_last_error()`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_create(cfg: *const VeriCacheConfigC) -> *mut VeriCacheHandle {
    if cfg.is_null() {
        set_last_error("cfg pointer is null");
        return core::ptr::null_mut();
    }
    let c = unsafe { &*cfg };
    if c.frame_size == 0
        || c.dram_frames == 0
        || c.frame_size
            .checked_mul(c.dram_frames as usize)
            .is_none_or(|size| size > isize::MAX as usize)
    {
        set_last_error("invalid frame_size or dram_frames");
        return core::ptr::null_mut();
    }
    if c.num_peers > (isize::MAX as usize) / size_of::<VeriCachePeerSpecC>() {
        set_last_error("num_peers exceeds the addressable peer array size");
        return core::ptr::null_mut();
    }
    let mut storage = StorageConfig::new(c.node_id, c.frame_size, c.dram_frames);
    if c.gpu_device_id >= 0 && c.hbm_frames > 0 {
        storage = storage.with_gpu(GpuConfig::new(c.gpu_device_id as u32, c.hbm_frames));
    }
    match unsafe { cstr_to_string(c.nvme_dir) } {
        Ok(Some(dir)) if c.nvme_frames > 0 => {
            storage = storage.with_nvme(NvmeConfig::new(PathBuf::from(dir), c.nvme_frames));
        }
        Ok(_) => {}
        Err(e) => {
            set_last_error(e);
            return core::ptr::null_mut();
        }
    }

    let mut peers = Vec::new();
    if c.num_peers > 0 {
        if c.peers.is_null() {
            set_last_error("num_peers > 0 but peers pointer is null");
            return core::ptr::null_mut();
        }
        let rdma_dev = match unsafe { cstr_to_string(c.rdma_device) } {
            Ok(d) => d,
            Err(e) => {
                set_last_error(e);
                return core::ptr::null_mut();
            }
        };
        let win = if c.rdma_windows_per_peer > 0 {
            c.rdma_windows_per_peer
        } else {
            8
        };
        storage = storage.with_rdma(RdmaConfig {
            device_name: rdma_dev,
            gid_index: c.rdma_gid_index,
            windows_per_peer: win,
            cq_depth: 128,
            qp_depth: 64,
        });
        let raw_peers = unsafe { core::slice::from_raw_parts(c.peers, c.num_peers) };
        for p in raw_peers {
            let addr = match unsafe { cstr_to_string(p.tcp_addr) } {
                Ok(Some(a)) => a,
                Ok(None) => {
                    set_last_error("peer tcp_addr is empty");
                    return core::ptr::null_mut();
                }
                Err(e) => {
                    set_last_error(e);
                    return core::ptr::null_mut();
                }
            };
            peers.push(OwnedPeerSpec {
                peer_node_id: p.peer_node_id,
                tcp_addr: addr,
                is_listener: p.is_listener != 0,
                psn_hint: if p.psn_hint == 0 { 0x1000 } else { p.psn_hint },
            });
        }
    }

    if let Err(error) = backend_mode::claim_distributed() {
        set_last_error(error);
        return core::ptr::null_mut();
    }

    let frame_size = c.frame_size;
    let owned_cfg = OwnedConfig { storage, peers };
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();

    let worker = match thread::Builder::new()
        .name(format!("vericache-node-{}", c.node_id))
        .stack_size(16 * 1024 * 1024)
        .spawn(move || worker_loop(owned_cfg, rx, ready_tx))
    {
        Ok(h) => h,
        Err(e) => {
            set_last_error(format!("failed to spawn worker thread: {e}"));
            return core::ptr::null_mut();
        }
    };

    match ready_rx.recv() {
        Ok(Ok(())) => Box::into_raw(Box::new(VeriCacheHandle {
            tx,
            worker: Some(worker),
            frame_size,
        })),
        Ok(Err(msg)) => {
            let _ = worker.join();
            set_last_error(msg);
            core::ptr::null_mut()
        }
        Err(_) => {
            let _ = worker.join();
            set_last_error("worker thread exited unexpectedly during initialization");
            core::ptr::null_mut()
        }
    }
}

/// Gracefully shut down the `VeriCache` worker thread and free its handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_destroy(handle: *mut VeriCacheHandle) {
    if handle.is_null() {
        return;
    }
    let mut boxed = unsafe { Box::from_raw(handle) };
    let (rtx, rrx) = mpsc::channel();
    let _ = boxed.tx.send(Command::Shutdown { reply: rtx });
    let _ = rrx.recv();
    if let Some(w) = boxed.worker.take() {
        let _ = w.join();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_put_host(
    handle: *mut VeriCacheHandle,
    key4: *const u64,
    data: *const u8,
    len: usize,
    out_ver_origin: *mut u16,
    out_ver_seq: *mut u64,
) -> i32 {
    if handle.is_null() || data.is_null() {
        set_last_error("null handle or data");
        return -1;
    }
    let Some(key) = (unsafe { read_key(key4) }) else {
        set_last_error("null key4");
        return -1;
    };
    let h = unsafe { &*handle };
    if len != h.frame_size {
        set_last_error(format!("length mismatch: got {len}, expected {}", h.frame_size));
        return -2;
    }
    let buf = unsafe { core::slice::from_raw_parts(data, len) }.to_vec();
    let (rtx, rrx) = mpsc::channel();
    if h.tx.send(Command::PutHost { key, data: buf, reply: rtx }).is_err() {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok((orig, seq))) => {
            if !out_ver_origin.is_null() {
                unsafe { *out_ver_origin = orig };
            }
            if !out_ver_seq.is_null() {
                unsafe { *out_ver_seq = seq };
            }
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_get_host(
    handle: *mut VeriCacheHandle,
    key4: *const u64,
    dst: *mut u8,
    len: usize,
    out_hit: *mut VeriCacheHitInfoC,
) -> i32 {
    if handle.is_null() || dst.is_null() {
        set_last_error("null handle or dst");
        return -1;
    }
    let Some(key) = (unsafe { read_key(key4) }) else {
        set_last_error("null key4");
        return -1;
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx.send(Command::GetHost { key, len, reply: rtx }).is_err() {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok((info, data))) => {
            if info.found != 0 {
                let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst, len) };
                dst_slice.copy_from_slice(&data);
            }
            if !out_hit.is_null() {
                unsafe { *out_hit = info };
            }
            info.found as i32
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_put_gpu_ptr(
    handle: *mut VeriCacheHandle,
    key4: *const u64,
    src_dev_ptr: usize,
    len: usize,
    cuda_stream: usize,
    out_ver_origin: *mut u16,
    out_ver_seq: *mut u64,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    let Some(key) = (unsafe { read_key(key4) }) else {
        set_last_error("null key4");
        return -1;
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx
        .send(Command::PutGpuPtr {
            key,
            src_dev_ptr,
            len,
            cuda_stream,
            reply: rtx,
        })
        .is_err()
    {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok((orig, seq))) => {
            if !out_ver_origin.is_null() {
                unsafe { *out_ver_origin = orig };
            }
            if !out_ver_seq.is_null() {
                unsafe { *out_ver_seq = seq };
            }
            0
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_get_gpu_ptr(
    handle: *mut VeriCacheHandle,
    key4: *const u64,
    dst_dev_ptr: usize,
    len: usize,
    cuda_stream: usize,
    out_hit: *mut VeriCacheHitInfoC,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    let Some(key) = (unsafe { read_key(key4) }) else {
        set_last_error("null key4");
        return -1;
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx
        .send(Command::GetGpuPtr {
            key,
            dst_dev_ptr,
            len,
            cuda_stream,
            reply: rtx,
        })
        .is_err()
    {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(info)) => {
            if !out_hit.is_null() {
                unsafe { *out_hit = info };
            }
            info.found as i32
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_put_gpu_batch(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    src_dev_ptrs: *const usize,
    count: usize,
    len: usize,
    cuda_stream: usize,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    if count == 0 {
        return 0;
    }
    if keys4.is_null() || src_dev_ptrs.is_null() {
        set_last_error("null keys4 or src_dev_ptrs");
        return -1;
    }
    let raw_keys = unsafe { core::slice::from_raw_parts(keys4, count * 4) };
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * 4;
        keys.push(ObjectKey([
            raw_keys[base],
            raw_keys[base + 1],
            raw_keys[base + 2],
            raw_keys[base + 3],
        ]));
    }
    let ptrs = unsafe { core::slice::from_raw_parts(src_dev_ptrs, count) }.to_vec();
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx
        .send(Command::PutGpuBatch {
            keys,
            src_dev_ptrs: ptrs,
            len,
            cuda_stream,
            reply: rtx,
        })
        .is_err()
    {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(n)) => n as i32,
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_get_gpu_batch(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    dst_dev_ptrs: *const usize,
    count: usize,
    len: usize,
    cuda_stream: usize,
    out_hits: *mut VeriCacheHitInfoC,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    if count == 0 {
        return 0;
    }
    if keys4.is_null() || dst_dev_ptrs.is_null() {
        set_last_error("null keys4 or dst_dev_ptrs");
        return -1;
    }
    let raw_keys = unsafe { core::slice::from_raw_parts(keys4, count * 4) };
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * 4;
        keys.push(ObjectKey([
            raw_keys[base],
            raw_keys[base + 1],
            raw_keys[base + 2],
            raw_keys[base + 3],
        ]));
    }
    let ptrs = unsafe { core::slice::from_raw_parts(dst_dev_ptrs, count) }.to_vec();
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx
        .send(Command::GetGpuBatch {
            keys,
            dst_dev_ptrs: ptrs,
            len,
            cuda_stream,
            reply: rtx,
        })
        .is_err()
    {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(hits)) => {
            let mut found_count = 0i32;
            if !out_hits.is_null() {
                let dst_hits = unsafe { core::slice::from_raw_parts_mut(out_hits, count) };
                for (i, hinfo) in hits.into_iter().enumerate() {
                    if hinfo.found != 0 {
                        found_count += 1;
                    }
                    dst_hits[i] = hinfo;
                }
            } else {
                for hinfo in hits {
                    if hinfo.found != 0 {
                        found_count += 1;
                    }
                }
            }
            found_count
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_contains_batch_local(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    count: usize,
    out_locs: *mut i32,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    if count == 0 {
        return 0;
    }
    if keys4.is_null() || out_locs.is_null() {
        set_last_error("null keys4 or out_locs");
        return -1;
    }
    let raw_keys = unsafe { core::slice::from_raw_parts(keys4, count * 4) };
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * 4;
        keys.push(ObjectKey([
            raw_keys[base],
            raw_keys[base + 1],
            raw_keys[base + 2],
            raw_keys[base + 3],
        ]));
    }
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx
        .send(Command::ContainsBatchLocal { keys, reply: rtx })
        .is_err()
    {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(locs) => {
            let out = unsafe { core::slice::from_raw_parts_mut(out_locs, count) };
            out.copy_from_slice(&locs);
            0
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_spill_to_nvme(
    handle: *mut VeriCacheHandle,
    key4: *const u64,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    let Some(key) = (unsafe { read_key(key4) }) else {
        set_last_error("null key4");
        return -1;
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx.send(Command::SpillToNvme { key, reply: rtx }).is_err() {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(spilled)) => i32::from(spilled),
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_evict_tier(
    handle: *mut VeriCacheHandle,
    key4: *const u64,
    tier_kind: i32,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    let Some(key) = (unsafe { read_key(key4) }) else {
        set_last_error("null key4");
        return -1;
    };
    let tier = match tier_kind {
        1 => TierKind::GpuVram,
        2 => TierKind::HostDram,
        3 => TierKind::LocalNvme,
        _ => {
            set_last_error(format!("invalid tier_kind {tier_kind}"));
            return -2;
        }
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx
        .send(Command::EvictTier {
            key,
            tier,
            reply: rtx,
        })
        .is_err()
    {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(evicted)) => i32::from(evicted),
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_poll(handle: *mut VeriCacheHandle) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx.send(Command::Poll { reply: rtx }).is_err() {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -4
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_stats(
    handle: *mut VeriCacheHandle,
    out_stats: *mut VeriCacheStatsC,
) -> i32 {
    if handle.is_null() || out_stats.is_null() {
        set_last_error("null handle or out_stats");
        return -1;
    }
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.tx.send(Command::Stats { reply: rtx }).is_err() {
        set_last_error("worker channel closed");
        return -3;
    }
    match rrx.recv() {
        Ok(stats) => {
            unsafe { *out_stats = stats };
            0
        }
        Err(_) => {
            set_last_error("worker reply channel closed");
            -3
        }
    }
}
