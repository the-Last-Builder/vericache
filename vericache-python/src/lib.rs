//! C ABI facade over `vericache::VeriCache` running inside a dedicated owning
//! worker thread.
//!
//! Architecture:
//! - Each `VeriCacheHandle` owns a dual-priority work queue (`high_pri` for
//!   latency-critical LOAD / LOOKUP / STATS, `low_pri` for background SAVE /
//!   maintenance) backed by a dedicated Rust worker thread that runs
//!   `VeriCache::with_session(...)` on its own stack.
//! - Urgent `high_pri` commands preempt chunked SAVE batches between every
//!   16-block chunk (`<1ms`) using strictly isolated CUDA events and batch
//!   buffers (`load_sync_event` + `load_batch_buf` vs `store_sync_events[2]` +
//!   `store_batch_buf` + `pinned_d2h[2]`).
//! - Whenever no Python command is queued, the worker thread automatically calls
//!   `cache.poll()` every ~100 us so inbound remote RDMA/GDR `FetchRequest`s are
//!   served concurrently with Python/PyTorch forward passes without blocking on GIL.

mod cuda_copy;

use core::ffi::{CStr, c_char};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use vericache::{
    BatchReadSource, CacheError, CacheStats, CascadeTiering, EvictionPolicy, FrameId, GpuConfig,
    HitLocation, LruEviction, NvmeConfig, ObjectKey, PeerLink, PeerRoutingPolicy, PolicySet,
    ProbeAllPeers, RdmaConfig, StorageConfig, TcpTransport, TierKind, TieringPolicy, VeriCache,
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
    gpu_device_id: i32,
}

/// Paged multi-layer GPU KV tensor layout descriptor.
/// Each region `r` resides in GPU HBM at `base_ptrs[r] + block_id * block_strides[r]`
/// with length `sizes[r]` bytes, mapping to byte range
/// `[offsets[r], offsets[r] + sizes[r])` within a `frame_size`-byte VeriCache frame.
struct PagedRegions {
    base_ptrs: Vec<usize>,
    block_strides: Vec<usize>,
    offsets: Vec<usize>,
    sizes: Vec<usize>,
}

impl PagedRegions {
    fn len(&self) -> usize {
        self.base_ptrs.len()
    }

    fn validate_bounds(&self, frame_size: usize) -> Result<(), String> {
        let n = self.base_ptrs.len();
        if n == 0
            || self.block_strides.len() != n
            || self.offsets.len() != n
            || self.sizes.len() != n
        {
            return Err("invalid paged region descriptor lengths".into());
        }
        for r in 0..n {
            let sz = self.sizes[r];
            let off = self.offsets[r];
            let end = off
                .checked_add(sz)
                .ok_or_else(|| format!("region {r} offset+size overflow"))?;
            if end > frame_size {
                return Err(format!(
                    "region {r} range [{off}..{end}) exceeds frame_size {frame_size}"
                ));
            }
            if sz > 0 && self.base_ptrs[r] == 0 {
                return Err(format!("region {r} has null base_ptr with size {sz}"));
            }
        }
        Ok(())
    }

    /// Validate region bounds and non-overlap in frame space, returning sorted
    /// uncovered `(start, end)` byte ranges within `0..frame_size` to zero-pad.
    fn validate_and_zero_gaps(&self, frame_size: usize) -> Result<Vec<(usize, usize)>, String> {
        self.validate_bounds(frame_size)?;
        let mut intervals: Vec<(usize, usize)> = (0..self.len())
            .filter(|&r| self.sizes[r] > 0)
            .map(|r| (self.offsets[r], self.offsets[r] + self.sizes[r]))
            .collect();
        intervals.sort_unstable_by_key(|&(start, _)| start);
        let mut gaps = Vec::new();
        let mut cursor = 0usize;
        for (start, end) in intervals {
            if start < cursor {
                return Err(format!(
                    "overlapping region frame intervals: [{start}..{end}) overlaps prior end {cursor}"
                ));
            }
            if start > cursor {
                gaps.push((cursor, start));
            }
            cursor = end;
        }
        if cursor < frame_size {
            gaps.push((cursor, frame_size));
        }
        Ok(gaps)
    }
}

/// High-priority commands that preempt chunked background SAVE batches.
enum HighCommand {
    GetHost {
        key: ObjectKey,
        len: usize,
        reply: Sender<Result<(VeriCacheHitInfoC, Vec<u8>), String>>,
    },
    GetGpuPtr {
        key: ObjectKey,
        dst_dev_ptr: usize,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<VeriCacheHitInfoC, String>>,
    },
    GetGpuBatch {
        keys: Vec<ObjectKey>,
        dst_dev_ptrs: Vec<usize>,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<Vec<VeriCacheHitInfoC>, String>>,
    },
    GetGpuPagedBatch {
        keys: Vec<ObjectKey>,
        gpu_block_ids: Vec<usize>,
        regions: PagedRegions,
        cuda_stream: usize,
        reply: Sender<Result<Vec<VeriCacheHitInfoC>, String>>,
    },
    #[cfg(test)]
    Noop,
}

/// Low-priority commands (background SAVE and tier maintenance).
enum LowCommand {
    PutHost {
        key: ObjectKey,
        data: Vec<u8>,
        reply: Sender<Result<(u16, u64), String>>,
    },
    PutGpuPtr {
        key: ObjectKey,
        src_dev_ptr: usize,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<(u16, u64), String>>,
    },
    PutGpuBatch {
        keys: Vec<ObjectKey>,
        src_dev_ptrs: Vec<usize>,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<usize, String>>,
    },
    PutGpuToDramBatch {
        keys: Vec<ObjectKey>,
        src_dev_ptrs: Vec<usize>,
        len: usize,
        cuda_stream: usize,
        reply: Sender<Result<usize, String>>,
    },
    PutGpuPagedDramBatch {
        keys: Vec<ObjectKey>,
        gpu_block_ids: Vec<usize>,
        regions: PagedRegions,
        cuda_stream: usize,
        reply: Sender<Result<usize, String>>,
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
    Shutdown {
        reply: Sender<Result<(), String>>,
    },
}

enum NextWork {
    High(HighCommand),
    Low(LowCommand),
    Timeout,
    Disconnected,
}

struct WorkQueueState {
    high_pri: VecDeque<HighCommand>,
    low_pri: VecDeque<LowCommand>,
    closed: bool,
}

/// Dual-priority thread-safe command queue with immediate condvar wakeup and
/// lock-free `has_high_pri()` fast check for mid-SAVE preemption.
struct WorkQueue {
    state: Mutex<WorkQueueState>,
    cv: Condvar,
    high_pri_pending: AtomicUsize,
    worker_alive: AtomicBool,
}

impl WorkQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(WorkQueueState {
                high_pri: VecDeque::new(),
                low_pri: VecDeque::new(),
                closed: false,
            }),
            cv: Condvar::new(),
            high_pri_pending: AtomicUsize::new(0),
            worker_alive: AtomicBool::new(true),
        }
    }

    #[inline]
    fn is_alive(&self) -> bool {
        self.worker_alive.load(Ordering::Acquire)
    }

    fn send_high(&self, cmd: HighCommand) -> Result<(), ()> {
        let mut guard = self.state.lock().map_err(|_| ())?;
        if guard.closed {
            return Err(());
        }
        guard.high_pri.push_back(cmd);
        self.high_pri_pending
            .store(guard.high_pri.len(), Ordering::Release);
        self.cv.notify_one();
        Ok(())
    }

    fn send_low(&self, cmd: LowCommand) -> Result<(), ()> {
        let mut guard = self.state.lock().map_err(|_| ())?;
        if guard.closed {
            return Err(());
        }
        guard.low_pri.push_back(cmd);
        self.cv.notify_one();
        Ok(())
    }

    #[inline]
    fn has_high_pri(&self) -> bool {
        self.high_pri_pending.load(Ordering::Acquire) > 0
    }

    fn try_pop_high(&self) -> Option<HighCommand> {
        if !self.has_high_pri() {
            return None;
        }
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let cmd = guard.high_pri.pop_front();
        if cmd.is_some() {
            self.high_pri_pending
                .store(guard.high_pri.len(), Ordering::Release);
        }
        cmd
    }

    fn recv_timeout(&self, timeout: Duration) -> NextWork {
        let deadline = Instant::now() + timeout;
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(cmd) = guard.high_pri.pop_front() {
                self.high_pri_pending
                    .store(guard.high_pri.len(), Ordering::Release);
                return NextWork::High(cmd);
            }
            if let Some(cmd) = guard.low_pri.pop_front() {
                return NextWork::Low(cmd);
            }
            if guard.closed {
                return NextWork::Disconnected;
            }
            let now = Instant::now();
            if now >= deadline {
                return NextWork::Timeout;
            }
            let rem = deadline.saturating_duration_since(now);
            let (g, _) = self
                .cv
                .wait_timeout(guard, rem)
                .unwrap_or_else(|e| e.into_inner());
            guard = g;
        }
    }

    fn close(&self) {
        self.worker_alive.store(false, Ordering::Release);
        let (dropped_high, dropped_low) = {
            let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
            guard.closed = true;
            self.high_pri_pending.store(0, Ordering::Release);
            let h = core::mem::take(&mut guard.high_pri);
            let l = core::mem::take(&mut guard.low_pri);
            self.cv.notify_all();
            (h, l)
        };
        // Drop queued commands outside the lock so every blocked caller's `reply_rx.recv()`
        // immediately wakes with `Err(RecvError)` instead of hanging forever on worker exit/panic.
        drop(dropped_high);
        drop(dropped_low);
    }
}

struct CloseQueueOnDrop(Arc<WorkQueue>);

impl Drop for CloseQueueOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

const KEY_MIRROR_SHARDS: usize = 64;
const MAX_EVICTED_KEYS: usize = 65_536;
const TIER_BIT_GPU: u8 = 1 << 0;
const TIER_BIT_DRAM: u8 = 1 << 1;
const TIER_BIT_NVME: u8 = 1 << 2;

#[inline]
fn tier_to_idx(tier: TierKind) -> Option<u8> {
    match tier {
        TierKind::GpuVram => Some(0),
        TierKind::HostDram => Some(1),
        TierKind::LocalNvme => Some(2),
        _ => None,
    }
}

#[inline]
fn tier_to_bit(tier: TierKind) -> u8 {
    match tier {
        TierKind::GpuVram => TIER_BIT_GPU,
        TierKind::HostDram => TIER_BIT_DRAM,
        TierKind::LocalNvme => TIER_BIT_NVME,
        _ => 0,
    }
}

#[inline]
fn mask_to_loc(mask: u8) -> u8 {
    if (mask & TIER_BIT_GPU) != 0 {
        1
    } else if (mask & TIER_BIT_DRAM) != 0 {
        2
    } else if (mask & TIER_BIT_NVME) != 0 {
        3
    } else {
        0
    }
}

#[inline]
fn key_shard(key: ObjectKey) -> usize {
    let h = key.0[0] ^ key.0[1].rotate_left(17) ^ key.0[2].rotate_right(13) ^ key.0[3];
    (h as usize) & (KEY_MIRROR_SHARDS - 1)
}

/// Lock-free (sharded RwLock) local-tier presence mirror + bounded evicted-key ring.
/// Allows `vericache_contains_batch_local[_flags]` and `vericache_take_evicted_keys`
/// to execute directly on the caller thread in `<50µs` without blocking on `worker_loop`.
struct LocalKeyMirror {
    shards: Box<[RwLock<HashMap<ObjectKey, u8>>]>,
    evicted: Mutex<VecDeque<ObjectKey>>,
}

impl LocalKeyMirror {
    fn new() -> Self {
        let mut v = Vec::with_capacity(KEY_MIRROR_SHARDS);
        for _ in 0..KEY_MIRROR_SHARDS {
            v.push(RwLock::new(HashMap::new()));
        }
        Self {
            shards: v.into_boxed_slice(),
            evicted: Mutex::new(VecDeque::new()),
        }
    }

    fn insert(&self, tier: TierKind, key: ObjectKey) {
        let bit = tier_to_bit(tier);
        if bit == 0 {
            return;
        }
        let mut guard = self.shards[key_shard(key)]
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *guard.entry(key).or_insert(0) |= bit;
    }

    fn remove(&self, tier: TierKind, key: ObjectKey) {
        let bit = tier_to_bit(tier);
        if bit == 0 {
            return;
        }
        let became_empty = {
            let mut guard = self.shards[key_shard(key)]
                .write()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(mask) = guard.get_mut(&key) {
                let old = *mask;
                *mask &= !bit;
                if *mask == 0 {
                    guard.remove(&key);
                    old != 0
                } else {
                    false
                }
            } else {
                false
            }
        };
        if became_empty {
            let mut ev = self.evicted.lock().unwrap_or_else(|e| e.into_inner());
            if ev.len() >= MAX_EVICTED_KEYS {
                ev.pop_front();
            }
            ev.push_back(key);
        }
    }

    #[inline]
    fn lookup_loc(&self, key: ObjectKey) -> u8 {
        let guard = self.shards[key_shard(key)]
            .read()
            .unwrap_or_else(|e| e.into_inner());
        guard.get(&key).copied().map(mask_to_loc).unwrap_or(0)
    }

    fn contains_batch_i32(&self, keys: &[ObjectKey], out: &mut [i32]) {
        for (i, &k) in keys.iter().enumerate() {
            out[i] = self.lookup_loc(k) as i32;
        }
    }

    fn contains_batch_u8(&self, keys: &[ObjectKey], out: &mut [u8]) {
        for (i, &k) in keys.iter().enumerate() {
            out[i] = self.lookup_loc(k);
        }
    }

    /// Drain up to `max_keys` evicted keys (still absent from all local tiers) as raw
    /// 32-byte little-endian `<4Q` records into `out_buf`.
    fn take_evicted(&self, out_buf: &mut [u8], max_keys: usize) -> usize {
        let cap = max_keys.min(out_buf.len() / 32);
        if cap == 0 {
            return 0;
        }
        let mut written = 0usize;
        let mut seen: HashSet<ObjectKey> = HashSet::new();
        let mut batch: Vec<ObjectKey> = Vec::with_capacity(cap.min(1024));
        while written < cap {
            batch.clear();
            {
                let mut ev = self.evicted.lock().unwrap_or_else(|e| e.into_inner());
                let want = (cap - written).min(1024);
                for _ in 0..want {
                    if let Some(k) = ev.pop_front() {
                        batch.push(k);
                    } else {
                        break;
                    }
                }
            }
            if batch.is_empty() {
                break;
            }
            // Check shard presence AFTER releasing `evicted` lock (strict lock order).
            // Filters out keys that were demoted to another tier or re-inserted since eviction.
            for &k in &batch {
                if written >= cap {
                    break;
                }
                if !seen.insert(k) {
                    continue;
                }
                if self.lookup_loc(k) == 0 {
                    let base = written * 32;
                    for w in 0..4 {
                        let le = k.0[w].to_le_bytes();
                        out_buf[base + w * 8..base + (w + 1) * 8].copy_from_slice(&le);
                    }
                    written += 1;
                }
            }
        }
        written
    }
}

/// `EvictionPolicy` wrapper around `LruEviction` that keeps `LocalKeyMirror` synchronized
/// across every local insert, removal, and victim selection (strictly non-TCB).
struct MirroredLruEviction {
    inner: LruEviction,
    mirror: Arc<LocalKeyMirror>,
    /// Worker-local `(tier_idx, ObjectKey) -> latest FrameId` so removing/popping a stale
    /// frame never clears the mirror bit of a newer frame for the same `(tier, key)`.
    key_frames: HashMap<(u8, ObjectKey), u32>,
}

impl MirroredLruEviction {
    fn new(mirror: Arc<LocalKeyMirror>) -> Self {
        Self {
            inner: LruEviction::new(),
            mirror,
            key_frames: HashMap::new(),
        }
    }

    fn clear_if_current(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        let Some(tidx) = tier_to_idx(tier) else {
            return;
        };
        if self.key_frames.get(&(tidx, key)) == Some(&frame.0) {
            self.key_frames.remove(&(tidx, key));
            self.mirror.remove(tier, key);
        }
    }
}

impl EvictionPolicy for MirroredLruEviction {
    fn on_insert(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_insert(tier, key, frame);
        if let Some(tidx) = tier_to_idx(tier) {
            self.key_frames.insert((tidx, key), frame.0);
            self.mirror.insert(tier, key);
        }
    }

    fn on_insert_cold(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_insert_cold(tier, key, frame);
        if let Some(tidx) = tier_to_idx(tier) {
            self.key_frames.insert((tidx, key), frame.0);
            self.mirror.insert(tier, key);
        }
    }

    fn on_access(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_access(tier, key, frame);
    }

    fn on_remove(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.inner.on_remove(tier, key, frame);
        self.clear_if_current(tier, key, frame);
    }

    fn select_victim(&mut self, tier: TierKind) -> Option<(ObjectKey, FrameId)> {
        // Do NOT clear `LocalKeyMirror` here: `kick_async_spills` and `ensure_tier_space`
        // call `select_victim` BEFORE the frame finishes spilling/demoting to a colder tier.
        // Every eviction path in `VeriCache` calls `on_insert(colder_tier)` before
        // `on_remove(hotter_tier)`, so clearing strictly in `on_remove` guarantees zero
        // transient misses and zero premature eviction notifications during spills/demotions.
        self.inner.select_victim(tier)
    }
}

/// Lock-free atomic mirror of `CacheStats` so `vericache_stats` never blocks on `worker_loop`.
struct AtomicCacheStats {
    puts: AtomicU64,
    gpu_hits: AtomicU64,
    dram_hits: AtomicU64,
    nvme_hits: AtomicU64,
    remote_hits: AtomicU64,
    misses: AtomicU64,
    policy_victims_rejected: AtomicU64,
    policy_victims_pinned: AtomicU64,
}

impl AtomicCacheStats {
    fn new() -> Self {
        Self {
            puts: AtomicU64::new(0),
            gpu_hits: AtomicU64::new(0),
            dram_hits: AtomicU64::new(0),
            nvme_hits: AtomicU64::new(0),
            remote_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            policy_victims_rejected: AtomicU64::new(0),
            policy_victims_pinned: AtomicU64::new(0),
        }
    }

    fn sync_from(&self, s: &CacheStats) {
        self.puts.store(s.puts, Ordering::Relaxed);
        self.gpu_hits.store(s.gpu_hits, Ordering::Relaxed);
        self.dram_hits.store(s.dram_hits, Ordering::Relaxed);
        self.nvme_hits.store(s.nvme_hits, Ordering::Relaxed);
        self.remote_hits.store(s.remote_hits, Ordering::Relaxed);
        self.misses.store(s.misses, Ordering::Relaxed);
        self.policy_victims_rejected
            .store(s.policy_victims_rejected, Ordering::Relaxed);
        self.policy_victims_pinned
            .store(s.policy_victims_pinned, Ordering::Relaxed);
    }

    fn snapshot(&self) -> VeriCacheStatsC {
        VeriCacheStatsC {
            puts: self.puts.load(Ordering::Relaxed),
            gpu_hits: self.gpu_hits.load(Ordering::Relaxed),
            dram_hits: self.dram_hits.load(Ordering::Relaxed),
            nvme_hits: self.nvme_hits.load(Ordering::Relaxed),
            remote_hits: self.remote_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            policy_victims_rejected: self.policy_victims_rejected.load(Ordering::Relaxed),
            policy_victims_pinned: self.policy_victims_pinned.load(Ordering::Relaxed),
        }
    }
}

/// Opaque thread-safe handle owning the dual-priority queue to the dedicated `VeriCache` worker thread.
pub struct VeriCacheHandle {
    queue: Arc<WorkQueue>,
    key_mirror: Arc<LocalKeyMirror>,
    stats_mirror: Arc<AtomicCacheStats>,
    worker: Option<JoinHandle<()>>,
    frame_size: usize,
}

impl VeriCacheHandle {
    fn send_high(&self, cmd: HighCommand) -> Result<(), ()> {
        self.queue.send_high(cmd)
    }

    fn send_low(&self, cmd: LowCommand) -> Result<(), ()> {
        self.queue.send_low(cmd)
    }
}

fn connect_peer_tcp(spec: &OwnedPeerSpec) -> Result<PeerLink<TcpTransport>, CacheError> {
    let deadline = Instant::now() + Duration::from_secs(180);
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
    let transport = TcpTransport::with_timeout(stream, Duration::from_secs(180))
        .map_err(|e| CacheError::Init(format!("TcpTransport::with_timeout failed: {e}")))?;
    Ok(PeerLink {
        peer_node_id: spec.peer_node_id,
        transport,
        psn_hint: spec.psn_hint,
    })
}

fn flush_batch_with_event(
    batch_buf: &mut cuda_copy::BatchCopyBuf,
    sync_event: Option<&cuda_copy::BlockingCudaEvent>,
    cuda_stream: usize,
) -> Result<(), CacheError> {
    if batch_buf.is_empty() {
        return Ok(());
    }
    let submit_res = batch_buf.submit_async(cuda_stream);
    batch_buf.clear();
    match submit_res {
        Ok(false) => Ok(()),
        Ok(true) => {
            let sync_res = match sync_event {
                Some(ev) => ev
                    .record_and_sync(cuda_stream)
                    .or_else(|_| cuda_copy::stream_synchronize(cuda_stream)),
                None => cuda_copy::stream_synchronize(cuda_stream),
            };
            sync_res.map_err(|code| {
                CacheError::Init(format!("DMA blocking sync failed with code {code}"))
            })
        }
        Err((code, dirty)) => {
            if dirty {
                let _ = cuda_copy::stream_synchronize(cuda_stream);
            }
            Err(CacheError::Init(format!(
                "cudaMemcpyBatchAsync/cudaMemcpyAsync failed with code {code}"
            )))
        }
    }
}

fn emit_load_slot(
    slot: usize,
    load_batch_bufs: &[RefCell<cuda_copy::BatchCopyBuf>; 2],
    load_sync_events: &Option<[cuda_copy::BlockingCudaEvent; 2]>,
    slot_dirty: &RefCell<[bool; 2]>,
    cuda_stream: usize,
) -> Result<(), CacheError> {
    let s = slot & 1;
    let batch = load_batch_bufs[s].borrow();
    if batch.is_empty() {
        slot_dirty.borrow_mut()[s] = false;
        return Ok(());
    }
    match batch.submit_async(cuda_stream) {
        Ok(false) => {
            slot_dirty.borrow_mut()[s] = false;
            Ok(())
        }
        Ok(true) => {
            if let Some(evts) = load_sync_events.as_ref() {
                if let Err(code) = evts[s].record(cuda_stream) {
                    let _ = cuda_copy::stream_synchronize(cuda_stream);
                    slot_dirty.borrow_mut()[s] = false;
                    return Err(CacheError::Init(format!(
                        "cudaEventRecord(load slot {s}) failed with code {code}"
                    )));
                }
            }
            slot_dirty.borrow_mut()[s] = true;
            Ok(())
        }
        Err((code, dirty)) => {
            if dirty {
                let _ = cuda_copy::stream_synchronize(cuda_stream);
            }
            slot_dirty.borrow_mut()[s] = false;
            Err(CacheError::Init(format!(
                "cudaMemcpyBatchAsync/cudaMemcpyAsync failed with code {code}"
            )))
        }
    }
}

fn sync_load_slot(
    slot: usize,
    load_batch_bufs: &[RefCell<cuda_copy::BatchCopyBuf>; 2],
    load_sync_events: &Option<[cuda_copy::BlockingCudaEvent; 2]>,
    slot_dirty: &RefCell<[bool; 2]>,
    cuda_stream: usize,
) -> Result<(), CacheError> {
    let s = slot & 1;
    let was_dirty = {
        let mut d = slot_dirty.borrow_mut();
        let v = d[s];
        d[s] = false;
        v
    };
    load_batch_bufs[s].borrow_mut().clear();
    if !was_dirty {
        return Ok(());
    }
    let sync_res = match load_sync_events.as_ref() {
        Some(evts) => evts[s]
            .synchronize()
            .or_else(|_| cuda_copy::stream_synchronize(cuda_stream)),
        None => cuda_copy::stream_synchronize(cuda_stream),
    };
    sync_res.map_err(|code| {
        CacheError::Init(format!(
            "DMA blocking sync(load slot {s}) failed with code {code}"
        ))
    })
}

fn ensure_load_sync_events(load_sync_events: &mut Option<[cuda_copy::BlockingCudaEvent; 2]>) {
    if load_sync_events.is_none() {
        if let (Ok(e0), Ok(e1)) = (
            cuda_copy::BlockingCudaEvent::new(),
            cuda_copy::BlockingCudaEvent::new(),
        ) {
            *load_sync_events = Some([e0, e1]);
        }
    }
}

fn exec_high_cmd<E, P, R>(
    cache: &mut VeriCache<'_, TcpTransport, E, P, R>,
    cmd: HighCommand,
    load_sync_events: &mut Option<[cuda_copy::BlockingCudaEvent; 2]>,
    load_batch_bufs: &[RefCell<cuda_copy::BatchCopyBuf>; 2],
) where
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    match cmd {
        HighCommand::GetHost { key, len, reply } => {
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
        HighCommand::GetGpuPtr {
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
        HighCommand::GetGpuBatch {
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
                ensure_load_sync_events(load_sync_events);
                load_batch_bufs[0].borrow_mut().clear();
                load_batch_bufs[1].borrow_mut().clear();
                let slot_dirty = RefCell::new([false, false]);
                let batch_res = cache.with_gpu_read_batch_pipelined(
                    &keys,
                    |slot, entries| {
                        let s = slot & 1;
                        {
                            let mut batch = load_batch_bufs[s].borrow_mut();
                            batch.clear();
                            for &(gi, src) in entries {
                                let dst_ptr = dst_dev_ptrs[gi];
                                let (src_addr, kind) = match src {
                                    BatchReadSource::GpuDevice(addr) => {
                                        (addr, cuda_copy::CUDA_MEMCPY_DEVICE_TO_DEVICE)
                                    }
                                    BatchReadSource::PinnedHost(addr) => {
                                        (addr, cuda_copy::CUDA_MEMCPY_HOST_TO_DEVICE)
                                    }
                                };
                                if let Err(code) = batch.push(dst_ptr, src_addr, len, kind) {
                                    batch.clear();
                                    return Err(CacheError::Init(format!(
                                        "invalid DMA copy descriptor (code {code})"
                                    )));
                                }
                            }
                        }
                        emit_load_slot(
                            s,
                            load_batch_bufs,
                            load_sync_events,
                            &slot_dirty,
                            cuda_stream,
                        )
                    },
                    |slot| {
                        sync_load_slot(
                            slot,
                            load_batch_bufs,
                            load_sync_events,
                            &slot_dirty,
                            cuda_stream,
                        )
                    },
                );
                if slot_dirty.borrow()[0] || slot_dirty.borrow()[1] {
                    let _ = cuda_copy::stream_synchronize(cuda_stream);
                }
                load_batch_bufs[0].borrow_mut().clear();
                load_batch_bufs[1].borrow_mut().clear();
                match batch_res {
                    Ok(read_hits) => {
                        let out = read_hits
                            .into_iter()
                            .map(|opt| match opt {
                                Some(hit) => VeriCacheHitInfoC {
                                    found: 1,
                                    location: loc_to_i32(hit.location),
                                    peer_node_id: loc_peer_node(hit.location),
                                    version_origin: hit.version.origin_node,
                                    version_seq: hit.version.seq,
                                },
                                None => VeriCacheHitInfoC::default(),
                            })
                            .collect();
                        Ok(out)
                    }
                    Err(e) => Err(e.to_string()),
                }
            };
            let _ = reply.send(res);
        }
        HighCommand::GetGpuPagedBatch {
            keys,
            gpu_block_ids,
            regions,
            cuda_stream,
            reply,
        } => {
            let frame_size = cache.frame_size();
            let res = if let Err(msg) = regions.validate_bounds(frame_size) {
                Err(msg)
            } else {
                ensure_load_sync_events(load_sync_events);
                let num_regions = regions.len();
                load_batch_bufs[0].borrow_mut().clear();
                load_batch_bufs[1].borrow_mut().clear();
                let slot_dirty = RefCell::new([false, false]);
                let batch_res = cache.with_gpu_read_batch_pipelined(
                    &keys,
                    |slot, entries| {
                        let s = slot & 1;
                        {
                            let mut batch = load_batch_bufs[s].borrow_mut();
                            batch.clear();
                            for &(gi, src) in entries {
                                let (src_base, kind) = match src {
                                    BatchReadSource::GpuDevice(addr) => {
                                        (addr, cuda_copy::CUDA_MEMCPY_DEVICE_TO_DEVICE)
                                    }
                                    BatchReadSource::PinnedHost(addr) => {
                                        (addr, cuda_copy::CUDA_MEMCPY_HOST_TO_DEVICE)
                                    }
                                };
                                let block_id = gpu_block_ids[gi];
                                for r in 0..num_regions {
                                    let sz = regions.sizes[r];
                                    if sz == 0 {
                                        continue;
                                    }
                                    let dst_off = match block_id.checked_mul(regions.block_strides[r]) {
                                        Some(v) => v,
                                        None => {
                                            batch.clear();
                                            return Err(CacheError::Init(
                                                "gpu_block_id * block_stride overflow".into(),
                                            ));
                                        }
                                    };
                                    let dst_addr = match regions.base_ptrs[r].checked_add(dst_off) {
                                        Some(v) => v,
                                        None => {
                                            batch.clear();
                                            return Err(CacheError::Init(
                                                "paged region dst address overflow".into(),
                                            ));
                                        }
                                    };
                                    let src_addr = match src_base.checked_add(regions.offsets[r]) {
                                        Some(v) => v,
                                        None => {
                                            batch.clear();
                                            return Err(CacheError::Init(
                                                "frame src address overflow".into(),
                                            ));
                                        }
                                    };
                                    if let Err(code) = batch.push(dst_addr, src_addr, sz, kind) {
                                        batch.clear();
                                        return Err(CacheError::Init(format!(
                                            "invalid paged DMA descriptor (code {code})"
                                        )));
                                    }
                                }
                            }
                        }
                        emit_load_slot(
                            s,
                            load_batch_bufs,
                            load_sync_events,
                            &slot_dirty,
                            cuda_stream,
                        )
                    },
                    |slot| {
                        sync_load_slot(
                            slot,
                            load_batch_bufs,
                            load_sync_events,
                            &slot_dirty,
                            cuda_stream,
                        )
                    },
                );
                if slot_dirty.borrow()[0] || slot_dirty.borrow()[1] {
                    let _ = cuda_copy::stream_synchronize(cuda_stream);
                }
                load_batch_bufs[0].borrow_mut().clear();
                load_batch_bufs[1].borrow_mut().clear();
                match batch_res {
                    Ok(read_hits) => {
                        let out = read_hits
                            .into_iter()
                            .map(|opt| match opt {
                                Some(hit) => VeriCacheHitInfoC {
                                    found: 1,
                                    location: loc_to_i32(hit.location),
                                    peer_node_id: loc_peer_node(hit.location),
                                    version_origin: hit.version.origin_node,
                                    version_seq: hit.version.seq,
                                },
                                None => VeriCacheHitInfoC::default(),
                            })
                            .collect();
                        Ok(out)
                    }
                    Err(e) => Err(e.to_string()),
                }
            };
            let _ = reply.send(res);
        }
        #[cfg(test)]
        HighCommand::Noop => {}
    }
}

fn drain_high_pri<E, P, R>(
    cache: &mut VeriCache<'_, TcpTransport, E, P, R>,
    queue: &WorkQueue,
    stats_mirror: &AtomicCacheStats,
    load_sync_events: &mut Option<[cuda_copy::BlockingCudaEvent; 2]>,
    load_batch_bufs: &[RefCell<cuda_copy::BatchCopyBuf>; 2],
) where
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    let mut ran = false;
    while let Some(cmd) = queue.try_pop_high() {
        exec_high_cmd(cache, cmd, load_sync_events, load_batch_bufs);
        ran = true;
    }
    if ran {
        stats_mirror.sync_from(cache.stats());
    }
}

fn worker_loop(
    cfg: OwnedConfig,
    queue: Arc<WorkQueue>,
    key_mirror: Arc<LocalKeyMirror>,
    stats_mirror: Arc<AtomicCacheStats>,
    ready_tx: Sender<Result<(), String>>,
) {
    let _close_guard = CloseQueueOnDrop(Arc::clone(&queue));
    // Ensure the worker thread's active CUDA device matches `gpu_device_id` even when
    // GPU HBM tiering is disabled (`hbm_frames == 0`), so BlockingCudaEvent and
    // PinnedHostBuf match the caller's `cuda_stream` device (e.g. GPU 4).
    if cfg.gpu_device_id >= 0 && cfg.storage.gpu.is_none() {
        if let Err(code) = cuda_copy::set_device(cfg.gpu_device_id) {
            let _ = ready_tx.send(Err(format!(
                "cudaSetDevice({}) failed with code {code}",
                cfg.gpu_device_id
            )));
            return;
        }
    }
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
    let has_nvme = cfg.storage.nvme.is_some();
    let dram_frames = cfg.storage.dram_frames as usize;
    let tiering = if has_nvme && dram_frames >= 64 {
        let high_water = (dram_frames / 16).clamp(16, 4096);
        let max_inflight = (high_water / 4).clamp(4, 64);
        CascadeTiering::serving_dram_first(high_water, max_inflight)
    } else {
        CascadeTiering::serving_dram_first(0, 0)
    };
    let policy = PolicySet {
        eviction: MirroredLruEviction::new(Arc::clone(&key_mirror)),
        tiering,
        routing: ProbeAllPeers,
    };
    let session_res = VeriCache::with_session(
        cfg.storage,
        peer_links,
        policy,
        |cache| {
            let mut gpu_io = if cache.engine().gpu_tier().is_some() {
                let stream = cache.create_gpu_stream()?;
                let slot = cache.alloc_gpu_slot()?;
                Some((stream, slot))
            } else {
                None
            };

            // Strictly separated SAVE vs LOAD DMA resources so high-priority LOADs
            // preempting mid-SAVE never alias events, streams, or scratch buffers.
            let mut pinned_d2h = [
                cuda_copy::PinnedHostBuf::new(),
                cuda_copy::PinnedHostBuf::new(),
            ];
            let mut store_sync_events: Option<[cuda_copy::BlockingCudaEvent; 2]> = None;
            let store_batch_buf = RefCell::new(cuda_copy::BatchCopyBuf::new());

            let mut load_sync_events: Option<[cuda_copy::BlockingCudaEvent; 2]> = None;
            let load_batch_bufs = [
                RefCell::new(cuda_copy::BatchCopyBuf::new()),
                RefCell::new(cuda_copy::BatchCopyBuf::new()),
            ];

            if let Some(tx) = ready_tx.take() {
                let _ = tx.send(Ok(()));
            }

            let idle_poll = Duration::from_micros(100);
            loop {
                match queue.recv_timeout(idle_poll) {
                    NextWork::High(cmd) => {
                        exec_high_cmd(cache, cmd, &mut load_sync_events, &load_batch_bufs);
                        stats_mirror.sync_from(cache.stats());
                    }
                    NextWork::Low(cmd) => {
                        let mut shutdown = false;
                        match cmd {
                        LowCommand::PutHost { key, data, reply } => {
                            let res = cache
                                .put_host(key, &data)
                                .map(|v| (v.origin_node, v.seq))
                                .map_err(|e| e.to_string());
                            let _ = reply.send(res);
                        }
                        LowCommand::PutGpuPtr {
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
                        LowCommand::PutGpuBatch {
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
                                        let total_keys = keys.len();
                                        let _ = cache.poll();
                                        for (idx, (key, src_ptr)) in
                                            keys.into_iter().zip(src_dev_ptrs.into_iter()).enumerate()
                                        {
                                            if idx > 0 && idx % 4 == 0 {
                                                let _ = cache.poll();
                                                if queue.has_high_pri() {
                                                    drain_high_pri(
                                                        cache,
                                                        &queue,
                                                        &stats_mirror,
                                                        &mut load_sync_events,
                                                        &load_batch_bufs,
                                                    );
                                                }
                                            }
                                            if cache.contains_local(key).is_some() {
                                                count += 1;
                                                continue;
                                            }
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
                                                eprintln!(
                                                    "vericache-python: PutGpuBatch partial stop at {idx}/{total_keys}: {e}"
                                                );
                                                err = Some(e.to_string());
                                                break;
                                            }
                                            count += 1;
                                        }
                                        let _ = cache.poll();
                                        match err {
                                            Some(_) if count > 0 => Ok(count),
                                            Some(e) => Err(e),
                                            None => Ok(count),
                                        }
                                    }
                                }
                            };
                            let _ = reply.send(res);
                        }
                        LowCommand::PutGpuToDramBatch {
                            keys,
                            src_dev_ptrs,
                            len,
                            cuda_stream,
                            reply,
                        } => {
                            const CHUNK: usize = 32;
                            let frame_size = cache.frame_size();
                            let res = if len != frame_size {
                                Err(format!(
                                    "length mismatch: got {len}, expected {frame_size}"
                                ))
                            } else {
                                if store_sync_events.is_none() {
                                    if let (Ok(e0), Ok(e1)) = (
                                        cuda_copy::BlockingCudaEvent::new(),
                                        cuda_copy::BlockingCudaEvent::new(),
                                    ) {
                                        store_sync_events = Some([e0, e1]);
                                    }
                                }
                                let total_keys = keys.len();
                                let mut count = 0usize;
                                let mut err: Option<String> = None;
                                let _ = cache.poll();
                                if queue.has_high_pri() {
                                    drain_high_pri(
                                        cache,
                                        &queue,
                                        &stats_mirror,
                                        &mut load_sync_events,
                                        &load_batch_bufs,
                                    );
                                }

                                if cache.is_host_pinned() {
                                    // Zero-copy 1-hop D2H DMA directly into VeriCache's CUDA-pinned
                                    // HostBackend slab in 16-block chunks (preemptible in <1ms).
                                    let mut cs = 0usize;
                                    while cs < total_keys {
                                        let ce = (cs + CHUNK).min(total_keys);
                                        store_batch_buf.borrow_mut().clear();
                                        let chunk_res = cache.with_host_write_batch(
                                            &keys[cs..ce],
                                            |li, dst_host_addr| {
                                                let orig_i = cs + li;
                                                let push_res = store_batch_buf.borrow_mut().push(
                                                    dst_host_addr,
                                                    src_dev_ptrs[orig_i],
                                                    len,
                                                    cuda_copy::CUDA_MEMCPY_DEVICE_TO_HOST,
                                                );
                                                if let Err(code) = push_res {
                                                    store_batch_buf.borrow_mut().clear();
                                                    return Err(CacheError::Init(format!(
                                                        "invalid D2H DMA descriptor (code {code})"
                                                    )));
                                                }
                                                Ok(())
                                            },
                                            || {
                                                flush_batch_with_event(
                                                    &mut store_batch_buf.borrow_mut(),
                                                    store_sync_events.as_ref().map(|ev| &ev[0]),
                                                    cuda_stream,
                                                )
                                            },
                                        );
                                        store_batch_buf.borrow_mut().clear();
                                        match chunk_res {
                                            Ok(n) => {
                                                count += n;
                                                if n < ce - cs {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                    "vericache-python: PutGpuToDramBatch partial stop at {count}/{total_keys}: {e}"
                                                );
                                                err = Some(e.to_string());
                                                break;
                                            }
                                        }
                                        cs = ce;
                                        if queue.has_high_pri() {
                                            drain_high_pri(
                                                cache,
                                                &queue,
                                                &stats_mirror,
                                                &mut load_sync_events,
                                                &load_batch_bufs,
                                            );
                                        }
                                    }
                                } else {
                                    struct PreparedChunk {
                                        start: usize,
                                        end: usize,
                                        unc: Vec<(usize, ObjectKey)>,
                                        buf_idx: usize,
                                    }

                                    let mut stream_dirty = false;

                                    let launch_chunk = |cs: usize,
                                                        ce: usize,
                                                        buf_idx: usize,
                                                        cache_ref: &mut VeriCache<'_, _, _, _, _>,
                                                        pinned: &mut [cuda_copy::PinnedHostBuf; 2],
                                                        evts: &Option<[cuda_copy::BlockingCudaEvent; 2]>,
                                                        batch_buf: &RefCell<cuda_copy::BatchCopyBuf>,
                                                        dirty: &mut bool|
                                     -> Result<PreparedChunk, String> {
                                        let mut unc = Vec::with_capacity(ce - cs);
                                        for i in cs..ce {
                                            let key = keys[i];
                                            if cache_ref.contains_local(key).is_none() {
                                                unc.push((i, key));
                                            }
                                        }
                                        if !unc.is_empty() {
                                            pinned[buf_idx]
                                                .ensure_capacity(unc.len() * len)
                                                .map_err(|code| {
                                                    format!(
                                                        "cudaHostAlloc({} bytes) failed with code {code}",
                                                        unc.len() * len
                                                    )
                                                })?;
                                            let host_base = pinned[buf_idx].addr();
                                            let mut batch = batch_buf.borrow_mut();
                                            batch.clear();
                                            for (j, &(orig_i, _key)) in unc.iter().enumerate() {
                                                batch
                                                    .push(
                                                        host_base + j * len,
                                                        src_dev_ptrs[orig_i],
                                                        len,
                                                        cuda_copy::CUDA_MEMCPY_DEVICE_TO_HOST,
                                                    )
                                                    .map_err(|code| {
                                                        format!(
                                                            "invalid D2H DMA descriptor (code {code})"
                                                        )
                                                    })?;
                                            }
                                            let submit_res = batch.submit_async(cuda_stream);
                                            batch.clear();
                                            match submit_res {
                                                Ok(submitted) => {
                                                    if submitted {
                                                        *dirty = true;
                                                    }
                                                }
                                                Err((code, submitted)) => {
                                                    if submitted {
                                                        *dirty = true;
                                                    }
                                                    return Err(format!(
                                                        "cudaMemcpyBatchAsync/cudaMemcpyAsync(D2H) failed with code {code}"
                                                    ));
                                                }
                                            }
                                            if let Some(ev) = evts.as_ref() {
                                                ev[buf_idx].record(cuda_stream).map_err(|code| {
                                                    format!(
                                                        "cudaEventRecord failed with code {code}"
                                                    )
                                                })?;
                                            }
                                        }
                                        Ok(PreparedChunk {
                                            start: cs,
                                            end: ce,
                                            unc,
                                            buf_idx,
                                        })
                                    };

                                    let mut next_start = 0usize;
                                    let mut cur_opt = if next_start < total_keys {
                                        let ce = (next_start + CHUNK).min(total_keys);
                                        next_start = ce;
                                        match launch_chunk(
                                            0,
                                            ce,
                                            0,
                                            cache,
                                            &mut pinned_d2h,
                                            &store_sync_events,
                                            &store_batch_buf,
                                            &mut stream_dirty,
                                        ) {
                                            Ok(pc) => Some(pc),
                                            Err(e) => {
                                                err = Some(e);
                                                None
                                            }
                                        }
                                    } else {
                                        None
                                    };

                                    while let Some(cur) = cur_opt.take() {
                                        if !cur.unc.is_empty() {
                                            let sync_res = match store_sync_events.as_ref() {
                                                Some(ev) => ev[cur.buf_idx].synchronize().or_else(
                                                    |_| cuda_copy::stream_synchronize(cuda_stream),
                                                ),
                                                None => cuda_copy::stream_synchronize(cuda_stream),
                                            };
                                            if let Err(code) = sync_res {
                                                err = Some(format!(
                                                    "D2H sync failed with code {code}"
                                                ));
                                                break;
                                            }
                                        }

                                        // Kick off next chunk's D2H on the alternate pinned buffer so
                                        // PCIe DMA overlaps with `cache.put_host` below.
                                        if next_start < total_keys {
                                            let cs = next_start;
                                            let ce = (cs + CHUNK).min(total_keys);
                                            next_start = ce;
                                            let next_buf = 1 - cur.buf_idx;
                                            match launch_chunk(
                                                cs,
                                                ce,
                                                next_buf,
                                                cache,
                                                &mut pinned_d2h,
                                                &store_sync_events,
                                                &store_batch_buf,
                                                &mut stream_dirty,
                                            ) {
                                                Ok(pc) => {
                                                    cur_opt = Some(pc);
                                                }
                                                Err(e) => {
                                                    err = Some(e);
                                                    break;
                                                }
                                            }
                                        }

                                        let mut chunk_failed = false;
                                        if !cur.unc.is_empty() {
                                            let host_bytes = pinned_d2h[cur.buf_idx]
                                                .as_slice(cur.unc.len() * len);
                                            for (j, &(orig_i, key)) in cur.unc.iter().enumerate() {
                                                if j > 0 && j % 8 == 0 {
                                                    let _ = cache.poll();
                                                }
                                                if cache.contains_local(key).is_some() {
                                                    continue;
                                                }
                                                let slice = &host_bytes[j * len..(j + 1) * len];
                                                if let Err(e) = cache.put_host(key, slice) {
                                                    eprintln!(
                                                        "vericache-python: PutGpuToDramBatch partial stop at {orig_i}/{total_keys}: {e}"
                                                    );
                                                    err = Some(e.to_string());
                                                    count = orig_i;
                                                    chunk_failed = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if !chunk_failed {
                                            let mut verified_end = cur.end;
                                            for idx in cur.start..cur.end {
                                                if cache.contains_local(keys[idx]).is_none() {
                                                    verified_end = idx;
                                                    err = Some(format!(
                                                        "prescanned key at {idx}/{total_keys} evicted before commit"
                                                    ));
                                                    break;
                                                }
                                            }
                                            count = verified_end;
                                        }
                                        let _ = cache.poll();
                                        if err.is_some() {
                                            break;
                                        }
                                        if queue.has_high_pri() {
                                            drain_high_pri(
                                                cache,
                                                &queue,
                                                &stats_mirror,
                                                &mut load_sync_events,
                                                &load_batch_bufs,
                                            );
                                        }
                                    }
                                    if err.is_some() && stream_dirty {
                                        let _ = cuda_copy::stream_synchronize(cuda_stream);
                                    }
                                }
                                match err {
                                    Some(_) if count > 0 => Ok(count),
                                    Some(e) => Err(e),
                                    None => Ok(count),
                                }
                            };
                            let _ = reply.send(res);
                        }
                        LowCommand::PutGpuPagedDramBatch {
                            keys,
                            gpu_block_ids,
                            regions,
                            cuda_stream,
                            reply,
                        } => {
                            const CHUNK: usize = 32;
                            let frame_size = cache.frame_size();
                            let res = match regions.validate_and_zero_gaps(frame_size) {
                                Err(msg) => Err(msg),
                                Ok(zero_gaps) => {
                                    if store_sync_events.is_none() {
                                        if let (Ok(e0), Ok(e1)) = (
                                            cuda_copy::BlockingCudaEvent::new(),
                                            cuda_copy::BlockingCudaEvent::new(),
                                        ) {
                                            store_sync_events = Some([e0, e1]);
                                        }
                                    }
                                    let total_keys = keys.len();
                                    let num_regions = regions.len();
                                    let mut count = 0usize;
                                    let mut err: Option<String> = None;
                                    let _ = cache.poll();
                                    if queue.has_high_pri() {
                                        drain_high_pri(
                                            cache,
                                            &queue,
                                            &stats_mirror,
                                            &mut load_sync_events,
                                            &load_batch_bufs,
                                        );
                                    }

                                    if cache.is_host_pinned() {
                                        // Zero-copy 1-hop paged D2H DMA directly from vLLM's KV
                                        // layer tensors into VeriCache's CUDA-pinned HostBackend
                                        // slab in 16-block chunks (preemptible in <1ms).
                                        let mut cs = 0usize;
                                        while cs < total_keys {
                                            let ce = (cs + CHUNK).min(total_keys);
                                            store_batch_buf.borrow_mut().clear();
                                            let chunk_res = cache.with_host_write_batch(
                                                &keys[cs..ce],
                                                |li, dst_host_addr| {
                                                    let orig_i = cs + li;
                                                    let block_id = gpu_block_ids[orig_i];
                                                    // Zero-pad uncovered gap/tail bytes BEFORE
                                                    // D2H DMA is submitted in `on_flush`.
                                                    if !zero_gaps.is_empty() {
                                                        let dst_slice = unsafe {
                                                            core::slice::from_raw_parts_mut(
                                                                dst_host_addr as *mut u8,
                                                                frame_size,
                                                            )
                                                        };
                                                        for &(gs, ge) in &zero_gaps {
                                                            dst_slice[gs..ge].fill(0);
                                                        }
                                                    }
                                                    let mut batch = store_batch_buf.borrow_mut();
                                                    for r in 0..num_regions {
                                                        let sz = regions.sizes[r];
                                                        if sz == 0 {
                                                            continue;
                                                        }
                                                        let src_off = match block_id
                                                            .checked_mul(regions.block_strides[r])
                                                        {
                                                            Some(v) => v,
                                                            None => {
                                                                batch.clear();
                                                                return Err(CacheError::Init(
                                                                    "gpu_block_id * block_stride overflow".into(),
                                                                ));
                                                            }
                                                        };
                                                        let src_addr = match regions.base_ptrs[r]
                                                            .checked_add(src_off)
                                                        {
                                                            Some(v) => v,
                                                            None => {
                                                                batch.clear();
                                                                return Err(CacheError::Init(
                                                                    "paged region src address overflow".into(),
                                                                ));
                                                            }
                                                        };
                                                        let dst_addr =
                                                            dst_host_addr + regions.offsets[r];
                                                        if let Err(code) = batch.push(
                                                            dst_addr,
                                                            src_addr,
                                                            sz,
                                                            cuda_copy::CUDA_MEMCPY_DEVICE_TO_HOST,
                                                        ) {
                                                            batch.clear();
                                                            return Err(CacheError::Init(format!(
                                                                "invalid paged D2H DMA descriptor (code {code})"
                                                            )));
                                                        }
                                                    }
                                                    Ok(())
                                                },
                                                || {
                                                    flush_batch_with_event(
                                                        &mut store_batch_buf.borrow_mut(),
                                                        store_sync_events.as_ref().map(|ev| &ev[0]),
                                                        cuda_stream,
                                                    )
                                                },
                                            );
                                            store_batch_buf.borrow_mut().clear();
                                            match chunk_res {
                                                Ok(n) => {
                                                    count += n;
                                                    if n < ce - cs {
                                                        break;
                                                    }
                                                }
                                                Err(e) => {
                                                    eprintln!(
                                                        "vericache-python: PutGpuPagedDramBatch partial stop at {count}/{total_keys}: {e}"
                                                    );
                                                    err = Some(e.to_string());
                                                    break;
                                                }
                                            }
                                            cs = ce;
                                            if queue.has_high_pri() {
                                                drain_high_pri(
                                                    cache,
                                                    &queue,
                                                    &stats_mirror,
                                                    &mut load_sync_events,
                                                    &load_batch_bufs,
                                                );
                                            }
                                        }
                                    } else {
                                        struct PreparedChunk {
                                            start: usize,
                                            end: usize,
                                            unc: Vec<(usize, ObjectKey)>,
                                            buf_idx: usize,
                                        }

                                        let mut stream_dirty = false;

                                        let launch_chunk = |cs: usize,
                                                            ce: usize,
                                                            buf_idx: usize,
                                                            cache_ref: &mut VeriCache<'_, _, _, _, _>,
                                                            pinned: &mut [cuda_copy::PinnedHostBuf; 2],
                                                            evts: &Option<
                                            [cuda_copy::BlockingCudaEvent; 2],
                                        >,
                                                            batch_buf: &RefCell<
                                            cuda_copy::BatchCopyBuf,
                                        >,
                                                            dirty: &mut bool|
                                         -> Result<PreparedChunk, String> {
                                            let mut unc = Vec::with_capacity(ce - cs);
                                            for i in cs..ce {
                                                let key = keys[i];
                                                if cache_ref.contains_local(key).is_none() {
                                                    unc.push((i, key));
                                                }
                                            }
                                            if !unc.is_empty() {
                                                pinned[buf_idx]
                                                    .ensure_capacity(unc.len() * frame_size)
                                                    .map_err(|code| {
                                                        format!(
                                                            "cudaHostAlloc({} bytes) failed with code {code}",
                                                            unc.len() * frame_size
                                                        )
                                                    })?;
                                                let host_base = pinned[buf_idx].addr();
                                                let mut batch = batch_buf.borrow_mut();
                                                batch.clear();
                                                for (j, &(orig_i, _key)) in unc.iter().enumerate() {
                                                    let block_id = gpu_block_ids[orig_i];
                                                    let dst_block_base = host_base + j * frame_size;
                                                    for r in 0..num_regions {
                                                        let sz = regions.sizes[r];
                                                        if sz == 0 {
                                                            continue;
                                                        }
                                                        let src_off = block_id
                                                            .checked_mul(regions.block_strides[r])
                                                            .ok_or_else(|| {
                                                                "gpu_block_id * block_stride overflow"
                                                                    .to_string()
                                                            })?;
                                                        let src_addr = regions.base_ptrs[r]
                                                            .checked_add(src_off)
                                                            .ok_or_else(|| {
                                                                "paged region src address overflow"
                                                                    .to_string()
                                                            })?;
                                                        let dst_addr =
                                                            dst_block_base + regions.offsets[r];
                                                        batch
                                                            .push(
                                                                dst_addr,
                                                                src_addr,
                                                                sz,
                                                                cuda_copy::CUDA_MEMCPY_DEVICE_TO_HOST,
                                                            )
                                                            .map_err(|code| {
                                                                format!(
                                                                    "invalid paged D2H DMA descriptor (code {code})"
                                                                )
                                                            })?;
                                                    }
                                                }
                                                let submit_res = batch.submit_async(cuda_stream);
                                                batch.clear();
                                                match submit_res {
                                                    Ok(submitted) => {
                                                        if submitted {
                                                            *dirty = true;
                                                        }
                                                    }
                                                    Err((code, submitted)) => {
                                                        if submitted {
                                                            *dirty = true;
                                                        }
                                                        return Err(format!(
                                                            "paged D2H DMA submit failed with code {code}"
                                                        ));
                                                    }
                                                }
                                                if let Some(ev) = evts.as_ref() {
                                                    ev[buf_idx].record(cuda_stream).map_err(
                                                        |code| {
                                                            format!(
                                                                "cudaEventRecord failed with code {code}"
                                                            )
                                                        },
                                                    )?;
                                                }
                                            }
                                            Ok(PreparedChunk {
                                                start: cs,
                                                end: ce,
                                                unc,
                                                buf_idx,
                                            })
                                        };

                                        let mut next_start = 0usize;
                                        let mut cur_opt = if next_start < total_keys {
                                            let ce = (next_start + CHUNK).min(total_keys);
                                            next_start = ce;
                                            match launch_chunk(
                                                0,
                                                ce,
                                                0,
                                                cache,
                                                &mut pinned_d2h,
                                                &store_sync_events,
                                                &store_batch_buf,
                                                &mut stream_dirty,
                                            ) {
                                                Ok(pc) => Some(pc),
                                                Err(e) => {
                                                    err = Some(e);
                                                    None
                                                }
                                            }
                                        } else {
                                            None
                                        };

                                        while let Some(cur) = cur_opt.take() {
                                            if !cur.unc.is_empty() {
                                                let sync_res = match store_sync_events.as_ref() {
                                                    Some(ev) => ev[cur.buf_idx]
                                                        .synchronize()
                                                        .or_else(|_| {
                                                            cuda_copy::stream_synchronize(
                                                                cuda_stream,
                                                            )
                                                        }),
                                                    None => {
                                                        cuda_copy::stream_synchronize(cuda_stream)
                                                    }
                                                };
                                                if let Err(code) = sync_res {
                                                    err = Some(format!(
                                                        "paged D2H sync failed with code {code}"
                                                    ));
                                                    break;
                                                }
                                            }

                                            // Kick off next chunk's D2H on the alternate pinned buffer
                                            // so PCIe DMA overlaps with zero-pad + `cache.put_host`.
                                            if next_start < total_keys {
                                                let cs = next_start;
                                                let ce = (cs + CHUNK).min(total_keys);
                                                next_start = ce;
                                                let next_buf = 1 - cur.buf_idx;
                                                match launch_chunk(
                                                    cs,
                                                    ce,
                                                    next_buf,
                                                    cache,
                                                    &mut pinned_d2h,
                                                    &store_sync_events,
                                                    &store_batch_buf,
                                                    &mut stream_dirty,
                                                ) {
                                                    Ok(pc) => {
                                                        cur_opt = Some(pc);
                                                    }
                                                    Err(e) => {
                                                        err = Some(e);
                                                        break;
                                                    }
                                                }
                                            }

                                            // Zero-pad uncovered gap/tail bytes in `pinned_d2h[cur.buf_idx]`
                                            // after D2H sync completed on this buffer.
                                            if !cur.unc.is_empty() && !zero_gaps.is_empty() {
                                                let buf = pinned_d2h[cur.buf_idx]
                                                    .as_mut_slice(cur.unc.len() * frame_size);
                                                for j in 0..cur.unc.len() {
                                                    let base = j * frame_size;
                                                    for &(gs, ge) in &zero_gaps {
                                                        buf[base + gs..base + ge].fill(0);
                                                    }
                                                }
                                            }

                                            let mut chunk_failed = false;
                                            if !cur.unc.is_empty() {
                                                let host_bytes = pinned_d2h[cur.buf_idx]
                                                    .as_slice(cur.unc.len() * frame_size);
                                                for (j, &(orig_i, key)) in
                                                    cur.unc.iter().enumerate()
                                                {
                                                    if j > 0 && j % 8 == 0 {
                                                        let _ = cache.poll();
                                                    }
                                                    if cache.contains_local(key).is_some() {
                                                        continue;
                                                    }
                                                    let slice = &host_bytes
                                                        [j * frame_size..(j + 1) * frame_size];
                                                    if let Err(e) = cache.put_host(key, slice) {
                                                        eprintln!(
                                                            "vericache-python: PutGpuPagedDramBatch partial stop at {orig_i}/{total_keys}: {e}"
                                                        );
                                                        err = Some(e.to_string());
                                                        count = orig_i;
                                                        chunk_failed = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            if !chunk_failed {
                                                let mut verified_end = cur.end;
                                                for idx in cur.start..cur.end {
                                                    if cache.contains_local(keys[idx]).is_none() {
                                                        verified_end = idx;
                                                        err = Some(format!(
                                                            "prescanned key at {idx}/{total_keys} evicted before commit"
                                                        ));
                                                        break;
                                                    }
                                                }
                                                count = verified_end;
                                            }
                                            let _ = cache.poll();
                                            if err.is_some() {
                                                break;
                                            }
                                            if queue.has_high_pri() {
                                                drain_high_pri(
                                                    cache,
                                                    &queue,
                                                    &stats_mirror,
                                                    &mut load_sync_events,
                                                    &load_batch_bufs,
                                                );
                                            }
                                        }
                                        if err.is_some() && stream_dirty {
                                            let _ = cuda_copy::stream_synchronize(cuda_stream);
                                        }
                                    }
                                    match err {
                                        Some(_) if count > 0 => Ok(count),
                                        Some(e) => Err(e),
                                        None => Ok(count),
                                    }
                                }
                            };
                            let _ = reply.send(res);
                        }
                        LowCommand::SpillToNvme { key, reply } => {
                            let _ = reply
                                .send(cache.spill_key_to_nvme(key).map_err(|e| e.to_string()));
                        }
                        LowCommand::EvictTier { key, tier, reply } => {
                            let _ = reply
                                .send(cache.evict_from_tier(key, tier).map_err(|e| e.to_string()));
                        }
                        LowCommand::Poll { reply } => {
                            let _ = reply.send(cache.poll().map(|_| ()).map_err(|e| e.to_string()));
                        }
                        LowCommand::Shutdown { reply } => {
                            let _ = cache.poll();
                            let _ = reply.send(Ok(()));
                            shutdown = true;
                        }
                    }
                    stats_mirror.sync_from(cache.stats());
                    if shutdown {
                        break;
                    }
                    }
                    NextWork::Timeout => {
                        let _ = cache.poll();
                        stats_mirror.sync_from(cache.stats());
                    }
                    NextWork::Disconnected => {
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

unsafe fn read_keys(keys4: *const u64, count: usize) -> Vec<ObjectKey> {
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
    keys
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
    let mut storage = StorageConfig::new(c.node_id, c.frame_size, c.dram_frames);
    storage.max_leases = 256;
    if c.gpu_device_id >= 0 && c.hbm_frames > 0 {
        storage = storage.with_gpu(GpuConfig::new(c.gpu_device_id as u32, c.hbm_frames));
    }
    match unsafe { cstr_to_string(c.nvme_dir) } {
        Ok(Some(dir)) if c.nvme_frames > 0 => {
            let mut ncfg = NvmeConfig::new(PathBuf::from(dir), c.nvme_frames);
            ncfg.queue_depth = 128;
            storage = storage.with_nvme(ncfg);
        }
        Ok(_) => {}
        Err(e) => {
            set_last_error(e);
            return core::ptr::null_mut();
        }
    }

    let mut peers = Vec::with_capacity(c.num_peers);
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

    let frame_size = c.frame_size;
    let owned_cfg = OwnedConfig {
        storage,
        peers,
        gpu_device_id: c.gpu_device_id,
    };
    let queue = Arc::new(WorkQueue::new());
    let key_mirror = Arc::new(LocalKeyMirror::new());
    let stats_mirror = Arc::new(AtomicCacheStats::new());
    let worker_queue = Arc::clone(&queue);
    let worker_key_mirror = Arc::clone(&key_mirror);
    let worker_stats_mirror = Arc::clone(&stats_mirror);
    let (ready_tx, ready_rx) = mpsc::channel();

    let worker = match thread::Builder::new()
        .name(format!("vericache-node-{}", c.node_id))
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            worker_loop(
                owned_cfg,
                worker_queue,
                worker_key_mirror,
                worker_stats_mirror,
                ready_tx,
            )
        })
    {
        Ok(h) => h,
        Err(e) => {
            set_last_error(format!("failed to spawn worker thread: {e}"));
            return core::ptr::null_mut();
        }
    };

    match ready_rx.recv() {
        Ok(Ok(())) => Box::into_raw(Box::new(VeriCacheHandle {
            queue,
            key_mirror,
            stats_mirror,
            worker: Some(worker),
            frame_size,
        })),
        Ok(Err(msg)) => {
            queue.close();
            let _ = worker.join();
            set_last_error(msg);
            core::ptr::null_mut()
        }
        Err(_) => {
            queue.close();
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
    if boxed.send_low(LowCommand::Shutdown { reply: rtx }).is_ok() {
        let _ = rrx.recv();
    }
    boxed.queue.close();
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
        set_last_error(format!(
            "length mismatch: got {len}, expected {}",
            h.frame_size
        ));
        return -2;
    }
    let buf = unsafe { core::slice::from_raw_parts(data, len) }.to_vec();
    let (rtx, rrx) = mpsc::channel();
    if h.send_low(LowCommand::PutHost {
        key,
        data: buf,
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
    if h.send_high(HighCommand::GetHost {
        key,
        len,
        reply: rtx,
    })
    .is_err()
    {
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
    if h.send_low(LowCommand::PutGpuPtr {
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
    if h.send_high(HighCommand::GetGpuPtr {
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
    let keys = unsafe { read_keys(keys4, count) };
    let ptrs = unsafe { core::slice::from_raw_parts(src_dev_ptrs, count) }.to_vec();
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.send_low(LowCommand::PutGpuBatch {
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
pub unsafe extern "C" fn vericache_put_gpu_dram_batch(
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
    let keys = unsafe { read_keys(keys4, count) };
    let ptrs = unsafe { core::slice::from_raw_parts(src_dev_ptrs, count) }.to_vec();
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.send_low(LowCommand::PutGpuToDramBatch {
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
pub unsafe extern "C" fn vericache_put_gpu_paged_dram_batch(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    gpu_block_ids: *const usize,
    count: usize,
    region_base_ptrs: *const usize,
    region_block_strides: *const usize,
    region_offsets: *const usize,
    region_sizes: *const usize,
    num_regions: usize,
    cuda_stream: usize,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    if count == 0 {
        return 0;
    }
    if num_regions == 0 {
        set_last_error("num_regions must be > 0");
        return -2;
    }
    if keys4.is_null()
        || gpu_block_ids.is_null()
        || region_base_ptrs.is_null()
        || region_block_strides.is_null()
        || region_offsets.is_null()
        || region_sizes.is_null()
    {
        set_last_error("null pointer argument to vericache_put_gpu_paged_dram_batch");
        return -1;
    }
    let keys = unsafe { read_keys(keys4, count) };
    let block_ids = unsafe { core::slice::from_raw_parts(gpu_block_ids, count) }.to_vec();
    let regions = PagedRegions {
        base_ptrs: unsafe { core::slice::from_raw_parts(region_base_ptrs, num_regions) }.to_vec(),
        block_strides: unsafe { core::slice::from_raw_parts(region_block_strides, num_regions) }
            .to_vec(),
        offsets: unsafe { core::slice::from_raw_parts(region_offsets, num_regions) }.to_vec(),
        sizes: unsafe { core::slice::from_raw_parts(region_sizes, num_regions) }.to_vec(),
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.send_low(LowCommand::PutGpuPagedDramBatch {
        keys,
        gpu_block_ids: block_ids,
        regions,
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
    let keys = unsafe { read_keys(keys4, count) };
    let ptrs = unsafe { core::slice::from_raw_parts(dst_dev_ptrs, count) }.to_vec();
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.send_high(HighCommand::GetGpuBatch {
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
pub unsafe extern "C" fn vericache_get_gpu_paged_batch(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    gpu_block_ids: *const usize,
    count: usize,
    region_base_ptrs: *const usize,
    region_block_strides: *const usize,
    region_offsets: *const usize,
    region_sizes: *const usize,
    num_regions: usize,
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
    if num_regions == 0 {
        set_last_error("num_regions must be > 0");
        return -2;
    }
    if keys4.is_null()
        || gpu_block_ids.is_null()
        || region_base_ptrs.is_null()
        || region_block_strides.is_null()
        || region_offsets.is_null()
        || region_sizes.is_null()
    {
        set_last_error("null pointer argument to vericache_get_gpu_paged_batch");
        return -1;
    }
    let keys = unsafe { read_keys(keys4, count) };
    let block_ids = unsafe { core::slice::from_raw_parts(gpu_block_ids, count) }.to_vec();
    let regions = PagedRegions {
        base_ptrs: unsafe { core::slice::from_raw_parts(region_base_ptrs, num_regions) }.to_vec(),
        block_strides: unsafe { core::slice::from_raw_parts(region_block_strides, num_regions) }
            .to_vec(),
        offsets: unsafe { core::slice::from_raw_parts(region_offsets, num_regions) }.to_vec(),
        sizes: unsafe { core::slice::from_raw_parts(region_sizes, num_regions) }.to_vec(),
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.send_high(HighCommand::GetGpuPagedBatch {
        keys,
        gpu_block_ids: block_ids,
        regions,
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
pub unsafe extern "C" fn vericache_get_gpu_paged_batch_flags(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    gpu_block_ids: *const usize,
    count: usize,
    region_base_ptrs: *const usize,
    region_block_strides: *const usize,
    region_offsets: *const usize,
    region_sizes: *const usize,
    num_regions: usize,
    cuda_stream: usize,
    out_flags: *mut u8,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    if count == 0 {
        return 0;
    }
    if num_regions == 0 {
        set_last_error("num_regions must be > 0");
        return -2;
    }
    if keys4.is_null()
        || gpu_block_ids.is_null()
        || region_base_ptrs.is_null()
        || region_block_strides.is_null()
        || region_offsets.is_null()
        || region_sizes.is_null()
    {
        set_last_error("null pointer argument to vericache_get_gpu_paged_batch_flags");
        return -1;
    }
    let keys = unsafe { read_keys(keys4, count) };
    let block_ids = unsafe { core::slice::from_raw_parts(gpu_block_ids, count) }.to_vec();
    let regions = PagedRegions {
        base_ptrs: unsafe { core::slice::from_raw_parts(region_base_ptrs, num_regions) }.to_vec(),
        block_strides: unsafe { core::slice::from_raw_parts(region_block_strides, num_regions) }
            .to_vec(),
        offsets: unsafe { core::slice::from_raw_parts(region_offsets, num_regions) }.to_vec(),
        sizes: unsafe { core::slice::from_raw_parts(region_sizes, num_regions) }.to_vec(),
    };
    let h = unsafe { &*handle };
    let (rtx, rrx) = mpsc::channel();
    if h.send_high(HighCommand::GetGpuPagedBatch {
        keys,
        gpu_block_ids: block_ids,
        regions,
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
            if !out_flags.is_null() {
                let dst = unsafe { core::slice::from_raw_parts_mut(out_flags, count) };
                for (i, hinfo) in hits.into_iter().enumerate() {
                    if hinfo.found != 0 {
                        found_count += 1;
                        dst[i] = hinfo.location as u8;
                    } else {
                        dst[i] = 0;
                    }
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
    let h = unsafe { &*handle };
    let out = unsafe { core::slice::from_raw_parts_mut(out_locs, count) };
    if !h.queue.is_alive() {
        out.fill(0);
        return 0;
    }
    let keys = unsafe { read_keys(keys4, count) };
    h.key_mirror.contains_batch_i32(&keys, out);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_contains_batch_local_flags(
    handle: *mut VeriCacheHandle,
    keys4: *const u64,
    count: usize,
    out_flags: *mut u8,
) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return -1;
    }
    if count == 0 {
        return 0;
    }
    if keys4.is_null() || out_flags.is_null() {
        set_last_error("null keys4 or out_flags");
        return -1;
    }
    let h = unsafe { &*handle };
    let out = unsafe { core::slice::from_raw_parts_mut(out_flags, count) };
    if !h.queue.is_alive() {
        out.fill(0);
        return 0;
    }
    let keys = unsafe { read_keys(keys4, count) };
    h.key_mirror.contains_batch_u8(&keys, out);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_take_evicted_keys(
    handle: *mut VeriCacheHandle,
    out_keys_buf: *mut u8,
    max_keys: usize,
) -> usize {
    if handle.is_null() || out_keys_buf.is_null() || max_keys == 0 {
        return 0;
    }
    let Some(byte_len) = max_keys.checked_mul(32) else {
        return 0;
    };
    let h = unsafe { &*handle };
    let out = unsafe { core::slice::from_raw_parts_mut(out_keys_buf, byte_len) };
    h.key_mirror.take_evicted(out, max_keys)
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
    if h.send_low(LowCommand::SpillToNvme { key, reply: rtx })
        .is_err()
    {
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
    if h.send_low(LowCommand::EvictTier {
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
    if h.send_low(LowCommand::Poll { reply: rtx }).is_err() {
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
    unsafe { *out_stats = h.stats_mirror.snapshot() };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_regions_zero_gaps_exact_and_tail() {
        let exact = PagedRegions {
            base_ptrs: vec![0x1000, 0x2000],
            block_strides: vec![4096, 4096],
            offsets: vec![0, 2048],
            sizes: vec![2048, 2048],
        };
        assert_eq!(exact.validate_and_zero_gaps(4096).unwrap(), vec![]);

        let with_tail = PagedRegions {
            base_ptrs: vec![0x1000, 0x2000],
            block_strides: vec![2048, 2048],
            offsets: vec![0, 1024],
            sizes: vec![1024, 1500],
        };
        assert_eq!(
            with_tail.validate_and_zero_gaps(4096).unwrap(),
            vec![(2524, 4096)]
        );

        let unordered_with_gap = PagedRegions {
            base_ptrs: vec![0x1000, 0x2000],
            block_strides: vec![1024, 1024],
            offsets: vec![2048, 512],
            sizes: vec![1024, 512],
        };
        assert_eq!(
            unordered_with_gap.validate_and_zero_gaps(4096).unwrap(),
            vec![(0, 512), (1024, 2048), (3072, 4096)]
        );
    }

    #[test]
    fn paged_regions_rejects_invalid_and_overlapping() {
        let overlap = PagedRegions {
            base_ptrs: vec![0x1000, 0x2000],
            block_strides: vec![4096, 4096],
            offsets: vec![0, 1024],
            sizes: vec![2048, 1024],
        };
        assert!(overlap.validate_and_zero_gaps(4096).is_err());

        let oob = PagedRegions {
            base_ptrs: vec![0x1000],
            block_strides: vec![4096],
            offsets: vec![3000],
            sizes: vec![2000],
        };
        assert!(oob.validate_bounds(4096).is_err());

        let null_base = PagedRegions {
            base_ptrs: vec![0],
            block_strides: vec![4096],
            offsets: vec![0],
            sizes: vec![128],
        };
        assert!(null_base.validate_bounds(4096).is_err());
    }

    #[test]
    fn work_queue_prioritizes_high_over_low_and_wakes() {
        let q = Arc::new(WorkQueue::new());
        let (ltx, _lrx) = mpsc::channel();
        q.send_low(LowCommand::Poll { reply: ltx }).unwrap();
        assert!(!q.has_high_pri());
        q.send_high(HighCommand::Noop).unwrap();
        assert!(q.has_high_pri());

        assert!(matches!(
            q.recv_timeout(Duration::from_millis(10)),
            NextWork::High(HighCommand::Noop)
        ));
        assert!(!q.has_high_pri());
        assert!(matches!(
            q.recv_timeout(Duration::from_millis(10)),
            NextWork::Low(LowCommand::Poll { .. })
        ));

        // Verify immediate wakeup across threads
        let q2 = Arc::clone(&q);
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            q2.send_high(HighCommand::Noop).unwrap();
        });
        let t0 = Instant::now();
        assert!(matches!(
            q.recv_timeout(Duration::from_secs(2)),
            NextWork::High(HighCommand::Noop)
        ));
        assert!(t0.elapsed() < Duration::from_millis(500));
        t.join().unwrap();

        assert!(q.is_alive());
        q.close();
        assert!(!q.is_alive());
        assert!(q.send_high(HighCommand::Noop).is_err());
        assert!(matches!(
            q.recv_timeout(Duration::from_millis(10)),
            NextWork::Disconnected
        ));
    }

    #[test]
    fn dead_worker_contains_batch_returns_all_miss() {
        let queue = Arc::new(WorkQueue::new());
        let mirror = Arc::new(LocalKeyMirror::new());
        let k = ObjectKey([7, 8, 9, 10]);
        mirror.insert(TierKind::HostDram, k);

        let mut handle = VeriCacheHandle {
            queue: Arc::clone(&queue),
            key_mirror: Arc::clone(&mirror),
            stats_mirror: Arc::new(AtomicCacheStats::new()),
            worker: None,
            frame_size: 4096,
        };
        let raw_k = k.0;
        let mut locs = [0i32; 1];
        let mut flags = [0u8; 1];

        // Alive worker -> returns LocalDram (2)
        unsafe {
            assert_eq!(
                vericache_contains_batch_local(&mut handle, raw_k.as_ptr(), 1, locs.as_mut_ptr()),
                0
            );
            assert_eq!(
                vericache_contains_batch_local_flags(
                    &mut handle,
                    raw_k.as_ptr(),
                    1,
                    flags.as_mut_ptr(),
                ),
                0
            );
        }
        assert_eq!(locs, [2]);
        assert_eq!(flags, [2]);

        // Enqueue pending high_pri + low_pri commands BEFORE simulated panic; both reply
        // receivers MUST wake immediately with Err when CloseQueueOnDrop runs!
        let (hrtx, hrrx) = mpsc::channel();
        queue
            .send_high(HighCommand::GetHost {
                key: k,
                len: 4096,
                reply: hrtx,
            })
            .unwrap();
        let (lrtx, lrrx) = mpsc::channel();
        queue.send_low(LowCommand::Poll { reply: lrtx }).unwrap();

        // Simulate worker panic / unwind via CloseQueueOnDrop
        let q_drop = Arc::clone(&queue);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = CloseQueueOnDrop(q_drop);
            panic!("simulated worker panic");
        }));
        assert!(!queue.is_alive());
        assert!(!queue.has_high_pri());
        assert!(hrrx.recv_timeout(Duration::from_millis(100)).is_err());
        assert!(lrrx.recv_timeout(Duration::from_millis(100)).is_err());

        // Dead worker -> immediately returns all 0 (MISS)
        unsafe {
            assert_eq!(
                vericache_contains_batch_local(&mut handle, raw_k.as_ptr(), 1, locs.as_mut_ptr()),
                0
            );
            assert_eq!(
                vericache_contains_batch_local_flags(
                    &mut handle,
                    raw_k.as_ptr(),
                    1,
                    flags.as_mut_ptr(),
                ),
                0
            );
        }
        assert_eq!(locs, [0]);
        assert_eq!(flags, [0]);
    }

    #[test]
    fn mirrored_lru_tracks_tiers_evictions_and_filters_demoted_keys() {
        let mirror = Arc::new(LocalKeyMirror::new());
        let mut policy = MirroredLruEviction::new(Arc::clone(&mirror));
        let k1 = ObjectKey([1, 2, 3, 4]);
        let k2 = ObjectKey([10, 20, 30, 40]);
        let k3 = ObjectKey([100, 200, 300, 400]);

        // Insert k1 on DRAM (FrameId 1), then GPU (FrameId 10) -> GPU takes priority
        policy.on_insert(TierKind::HostDram, k1, FrameId(1));
        assert_eq!(mirror.lookup_loc(k1), 2);
        policy.on_insert(TierKind::GpuVram, k1, FrameId(10));
        assert_eq!(mirror.lookup_loc(k1), 1);

        // Overwrite k1 on DRAM with FrameId(11); removing stale FrameId(1) must NOT clear DRAM bit
        policy.on_insert(TierKind::HostDram, k1, FrameId(11));
        policy.on_remove(TierKind::HostDram, k1, FrameId(1));
        assert_eq!(mirror.lookup_loc(k1), 1);

        // Remove GPU frame -> falls back to DRAM, NOT evicted
        policy.on_remove(TierKind::GpuVram, k1, FrameId(10));
        assert_eq!(mirror.lookup_loc(k1), 2);
        let mut buf = [0u8; 128];
        assert_eq!(mirror.take_evicted(&mut buf, 4), 0);

        // Re-insert k1 on DRAM (FrameId 11) and insert k2 (FrameId 2) to test select_victim + demotion
        policy.on_insert(TierKind::HostDram, k1, FrameId(11));
        policy.on_insert(TierKind::HostDram, k2, FrameId(2));
        let v = policy.select_victim(TierKind::HostDram);
        assert_eq!(v, Some((k1, FrameId(11))));
        // While async spill / sync demotion is in flight, k1 MUST remain visible in DRAM and NOT evicted!
        assert_eq!(mirror.lookup_loc(k1), 2);
        assert_eq!(mirror.take_evicted(&mut buf, 4), 0);

        // Spill completes: insert into NVMe, then remove from DRAM -> k1 stays live on NVMe, NOT evicted!
        policy.on_insert(TierKind::LocalNvme, k1, FrameId(50));
        policy.on_remove(TierKind::HostDram, k1, FrameId(11));
        assert_eq!(mirror.lookup_loc(k1), 3);
        assert_eq!(mirror.take_evicted(&mut buf, 4), 0);

        // True eviction of k2 (DRAM only -> empty)
        policy.on_remove(TierKind::HostDram, k2, FrameId(2));
        assert_eq!(mirror.lookup_loc(k2), 0);
        // Batch lookup check
        let mut flags = [0u8; 3];
        mirror.contains_batch_u8(&[k1, k2, k3], &mut flags);
        assert_eq!(flags, [3, 0, 0]);

        let n = mirror.take_evicted(&mut buf, 4);
        assert_eq!(n, 1);
        let got = ObjectKey([
            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        ]);
        assert_eq!(got, k2);
        // Drained once -> empty now
        assert_eq!(mirror.take_evicted(&mut buf, 4), 0);

        // Final eviction of k1 from NVMe -> now k1 becomes evicted!
        policy.on_remove(TierKind::LocalNvme, k1, FrameId(50));
        assert_eq!(mirror.lookup_loc(k1), 0);
        let n2 = mirror.take_evicted(&mut buf, 4);
        assert_eq!(n2, 1);
        let got1 = ObjectKey([
            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        ]);
        assert_eq!(got1, k1);
    }

    #[test]
    fn atomic_stats_mirror_round_trip() {
        let m = AtomicCacheStats::new();
        let s = CacheStats {
            puts: 7,
            gpu_hits: 1,
            dram_hits: 2,
            nvme_hits: 3,
            remote_hits: 4,
            misses: 5,
            policy_victims_rejected: 6,
            policy_victims_pinned: 8,
        };
        m.sync_from(&s);
        let snap = m.snapshot();
        assert_eq!(snap.puts, 7);
        assert_eq!(snap.gpu_hits, 1);
        assert_eq!(snap.dram_hits, 2);
        assert_eq!(snap.nvme_hits, 3);
        assert_eq!(snap.remote_hits, 4);
        assert_eq!(snap.misses, 5);
        assert_eq!(snap.policy_victims_rejected, 6);
        assert_eq!(snap.policy_victims_pinned, 8);
    }
}
