use std::sync::{Mutex, OnceLock};
use transfer::context::{
    AllocSource, EventSourceAllocator, EventSourceKind, EventSourceToken, NodeContext,
    ShardAllocator, ShardSources, ShareSource, TransferInstance, VersionInstance, VersionSource,
};
use vericache_core::kernel::CacheCore;
use vericache_core::spec::ObjectKey;
use vstd::prelude::{Tracked, verus};

/// Default per-shard version sequence budget (`2^40` versions per shard, allowing over
/// 16 million independent share-nothing shards per OS process without exhausting `u64::MAX`).
pub const DEFAULT_SHARD_VERSION_QUOTA: u64 = 1u64 << 40;

static BENCH_NODE_ALLOCATOR: OnceLock<Mutex<(u64, ShardAllocator)>> = OnceLock::new();

/// Initialize the process-wide `ShardAllocator` for `node_id` at most once per OS process.
pub fn ensure_bench_node_allocator(node_id: u64) {
    let lock = BENCH_NODE_ALLOCATOR.get_or_init(|| {
        let ctx = NodeContext::try_bootstrap(node_id)
            .expect("NodeContext::try_bootstrap once per OS process");
        Mutex::new((node_id, ShardAllocator::from_node_context(ctx)))
    });
    let guard = lock.lock().expect("bench shard allocator mutex");
    assert_eq!(
        guard.0, node_id,
        "process already bootstrapped for node_id {}, cannot switch to {}",
        guard.0, node_id
    );
}

/// Carve an independent share-nothing `ShardSources` bundle with `version_quota` local version IDs.
pub fn take_bench_shard_with_quota(node_id: u64, version_quota: u64) -> ShardSources {
    let lock = BENCH_NODE_ALLOCATOR.get_or_init(|| {
        let ctx = NodeContext::try_bootstrap(node_id)
            .expect("NodeContext::try_bootstrap once per OS process");
        Mutex::new((node_id, ShardAllocator::from_node_context(ctx)))
    });
    let mut guard = lock.lock().expect("bench shard allocator mutex");
    assert_eq!(
        guard.0, node_id,
        "process already bootstrapped for node_id {}, cannot request {}",
        guard.0, node_id
    );
    guard
        .1
        .split_shard(version_quota)
        .expect("sufficient bench version quota remaining in ShardAllocator")
}

verus! {

/// Per-shard linear capability bundle backed by `ShardAllocator::split_shard` with zero forged capabilities.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct ShardCapabilityBundle {
    pub node: u64,
    pub instance: Tracked<TransferInstance>,
    pub alloc: Tracked<AllocSource>,
    pub events: Tracked<EventSourceAllocator>,
    pub shares: Tracked<ShareSource>,
    pub version_instance: Tracked<VersionInstance>,
    pub versions: Option<Tracked<VersionSource>>,
    pub version_start: u64,
    pub version_limit: u64,
}

impl ShardCapabilityBundle {
    pub fn from_sources(shard: ShardSources) -> Self {
        Self {
            node: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            version_instance: shard.version_instance,
            versions: Some(shard.versions),
            version_start: shard.version_start,
            version_limit: shard.version_limit,
        }
    }

    pub fn take(node_id: u64) -> Self {
        Self::from_sources(take_bench_shard_with_quota(node_id, DEFAULT_SHARD_VERSION_QUOTA))
    }

    pub fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::host()))
    }

    pub fn mint_cuda(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::cuda_stream()))
    }

    pub fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::io_uring()))
    }

    pub fn mint_rdma(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::rdma_qp()))
    }

    /// Construct a shard-isolated `CacheCore` over `[version_start, version_limit)` for local engines.
    pub fn make_local_core(&self, frame_size: usize, max_leases: usize) -> CacheCore {
        CacheCore::new_with_version_range(
            self.node as u16,
            frame_size,
            Vec::new(),
            max_leases,
            self.version_start,
            self.version_limit,
            Tracked(self.instance.borrow()),
        )
    }

    /// Construct a shard-isolated `CacheCore` with an attached bounded `VersionSource` for RDMA engines.
    pub fn make_versioned_core(&mut self, frame_size: usize, max_leases: usize) -> CacheCore {
        let ver_src = self.versions.take().expect("shard VersionSource already consumed");
        CacheCore::from_shard_versions(
            self.node as u16,
            frame_size,
            Vec::new(),
            max_leases,
            self.version_start,
            self.version_limit,
            Tracked(self.instance.borrow()),
            Tracked(self.version_instance.borrow()),
            ver_src,
        )
    }
}

} // verus!

/// Deterministically map a 64-bit trace `hash_id` to a 256-bit VeriCache `ObjectKey`.
#[inline]
pub fn key_from_hash_id(hash_id: u64) -> ObjectKey {
    let h1 = splitmix64(hash_id ^ 0x9e37_79b9_7f4a_7c15);
    let h2 = splitmix64(h1 ^ 0xbf58_476d_1ce4_e5b9);
    let h3 = splitmix64(h2 ^ 0x94d0_49bb_1331_11eb);
    ObjectKey([hash_id, h1, h2, h3])
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Key-hash multi-engine shard router driving independent `LocalEngine` or
/// `DistributedEngine` shards with disjoint key spaces.
pub struct ShardRouter<T> {
    shards: Vec<T>,
}

impl<T> ShardRouter<T> {
    pub fn new(shards: Vec<T>) -> Self {
        assert!(!shards.is_empty(), "ShardRouter requires at least 1 shard");
        Self { shards }
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    #[inline]
    pub fn shard_index(&self, key: &ObjectKey) -> usize {
        let mixed = splitmix64(key.0[0] ^ key.0[1]);
        (mixed as usize) % self.shards.len()
    }

    #[inline]
    pub fn shard_for(&self, key: &ObjectKey) -> &T {
        let idx = self.shard_index(key);
        &self.shards[idx]
    }

    #[inline]
    pub fn shard_for_mut(&mut self, key: &ObjectKey) -> &mut T {
        let idx = self.shard_index(key);
        &mut self.shards[idx]
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.shards.iter_mut()
    }
}

/// Deterministic payload checksum header + trailer for Suite 4 integrity verification.
pub const FRAME_MAGIC: u64 = 0x5643_4143_4845_3034; // "VCACHE04"

/// Stamp a frame buffer of length `>= 32` with `(magic, hash_id, version_seq)` header,
/// deterministic body bytes, and trailing 8-byte FNV-1a checksum.
pub fn stamp_checksummed_frame(buf: &mut [u8], hash_id: u64, version_seq: u64) {
    assert!(buf.len() >= 32, "checksummed frame must be >= 32 bytes");
    buf[0..8].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    buf[8..16].copy_from_slice(&hash_id.to_le_bytes());
    buf[16..24].copy_from_slice(&version_seq.to_le_bytes());
    let body_end = buf.len() - 8;
    // Fast 8-byte deterministic fill so large frames stamp at memory bandwidth.
    let mut state = splitmix64(hash_id ^ version_seq.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut off = 24;
    while off + 8 <= body_end {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        buf[off..off + 8].copy_from_slice(&state.to_le_bytes());
        off += 8;
    }
    while off < body_end {
        buf[off] = (state & 0xff) as u8;
        state >>= 8;
        off += 1;
    }
    let csum = frame_checksum(&buf[..body_end]);
    buf[body_end..].copy_from_slice(&csum.to_le_bytes());
}

/// Verify magic, expected `hash_id` (if `Some`), and full-payload checksum.
/// Returns `Ok(version_seq)` on intact frame, `Err(reason)` on corruption/tearing.
pub fn verify_checksummed_frame(
    buf: &[u8],
    expected_hash_id: Option<u64>,
) -> Result<u64, &'static str> {
    if buf.len() < 32 {
        return Err("short_frame");
    }
    let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    if magic != FRAME_MAGIC {
        return Err("bad_magic");
    }
    let hash_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    if let Some(exp) = expected_hash_id {
        if hash_id != exp {
            return Err("hash_id_mismatch");
        }
    }
    let version_seq = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    let body_end = buf.len() - 8;
    let stored = u64::from_le_bytes(buf[body_end..].try_into().unwrap());
    let actual = frame_checksum(&buf[..body_end]);
    if stored != actual {
        return Err("checksum_mismatch");
    }
    Ok(version_seq)
}

#[inline]
pub fn frame_checksum(data: &[u8]) -> u64 {
    // Fast word-granularity FNV-1a-64 mixing over every 8-byte word.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut chunks = data.chunks_exact(8);
    for c in chunks.by_ref() {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h ^= w;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &b in chunks.remainder() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
