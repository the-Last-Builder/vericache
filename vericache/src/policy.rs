//! Composable, unverified placement/eviction/routing policy traits.
//!
//! VeriCache separates **unverified policy** (heuristic hints on what to evict,
//! where to promote/demote, and which peers to probe) from **verified enforcement**
//! (`vericache-core::CacheCore` and `DistributedEngine`, which guarantee memory
//! safety, lease/RDMA MW fencing, and `VersionBinding` value integrity even if a
//! policy suggests an invalid, stale, or actively-leased `FrameId`).

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use vericache_core::spec::{FrameId, ObjectKey, TierKind};

/// Consumer or producer memory domain at the API boundary (not a storage tier).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataDomain {
    Host,
    Gpu,
}

/// Observable location where a `get` / `lease` satisfied a request.
///
/// This metadata is purely diagnostic/statistical and does not alter data
/// correctness semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitLocation {
    LocalGpu,
    LocalDram,
    LocalNvme,
    RemotePeer { node_id: u64, gdr: bool },
}

/// Read-only descriptor of a connected remote peer passed to [`PeerRoutingPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerMeta {
    pub index: usize,
    pub node_id: u64,
    pub supports_gdr: bool,
    pub supports_host_rdma: bool,
}

/// Async DRAM -> NVMe spill watermark parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillWatermark {
    /// Start proactive async NVMe spills when free DRAM frames drop to or below this count.
    pub high_water_free_frames: usize,
    /// Maximum number of concurrent in-flight `io_uring` NVMe spill operations.
    pub max_in_flight: usize,
}

// ============================================================================
// 1. EvictionPolicy: capacity victim selection (unverified hint)
// ============================================================================

/// Selects victim frames when a storage tier needs space.
///
/// Implementations are **unverified** and need not prove safety: if `select_victim`
/// returns a stale, out-of-bounds, or actively-leased `FrameId`, the verified
/// `DistributedEngine` enforcement layer safely drains/fences or rejects it.
pub trait EvictionPolicy {
    fn on_insert(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId);
    /// Insert a frame at the cold (LRU) tail so transient/cold-warmed entries are
    /// evicted first without displacing hot entries. Defaults to [`Self::on_insert`]
    /// for backwards compatibility with external policy wrappers.
    fn on_insert_cold(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.on_insert(tier, key, frame);
    }
    fn on_access(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId);
    fn on_remove(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId);
    /// Track a temporary frame for eviction without advertising a resident replica.
    fn on_insert_transient(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.on_insert(tier, key, frame);
    }
    fn on_remove_transient(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        self.on_remove(tier, key, frame);
    }
    fn select_victim(&mut self, tier: TierKind) -> Option<(ObjectKey, FrameId)>;
}

// ============================================================================
// 2. TieringPolicy: cross-tier placement, promotion, demotion & spill hints
// ============================================================================

/// Controls cross-tier movement (promotion on cold hit, demotion on eviction,
/// and proactive async DRAM -> NVMe spilling).
pub trait TieringPolicy {
    /// Keep a local replica after a remote fetch has been consumed.
    fn retain_remote_fetches(&self) -> bool {
        false
    }

    /// When `get` hits a cold/remote location, optionally promote into a hotter local tier.
    fn promote_on_hit(&self, from: HitLocation, consumer: DataDomain) -> Option<TierKind>;

    /// When a frame is evicted from `from`, optionally cascade-demote it to a colder local tier.
    fn demote_on_evict(&self, from: TierKind, _key: ObjectKey) -> Option<TierKind>;

    /// Optional proactive async DRAM -> NVMe spill watermark.
    fn dram_spill_watermark(&self) -> Option<SpillWatermark> {
        None
    }
}

// ============================================================================
// 3. PeerRoutingPolicy: remote peer probe order and RDMA/GDR preference
// ============================================================================

/// Controls which remote peers are probed on a local miss and in what order.
pub trait PeerRoutingPolicy {
    /// Append candidate peer indices (into `peers`) in probe order into `out`.
    fn probe_order(&self, key: ObjectKey, peers: &[PeerMeta], out: &mut Vec<usize>);

    /// Whether to prefer GPUDirect RDMA (`PullTarget::GpuGdr`) over Host RDMA
    /// for a given consumer domain and peer capability.
    fn prefer_gdr(&self, consumer: DataDomain, peer: &PeerMeta) -> bool {
        consumer == DataDomain::Gpu && peer.supports_gdr
    }

    fn on_remote_hit(&mut self, _key: ObjectKey, _peer_idx: usize) {}
    fn on_remote_miss(&mut self, _key: ObjectKey, _peer_idx: usize) {}
}

// ============================================================================
// Composable Policy Bundle
// ============================================================================

/// Composable bundle combining the three orthogonal policy traits.
pub struct PolicySet<E, P, R> {
    pub eviction: E,
    pub tiering: P,
    pub routing: R,
}

pub type DefaultPolicy = PolicySet<LruEviction, CascadeTiering, ProbeAllPeers>;

impl Default for DefaultPolicy {
    fn default() -> Self {
        Self {
            eviction: LruEviction::new(),
            tiering: CascadeTiering::default(),
            routing: ProbeAllPeers,
        }
    }
}

impl<E, P, R> PolicySet<E, P, R> {
    pub fn new(eviction: E, tiering: P, routing: R) -> Self {
        Self {
            eviction,
            tiering,
            routing,
        }
    }

    pub fn with_eviction<E2>(self, eviction: E2) -> PolicySet<E2, P, R> {
        PolicySet {
            eviction,
            tiering: self.tiering,
            routing: self.routing,
        }
    }

    pub fn with_tiering<P2>(self, tiering: P2) -> PolicySet<E, P2, R> {
        PolicySet {
            eviction: self.eviction,
            tiering,
            routing: self.routing,
        }
    }

    pub fn with_routing<R2>(self, routing: R2) -> PolicySet<E, P, R2> {
        PolicySet {
            eviction: self.eviction,
            tiering: self.tiering,
            routing,
        }
    }
}

// ============================================================================
// Built-in Eviction Policies
// ============================================================================

struct LruNode<K, V> {
    key: K,
    val: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// O(1) intrusive doubly-linked-list LRU tracker for a single tier.
pub struct LruList<K, V> {
    nodes: Vec<Option<LruNode<K, V>>>,
    free: Vec<usize>,
    map: HashMap<K, usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl<K: Eq + Hash + Copy, V: Copy> Default for LruList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Copy, V: Copy> LruList<K, V> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            map: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn insert(&mut self, key: K, val: V) {
        if let Some(&idx) = self.map.get(&key) {
            if let Some(node) = self.nodes[idx].as_mut() {
                node.val = val;
            }
            self.detach(idx);
            self.push_front(idx);
            return;
        }
        let idx = if let Some(free_idx) = self.free.pop() {
            self.nodes[free_idx] = Some(LruNode {
                key,
                val,
                prev: None,
                next: None,
            });
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(LruNode {
                key,
                val,
                prev: None,
                next: None,
            }));
            idx
        };
        self.push_front(idx);
        self.map.insert(key, idx);
    }

    /// Insert `(key, val)` at the LRU tail without displacing MRU entries.
    /// If `key` is already tracked, updates its value in place without promoting it.
    pub fn insert_cold(&mut self, key: K, val: V) {
        if let Some(&idx) = self.map.get(&key) {
            if let Some(node) = self.nodes[idx].as_mut() {
                node.val = val;
            }
            return;
        }
        let idx = if let Some(free_idx) = self.free.pop() {
            self.nodes[free_idx] = Some(LruNode {
                key,
                val,
                prev: None,
                next: None,
            });
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(LruNode {
                key,
                val,
                prev: None,
                next: None,
            }));
            idx
        };
        self.push_back(idx);
        self.map.insert(key, idx);
    }

    pub fn touch(&mut self, key: &K) -> Option<V> {
        let &idx = self.map.get(key)?;
        self.detach(idx);
        self.push_front(idx);
        self.nodes[idx].as_ref().map(|n| n.val)
    }

    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        let idx = self.tail?;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.free.push(idx);
        self.map.remove(&node.key);
        Some((node.key, node.val))
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.free.push(idx);
        Some(node.val)
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.nodes[idx].as_mut().expect("valid LRU node");
            (n.prev.take(), n.next.take())
        };
        match prev {
            Some(p) => self.nodes[p].as_mut().expect("prev node").next = next,
            None => self.head = next,
        }
        match next {
            Some(nx) => self.nodes[nx].as_mut().expect("next node").prev = prev,
            None => self.tail = prev,
        }
    }

    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        {
            let n = self.nodes[idx].as_mut().expect("valid LRU node");
            n.prev = None;
            n.next = old_head;
        }
        if let Some(h) = old_head {
            self.nodes[h].as_mut().expect("head node").prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    fn push_back(&mut self, idx: usize) {
        let old_tail = self.tail;
        {
            let n = self.nodes[idx].as_mut().expect("valid LRU node");
            n.prev = old_tail;
            n.next = None;
        }
        if let Some(t) = old_tail {
            self.nodes[t].as_mut().expect("tail node").next = Some(idx);
        } else {
            self.head = Some(idx);
        }
        self.tail = Some(idx);
    }
}

/// Bounded O(1) two-generation 2-hit promotion gate for NVMe -> DRAM reads.
///
/// The first NVMe hit records the key and returns `false` (transient read — DRAM copy
/// is reclaimed immediately after DMA with zero NVMe write amplification). A second hit
/// within two generations returns `true` so genuinely hot keys promote into DRAM.
pub struct NvmePromoFilter {
    current: HashSet<ObjectKey>,
    previous: HashSet<ObjectKey>,
    max_entries_per_gen: usize,
}

impl Default for NvmePromoFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl NvmePromoFilter {
    pub const DEFAULT_MAX_ENTRIES_PER_GEN: usize = 131_072;

    pub fn new() -> Self {
        Self::with_capacity_per_gen(Self::DEFAULT_MAX_ENTRIES_PER_GEN)
    }

    pub fn with_capacity_per_gen(max_entries_per_gen: usize) -> Self {
        Self {
            current: HashSet::new(),
            previous: HashSet::new(),
            max_entries_per_gen: max_entries_per_gen.max(1),
        }
    }

    /// Record an NVMe read hit for `key`. Returns `true` iff `key` has already been seen
    /// in the current or previous generation (2nd+ hit -> promote into DRAM).
    pub fn record_hit_and_should_promote(&mut self, key: ObjectKey) -> bool {
        if self.current.contains(&key) {
            return true;
        }
        let seen_prev = self.previous.contains(&key);
        if self.current.len() >= self.max_entries_per_gen {
            self.previous = std::mem::take(&mut self.current);
        }
        self.current.insert(key);
        seen_prev
    }
}

/// Per-tier O(1) LRU eviction policy across GPU VRAM, Host DRAM, and Local NVMe.
#[derive(Default)]
pub struct LruEviction {
    gpu: LruList<ObjectKey, FrameId>,
    dram: LruList<ObjectKey, FrameId>,
    nvme: LruList<ObjectKey, FrameId>,
}

impl LruEviction {
    pub fn new() -> Self {
        Self::default()
    }

    fn list_mut(&mut self, tier: TierKind) -> Option<&mut LruList<ObjectKey, FrameId>> {
        match tier {
            TierKind::GpuVram => Some(&mut self.gpu),
            TierKind::HostDram => Some(&mut self.dram),
            TierKind::LocalNvme => Some(&mut self.nvme),
            _ => None,
        }
    }
}

impl EvictionPolicy for LruEviction {
    fn on_insert(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        if let Some(list) = self.list_mut(tier) {
            list.insert(key, frame);
        }
    }

    fn on_insert_cold(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        if let Some(list) = self.list_mut(tier) {
            list.insert_cold(key, frame);
        }
    }

    fn on_access(&mut self, tier: TierKind, key: ObjectKey, frame: FrameId) {
        if let Some(list) = self.list_mut(tier) {
            if list.touch(&key).is_none() {
                list.insert(key, frame);
            }
        }
    }

    fn on_remove(&mut self, tier: TierKind, key: ObjectKey, _frame: FrameId) {
        if let Some(list) = self.list_mut(tier) {
            let _ = list.remove(&key);
        }
    }

    fn select_victim(&mut self, tier: TierKind) -> Option<(ObjectKey, FrameId)> {
        self.list_mut(tier)?.pop_lru()
    }
}

/// Never evicts frames; surfaces `CapacityExhausted` when a tier is full.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoEviction;

impl EvictionPolicy for NoEviction {
    fn on_insert(&mut self, _tier: TierKind, _key: ObjectKey, _frame: FrameId) {}
    fn on_access(&mut self, _tier: TierKind, _key: ObjectKey, _frame: FrameId) {}
    fn on_remove(&mut self, _tier: TierKind, _key: ObjectKey, _frame: FrameId) {}
    fn select_victim(&mut self, _tier: TierKind) -> Option<(ObjectKey, FrameId)> {
        None
    }
}

// ============================================================================
// Built-in Tiering Policies
// ============================================================================

/// Default cascading tiering policy:
/// - Evicting from GPU HBM demotes to Host DRAM.
/// - Evicting from Host DRAM spills to Local NVMe.
/// - GPU reads hitting DRAM/NVMe promote into GPU HBM.
/// - Optional proactive async NVMe spill watermark for DRAM.
#[derive(Clone, Copy, Debug)]
pub struct CascadeTiering {
    pub retain_remote_fetches: bool,
    pub promote_gpu_on_cold_hit: bool,
    pub demote_gpu_to_dram: bool,
    pub demote_dram_to_nvme: bool,
    pub spill_watermark: Option<SpillWatermark>,
}

impl Default for CascadeTiering {
    fn default() -> Self {
        Self {
            retain_remote_fetches: false,
            promote_gpu_on_cold_hit: true,
            demote_gpu_to_dram: true,
            demote_dram_to_nvme: true,
            spill_watermark: None,
        }
    }
}

impl CascadeTiering {
    /// Serving-oriented policy for inference engines (e.g., vLLM) that already
    /// maintain their own large hot GPU KV cache:
    /// - Avoids write amplification from promoting cold DRAM/NVMe hits into a
    ///   secondary GPU HBM tier (reads stream directly via 1-hop pinned H2D).
    /// - Proactively spills cold DRAM frames to NVMe asynchronously via `io_uring`
    ///   before DRAM fills up.
    pub fn serving_dram_first(high_water_free_frames: usize, max_in_flight: usize) -> Self {
        let wm = if high_water_free_frames > 0 && max_in_flight > 0 {
            Some(SpillWatermark {
                high_water_free_frames,
                max_in_flight,
            })
        } else {
            None
        };
        Self {
            retain_remote_fetches: false,
            promote_gpu_on_cold_hit: false,
            demote_gpu_to_dram: true,
            demote_dram_to_nvme: true,
            spill_watermark: wm,
        }
    }
}

impl TieringPolicy for CascadeTiering {
    fn retain_remote_fetches(&self) -> bool {
        self.retain_remote_fetches
    }

    fn promote_on_hit(&self, from: HitLocation, consumer: DataDomain) -> Option<TierKind> {
        if !self.promote_gpu_on_cold_hit {
            return None;
        }
        match (consumer, from) {
            (DataDomain::Gpu, HitLocation::LocalDram | HitLocation::LocalNvme) => {
                Some(TierKind::GpuVram)
            }
            (DataDomain::Gpu, HitLocation::RemotePeer { gdr: false, .. }) => {
                Some(TierKind::GpuVram)
            }
            _ => None,
        }
    }

    fn demote_on_evict(&self, from: TierKind, _key: ObjectKey) -> Option<TierKind> {
        match from {
            TierKind::GpuVram if self.demote_gpu_to_dram => Some(TierKind::HostDram),
            TierKind::HostDram if self.demote_dram_to_nvme => Some(TierKind::LocalNvme),
            _ => None,
        }
    }

    fn dram_spill_watermark(&self) -> Option<SpillWatermark> {
        self.spill_watermark
    }
}

/// Never promotes on cold hit and never cascade-demotes on eviction.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoTiering;

impl TieringPolicy for NoTiering {
    fn promote_on_hit(&self, _from: HitLocation, _consumer: DataDomain) -> Option<TierKind> {
        None
    }

    fn demote_on_evict(&self, _from: TierKind, _key: ObjectKey) -> Option<TierKind> {
        None
    }
}

// ============================================================================
// Built-in Peer Routing Policies
// ============================================================================

/// Sequentially probes every connected peer on a local miss.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeAllPeers;

impl PeerRoutingPolicy for ProbeAllPeers {
    fn probe_order(&self, _key: ObjectKey, peers: &[PeerMeta], out: &mut Vec<usize>) {
        out.clear();
        for p in peers {
            out.push(p.index);
        }
    }
}

/// Probes the consistent-hash home peer first, then falls back to remaining peers.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConsistentHashPeers {
    pub fallback_to_others: bool,
}

impl PeerRoutingPolicy for ConsistentHashPeers {
    fn probe_order(&self, key: ObjectKey, peers: &[PeerMeta], out: &mut Vec<usize>) {
        out.clear();
        if peers.is_empty() {
            return;
        }
        let mut h = key.0[0] ^ key.0[1].wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let start = ((h ^ (h >> 31)) as usize) % peers.len();
        out.push(peers[start].index);
        if self.fallback_to_others {
            for offset in 1..peers.len() {
                out.push(peers[(start + offset) % peers.len()].index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: u64) -> ObjectKey {
        ObjectKey([id, 0, 0, 0])
    }

    #[test]
    fn lru_list_insert_cold_places_at_tail_without_displacing_mru() {
        let mut lru: LruList<u64, u32> = LruList::new();
        lru.insert(1, 10);
        lru.insert(2, 20);
        // Order: head=2, tail=1. Insert 3 at cold tail -> head=2, 1, tail=3.
        lru.insert_cold(3, 30);
        assert_eq!(lru.pop_lru(), Some((3, 30)));
        assert_eq!(lru.pop_lru(), Some((1, 10)));
        assert_eq!(lru.pop_lru(), Some((2, 20)));
        assert_eq!(lru.pop_lru(), None);

        // insert_cold on empty list works cleanly.
        lru.insert_cold(9, 90);
        assert_eq!(lru.pop_lru(), Some((9, 90)));

        // insert_cold on existing key updates val without moving it to MRU.
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert_cold(1, 11);
        assert_eq!(lru.pop_lru(), Some((1, 11)));
    }

    #[test]
    fn nvme_promo_filter_two_hit_gate_and_rotation() {
        let mut filter = NvmePromoFilter::with_capacity_per_gen(4);
        assert!(!filter.record_hit_and_should_promote(key(1)));
        assert!(filter.record_hit_and_should_promote(key(1)));

        // Fill generation 1 and rotate into generation 2.
        assert!(!filter.record_hit_and_should_promote(key(2)));
        assert!(!filter.record_hit_and_should_promote(key(3)));
        assert!(!filter.record_hit_and_should_promote(key(4)));
        // Triggers rotation: previous = {1,2,3,4}, current = {5}.
        assert!(!filter.record_hit_and_should_promote(key(5)));
        // key(2) is still in `previous`, so 2nd hit promotes!
        assert!(filter.record_hit_and_should_promote(key(2)));
    }
}
