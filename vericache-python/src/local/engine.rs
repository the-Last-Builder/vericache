use super::shard_cap::ShardCapabilityBundle;
use crate::{backend_mode, set_last_error};
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
use transfer::context::ShareCap;
use transfer::cuda::{self, CudaHostRegistration};
use transfer::host::{self, HostContext};
use transfer::nvme::{DIRECT_IO_ALIGNMENT, MAX_IO_LEN};
use vericache_core::backends::{host::HostBackend, nvme::NvmeBackend};
use vericache_core::engine::{HostLease, LocalEngine};
use vericache_core::kernel::{CacheCore, ReadLeaseHandle};
use vericache_core::spec::{CoreError, FrameId, ObjectKey, Ticket, TierId};
use vstd::prelude::*;

const ALIGNMENT: usize = DIRECT_IO_ALIGNMENT as usize;
const SPILL_DEPTH: usize = 4;
const PROMOTION_DEPTH: usize = 4;
const PARALLEL_BATCH_BYTES: usize = 256 * 1024;

fn promotion_reserve(host_capacity: usize, nvme_capacity: usize) -> usize {
    if nvme_capacity == 0 {
        0
    } else {
        host_capacity.min(PROMOTION_DEPTH)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    Invalid,
    Capacity(usize),
    Backend,
    Busy,
}

impl From<CoreError> for Error {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::WouldConflict | CoreError::CapacityExhausted => Self::Busy,
            _ => Self::Backend,
        }
    }
}

#[derive(Default)]
pub(crate) struct Stats {
    pub puts: u64,
    pub gets: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
}

#[derive(Default)]
pub(crate) struct ExtendedStats {
    pub dram_hits: u64,
    pub nvme_hits: u64,
    pub nvme_promotions: u64,
    pub async_writebacks: u64,
    pub clean_dram_evictions: u64,
    pub active_leases: u64,
    pub nvme_read_bytes: u64,
    pub nvme_write_bytes: u64,
}

#[derive(Clone, Copy)]
enum Residency {
    HostOnly {
        host_frame: FrameId,
    },
    Dual {
        host_frame: FrameId,
        nvme_frame: FrameId,
    },
    NvmeOnly {
        nvme_frame: FrameId,
    },
}

impl Residency {
    fn host(self) -> Option<FrameId> {
        match self {
            Self::HostOnly { host_frame } | Self::Dual { host_frame, .. } => Some(host_frame),
            Self::NvmeOnly { .. } => None,
        }
    }

    fn nvme(self) -> Option<FrameId> {
        match self {
            Self::Dual { nvme_frame, .. } | Self::NvmeOnly { nvme_frame } => Some(nvme_frame),
            Self::HostOnly { .. } => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    residency: Residency,
    lru_seq: u64,
    len: usize,
    pin_count: usize,
    spilling: bool,
    prefetched: bool,
    probation: bool,
}

struct ActiveLease {
    handle: ReadLeaseHandle,
    cap: Tracked<ShareCap>,
    key: ObjectKey,
}

#[derive(Debug)]
pub(crate) struct Lease {
    pub id: u64,
    pub address: usize,
    pub len: usize,
}

#[derive(Clone, Copy)]
struct InflightSpill {
    key: ObjectKey,
    ticket: Ticket,
    eager: bool,
}

#[derive(Clone, Copy)]
struct Promotion {
    key: ObjectKey,
    ticket: Ticket,
    prefetched: bool,
}

struct ShardState {
    core: CacheCore,
    host_ctx: HostContext,
    nvme: Option<NvmeBackend>,
}

type Lru = BTreeMap<u64, ObjectKey>;

struct Directory {
    entries: HashMap<ObjectKey, Entry>,
    host_order: Lru,
    nvme_order: Lru,
    dirty: Lru,
    clean: Lru,
    probation: Lru,
    probation_count: usize,
    nvme_only: Lru,
    cold_shards: Arc<AtomicUsize>,
    replicas: Lru,
    sequence: u64,
    scratch: Vec<u8>,
    inflight_spills: Vec<InflightSpill>,
    promotions: Vec<Promotion>,
    leases: HashMap<u64, ActiveLease>,
    lease_seq: u64,
    shard_index: u64,
    num_shards: u64,
    host_capacity: usize,
    nvme_capacity: usize,
    high_watermark: usize,
    spill_depth: usize,
    promotion_depth: usize,
    probation_enabled: bool,
    evicted: Option<VecDeque<ObjectKey>>,
    stats: Stats,
    extended: ExtendedStats,
}

struct ShardInstance {
    registration: Option<CudaHostRegistration<'static>>,
    host: Option<Arc<HostBackend>>,
    state: Option<ShardState>,
    caps: ShardCapabilityBundle,
    data: Directory,
    // Each ring addresses offsets starting at zero in its own backing file.
    _file_lock: Option<File>,
}

// The shard mutex serializes use of the backends and owns the slab and ring.
// External readers only see frames held by read leases; the directory and core
// both prevent writes/recycling of those frames until their leases retire.
unsafe impl Send for ShardInstance {}

pub struct VericacheVllmEngine {
    shards: Vec<Mutex<ShardInstance>>,
    workers: OnceLock<Option<ThreadPool>>,
    cold_shards: Arc<AtomicUsize>,
    frame_size: usize,
    has_nvme: bool,
    cuda_pinned: bool,
}

pub(crate) struct Options {
    pub pin_cuda: bool,
    pub track_evictions: bool,
}

impl VericacheVllmEngine {
    pub(crate) fn new(
        num_shards: usize,
        host_frames: usize,
        nvme_frames: usize,
        frame_size: usize,
        nvme_path: Option<&str>,
    ) -> Result<Self, Error> {
        Self::with_options(
            num_shards,
            host_frames,
            nvme_frames,
            frame_size,
            nvme_path,
            Options {
                pin_cuda: false,
                track_evictions: false,
            },
        )
    }

    pub(crate) fn with_options(
        num_shards: usize,
        host_frames: usize,
        nvme_frames: usize,
        frame_size: usize,
        nvme_path: Option<&str>,
        options: Options,
    ) -> Result<Self, Error> {
        if num_shards == 0
            || num_shards > u32::MAX as usize
            || host_frames == 0
            || frame_size == 0
            || frame_size > isize::MAX as usize
        {
            return Err(Error::Invalid);
        }
        let path = nvme_path.filter(|_| nvme_frames > 0);
        let has_nvme = path.is_some();
        let nvme_frames = path.map_or(0, |_| nvme_frames);
        let storage_frame_size = if has_nvme {
            frame_size
                .checked_next_multiple_of(ALIGNMENT)
                .filter(|size| *size <= MAX_IO_LEN)
                .ok_or(Error::Invalid)?
        } else {
            frame_size
        };
        // Reserve space for parallel promotions even with a full DRAM tier.
        let host_slots = host_frames
            .checked_add(promotion_reserve(host_frames, nvme_frames))
            .ok_or(Error::Invalid)?;
        if host_slots
            .checked_add(nvme_frames)
            .is_none_or(|n| n > u32::MAX as usize)
            || host_slots
                .checked_mul(storage_frame_size)
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(Error::Invalid);
        }
        let mut shards = Vec::new();
        shards
            .try_reserve_exact(num_shards)
            .map_err(|_| Error::Invalid)?;
        if let Err(error) = backend_mode::claim_local() {
            set_last_error(error);
            return Err(Error::Backend);
        }
        let pin_cuda = options.pin_cuda && cuda::device_count().is_ok_and(|n| n != 0);
        let mut cuda_pinned = pin_cuda;
        let cold_shards = Arc::new(AtomicUsize::new(0));
        for index in 0..num_shards {
            let shard_path = path.map(|path| {
                if num_shards == 1 {
                    path.to_owned()
                } else {
                    format!("{path}.shard-{index}")
                }
            });
            let dir = Directory::new(
                index,
                num_shards,
                host_frames,
                nvme_frames,
                options.track_evictions,
                Arc::clone(&cold_shards),
            );
            let shard = ShardInstance::new(
                dir,
                host_slots as u32,
                storage_frame_size,
                shard_path.as_deref(),
                pin_cuda,
            )?;
            cuda_pinned &= shard.registration.is_some();
            shards.push(Mutex::new(shard));
        }
        Ok(Self {
            shards,
            workers: OnceLock::new(),
            cold_shards,
            frame_size,
            has_nvme,
            cuda_pinned,
        })
    }

    pub(crate) fn frame_size(&self) -> usize {
        self.frame_size
    }
    pub(crate) fn num_shards(&self) -> usize {
        self.shards.len()
    }
    pub(crate) fn is_cuda_pinned(&self) -> bool {
        self.cuda_pinned
    }

    fn shard_index(&self, key: ObjectKey) -> usize {
        let [a, b, c, d] = key.0;
        let mixed = splitmix64(a ^ splitmix64(b ^ splitmix64(c ^ splitmix64(d))));
        (mixed % self.shards.len() as u64) as usize
    }

    fn lock_shard(&self, index: usize) -> Result<MutexGuard<'_, ShardInstance>, Error> {
        self.shards
            .get(index)
            .ok_or(Error::Invalid)?
            .lock()
            .map_err(|_| Error::Backend)
    }

    pub(crate) fn put(&self, key: ObjectKey, src: &[u8]) -> Result<(), Error> {
        if src.len() > self.frame_size {
            return Err(Error::Invalid);
        }
        self.lock_shard(self.shard_index(key))?
            .with_engine(|engine, caps, dir| {
                // With eager writeback disabled, a scalar put spills just
                // enough for its own admission. Batches still pipeline.
                let depth = if dir.high_watermark == usize::MAX {
                    1
                } else {
                    dir.spill_depth
                };
                dir.put(engine, caps, key, src, depth)
            })
    }

    pub(crate) fn get(&self, key: ObjectKey, dst: &mut [u8]) -> Result<Option<usize>, Error> {
        self.lock_shard(self.shard_index(key))?
            .with_engine(|engine, caps, dir| dir.get(engine, caps, key, dst))
    }

    pub(crate) fn contains(&self, key: ObjectKey) -> Result<bool, Error> {
        Ok(self
            .lock_shard(self.shard_index(key))?
            .data
            .entries
            .contains_key(&key))
    }

    pub(crate) fn evict(&self, key: ObjectKey) -> Result<bool, Error> {
        self.lock_shard(self.shard_index(key))?
            .with_engine(|engine, caps, dir| dir.remove(engine, caps, key, true))
    }

    pub(crate) fn acquire(&self, key: ObjectKey) -> Result<Option<Lease>, Error> {
        self.lock_shard(self.shard_index(key))?
            .with_engine(|engine, caps, dir| {
                dir.poll(engine, caps)?;
                dir.acquire(engine, caps, key, None)
            })
    }

    pub(crate) fn release(&self, id: u64) -> Result<(), Error> {
        self.lock_shard(self.lease_shard(id))?
            .with_engine(|engine, caps, dir| dir.release(engine, caps, id))
    }

    fn lease_shard(&self, id: u64) -> usize {
        (id % self.num_shards() as u64) as usize
    }

    fn batch<K: Copy + Send, T: Send, R: Send>(
        &self,
        items: impl IntoIterator<Item = (K, T)>,
        index: impl Fn(K) -> usize,
        parallel: bool,
        read_key: Option<fn(K) -> ObjectKey>,
        op: impl Fn(&mut ShardInstance, K, T, Option<&HashSet<ObjectKey>>) -> Result<R, Error> + Sync,
    ) -> Vec<Result<R, Error>> {
        let mut groups: Vec<Vec<_>> = (0..self.num_shards()).map(|_| Vec::new()).collect();
        let mut results = Vec::new();
        for (position, (key, item)) in items.into_iter().enumerate() {
            results.push(Err(Error::Invalid));
            if let Some(group) = groups.get_mut(index(key)) {
                group.push((position, key, item));
            }
        }
        let groups: Vec<_> = groups
            .into_iter()
            .enumerate()
            .filter(|(_, group)| !group.is_empty())
            .collect();
        let run = |(shard_index, group): (_, Vec<(usize, K, T)>)| {
            let mut shard = self.lock_shard(shard_index).and_then(|mut shard| {
                let mut requested = None;
                if let Some(read_key) = read_key {
                    shard.with_engine(|engine, caps, dir| {
                        dir.poll(engine, caps)?;
                        if group.len() > 1 && !dir.nvme_only.is_empty() {
                            let keys: Vec<_> =
                                group.iter().map(|(_, key, _)| read_key(*key)).collect();
                            let batch_keys = keys.iter().copied().collect();
                            dir.prewarm(engine, caps, &keys, &batch_keys)?;
                            requested = Some(batch_keys);
                        }
                        Ok(())
                    })?;
                }
                Ok((shard, requested))
            });
            group
                .into_iter()
                .map(|(pos, key, item)| {
                    let result = match &mut shard {
                        Ok((shard, requested)) => catch_unwind(AssertUnwindSafe(|| {
                            op(shard, key, item, requested.as_ref())
                        }))
                        .unwrap_or(Err(Error::Backend)),
                        Err(error) => Err(*error),
                    };
                    (pos, result)
                })
                .collect::<Vec<_>>()
        };
        let mut merge = |group: Vec<(usize, Result<R, Error>)>| {
            for (pos, result) in group {
                results[pos] = result;
            }
        };
        let workers = if parallel && groups.len() > 1 {
            self.workers
                .get_or_init(|| {
                    let cores = std::thread::available_parallelism().map_or(1, usize::from);
                    ThreadPoolBuilder::new()
                        .num_threads(self.num_shards().min(cores))
                        .thread_name(|index| format!("vericache-shard-{index}"))
                        .build()
                        .ok()
                })
                .as_ref()
        } else {
            None
        };
        if let Some(workers) = workers {
            for group in workers.install(|| groups.into_par_iter().map(run).collect::<Vec<_>>()) {
                merge(group);
            }
        } else {
            for group in groups {
                merge(run(group));
            }
        }
        results
    }

    pub(crate) fn has_cold_data(&self) -> bool {
        self.cold_shards.load(Ordering::Relaxed) != 0
    }

    pub(crate) fn contains_batch(
        &self,
        keys: impl ExactSizeIterator<Item = ObjectKey>,
    ) -> Vec<Result<bool, Error>> {
        self.batch(
            keys.map(|key| (key, ())),
            |key| self.shard_index(key),
            false,
            None,
            |shard, key, (), _| Ok(shard.data.entries.contains_key(&key)),
        )
    }

    pub(crate) fn put_batch(
        &self,
        keys: impl ExactSizeIterator<Item = ObjectKey>,
        sources: Vec<Option<&[u8]>>,
    ) -> Vec<Result<(), Error>> {
        let parallel = keys.len().saturating_mul(self.frame_size) >= PARALLEL_BATCH_BYTES;
        self.batch(
            keys.zip(sources),
            |key| self.shard_index(key),
            parallel,
            None,
            |shard, key, src, _| {
                let src = src
                    .filter(|src| src.len() <= self.frame_size)
                    .ok_or(Error::Invalid)?;
                shard.with_engine(|engine, caps, dir| dir.put(engine, caps, key, src, dir.spill_depth))
            },
        )
    }

    pub(crate) fn get_batch(
        &self,
        keys: impl ExactSizeIterator<Item = ObjectKey>,
        destinations: Vec<Option<&mut [u8]>>,
    ) -> Vec<Result<Option<usize>, Error>> {
        let parallel = keys.len().saturating_mul(self.frame_size) >= PARALLEL_BATCH_BYTES;
        self.batch(
            keys.zip(destinations),
            |key| self.shard_index(key),
            parallel,
            None,
            |shard, key, dst, _| {
                let dst = dst.ok_or(Error::Invalid)?;
                shard.with_engine(|engine, caps, dir| dir.get(engine, caps, key, dst))
            },
        )
    }

    pub(crate) fn acquire_batch(
        &self,
        keys: impl ExactSizeIterator<Item = ObjectKey>,
    ) -> Vec<Result<Option<Lease>, Error>> {
        self.batch(
            keys.map(|key| (key, ())),
            |key| self.shard_index(key),
            self.has_cold_data(),
            self.has_nvme.then_some(std::convert::identity),
            |shard, key, (), requested| {
                shard.with_engine(|engine, caps, dir| dir.acquire(engine, caps, key, requested))
            },
        )
    }

    pub(crate) fn release_batch(&self, ids: &[u64]) -> Result<(), Error> {
        self.batch(
            ids.iter().map(|&id| (id, ())),
            |id| self.lease_shard(id),
            false,
            None,
            |shard, id, (), _| shard.with_engine(|engine, caps, dir| dir.release(engine, caps, id)),
        )
        .into_iter()
        .find(Result::is_err)
        .unwrap_or(Ok(()))
    }

    pub(crate) fn poll_writeback(&self) -> Result<usize, Error> {
        let mut count = 0;
        for shard in &self.shards {
            let mut shard = match shard.try_lock() {
                Ok(shard) => shard,
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Poisoned(_)) => return Err(Error::Backend),
            };
            count += shard.with_engine(|engine, caps, dir| {
                let count = dir.poll(engine, caps)?;
                dir.schedule_spills(engine, caps)?;
                Ok(count)
            })?;
        }
        Ok(count)
    }

    pub(crate) fn set_writeback_policy(&self, percent: usize) -> Result<(), Error> {
        if percent > 100 {
            return Err(Error::Invalid);
        }
        for index in 0..self.num_shards() {
            let mut shard = self.lock_shard(index)?;
            shard.data.high_watermark = if percent == 0 {
                usize::MAX
            } else {
                ((shard.data.host_capacity as u64 * percent as u64).div_ceil(100)) as usize
            };
        }
        Ok(())
    }

    pub(crate) fn drain_evictions(
        &self,
        capacity: usize,
        mut write: impl FnMut(usize, ObjectKey),
    ) -> Result<usize, Error> {
        let mut count = 0;
        for index in 0..self.num_shards() {
            if let Some(evicted) = &mut self.lock_shard(index)?.data.evicted {
                let n = (capacity - count).min(evicted.len());
                for key in evicted.drain(..n) {
                    write(count, key);
                    count += 1;
                }
                if count == capacity {
                    break;
                }
            }
        }
        Ok(count)
    }

    pub(crate) fn stats(&self) -> Result<Stats, Error> {
        let mut total = Stats::default();
        for index in 0..self.num_shards() {
            let s = &self.lock_shard(index)?.data.stats;
            total.puts += s.puts;
            total.gets += s.gets;
            total.hits += s.hits;
            total.misses += s.misses;
            total.evictions += s.evictions;
            total.bytes_written += s.bytes_written;
            total.bytes_read += s.bytes_read;
        }
        Ok(total)
    }

    pub(crate) fn extended_stats(&self) -> Result<ExtendedStats, Error> {
        let mut total = ExtendedStats::default();
        for index in 0..self.num_shards() {
            let shard = self.lock_shard(index)?;
            let s = &shard.data.extended;
            total.dram_hits += s.dram_hits;
            total.nvme_hits += s.nvme_hits;
            total.nvme_promotions += s.nvme_promotions;
            total.async_writebacks += s.async_writebacks;
            total.clean_dram_evictions += s.clean_dram_evictions;
            total.active_leases += shard.data.leases.len() as u64;
            total.nvme_read_bytes += s.nvme_read_bytes;
            total.nvme_write_bytes += s.nvme_write_bytes;
        }
        Ok(total)
    }

    pub(crate) fn set_nvme_policy(&self, depth: usize, probation: bool) -> Result<(), Error> {
        if !matches!(depth, 1 | 4) {
            return Err(Error::Invalid);
        }
        let mut shards = (0..self.num_shards())
            .map(|index| self.lock_shard(index))
            .collect::<Result<Vec<_>, _>>()?;
        if shards.iter().any(|shard| !shard.data.entries.is_empty()) {
            return Err(Error::Busy);
        }
        for shard in &mut shards {
            shard.data.spill_depth = depth;
            shard.data.promotion_depth = depth;
            shard.data.probation_enabled = probation;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn slab_range(&self, index: usize) -> std::ops::Range<usize> {
        let shard = self.lock_shard(index).unwrap();
        let host = shard.host.as_ref().unwrap();
        let base = host.buffer().slot_slice(0).addr();
        base..base + host.num_frames as usize * host.buffer().slot_len_val()
    }
}

impl ShardInstance {
    fn with_engine<T>(
        &mut self,
        run: impl FnOnce(
            &mut LocalEngine<'_>,
            &mut ShardCapabilityBundle,
            &mut Directory,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let ShardState {
            core,
            host_ctx,
            nvme,
        } = self.state.take().ok_or(Error::Backend)?;
        let mut engine = LocalEngine::new(core, self.host.as_ref().unwrap(), host_ctx, None, nvme);
        let result = catch_unwind(AssertUnwindSafe(|| {
            run(&mut engine, &mut self.caps, &mut self.data)
        }));
        let LocalEngine {
            core,
            host_ctx,
            nvme,
            ..
        } = engine;
        self.state = Some(ShardState {
            core,
            host_ctx,
            nvme,
        });
        result.unwrap_or_else(|panic| resume_unwind(panic))
    }
}

verus! {

impl ShardInstance {
    fn new(
        data: Directory,
        host_slots: u32,
        frame_size: usize,
        path: Option<&str>,
        pin_cuda: bool,
    ) -> Result<Self, Error> {
        let mut caps = ShardCapabilityBundle::take().ok_or(Error::Backend)?;
        let mut core = caps.make_local_core(frame_size);
        let source = caps.mint_host();
        let host = Arc::new(HostBackend::new(
            &mut core, Tracked(caps.instance.borrow()), Tracked(caps.alloc.borrow_mut()),
            source, host_slots, ALIGNMENT,
        )?);
        let host_ctx = HostContext::new(caps.mint_host());
        let registration = if pin_cuda {
            cuda::host_register(host.buffer()).ok().map(extend_registration)
        } else {
            None
        };
        let mut shard = Self {
            registration,
            host: Some(host),
            state: Some(ShardState { core, host_ctx, nvme: None }),
            caps,
            data,
            _file_lock: None,
        };
        if let Some(path) = path {
            shard._file_lock = Some(lock_file(path)?);
            let state = shard.state.as_mut().unwrap();
            let src = shard.caps.mint_nvme();
            state.nvme = Some(NvmeBackend::new_with_path(
                &mut state.core, &state.host_ctx,
                Tracked(shard.caps.instance.borrow()), Tracked(shard.caps.alloc.borrow_mut()),
                src, path, shard.data.nvme_capacity as u32, 8, false, frame_size,
            )?);
        }
        Ok(shard)
    }
}

impl Drop for ShardInstance {
    #[cfg_attr(not(verus_only), allow(unused_mut, unused_variables, non_shorthand_field_patterns))]
    fn drop(&mut self) {
        let drained = self.with_engine(|engine, caps, dir| dir.drain(engine, caps));
        let Some(ShardState { mut core, mut host_ctx, nvme }) = self.state.take() else { return; };
        drop(nvme);
        drop(self.registration.take());
        if let Some(host) = self.host.take() {
            if drained.is_err() {
                // A failed ring can leave ownership of slab bytes with the
                // kernel. Retain the allocation if retirement was uncertain.
                return;
            }
            let host = Arc::try_unwrap(host).ok().unwrap();
            let tracked mut tokens = Map::tracked_empty();
            proof {
                let mut index = 0int;
                while index < host.num_frames as int {
                    let tracked frame = core.frames.ghosts.borrow_mut().tracked_remove(
                        (host.first_frame.0 as int + index) as nat,
                    );
                    tokens.tracked_insert(index as usize, frame.token.tracked_unwrap());
                    index = index + 1;
                }
            }
            let _ = host::free_slab(
                &mut host_ctx, Tracked(self.caps.instance.borrow()), host.buffer, Tracked(tokens),
                Tracked(&core.ledger.borrow().submitted), Tracked(&core.ledger.borrow().knows),
                Ghost(core.citations()),
            );
        }
    }
}

} // verus!

fn extend_registration(registration: CudaHostRegistration<'_>) -> CudaHostRegistration<'static> {
    // SAFETY: the borrow targets the stable Arc allocation owned by ShardInstance. The
    // registration never escapes and is dropped before taking apart that allocation.
    unsafe { std::mem::transmute(registration) }
}

impl Directory {
    fn new(
        shard_index: usize,
        num_shards: usize,
        host_capacity: usize,
        nvme_capacity: usize,
        track: bool,
        cold_shards: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            host_order: Lru::new(),
            nvme_order: Lru::new(),
            dirty: Lru::new(),
            clean: Lru::new(),
            probation: Lru::new(),
            probation_count: 0,
            nvme_only: Lru::new(),
            cold_shards,
            replicas: Lru::new(),
            sequence: 0,
            scratch: Vec::new(),
            inflight_spills: Vec::with_capacity(if nvme_capacity > 0 { SPILL_DEPTH } else { 0 }),
            promotions: Vec::new(),
            leases: HashMap::new(),
            lease_seq: 0,
            shard_index: shard_index as u64,
            num_shards: num_shards as u64,
            host_capacity,
            nvme_capacity,
            high_watermark: host_capacity,
            spill_depth: SPILL_DEPTH,
            promotion_depth: PROMOTION_DEPTH,
            probation_enabled: true,
            evicted: track.then(VecDeque::new),
            stats: Stats::default(),
            extended: ExtendedStats::default(),
        }
    }

    fn record_eviction(&mut self, key: ObjectKey) {
        self.stats.evictions += 1;
        if let Some(evicted) = &mut self.evicted {
            evicted.push_back(key);
        }
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    fn orders(&mut self, entry: Entry) -> impl Iterator<Item = &mut Lru> {
        let available = entry.pin_count == 0 && !entry.spilling;
        [
            entry
                .residency
                .host()
                .is_some()
                .then_some(&mut self.host_order),
            entry
                .residency
                .nvme()
                .is_some()
                .then_some(&mut self.nvme_order),
            (available && matches!(entry.residency, Residency::HostOnly { .. }))
                .then_some(&mut self.dirty),
            (available && !entry.probation && matches!(entry.residency, Residency::Dual { .. }))
                .then_some(&mut self.clean),
            (available && entry.probation).then_some(&mut self.probation),
            matches!(entry.residency, Residency::NvmeOnly { .. }).then_some(&mut self.nvme_only),
            matches!(entry.residency, Residency::Dual { .. }).then_some(&mut self.replicas),
        ]
        .into_iter()
        .flatten()
    }

    fn set(&mut self, key: ObjectKey, next: Option<Entry>) {
        let had_cold_data = !self.nvme_only.is_empty();
        if let Some(old) = self.entries.remove(&key) {
            self.probation_count -= usize::from(old.probation);
            for order in self.orders(old) {
                order.remove(&old.lru_seq);
            }
        }
        if let Some(next) = next {
            self.probation_count += usize::from(next.probation);
            for order in self.orders(next) {
                order.insert(next.lru_seq, key);
            }
            self.entries.insert(key, next);
        }
        match (had_cold_data, !self.nvme_only.is_empty()) {
            (false, true) => {
                self.cold_shards.fetch_add(1, Ordering::Relaxed);
            }
            (true, false) => {
                self.cold_shards.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn recycle(engine: &mut LocalEngine<'_>, tier: TierId, frame: FrameId) -> Result<(), Error> {
        engine.evict_frame(frame)?;
        engine.reclaim_draining(tier, usize::MAX);
        Ok(())
    }

    fn drop_host(&mut self, engine: &mut LocalEngine<'_>, key: ObjectKey) -> Result<(), Error> {
        let mut entry = self.entries[&key];
        let Residency::Dual {
            host_frame,
            nvme_frame,
        } = entry.residency
        else {
            return Err(Error::Backend);
        };
        debug_assert_eq!(entry.pin_count, 0);
        Self::recycle(engine, engine.host_tier(), host_frame)?;
        entry.residency = Residency::NvmeOnly { nvme_frame };
        entry.prefetched = false;
        entry.probation = false;
        self.set(key, Some(entry));
        self.extended.clean_dram_evictions += 1;
        Ok(())
    }

    fn drop_nvme(&mut self, engine: &mut LocalEngine<'_>, key: ObjectKey) -> Result<(), Error> {
        let mut entry = self.entries[&key];
        let frame = entry.residency.nvme().ok_or(Error::Backend)?;
        Self::recycle(engine, engine.nvme_tier().ok_or(Error::Backend)?, frame)?;
        if let Some(host_frame) = entry.residency.host() {
            entry.residency = Residency::HostOnly { host_frame };
            entry.probation = false;
            self.set(key, Some(entry));
        } else {
            self.set(key, None);
            self.record_eviction(key);
        }
        Ok(())
    }

    fn nvme_used(&self) -> usize {
        self.nvme_order.len() + self.inflight_spills.len()
    }

    fn probation_victim(&self, requested: Option<&HashSet<ObjectKey>>) -> Option<ObjectKey> {
        self.probation
            .values()
            .copied()
            .find(|key| requested.is_none_or(|keys| !keys.contains(key)))
    }

    fn host_victim(&self) -> Option<ObjectKey> {
        self.probation_victim(None)
            .or_else(|| self.clean.first_key_value().map(|(_, &key)| key))
    }

    fn first_unpinned_spill(&self) -> Option<Ticket> {
        self.inflight_spills
            .iter()
            .find_map(|spill| (self.entries[&spill.key].pin_count == 0).then_some(spill.ticket))
    }
}

verus! {

impl Directory {
    #[cfg_attr(not(verus_only), allow(unused_variables, non_shorthand_field_patterns))]
    fn poll(&mut self, engine: &mut LocalEngine, caps: &ShardCapabilityBundle) -> Result<usize, Error> {
        if self.inflight_spills.is_empty() && self.promotions.is_empty() { return Ok(0); }
        // A partial submit still owns its ticket. Retry staged SQEs without
        // waiting for a completion, and reconcile whatever the core retires.
        let submit = match &mut engine.nvme {
            Some(nvme) if nvme.staged_len() != 0 => nvme.submit(Ghost(&engine.core)).map(|_| ()),
            _ => Ok(()),
        };
        let progress = engine.poll_progress(Tracked(caps.instance.borrow()));
        let (settled, failed) = self.retire(engine)?;
        submit?;
        progress.map_err(|(error, _)| Error::from(error))?;
        if failed { return Err(Error::Backend); }
        Ok(settled)
    }

    fn retire(&mut self, engine: &mut LocalEngine) -> Result<(usize, bool), Error> {
        let mut settled = 0;
        let mut failed = false;
        let mut index = 0;
        while index < self.inflight_spills.len() {
            let spill = self.inflight_spills[index];
            if !engine.core.timelines.is_retired(spill.ticket) {
                index += 1;
                continue;
            }
            self.inflight_spills.remove(index);
            let mut entry = self.entries[&spill.key];
            entry.spilling = false;
            let nvme_tier = engine.nvme_tier().ok_or(Error::Backend)?;
            if let Some((nvme_frame, _)) = engine.core.lookup_tier(spill.key, nvme_tier) {
                let host_frame = entry.residency.host().ok_or(Error::Backend)?;
                entry.residency = Residency::Dual { host_frame, nvme_frame };
                self.extended.nvme_write_bytes += engine.core.frames.frame_size as u64;
                if spill.eager { self.extended.async_writebacks += 1; }
            } else {
                engine.reclaim_draining(nvme_tier, usize::MAX);
                failed = true;
            }
            self.set(spill.key, Some(entry));
            settled += 1;
        }
        let mut index = 0;
        while index < self.promotions.len() {
            let promotion = self.promotions[index];
            if !engine.core.timelines.is_retired(promotion.ticket) {
                index += 1;
                continue;
            }
            self.promotions.remove(index);
            let mut entry = self.entries[&promotion.key];
            if let Some((host_frame, _)) = engine.core.lookup_tier(promotion.key, engine.host_tier()) {
                let nvme_frame = entry.residency.nvme().ok_or(Error::Backend)?;
                entry.residency = Residency::Dual { host_frame, nvme_frame };
                entry.prefetched = promotion.prefetched;
                entry.probation = self.probation_enabled;
                self.extended.nvme_promotions += 1;
                self.extended.nvme_read_bytes += engine.core.frames.frame_size as u64;
                self.set(promotion.key, Some(entry));
            } else {
                engine.reclaim_draining(engine.host_tier(), usize::MAX);
                failed = true;
            }
        }
        Ok((settled, failed))
    }

    #[cfg_attr(not(verus_only), allow(unused_variables, non_shorthand_field_patterns))]
    fn wait(&mut self, engine: &mut LocalEngine, caps: &ShardCapabilityBundle, ticket: Ticket)
        -> Result<(), Error>
    {
        let waited = engine.wait_nvme_ticket(Tracked(caps.instance.borrow()), ticket);
        let (_, failed) = self.retire(engine)?;
        waited.map_err(|(error, _)| Error::from(error))?;
        if failed { return Err(Error::Backend); }
        Ok(())
    }

    #[cfg_attr(not(verus_only), allow(unused_variables, non_shorthand_field_patterns))]
    fn start_spill(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, eager: bool) -> Result<Ticket, Error>
    {
        let ticket = match engine.tier_dram_to_nvme(
            Tracked(caps.instance.borrow()), Tracked(caps.shares.borrow_mut()), key,
        ) {
            Ok((_, ticket)) | Err((_, Some(ticket))) => ticket,
            Err((error, None)) => return Err(error.into()),
        };
        let mut entry = self.entries[&key];
        entry.spilling = true;
        self.set(key, Some(entry));
        self.inflight_spills.push(InflightSpill { key, ticket, eager });
        Ok(ticket)
    }

    fn schedule_spills(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle)
        -> Result<(), Error>
    {
        if self.nvme_capacity == 0 || self.host_order.len() < self.high_watermark
            || self.host_order.len() > self.host_capacity || !self.promotions.is_empty()
        {
            return Ok(());
        }
        while self.inflight_spills.len() < self.spill_depth {
            let Some((&seq, &key)) = self.dirty.first_key_value() else { break; };
            if self.nvme_used() == self.nvme_capacity {
                // Prioritize writeback of cold DRAM entries. Rotate replicas
                // away from hotter DRAM blocks, and preserve NVMe-only keys
                // until admission actually needs to displace one.
                let Some((&hot, &replica)) = self.replicas.last_key_value() else { break; };
                if seq >= hot { break; }
                self.drop_nvme(engine, replica)?;
            }
            self.start_spill(engine, caps, key, true)?;
        }
        Ok(())
    }

    fn remove(&mut self, engine: &mut LocalEngine, caps: &ShardCapabilityBundle,
        key: ObjectKey, count: bool) -> Result<bool, Error>
    {
        let Some(entry) = self.entries.get(&key) else { return Ok(false); };
        if entry.pin_count != 0 { return Err(Error::Busy); }
        if let Some(pending) = self.promotions.iter().find(|p| p.key == key).copied() {
            self.wait(engine, caps, pending.ticket)?;
        }
        if let Some(spill) = self.inflight_spills.iter().find(|s| s.key == key) {
            let ticket = spill.ticket;
            self.wait(engine, caps, ticket)?;
        }
        let mut entry = self.entries[&key];
        if let Some(frame) = entry.residency.nvme() {
            Self::recycle(engine, engine.nvme_tier().ok_or(Error::Backend)?, frame)?;
            if let Some(host_frame) = entry.residency.host() {
                entry.residency = Residency::HostOnly { host_frame };
                entry.probation = false;
                self.set(key, Some(entry));
            }
        }
        if let Some(frame) = entry.residency.host() {
            Self::recycle(engine, engine.host_tier(), frame)?;
        }
        self.set(key, None);
        if count { self.record_eviction(key); }
        Ok(true)
    }

    fn make_put_room(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        spill_depth: usize) -> Result<(), Error>
    {
        while self.host_order.len() >= self.host_capacity {
            if let Some(key) = self.host_victim() {
                self.drop_host(engine, key)?;
                continue;
            }
            if self.nvme_capacity == 0 {
                let (_, &key) = self.dirty.first_key_value().ok_or(Error::Busy)?;
                self.remove(engine, caps, key, true)?;
                continue;
            }
            self.poll(engine, caps)?;
            if let Some(key) = self.host_victim() {
                self.drop_host(engine, key)?;
                continue;
            }
            while self.inflight_spills.len() < spill_depth {
                let Some((_, &key)) = self.dirty.first_key_value() else { break; };
                if self.nvme_used() == self.nvme_capacity {
                    let victim = self.replicas.last_key_value().or_else(|| {
                        // Displace an NVMe-only key only when this admission
                        // cannot proceed with the spills already queued.
                        if self.first_unpinned_spill().is_some() { None }
                        else { self.nvme_only.first_key_value() }
                    });
                    let Some((_, &victim)) = victim else { break; };
                    self.drop_nvme(engine, victim)?;
                }
                self.start_spill(engine, caps, key, false)?;
            }
            let ticket = self.first_unpinned_spill().ok_or(Error::Busy)?;
            self.wait(engine, caps, ticket)?;
        }
        Ok(())
    }

    fn put(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, src: &[u8], spill_depth: usize) -> Result<(), Error>
    {
        self.poll(engine, caps)?;
        self.finish_promotions(engine, caps)?;
        let replaces = self.entries.get(&key).copied();
        if let Some(entry) = replaces {
            if entry.pin_count != 0 { return Err(Error::Busy); }
            if entry.residency.host().is_none() && self.host_order.len() > self.host_capacity {
                let victim = self.host_victim().ok_or(Error::Busy)?;
                self.drop_host(engine, victim)?;
            }
            self.remove(engine, caps, key, false)?;
        } else {
            self.make_put_room(engine, caps, spill_depth)?;
        }
        let frame_size = engine.core.frames.frame_size;
        let payload = if src.len() == frame_size {
            src
        } else {
            self.scratch.resize(frame_size, 0);
            self.scratch[..src.len()].copy_from_slice(src);
            self.scratch[src.len()..].fill(0);
            &self.scratch
        };
        let (_, host_frame) = match engine.put_host(Tracked(caps.instance.borrow()), key, payload) {
            Ok(resident) => resident,
            Err(error) => {
                if replaces.is_some() { self.record_eviction(key); }
                return Err(error.into());
            },
        };
        let lru_seq = self.next_sequence();
        self.set(key, Some(Entry {
            residency: Residency::HostOnly { host_frame }, lru_seq, len: src.len(),
            pin_count: 0, spilling: false, prefetched: false, probation: false,
        }));
        self.stats.puts += 1;
        self.stats.bytes_written += src.len() as u64;
        // The put is already visible. A failed eager write can be retried by
        // a later poll or admission, without invalidating committed bytes.
        let _ = self.schedule_spills(engine, caps);
        Ok(())
    }

    fn promote(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, requested: Option<&HashSet<ObjectKey>>) -> Result<(), Error>
    {
        self.finish_promotions(engine, caps)?;
        if self.entries[&key].residency.host().is_some() { return Ok(()); }
        if !self.make_promotion_room(engine, requested)? { return Err(Error::Busy); }
        let ticket = self.start_promotion(engine, caps, key, false)?;
        self.wait(engine, caps, ticket)
    }

    fn make_promotion_room(&mut self, engine: &mut LocalEngine,
        requested: Option<&HashSet<ObjectKey>>) -> Result<bool, Error>
    {
        let reserve = promotion_reserve(self.host_capacity, self.nvme_capacity);
        if self.host_order.len() + self.promotions.len() < self.host_capacity + reserve {
            return Ok(true);
        }
        let victim = self.probation_victim(requested).or_else(|| {
            // A second hit may have graduated a block in a reserve slot.
            // Reclaim that space from the protected LRU only when it exceeds
            // the configured resident capacity.
            if self.host_order.len() - self.probation_count <= self.host_capacity {
                return None;
            }
            self.clean.values().copied().find(|key| requested.is_none_or(|keys| !keys.contains(key)))
        });
        let Some(victim) = victim else { return Ok(false); };
        self.drop_host(engine, victim)?;
        Ok(true)
    }

    #[cfg_attr(not(verus_only), allow(unused_variables))]
    fn start_promotion(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, prefetched: bool) -> Result<Ticket, Error>
    {
        let ticket = match engine.warm_host_from_nvme(
            Tracked(caps.instance.borrow()), Tracked(caps.shares.borrow_mut()), key,
        ) {
            Ok((_, ticket)) | Err((_, Some(ticket))) => ticket,
            Err((error, None)) => return Err(error.into()),
        };
        self.promotions.push(Promotion { key, ticket, prefetched });
        Ok(ticket)
    }

    fn finish_promotions(&mut self, engine: &mut LocalEngine, caps: &ShardCapabilityBundle)
        -> Result<(), Error>
    {
        if let Some(pending) = self.promotions.last().copied() {
            self.wait(engine, caps, pending.ticket)?;
        }
        Ok(())
    }

    fn prewarm(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        keys: &[ObjectKey], requested: &HashSet<ObjectKey>) -> Result<(), Error>
    {
        self.finish_promotions(engine, caps)?;
        for &key in keys {
            let Some(entry) = self.entries.get(&key) else { continue; };
            if entry.residency.host().is_some() || self.promotions.iter().any(|p| p.key == key) {
                continue;
            }
            if !self.make_promotion_room(engine, Some(requested))? { break; }
            self.start_promotion(engine, caps, key, true)?;
            if self.promotions.len() == self.promotion_depth {
                self.finish_promotions(engine, caps)?;
            }
        }
        self.finish_promotions(engine, caps)
    }

    fn read_entry(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, capacity: usize, requested: Option<&HashSet<ObjectKey>>)
        -> Result<Option<(Entry, bool)>, Error>
    {
        self.stats.gets += 1;
        let Some(initial) = self.entries.get(&key).copied() else {
            self.stats.misses += 1;
            return Ok(None);
        };
        if capacity < initial.len { return Err(Error::Capacity(initial.len)); }
        let needs_promotion = initial.residency.host().is_none();
        if needs_promotion { self.promote(engine, caps, key, requested)?; }
        Ok(Some((self.entries[&key], needs_promotion || initial.prefetched)))
    }

    fn get(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, dst: &mut [u8]) -> Result<Option<usize>, Error>
    {
        self.poll(engine, caps)?;
        let Some((entry, nvme_hit)) = self.read_entry(engine, caps, key, dst.len(), None)? else {
            return Ok(None);
        };
        let frame_size = engine.core.frames.frame_size;
        let target = if dst.len() >= frame_size {
            &mut dst[..frame_size]
        } else {
            self.scratch.resize(frame_size, 0);
            &mut self.scratch
        };
        let read = engine.get_into_host(
            Tracked(caps.instance.borrow()), Tracked(caps.shares.borrow_mut()), key, target,
        )?;
        if read.is_none() { return Err(Error::Backend); }
        if dst.len() < frame_size { dst[..entry.len].copy_from_slice(&self.scratch[..entry.len]); }
        self.touch(key, entry, nvme_hit);
        Ok(Some(entry.len))
    }

    #[cfg_attr(not(verus_only), allow(unused_variables, non_shorthand_field_patterns))]
    fn acquire(&mut self, engine: &mut LocalEngine, caps: &mut ShardCapabilityBundle,
        key: ObjectKey, requested: Option<&HashSet<ObjectKey>>) -> Result<Option<Lease>, Error>
    {
        let Some((mut entry, nvme_hit)) = self.read_entry(engine, caps, key, usize::MAX, requested)? else {
            return Ok(None);
        };
        let sequence = self.lease_seq.checked_add(1).ok_or(Error::Backend)?;
        let id = sequence.checked_mul(self.num_shards)
            .and_then(|base| base.checked_add(self.shard_index)).ok_or(Error::Backend)?;
        let HostLease { handle, cap, slice, .. } = engine.get_lease_host(
            Tracked(caps.instance.borrow()), Tracked(caps.shares.borrow_mut()), key,
        )?.ok_or(Error::Backend)?;
        debug_assert_eq!(entry.residency.host(), Some(handle.frame));
        self.lease_seq = sequence;
        self.leases.insert(id, ActiveLease { handle, cap, key });
        entry.pin_count += 1;
        self.touch(key, entry, nvme_hit);
        Ok(Some(Lease { id, address: slice.addr(), len: entry.len }))
    }

    #[cfg_attr(not(verus_only), allow(unused_variables, non_shorthand_field_patterns))]
    fn release(&mut self, engine: &mut LocalEngine, caps: &ShardCapabilityBundle,
        id: u64) -> Result<(), Error>
    {
        let ActiveLease { handle, cap, key } = self.leases.remove(&id).ok_or(Error::Invalid)?;
        let mut entry = self.entries[&key];
        debug_assert_eq!(entry.residency.host(), Some(handle.frame));
        engine.release_lease(Tracked(caps.instance.borrow()), handle, cap, Ticket::NONE)?;
        entry.pin_count -= 1;
        self.set(key, Some(entry));
        Ok(())
    }

    fn drain(&mut self, engine: &mut LocalEngine, caps: &ShardCapabilityBundle) -> Result<(), Error> {
        let ids: Vec<_> = self.leases.keys().copied().collect();
        for id in ids { self.release(engine, caps, id)?; }
        self.finish_promotions(engine, caps)?;
        while let Some(spill) = self.inflight_spills.first() {
            let ticket = spill.ticket;
            self.wait(engine, caps, ticket)?;
        }
        Ok(())
    }
}

} // verus!

impl Directory {
    fn touch(&mut self, key: ObjectKey, mut entry: Entry, nvme_hit: bool) {
        entry.lru_seq = self.next_sequence();
        entry.prefetched = false;
        if !nvme_hit {
            entry.probation = false;
        }
        self.set(key, Some(entry));
        self.stats.hits += 1;
        self.stats.bytes_read += entry.len as u64;
        if nvme_hit {
            self.extended.nvme_hits += 1;
        } else {
            self.extended.dram_hits += 1;
        }
    }
}

fn lock_file(path: &str) -> Result<File, Error> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
        .map_err(|_| Error::Backend)?;
    file.try_lock().map_err(|_| Error::Backend)?;
    Ok(file)
}

pub(super) fn object_key(key: u64) -> ObjectKey {
    let h1 = splitmix64(key ^ 0x9e37_79b9_7f4a_7c15);
    let h2 = splitmix64(h1 ^ 0xbf58_476d_1ce4_e5b9);
    let h3 = splitmix64(h2 ^ 0x94d0_49bb_1331_11eb);
    ObjectKey([key, h1, h2, h3])
}

fn splitmix64(mut key: u64) -> u64 {
    key = key.wrapping_add(0x9e37_79b9_7f4a_7c15);
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}
