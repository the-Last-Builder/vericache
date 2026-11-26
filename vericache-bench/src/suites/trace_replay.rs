use crate::policy::LruPolicy;
use crate::schema::{JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps};
use crate::shard::{
    ShardCapabilityBundle, ShardRouter, ensure_bench_node_allocator, key_from_hash_id,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};
use transfer::host::HostContext;
use vericache_core::backends::host::HostBackend;
use vericache_core::engine::LocalEngine;
use vericache_core::spec::{FrameId, ObjectKey};
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Tracked, verus};

#[derive(Debug, Deserialize)]
pub struct TraceRequest {
    pub timestamp: u64,
    #[allow(dead_code)]
    pub input_length: usize,
    #[allow(dead_code)]
    pub output_length: usize,
    pub hash_ids: Vec<u64>,
}

pub fn load_trace(path: &str, max_requests: Option<usize>) -> Vec<TraceRequest> {
    let f = File::open(path).unwrap_or_else(|e| panic!("failed to open trace {path}: {e}"));
    let reader = BufReader::new(f);
    let limit = max_requests.unwrap_or(usize::MAX);
    let mut reqs = Vec::new();
    for line in reader.lines() {
        if reqs.len() >= limit {
            break;
        }
        let line = line.expect("read trace line");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(req) = serde_json::from_str::<TraceRequest>(trimmed) {
            if !req.hash_ids.is_empty() {
                reqs.push(req);
            }
        }
    }
    assert!(!reqs.is_empty(), "trace {path} yielded 0 valid requests");
    reqs
}

fn box_host_as_static(host: &HostBackend) -> &'static HostBackend {
    let host_ptr: *const HostBackend = host;
    // SAFETY: `host` is boxed and owned by `OwnedHostShard`, and `OwnedHostShard::drop`
    // drops `engine` before dropping `host`.
    unsafe { &*host_ptr }
}

verus! {

#[cfg_attr(not(verus_only), allow(dead_code))]
struct OwnedHostShard {
    caps: ShardCapabilityBundle,
    host: Box<HostBackend>,
    engine: Option<LocalEngine<'static>>,
    lru: LruPolicy<ObjectKey, FrameId>,
}

impl OwnedHostShard {
    fn new(frame_size: usize, cap_frames: usize) -> Self {
        let mut caps = ShardCapabilityBundle::take(0);
        let mut core = caps.make_local_core(frame_size, 64);
        let host_src = caps.mint_host();
        let host = Box::new(
            HostBackend::new(
                &mut core,
                Tracked(caps.instance.borrow()),
                Tracked(caps.alloc.borrow_mut()),
                host_src,
                cap_frames as u32,
                4096,
            )
            .expect("HostBackend::new shard"),
        );
        let host_static: &'static HostBackend = box_host_as_static(&host);
        let eng_ctx = HostContext::new(caps.mint_host());
        let engine = LocalEngine::new(core, host_static, eng_ctx, None, None).expect("assemble local engine");
        Self {
            caps,
            host,
            engine: Some(engine),
            lru: LruPolicy::new(cap_frames),
        }
    }

    /// Try prefix-hit read from Host DRAM into `dst`, touching LRU on hit.
    fn get_hit(&mut self, key: ObjectKey, dst: &mut [u8]) -> bool {
        let eng = self.engine.as_mut().unwrap();
        match eng
            .get_into_host(
                Tracked(self.caps.instance.borrow()),
                Tracked(self.caps.shares.borrow_mut()),
                key,
                dst,
            )
            .expect("get_into_host")
        {
            Some(_ver) => {
                self.lru.touch(&key);
                true
            },
            None => false,
        }
    }

    /// Writeback a miss block into Host DRAM, evicting & reclaiming the LRU victim if full.
    fn put_with_lru(&mut self, key: ObjectKey, data: &[u8]) {
        let host_tier = self.engine.as_ref().unwrap().host_tier();
        // If key already somehow present (e.g. duplicate hash_id inside a single request suffix), touch & return.
        if self.lru.contains(&key) {
            self.lru.touch(&key);
            return;
        }
        if self.lru.is_full() {
            if let Some((_vic_key, vic_frame)) = self.lru.pop_lru() {
                let eng = self.engine.as_mut().unwrap();
                eng.evict_frame(vic_frame).expect("evict lru victim");
                let reclaimed = eng.reclaim_draining(host_tier, 8).unwrap();
                assert!(reclaimed >= 1, "expected lru victim reclaim");
            }
        }
        let eng = self.engine.as_mut().unwrap();
        let (_ver, frame) = eng
            .put_host(Tracked(self.caps.instance.borrow()), key, data)
            .expect("put_host writeback");
        let _ = self.lru.insert(key, frame);
    }
}

} // verus!

impl Drop for OwnedHostShard {
    fn drop(&mut self) {
        // Drop engine (holding &'static HostBackend) BEFORE dropping the boxed HostBackend.
        self.engine.take();
        let _ = &self.host;
    }
}

pub fn run(job: &JobSpec) -> Vec<MetricRecord> {
    let trace_path = job
        .trace_path
        .as_deref()
        .expect("s3_trace_replay requires trace_path");
    let reqs = load_trace(trace_path, job.max_requests);

    let mut unique_blocks = HashSet::new();
    let mut total_blocks_in_trace = 0usize;
    for r in &reqs {
        for &h in &r.hash_ids {
            unique_blocks.insert(h);
        }
        total_blocks_in_trace += r.hash_ids.len();
    }

    let frame_size = job.chunk_bytes;
    let ratio = job.capacity_ratio.unwrap_or(0.30);
    let total_cap_frames = job.capacity_frames.unwrap_or_else(|| {
        let target = ((unique_blocks.len() as f64) * ratio).round() as usize;
        // Clamp memory footprint safely (max ~2 GiB across host slabs so 4 MB sweeps stay fast & lean).
        let max_by_mem = (2usize * 1024 * 1024 * 1024) / frame_size;
        target.max(16).min(max_by_mem.max(16))
    });

    ensure_bench_node_allocator(0);
    let num_shards = job
        .num_shards
        .unwrap_or(4)
        .max(1)
        .min(total_cap_frames / 4)
        .max(1);
    let per_shard_cap = (total_cap_frames + num_shards - 1) / num_shards;

    let mut shards = Vec::with_capacity(num_shards);
    for _s in 0..num_shards {
        shards.push(OwnedHostShard::new(frame_size, per_shard_cap));
    }
    let mut router = ShardRouter::new(shards);

    let mut read_buf = vec![0u8; frame_size];
    let write_buf: Vec<u8> = (0..frame_size)
        .map(|i| ((i.wrapping_mul(29) + 7) & 0xff) as u8)
        .collect();

    let is_open_loop = job.mode == "open_loop";
    let time_scale = job.time_scale.unwrap_or(1000.0).max(1.0);
    let base_ts = reqs[0].timestamp;

    let mut hist = LatencyHist::new();
    let sampler = ProcSampler::start();
    let run_start = Instant::now();

    let mut hit_blocks = 0u64;
    let mut miss_blocks = 0u64;

    for req in &reqs {
        if is_open_loop {
            // Pace arrival according to trace timestamp (ms) / time_scale.
            let rel_ms = req.timestamp.saturating_sub(base_ts) as f64;
            let target_offset = Duration::from_secs_f64((rel_ms / 1000.0) / time_scale);
            let now_off = run_start.elapsed();
            if target_offset > now_off {
                let wait = target_offset - now_off;
                if wait > Duration::from_micros(50) {
                    std::thread::sleep(wait);
                } else {
                    while run_start.elapsed() < target_offset {
                        std::hint::spin_loop();
                    }
                }
            }
        }

        let req_t0 = Instant::now();
        // Prefix-match scan: hits while contiguous prefix matches; on first miss, writeback remaining suffix.
        let mut prefix_open = true;
        for &hid in &req.hash_ids {
            let key = key_from_hash_id(hid);
            let shard = router.shard_for_mut(&key);
            if prefix_open && shard.get_hit(key, &mut read_buf) {
                hit_blocks += 1;
            } else {
                prefix_open = false;
                miss_blocks += 1;
                shard.put_with_lru(key, &write_buf);
            }
        }
        let req_ns = req_t0.elapsed().as_nanos() as u64;
        hist.record_ns(req_ns);
    }

    let elapsed_s = run_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_blocks = hit_blocks + miss_blocks;
    let hit_rate = if total_blocks > 0 {
        (hit_blocks as f64) / (total_blocks as f64)
    } else {
        0.0
    };
    let total_bytes = total_blocks * (frame_size as u64);
    let hit_bytes = hit_blocks * (frame_size as u64);
    let throughput_gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);
    // Goodput = hit KV blocks served directly from cache + valid writebacks (report total valid KV throughput,
    // and note exact hit-only GB/s in notes so both metrics are crystal clear!).
    let goodput_gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);
    let hit_gbps = bytes_per_sec_to_gbps(hit_bytes, elapsed_s);

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.num_blocks = total_cap_frames;
    rec.throughput_gbps = throughput_gbps;
    rec.goodput_gbps = goodput_gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = hit_rate;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = format!(
        "reqs={}, unique_blks={}, total_blks={}, cap_frames={} (ratio={:.2}), shards={}, hit_only_gbps={:.2}",
        reqs.len(),
        unique_blocks.len(),
        total_blocks_in_trace,
        total_cap_frames,
        ratio,
        num_shards,
        hit_gbps
    );
    vec![rec]
}
