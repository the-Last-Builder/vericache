use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

fn default_num_blocks() -> usize {
    16
}
fn default_concurrency() -> usize {
    1
}
fn default_warmup() -> usize {
    16
}
fn default_iters() -> usize {
    128
}
fn default_nvme_dir() -> String {
    "/workspace/target/bench-nvme".to_string()
}
fn default_rdma_dev() -> String {
    std::env::var("VERICACHE_RDMA_DEVICE")
        .or_else(|_| std::env::var("VERICACHE_RDMA_DEV"))
        .unwrap_or_else(|_| "mlx5_3".to_string())
}
fn default_gid_index() -> u8 {
    std::env::var("VERICACHE_RDMA_GID_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}
fn default_init_gpu() -> u32 {
    1
}

/// Unified JSON JobSpec passed from `experiments/harness/orchestrate.py` to
/// `vericache-bench` and Python baselines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    pub suite: String,
    pub case: String,
    pub mode: String,
    pub chunk_bytes: usize,
    #[serde(default = "default_num_blocks")]
    pub num_blocks: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_warmup")]
    pub warmup_iters: usize,
    #[serde(default = "default_iters")]
    pub iters: usize,
    #[serde(default)]
    pub trace_path: Option<String>,
    #[serde(default)]
    pub max_requests: Option<usize>,
    #[serde(default)]
    pub capacity_ratio: Option<f64>,
    #[serde(default)]
    pub capacity_frames: Option<usize>,
    #[serde(default)]
    pub revoke_rate: Option<f64>,
    #[serde(default)]
    pub num_shards: Option<usize>,
    #[serde(default = "default_nvme_dir")]
    pub nvme_dir: String,
    #[serde(default)]
    pub nvme_dir_node0: Option<String>,
    #[serde(default)]
    pub nvme_dir_node1: Option<String>,
    #[serde(default = "default_rdma_dev")]
    pub rdma_dev: String,
    #[serde(default = "default_gid_index")]
    pub gid_index: u8,
    #[serde(default)]
    pub donor_gpu: u32,
    #[serde(default = "default_init_gpu")]
    pub initiator_gpu: u32,
    #[serde(default)]
    pub time_scale: Option<f64>,
    #[serde(default)]
    pub hbm_gb_per_node: Option<f64>,
    #[serde(default)]
    pub dram_gb_per_node: Option<f64>,
    #[serde(default)]
    pub nvme_gb_per_node: Option<f64>,
    #[serde(default)]
    pub shards_per_node: Option<usize>,
    #[serde(default)]
    pub sticky_ratio: Option<f64>,
}

impl JobSpec {
    pub fn nvme_dir_for_node(&self, node_id: u64) -> &str {
        match node_id {
            0 => self.nvme_dir_node0.as_deref().unwrap_or(&self.nvme_dir),
            1 => self.nvme_dir_node1.as_deref().unwrap_or(&self.nvme_dir),
            _ => &self.nvme_dir,
        }
    }
}

/// Unified JSONL MetricRecord emitted by VeriCache, Mooncake, and LMCache runners.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub system: String,
    pub suite: String,
    pub case: String,
    pub mode: String,
    pub chunk_bytes: usize,
    pub num_blocks: usize,
    pub concurrency: usize,
    pub throughput_gbps: f64,
    pub goodput_gbps: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub hit_rate: f64,
    pub cpu_pct: f64,
    pub rss_mb: f64,
    pub corrupted_frames: u64,
    pub recovery_us: f64,
    pub notes: String,
}

impl MetricRecord {
    pub fn from_job(system: &str, job: &JobSpec) -> Self {
        Self {
            system: system.to_string(),
            suite: job.suite.clone(),
            case: job.case.clone(),
            mode: job.mode.clone(),
            chunk_bytes: job.chunk_bytes,
            num_blocks: job.num_blocks,
            concurrency: job.concurrency,
            throughput_gbps: 0.0,
            goodput_gbps: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
            hit_rate: 0.0,
            cpu_pct: 0.0,
            rss_mb: 0.0,
            corrupted_frames: 0,
            recovery_us: 0.0,
            notes: String::new(),
        }
    }
}

/// Nanosecond-resolution latency histogram backed by `hdrhistogram`.
pub struct LatencyHist {
    hist: Histogram<u64>,
}

impl LatencyHist {
    pub fn new() -> Self {
        let hist =
            Histogram::<u64>::new_with_bounds(1, 120_000_000_000, 3).expect("create hdrhistogram");
        Self { hist }
    }

    #[inline]
    pub fn record_ns(&mut self, ns: u64) {
        let clamped = ns.clamp(1, 120_000_000_000);
        let _ = self.hist.record(clamped);
    }

    pub fn p50_us(&self) -> f64 {
        if self.hist.is_empty() {
            return 0.0;
        }
        self.hist.value_at_quantile(0.50) as f64 / 1_000.0
    }

    pub fn p95_us(&self) -> f64 {
        if self.hist.is_empty() {
            return 0.0;
        }
        self.hist.value_at_quantile(0.95) as f64 / 1_000.0
    }

    pub fn p99_us(&self) -> f64 {
        if self.hist.is_empty() {
            return 0.0;
        }
        self.hist.value_at_quantile(0.99) as f64 / 1_000.0
    }

    pub fn p999_us(&self) -> f64 {
        if self.hist.is_empty() {
            return 0.0;
        }
        self.hist.value_at_quantile(0.999) as f64 / 1_000.0
    }

    pub fn merge(&mut self, other: &LatencyHist) {
        let _ = self.hist.add(&other.hist);
    }

    pub fn export_buckets(&self) -> Vec<(u64, u64)> {
        self.hist
            .iter_recorded()
            .map(|v| (v.value_iterated_to(), v.count_at_value()))
            .collect()
    }

    pub fn merge_buckets(&mut self, buckets: &[(u64, u64)]) {
        for &(val, cnt) in buckets {
            if cnt > 0 {
                let clamped = val.clamp(1, 120_000_000_000);
                let _ = self.hist.record_n(clamped, cnt);
            }
        }
    }

    pub fn mean_us(&self) -> f64 {
        if self.hist.is_empty() {
            return 0.0;
        }
        self.hist.mean() / 1_000.0
    }
}

/// Process CPU utilization and RSS sampler using Linux `/proc/self/{schedstat,stat,status}`.
pub struct ProcSampler {
    start_wall: Instant,
    start_cpu_ns: u64,
}

impl ProcSampler {
    pub fn start() -> Self {
        Self {
            start_wall: Instant::now(),
            start_cpu_ns: read_proc_cpu_ns(),
        }
    }

    pub fn finish(&self) -> (f64, f64) {
        let wall_ns = self.start_wall.elapsed().as_nanos().max(1) as f64;
        let end_cpu_ns = read_proc_cpu_ns();
        let cpu_ns = end_cpu_ns.saturating_sub(self.start_cpu_ns) as f64;
        let cpu_pct = (cpu_ns / wall_ns) * 100.0;
        let rss_mb = read_proc_rss_mb();
        (cpu_pct, rss_mb)
    }
}

fn read_proc_cpu_ns() -> u64 {
    // Use `/proc/self/stat` (fields 14+15: utime + stime across ALL threads in the process,
    // matching `psutil.Process.cpu_times()`), falling back to `/proc/self/schedstat`.
    if let Ok(stat) = fs::read_to_string("/proc/self/stat") {
        if let Some(rparen) = stat.rfind(')') {
            let fields: Vec<&str> = stat[rparen + 2..].split_whitespace().collect();
            if fields.len() > 12 {
                let utime: u64 = fields[11].parse().unwrap_or(0);
                let stime: u64 = fields[12].parse().unwrap_or(0);
                return utime.saturating_add(stime).saturating_mul(10_000_000); // USER_HZ=100 -> 10ms/tick
            }
        }
    }
    if let Ok(sched) = fs::read_to_string("/proc/self/schedstat") {
        if let Some(first) = sched.split_whitespace().next() {
            if let Ok(ns) = first.parse::<u64>() {
                return ns;
            }
        }
    }
    0
}

pub fn read_proc_rss_mb() -> f64 {
    read_proc_status_kb("VmRSS:") / 1024.0
}

pub fn read_proc_hwm_mb() -> f64 {
    read_proc_status_kb("VmHWM:") / 1024.0
}

fn read_proc_status_kb(prefix: &str) -> f64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            let kb: f64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            return kb;
        }
    }
    0.0
}

/// Convert total bytes over elapsed seconds into GB/s (decimal 10^9 B/s, standard for RDMA/NVMe/PCIe).
#[inline]
pub fn bytes_per_sec_to_gbps(bytes: u64, elapsed_secs: f64) -> f64 {
    if elapsed_secs <= 0.0 {
        0.0
    } else {
        (bytes as f64) / (elapsed_secs * 1e9)
    }
}
