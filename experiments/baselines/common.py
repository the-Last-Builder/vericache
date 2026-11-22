"""Shared schema, timing, checksum, LRU, and trace utilities for Python baselines."""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import time
from typing import Any

import numpy as np
import psutil

FRAME_MAGIC = np.uint64(0x5643414348453034)


@dataclasses.dataclass
class MetricRecord:
    system: str
    suite: str
    case: str
    mode: str
    chunk_bytes: int
    num_blocks: int
    concurrency: int
    throughput_gbps: float = 0.0
    goodput_gbps: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    hit_rate: float = 0.0
    cpu_pct: float = 0.0
    rss_mb: float = 0.0
    corrupted_frames: int = 0
    recovery_us: float = 0.0
    notes: str = ""

    @classmethod
    def from_job(cls, system: str, job: dict[str, Any]) -> "MetricRecord":
        return cls(
            system=system,
            suite=str(job["suite"]),
            case=str(job["case"]),
            mode=str(job["mode"]),
            chunk_bytes=int(job["chunk_bytes"]),
            num_blocks=int(job.get("num_blocks", 16)),
            concurrency=int(job.get("concurrency", 1)),
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))


class LatencyRecorder:
    def __init__(self) -> None:
        self._samples_ns: list[int] = []

    def record_ns(self, ns: int) -> None:
        self._samples_ns.append(max(1, int(ns)))

    def extend_ns(self, samples: list[int]) -> None:
        self._samples_ns.extend(samples)

    @property
    def samples_ns(self) -> list[int]:
        return self._samples_ns

    def percentiles_us(self) -> tuple[float, float, float]:
        if not self._samples_ns:
            return (0.0, 0.0, 0.0)
        arr = np.asarray(self._samples_ns, dtype=np.float64) / 1_000.0
        p50, p95, p99 = np.percentile(arr, [50.0, 95.0, 99.0])
        return (float(p50), float(p95), float(p99))

    def percentiles_us_full(self) -> tuple[float, float, float, float]:
        if not self._samples_ns:
            return (0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(self._samples_ns, dtype=np.float64) / 1_000.0
        p50, p95, p99, p999 = np.percentile(arr, [50.0, 95.0, 99.0, 99.9])
        return (float(p50), float(p95), float(p99), float(p999))

    def mean_us(self) -> float:
        if not self._samples_ns:
            return 0.0
        return float(np.mean(self._samples_ns) / 1_000.0)


class ProcSampler:
    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())
        self._start_wall = time.perf_counter()
        ct = self._proc.cpu_times()
        self._start_cpu = ct.user + ct.system

    def finish(self) -> tuple[float, float]:
        wall = max(1e-9, time.perf_counter() - self._start_wall)
        ct = self._proc.cpu_times()
        cpu_s = max(0.0, (ct.user + ct.system) - self._start_cpu)
        cpu_pct = (cpu_s / wall) * 100.0
        rss_mb = self._proc.memory_info().rss / (1024.0 * 1024.0)
        return (cpu_pct, rss_mb)


def bytes_per_sec_to_gbps(total_bytes: int, elapsed_s: float) -> float:
    if elapsed_s <= 0.0:
        return 0.0
    return float(total_bytes) / (elapsed_s * 1e9)


def load_trace(path: str, max_requests: int | None = None) -> list[dict[str, Any]]:
    reqs: list[dict[str, Any]] = []
    limit = max_requests if max_requests is not None else 10**9
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(reqs) >= limit:
                break
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if obj.get("hash_ids"):
                reqs.append(obj)
    if not reqs:
        raise RuntimeError(f"trace {path} produced 0 valid requests")
    return reqs


class LruTracker:
    """O(1) LRU tracker backed by OrderedDict."""

    def __init__(self, cap: int) -> None:
        self.cap = max(1, cap)
        self._od: collections.OrderedDict[int, Any] = collections.OrderedDict()

    def __len__(self) -> int:
        return len(self._od)

    def __contains__(self, key: int) -> bool:
        return key in self._od

    def is_full(self) -> bool:
        return len(self._od) >= self.cap

    def touch(self, key: int) -> Any | None:
        if key not in self._od:
            return None
        self._od.move_to_end(key, last=True)
        return self._od[key]

    def insert(self, key: int, val: Any) -> tuple[int, Any] | None:
        if key in self._od:
            self._od[key] = val
            self._od.move_to_end(key, last=True)
            return None
        evicted = None
        if len(self._od) >= self.cap:
            evicted = self._od.popitem(last=False)
        self._od[key] = val
        return evicted

    def pop_lru(self) -> tuple[int, Any] | None:
        if not self._od:
            return None
        return self._od.popitem(last=False)

    def remove(self, key: int) -> Any | None:
        return self._od.pop(key, None)


def stamp_checksummed_array(arr_u64: np.ndarray, hash_id: int, version_seq: int) -> None:
    """Stamp a uint64 view of a frame with (magic, hash_id, version, patterned body, xor-sum)."""
    n = arr_u64.shape[0]
    assert n >= 4
    arr_u64[0] = FRAME_MAGIC
    arr_u64[1] = np.uint64(hash_id & 0xFFFFFFFFFFFFFFFF)
    arr_u64[2] = np.uint64(version_seq & 0xFFFFFFFFFFFFFFFF)
    fill_val = np.uint64(((hash_id * 0x9E3779B97F4A7C15) ^ (version_seq * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF)
    arr_u64[3 : n - 1] = fill_val
    # Fast checksum mixing header + body endpoints + stride sample + full XOR
    csum = np.bitwise_xor.reduce(arr_u64[: n - 1])
    arr_u64[n - 1] = csum


def verify_checksummed_array(arr_u64: np.ndarray, expected_hash_id: int | None = None) -> tuple[bool, str]:
    n = arr_u64.shape[0]
    if n < 4:
        return (False, "short")
    if arr_u64[0] != FRAME_MAGIC:
        return (False, "bad_magic")
    hid = int(arr_u64[1])
    if expected_hash_id is not None and hid != (expected_hash_id & 0xFFFFFFFFFFFFFFFF):
        return (False, "hash_id_mismatch")
    ver = int(arr_u64[2])
    expected_fill = np.uint64(((hid * 0x9E3779B97F4A7C15) ^ (ver * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF)
    # Check both body uniformity (catches partial-frame tearing!) and trailing checksum
    if not bool(np.all(arr_u64[3 : n - 1] == expected_fill)):
        return (False, "torn_body")
    actual = np.bitwise_xor.reduce(arr_u64[: n - 1])
    if arr_u64[n - 1] != actual:
        return (False, "checksum_mismatch")
    return (True, "ok")


def should_revoke(op_idx: int, revoke_rate: float) -> bool:
    if revoke_rate <= 0.0:
        return False
    period = max(1, int(round(1.0 / revoke_rate)))
    return ((op_idx + 1) % period) == 0


def splitmix64(x: int) -> int:
    """Exact bit-for-bit match with Rust `vericache-bench/src/shard.rs::splitmix64`."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def route_request_scaleup(
    req_idx: int,
    hash_ids: list[int],
    shards_per_node: int,
    sticky_pct: int,
) -> tuple[int, int]:
    """Deterministic 2-node 70/30 session-sticky routing + 8-shard affinity.
    Exact bit-for-bit match with `vericache-bench/src/suites/scaleup_cluster.rs`."""
    pos = 12 if len(hash_ids) > 12 else len(hash_ids) - 1
    sk = int(hash_ids[pos]) & 0xFFFFFFFFFFFFFFFF
    h = splitmix64(sk)
    home_node = h & 1
    shard_id = (h >> 1) % shards_per_node
    drift_hash = splitmix64(
        (req_idx & 0xFFFFFFFFFFFFFFFF) ^ ((sk * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    )
    node_id = home_node if (drift_hash % 100) < sticky_pct else (1 - home_node)
    return (int(node_id), int(shard_id))


def scaleup_config_from_job(job: dict[str, Any]) -> dict[str, Any]:
    frame_size = int(job["chunk_bytes"])
    shards = max(1, int(job.get("shards_per_node", 8)))
    hbm_gb = float(job.get("hbm_gb_per_node", 50.0))
    dram_gb = float(job.get("dram_gb_per_node", 200.0))
    nvme_gb = float(job.get("nvme_gb_per_node", 50.0))
    sticky_pct = max(0, min(100, int(round(float(job.get("sticky_ratio", 0.70)) * 100.0))))

    def _gb_to_frames(gb: float) -> int:
        total_bytes = int(round(gb * (1024.0**3)))
        total_frames = total_bytes // frame_size
        return max(4, total_frames // shards)

    return {
        "frame_size": frame_size,
        "shards_per_node": shards,
        "hbm_frames_per_shard": _gb_to_frames(hbm_gb),
        "dram_frames_per_shard": _gb_to_frames(dram_gb),
        "nvme_frames_per_shard": _gb_to_frames(nvme_gb),
        "hbm_gb_per_node": hbm_gb,
        "dram_gb_per_node": dram_gb,
        "nvme_gb_per_node": nvme_gb,
        "sticky_pct": sticky_pct,
    }
