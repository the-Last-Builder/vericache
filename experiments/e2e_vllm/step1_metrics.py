"""Local resource and cache telemetry for Step 1 replay and smoke tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import threading
import time

import psutil

GIB = 1 << 30
_STATS_SNAPSHOT = 0xFFFFFFFF
_STATS_DRAIN = 0xFFFFFFFE
_HEADER = struct.Struct("<I")
_METRIC = re.compile(r"^([\w:]+)(\{.*\})?\s+(\S+)(?:\s+\d+)?$")


def parse_prometheus(body: str) -> dict[str, float]:
    samples = {}
    for line in body.splitlines():
        match = _METRIC.fullmatch(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name.endswith(("_bucket", "_created")):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            samples[name + (labels or "")] = number
    return samples


def sum_metrics(samples: dict[str, float]) -> dict[str, float]:
    sums: dict[str, float] = {}
    for sample, value in samples.items():
        name = sample.split("{", 1)[0]
        sums[name] = sums.get(name, 0.0) + value
    return sums


def numeric_delta(before: dict, after: dict) -> dict:
    # A negative delta is useful evidence of a reset. Keep it visible.
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if type(after[key]) in (int, float) and type(before.get(key, 0)) in (int, float)
    }


def read_vericache(socket_path: str, *, drain: bool = False, timeout: float = 5) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        sock.sendall(_HEADER.pack(_STATS_DRAIN if drain else _STATS_SNAPSHOT))
        with sock.makefile("rb") as response:
            size, = _HEADER.unpack(response.read(_HEADER.size))
            if not 0 < size <= 1024 * 1024:
                raise ValueError(f"Invalid stats reply size from {socket_path}: {size}")
            snapshot = json.loads(response.read(size))
    if snapshot["worker_dead"]:
        raise RuntimeError(f"VeriCache worker is dead: {socket_path}")
    return snapshot


def read_hca(devices: list[str], root: Path = Path("/sys/class/infiniband")) -> dict[str, dict[str, int]]:
    results = {}
    for device in devices:
        port = root / device / "ports" / "1"
        counts = {}
        for group in ("counters", "hw_counters"):
            for path in (port / group).glob("*"):
                try:
                    counts[f"{group}/{path.name}"] = int(path.read_text().strip(), 0)
                except (OSError, ValueError):
                    continue
        if not all(f"counters/port_{direction}_data" in counts for direction in ("xmit", "rcv")):
            raise RuntimeError(f"RDMA byte counters are unavailable for {port}")
        # Standard IB port data counters are in 32-bit words, also on RoCE.
        for direction in ("xmit", "rcv"):
            counts[f"port_{direction}_bytes"] = 4 * counts[f"counters/port_{direction}_data"]
        results[device] = counts
    return results


def read_nvme_usage(directory: str, *, include_usage: bool = True) -> dict:
    path = Path(directory)
    fs_dev = path.stat().st_dev
    device = f"{os.major(fs_dev)}:{os.minor(fs_dev)}"
    stat_path = Path("/sys/dev/block") / device / "stat"
    result: dict = {"path": directory, "device": device}
    if stat_path.exists():
        fields = stat_path.read_text().split()
        result["device_name"] = stat_path.resolve().parent.name
        result["device_read_bytes"] = int(fields[2]) * 512
        result["device_write_bytes"] = int(fields[6]) * 512
    if include_usage:
        apparent = allocated = files = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    meta = item.stat()
                    apparent += meta.st_size
                    allocated += meta.st_blocks * 512
                    files += 1
            except FileNotFoundError:
                # Some backends replace spill files while sampling.
                continue
        result.update(apparent_bytes=apparent, allocated_bytes=allocated, files=files)
    return result


def read_pss_bytes(pid: int) -> int:
    with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as status:
        for line in status:
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    raise ValueError(f"No Pss field for PID {pid}")


class ResourceSampler:
    """Sample only the launched server trees, retaining CPU totals of reaped children."""

    def __init__(self, roots: list[int], gpu_ids: list[str], interval: float = 0.5):
        self.roots = [psutil.Process(pid) for pid in roots]
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="step1-server-resources", daemon=True)
        self.gpu_pool = ThreadPoolExecutor(max_workers=1) if gpu_ids else None
        self.gpu_sample = None
        self.next_gpu_sample = 0.0
        self.initial_cpu: dict[psutil.Process, float] = {}
        self.last_cpu: dict[psutil.Process, float] = {}
        self.initial_kernel_cpu = self.peak_kernel_cpu = 0.0
        self.peak_pss: int | None = None
        self.peak_rss = 0
        self.peak_gpu_bytes: dict[str, int] = {}
        self.sample_count = self.incomplete_pss_samples = 0

    def _sample_gpu(self) -> dict[str, int]:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", f"--id={','.join(self.gpu_ids)}",
                 "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            )
            return {idx.strip(): int(value.strip()) * (1 << 20)
                    for idx, value in (line.split(",") for line in output.splitlines())}
        except (OSError, subprocess.SubprocessError, ValueError):
            return {}

    def _collect_gpu(self) -> None:
        for gpu, used in self.gpu_sample.result().items():
            self.peak_gpu_bytes[gpu] = max(self.peak_gpu_bytes.get(gpu, 0), used)
        self.gpu_sample = None

    def sample(self, *, baseline: bool = False) -> None:
        live = set()
        for process in {*self.roots, *self.last_cpu}:
            try:
                if process.is_running():
                    live.add(process)
                    if process in self.roots:
                        live.update(process.children(recursive=True))
            except psutil.Error:
                continue
        rss = pss = 0
        kernel_cpu = 0.0
        measured = 0
        pss_complete = True
        for process in live:
            try:
                with process.oneshot():
                    cpu = process.cpu_times()
                    total = cpu.user + cpu.system
                    kernel_cpu += total + cpu.children_user + cpu.children_system
                    rss += process.memory_info().rss
                if baseline:
                    self.initial_cpu[process] = total
                self.last_cpu[process] = total
                measured += 1
                pss += read_pss_bytes(process.pid)
            except (OSError, ValueError, psutil.Error):
                pss_complete = False
        # Reaped child CPU moves into the parent's children_* clocks. These
        # counters capture work even for children that live between samples.
        if baseline:
            self.initial_kernel_cpu = kernel_cpu
        self.peak_kernel_cpu = max(self.peak_kernel_cpu, kernel_cpu)
        self.sample_count += 1
        self.peak_rss = max(self.peak_rss, rss)
        if pss_complete and measured:
            self.peak_pss = max(self.peak_pss or 0, pss)
        elif live:
            self.incomplete_pss_samples += 1
        if self.gpu_pool is not None:
            now = time.perf_counter()
            if self.gpu_sample is not None and self.gpu_sample.done():
                self._collect_gpu()
            if self.gpu_sample is None and now >= self.next_gpu_sample:
                self.gpu_sample = self.gpu_pool.submit(self._sample_gpu)
                self.next_gpu_sample = now + 2

    def _run(self) -> None:
        next_sample = time.perf_counter() + self.interval
        while not self.stop_event.wait(max(0, next_sample - time.perf_counter())):
            self.sample()
            next_sample = max(next_sample + self.interval, time.perf_counter())

    def start(self) -> None:
        self.sample(baseline=True)
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        self.thread.join()
        # The final CPU and memory boundary must not launch another GPU query.
        self.next_gpu_sample = math.inf
        self.sample()
        if self.gpu_pool is not None:
            self.gpu_pool.shutdown()
            if self.gpu_sample is not None:
                self._collect_gpu()
        observed_cpu = sum(max(0.0, cpu - self.initial_cpu.get(proc, 0.0))
                           for proc, cpu in self.last_cpu.items())
        inherited_cpu = self.peak_kernel_cpu - self.initial_kernel_cpu
        return {
            "peak_rss_bytes": self.peak_rss,
            "peak_pss_bytes": self.peak_pss,
            "peak_gpu_used_bytes": self.peak_gpu_bytes,
            "cpu_core_seconds": max(observed_cpu, inherited_cpu),
            "processes_seen": len(self.last_cpu),
            "sample_interval_s": self.interval,
            "samples": self.sample_count,
            "incomplete_pss_samples": self.incomplete_pss_samples,
        }
