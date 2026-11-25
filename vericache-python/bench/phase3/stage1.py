"""Run one Phase 3 single-GPU serving comparison point."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
from threading import Event, Thread
import time
from urllib.request import urlopen

import psutil
from prometheus_client.parser import text_string_to_metric_families

from prepare_traces import TRACES, read_rows
from prepare_routed_traces import ROUTED_TRACES
from prepare_bailian_trace_a import TRACE_A
from stage0_correctness import MODEL, PHASE3
from stage0_layout import model_command, reserve_port, start_server, write_manifest


GIB = 1 << 30
BLOCK_TOKENS = 64
LAYERS = 48
FRAME_BYTES = 128 << 10
KV_BLOCK_BYTES = LAYERS * FRAME_BYTES
HBM_BLOCKS = 3328
DEFAULT_WARMUP_END_MS = 12000
DEFAULT_REQUEST_TIMEOUT_SECS = 300
SYSTEMS = ("vanilla", "offloading", "lmcache", "mooncake", "vericache")
ABLATIONS = {
    "p1_sync": (False, False, 0, 1, False, 0),
    "zero_copy_lease": (True, False, 0, 1, False, 0),
    "async_pipeline": (True, True, 2, 1, False, 0),
    "qd4_probation": (True, True, 2, 4, True, 100),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_timestamp_ms(row, trace_type):
    ts = row["timestamp"]
    return int(ts) if trace_type == "mooncake" else int(float(ts) * 1000.0)


def capacity(job):
    profile, system = job["profile"], job["system"]
    assert profile in ("A", "B") and system in SYSTEMS
    if system == "vanilla":
        return 0, 0
    dram = (136 if profile == "A" else 36) * GIB // KV_BLOCK_BYTES
    nvme = (0 if profile == "A" else 80) * GIB // KV_BLOCK_BYTES
    if system == "offloading":
        return dram + nvme, 0
    return dram, nvme


def budgets(dram, nvme):
    return {
        tier: {"blocks": blocks, "bytes": blocks * KV_BLOCK_BYTES,
               "gib": blocks * KV_BLOCK_BYTES / GIB, "recoverable_tokens": blocks * BLOCK_TOKENS}
        for tier, blocks in (("hbm", HBM_BLOCKS), ("dram", dram), ("nvme", nvme))
    }


DEFAULT_NVME_ROOT = Path("/data/nvme3/zxw/vericache_phase3_nvme")


def resolve_nvme_run_dir(run: Path) -> Path:
    override = os.environ.get("PHASE3_NVME_ROOT")
    base = Path(override) if override else DEFAULT_NVME_ROOT
    if base.is_dir() and os.access(base, os.W_OK | os.X_OK):
        target = base / run.name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return run


def _rdma_device_for_numa(out: Path) -> str:
    numa_file = out / "numa-node"
    if numa_file.is_file():
        numa = int(numa_file.read_text().strip())
    else:
        gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        numa = gpu // 4
    return "mlx5_1" if numa == 0 else "mlx5_3"


@contextmanager
def maybe_start_mooncake_master(job, nvme_run: Path, out: Path):
    if job["system"] != "mooncake":
        yield None, None
        return
    _, nvme = capacity(job)
    rpc_port = reserve_port()
    metrics_port = reserve_port()
    argv = [
        "mooncake_master",
        "--rpc_port", str(rpc_port),
        "--metrics_port", str(metrics_port),
        "--enable_http_metadata_server=false",
        "--default_kv_lease_ttl=500",
        "--eviction_high_watermark_ratio=0.9",
        "--eviction_ratio=0.1",
    ]
    if nvme:
        mc_disk = nvme_run / "mooncake-nvme"
        mc_disk.mkdir(parents=True, exist_ok=True)
        argv.extend([
            "--enable_offload=true",
            "--offload_on_evict=true",
            "--promotion_on_hit=true",
            "--root_fs_dir", str(mc_disk),
            "--quota_bytes", str(int(nvme * KV_BLOCK_BYTES)),
        ])
    with (out / "mooncake_master.log").open("w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15.0
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", rpc_port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"mooncake_master did not start; see {log.name}")
            yield rpc_port, metrics_port
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)


def serving_options(job, run, out, port, *, mooncake_rpc_port=None, nvme_run=None):
    if nvme_run is None:
        nvme_run = run
    dram, nvme = capacity(job)
    system = job["system"]
    tp = int(job.get("tensor_parallel_size", 1))
    max_num_seqs = int(job.get("max_num_seqs", 16))
    assert tp == len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert tp == 1 or (tp == 2 and system == "vanilla")
    settings = budgets(dram, nvme)
    transfer = {"kv_role": "kv_both"}
    if system == "offloading":
        transfer.update({
            "kv_connector": "OffloadingConnector",
            "kv_connector_extra_config": {"num_cpu_blocks": dram, "block_size": BLOCK_TOKENS},
        })
    elif system == "lmcache":
        transfer["kv_connector"] = "LMCacheConnectorV1"
        profile = {"chunk_size": BLOCK_TOKENS, "local_cpu": True,
                   "max_local_cpu_size": settings["dram"]["gib"]}
        if nvme:
            lmc_disk = nvme_run / "lmcache-nvme"
            lmc_disk.mkdir(parents=True, exist_ok=True)
            profile.update({"local_disk": str(lmc_disk),
                            "max_local_disk_size": settings["nvme"]["gib"],
                            "extra_config": {"use_odirect": True}})
        config = out / "lmcache.yaml"
        config.write_text(json.dumps(profile, indent=2) + "\n")  # JSON is valid YAML.
        os.environ["LMCACHE_CONFIG_FILE"] = str(config)
        # LMCache places ZMQ IPC sockets under TMPDIR; keep path < 107-byte sun_path limit.
        lmc_tmp = Path("/tmp") / f"vc_lmc_{os.getpid()}"
        lmc_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(lmc_tmp)
    elif system == "mooncake":
        assert mooncake_rpc_port is not None
        transfer["kv_connector"] = "LMCacheConnectorV1"
        staging_gib = 4.0
        align_16m = 16 << 20
        seg_bytes = max(64 << 20, int((settings["dram"]["gib"] - staging_gib) * GIB))
        seg_bytes = (seg_bytes // align_16m) * align_16m
        rdma_dev = _rdma_device_for_numa(out)
        mc_extra = {
            "save_chunk_meta": False,
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": seg_bytes,
            "local_buffer_size": 64 << 20,
            "protocol": "rdma",
            "device_name": rdma_dev,
            "master_server_address": f"127.0.0.1:{mooncake_rpc_port}",
            "transfer_timeout": 15,
            "mooncake_prefer_local_alloc": True,
        }
        if nvme:
            mc_disk = nvme_run / "mooncake-nvme"
            mc_disk.mkdir(parents=True, exist_ok=True)
            mc_extra["storage_root_dir"] = str(mc_disk)
        profile = {
            "chunk_size": BLOCK_TOKENS,
            "local_cpu": False,
            "max_local_cpu_size": staging_gib,
            "remote_url": f"mooncakestore://127.0.0.1:{mooncake_rpc_port}/",
            "remote_serde": "naive",
            "extra_config": mc_extra,
        }
        config = out / "lmcache.yaml"
        config.write_text(json.dumps(profile, indent=2) + "\n")
        os.environ["LMCACHE_CONFIG_FILE"] = str(config)
        os.environ["MC_STORE_MEMCPY"] = "1" if job.get("mooncake_memcpy", True) else "0"
        lmc_tmp = Path("/tmp") / f"vc_lmc_{os.getpid()}"
        lmc_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(lmc_tmp)
    elif system == "vericache":
        shards = 4
        assert LAYERS % shards == 0
        cfg = {
            "num_shards": shards,
            "host_frames_per_shard": dram * LAYERS // shards,
            "nvme_frames_per_shard": nvme * LAYERS // shards,
            "frame_size": FRAME_BYTES,
            "prefer_cuda_pinned": True,
            "io_workers": 4,
            "staging_pool_size": 16,
        }
        if nvme:
            cfg["nvme_path"] = str(nvme_run / "vericache-nvme")
            cfg["lib_path"] = str(
                PHASE3 / "vericache-python/target/phase3/libvericache_python.so"
            )
            if not Path(cfg["lib_path"]).is_file():
                raise RuntimeError("the Phase 3 NVMe instrumentation library is missing")
            assert 64 * GIB <= cfg["nvme_frames_per_shard"] * shards * FRAME_BYTES <= 80 * GIB
        if "ablation" in job:
            assert nvme > 0
            zero_copy, async_pipeline, workers, depth, probation, watermark = ABLATIONS[job["ablation"]]
            cfg.update({"use_zero_copy_load": zero_copy, "enable_async_pipeline": async_pipeline,
                        "io_workers": workers})
            os.environ["PHASE3_NVME_POLICY"] = json.dumps([depth, probation, watermark])
        transfer.update({
            "kv_connector": "MeasuredVeriCacheV1Connector",
            "kv_connector_module_path": "stage1_connector",
            "kv_connector_extra_config": cfg,
        })
    argv = model_command(
        MODEL, BLOCK_TOKENS, port, "vanilla",
        tensor_parallel_size=tp, max_num_seqs=max_num_seqs,
    )
    argv.extend(["--num-gpu-blocks-override", str(HBM_BLOCKS)])
    if os.environ.get("VLLM_BATCH_INVARIANT") == "0":
        argv.extend(["--attention-backend", "FLASH_ATTN"])
    if system != "vanilla":
        argv.extend(["--kv-transfer-config", json.dumps(transfer)])
    return argv, settings


def trace_source(name):
    if name == TRACE_A:
        return "bailian", PHASE3 / "traces" / TRACE_A / "frozen"
    if name in ROUTED_TRACES:
        return "mooncake", PHASE3 / "traces/routed_frozen"
    return "mooncake", PHASE3 / "traces/frozen"


def select_cohort(rows, trace_type, begin_ms, end_ms):
    cohort = []
    for idx, row in enumerate(rows):
        ts_ms = row_timestamp_ms(row, trace_type)
        if begin_ms is not None and ts_ms < begin_ms:
            continue
        if end_ms is not None and ts_ms > end_ms:
            continue
        cohort.append((idx, ts_ms, row))
    return cohort


def replay_options(endpoint, out, name, scale, *, warmup=False,
                   warmup_end_ms=DEFAULT_WARMUP_END_MS,
                   measurement_end_ms=None,
                   request_timeout_secs=DEFAULT_REQUEST_TIMEOUT_SECS):
    trace_type, frozen = trace_source(name)
    dataset = frozen / f"{name}.jsonl"
    prompt_cache = dataset.with_suffix(".prompts.bin")
    rows = read_rows(dataset)
    mode = "feedback" if scale == "feedback" else "trace-replay"

    begin_ms = None if warmup else (warmup_end_ms + 1 if warmup_end_ms is not None else None)
    end_ms = warmup_end_ms if warmup else measurement_end_ms
    cohort = select_cohort(rows, trace_type, begin_ms, end_ms)
    if not cohort:
        raise ValueError(f"Empty cohort for {name} (warmup={warmup}, begin={begin_ms}, end={end_ms})")

    if mode == "feedback":
        dispatch_secs = 60.0
        run_seconds = 60
        t_dispatch = 60.0
    else:
        scale_f = float(scale)
        assert math.isfinite(scale_f) and scale_f > 0.0
        origin_ms = begin_ms or 0
        dispatch_secs = max(1.0, (cohort[-1][1] - origin_ms) / 1000.0 / scale_f)
        t_dispatch = max(1e-3, (cohort[-1][1] - cohort[0][1]) / 1000.0 / scale_f)
        run_seconds = math.ceil(dispatch_secs) + 30

    watchdog_secs = math.ceil(dispatch_secs) + int(request_timeout_secs) + 90

    argv = [
        str(PHASE3 / "bin/request-sim"),
        "--dataset", trace_type, "--dataset-path", str(dataset),
        "--api", "openai", "--endpoint", endpoint + "/v1/chat/completions",
        "--model-name", "Qwen3-30B-A3B",
        "--tokenizer", str(MODEL / "tokenizer.json"),
        "--tokenizer-config", str(MODEL / "tokenizer_config.json"),
        "--cache", "file", "--cache-path", str(prompt_cache),
        "--stream", "--ignore-eos", "--disable-thinking", "--context-length", "32768",
        "--mode", mode, "--time-in-secs", str(run_seconds),
        "--ttft-slo", "2.0", "--tpot-slo", "0.05",
        "--request-timeout-secs", str(int(request_timeout_secs)),
        "--metric-percentile", "50,95,99",
        "--num-producer", "8", "--channel-capacity", "1", "--threads", "4",
        "--output-path", str(out / ("warmup.jsonl" if warmup else "requests.jsonl")),
        "--summary-path", str(out / ("warmup-summary.json" if warmup else "request-sim-summary.json")),
    ]
    if mode == "feedback":
        argv.extend(["--bs-limit", "128", "--tpot-limit", "50"])
    else:
        argv.extend(["--scale-factor", str(scale)])
    if begin_ms is not None:
        argv.extend(["--begin-time", str(begin_ms)])
    if end_ms is not None:
        argv.extend(["--end-time", str(end_ms)])

    plan = {
        "cohort_indices": [item[0] for item in cohort],
        "cohort_count": len(cohort),
        "begin_ms": begin_ms,
        "end_ms": end_ms,
        "first_ts_ms": cohort[0][1],
        "last_ts_ms": cohort[-1][1],
        "planned_dispatch_seconds": t_dispatch,
        "run_seconds": run_seconds,
        "watchdog_seconds": watchdog_secs,
    }
    return argv, plan


def scrape(endpoint, out, name):
    with urlopen(endpoint + "/metrics", timeout=10) as response:
        (out / name).write_bytes(response.read())


def native_stats(run):
    return [json.loads(path.read_text()) for path in sorted(run.glob("vericache-stats-*.json"))]


def sample_cpu(done, samples):
    parent = psutil.Process()
    while not done.is_set():
        processes = [p for p in parent.children(recursive=True) if p.name() != "request-sim"]
        usage = {}
        for process in processes:
            try:
                if process.name() in {"vllm", "python", "python3", "mooncake_master"} or process.name().startswith("VLLM"):
                    cpu = process.cpu_times()
                    usage[process.pid] = {"cpu_seconds": cpu.user + cpu.system,
                                          "rss_bytes": process.memory_info().rss}
            except psutil.NoSuchProcess:
                continue
        samples.append({"monotonic": time.monotonic(), "processes": usage})
        done.wait(0.5)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def analyze(rows, plan=None):
    if not rows:
        raise ValueError("request-sim produced no responses")

    if plan is not None:
        planned_ids = plan["cohort_indices"]
        planned_set = set(planned_ids)
        by_idx = {}
        for r in rows:
            if "request_index" in r:
                idx = int(r["request_index"])
                if idx in planned_set:
                    by_idx[idx] = r
        if by_idx:
            cohort_rows = [by_idx.get(idx) for idx in planned_ids]
        else:
            cohort_rows = list(rows)
        n_cohort = len(planned_ids)
        t_dispatch = float(plan["planned_dispatch_seconds"])
    else:
        cohort_rows = list(rows)
        n_cohort = len(rows)
        t_dispatch = None

    observed = [r for r in cohort_rows if r is not None]
    missing = n_cohort - len(observed)
    wall = (max(float(r["e_time"]) for r in observed) - min(float(r["s_time"]) for r in observed)) / 1000.0
    assert wall > 0
    if t_dispatch is None or t_dispatch <= 0:
        t_dispatch = wall

    def is_completed(r):
        return (
            r.get("status") == "200"
            and "stream_error" not in r
            and r.get("output_tokens_source") == "usage"
            and "first_token_time" in r
        )

    completed = [r for r in observed if is_completed(r)]
    single_token = [r for r in completed if int(r["actual_output_tokens"]) <= 1]
    multi_token = [r for r in completed if int(r["actual_output_tokens"]) >= 2]

    ttft = [float(r["first_token_time"]) for r in completed]
    tpot = [
        float(r.get("total_time", 0.0)) / (int(r["actual_output_tokens"]) - 1)
        for r in multi_token
    ]
    e2e = [float(r["span_time"]) for r in completed]
    ttft_all_observed = [float(r["first_token_time"]) for r in observed if "first_token_time" in r]

    qualified = 0
    for r in completed:
        req_ttft = float(r["first_token_time"])
        out_toks = int(r["actual_output_tokens"])
        if out_toks <= 1:
            if req_ttft <= 2000.0:
                qualified += 1
        else:
            req_tpot = float(r.get("total_time", 0.0)) / (out_toks - 1)
            if req_ttft <= 2000.0 and req_tpot <= 50.0:
                qualified += 1

    status_counts = {}
    for r in observed:
        st = str(r.get("status", "unknown"))
        status_counts[st] = status_counts.get(st, 0) + 1
    if missing:
        status_counts["missing"] = missing

    return {
        "planned_cohort": n_cohort,
        "submitted": len(observed),
        "completed": len(completed),
        "single_token_completed": len(single_token),
        "tpot_sample_count": len(multi_token),
        "timeout_count": status_counts.get("timeout", 0),
        "stream_error_count": status_counts.get("stream_error", 0),
        "other_error_count": status_counts.get("other_error", 0),
        "missing_count": missing,
        "status_counts": dict(sorted(status_counts.items())),
        "planned_dispatch_seconds": t_dispatch,
        "measurement_wall_seconds": wall,
        "slo_qualified": qualified,
        "slo_attainment": qualified / n_cohort,
        "goodput_dispatch_requests_per_second": qualified / t_dispatch,
        "wall_goodput_requests_per_second": qualified / wall,
        "slo_goodput_requests_per_second": qualified / t_dispatch,
        "output_tokens_per_second": sum(int(r["actual_output_tokens"]) for r in completed) / wall,
        "output_tokens_dispatch_per_second": sum(int(r["actual_output_tokens"]) for r in completed) / t_dispatch,
        "cached_prompt_tokens": sum(int(r.get("cached_tokens", 0)) for r in completed),
        "prompt_tokens": sum(int(r["prompt_tokens"]) for r in completed),
        "latency_ms": {
            name: {label: percentile(values, q) for label, q in
                   (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p99.9", 0.999))}
            for name, values in (
                ("ttft", ttft),
                ("tpot", tpot),
                ("e2e", e2e),
                ("ttft_observed_all", ttft_all_observed),
            )
        },
    }


def run_replay(argv, log, timeout_secs):
    with log.open("w") as output:
        subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=timeout_secs)
    assert "Loaded prompt cache from" in log.read_text(), log


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metric_totals(path):
    totals = {}
    for group in text_string_to_metric_families(path.read_text()):
        for sample in group.samples:
            if sample.name.startswith(("vllm:", "lmcache:")) and sample.name.endswith("_total"):
                totals[sample.name] = totals.get(sample.name, 0) + sample.value
    return totals


def parse_server_log_peaks(server_log_path: Path):
    if not server_log_path.is_file():
        return {}
    text = server_log_path.read_text(errors="replace")
    running = [int(m) for m in re.findall(r"Running:\s*(\d+)\s*reqs", text)]
    waiting = [int(m) for m in re.findall(r"Waiting:\s*(\d+)\s*reqs", text)]
    kv_pct = [float(m) for m in re.findall(r"GPU KV cache usage:\s*([0-9.]+)%", text)]
    preempt = len(re.findall(r"preempt", text, flags=re.IGNORECASE))
    oom = len(re.findall(r"OutOfMemoryError|CUDA out of memory", text))
    return {
        "peak_running_reqs": max(running) if running else 0,
        "peak_waiting_reqs": max(waiting) if waiting else 0,
        "peak_gpu_kv_cache_percent": max(kv_pct) if kv_pct else 0.0,
        "preemption_log_mentions": preempt,
        "oom_log_mentions": oom,
    }


def summarize_server(out, before, after, samples):
    previous = metric_totals(out / "metrics-before.prom")
    current = metric_totals(out / "metrics-after.prom")
    counters = {key: value - previous.get(key, 0) for key, value in current.items()}
    cpu_seconds = 0.0
    rss_peak = 0
    period = 0.0
    if samples:
        first, last = samples[0], samples[-1]
        period = last["monotonic"] - first["monotonic"]
        cpu_seconds = sum(
            row["cpu_seconds"] - first["processes"].get(pid, {}).get("cpu_seconds", 0)
            for pid, row in last["processes"].items()
        )
        rss_peak = max(sum(proc["rss_bytes"] for proc in row["processes"].values()) for row in samples)
    metrics = {
        "counters": counters,
        "hbm_apc_hit_tokens": counters.get("vllm:prefix_cache_hits_total"),
        "external_prefix_hit_tokens": counters.get("vllm:external_prefix_cache_hits_total"),
        "preemptions_total": counters.get("vllm:num_preemptions_total"),
        "server_log_peaks": parse_server_log_peaks(out / "server.log"),
        "host_cpu_percent_of_one_core": 100 * cpu_seconds / period if period else None,
        "host_peak_process_rss_bytes": rss_peak,
        "vericache_before": before,
        "vericache_after": after,
        "process_samples": samples,
    }
    if before and after:
        assert len(before) == len(after) == 1
        extended = {name: value - before[0]["extended"][name]
                    for name, value in after[0]["extended"].items()}
        metrics.update({
            "vericache_deltas": extended,
            "dram_restore_token_equivalents": extended["dram_hits"] * BLOCK_TOKENS / LAYERS,
            "nvme_restore_token_equivalents": extended["nvme_hits"] * BLOCK_TOKENS / LAYERS,
            "nvme_promoted_payload_bytes": extended["nvme_promotions"] * FRAME_BYTES,
            "nvme_async_writeback_payload_bytes": extended["async_writebacks"] * FRAME_BYTES,
        })
        if "nvme_io" in after[0]:
            metrics["nvme_io_bytes"] = {
                name: value - before[0]["nvme_io"][name]
                for name, value in after[0]["nvme_io"].items()
            }
    return metrics


def main():
    job = json.loads(Path(os.environ["PHASE3_STAGE1_JOB"]).read_text())
    assert job["trace"] in (*TRACES, *ROUTED_TRACES, TRACE_A)
    scale = job["scale"]
    if scale != "feedback":
        scale_f = float(scale)
        assert math.isfinite(scale_f) and scale_f > 0.0
    run = Path(os.environ["PHASE3_TEST_DIR"])
    out = Path(os.environ["PHASE3_RESULT_DIR"])
    dump(out / "job.json", job)
    port = reserve_port()
    endpoint = f"http://127.0.0.1:{port}"
    _, frozen = trace_source(job["trace"])
    subprocess.run(["sha256sum", "-c", "SHA256SUMS", "--quiet"], cwd=frozen, check=True)

    batch_invariant = bool(job.get("batch_invariant", False))
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant else "0"
    max_num_seqs = int(job.get("max_num_seqs", 16))
    request_timeout_secs = int(job.get("request_timeout_secs", DEFAULT_REQUEST_TIMEOUT_SECS))
    warmup_end_ms = job.get("warmup_end_ms", DEFAULT_WARMUP_END_MS)
    measurement_end_ms = job.get("measurement_end_ms")
    warmup_scale = job.get("warmup_scale", 1.0 if scale == "feedback" else scale)

    lmc_tmp = Path("/tmp") / f"vc_lmc_{os.getpid()}"
    _, nvme_blocks = capacity(job)
    nvme_run = resolve_nvme_run_dir(run) if nvme_blocks > 0 else run
    try:
        with maybe_start_mooncake_master(job, nvme_run, out) as (mc_rpc_port, mc_metrics_port):
            argv, settings = serving_options(
                job, run, out, port, mooncake_rpc_port=mc_rpc_port, nvme_run=nvme_run
            )
            warmup_argv, warmup_plan = replay_options(
                endpoint, out, job["trace"], warmup_scale,
                warmup=True, warmup_end_ms=warmup_end_ms,
                request_timeout_secs=request_timeout_secs,
            )
            measure_argv, measure_plan = replay_options(
                endpoint, out, job["trace"], scale,
                warmup=False, warmup_end_ms=warmup_end_ms,
                measurement_end_ms=measurement_end_ms,
                request_timeout_secs=request_timeout_secs,
            )
            write_manifest(out, MODEL, argv)
            meta = json.loads((out / "manifest.json").read_text())
            if job["trace"] == TRACE_A:
                meta["trace_window"] = json.loads((frozen / "window.json").read_text())
            dump(out / "manifest.json", {
                **meta,
                "request_sim_bin_sha256": file_sha256(PHASE3 / "bin/request-sim"),
                "nvme_run_dir": str(nvme_run) if nvme_blocks > 0 else None,
                "max_num_seqs": max_num_seqs,
                "request_timeout_secs": request_timeout_secs,
                "job": job,
                "budgets": settings,
                "warmup_argv": warmup_argv,
                "warmup_plan": {k: v for k, v in warmup_plan.items() if k != "cohort_indices"},
                "measurement_argv": measure_argv,
                "measurement_plan": {k: v for k, v in measure_plan.items() if k != "cohort_indices"},
            })
            with start_server(out, argv, endpoint):
                expected = f"{HBM_BLOCKS * BLOCK_TOKENS:,} tokens"
                server_log = (out / "server.log").read_text()
                if not re.search(r"GPU KV cache size: " + re.escape(expected), server_log):
                    raise RuntimeError(f"HBM KV budget was not confirmed in the server log: {expected}")
                if warmup_end_ms is not None and warmup_end_ms > 0:
                    run_replay(warmup_argv, out / "warmup.log", warmup_plan["watchdog_seconds"])
                before = native_stats(run)
                scrape(endpoint, out, "metrics-before.prom")
                if mc_metrics_port is not None:
                    try:
                        scrape(f"http://127.0.0.1:{mc_metrics_port}", out, "mooncake-metrics-before.prom")
                    except Exception:
                        pass
                done, samples = Event(), []
                sampler = Thread(target=sample_cpu, args=(done, samples), daemon=True)
                sampler.start()
                try:
                    run_replay(measure_argv, out / "replay.log", measure_plan["watchdog_seconds"])
                finally:
                    done.set()
                    sampler.join(timeout=2)
                time.sleep(0.6)
                after = native_stats(run)
                scrape(endpoint, out, "metrics-after.prom")
                if mc_metrics_port is not None:
                    try:
                        scrape(f"http://127.0.0.1:{mc_metrics_port}", out, "mooncake-metrics-after.prom")
                    except Exception:
                        pass
    finally:
        if lmc_tmp.exists():
            shutil.rmtree(lmc_tmp, ignore_errors=True)
        if nvme_run != run and nvme_run.exists():
            shutil.rmtree(nvme_run, ignore_errors=True)
    report = analyze(read_rows(out / "requests.jsonl"), plan=measure_plan)
    dump(out / "summary.json", report)
    dump(out / "server_metrics.json", summarize_server(out, before, after, samples))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
