#!/usr/bin/env python3
"""NUMA-pinned benchmark orchestrator for VeriCache, Mooncake 0.3.14, and LMCache 0.5.7.post1."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOAD_DIR = ROOT / "experiments" / "workloads"
BASELINES_DIR = ROOT / "experiments" / "baselines"
RESULTS_ROOT = ROOT / "experiments" / "results"
ALL_SUITES = [
    "s1_local_tiers",
    "s2_rdma_ablation",
    "s3_trace_replay",
    "s4_safety_revoke",
    "s5_scaleup_cluster",
]

sys.path.insert(0, str(ROOT / "experiments" / "harness"))
from report import generate_summary  # noqa: E402


def _run_cmd(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _detect_numa_prefix(preferred: bool = False) -> list[str]:
    if shutil.which("numactl") is None:
        return []
    numa_node = os.environ.get("VERICACHE_NUMA_NODE", "").strip()
    if not numa_node:
        return []
    # For s5_scaleup_cluster (large host pinning across 2 nodes), use --preferred=<node>
    # (NOT --membind=<node>) so Linux page-cache reclaim works cleanly without single-node OOM.
    flag = f"--preferred={numa_node}" if preferred else f"--membind={numa_node}"
    probe = subprocess.run(
        ["numactl", f"--cpunodebind={numa_node}", flag, "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return ["numactl", f"--cpunodebind={numa_node}", flag]
    return []


def _verify_and_clean_nvme_dir(nvme_dir: pathlib.Path, label: str) -> None:
    if not nvme_dir.exists():
        return
    leaked = list(nvme_dir.iterdir())
    if leaked:
        names = [p.name for p in leaked[:10]]
        print(
            f"[orchestrate] WARNING: {len(leaked)} leaked scratch entries in {nvme_dir} after {label}: {names}; cleaning up...",
            flush=True,
        )
        for p in leaked:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except OSError:
                    pass


def _collect_manifest(run_id: str, suites: list[str], numa_prefix: list[str]) -> dict[str, Any]:
    def _try_out(cmd: list[str]) -> str:
        try:
            res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=10)
            return res.stdout.strip() if res.returncode == 0 else ""
        except Exception:
            return ""

    gpus = _try_out(
        [
            sys.executable,
            "-c",
            "import torch; print(', '.join(f'{i}:{torch.cuda.get_device_name(i)}' for i in range(torch.cuda.device_count())))",
        ]
    )
    versions = {
        "rustc": _try_out(["rustc", "--version"]),
        "python": sys.version.split()[0],
        "torch": _try_out([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "mooncake": _try_out(
            [sys.executable, "-c", "import importlib.metadata as m; print(m.version('mooncake-transfer-engine-cuda13'))"]
        ),
        "lmcache": _try_out(
            [sys.executable, "-c", "import importlib.metadata as m; print(m.version('lmcache'))"]
        ),
    }
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_rev": _try_out(["git", "rev-parse", "--short", "HEAD"]),
        "suites": suites,
        "numa_cmd": " ".join(numa_prefix) if numa_prefix else "none",
        "hardware": {
            "gpus": gpus,
            "rdma_dev": os.environ.get("VERICACHE_RDMA_DEVICE", "mlx5_3"),
            "gid_index": int(os.environ.get("VERICACHE_RDMA_GID_INDEX", "3")),
            "nvme_dir": "/workspace/target/bench-nvme",
        },
        "versions": versions,
    }


def _expand_toml_jobs(suite_name: str) -> list[tuple[str, dict[str, Any]]]:
    toml_path = WORKLOAD_DIR / f"{suite_name}.toml"
    with toml_path.open("rb") as f:
        cfg = tomllib.load(f)
    defaults = cfg.get("defaults", {})
    expanded: list[tuple[str, dict[str, Any]]] = []
    for entry in cfg.get("jobs", []):
        systems = entry.get("systems", ["vericache"])
        chunk_list = entry.get("chunk_bytes_list", [entry.get("chunk_bytes", 262144)])
        for chunk_bytes in chunk_list:
            for sys_name in systems:
                spec: dict[str, Any] = dict(defaults)
                for k, v in entry.items():
                    if k not in ("systems", "chunk_bytes_list"):
                        spec[k] = v
                spec["suite"] = cfg["suite"]
                spec["chunk_bytes"] = int(chunk_bytes)
                if "VERICACHE_RDMA_DEVICE" in os.environ:
                    spec["rdma_dev"] = os.environ["VERICACHE_RDMA_DEVICE"]
                if "VERICACHE_RDMA_GID_INDEX" in os.environ:
                    spec["gid_index"] = int(os.environ["VERICACHE_RDMA_GID_INDEX"])
                expanded.append((str(sys_name), spec))
    return expanded


def _ensure_traces_and_nvme() -> None:
    pathlib.Path("/workspace/target/bench-nvme").mkdir(parents=True, exist_ok=True)
    trace_dir = pathlib.Path("/workspace/target/bench-traces")
    synth = trace_dir / "synthetic_zipf_trace.jsonl"
    if not synth.exists() or synth.stat().st_size == 0:
        print("[orchestrate] Generating synthetic_zipf_trace.jsonl...", flush=True)
        _run_cmd([sys.executable, str(WORKLOAD_DIR / "gen_trace.py"), "--out-dir", str(trace_dir)])


def _build_vericache_bench() -> pathlib.Path:
    print("[orchestrate] Building vericache-bench (--release --offline)...", flush=True)
    _run_cmd(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(ROOT / "vericache-bench" / "Cargo.toml"),
            "--target-dir",
            str(ROOT / "target"),
            "--offline",
        ]
    )
    bin_path = ROOT / "target" / "release" / "vericache-bench"
    if not bin_path.exists():
        raise RuntimeError(f"missing release binary at {bin_path}")
    return bin_path


def _execute_job(
    numa_prefix: list[str],
    vericache_bin: pathlib.Path,
    sys_name: str,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(spec, tf)
        job_file = tf.name
    try:
        if sys_name == "vericache":
            cmd = numa_prefix + [str(vericache_bin), "--job-file", job_file]
        elif sys_name == "mooncake":
            cmd = numa_prefix + [
                sys.executable,
                str(BASELINES_DIR / "mooncake_runner.py"),
                "--job-file",
                job_file,
            ]
        elif sys_name == "lmcache":
            cmd = numa_prefix + [
                sys.executable,
                str(BASELINES_DIR / "lmcache_runner.py"),
                "--job-file",
                job_file,
            ]
        else:
            raise ValueError(f"unknown system '{sys_name}'")

        res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"Job FAILED ({sys_name} {spec['suite']}/{spec['case']} mode={spec['mode']} "
                f"chunk={spec['chunk_bytes']} rc={res.returncode}):\n"
                f"--- STDOUT ---\n{res.stdout}\n--- STDERR ---\n{res.stderr}"
            )
        records: list[dict[str, Any]] = []
        for line in res.stdout.splitlines():
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if "system" in obj and "throughput_gbps" in obj:
                        records.append(obj)
                except json.JSONDecodeError:
                    pass
        if not records:
            raise RuntimeError(
                f"Job emitted 0 MetricRecords ({sys_name} {spec['suite']}/{spec['case']}):\n"
                f"STDOUT={res.stdout}\nSTDERR={res.stderr}"
            )
        return records
    finally:
        try:
            os.unlink(job_file)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="VeriCache Benchmark Orchestrator (Suites 1-5)")
    ap.add_argument("--all", action="store_true", help="Run Suites 1-4 (or pass --suites s5_scaleup_cluster)")
    ap.add_argument(
        "--suites",
        "--suite",
        dest="suites",
        nargs="+",
        choices=ALL_SUITES,
        help="Specific suite(s) to run",
    )
    ap.add_argument("--systems", nargs="+", choices=["vericache", "mooncake", "lmcache"], help="Filter systems")
    ap.add_argument("--cases", nargs="+", help="Filter case names")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Fast 4-tier smoke mode for s5_scaleup_cluster (1 GiB HBM + 4 GiB DRAM + 1 GiB NVMe/node, 500 reqs)",
    )
    ap.add_argument("--run-id", type=str, default=None, help="Explicit run-id directory name")
    args = ap.parse_args()

    suites = ALL_SUITES if args.all or not args.suites else args.suites
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_ROOT / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _ensure_traces_and_nvme()
    vericache_bin = _build_vericache_bench()
    numa_strict = _detect_numa_prefix(preferred=False)
    numa_preferred = _detect_numa_prefix(preferred=True)
    # Record effective NUMA policy in manifest
    display_numa = numa_preferred if suites == ["s5_scaleup_cluster"] else numa_strict
    print(
        f"[orchestrate] run_id={run_id}, numa_s1_s4={' '.join(numa_strict) or 'none'}, "
        f"numa_s5={' '.join(numa_preferred) or 'none'}, smoke={args.smoke}",
        flush=True,
    )

    manifest = _collect_manifest(run_id, suites, display_numa)
    manifest["smoke"] = bool(args.smoke)
    manifest["numa_cmd_s5"] = " ".join(numa_preferred) if numa_preferred else "none"
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_start = time.perf_counter()
    total_records = 0
    nvme_dir = pathlib.Path("/workspace/target/bench-nvme")

    for suite in suites:
        jobs = _expand_toml_jobs(suite)
        if args.systems:
            jobs = [(s, spec) for s, spec in jobs if s in args.systems]
        if args.cases:
            jobs = [(s, spec) for s, spec in jobs if spec["case"] in args.cases]
        if args.smoke and suite == "s5_scaleup_cluster":
            for _s, spec in jobs:
                spec["hbm_gb_per_node"] = 1.0
                spec["dram_gb_per_node"] = 4.0
                spec["nvme_gb_per_node"] = 1.0
                spec["max_requests"] = 500
                spec["case"] = f"{spec['case']}_smoke"
        suite_numa = numa_preferred if suite == "s5_scaleup_cluster" else numa_strict
        raw_path = raw_dir / f"{suite}.jsonl"
        print(
            f"\n[orchestrate] === Suite {suite}: {len(jobs)} serial jobs (numa={' '.join(suite_numa) or 'none'}) ===",
            flush=True,
        )
        with raw_path.open("w", encoding="utf-8") as rf:
            for idx, (sys_name, spec) in enumerate(jobs, 1):
                t0 = time.perf_counter()
                try:
                    recs = _execute_job(suite_numa, vericache_bin, sys_name, spec)
                finally:
                    _verify_and_clean_nvme_dir(nvme_dir, f"{sys_name} {suite}/{spec['case']}")
                dt = time.perf_counter() - t0
                for r in recs:
                    rf.write(json.dumps(r, separators=(",", ":")) + "\n")
                    rf.flush()
                    total_records += 1
                    print(
                        f"  [{idx:02d}/{len(jobs):02d}] {r['system']:<24} {r['case']:<30} "
                        f"mode={r['mode']:<14} sz={r['chunk_bytes']:>8}B "
                        f"thr={r['throughput_gbps']:>6.2f} GB/s good={r['goodput_gbps']:>6.2f} GB/s "
                        f"p50={r['p50_us']:>7.1f}us p99={r['p99_us']:>7.1f}us "
                        f"corr={r['corrupted_frames']} ({dt:.1f}s)",
                        flush=True,
                    )

    summary_path = generate_summary(run_dir)
    total_elapsed = time.perf_counter() - total_start
    print(
        f"\n[orchestrate] COMPLETE: run_id={run_id}, records={total_records}, "
        f"elapsed={total_elapsed:.1f}s, summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
