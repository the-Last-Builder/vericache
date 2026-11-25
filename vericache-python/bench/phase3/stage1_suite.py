"""Resumable, fault-isolated, zero-LLM overnight Stage 1 experiment runner."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from generate_stage1_report import write_live_report
from prepare_bailian_trace_a import TRACE_A
from prepare_traces import TRACES, read_rows
from stage0_correctness import PHASE3
from stage1 import (
    DEFAULT_REQUEST_TIMEOUT_SECS,
    DEFAULT_WARMUP_END_MS,
    row_timestamp_ms,
    select_cohort,
    trace_source,
)


HERE = Path(__file__).resolve().parent
RESULTS = PHASE3 / "results"
SCHEDULER = "/usr/local/bin/scheduler"


def fmt_scale(scale):
    if scale == "feedback":
        return "feedback"
    return f"{float(scale):g}"


def identifier(job):
    parts = [
        f"exp{job['exp']}",
        f"prof{job['profile']}",
        job["system"],
        job["trace"],
        f"s{fmt_scale(job['scale'])}",
        f"m{int(job.get('max_num_seqs', 16))}",
        f"w{int(job.get('warmup_end_ms', DEFAULT_WARMUP_END_MS))}",
    ]
    if job.get("measurement_end_ms") is not None:
        parts.append(f"end{int(job['measurement_end_ms'])}")
    if job.get("ablation"):
        parts.append(f"abl_{job['ablation']}")
    if job.get("batch_invariant", False):
        parts.append("bi1")
    parts.append(f"rep{int(job.get('repeat', 1))}")
    return "_".join(parts)


def estimate_duration_seconds(job):
    if "duration_seconds" in job:
        return int(job["duration_seconds"])
    trace_type, frozen = trace_source(job["trace"])
    rows = read_rows(frozen / f"{job['trace']}.jsonl")
    scale = job["scale"]
    warmup_end_ms = job.get("warmup_end_ms", DEFAULT_WARMUP_END_MS)
    measurement_end_ms = job.get("measurement_end_ms")
    warmup_scale = job.get("warmup_scale", 1.0 if scale == "feedback" else scale)
    req_timeout = int(job.get("request_timeout_secs", DEFAULT_REQUEST_TIMEOUT_SECS))

    warmup_secs = 0.0
    if warmup_end_ms is not None and warmup_end_ms > 0:
        w_cohort = select_cohort(rows, trace_type, None, warmup_end_ms)
        if w_cohort:
            warmup_secs = w_cohort[-1][1] / 1000.0 / float(warmup_scale) + 60.0

    if scale == "feedback":
        meas_secs = 60.0
    else:
        begin_ms = (warmup_end_ms + 1) if warmup_end_ms is not None else None
        m_cohort = select_cohort(rows, trace_type, begin_ms, measurement_end_ms)
        if not m_cohort:
            raise ValueError(f"Empty measurement cohort for {job}")
        origin_ms = begin_ms or 0
        meas_secs = (m_cohort[-1][1] - origin_ms) / 1000.0 / float(scale)

    # Server startup (~120s) + warmup + measurement + drain headroom (min(req_timeout, 300)) + margin
    total = int(math.ceil(150.0 + warmup_secs + meas_secs + min(req_timeout, 300) + 90.0))
    return max(760, min(total, 3300))


def finished_jobs():
    finished = {}
    for path in sorted(RESULTS.glob("stage1_run.*")):
        status = path / "exit-status"
        if not status.is_file() or status.read_text().strip() != "0":
            continue
        ok = True
        for file in ("summary.json", "server_metrics.json", "requests.jsonl", "cleanup.log", "release.json", "job.json"):
            p = path / file
            if not p.is_file() or p.stat().st_size == 0:
                ok = False
                break
        if not ok:
            continue
        try:
            release = json.loads((path / "release.json").read_text())["lease"]
            if release.get("state") not in ("released", "reused") or release.get("active_gpu_ids"):
                continue
            job = json.loads((path / "job.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            if "planned_cohort" not in summary:
                continue
        except Exception:
            continue
        finished[identifier(job)] = path.name
    return finished


def wait_for_reservation(gpu, res_id, min_seconds):
    while True:
        info = json.loads(subprocess.check_output([SCHEDULER, "status", "--gpus", str(gpu)]))
        now = datetime.fromisoformat(info["now"])
        matches = [l for l in info.get("my_allocations", []) if l.get("id") == res_id]
        if not matches:
            raise RuntimeError(f"Reservation {res_id} not found in my_allocations")
        lease = matches[0]
        start = datetime.fromisoformat(lease["start"])
        end = datetime.fromisoformat(lease["end"])
        if lease.get("state") != "active":
            raise RuntimeError(f"Reservation {res_id} is not active: {lease}")
        if now < start:
            wait_s = max(1.0, (start - now).total_seconds())
            print(f"[{now.isoformat()}] Waiting {wait_s:.0f}s for reservation {res_id} start ({start.isoformat()})...", flush=True)
            time.sleep(min(wait_s, 15.0))
            continue
        rem = (end - now).total_seconds()
        if rem < min_seconds + 30:
            raise RuntimeError(f"Reservation {res_id} only has {rem:.0f}s left (< {min_seconds + 30}s needed)")
        return gpu


def wait_for_claim_slot(gpus, min_seconds):
    while True:
        best_gpu = None
        best_rem = -1.0
        now_iso = ""
        for gpu in gpus:
            info = json.loads(subprocess.check_output([SCHEDULER, "status", "--gpus", str(gpu)]))
            now = datetime.fromisoformat(info["now"])
            now_iso = now.isoformat()
            end = datetime.fromisoformat(info["availability_end"])
            rem = (end - now).total_seconds()
            if str(gpu) in info.get("available_gpu_ids", []) and rem >= min_seconds + 45:
                if rem > best_rem:
                    best_rem = rem
                    best_gpu = gpu
        if best_gpu is not None:
            return best_gpu
        print(
            f"[{now_iso}] Waiting for GPU in {gpus} with >= {min_seconds + 45}s remaining...",
            flush=True,
        )
        time.sleep(30)


def run_job(job, gpus, vericache_sha, reservation_id=None):
    job_id = identifier(job)
    duration = estimate_duration_seconds(job)
    config = RESULTS / "stage1_jobs" / f"{job_id}.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(job, indent=2) + "\n")
    if reservation_id:
        assert len(gpus) == 1
        gpu = wait_for_reservation(gpus[0], reservation_id, duration)
    else:
        gpu = wait_for_claim_slot(gpus, duration)
    environment = {
        **os.environ,
        "PHASE3_STAGE1_JOB": str(config),
        "PHASE3_STAGE1_DURATION": str(duration),
        "PHASE3_VERICACHE_SHA": vericache_sha,
    }
    if reservation_id:
        environment["PHASE3_RESERVATION_ID"] = reservation_id
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Running {job_id} on GPU {gpu} (duration={duration}s, res={reservation_id or 'claim'})", flush=True)
    worker = subprocess.Popen(
        ["bash", str(HERE / "stage0_hw.sh"), "stage1", str(gpu)],
        env=environment,
    )
    try:
        status = worker.wait()
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
    return status, gpu, duration


def stop(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", required=True, type=int, choices=range(8))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--vericache-sha", default=os.environ.get("PHASE3_VERICACHE_SHA", "a45c22a2d7e4ca56db7068142dc40466592e8021"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--report-name", default="stage1_live_report.md")
    parser.add_argument("--reservation-id", default=None)
    parser.add_argument("--auto-cancel-reservation", action="store_true")
    parser.add_argument("--state-prefix", default="stage1")
    args = parser.parse_args()

    for signum in (signal.SIGTERM,):
        signal.signal(signum, stop)

    plan_jobs = json.loads(args.plan.read_text())
    failed_path = RESULTS / f"{args.state_prefix}_failed_jobs.json"
    index_path = RESULTS / f"{args.state_prefix}_plan_index.json"
    failed = json.loads(failed_path.read_text()) if failed_path.is_file() else {}

    completed = finished_jobs()
    write_live_report(RESULTS, plan_jobs=plan_jobs, failed=failed, report_name=args.report_name)

    try:
        for idx, job in enumerate(plan_jobs, 1):
            job_id = identifier(job)
            if job_id in completed:
                print(f"[{idx}/{len(plan_jobs)}] SKIP already done: {job_id} ({completed[job_id]})", flush=True)
                continue

            attempts = 0
            succeeded = False
            last_rc = None
            while attempts <= args.max_retries:
                attempts += 1
                rc, gpu, duration = run_job(job, args.gpus, args.vericache_sha, reservation_id=args.reservation_id)
                last_rc = rc
                completed = finished_jobs()
                if rc == 0 and job_id in completed:
                    succeeded = True
                    failed.pop(job_id, None)
                    print(f"[{idx}/{len(plan_jobs)}] DONE {job_id} -> {completed[job_id]}", flush=True)
                    break
                print(
                    f"[{idx}/{len(plan_jobs)}] ATTEMPT {attempts} FAILED for {job_id} on GPU {gpu} (rc={rc})",
                    flush=True,
                )
                time.sleep(10)

            if not succeeded:
                failed[job_id] = {
                    "job": job,
                    "exit_code": last_rc,
                    "attempts": attempts,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                failed_path.write_text(json.dumps(failed, indent=2) + "\n")
                print(f"[{idx}/{len(plan_jobs)}] RECORDED FAILURE {job_id}, continuing queue.", flush=True)

            write_live_report(RESULTS, plan_jobs=plan_jobs, failed=failed, report_name=args.report_name)
    finally:
        if args.reservation_id and args.auto_cancel_reservation:
            print(f"Auto-cancelling reservation {args.reservation_id}...", flush=True)
            subprocess.run([SCHEDULER, "reserve", "cancel", args.reservation_id], check=False)

    completed = finished_jobs()
    index = {identifier(j): completed.get(identifier(j)) for j in plan_jobs}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    write_live_report(RESULTS, plan_jobs=plan_jobs, failed=failed, report_name=args.report_name)
    done_cnt = sum(v is not None for v in index.values())
    print(f"Plan finished: {done_cnt}/{len(plan_jobs)} succeeded, {len(failed)} failed.", flush=True)


if __name__ == "__main__":
    main()
