"""Detached Stage 2 suite runner (Exp 2.1 safety gate + Exp 2.2 4-system cross-NUMA RDMA comparison)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PHASE3 = Path("/data/nvme2/zxw/vericache_phase3")
SCHEDULER = "/usr/local/bin/scheduler"
STAGE2_HW = PHASE3 / "vericache-python/bench/phase3/stage2_hw.sh"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cancel_reservation(res_id: str | None) -> None:
    if not res_id:
        return
    try:
        subprocess.run([SCHEDULER, "reserve", "cancel", res_id], check=False, timeout=20)
    except Exception:
        pass


def render_live_report(status: dict, md_path: Path) -> None:
    lines = [
        "# Stage 2 Live Report — Cross-NUMA RoCEv2 RDMA",
        "",
        f"- Updated (UTC): `{status['updated_at']}`",
        f"- GPUs: `GPU {status['gpu0']} (NUMA 0)` ↔ `GPU {status['gpu1']} (NUMA 1)`",
        f"- Reservation: `{status.get('reservation_id')}`",
        f"- State: `{status['state']}` ({status['completed']}/{status['total']} jobs)",
        "",
    ]
    exp21 = status.get("exp21")
    if exp21:
        lines.extend([
            "## Exp 2.1 — Cross-Instance RDMA & Fault/Eviction Safety Gate",
            "",
            f"- Status: **{'PASS' if exp21.get('passed') else 'FAIL'}** (`{exp21.get('result_dir')}`)",
            "",
        ])
    rows = status.get("exp22_runs", [])
    if rows:
        lines.extend([
            "## Exp 2.2 — Dual-Instance Trace Replay Comparison (`toolagent_routed`, 70/30 Sticky/Drift)",
            "",
            "| Scale | System | State | SLO% | Goodput | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | Preempts | Ext Hits (tok) | RDMA Hits (frm) | HCA0+1 Xmit (GiB) | Result Dir |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for r in rows:
            s = r.get("summary") or {}
            m = r.get("metrics") or {}
            lat = s.get("latency_ms") or {}
            ttft = lat.get("ttft") or {}
            tpot = lat.get("tpot") or {}
            slo = f"{100 * s['slo_attainment']:.1f}%" if "slo_attainment" in s else "—"
            gp = f"{s['slo_goodput_requests_per_second']:.3f}" if "slo_goodput_requests_per_second" in s else "—"
            ttft_s = f"{ttft.get('p50', 0):.0f} / {ttft.get('p95', 0):.0f}" if ttft.get("p50") is not None else "—"
            tpot_s = f"{tpot.get('p50', 0):.1f} / {tpot.get('p95', 0):.1f}" if tpot.get("p50") is not None else "—"
            pre = str(int(m.get("total_preemptions", 0))) if m else "—"
            ext = f"{m.get('total_external_prefix_hit_tokens', 0) / 1000:.1f}k" if m else "—"
            vc = m.get("vericache_rdma_deltas") or {}
            rfrm = (
                str(vc.get("inst0", {}).get("remote_hits", 0) + vc.get("inst1", {}).get("remote_hits", 0))
                if vc
                else "—"
            )
            hca_gib = (
                f"{((m.get('hca0_delta', {}).get('port_xmit_data', 0) + m.get('hca1_delta', {}).get('port_xmit_data', 0)) / (1 << 30)):.2f}"
                if m
                else "—"
            )
            lines.append(
                f"| {r['scale']} | {r['system']} | {r['state']} | {slo} | {gp} | {ttft_s} | {tpot_s} | {pre} | {ext} | {rfrm} | {hca_gib} | `{r.get('result_dir', '')}` |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n")


def run_hw(mode: str, gpu0: int, gpu1: int, res_id: str | None, extra_env: dict, log_path: Path, res_start_iso: str | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    use_res = res_id
    if use_res and res_start_iso:
        start_dt = datetime.fromisoformat(res_start_iso.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) < start_dt:
            use_res = None
    if use_res:
        env["PHASE3_RESERVATION_ID"] = use_res
    else:
        env.pop("PHASE3_RESERVATION_ID", None)
    env.update(extra_env)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [str(STAGE2_HW), mode, str(gpu0), str(gpu1)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        result_dir = ""
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith("Stage 2 Results: "):
                result_dir = line.split("Stage 2 Results: ", 1)[1].strip()
        rc = proc.wait()
    return rc, result_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gpu0", type=int, required=True)
    parser.add_argument("--gpu1", type=int, required=True)
    parser.add_argument("--reservation-id", default=None)
    parser.add_argument("--reservation-start", default=None)
    parser.add_argument("--auto-cancel-reservation", action="store_true")
    parser.add_argument("--status-dir", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_json = status_dir / "stage2_live_status.json"
    report_md = status_dir / "stage2_live_report.md"

    jobs = plan.get("exp22_jobs", [])
    run_exp21 = bool(plan.get("run_exp21", True))
    status = {
        "state": "running",
        "started_at": now_utc(),
        "updated_at": now_utc(),
        "gpu0": args.gpu0,
        "gpu1": args.gpu1,
        "reservation_id": args.reservation_id,
        "total": (1 if run_exp21 else 0) + len(jobs),
        "completed": 0,
        "failed": 0,
        "exp21": None,
        "exp22_runs": [],
    }

    def save():
        status["updated_at"] = now_utc()
        status_json.write_text(json.dumps(status, indent=2) + "\n")
        render_live_report(status, report_md)

    stop_requested = False

    def _sig(_s, _f):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    try:
        save()
        if run_exp21 and not stop_requested:
            rc, rdir = run_hw(
                "exp21",
                args.gpu0,
                args.gpu1,
                args.reservation_id,
                {"PHASE3_STAGE2_DURATION": "900"},
                status_dir / "exp21_driver.log",
            )
            rep_file = Path(rdir) / "exp21_report.json" if rdir else None
            exp21_data = json.loads(rep_file.read_text()) if rep_file and rep_file.is_file() else {}
            passed = rc == 0 and bool(exp21_data.get("all_passed"))
            status["exp21"] = {
                "rc": rc,
                "passed": passed,
                "result_dir": Path(rdir).name if rdir else "",
                "report": exp21_data,
            }
            if passed:
                status["completed"] += 1
            else:
                status["failed"] += 1
                save()
                if plan.get("abort_on_exp21_fail", True):
                    status["state"] = "aborted_exp21_failed"
                    save()
                    sys.exit(1)
            save()

        for idx, job in enumerate(jobs):
            if stop_requested:
                break
            job_file = status_dir / f"job_{idx:02d}_{job['system']}_{job['scale']}.json"
            job_file.write_text(json.dumps(job, indent=2) + "\n")
            entry = {
                "index": idx,
                "system": job["system"],
                "scale": job["scale"],
                "trace": job.get("trace", "toolagent_routed"),
                "state": "running",
                "started_at": now_utc(),
            }
            status["exp22_runs"].append(entry)
            save()

            dur = str(int(job.get("duration_seconds", 1100)))
            rc, rdir = run_hw(
                "exp22",
                args.gpu0,
                args.gpu1,
                args.reservation_id,
                {
                    "PHASE3_STAGE2_JOB": str(job_file),
                    "PHASE3_STAGE2_DURATION": dur,
                },
                status_dir / f"job_{idx:02d}_{job['system']}_{job['scale']}.log",
                res_start_iso=args.reservation_start,
            )
            entry["rc"] = rc
            entry["finished_at"] = now_utc()
            entry["result_dir"] = Path(rdir).name if rdir else ""
            if rdir:
                sf = Path(rdir) / "summary.json"
                mf = Path(rdir) / "server_metrics.json"
                if sf.is_file():
                    entry["summary"] = json.loads(sf.read_text())
                if mf.is_file():
                    entry["metrics"] = json.loads(mf.read_text())
            if rc == 0 and "summary" in entry:
                entry["state"] = "succeeded"
                status["completed"] += 1
            else:
                entry["state"] = "failed"
                status["failed"] += 1
            save()

        status["state"] = "done" if status["failed"] == 0 else "done_with_failures"
        save()
    finally:
        if args.auto_cancel_reservation and args.reservation_id:
            cancel_reservation(args.reservation_id)


if __name__ == "__main__":
    main()
