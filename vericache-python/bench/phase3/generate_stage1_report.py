"""Aggregate Stage 1 runs into a live Markdown report and JSON status."""

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import median


def _fmt(val, fmt=".1f"):
    if val is None:
        return "-"
    return format(val, fmt)


def load_runs(results_dir: Path):
    runs = []
    for path in sorted(results_dir.glob("stage1_run.*")):
        status_file = path / "exit-status"
        if not status_file.is_file() or status_file.read_text().strip() != "0":
            continue
        try:
            job = json.loads((path / "job.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            server = json.loads((path / "server_metrics.json").read_text())
            if "planned_cohort" not in summary:
                continue
        except Exception:
            continue
        runs.append({"dir": path.name, "job": job, "summary": summary, "server": server})
    return runs


def write_live_report(results_dir: Path, *, plan_jobs=None, failed=None, report_name="stage1_live_report.md"):
    runs = load_runs(results_dir)
    failed = failed or {}
    grouped = defaultdict(list)
    for item in runs:
        j = item["job"]
        key = (
            str(j.get("exp", "")),
            str(j.get("profile", "")),
            str(j.get("trace", "")),
            int(j.get("max_num_seqs", 16)),
            str(j.get("scale", "")),
            str(j.get("system", "")),
            str(j.get("ablation", "")),
        )
        grouped[key].append(item)

    lines = [
        "# Phase 3 Stage 1 Live Experiment Report",
        "",
        f"- Updated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Completed new-harness runs: **{len(runs)}**",
        f"- Failed jobs recorded: **{len(failed)}**",
    ]
    if plan_jobs is not None:
        lines.append(f"- Active plan size: **{len(plan_jobs)}**")
    lines.append("")

    if grouped:
        lines.extend([
            "## Summary (Median across repeats)",
            "",
            "| Exp | Prof | Trace | mSeq | Scale | System | Abl | Reps | Comp/Coh | TO | SLO% | Goodput (req/s) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | Out TPS | HBM/Ext/DRAM/NVMe tok | Peak R/W/KV% | Preempt |",
            "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key in sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2], k[3], 999.0 if k[4] == "feedback" else float(k[4]), k[5], k[6])):
            items = grouped[key]
            exp, prof, trace, mseq, scale, system, abl = key
            sums = [it["summary"] for it in items]
            srvs = [it["server"] for it in items]
            comp = int(median([s["completed"] for s in sums]))
            coh = int(median([s["planned_cohort"] for s in sums]))
            to = int(median([s.get("timeout_count", 0) for s in sums]))
            slo_pct = 100.0 * median([s["slo_attainment"] for s in sums])
            gp = median([s["goodput_dispatch_requests_per_second"] for s in sums])
            ttft50 = median([s["latency_ms"]["ttft"]["p50"] for s in sums if s["latency_ms"]["ttft"]["p50"] is not None] or [0.0])
            ttft95 = median([s["latency_ms"]["ttft"]["p95"] for s in sums if s["latency_ms"]["ttft"]["p95"] is not None] or [0.0])
            tpot50_vals = [s["latency_ms"]["tpot"]["p50"] for s in sums if s["latency_ms"]["tpot"]["p50"] is not None]
            tpot95_vals = [s["latency_ms"]["tpot"]["p95"] for s in sums if s["latency_ms"]["tpot"]["p95"] is not None]
            tpot50 = median(tpot50_vals) if tpot50_vals else None
            tpot95 = median(tpot95_vals) if tpot95_vals else None
            tps = median([s["output_tokens_dispatch_per_second"] for s in sums])

            hbm_hit = median([sv.get("hbm_apc_hit_tokens") or 0 for sv in srvs])
            ext_hit = median([sv.get("external_prefix_hit_tokens") or 0 for sv in srvs])
            dram_hit = median([sv.get("dram_restore_token_equivalents") or 0 for sv in srvs])
            nvme_hit = median([sv.get("nvme_restore_token_equivalents") or 0 for sv in srvs])

            peaks = [sv.get("server_log_peaks") or {} for sv in srvs]
            pr = int(median([p.get("peak_running_reqs", 0) for p in peaks]))
            pw = int(median([p.get("peak_waiting_reqs", 0) for p in peaks]))
            pkv = median([p.get("peak_gpu_kv_cache_percent", 0.0) for p in peaks])
            preempt = int(median([sv.get("preemptions_total") or 0 for sv in srvs]))

            lines.append(
                f"| {exp} | {prof} | {trace} | {mseq} | {scale} | {system} | {abl or '-'} | {len(items)} "
                f"| {comp}/{coh} | {to} | {slo_pct:.1f}% | {gp:.3f} "
                f"| {ttft50:.0f}/{ttft95:.0f} | {_fmt(tpot50)}/{_fmt(tpot95)} | {tps:.1f} "
                f"| {hbm_hit/1e3:.0f}k/{ext_hit/1e3:.0f}k/{dram_hit/1e3:.0f}k/{nvme_hit/1e3:.0f}k "
                f"| {pr}/{pw}/{pkv:.1f}% | {preempt} |"
            )

    if failed:
        lines.extend(["", "## Failed Jobs", ""])
        for jid, info in sorted(failed.items()):
            lines.append(f"- `{jid}`: rc={info.get('exit_code')}, attempts={info.get('attempts')}, at={info.get('timestamp')}")

    import os
    tmp_md = results_dir / f".{report_name}.{os.getpid()}.tmp"
    tmp_md.write_text("\n".join(lines) + "\n")
    tmp_md.replace(results_dir / report_name)
    tmp_json = results_dir / f".stage1_live_status.{os.getpid()}.tmp"
    tmp_json.write_text(json.dumps({
        "updated": datetime.now().isoformat(timespec="seconds"),
        "completed_runs": len(runs),
        "failed_jobs": len(failed),
        "plan_size": len(plan_jobs) if plan_jobs is not None else None,
        "runs": [
            {
                "dir": r["dir"],
                "job": r["job"],
                "completed": r["summary"]["completed"],
                "planned_cohort": r["summary"]["planned_cohort"],
                "timeout_count": r["summary"].get("timeout_count", 0),
                "slo_attainment": r["summary"]["slo_attainment"],
                "goodput_dispatch": r["summary"]["goodput_dispatch_requests_per_second"],
                "ttft_p50_ms": r["summary"]["latency_ms"]["ttft"]["p50"],
                "ttft_p95_ms": r["summary"]["latency_ms"]["ttft"]["p95"],
                "tpot_p50_ms": r["summary"]["latency_ms"]["tpot"]["p50"],
                "tpot_p95_ms": r["summary"]["latency_ms"]["tpot"]["p95"],
                "hbm_apc_hit_tokens": r["server"].get("hbm_apc_hit_tokens"),
                "external_prefix_hit_tokens": r["server"].get("external_prefix_hit_tokens"),
                "dram_restore_token_equivalents": r["server"].get("dram_restore_token_equivalents"),
                "nvme_restore_token_equivalents": r["server"].get("nvme_restore_token_equivalents"),
                "server_log_peaks": r["server"].get("server_log_peaks"),
                "preemptions_total": r["server"].get("preemptions_total"),
            }
            for r in runs
        ],
    }, indent=2) + "\n")
    tmp_json.replace(results_dir / "stage1_live_status.json")


if __name__ == "__main__":
    import sys
    write_live_report(Path(sys.argv[1]))
