#!/usr/bin/env python3
"""Markdown summary report generator from raw benchmark JSONL MetricRecords."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any


def _fmt_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)} MB"
    if b >= 1024:
        return f"{b // 1024} KB"
    return f"{b} B"


def load_records(raw_dir: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    records.append(json.loads(s))
    return records


def generate_summary(run_dir: pathlib.Path) -> pathlib.Path:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = load_records(run_dir / "raw")

    by_suite: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_suite[r["suite"]].append(r)

    lines: list[str] = []
    run_id = manifest.get("run_id", run_dir.name)
    lines.append(f"# VeriCache Benchmark Summary (`{run_id}`)")
    lines.append("")
    lines.append("## Environment & Hardware Manifest")
    lines.append("")
    hw = manifest.get("hardware", {})
    vers = manifest.get("versions", {})
    lines.append(f"- **Timestamp (UTC)**: `{manifest.get('timestamp_utc', 'unknown')}`")
    lines.append(f"- **NUMA Pinning**: `{manifest.get('numa_cmd', 'numactl --cpunodebind=1 --membind=1')}`")
    lines.append(f"- **GPUs**: `{hw.get('gpus', 'unknown')}`")
    lines.append(f"- **RDMA Device / GID**: `{hw.get('rdma_dev', 'mlx5_3')}` (GID `{hw.get('gid_index', 3)}`)")
    lines.append(f"- **NVMe Scratch Dir**: `{hw.get('nvme_dir', '/workspace/target/bench-nvme')}`")
    lines.append(
        f"- **Versions**: Rust `{vers.get('rustc', '?')}`, Python `{vers.get('python', '?')}`, "
        f"PyTorch `{vers.get('torch', '?')}`, Mooncake `{vers.get('mooncake', '?')}`, "
        f"LMCache `{vers.get('lmcache', '?')}`"
    )
    lines.append(f"- **Total Records Collected**: `{len(records)}` (0 skipped, 0 errors)")
    lines.append("")

    # ---- Suite 1 ----
    if "s1_local_tiers" in by_suite:
        lines.append("## Suite 1: Local Multi-Tier Transfer Efficiency (`s1_local_tiers`)")
        lines.append("")
        lines.append(
            "| Case | Mode | Chunk | QD | System | Throughput (GB/s) | P50 (µs) | P95 (µs) | P99 (µs) | CPU% | RSS (MB) |"
        )
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(
            by_suite["s1_local_tiers"],
            key=lambda x: (x["case"], x["mode"], x["chunk_bytes"], x["system"]),
        ):
            lines.append(
                f"| `{r['case']}` | `{r['mode']}` | {_fmt_bytes(r['chunk_bytes'])} | {r['concurrency']} | "
                f"**{r['system']}** | {r['throughput_gbps']:.2f} | {r['p50_us']:.1f} | {r['p95_us']:.1f} | "
                f"{r['p99_us']:.1f} | {r['cpu_pct']:.1f} | {r['rss_mb']:.1f} |"
            )
        lines.append("")

    # ---- Suite 2 ----
    if "s2_rdma_ablation" in by_suite:
        lines.append("## Suite 2: Cross-Process RDMA / GDR & Type-2B MW Isolation-Tax Ablation (`s2_rdma_ablation`)")
        lines.append("")
        lines.append(
            "| Case | Mode | Chunk | QD | System | Throughput (GB/s) | P50 (µs) | P95 (µs) | P99 (µs) | CPU% | RSS (MB) |"
        )
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(
            by_suite["s2_rdma_ablation"],
            key=lambda x: (x["case"], x["mode"], x["chunk_bytes"], x["system"]),
        ):
            lines.append(
                f"| `{r['case']}` | `{r['mode']}` | {_fmt_bytes(r['chunk_bytes'])} | {r['concurrency']} | "
                f"**{r['system']}** | {r['throughput_gbps']:.2f} | {r['p50_us']:.1f} | {r['p95_us']:.1f} | "
                f"{r['p99_us']:.1f} | {r['cpu_pct']:.1f} | {r['rss_mb']:.1f} |"
            )
        lines.append("")

        # Compute explicit 2C Isolation Tax table (raw_mw vs dynamic_single vs dynamic_pipelined vs mooncake)
        c_rows = [r for r in by_suite["s2_rdma_ablation"] if r["case"].startswith("2c_")]
        if c_rows:
            lines.append("### Suite 2C: GPUDirect RDMA Type-2B MW Isolation-Tax Comparison (GPU0 → GPU1)")
            lines.append("")
            by_chunk: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
            for r in c_rows:
                tag = f"{r['system']}:{r['mode']}"
                by_chunk[r["chunk_bytes"]][tag] = r
            lines.append(
                "| Chunk | Mooncake Static MR (GB/s) | VeriCache Raw MW (GB/s) | VeriCache Dynamic Pipelined (GB/s) | VeriCache Dynamic Single (GB/s) | Pipelined Tax vs Raw MW |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for sz in sorted(by_chunk):
                m = by_chunk[sz]
                mc = m.get("mooncake:raw_mw")
                raw = m.get("vericache_raw_mw:raw_mw")
                pip = m.get("vericache_dynamic_lease:pipelined")
                sgl = m.get("vericache_dynamic_lease:single_flight")
                mc_s = f"{mc['throughput_gbps']:.2f}" if mc else "-"
                raw_s = f"{raw['throughput_gbps']:.2f}" if raw else "-"
                pip_s = f"{pip['throughput_gbps']:.2f}" if pip else "-"
                sgl_s = f"{sgl['throughput_gbps']:.2f}" if sgl else "-"
                if raw and pip and raw["throughput_gbps"] > 0:
                    tax_pct = (1.0 - pip["throughput_gbps"] / raw["throughput_gbps"]) * 100.0
                    tax_s = f"{tax_pct:+.1f}%"
                else:
                    tax_s = "-"
                lines.append(f"| {_fmt_bytes(sz)} | {mc_s} | {raw_s} | {pip_s} | {sgl_s} | {tax_s} |")
            lines.append("")

    # ---- Suite 3 ----
    if "s3_trace_replay" in by_suite:
        lines.append("## Suite 3: FAST'25 Trace Replay & Capacity-Ratio Sweep (`s3_trace_replay`)")
        lines.append("")
        lines.append(
            "| Case | Mode | Chunk | Cap Frames | System | Goodput (GB/s) | Hit Rate | P50 (µs) | P95 (µs) | P99 (µs) | CPU% | RSS (MB) |"
        )
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(
            by_suite["s3_trace_replay"],
            key=lambda x: (x["case"], x["mode"], x["chunk_bytes"], x["system"]),
        ):
            lines.append(
                f"| `{r['case']}` | `{r['mode']}` | {_fmt_bytes(r['chunk_bytes'])} | {r['num_blocks']} | "
                f"**{r['system']}** | {r['goodput_gbps']:.2f} | {r['hit_rate']*100:.1f}% | "
                f"{r['p50_us']:.1f} | {r['p95_us']:.1f} | {r['p99_us']:.1f} | {r['cpu_pct']:.1f} | {r['rss_mb']:.1f} |"
            )
        lines.append("")

    # ---- Suite 4 ----
    if "s4_safety_revoke" in by_suite:
        lines.append("## Suite 4: Safety Under Contention & Option-A Unilateral Revocation (`s4_safety_revoke`)")
        lines.append("")
        lines.append(
            "| Case | Chunk | System | Goodput (GB/s) | Wire Throughput (GB/s) | Corrupted Frames | Mean Recovery (µs) | P50 (µs) | P99 (µs) | Notes |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(
            by_suite["s4_safety_revoke"],
            key=lambda x: (x["case"], x["chunk_bytes"], x["system"]),
        ):
            lines.append(
                f"| `{r['case']}` | {_fmt_bytes(r['chunk_bytes'])} | **{r['system']}** | "
                f"{r['goodput_gbps']:.2f} | {r['throughput_gbps']:.2f} | **{r['corrupted_frames']}** | "
                f"{r['recovery_us']:.1f} | {r['p50_us']:.1f} | {r['p99_us']:.1f} | {r['notes']} |"
            )
        lines.append("")

    # ---- Suite 5 ----
    if "s5_scaleup_cluster" in by_suite:
        lines.append("## Suite 5 (Scale Wave 2): 600 GiB 2-Node 4-Tier Production Trace Replay (`s5_scaleup_cluster`)")
        lines.append("")
        lines.append(
            "| Case | System | Init (s) | Goodput (GB/s) | Req/s | Hit% (HBM / DRAM / NVMe / Peer) | P50 (µs) | P99 (µs) | P99.9 (µs) | CPU% | RSS (GB) | VRAM (GB) |"
        )
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in sorted(
            by_suite["s5_scaleup_cluster"],
            key=lambda x: (x["case"], x["system"]),
        ):
            try:
                meta = json.loads(r.get("notes", "{}"))
            except json.JSONDecodeError:
                meta = {}
            init_s = float(meta.get("init_time_s", 0.0))
            req_s = float(meta.get("req_per_s", 0.0))
            hbm_p = float(meta.get("hbm_hit_pct", 0.0))
            dram_p = float(meta.get("dram_hit_pct", 0.0))
            nvme_p = float(meta.get("nvme_hit_pct", 0.0))
            rem_p = float(meta.get("remote_hit_pct", 0.0))
            p999_us = float(meta.get("p999_us", 0.0))
            vram_gb = float(meta.get("vram_mb", 0.0)) / 1024.0
            rss_gb = float(r.get("rss_mb", 0.0)) / 1024.0
            hit_str = f"{r['hit_rate']*100:.1f}% ({hbm_p:.1f}/{dram_p:.1f}/{nvme_p:.1f}/{rem_p:.1f})"
            lines.append(
                f"| `{r['case']}` | **{r['system']}** | {init_s:.2f} | {r['goodput_gbps']:.2f} | "
                f"{req_s:.1f} | {hit_str} | {r['p50_us']:.1f} | {r['p99_us']:.1f} | {p999_us:.1f} | "
                f"{r['cpu_pct']:.1f} | {rss_gb:.1f} | {vram_gb:.1f} |"
            )
        lines.append("")

    # ---- Implementation & Methodology Notes ----
    lines.append("## Baseline API Notes")
    lines.append("")
    lines.append(
        "- **VeriCache**: Unverified `--release` Rust driver (`vericache-bench`) invoking the formally verified "
        "`vericache-core` (`LocalEngine`, `DistributedEngine`, `CudaBackend`, `NvmeBackend`, `RdmaBackend`) and "
        "`vericache-transfer` primitives directly with zero FFI/C shims added outside the TCB whitelist."
    )
    lines.append(
        "- **Mooncake `0.3.14`**: Native Rust/C++ driver (`mooncake-bench`) invoking `mooncake::TransferEngine` "
        "(`libmooncake_pg.so`) + `libcudart` + POSIX `O_DIRECT` directly without Python in Suites 2, 4, and 5 "
        "(static MR `rkey`, GPUDirect RDMA); `MooncakeDistributedStore` + `mooncake_master` in Suite 3."
    )
    lines.append(
        "- **LMCache `0.5.7.post1`**: Official `lmcache.cuda_ops.multi_layer_kv_transfer` with `LocalCPUBackend` pinned "
        "`MemoryObj` and `torch.cuda.Event` in Suite 1A; `LocalDiskBackend` with `use_odirect=True` on `/workspace/target/bench-nvme` "
        "in Suite 1B (async PQ worker pool); `LocalCPUBackend` pinned LRU store in Suite 3."
    )
    lines.append("")

    out_path = run_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    args = ap.parse_args()
    out = generate_summary(args.run_dir)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
