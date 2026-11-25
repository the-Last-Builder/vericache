#!/usr/bin/env python3
"""Freeze the first qualifying contiguous ToolAgent window using prompt-block LRU simulation."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Sequence
from itertools import islice
import hashlib
import json
import pathlib

from trace_tokens import FRAME_BYTES, build_safe_vocab, iter_trace_requests, prefix_block_keys

GIB = 1 << 30
WARMUP = 500
MEASURE = 1500
CAPACITY = tuple(gib * GIB // FRAME_BYTES for gib in (16, 32, 64))


class TieredLRU:
    def __init__(self, capacities: tuple[int, int, int]):
        self.tiers = [OrderedDict() for _ in capacities]
        self.capacities = capacities

    def location(self, key: bytes) -> int | None:
        return next((i for i, tier in enumerate(self.tiers) if key in tier), None)

    def admit(self, key: bytes) -> bool:
        """Touch/promote in HBM and cascade displaced frames down. Return SSD eviction."""
        old = self.location(key)
        if old is not None:
            del self.tiers[old][key]
        for tier, cap in zip(self.tiers, self.capacities):
            tier[key] = None
            if len(tier) <= cap:
                return False
            key, _ = tier.popitem(last=False)
        return True


def evaluate_window(rows: Iterable[tuple[dict, tuple[bytes, ...]]], warmup: int, capacities=CAPACITY):
    caches = [TieredLRU(capacities), TieredLRU(capacities)]
    seen = [set(), set()]
    cross_reused = [set(), set()]
    warmup_nvme = [0, 0]
    stats = [Counter(), Counter()]
    for i, (req, keys) in enumerate(rows):
        node = req["dest_node"]
        local, peer = caches[node], caches[1 - node]
        counts = stats[node] if i >= warmup else Counter()
        counts["requests"] += 1
        nvme_blocks = remote_blocks = 0
        # vLLM keeps at least the last prompt token for computation, even on a full hit.
        for key in islice(keys, (req["input_tokens"] - 1) // 16):
            local_tier = local.location(key)
            if local_tier is not None:
                counts[("hbm_blocks", "dram_blocks", "local_nvme_blocks")[local_tier]] += 1
                nvme_blocks += int(local_tier == 2)
                continue
            peer_tier = peer.location(key)
            if peer_tier not in (1, 2):
                break
            counts["remote_dram_blocks" if peer_tier == 1 else "remote_nvme_blocks"] += 1
            remote_blocks += 1
            nvme_blocks += int(peer_tier == 2)
            if key not in cross_reused[node]:
                counts["first_cross_node_blocks"] += 1
                cross_reused[node].add(key)

        counts["nvme_reuse_blocks"] += nvme_blocks
        counts["cross_node_blocks"] += remote_blocks
        counts["nvme_reuse_requests"] += int(nvme_blocks > 0)
        counts["cross_node_requests"] += int(remote_blocks > 0)
        for key in keys:
            if key not in seen[0] and key not in seen[1]:
                counts["new_blocks"] += 1
            seen[node].add(key)
            counts["nvme_evictions"] += int(local.admit(key))
        if i + 1 == warmup:
            warmup_nvme = [len(cache.tiers[2]) for cache in caches]
            warmup_seen = [len(keys_seen) for keys_seen in seen]
    return {
        "warmup_distinct_blocks": warmup_seen if warmup else [0, 0],
        "warmup_nvme_frames": warmup_nvme,
        "measurement_per_node": [
            dict(node, new_bytes=node["new_blocks"] * FRAME_BYTES,
                 first_cross_node_bytes=node["first_cross_node_blocks"] * FRAME_BYTES)
            for node in stats
        ],
    }


def select_window(requests: Iterable[dict], warmup: int, measure: int,
                  min_blocks: int, min_requests: int, capacities=CAPACITY):
    """Slide by one request. Skip simulation only if exact warmup pressure is impossible."""
    rows: deque[tuple[dict, tuple[bytes, ...]]] = deque()
    warm_refs = [Counter(), Counter()]

    def update_warm(row, increase: bool) -> None:
        req, keys = row
        refs = warm_refs[req["dest_node"]]
        if increase:
            refs.update(keys)
        else:
            for key in keys:
                refs[key] -= 1
                if refs[key] == 0:
                    del refs[key]

    for req in requests:
        row = req, prefix_block_keys(req["prompt_token_ids"])
        rows.append(row)
        if len(rows) <= warmup:
            update_warm(row, True)
        if len(rows) < warmup + measure:
            continue
        if all(len(refs) > sum(capacities[:2]) for refs in warm_refs):
            summary = evaluate_window(rows, warmup, capacities)
            if all(summary["warmup_nvme_frames"]) and all(
                stats["nvme_reuse_blocks"] >= min_blocks
                and stats["cross_node_blocks"] >= min_blocks
                and stats["nvme_reuse_requests"] >= min_requests
                and stats["cross_node_requests"] >= min_requests
                for stats in summary["measurement_per_node"]
            ):
                base_ms = rows[0][0]["timestamp_ms"]
                frozen = [
                    dict(req, is_warmup=i < warmup, timestamp_ms=req["timestamp_ms"] - base_ms)
                    for i, (req, _) in enumerate(rows)
                ]
                return frozen, summary
        update_warm(rows.popleft(), False)
        update_warm(rows[warmup - 1], True)
    raise ValueError("No contiguous window meets both-node pressure and reuse thresholds")


def add_turnover_rates(summary: dict, frozen: Sequence[dict], warmup: int, qps: float) -> None:
    measured = frozen[warmup:]
    # Evaluate the frozen ToolAgent cohort for Exp 1.1 at the specified offered rate.
    span = (measured[-1]["timestamp_ms"] - measured[0]["timestamp_ms"]) / 1000
    duration = len(measured) / qps
    summary["turnover"] = {"assumed_global_qps": qps, "trace_measurement_span_s": span}
    for node in summary["measurement_per_node"]:
        node["new_gib_s"] = node["new_bytes"] / GIB / duration
        node["first_cross_node_gib_s"] = node["first_cross_node_bytes"] / GIB / duration


def sha256_file(path: pathlib.Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=pathlib.Path)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--min-reuse-mib", type=int, default=256)
    parser.add_argument("--min-reuse-requests", type=int, default=10)
    parser.add_argument("--turnover-qps", type=float, default=1.0)
    parser.add_argument("--evaluate-frozen", type=pathlib.Path)
    args = parser.parse_args()
    if min(args.min_reuse_mib, args.min_reuse_requests, args.turnover_qps) <= 0:
        parser.error("Reuse thresholds and turnover QPS must be positive")
    min_blocks = (args.min_reuse_mib * (1 << 20) + FRAME_BYTES - 1) // FRAME_BYTES
    if args.evaluate_frozen:
        from replayer import load_frozen_window

        frozen = load_frozen_window(args.evaluate_frozen, WARMUP, MEASURE)
        summary = evaluate_window(
            ((req, prefix_block_keys(req["prompt_token_ids"])) for req in frozen), WARMUP,
        )
        source = args.evaluate_frozen
    else:
        if args.trace_path is None or args.model_path is None:
            parser.error("--trace-path and --model-path are required when freezing a trace")
        source = args.trace_path
        requests = iter_trace_requests(source, build_safe_vocab(args.model_path), max_input_tokens=32767)
        frozen, summary = select_window(requests, WARMUP, MEASURE, min_blocks, args.min_reuse_requests)
    add_turnover_rates(summary, frozen, WARMUP, args.turnover_qps)
    summary.update(
        estimator="serial exclusive prompt-block LRU, vLLM sha256 seed 0; generated KV omitted",
        source=str(source), source_sha256=sha256_file(source),
        window_start=frozen[0]["req_idx"], window_end_exclusive=frozen[-1]["req_idx"] + 1,
        warmup_requests=WARMUP, measurement_requests=MEASURE,
        capacities_frames=dict(zip(("hbm", "dram", "nvme"), CAPACITY)),
        frame_bytes=FRAME_BYTES, block_size=16, sticky_ratio=0.70,
        session_key_rule="hash_ids[1] else hash_ids[-1]",
        min_reuse_blocks=min_blocks, min_reuse_requests=args.min_reuse_requests,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    window_path, summary_path = out / "step1_window.jsonl", out / "summary.json"
    with window_path.open("w", encoding="utf-8") as file:
        for req in frozen:
            file.write(json.dumps(req, separators=(",", ":")) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in (window_path, summary_path)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
