"""Prepare deterministic prefix-routed (K=8) Mooncake trace shards for Stage 1.

In production multi-instance serving (e.g., an 8-instance cluster with prefix-aware
routing on the per-session prefix block hash_ids[1] % 8), each GPU instance serves
1/8 of the cluster's sessions. Using a 350-request prefix-routed shard (115 warmup
+ 235 measurement, timestamps proportionally normalized to 1.0 req/s nominal):
- Warmup populates ~70-77 GiB of unique KV (> 19.5 GiB HBM and > 36 GiB Profile B DRAM),
- Measurement exhibits ~18% HBM hit, ~24-25% DRAM-36G hit, ~42-43% Profile A (136G DRAM)
  hit (+24-26% DRAM gain over HBM), and ~17.5% NVMe hit in Profile B (36G DRAM + 80G NVMe).
"""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from statistics import median

from prepare_traces import BYTES_PER_KV_TOKEN, CHAT_TEMPLATE_MARGIN, CONTEXT_LENGTH, TRACE_BLOCK_SIZE, fits, read_rows, write_rows


ROUTED_TRACES = ("toolagent_routed", "conversation_routed")
SOURCE_MAP = {
    "toolagent_routed": "toolagent_trace.jsonl",
    "conversation_routed": "conversation_trace.jsonl",
}
SHARD_K = 8
WINDOW_LEN = 350
WARMUP_COUNT = 115
NOMINAL_DURATION_MS = 350_000


def select_routed_window(rows):
    elig = [r for r in rows if fits(r)]
    buckets = {}
    for r in elig:
        h = r["hash_ids"]
        key = h[1] if len(h) > 1 else h[0]
        buckets.setdefault(key % SHARD_K, []).append(r)
    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]))
    shard_id, bucket = ordered[len(ordered) // 2]
    kept = bucket[:WINDOW_LEN]
    assert len(kept) == WINDOW_LEN
    t0, t1 = kept[0]["timestamp"], kept[-1]["timestamp"]
    span = t1 - t0
    assert span > 0
    scale = NOMINAL_DURATION_MS / span
    normalized = [
        {**r, "timestamp": int(round((r["timestamp"] - t0) * scale))}
        for r in kept
    ]
    warmup_end_ms = normalized[WARMUP_COUNT - 1]["timestamp"]
    # Ensure strict separation if two consecutive rows rounded to same ms
    while (
        WARMUP_COUNT < len(normalized)
        and normalized[WARMUP_COUNT]["timestamp"] <= warmup_end_ms
    ):
        normalized[WARMUP_COUNT]["timestamp"] = warmup_end_ms + 1

    # Simulate LRU hit curves for metadata audit
    hit_curve = {}
    for cap_gib in (19.5, 36.0, 116.0, 136.0):
        cb = int(cap_gib * (1 << 30) // (TRACE_BLOCK_SIZE * BYTES_PER_KV_TOKEN))
        lru = OrderedDict()
        mp = mh = 0
        for i, r in enumerate(normalized):
            h = tuple(r["hash_ids"])
            matched = True
            hits = 0
            for d in range(1, len(h) + 1):
                k = h[:d]
                if matched and k in lru:
                    lru.move_to_end(k)
                    hits += TRACE_BLOCK_SIZE
                else:
                    matched = False
                    if k in lru:
                        lru.move_to_end(k)
                    else:
                        lru[k] = True
                        if len(lru) > cb:
                            lru.popitem(last=False)
            hits = min(hits, r["input_length"])
            if i >= WARMUP_COUNT:
                mp += r["input_length"]
                mh += hits
        hit_curve[str(cap_gib)] = mh / max(1, mp)

    meta = {
        "shard_k": SHARD_K,
        "selected_shard_id": shard_id,
        "rows_retained": len(normalized),
        "warmup_count": WARMUP_COUNT,
        "measurement_count": len(normalized) - WARMUP_COUNT,
        "warmup_end_ms": warmup_end_ms,
        "source_span_seconds": span / 1000.0,
        "normalized_duration_ms": normalized[-1]["timestamp"],
        "median_input_tokens": median(r["input_length"] for r in normalized),
        "median_output_tokens": median(r["output_length"] for r in normalized),
        "lru_measurement_hit_curve": hit_curve,
    }
    return normalized, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"traces": {}}
    for name, src_name in SOURCE_MAP.items():
        src = args.source / src_name
        rows = read_rows(src)
        window, meta = select_routed_window(rows)
        out_path = args.output / f"{name}.jsonl"
        write_rows(out_path, window)
        meta["source_file"] = src_name
        meta["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
        metadata["traces"][name] = meta
    (args.output / "windows.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
