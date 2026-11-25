"""Prepare a deterministic Trace A window for the matched 1/2 GPU pilot."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median, quantiles

from prepare_traces import CHAT_TEMPLATE_MARGIN, CONTEXT_LENGTH, read_rows, write_rows


TRACE_A = "bailian_traceA"
SOURCE_COMMIT = "5f7439c51ec248a0c585f7d90a41a6f57773b912"
SOURCE_SHA256 = "07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17"
HASH_BLOCK_TOKENS = 16
WINDOW_REQUESTS = 317
NORMALIZED_DURATION_SECONDS = 60.0


def excluded_reason(row):
    if row["type"] != "text":
        return "non_text"
    prompt_length = row["input_length"]
    if prompt_length <= 0 or row["output_length"] <= 0:
        return "empty_input_or_output"
    if prompt_length + row["output_length"] + CHAT_TEMPLATE_MARGIN > CONTEXT_LENGTH:
        return "context_limit"
    expected_blocks = (prompt_length + HASH_BLOCK_TOKENS - 1) // HASH_BLOCK_TOKENS
    if len(row["hash_ids"]) != expected_blocks:
        return "hash_block_count"
    return None


def distribution(rows):
    stats = {}
    for key in ("input_length", "output_length"):
        samples = [row[key] for row in rows]
        stats[key] = {"p50": median(samples), "p95": quantiles(samples, n=20)[-1]}
    return stats


def select_window(rows):
    exclusion = [excluded_reason(row) for row in rows]
    selected = [(i, row) for i, row in enumerate(rows) if exclusion[i] is None]
    reference = distribution([row for _, row in selected])
    candidates = []
    for fraction in range(1, 10):
        start = len(selected) * fraction // 10
        window = selected[start:start + WINDOW_REQUESTS]
        if len(window) != WINDOW_REQUESTS:
            continue
        statistics = distribution([row for _, row in window])
        distance = sum(
            abs(statistics[key][quantile] - value) / max(value, 1)
            for key, stats in reference.items() for quantile, value in stats.items()
        )
        candidates.append((distance, start, window))
    _, _, window = min(candidates)
    start, stop = window[0][0], window[-1][0] + 1
    kept = [row for _, row in window]
    times = [row["timestamp"] for row in kept]
    seconds = times[-1] - times[0]
    assert seconds > 0
    factor = NORMALIZED_DURATION_SECONDS / seconds
    normalized = [
        {**row, "timestamp": round((row["timestamp"] - times[0]) * factor, 6)}
        for row in kept
    ]
    return normalized, {
        "source_rows": len(rows),
        "source_excluded_by_reason": dict(Counter(reason for reason in exclusion if reason)),
        "source_window_rows_zero_based": [start, stop],
        "source_window_seconds": [times[0], times[-1]],
        "source_window_excluded_by_reason": dict(
            Counter(reason for reason in exclusion[start:stop] if reason)
        ),
        "source_window_rows_retained": len(kept),
        "source_window_retained_requests_per_second": len(kept) / seconds,
        "normalized_duration_seconds": NORMALIZED_DURATION_SECONDS,
        "timestamp_stretch_before_replay_scaling": factor,
        "normalized_requests_per_second": len(kept) / NORMALIZED_DURATION_SECONDS,
        "source_eligible_lengths_tokens": reference,
        "window_lengths_tokens": distribution(kept),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.source.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == SOURCE_SHA256
    rows = read_rows(args.source)
    assert all(a["timestamp"] <= b["timestamp"] for a, b in zip(rows, rows[1:]))
    window, metadata = select_window(rows)
    write_rows(args.output / f"{TRACE_A}.jsonl", window)
    metadata.update({
        "source": "alibaba-edu/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "context_length": CONTEXT_LENGTH,
        "chat_template_margin": CHAT_TEMPLATE_MARGIN,
        "hash_block_tokens": HASH_BLOCK_TOKENS,
        "selection": "Among windows of 317 eligible text requests starting at each "
                     "decile, choose the closest p50 and p95 input/output lengths.",
    })
    (args.output / "window.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
