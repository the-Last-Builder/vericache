"""Select and stage contiguous FAST'25 trace windows for Stage 0 calibration."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


TRACES = ("toolagent_trace", "conversation_trace")
TRACE_BLOCK_SIZE = 512
SERVING_BLOCK_SIZE = 64
CONTEXT_LENGTH = 32768
CHAT_TEMPLATE_MARGIN = 32  # request-sim OpenAI context-length rule
BYTES_PER_KV_TOKEN = 2 * 4 * 128 * 2 * 48
TARGET_BYTES = 130 << 30


def read_rows(path):
    with path.open() as source:
        return [json.loads(row) for row in source]


def fits(row):
    return row["input_length"] + row["output_length"] + CHAT_TEMPLATE_MARGIN <= CONTEXT_LENGTH


def statistics(rows):
    return {
        key: median(row[key] for row in rows)
        for key in ("input_length", "output_length")
    }


def choose_window(rows):
    reference = statistics([row for row in rows if fits(row)])
    candidates = []
    for numerator in (1, 2, 3):
        start = len(rows) * numerator // 4
        prefixes = set()
        used_tokens = 0
        retained = []
        for stop in range(start, len(rows)):
            row = rows[stop]
            if not fits(row):
                continue
            retained.append(row)
            length = row["input_length"]
            full_size, tail = divmod(length, TRACE_BLOCK_SIZE)
            hashes = tuple(row["hash_ids"])
            assert len(hashes) == full_size + bool(tail)
            for depth in range(1, full_size + 1):
                prefix = hashes[:depth]
                if prefix not in prefixes:
                    prefixes.add(prefix)
                    used_tokens += TRACE_BLOCK_SIZE
            used_tokens += (tail + SERVING_BLOCK_SIZE - 1) // SERVING_BLOCK_SIZE * SERVING_BLOCK_SIZE
            used_bytes = used_tokens * BYTES_PER_KV_TOKEN
            if used_bytes >= TARGET_BYTES:
                medians = statistics(retained)
                distance = sum(abs(medians[key] - value) / value for key, value in reference.items())
                candidates.append((distance, start, stop + 1, used_bytes, retained, medians))
                break
    if not candidates:
        raise ValueError("no contiguous window reached the 130 GiB target")
    _, start, stop, used_bytes, kept, medians = min(candidates)
    assert 100 << 30 <= used_bytes <= 160 << 30
    origin = kept[0]["timestamp"]
    normalized = [{**row, "timestamp": row["timestamp"] - origin} for row in kept]
    filtered = stop - start - len(normalized)
    report = {
        "source_rows": len(rows),
        "source_context_filtered": sum(not fits(row) for row in rows),
        "source_max_input_tokens": max(row["input_length"] for row in rows),
        "source_medians_tokens": reference,
        "source_window_start_zero_based": start,
        "source_window_stop_exclusive": stop,
        "source_window_rows": stop - start,
        "window_context_filtered": filtered,
        "window_context_filtered_percent": 100 * filtered / (stop - start),
        "rows_retained": len(normalized),
        "median_tokens": medians,
        "duration_ms": normalized[-1]["timestamp"],
        "estimated_unique_prefill_kv_bytes": used_bytes,
        "estimated_unique_prefill_kv_gib": used_bytes / (1 << 30),
    }
    return normalized, report


def write_rows(path, rows):
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "context_length": CONTEXT_LENGTH,
        "reserved_chat_template_tokens": CHAT_TEMPLATE_MARGIN,
        "trace_hash_block_tokens": TRACE_BLOCK_SIZE,
        "serving_block_tokens": SERVING_BLOCK_SIZE,
        "kv_bytes_per_token": BYTES_PER_KV_TOKEN,
        "window_selection": "Start at a trace quartile, extend to 130 GiB, "
                            "choose the closest input/output medians to the filtered source.",
        "working_set_method": "Unique full-block prefix paths plus per-request "
                              "partial tails rounded to the vLLM block size.",
        "traces": {},
    }
    for name in TRACES:
        source = args.source / f"{name}.jsonl"
        rows, stats = choose_window(read_rows(source))
        write_rows(args.output / source.name, rows)
        stats["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata["traces"][name] = stats
    (args.output / "windows.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
