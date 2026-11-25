#!/usr/bin/env python3
"""Build frozen high-turnover and BI=1 exactness cohorts from real ToolAgent prompts for Step 1 Exp 1.1 & 1.2."""

from __future__ import annotations

import argparse
import json
import pathlib

from prepare_step1_window import CAPACITY, GIB, add_turnover_rates, evaluate_window, sha256_file
from trace_tokens import FRAME_BYTES, build_safe_vocab, iter_trace_requests, prefix_block_keys


def collect_unique_prompts(
    trace_path: pathlib.Path,
    model_path: str,
    needed: int,
    *,
    min_input_tokens: int,
    max_input_tokens: int,
) -> list[dict]:
    """Extract real ToolAgent requests with distinct full-block prefixes within token bounds."""
    vocab = build_safe_vocab(model_path)
    seen_full: set[tuple[bytes, ...]] = set()
    selected: list[dict] = []
    for req in iter_trace_requests(trace_path, vocab, max_input_tokens=max_input_tokens):
        n_tok = req["input_tokens"]
        if n_tok < min_input_tokens or n_tok > max_input_tokens:
            continue
        keys = prefix_block_keys(req["prompt_token_ids"])
        if not keys or keys in seen_full:
            continue
        seen_full.add(keys)
        selected.append(req)
        if len(selected) >= needed:
            break
    if len(selected) < needed:
        raise ValueError(
            f"Only found {len(selected)} unique prompts in [{min_input_tokens}, {max_input_tokens}] "
            f"(needed {needed})"
        )
    return selected


def build_turnover_cohort(
    pool: list[dict],
    warmup_donor: int,
    warmup_requester: int,
    measure_pairs: int,
    *,
    dram_lag_min: int = 8,
    dram_lag_max: int = 24,
    nvme_lag: int = 55,
    nvme_every: int = 5,
    step_ms: int = 200,
) -> list[dict]:
    """Interleave Donor (Node 0) new-prefix admissions with Requester (Node 1) first-time cross-node reads.

    Warmup populates Donor (Node 0) through HBM + DRAM + NVMe and primes Requester (Node 1).
    Measurement alternates:
      - Node 0: brand-new unique real prompt (forces continuous DRAM admission & eviction on Donor).
      - Node 1: first-time cross-node read of a prefix admitted to Node 0 `lag` admissions earlier
        (mostly in Node 0 DRAM, periodically in Node 0 NVMe).
    """
    needed_unique = warmup_donor + warmup_requester + measure_pairs
    if len(pool) < needed_unique:
        raise ValueError(f"Pool has {len(pool)} prompts, need {needed_unique}")

    donor_history: list[dict] = []
    cross_sent: set[int] = set()
    rows: list[dict] = []
    pool_idx = 0
    req_idx = 0

    def emit(src: dict, dest_node: int, is_warmup: bool) -> None:
        nonlocal req_idx
        rows.append(
            {
                "req_idx": req_idx,
                "timestamp_ms": req_idx * step_ms,
                "prompt_token_ids": src["prompt_token_ids"],
                "input_tokens": src["input_tokens"],
                "output_tokens": src["output_tokens"],
                "hash_ids": src["hash_ids"],
                "session_key": src["session_key"],
                "dest_node": dest_node,
                "is_warmup": is_warmup,
            }
        )
        req_idx += 1

    # Warmup: alternate donor/requester early, then finish filling donor DRAM + NVMe.
    for i in range(warmup_donor + warmup_requester):
        src = pool[pool_idx]
        pool_idx += 1
        if i < warmup_requester * 2 and i % 2 == 1:
            emit(src, 1, True)
        else:
            emit(src, 0, True)
            donor_history.append(src)

    # Measurement: alternating Donor new-prefix admission & Requester rotating first-time cross-node read.
    lag_span = max(1, dram_lag_max - dram_lag_min + 1)
    for step in range(measure_pairs):
        # 1) Donor Node 0 admits a brand-new unique prompt.
        new_src = pool[pool_idx]
        pool_idx += 1
        emit(new_src, 0, False)
        donor_history.append(new_src)

        # 2) Requester Node 1 performs a first-time cross-node read from Node 0's DRAM (or periodically NVMe).
        use_nvme = nvme_every > 0 and (step % nvme_every == nvme_every - 1) and len(donor_history) > nvme_lag
        if use_nvme:
            target_idx = len(donor_history) - 1 - nvme_lag
        else:
            lag = dram_lag_min + (step % lag_span)
            target_idx = max(0, len(donor_history) - 1 - lag)
        # Ensure first-time cross-node lookup (never previously requested by Node 1).
        while target_idx in cross_sent and target_idx > 0:
            target_idx -= 1
        if target_idx in cross_sent:
            for cand in range(len(donor_history) - 2, -1, -1):
                if cand not in cross_sent:
                    target_idx = cand
                    break
        cross_sent.add(target_idx)
        emit(donor_history[target_idx], 1, False)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", type=pathlib.Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--warmup-donor", type=int, default=140)
    parser.add_argument("--warmup-requester", type=int, default=60)
    parser.add_argument("--measure-pairs", type=int, default=200)
    parser.add_argument("--min-input-tokens", type=int, default=10240)
    parser.add_argument("--max-input-tokens", type=int, default=28672)
    parser.add_argument("--dram-lag-min", type=int, default=10)
    parser.add_argument("--dram-lag-max", type=int, default=24)
    parser.add_argument("--nvme-lag", type=int, default=55)
    parser.add_argument("--nvme-every", type=int, default=5)
    parser.add_argument("--turnover-qps", type=float, default=2.0)
    args = parser.parse_args()

    needed = args.warmup_donor + args.warmup_requester + args.measure_pairs
    pool = collect_unique_prompts(
        args.trace_path,
        args.model_path,
        needed,
        min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    frozen = build_turnover_cohort(
        pool,
        args.warmup_donor,
        args.warmup_requester,
        args.measure_pairs,
        dram_lag_min=args.dram_lag_min,
        dram_lag_max=args.dram_lag_max,
        nvme_lag=args.nvme_lag,
        nvme_every=args.nvme_every,
    )
    warmup = args.warmup_donor + args.warmup_requester
    measure = args.measure_pairs * 2
    summary = evaluate_window(
        ((req, prefix_block_keys(req["prompt_token_ids"])) for req in frozen),
        warmup,
        capacities=CAPACITY,
    )
    add_turnover_rates(summary, frozen, warmup, args.turnover_qps)
    summary.update(
        cohort_type="exp1_high_turnover_cross_node",
        source=str(args.trace_path),
        source_sha256=sha256_file(args.trace_path),
        warmup_requests=warmup,
        measurement_requests=measure,
        warmup_donor=args.warmup_donor,
        warmup_requester=args.warmup_requester,
        measure_pairs=args.measure_pairs,
        min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens,
        dram_lag_range=[args.dram_lag_min, args.dram_lag_max],
        nvme_lag=args.nvme_lag,
        nvme_every=args.nvme_every,
        capacities_frames=dict(zip(("hbm", "dram", "nvme"), CAPACITY)),
        frame_bytes=FRAME_BYTES,
        block_size=16,
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
