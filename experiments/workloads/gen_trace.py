#!/usr/bin/env python3
"""Generate deterministic synthetic KV-cache prefix-sharing traces for Suite 3 capacity sweeps."""

import argparse
import json
import math
import pathlib
import random


def generate_zipf_prefix_trace(
    out_path: pathlib.Path,
    num_requests: int = 512,
    num_sessions: int = 64,
    shared_prefix_len: int = 4,
    suffix_len: int = 8,
    alpha: float = 1.15,
    seed: int = 20261121,
) -> dict:
    rng = random.Random(seed)
    # Zipf weights over sessions so prefix reuse creates clear capacity-ratio knees at 10/30/50/80%.
    weights = [1.0 / ((i + 1) ** alpha) for i in range(num_sessions)]
    total_w = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / total_w
        cdf.append(acc)

    def sample_session() -> int:
        u = rng.random()
        for idx, c in enumerate(cdf):
            if u <= c:
                return idx
        return num_sessions - 1

    # Each session has a fixed shared prefix + a slowly growing conversation chain (multi-turn reuse).
    next_block_id = 10_000
    session_prefixes = {}
    session_turns = {s: 0 for s in range(num_sessions)}
    for s in range(num_sessions):
        # System prompt shared across groups of 8 sessions + per-session prefix
        sys_root = s // 8
        pref = [sys_root] + [next_block_id + i for i in range(shared_prefix_len - 1)]
        next_block_id += shared_prefix_len - 1
        session_prefixes[s] = pref

    unique_ids = set()
    ts_ms = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in range(num_requests):
            s = sample_session()
            turn = session_turns[s]
            session_turns[s] = turn + 1
            # Every 4th turn on a session introduces 2 new suffix blocks; otherwise reuses session prefix+recent turn
            base = list(session_prefixes[s])
            if turn == 0 or turn % 3 == 0:
                extra = [next_block_id + i for i in range(2)]
                next_block_id += 2
                if len(base) < shared_prefix_len + suffix_len:
                    session_prefixes[s].extend(extra)
                hash_ids = base + extra
            else:
                # Re-hit existing prefix with 1 fresh turn suffix block
                fresh = next_block_id
                next_block_id += 1
                hash_ids = base[: shared_prefix_len + suffix_len - 1] + [fresh]

            for h in hash_ids:
                unique_ids.add(h)
            rec = {
                "timestamp": ts_ms,
                "input_length": len(hash_ids) * 512,
                "output_length": 128,
                "hash_ids": hash_ids,
            }
            f.write(json.dumps(rec) + "\n")
            # Poisson-like inter-arrival gap in ms
            gap = max(1, int(-math.log(max(1e-9, rng.random())) * 8.0))
            ts_ms += gap

    return {
        "path": str(out_path),
        "num_requests": num_requests,
        "unique_blocks": len(unique_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/target/bench-traces"),
    )
    ap.add_argument("--num-requests", type=int, default=512)
    args = ap.parse_args()
    info = generate_zipf_prefix_trace(
        args.out_dir / "synthetic_zipf_trace.jsonl",
        num_requests=args.num_requests,
    )
    print(json.dumps(info))


if __name__ == "__main__":
    main()
