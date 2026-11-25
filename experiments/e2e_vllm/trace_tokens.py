#!/usr/bin/env python3
"""Deterministic FAST'25 Mooncake trace -> raw token_ids converter for vLLM E2E replay.

Each Mooncake trace entry contains:
  {"timestamp": <ms>, "input_length": <int>, "output_length": <int>, "hash_ids": [h0, h1, ...]}

To guarantee 100% bit-exact prefix sharing across vLLM Baseline, VeriCache,
LMCache, and MooncakeStore without any BPE boundary drift:
  - Each unique `hash_id` deterministically maps to `tokens_per_block` ordinary
    (non-special, non-EOS) token IDs from the target model's tokenizer vocabulary.
  - Two requests sharing prefix `hash_ids[:k]` therefore have strictly identical
    leading token IDs, yielding identical vLLM prefix block hashes!
  - Requests are sent to `/v1/completions` as raw `list[int]` token IDs.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
from typing import Any


def build_safe_vocab(tokenizer_or_dir: Any) -> list[int]:
    """Build a deterministic list of ordinary non-special token IDs for `model_dir`."""
    if isinstance(tokenizer_or_dir, (str, pathlib.Path)):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tokenizer_or_dir), trust_remote_code=True)
    else:
        tok = tokenizer_or_dir

    special = set(getattr(tok, "all_special_ids", []) or [])
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
        val = getattr(tok, attr, None)
        if isinstance(val, int):
            special.add(val)
        elif isinstance(val, (list, tuple)):
            special.update(int(x) for x in val)

    # Use safe ordinary ASCII/word token range [1000, min(vocab_size, 65000)) excluding special IDs.
    vocab_size = int(getattr(tok, "vocab_size", 32000))
    hi = min(vocab_size - 1, 65000)
    lo = min(1000, max(10, hi // 10))
    safe = [tid for tid in range(lo, hi) if tid not in special]
    if len(safe) < 1024:
        raise RuntimeError(f"Safe vocab too small ({len(safe)}) for tokenizer")
    return safe


def hash_id_to_tokens(hash_id: int, tokens_per_block: int, safe_vocab: list[int]) -> list[int]:
    """Deterministically expand one Mooncake `hash_id` into `tokens_per_block` safe token IDs."""
    n_vocab = len(safe_vocab)
    # Derive a 32-byte seed from hash_id and expand via counter-mode SHA-256 -> uint32 stream.
    out: list[int] = []
    ctr = 0
    while len(out) < tokens_per_block:
        digest = hashlib.sha256(struct.pack("<QI", int(hash_id) & 0xFFFFFFFFFFFFFFFF, ctr)).digest()
        ctr += 1
        for val in struct.unpack("<8I", digest):
            out.append(safe_vocab[val % n_vocab])
            if len(out) >= tokens_per_block:
                break
    return out


def prepare_trace_requests(
    trace_path: str | pathlib.Path,
    safe_vocab: list[int],
    *,
    start_index: int = 0,
    max_requests: int = 200,
    tokens_per_hash_id: int = 512,
    max_input_tokens: int = 16384,
    fixed_output_tokens: int | None = None,
    max_output_tokens: int = 128,
    min_output_tokens: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a contiguous window of FAST'25 trace entries and convert them to raw token IDs."""
    raw_reqs: list[dict[str, Any]] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        seen = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            hids = obj.get("hash_ids")
            if not hids:
                continue
            if seen < start_index:
                seen += 1
                continue
            raw_reqs.append(obj)
            seen += 1
            if max_requests > 0 and len(raw_reqs) >= max_requests:
                break

    if not raw_reqs:
        raise RuntimeError(f"No trace requests loaded from {trace_path}")

    max_blocks = max(1, max_input_tokens // tokens_per_hash_id)
    block_cache: dict[int, list[int]] = {}
    prepared: list[dict[str, Any]] = []
    base_ts = int(raw_reqs[0].get("timestamp", 0))

    total_blocks = 0
    unique_hids: set[int] = set()
    # Track theoretical infinite-cache prefix hit rate on this exact window
    seen_prefix_hids: set[int] = set()
    ideal_prefix_hits = 0

    for idx, obj in enumerate(raw_reqs):
        hids = [int(h) for h in obj["hash_ids"][:max_blocks]]
        token_ids: list[int] = []
        prefix_open = True
        for h in hids:
            toks = block_cache.get(h)
            if toks is None:
                toks = hash_id_to_tokens(h, tokens_per_hash_id, safe_vocab)
                block_cache[h] = toks
            token_ids.extend(toks)
            unique_hids.add(h)
            if prefix_open and h in seen_prefix_hids:
                ideal_prefix_hits += 1
            else:
                prefix_open = False
                seen_prefix_hids.add(h)
        total_blocks += len(hids)

        if fixed_output_tokens is not None:
            out_len = int(fixed_output_tokens)
        else:
            raw_out = int(obj.get("output_length", min_output_tokens))
            out_len = max(min_output_tokens, min(max_output_tokens, raw_out))

        prepared.append(
            {
                "req_idx": idx,
                "timestamp_ms": max(0, int(obj.get("timestamp", base_ts)) - base_ts),
                "hash_ids": hids,
                "prompt_token_ids": token_ids,
                "input_tokens": len(token_ids),
                "max_tokens": out_len,
            }
        )

    summary = {
        "trace_path": str(trace_path),
        "start_index": start_index,
        "num_requests": len(prepared),
        "tokens_per_hash_id": tokens_per_hash_id,
        "total_blocks": total_blocks,
        "unique_blocks": len(unique_hids),
        "total_input_tokens": sum(r["input_tokens"] for r in prepared),
        "avg_input_tokens": sum(r["input_tokens"] for r in prepared) / len(prepared),
        "max_input_tokens": max(r["input_tokens"] for r in prepared),
        "avg_output_tokens": sum(r["max_tokens"] for r in prepared) / len(prepared),
        "ideal_prefix_hit_rate": ideal_prefix_hits / max(1, total_blocks),
        "trace_span_s": (prepared[-1]["timestamp_ms"] - prepared[0]["timestamp_ms"]) / 1000.0,
    }
    return prepared, summary
