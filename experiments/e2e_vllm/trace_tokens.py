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
import pickle
import struct
from functools import lru_cache
from itertools import islice
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


FRAME_BYTES = 16 * 96 * 1024
MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def route_request(req_idx: int, hash_ids: list[int], sticky_ratio: float = 0.70) -> tuple[int, int]:
    session_key = hash_ids[1] if len(hash_ids) > 1 else hash_ids[-1]
    home = splitmix64(session_key) & 1
    drift = splitmix64(req_idx ^ ((session_key * 0x9E3779B97F4A7C15) & MASK64))
    return session_key, home if drift % 100 < round(sticky_ratio * 100) else 1 - home


def sha256_vllm(value: Any) -> bytes:
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).digest()


def prefix_block_keys(token_ids: list[int], block_size: int = 16) -> tuple[bytes, ...]:
    """vLLM SHA-256 with PYTHONHASHSEED=0 and no salt, adapters or multimodal keys."""
    parent = sha256_vllm("0")
    keys = []
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = sha256_vllm((parent, tuple(token_ids[start:start + block_size]), None))
        keys.append(parent)
    return tuple(keys)


def iter_trace_requests(
    trace_path: str | pathlib.Path,
    safe_vocab: list[int],
    *,
    tokens_per_hash_id: int = 512,
    max_input_tokens: int = 32768,
    sticky_ratio: float = 0.70,
):
    if tokens_per_hash_id < 1 or max_input_tokens < 1:
        raise ValueError("Token limits must be positive")

    @lru_cache(maxsize=4096)
    def expand(hash_id: int) -> list[int]:
        return hash_id_to_tokens(hash_id, tokens_per_hash_id, safe_vocab)

    with open(trace_path, encoding="utf-8") as source:
        req_idx = 0
        previous_ts = None
        for line in source:
            if not line.strip():
                continue
            raw = json.loads(line)
            hash_ids = [int(h) for h in raw["hash_ids"]]
            input_tokens = min(int(raw["input_length"]), max_input_tokens)
            output_tokens = int(raw["output_length"])
            timestamp_ms = int(raw["timestamp"])
            if not (0 < input_tokens <= len(hash_ids) * tokens_per_hash_id) or output_tokens < 1:
                raise ValueError(f"Request {req_idx} has inconsistent input/output lengths")
            if previous_ts is not None and timestamp_ms < previous_ts:
                raise ValueError(f"Request {req_idx} is out of timestamp order")
            previous_ts = timestamp_ms
            session_key, dest_node = route_request(req_idx, hash_ids, sticky_ratio)
            token_ids: list[int] = []
            for hash_id in hash_ids[:(input_tokens + tokens_per_hash_id - 1) // tokens_per_hash_id]:
                token_ids.extend(expand(hash_id))
            yield {
                "req_idx": req_idx,
                "timestamp_ms": timestamp_ms,
                "prompt_token_ids": token_ids[:input_tokens],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "hash_ids": hash_ids,
                "session_key": session_key,
                "dest_node": dest_node,
            }
            req_idx += 1


def prepare_trace_requests(
    trace_path: str | pathlib.Path,
    safe_vocab: list[int],
    *,
    start_index: int = 0,
    max_requests: int = 200,
    tokens_per_hash_id: int = 512,
    max_input_tokens: int = 32768,
    fixed_output_tokens: int | None = None,
    max_output_tokens: int | None = None,
    min_output_tokens: int = 1,
    sticky_ratio: float = 0.70,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a contiguous input window, preserving partial hash blocks."""
    stop = start_index + max_requests if max_requests > 0 else None
    requests = list(islice(iter_trace_requests(
        trace_path, safe_vocab,
        tokens_per_hash_id=tokens_per_hash_id,
        max_input_tokens=max_input_tokens,
        sticky_ratio=sticky_ratio,
    ), start_index, stop))
    if not requests:
        raise RuntimeError(f"No trace requests loaded from {trace_path}")

    base_ts = requests[0]["timestamp_ms"]
    for req in requests:
        req["timestamp_ms"] -= base_ts
        output = req["output_tokens"]
        if fixed_output_tokens is not None:
            output = fixed_output_tokens
        elif max_output_tokens is not None:
            output = min(max_output_tokens, max(min_output_tokens, output))
        req["max_tokens"] = output

    summary = {
        "trace_path": str(trace_path),
        "start_index": start_index,
        "num_requests": len(requests),
        "total_input_tokens": sum(r["input_tokens"] for r in requests),
        "avg_input_tokens": sum(r["input_tokens"] for r in requests) / len(requests),
        "max_input_tokens": max(r["input_tokens"] for r in requests),
        "avg_output_tokens": sum(r["max_tokens"] for r in requests) / len(requests),
        "trace_span_s": requests[-1]["timestamp_ms"] / 1000.0,
    }
    return requests, summary
