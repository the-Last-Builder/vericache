#!/usr/bin/env python3
"""2-Node Peer-Shared (Morphology I) Open-Loop vLLM Trace Replayer.

Replays official Mooncake FAST'25 KV-cache traces (`toolagent_trace.jsonl` and
`conversation_trace.jsonl`) against two live vLLM OpenAI-compatible servers
using deterministic 70/30 session-sticky routing (matching Suite 5) and raw
token_id prompts (zero BPE boundary drift).

Collects:
  - TTFT (P50 / P95 / P99 / mean, ms)
  - TPOT / ITL (P50 / P95 / P99 / mean, ms/tok)
  - E2E Latency (P50 / P95 / P99, ms)
  - Request throughput (req/s), prompt throughput (tok/s), output throughput (tok/s)
  - TTFT SLO attainment (<500ms, <1000ms, <2000ms)
  - Per-node vLLM `/metrics` prefix-cache hit counters (before/after delta)
  - Optional greedy (`temperature=0`) output consistency signatures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
import time
from typing import Any

import aiohttp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "baselines"))
from common import route_request_scaleup  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trace_tokens import build_safe_vocab, prepare_trace_requests  # noqa: E402


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+\-]+)$")


async def scrape_vllm_metrics(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    """Scrape key counter/gauge values from vLLM's `/metrics` Prometheus endpoint."""
    wanted_prefixes = (
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    )
    out: dict[str, float] = {}
    try:
        async with session.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return out
            text = await resp.text()
    except Exception:
        return out

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line.strip())
        if not m:
            continue
        name, _labels, val_s = m.groups()
        if any(name.startswith(p) for p in wanted_prefixes):
            try:
                out[name] = out.get(name, 0.0) + float(val_s)
            except ValueError:
                pass
    return out


def diff_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {k: max(0.0, after.get(k, 0.0) - before.get(k, 0.0)) for k in sorted(keys)}


async def send_streaming_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    req: dict[str, Any],
    node_id: int,
    *,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Send one streaming `/v1/completions` request with raw `token_ids`."""
    payload = {
        "model": model,
        "prompt": req["prompt_token_ids"],
        "max_tokens": int(req["max_tokens"]),
        "temperature": temperature,
        "seed": 42,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_start = time.perf_counter()
    t_first: float | None = None
    chunks: list[str] = []
    completion_tokens = 0
    prompt_tokens = int(req["input_tokens"])
    error: str | None = None

    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                error = f"HTTP {resp.status}: {body[:200]}"
            else:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_s = line[5:].strip()
                    if data_s == "[DONE]":
                        break
                    try:
                        evt = json.loads(data_s)
                    except json.JSONDecodeError:
                        continue
                    choices = evt.get("choices") or []
                    if choices:
                        txt = choices[0].get("text") or ""
                        if txt:
                            if t_first is None:
                                t_first = time.perf_counter()
                            chunks.append(txt)
                    usage = evt.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                        completion_tokens = int(usage.get("completion_tokens", completion_tokens))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    t_end = time.perf_counter()
    if completion_tokens == 0 and chunks:
        completion_tokens = len(chunks)
    if error is None and completion_tokens == 0 and int(req["max_tokens"]) > 0:
        error = "Empty completion (0 output tokens generated)"
    if t_first is None:
        t_first = t_end

    ttft_ms = (t_first - t_start) * 1000.0
    e2e_ms = (t_end - t_start) * 1000.0
    tpot_ms = (
        ((t_end - t_first) * 1000.0) / max(1, completion_tokens - 1)
        if completion_tokens > 1
        else 0.0
    )

    full_text = "".join(chunks)
    return {
        "req_idx": req["req_idx"],
        "node_id": node_id,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "text": full_text,
        "text_prefix": full_text[:128],
        "error": error,
    }


async def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    safe_vocab = build_safe_vocab(args.model_path)
    reqs, trace_summary = prepare_trace_requests(
        args.trace_path,
        safe_vocab,
        start_index=args.start_index,
        max_requests=args.max_requests,
        tokens_per_hash_id=args.tokens_per_hash_id,
        max_input_tokens=args.max_input_tokens,
        fixed_output_tokens=args.fixed_output_tokens,
        max_output_tokens=args.max_output_tokens,
        min_output_tokens=args.min_output_tokens,
    )

    urls = [args.node0_url.rstrip("/"), args.node1_url.rstrip("/")]
    sticky_pct = int(round(args.sticky_ratio * 100.0))

    connector = aiohttp.TCPConnector(limit=0, force_close=False, keepalive_timeout=2.0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Optional quick warmup (1 tiny request per node to trigger CUDA graphs / JIT)
        if args.warmup:
            warm_req = {
                "req_idx": -1,
                "prompt_token_ids": safe_vocab[:64],
                "input_tokens": 64,
                "max_tokens": 8,
            }
            for nid, url in enumerate(urls):
                w = await send_streaming_completion(
                    session, url, args.served_model_name, warm_req, nid, ignore_eos=True
                )
                if w["error"]:
                    raise RuntimeError(f"Warmup failed on node {nid} ({url}): {w['error']}")

        metrics_before = [await scrape_vllm_metrics(session, u) for u in urls]

        sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency > 0 else None
        total_n = len(reqs)
        results: list[dict[str, Any]] = [{}] * total_n

        # Real-time progress tracking state
        done_count = 0
        err_count = 0
        last_log_done = 0
        last_log_t = 0.0
        win_ttfts: list[float] = []
        progress_reqs = max(1, int(args.progress_interval_reqs))
        progress_s = max(1.0, float(args.progress_interval_s))
        mode = (
            "open_loop"
            if (args.qps is not None and args.qps > 0) or (args.time_scale is not None and args.time_scale > 0)
            else "closed_loop"
        )
        mode_tag = f"qps={args.qps:.2f}" if (args.qps is not None and args.qps > 0) else mode

        def _maybe_log_progress(force: bool = False) -> None:
            nonlocal last_log_done, last_log_t, win_ttfts
            now = time.perf_counter()
            wall = max(1e-6, now - t_wall0)
            dt_win = max(1e-6, now - last_log_t)
            dn_win = done_count - last_log_done
            if not force and dn_win < progress_reqs and dt_win < progress_s:
                return
            avg_rps = done_count / wall
            live_rps = dn_win / dt_win if dn_win > 0 else 0.0
            rem = max(0, total_n - done_count)
            eta_m = (rem / max(1e-6, avg_rps)) / 60.0 if done_count > 0 else 0.0
            if win_ttfts:
                arr = np.asarray(win_ttfts, dtype=np.float64)
                p50 = float(np.percentile(arr, 50))
                p99 = float(np.percentile(arr, 99))
            else:
                p50 = 0.0
                p99 = 0.0
            print(
                f"[progress] {args.system}/{args.case} ({mode_tag}) "
                f"{done_count}/{total_n} ({done_count * 100.0 / max(1, total_n):.1f}%) "
                f"err={err_count} wall={wall:.0f}s live_rps={live_rps:.2f} avg_rps={avg_rps:.2f} "
                f"win_ttft_p50={p50:.0f}ms p99={p99:.0f}ms eta={eta_m:.1f}m",
                file=sys.stderr,
                flush=True,
            )
            last_log_done = done_count
            last_log_t = now
            win_ttfts = []

        async def _execute_one(i: int, req: dict[str, Any], sched_off_s: float) -> None:
            nonlocal done_count, err_count
            node_id, _shard = route_request_scaleup(i, req["hash_ids"], 8, sticky_pct)
            if args.single_node:
                node_id = 0
            if sem is not None:
                async with sem:
                    disp_off_s = time.perf_counter() - t_wall0
                    res = await send_streaming_completion(
                        session,
                        urls[node_id],
                        args.served_model_name,
                        req,
                        node_id,
                        temperature=0.0,
                        ignore_eos=True,
                        timeout_s=args.timeout_s,
                    )
            else:
                disp_off_s = time.perf_counter() - t_wall0
                res = await send_streaming_completion(
                    session,
                    urls[node_id],
                    args.served_model_name,
                    req,
                    node_id,
                    temperature=0.0,
                    ignore_eos=True,
                    timeout_s=args.timeout_s,
                )
            comp_off_s = time.perf_counter() - t_wall0
            res["sched_offset_s"] = round(sched_off_s, 4)
            res["dispatch_offset_s"] = round(disp_off_s, 4)
            res["complete_offset_s"] = round(comp_off_s, 4)
            results[i] = res
            done_count += 1
            if res.get("error"):
                err_count += 1
            else:
                win_ttfts.append(float(res["ttft_ms"]))
            _maybe_log_progress(force=(done_count == total_n))

        # Compute arrival offsets: either fixed target QPS or timestamp / time_scale or closed-loop
        offsets_s: list[float] = []
        if args.qps is not None and args.qps > 0:
            # Preserve trace relative inter-arrival burstiness, scaled so mean arrival rate == args.qps
            span_s = max(1e-6, trace_summary["trace_span_s"])
            desired_span_s = max(0.0, (len(reqs) - 1) / args.qps)
            if span_s > 0.01:
                scale = desired_span_s / span_s
                offsets_s = [(r["timestamp_ms"] / 1000.0) * scale for r in reqs]
            else:
                offsets_s = [i / args.qps for i in range(len(reqs))]
        elif args.time_scale is not None and args.time_scale > 0:
            offsets_s = [(r["timestamp_ms"] / 1000.0) / args.time_scale for r in reqs]
        else:
            # Closed-loop max-concurrency saturation
            offsets_s = [0.0 for _ in reqs]

        t_wall0 = time.perf_counter()
        last_log_t = t_wall0
        print(
            f"[progress] START {args.system}/{args.case} ({mode_tag}) "
            f"total={total_n} max_conc={args.max_concurrency} timeout={args.timeout_s:.0f}s "
            f"avg_in={trace_summary['avg_input_tokens']:.0f} avg_out={trace_summary['avg_output_tokens']:.1f}",
            file=sys.stderr,
            flush=True,
        )

        async def _heartbeat() -> None:
            while done_count < total_n:
                await asyncio.sleep(progress_s)
                if done_count < total_n:
                    _maybe_log_progress(force=False)

        hb_task = asyncio.create_task(_heartbeat())
        try:
            if mode == "closed_loop":
                # Deterministic FIFO bounded worker pool (avoids spawning 23k coroutines at t=0)
                n_workers = max(1, args.max_concurrency) if args.max_concurrency > 0 else total_n
                next_idx = 0

                async def _worker() -> None:
                    nonlocal next_idx
                    while True:
                        idx = next_idx
                        if idx >= total_n:
                            return
                        next_idx += 1
                        await _execute_one(idx, reqs[idx], 0.0)

                await asyncio.gather(*[asyncio.create_task(_worker()) for _ in range(min(n_workers, total_n))])
            else:
                # Incremental open-loop feeder preserving exact trace inter-arrival offsets
                in_flight: set[asyncio.Task[None]] = set()
                for i, req in enumerate(reqs):
                    target = offsets_s[i]
                    now_off = time.perf_counter() - t_wall0
                    if target > now_off:
                        await asyncio.sleep(target - now_off)
                    t = asyncio.create_task(_execute_one(i, req, target))
                    in_flight.add(t)
                    t.add_done_callback(in_flight.discard)
                if in_flight:
                    await asyncio.gather(*list(in_flight))
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
        wall_s = time.perf_counter() - t_wall0

        metrics_after = [await scrape_vllm_metrics(session, u) for u in urls]

    node_metric_deltas = [
        diff_metrics(metrics_before[i], metrics_after[i]) for i in range(len(urls))
    ]

    if args.save_outputs:
        out_p = pathlib.Path(args.save_outputs)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    ok_res = [r for r in results if not r.get("error")]
    err_res = [r for r in results if r.get("error")]
    if not ok_res:
        raise RuntimeError(f"All {len(results)} requests failed! First error: {err_res[0]['error']}")

    total_reqs = len(results)
    ttfts = np.asarray([r["ttft_ms"] for r in ok_res], dtype=np.float64)
    tpots = np.asarray([r["tpot_ms"] for r in ok_res if r["output_tokens"] > 1], dtype=np.float64)
    e2es = np.asarray([r["e2e_ms"] for r in ok_res], dtype=np.float64)
    e2es_all = np.asarray([r["e2e_ms"] for r in results], dtype=np.float64)

    total_in = sum(int(r["input_tokens"]) for r in ok_res)
    total_out = sum(int(r["output_tokens"]) for r in ok_res)

    # Aggregate prefix cache hit counters from both nodes
    gpu_hits = sum(d.get("vllm:prefix_cache_hits_total", 0.0) for d in node_metric_deltas)
    gpu_queries = sum(d.get("vllm:prefix_cache_queries_total", 0.0) for d in node_metric_deltas)
    ext_hits = sum(d.get("vllm:external_prefix_cache_hits_total", 0.0) for d in node_metric_deltas)
    ext_queries = sum(
        d.get("vllm:external_prefix_cache_queries_total", 0.0) for d in node_metric_deltas
    )
    preemptions = sum(d.get("vllm:num_preemptions_total", 0.0) for d in node_metric_deltas)

    summary = {
        "system": args.system,
        "case": args.case,
        "mode": mode,
        "model": args.served_model_name,
        "trace": trace_summary,
        "qps_target": args.qps,
        "time_scale": args.time_scale,
        "sticky_ratio": args.sticky_ratio,
        "max_concurrency": args.max_concurrency,
        "wall_s": wall_s,
        "total_requests": total_reqs,
        "completed": len(ok_res),
        "errors": len(err_res),
        "first_error": err_res[0]["error"] if err_res else None,
        "req_per_s": len(ok_res) / max(1e-9, wall_s),
        "input_tok_per_s": total_in / max(1e-9, wall_s),
        "output_tok_per_s": total_out / max(1e-9, wall_s),
        "total_tok_per_s": (total_in + total_out) / max(1e-9, wall_s),
        "ttft_mean_ms": float(np.mean(ttfts)),
        "ttft_p50_ms": float(np.percentile(ttfts, 50)),
        "ttft_p95_ms": float(np.percentile(ttfts, 95)),
        "ttft_p99_ms": float(np.percentile(ttfts, 99)),
        "tpot_mean_ms": float(np.mean(tpots)) if len(tpots) else 0.0,
        "tpot_p50_ms": float(np.percentile(tpots, 50)) if len(tpots) else 0.0,
        "tpot_p95_ms": float(np.percentile(tpots, 95)) if len(tpots) else 0.0,
        "tpot_p99_ms": float(np.percentile(tpots, 99)) if len(tpots) else 0.0,
        "e2e_mean_ms": float(np.mean(e2es)),
        "e2e_p50_ms": float(np.percentile(e2es, 50)),
        "e2e_p95_ms": float(np.percentile(e2es, 95)),
        "e2e_p99_ms": float(np.percentile(e2es, 99)),
        "e2e_p99_all_ms": float(np.percentile(e2es_all, 99)),
        # All-attempt SLO attainment (failed/timed-out requests count as SLO misses)
        "slo_ttft_500ms": float(np.sum(ttfts <= 500.0)) / max(1, total_reqs),
        "slo_ttft_1000ms": float(np.sum(ttfts <= 1000.0)) / max(1, total_reqs),
        "slo_ttft_2000ms": float(np.sum(ttfts <= 2000.0)) / max(1, total_reqs),
        "vllm_gpu_hit_rate": gpu_hits / max(1.0, gpu_queries) if gpu_queries > 0 else 0.0,
        "vllm_ext_hit_rate": ext_hits / max(1.0, ext_queries) if ext_queries > 0 else 0.0,
        "vllm_ext_hit_total_rate": ext_hits / max(1.0, gpu_queries) if gpu_queries > 0 else 0.0,
        "vllm_total_hit_rate": (gpu_hits + ext_hits) / max(1.0, gpu_queries) if gpu_queries > 0 else 0.0,
        "vllm_gpu_hits_tokens": gpu_hits,
        "vllm_gpu_queries_tokens": gpu_queries,
        "vllm_ext_hits_tokens": ext_hits,
        "vllm_ext_queries_tokens": ext_queries,
        "vllm_preemptions": preemptions,
        "node0_reqs": sum(1 for r in ok_res if r["node_id"] == 0),
        "node1_reqs": sum(1 for r in ok_res if r["node_id"] == 1),
        "node_metric_deltas": node_metric_deltas,
    }

    if args.output_json:
        jp = pathlib.Path(args.output_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="2-Node Peer-Shared vLLM Trace Replayer")
    ap.add_argument("--system", required=True, help="System label (baseline, vericache, lmcache, mooncake)")
    ap.add_argument("--case", default="toolagent", help="Case label")
    ap.add_argument("--model-path", required=True, help="Local HF model directory (for tokenizer)")
    ap.add_argument("--served-model-name", default="qwen2.5-7b", help="Model name passed to /v1/completions")
    ap.add_argument("--trace-path", required=True, help="FAST'25 trace jsonl path")
    ap.add_argument("--node0-url", default="http://127.0.0.1:8000")
    ap.add_argument("--node1-url", default="http://127.0.0.1:8001")
    ap.add_argument("--single-node", action="store_true", help="Route all requests to node0 only")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-requests", type=int, default=200)
    ap.add_argument("--tokens-per-hash-id", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--fixed-output-tokens", type=int, default=None)
    ap.add_argument("--max-output-tokens", type=int, default=64)
    ap.add_argument("--min-output-tokens", type=int, default=16)
    ap.add_argument("--qps", type=float, default=None, help="Target mean arrival rate (req/s)")
    ap.add_argument("--time-scale", type=float, default=None, help="Timestamp speedup factor")
    ap.add_argument("--sticky-ratio", type=float, default=0.70)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--progress-interval-reqs", type=int, default=200)
    ap.add_argument("--progress-interval-s", type=float, default=30.0)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--save-outputs", default=None)
    args = ap.parse_args()

    summary = asyncio.run(run_replay(args))
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
