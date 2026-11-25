#!/usr/bin/env python3
"""Replay frozen Step 1 arrivals across two vLLM nodes and measure joint SLO goodput."""

from __future__ import annotations

import argparse
import asyncio
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import threading
import time
from typing import Any

import aiohttp
import psutil

from step1_metrics import (
    GIB,
    ResourceSampler,
    numeric_delta,
    parse_prometheus,
    read_hca,
    read_nvme_usage,
    read_vericache,
    sum_metrics,
)
from trace_tokens import build_safe_vocab, prepare_trace_requests

WARMUP_REQUESTS = 500
MEASUREMENT_REQUESTS = 1500
WINDOW_S = 10
TTFT_SLO_MS = 2000
TPOT_SLO_MS = 50


def load_frozen_window(path: str | Path, warmup_count: int = WARMUP_REQUESTS,
                       measurement_count: int = MEASUREMENT_REQUESTS) -> list[dict]:
    with open(path, encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) != warmup_count + measurement_count:
        raise ValueError(
            f"Expected {warmup_count} warmup + {measurement_count} measured requests, got {len(rows)}"
        )
    required = {
        "req_idx", "timestamp_ms", "prompt_token_ids", "input_tokens", "output_tokens",
        "hash_ids", "session_key", "dest_node", "is_warmup",
    }
    for i, row in enumerate(rows):
        if not required <= row.keys():
            raise ValueError(f"Frozen request {i} is missing {sorted(required - row.keys())}")
        tokens = row["prompt_token_ids"]
        if (row["is_warmup"] is not (i < warmup_count)
                or type(row["dest_node"]) is not int or row["dest_node"] not in (0, 1)
                or not tokens or row["input_tokens"] != len(tokens)
                or type(row["output_tokens"]) is not int or row["output_tokens"] <= 0
                or type(row["timestamp_ms"]) is not int or row["timestamp_ms"] < 0
                or type(row["req_idx"]) is not int):
            raise ValueError(f"Invalid frozen request {i}")
        if i and (row["req_idx"] != rows[i - 1]["req_idx"] + 1
                  or row["timestamp_ms"] < rows[i - 1]["timestamp_ms"]):
            raise ValueError(f"Frozen requests must be contiguous and timestamp-ordered at row {i}")
    return rows


def arrival_offsets(rows: list[dict], qps: float | None, time_scale: float | None) -> list[float]:
    base = rows[0]["timestamp_ms"]
    relative = [(row["timestamp_ms"] - base) / 1000 for row in rows]
    if qps is not None:
        if relative[-1] == 0:
            return [idx / qps for idx in range(len(rows))]
        scale = (len(rows) - 1) / qps / relative[-1]
        return [offset * scale for offset in relative]
    if time_scale is not None:
        return [offset / time_scale for offset in relative]
    return [0.0] * len(rows)


def planned_duration(offsets: list[float], qps: float | None,
                     time_scale: float | None) -> float | None:
    if qps is not None:
        return len(offsets) / qps
    if time_scale is not None and len(offsets) > 1 and offsets[-1] > 0:
        return offsets[-1] * len(offsets) / (len(offsets) - 1)
    return None


async def scrape_vllm_metrics(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    async with session.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as response:
        response.raise_for_status()
        samples = parse_prometheus(await response.text())
    if not samples:
        raise RuntimeError(f"No Prometheus samples from {base_url}/metrics")
    return samples


async def wait_idle(session: aiohttp.ClientSession, urls: list[str], timeout_s: float = 60) -> None:
    deadline = time.perf_counter() + timeout_s
    while True:
        samples = await asyncio.gather(*(scrape_vllm_metrics(session, url) for url in urls))
        if all(not any(values.get(f"vllm:num_requests_{state}", 0) > 0
                       for state in ("running", "waiting", "swapped"))
               for values in map(sum_metrics, samples)):
            return
        if time.perf_counter() >= deadline:
            raise TimeoutError("vLLM requests did not drain before the snapshot")
        await asyncio.sleep(0.1)


async def snapshot_node(session: aiohttp.ClientSession, urls: list[str], args: argparse.Namespace,
                        node: int, *, include_usage: bool = True) -> dict:
    started = time.perf_counter()
    snapshot: dict[str, Any] = {
        "started_t": started,
        "vllm": await scrape_vllm_metrics(session, urls[node]),
    }
    if args.stats_socket:
        snapshot["vericache"] = await asyncio.to_thread(
            read_vericache, args.stats_socket[node], timeout=3,
        )
    if args.nvme_dir:
        snapshot["nvme"] = await asyncio.to_thread(
            read_nvme_usage, args.nvme_dir[node], include_usage=include_usage,
        )
    snapshot["finished_t"] = time.perf_counter()
    return snapshot


async def snapshot_cluster(session: aiohttp.ClientSession, urls: list[str],
                           args: argparse.Namespace) -> dict:
    nodes = await asyncio.gather(
        *(snapshot_node(session, urls, args, node) for node in range(len(urls))),
    )
    aux = await asyncio.gather(*(scrape_vllm_metrics(session, url) for url in args.aux_metrics_url))
    return {
        "timestamp_unix_ns": time.time_ns(),
        "nodes": nodes,
        "hca": read_hca(args.hca_device),
        "aux_metrics": dict(zip(args.aux_metrics_url, aux)),
    }


async def drained_snapshot(session: aiohttp.ClientSession, urls: list[str],
                           args: argparse.Namespace, on_drained=None) -> dict:
    async def control(url: str, endpoint: str, status: str) -> None:
        async with session.post(url + endpoint, timeout=aiohttp.ClientTimeout(total=60)) as response:
            response.raise_for_status()
            if (await response.json()).get("status") != status:
                raise RuntimeError(f"{url}{endpoint} did not report {status}")

    async def all_servers(endpoint: str, status: str) -> None:
        results = await asyncio.gather(
            *(control(url, endpoint, status) for url in urls), return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                raise result

    # vLLM's wait barrier drains scheduler-owned connector work. Preserve the
    # warmed prefix cache. The launcher exposes this API only on localhost.
    try:
        if args.vllm_drain:
            await all_servers("/pause?mode=wait&clear_cache=false", "paused")
        await wait_idle(session, urls)
        if args.stats_socket:
            await asyncio.gather(*(asyncio.to_thread(
                read_vericache, path, drain=True, timeout=60,
            ) for path in args.stats_socket))
        drained_t = time.perf_counter()
        if on_drained:
            on_drained(drained_t)
        return await snapshot_cluster(session, urls, args)
    finally:
        if args.vllm_drain:
            await all_servers("/resume", "resumed")


def _top1_logprobs(choice: dict, token_ids: list[int]) -> list[tuple[int, float]]:
    logprobs = choice["logprobs"]
    candidates = logprobs["top_logprobs"]
    if len(candidates) != len(token_ids):
        raise ValueError("Top-1 logprob count differs from token count")
    top1 = []
    for entries in candidates:
        token, value = max(entries.items(), key=lambda item: item[1])
        token_id = int(token.removeprefix("token_id:")) if token.startswith("token_id:") else None
        if token_id is None or not math.isfinite(value):
            raise ValueError("vLLM did not return a token ID and finite top-1 logprob")
        top1.append((token_id, float(value)))
    return top1


async def send_streaming_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    req: dict,
    node_id: int,
    *,
    sched_t: float | None = None,
    record_logprobs: bool = False,
    timeout_s: float = 300.0,
) -> dict:
    payload = {
        "model": model,
        "prompt": req["prompt_token_ids"],
        "max_tokens": req["max_tokens"],
        "temperature": 0.0,
        "seed": 42,
        "ignore_eos": True,
        "echo": False,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    if record_logprobs:
        payload.update(logprobs=1, return_tokens_as_token_ids=True)

    first_token_t = last_token_t = None
    completion_tokens = 0
    usage = None
    saw_done = False
    finish_reason = None
    exact_ids: list[int] = []
    top1: list[tuple[int, float]] = []
    error = None
    send_t = time.perf_counter()
    if sched_t is None:
        sched_t = send_t
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
            async for raw_line in response.content:
                received_t = time.perf_counter()
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    saw_done = True
                    break
                event = json.loads(data)
                if event.get("error") or event.get("object") == "error":
                    raise RuntimeError(str(event.get("error") or event)[:300])
                for choice in event.get("choices") or ():
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    token_ids = choice.get("token_ids") or []
                    if not token_ids:
                        continue
                    if first_token_t is None:
                        first_token_t = received_t
                    last_token_t = received_t
                    completion_tokens += len(token_ids)
                    if record_logprobs:
                        exact_ids.extend(token_ids)
                        top1.extend(_top1_logprobs(choice, token_ids))
                if event.get("usage") is not None:
                    usage = event["usage"]
        if not saw_done or finish_reason not in ("stop", "length"):
            raise ValueError(f"Completion stream did not finish normally ({finish_reason!r})")
        if not completion_tokens or usage is None or usage.get("completion_tokens") != completion_tokens:
            raise ValueError("Incomplete token IDs or inconsistent final token usage")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    finished_t = time.perf_counter()
    result = {
        "req_idx": req["req_idx"],
        "node_id": node_id,
        "input_tokens": req["input_tokens"],
        "planned_output_tokens": req["max_tokens"],
        "output_tokens": completion_tokens,
        "sched_t": sched_t,
        "send_t": send_t,
        "first_token_t": first_token_t,
        "last_token_t": last_token_t,
        "finished_t": finished_t,
        "ttft_sched_ms": None if first_token_t is None else (first_token_t - sched_t) * 1000,
        "ttft_send_ms": None if first_token_t is None else (first_token_t - send_t) * 1000,
        "tpot_ms": ((last_token_t - first_token_t) * 1000 / (completion_tokens - 1)
                    if completion_tokens > 1 else None),
        "e2e_ms": (finished_t - sched_t) * 1000,
        "finish_reason": finish_reason,
        "error": error,
    }
    if record_logprobs:
        result["token_ids"] = exact_ids
        result["top1_token_ids"] = [token for token, _ in top1]
        result["top1_logprobs"] = [value for _, value in top1]
    return result


def exactness(expected: dict, actual: dict) -> dict:
    """Compare IDs and IEEE-754 representations of the logprobs returned by vLLM."""
    count = expected["output_tokens"]
    if (expected.get("error") or actual.get("error") or count == 0
            or any(len(expected.get(field, ())) != count or len(actual.get(field, ())) != count
                   for field in ("token_ids", "top1_token_ids", "top1_logprobs"))):
        return {"exact": False, "missing_or_incomplete": True}
    token_mismatches = sum(a != b for a, b in zip(expected["token_ids"], actual["token_ids"]))
    top1_id_mismatches = sum(a != b for a, b in zip(expected["top1_token_ids"], actual["top1_token_ids"]))
    logprob_mismatches = sum(
        struct.pack("!d", a) != struct.pack("!d", b)
        for a, b in zip(expected["top1_logprobs"], actual["top1_logprobs"])
    )
    return {
        "exact": token_mismatches == top1_id_mismatches == logprob_mismatches == 0,
        "compared_tokens": count,
        "token_mismatches": token_mismatches,
        "top1_id_mismatches": top1_id_mismatches,
        "logprob_mismatches": logprob_mismatches,
    }


def joint_slo(result: dict) -> bool:
    ttft, tpot = result["ttft_sched_ms"], result["tpot_ms"]
    return (
        not result["error"] and ttft is not None and ttft <= TTFT_SLO_MS
        and (result["output_tokens"] == 1 or tpot is not None and tpot <= TPOT_SLO_MS)
        and result["output_tokens"] > 0
    )


def percentiles(values) -> dict:
    values = sorted(x for x in values if x is not None)
    result = {}
    for p in (50, 95, 99):
        if not values:
            result[f"p{p}"] = None
            continue
        index = (len(values) - 1) * p / 100
        low, high = math.floor(index), math.ceil(index)
        result[f"p{p}"] = values[low] + (index - low) * (values[high] - values[low])
    return result


def goodput_series(results: list[dict], start_t: float, wall_s: float) -> list[dict]:
    counts = [0] * max(1, math.ceil(wall_s / WINDOW_S))
    for result in results:
        if joint_slo(result):
            index = min(len(counts) - 1, int((result["last_token_t"] - start_t) // WINDOW_S))
            counts[index] += 1
    return [
        {"start_s": i * WINDOW_S, "duration_s": duration, "slo_met": count,
         "goodput_req_s": count / duration if duration > 0 else 0.0}
        for i, count in enumerate(counts)
        for duration in [min(WINDOW_S, wall_s - i * WINDOW_S)]
    ]


def recovery_after(resume_offset: float, current: list[dict], baseline: list[dict],
                   next_pause: float = math.inf) -> dict:
    consecutive = 0
    for window, ref in zip(current, baseline):
        if window["start_s"] + WINDOW_S > next_pause:
            break
        if (window["start_s"] >= resume_offset
                and window["start_s"] == ref["start_s"]
                and window["duration_s"] == ref["duration_s"] == WINDOW_S
                and ref["goodput_req_s"] > 0
                and window["goodput_req_s"] >= 0.9 * ref["goodput_req_s"]):
            consecutive += 1
            if consecutive == 3:
                return {"recovered": True,
                        "first_qualifying_window_s": window["start_s"] - 2 * WINDOW_S,
                        "time_to_confirm_recovery_s": window["start_s"] + WINDOW_S - resume_offset}
        else:
            consecutive = 0
    return {"recovered": False, "time_to_confirm_recovery_s": None}


class ScheduledFaults:
    """Send POSIX signals on monotonic deadlines independently of request processing."""

    def __init__(self, pid: int, offsets: list[float], duration: float, capture):
        self.pid = pid
        self.offsets = offsets
        self.duration = duration
        self.capture = capture
        self.events: list[dict] = []
        self.error: Exception | None = None
        self.stop_event = threading.Event()
        self.process = psutil.Process(pid)
        self.pidfd = None
        # Some Python builds omit pidfd bindings. psutil verifies process identity
        # before sending each signal in that case.
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                self.pidfd = os.pidfd_open(pid)
            except OSError as error:
                if error.errno != errno.ENOSYS:
                    raise
        self.thread: threading.Thread | None = None

    def _signal(self, sig: int) -> None:
        if self.pidfd is None:
            self.process.send_signal(sig)
        else:
            signal.pidfd_send_signal(self.pidfd, sig)

    def start(self, start_t: float) -> None:
        self.thread = threading.Thread(target=self._run, args=(start_t,), name="step1-fault-clock")
        self.thread.start()

    def _run(self, start_t: float) -> None:
        try:
            for offset in self.offsets:
                if self.stop_event.wait(max(0, start_t + offset - time.perf_counter())):
                    break
                event = {"pid": self.pid, "scheduled_pause_offset_s": offset,
                         "scheduled_resume_offset_s": offset + self.duration}
                self.events.append(event)
                self._signal(signal.SIGSTOP)
                try:
                    event.update(pause_t=time.perf_counter(), pause_unix_ns=time.time_ns())
                    event["donor_before"] = self.capture()
                    remaining = start_t + offset + self.duration - time.perf_counter()
                    if remaining > 0:
                        # Finish an already-started pause even if all HTTP requests have drained.
                        time.sleep(remaining)
                finally:
                    self._signal(signal.SIGCONT)
                    event.update(resume_t=time.perf_counter(), resume_unix_ns=time.time_ns())
                    event["donor_after"] = self.capture()
        except Exception as exc:
            self.error = exc

    async def finish(self) -> list[dict]:
        self.stop_event.set()
        try:
            if self.thread is not None:
                await asyncio.to_thread(self.thread.join)
        finally:
            if self.pidfd is not None:
                os.close(self.pidfd)
        if self.error:
            raise RuntimeError("External fault injection failed") from self.error
        for event in self.events:
            for boundary in ("donor_before", "donor_after"):
                try:
                    event[boundary] = await asyncio.wrap_future(event[boundary])
                except Exception as exc:
                    event[boundary] = {"error": str(exc)}
        return self.events


async def run_cohort(session: aiohttp.ClientSession, urls: list[str], args: argparse.Namespace,
                     rows: list[dict], offsets: list[float], phase: str,
                     start_t: float) -> list[dict]:
    semaphore = asyncio.Semaphore(args.max_concurrency)
    completed = 0

    async def execute(index: int, req: dict) -> dict:
        is_open = args.qps is not None or args.time_scale is not None
        if is_open:
            scheduled_t = start_t + offsets[index]
            await asyncio.sleep(max(0, scheduled_t - time.perf_counter()))
        else:
            scheduled_t = None
        async with semaphore:
            node = req["dest_node"]
            result = await send_streaming_completion(
                session, urls[node], args.served_model_name, req, node,
                sched_t=scheduled_t,
                record_logprobs=args.record_logprobs and phase == "measure",
                timeout_s=args.timeout_s,
            )
        nonlocal completed
        completed += 1
        if completed % args.progress_interval_reqs == 0 or completed == len(rows):
            print(f"[{args.system}/{phase}] {completed}/{len(rows)} finished", file=sys.stderr, flush=True)
        result["joint_slo_met"] = joint_slo(result)
        return result

    if args.qps is not None or args.time_scale is not None:
        return await asyncio.gather(*(execute(i, row) for i, row in enumerate(rows)))

    results: list[dict] = [{} for _ in rows]
    work = iter(enumerate(rows))

    async def worker() -> None:
        for i, row in work:
            results[i] = await execute(i, row)

    await asyncio.gather(*(worker() for _ in range(min(len(rows), args.max_concurrency))))
    return results


def snapshot_delta(before: dict, after: dict) -> dict:
    deltas = {
        "nodes": [
            {name: numeric_delta(old.get(name, {}), new[name])
             for name in ("vllm", "vericache", "nvme") if name in new}
            for old, new in zip(before["nodes"], after["nodes"])
        ],
        "hca": {dev: numeric_delta(before["hca"][dev], values)
                for dev, values in after["hca"].items()},
        "aux_metrics": {url: numeric_delta(before["aux_metrics"][url], values)
                        for url, values in after["aux_metrics"].items()},
    }
    return deltas


def path_rates(before: dict, after: dict, elapsed: float) -> dict:
    stats_before = before.get("vericache")
    stats_after = after.get("vericache")
    if stats_before is None or stats_after is None or elapsed <= 0:
        return {}
    deltas = numeric_delta(stats_before, stats_after)
    frame_bytes = stats_after["frame_bytes"]
    offered = deltas.get("store_offered_blocks", deltas["put_attempts"])
    hit_frames = sum(deltas[k] for k in ("dram_hits", "nvme_hits", "remote_hits", "gpu_hits"))
    lookup_frames = hit_frames + deltas["misses"]
    return {
        "offered_bytes": offered * frame_bytes,
        "admitted_bytes": deltas["puts"] * frame_bytes,
        "admission_fraction": deltas["puts"] / offered if offered > 0 else None,
        "admitted_gib_s": deltas["puts"] * frame_bytes / GIB / elapsed,
        "dram_evicted_gib_s": deltas["dram_evictions"] * frame_bytes / GIB / elapsed,
        "dram_reclaimed_gib_s": deltas["dram_reclaims"] * frame_bytes / GIB / elapsed,
        "nvme_spill_bytes": deltas["nvme_spills"] * frame_bytes,
        "local_nvme_restore_bytes": deltas["nvme_hits"] * frame_bytes,
        "remote_nvme_restore_bytes": deltas["remote_nvme_warm_hits"] * frame_bytes,
        "effective_external_restore_fraction": hit_frames / lookup_frames if lookup_frames else None,
    }


def summarize_faults(events: list[dict], results: list[dict], args: argparse.Namespace,
                     start_t: float, series: list, baseline: dict | None) -> list[dict]:
    for index, event in enumerate(events):
        begin, end = event["pause_t"], event["resume_t"]
        next_pause = events[index + 1]["pause_t"] - start_t if index + 1 < len(events) else math.inf
        event["pause_offset_s"], event["resume_offset_s"] = begin - start_t, end - start_t
        event["actual_duration_s"] = end - begin
        affected = [
            r for r in results
            if r["node_id"] == args.fault_requester_node
            and r["sched_t"] < end and r["finished_t"] > begin
        ]
        event["affected_requests"] = len(affected)
        event["affected_req_indices"] = [r["req_idx"] for r in affected]
        event["affected_ttft_sched_ms"] = percentiles(r["ttft_sched_ms"] for r in affected)
        event["affected_ttft_missing"] = sum(r["ttft_sched_ms"] is None for r in affected)
        event["affected_errors"] = sum(r["error"] is not None for r in affected)
        event["affected_timeouts"] = sum("Timeout" in (r["error"] or "") for r in affected)
        event["affected_completed"] = len(affected) - event["affected_errors"]
        donor = [r for r in results if r["node_id"] != args.fault_requester_node
                 and r["error"] is None and begin <= r["last_token_t"] < end]
        event["donor_completed"] = len(donor)
        event["donor_goodput_req_s"] = sum(map(joint_slo, donor)) / (end - begin)
        event["donor_path_rates"] = path_rates(event["donor_before"], event["donor_after"], end - begin)
        event["donor_vllm_delta"] = numeric_delta(
            event["donor_before"].get("vllm", {}), event["donor_after"].get("vllm", {}),
        )
        event["donor_nvme_delta"] = numeric_delta(
            event["donor_before"].get("nvme", {}), event["donor_after"].get("nvme", {}),
        )
        event["recovery"] = recovery_after(
            event["resume_offset_s"], series, baseline["goodput_10s"], next_pause,
        ) if baseline is not None else None
    return events


def replay_plan_id(rows: list[dict], offsets: list[float], args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    digest.update(repr((args.served_model_name, args.max_concurrency, args.record_logprobs,
                        args.deployment_id)).encode())
    for req, offset in zip(rows, offsets):
        digest.update(json.dumps(
            [req["req_idx"], req["dest_node"], offset, req["max_tokens"], req["prompt_token_ids"]],
            separators=(",", ":"),
        ).encode())
    return digest.hexdigest()


def compare_oracle(path: str, results: list[dict], selected_ids: set[int] | None = None) -> dict:
    with open(path, encoding="utf-8") as source:
        oracle = {row["req_idx"]: row for line in source if (row := json.loads(line))}
    matched = []
    for result in results:
        idx = result["req_idx"]
        if selected_ids is None or idx in selected_ids:
            compare = exactness(oracle[idx], result) if idx in oracle else {"exact": False, "missing_oracle": True}
            matched.append({"req_idx": idx, **compare})
    return {
        "checked": len(matched),
        "exact": sum(row["exact"] for row in matched),
        "mismatches": [row for row in matched if not row["exact"]],
    }


def _prepare_requests(args: argparse.Namespace) -> list[dict]:
    if args.frozen_window:
        rows = load_frozen_window(args.frozen_window, args.warmup_requests, args.measure_requests)
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required for raw trace replay")
        rows, _ = prepare_trace_requests(
            args.trace_path, build_safe_vocab(args.model_path),
            max_requests=args.warmup_requests + args.measure_requests, start_index=args.start_index,
            max_input_tokens=min(args.max_input_tokens, args.max_model_len - 1),
            sticky_ratio=args.sticky_ratio, tokens_per_hash_id=args.tokens_per_hash_id,
        )
        if len(rows) != args.warmup_requests + args.measure_requests:
            raise ValueError("Raw trace is shorter than the requested warmup + measure cohort")
    for row in rows:
        room = args.max_model_len - row["input_tokens"]
        if room < 1:
            raise ValueError(f"Frozen prompt {row['req_idx']} leaves no space under --max-model-len")
        row["max_tokens"] = min(row["output_tokens"], room)
        if args.clamp_output_max is not None:
            row["max_tokens"] = min(row["max_tokens"], args.clamp_output_max)
    return rows


async def run_replay(args: argparse.Namespace) -> dict:
    rows = _prepare_requests(args)
    warm, measured = rows[:args.warmup_requests], rows[args.warmup_requests:]
    urls = [args.node0_url.rstrip("/"), args.node1_url.rstrip("/")]
    measure_offsets = arrival_offsets(measured, args.qps, args.time_scale)
    planned_s = planned_duration(measure_offsets, args.qps, args.time_scale)
    plan_id = replay_plan_id(rows, arrival_offsets(warm, args.qps, args.time_scale) + measure_offsets, args)
    baseline = None
    if args.paired_baseline:
        baseline = json.loads(Path(args.paired_baseline).read_text())
        if (baseline["plan_id"] != plan_id or baseline["system"] != args.system
                or baseline.get("faults") or baseline.get("after") is None):
            raise ValueError("The paired baseline needs complete snapshots, the same plan/system and no faults")

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        faults = None
        if args.fault_pid_file:
            fault_pid = int(Path(args.fault_pid_file).read_text().strip())
            if fault_pid <= 1:
                raise ValueError("--fault-pid-file must identify a vLLM worker PID above 1")
            loop = asyncio.get_running_loop()
            donor_node = 1 - args.fault_requester_node
            faults = ScheduledFaults(fault_pid, args.fault_offsets_s, args.fault_duration_s, lambda:
                asyncio.run_coroutine_threadsafe(
                    snapshot_node(session, urls, args, donor_node, include_usage=False), loop,
                )
            )

        sampler = None
        resource_usage = {}
        drained_t = None
        after_error = None

        def stop_sampling(when: float) -> None:
            nonlocal sampler, resource_usage, drained_t
            drained_t = when
            if sampler is not None:
                active, sampler = sampler, None
                resource_usage = active.stop()

        try:
            warm_results = await run_cohort(
                session, urls, args, warm, arrival_offsets(warm, args.qps, args.time_scale),
                "warmup", time.perf_counter(),
            )
            if all(r["error"] is not None for r in warm_results):
                raise RuntimeError(f"All warmup requests failed: {warm_results[0]['error']}")
            before = await drained_snapshot(session, urls, args)
            if args.server_pid:
                sampler = ResourceSampler(args.server_pid, args.gpu_id)
            if sampler:
                sampler.start()
            start_t = time.perf_counter()
            measurement_start_unix_ns = time.time_ns()
            if faults:
                faults.start(start_t)
            results = await run_cohort(session, urls, args, measured, measure_offsets, "measure", start_t)
            completed_t = time.perf_counter()
            finished_faults, faults = faults, None
            fault_events = await finished_faults.finish() if finished_faults else []
            try:
                after = await drained_snapshot(session, urls, args, on_drained=stop_sampling)
            except Exception as error:
                after = None
                after_error = f"{type(error).__name__}: {error}"
                if drained_t is None:
                    stop_sampling(time.perf_counter())
        finally:
            try:
                if faults is not None:
                    await faults.finish()
            finally:
                if sampler is not None:
                    sampler.stop()
    wall_s = drained_t - start_t
    series = goodput_series(results, start_t, wall_s)
    deltas = snapshot_delta(before, after) if after is not None else None
    completed = [r for r in results if r["error"] is None]
    slo_met = sum(map(joint_slo, results))
    metrics = [sum_metrics(node["vllm"]) for node in deltas["nodes"]] if deltas else []
    summary = {
        "system": args.system, "case": args.case, "model": args.served_model_name,
        "mode": "open_loop" if args.qps is not None or args.time_scale is not None else "closed_loop",
        "qps_target": args.qps, "time_scale": args.time_scale, "plan_id": plan_id,
        "max_concurrency": args.max_concurrency,
        "output_clamp_max": args.clamp_output_max,
        "warmup_requests": len(warm_results),
        "warmup_errors": sum(r["error"] is not None for r in warm_results),
        "total_requests": len(results),
        "completed": len(completed), "errors": len(results) - len(completed),
        "measurement_start_unix_ns": measurement_start_unix_ns,
        "wall_s": wall_s, "request_wall_s": completed_t - start_t, "planned_window_s": planned_s,
        "req_per_s": len(completed) / wall_s,
        "joint_slo_met": slo_met,
        "joint_slo_attainment": slo_met / len(results),
        "planned_window_goodput_req_s": slo_met / planned_s if planned_s is not None else None,
        "wall_clock_goodput_req_s": slo_met / wall_s,
        "goodput_10s": series,
        "preemptions": sum(m.get("vllm:num_preemptions_total", 0) for m in metrics) if metrics else None,
        "before": before, "after": after, "delta": deltas, "after_snapshot_error": after_error,
        "vericache_path_rates": [
            path_rates(old, new, wall_s) for old, new in zip(before["nodes"], after["nodes"])
        ] if after is not None else None,
        "server_resources": resource_usage,
        "cpu_core_s_per_request": (resource_usage["cpu_core_seconds"] / len(results)
                                   if resource_usage else None),
        "faults": summarize_faults(fault_events, results, args, start_t, series, baseline),
        "faults_scheduled": len(args.fault_offsets_s or []),
        "faults_applied": len(fault_events),
    }
    for label in ("ttft_sched_ms", "ttft_send_ms", "tpot_ms", "e2e_ms"):
        summary[label] = percentiles(r[label] for r in completed)
    summary["input_tokens"] = sum(r["input_tokens"] for r in completed)
    summary["output_tokens"] = sum(r["output_tokens"] for r in completed)
    summary["output_length_tokens"] = percentiles(r["output_tokens"] for r in completed)
    summary["requested_output_length_tokens"] = percentiles(r["planned_output_tokens"] for r in results)
    if args.cold_oracle:
        affected = {idx for event in summary["faults"] for idx in event["affected_req_indices"]}
        summary["cold_oracle_exactness"] = compare_oracle(args.cold_oracle, results)
        if args.fault_pid_file:
            summary["fault_affected_exactness"] = compare_oracle(args.cold_oracle, results, affected)
    if args.save_outputs:
        path = Path(args.save_outputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True)
    ap.add_argument("--case", default="toolagent")
    ap.add_argument("--model-path")
    ap.add_argument("--served-model-name", default="qwen3-30b-a3b")
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--frozen-window")
    inputs.add_argument("--trace-path")
    ap.add_argument("--node0-url", default="http://127.0.0.1:8000")
    ap.add_argument("--node1-url", default="http://127.0.0.1:8001")
    ap.add_argument("--warmup-requests", type=int, default=WARMUP_REQUESTS)
    ap.add_argument("--measure-requests", type=int, default=MEASUREMENT_REQUESTS)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--tokens-per-hash-id", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--clamp-output-max", type=int)
    load = ap.add_mutually_exclusive_group()
    load.add_argument("--qps", type=float)
    load.add_argument("--time-scale", type=float)
    ap.add_argument("--sticky-ratio", type=float, default=0.70)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=300)
    ap.add_argument("--progress-interval-reqs", type=int, default=200)
    ap.add_argument("--vllm-drain", action=argparse.BooleanOptionalAction, default=True,
                    help="Use the vLLM wait barrier without clearing KV at phase boundaries")
    ap.add_argument("--stats-socket", nargs=2, metavar=("NODE0", "NODE1"))
    ap.add_argument("--nvme-dir", nargs=2, metavar=("NODE0", "NODE1"))
    ap.add_argument("--hca-device", action="append", default=[])
    ap.add_argument("--server-pid", type=int, action="append", default=[])
    ap.add_argument("--gpu-id", action="append", default=[])
    ap.add_argument("--aux-metrics-url", action="append", default=[])
    ap.add_argument("--fault-pid-file")
    ap.add_argument("--fault-offsets-s", type=float, nargs="+")
    ap.add_argument("--fault-duration-s", type=float)
    ap.add_argument("--fault-requester-node", type=int, choices=(0, 1), default=1)
    ap.add_argument("--deployment-id", help="Stable launcher configuration fingerprint")
    ap.add_argument("--paired-baseline", help="Matching no-fault replay summary JSON")
    ap.add_argument("--record-logprobs", action="store_true")
    ap.add_argument("--cold-oracle", help="Cold BI=1 result JSONL keyed by req_idx")
    ap.add_argument("--output-json")
    ap.add_argument("--save-outputs")
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (args.warmup_requests, args.measure_requests, args.max_model_len, args.max_concurrency,
                args.timeout_s, args.progress_interval_reqs, args.qps, args.time_scale,
                args.clamp_output_max)
    if any(val is not None and (not math.isfinite(val) or val <= 0) for val in positive):
        ap.error("Request counts, limits and rates must be positive")
    fault = (args.fault_pid_file, args.fault_offsets_s, args.fault_duration_s)
    if any(value is not None for value in fault):
        if not all(fault) or not math.isfinite(args.fault_duration_s) or args.fault_duration_s <= 0:
            ap.error("Fault injection requires a PID file, offsets and positive duration")
        if not all(math.isfinite(offset) and offset >= 0 for offset in args.fault_offsets_s):
            ap.error("Fault offsets must be finite and nonnegative")
        if any(b < a + args.fault_duration_s for a, b in zip(args.fault_offsets_s, args.fault_offsets_s[1:])):
            ap.error("Fault offsets must be ordered without overlapping pauses")
    if args.cold_oracle and not args.record_logprobs:
        ap.error("--cold-oracle requires --record-logprobs")


def main() -> None:
    ap = parser()
    args = ap.parse_args()
    validate_args(ap, args)
    result = asyncio.run(run_replay(args))
    print(json.dumps(result, indent=2))
    if result["after_snapshot_error"]:
        raise SystemExit(f"After-snapshot failed: {result['after_snapshot_error']}")
    if result["faults_applied"] != result["faults_scheduled"]:
        raise SystemExit("Measurement ended before all scheduled faults applied")
    if args.cold_oracle and result["cold_oracle_exactness"]["mismatches"]:
        raise SystemExit("Measured outputs did not match the cold oracle")


if __name__ == "__main__":
    main()
