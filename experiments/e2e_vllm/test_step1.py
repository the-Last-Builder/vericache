from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time

from aiohttp import ClientSession, web
import pytest

from prepare_step1_window import select_window
from replayer import (
    ScheduledFaults,
    _prepare_requests,
    exactness,
    joint_slo,
    recovery_after,
    run_replay,
    send_streaming_completion,
)
from replayer import parser as replay_parser
from run_e2e import _numa_cmd, _vericache_frames, _vllm_flags, add_cluster_args
from smoke_step1 import verify_rdma, verify_storage_layout
from step1_metrics import ResourceSampler, read_hca
from trace_tokens import FRAME_BYTES, prefix_block_keys, prepare_trace_requests, route_request


def request(index: int, token: int, node: int, *, warmup: bool = False, length: int = 49) -> dict:
    return {
        "req_idx": index,
        "timestamp_ms": index * 10,
        "prompt_token_ids": [token] * length,
        "input_tokens": length,
        "output_tokens": 8,
        "hash_ids": [token],
        "session_key": token,
        "dest_node": node,
        "is_warmup": warmup,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_trace_preserves_partial_tail_and_prefix_parent(tmp_path):
    trace = tmp_path / "toolagent.jsonl"
    write_jsonl(trace, [
        {"timestamp": 0, "input_length": 513, "output_length": 182, "hash_ids": [10, 20]},
        {"timestamp": 1, "input_length": 529, "output_length": 99, "hash_ids": [10, 20]},
        {"timestamp": 2, "input_length": 529, "output_length": 11, "hash_ids": [11, 20]},
    ])
    rows, _ = prepare_trace_requests(trace, list(range(1000, 2500)), max_input_tokens=528)
    a, b, c = (r["prompt_token_ids"] for r in rows)
    assert list(map(len, (a, b, c))) == [513, 528, 528]
    assert b[:513] == a
    assert b[512:] == c[512:]
    assert prefix_block_keys(a) == prefix_block_keys(b)[:32]
    assert prefix_block_keys(b)[-1] != prefix_block_keys(c)[-1]
    assert rows[0]["max_tokens"] == 182
    assert route_request(1, [10, 20])[0] == route_request(1, [20])[0] == 20


def test_scanner_selects_first_contiguous_window_with_both_paths():
    rows = [request(0, 9, 0), request(1, 10, 0), request(2, 20, 1),
            request(3, 20, 0), request(4, 10, 1), request(5, 99, 0)]
    frozen, summary = select_window(rows, 2, 2, 1, 1, capacities=(1, 1, 12))
    assert [row["req_idx"] for row in frozen] == [1, 2, 3, 4]
    assert [row["is_warmup"] for row in frozen] == [True, True, False, False]
    assert [row["dest_node"] for row in frozen] == [0, 1, 0, 1]
    for node in summary["measurement_per_node"]:
        assert node["remote_nvme_blocks"] >= 1
        assert node["first_cross_node_bytes"] > 0


def test_frozen_window_native_output_length_and_clamp(tmp_path):
    window = tmp_path / "window.jsonl"
    rows = [request(10, 11, 1, warmup=True), request(11, 99, 0)]
    write_jsonl(window, rows)
    argv = ["--system", "test", "--frozen-window", str(window),
            "--warmup-requests", "1", "--measure-requests", "1", "--max-model-len", "55"]
    assert [row["max_tokens"] for row in _prepare_requests(replay_parser().parse_args(argv))] == [6, 6]
    parsed = _prepare_requests(replay_parser().parse_args([*argv, "--clamp-output-max", "3"]))
    assert parsed[0]["dest_node"] == 1 and parsed[0]["max_tokens"] == 3
    assert parsed[0]["prompt_token_ids"] == rows[0]["prompt_token_ids"]
    rows[1]["is_warmup"] = True
    write_jsonl(window, rows)
    with pytest.raises(ValueError, match="Invalid frozen request"):
        _prepare_requests(replay_parser().parse_args(argv))


async def with_server(handler, metrics, operation):
    app = web.Application()
    async def pause(request):
        assert request.query["mode"] == "wait"
        assert request.query["clear_cache"] == "false"
        return web.json_response({"status": "paused"})

    async def resume(_request):
        return web.json_response({"status": "resumed"})

    app.router.add_post("/v1/completions", handler)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/pause", pause)
    app.router.add_post("/resume", resume)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        host, port = runner.addresses[0]
        await operation(f"http://{host}:{port}")
    finally:
        await runner.cleanup()


async def sse(response, data):
    await response.write(f"data: {json.dumps(data)}\n\n".encode())


def test_replay_timing_and_snapshots_exclude_warmup(tmp_path):
    counts = []
    posts = []

    async def handler(request):
        payload = await request.json()
        posts.append(payload["prompt"][0])
        assert payload["return_token_ids"] and payload["max_tokens"] == 8
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        # Empty text can still carry a generated token.
        await sse(response, {"choices": [{"token_ids": [10], "text": ""}]})
        await asyncio.sleep(0.02)
        await sse(response, {"choices": [{"token_ids": [20, 21], "finish_reason": "length"}]})
        await asyncio.sleep(0.13)
        await sse(response, {"usage": {"completion_tokens": 3}})
        await response.write(b"data: [DONE]\n\n")
        return response

    async def metrics(_request):
        counts.append(len(posts))
        return web.Response(text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                                 f"vllm:num_preemptions_total {len(posts)}\n")

    async def operation(url):
        args = replay_parser().parse_args([
            "--system", "test", "--frozen-window", str(tmp_path / "window.jsonl"),
            "--node0-url", url, "--node1-url", url, "--warmup-requests", "1",
            "--measure-requests", "2", "--qps", "20", "--max-concurrency", "1",
            "--save-outputs", str(tmp_path / "requests.jsonl"),
        ])
        report = await run_replay(args)
        saved = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
        assert counts == [1] * 4 + [3] * 4
        assert posts == [11, 22, 33]
        assert [r["node_id"] for r in saved] == [0, 1]
        assert report["completed"] == report["joint_slo_met"] == report["total_requests"] == 2
        assert report["planned_window_goodput_req_s"] == 20
        assert report["wall_clock_goodput_req_s"] < 20
        assert saved[1]["ttft_sched_ms"] > saved[1]["ttft_send_ms"] + 50
        for result in saved:
            expected = (result["last_token_t"] - result["first_token_t"]) * 1000 / 2
            assert result["tpot_ms"] == expected < 50
            assert result["finished_t"] - result["last_token_t"] > 0.10

    write_jsonl(tmp_path / "window.jsonl", [
        request(0, 11, 1, warmup=True), request(1, 22, 0), request(2, 33, 1),
    ])
    asyncio.run(with_server(handler, metrics, operation))


def test_missing_after_snapshot_preserves_measurement_slo_results(tmp_path):
    sent = []

    async def handler(request):
        sent.append((await request.json())["prompt"][0])
        response = web.StreamResponse()
        await response.prepare(request)
        await sse(response, {"choices": [{"token_ids": [9], "finish_reason": "length"}]})
        await sse(response, {"usage": {"completion_tokens": 1}})
        await response.write(b"data: [DONE]\n\n")
        return response

    async def metrics(_request):
        if len(sent) > 1:
            return web.Response(status=503)
        return web.Response(text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n")

    async def operation(url):
        args = replay_parser().parse_args([
            "--system", "test", "--frozen-window", str(tmp_path / "window.jsonl"),
            "--node0-url", url, "--node1-url", url, "--warmup-requests", "1",
            "--measure-requests", "1", "--output-json", str(tmp_path / "report.json"),
        ])
        result = await run_replay(args)
        assert result["joint_slo_met"] == 1
        assert result["preemptions"] is None and result["delta"] is None
        assert "503" in result["after_snapshot_error"]
        assert json.loads((tmp_path / "report.json").read_text())["completed"] == 1

    write_jsonl(tmp_path / "window.jsonl", [request(0, 11, 0, warmup=True), request(1, 22, 1)])
    asyncio.run(with_server(handler, metrics, operation))


def test_token_ids_and_bit_exact_logprobs():
    async def handler(request):
        payload = await request.json()
        assert payload["logprobs"] == 1
        assert payload["return_tokens_as_token_ids"] and payload["echo"] is False
        response = web.StreamResponse()
        await response.prepare(request)
        await sse(response, {"choices": [{
            "token_ids": [17], "finish_reason": "length",
            "logprobs": {"top_logprobs": [{"token_id:17": -0.0, "token_id:99": -14.0}]},
        }], "usage": {"completion_tokens": 1}})
        await response.write(b"data: [DONE]\n\n")
        return response

    async def metrics(_request):
        return web.Response()

    async def operation(url):
        req = dict(request(1, 2, 0), max_tokens=1)
        async with ClientSession() as session:
            result = await send_streaming_completion(session, url, "model", req, 0, record_logprobs=True)
        assert joint_slo(result) and result["tpot_ms"] is None
        assert result["token_ids"] == [17]
        assert exactness(result, result)["exact"]
        assert exactness(result, dict(result, top1_logprobs=[0.0]))["logprob_mismatches"] == 1
        assert not joint_slo(dict(result, error="failed"))

    asyncio.run(with_server(handler, metrics, operation))


def test_legacy_text_only_stream_counts_as_a_miss():
    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await sse(response, {"choices": [{"text": "hello", "finish_reason": "length"}]})
        await sse(response, {"usage": {"completion_tokens": 1}})
        await response.write(b"data: [DONE]\n\n")
        return response

    async def metrics(_request):
        return web.Response()

    async def operation(url):
        async with ClientSession() as session:
            result = await send_streaming_completion(
                session, url, "test", dict(request(1, 2, 0), max_tokens=1), 0,
            )
        assert result["output_tokens"] == 0
        assert result["ttft_sched_ms"] is None and result["tpot_ms"] is None
        assert not joint_slo(result)

    asyncio.run(with_server(handler, metrics, operation))


def test_external_worker_fault_schedule():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        async def exercise():
            async def capture():
                return {"captured": True}

            loop = asyncio.get_running_loop()
            fault = ScheduledFaults(
                proc.pid, [0.02], 0.15, lambda: asyncio.run_coroutine_threadsafe(capture(), loop),
            )
            start = time.perf_counter()
            fault.start(start)
            try:
                limit = time.perf_counter() + 0.12
                state = ""
                while time.perf_counter() < limit:
                    state = Path(f"/proc/{proc.pid}/status").read_text().split("State:")[1].lstrip()[0]
                    if state == "T":
                        break
                    await asyncio.sleep(0.005)
                assert state == "T"
            finally:
                events = await fault.finish()
            assert len(events) == 1
            event = events[0]
            assert event["resume_t"] >= event["pause_t"] + 0.10
            assert event["resume_t"] >= start + 0.17 - 0.005
            assert event["donor_before"] == event["donor_after"] == {"captured": True}

        asyncio.run(exercise())
    finally:
        proc.kill()
        proc.wait()


def test_recovery_requires_three_full_nonzero_paired_windows():
    baseline = [{"start_s": 10 * i, "duration_s": 10, "goodput_req_s": 1} for i in range(6)]
    current = [dict(window, goodput_req_s=0.95) for window in baseline]
    assert recovery_after(11, current, baseline)["time_to_confirm_recovery_s"] == 39
    assert not recovery_after(11, current, baseline, next_pause=49)["recovered"]
    baseline[3]["goodput_req_s"] = 0
    assert not recovery_after(11, current, baseline)["recovered"]


def test_launcher_zero_frames_frozen_flags_and_numa_failure(monkeypatch):
    parser = argparse.ArgumentParser()
    add_cluster_args(parser)
    args = parser.parse_args(["--model-path", "/tmp/model"])
    assert args.mooncake_lease_ttl_ms == 10000
    assert _vericache_frames(args, FRAME_BYTES)[0] == 0
    assert parser.parse_args(["--model-path", "/tmp/model", "--lmcache-chunk-size", "64"]).lmcache_chunk_size == 64
    flags = _vllm_flags(args, FRAME_BYTES * 100)
    assert "--async-scheduling" in flags and "FLASH_ATTN" in flags
    assert json.loads(flags[flags.index("--compilation-config") + 1])["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    args.mode = "correctness"
    flags = _vllm_flags(args, FRAME_BYTES * 100)
    assert "--enforce-eager" in flags and "--no-async-scheduling" in flags
    monkeypatch.setattr("run_e2e.shutil.which", lambda _: "/usr/bin/numactl")
    monkeypatch.setattr("run_e2e.subprocess.run", lambda *_a, **_kw:
                        subprocess.CompletedProcess([], 1, stderr="set_mempolicy: Operation not permitted"))
    with pytest.raises(RuntimeError, match="Operation not permitted"):
        _numa_cmd(0)


def test_hca_words_and_gate_require_both_directions(tmp_path):
    for dev in ("mlx5_0", "mlx5_1"):
        for group in ("counters", "hw_counters"):
            (tmp_path / dev / "ports" / "1" / group).mkdir(parents=True)
        for key in ("port_xmit_data", "port_rcv_data"):
            (tmp_path / dev / "ports" / "1" / "counters" / key).write_text("10\n")
        (tmp_path / dev / "ports" / "1" / "hw_counters" / "out_of_buffer").write_text("2\n")
    devices = ["mlx5_0", "mlx5_1"]
    before = read_hca(devices, tmp_path)
    assert before["mlx5_0"]["port_rcv_bytes"] == 40
    assert before["mlx5_0"]["hw_counters/out_of_buffer"] == 2
    after = {dev: dict(stats) for dev, stats in before.items()}
    after["mlx5_0"]["port_xmit_bytes"] += FRAME_BYTES
    with pytest.raises(RuntimeError, match="shortfall"):
        verify_rdma(before, after, devices, FRAME_BYTES)
    after["mlx5_1"]["port_rcv_bytes"] += FRAME_BYTES
    assert verify_rdma(before, after, devices, FRAME_BYTES)["expected_payload_bytes"] == FRAME_BYTES


def test_gate1_checks_actual_worker_geometry():
    good = dict(hbm_frames=0, host_pinned=1, is_host_pinned=True, direct_slab_dma=True,
                load_direct_slab_dma=True, staging_fallback=False, gdr_enabled=0, host_rdma=True,
                frame_bytes=FRAME_BYTES, num_kv_groups=1, block_size=16, dram_frames=16, nvme_frames=32)
    verify_storage_layout([good, good])
    with pytest.raises(RuntimeError, match="direct_slab_dma"):
        verify_storage_layout([good, dict(good, direct_slab_dma=False)])


def test_server_sampler_counts_exited_process_cpu():
    child = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    sampler = ResourceSampler([child.pid], [], interval=0.03)
    sampler.start()
    try:
        start = time.monotonic()
        while time.monotonic() < start + 2 and max(sampler.last_cpu.values(), default=0) < 0.05:
            time.sleep(0.03)
        child.terminate()
        child.wait(timeout=2)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        result = sampler.stop()
    assert result["cpu_core_seconds"] > 0
    assert result["peak_rss_bytes"] > 0 and result["peak_pss_bytes"] > 0
