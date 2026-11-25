#!/usr/bin/env python3
"""Run the Step 1 pinned-slab, storage-path and BI=1 exactness gates on two H20s."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import datetime
import json
from pathlib import Path
import random
import re
import signal
import time

import aiohttp

from replayer import _top1_logprobs, exactness, scrape_vllm_metrics, send_streaming_completion, wait_idle
from run_e2e import ClusterContext, RESULTS_ROOT, _vericache_frames, add_cluster_args
from step1_metrics import GIB, numeric_delta, read_hca, read_nvme_usage, read_vericache, sum_metrics
from trace_tokens import FRAME_BYTES, build_safe_vocab

PROBE_TOKENS = 2049
OUTPUT_TOKENS = 8
GPU_HITS = "vllm:prefix_cache_hits_total"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits_total"
PATHS = ("hbm", "dram", "local_nvme", "remote_dram", "remote_nvme")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_storage_layout(stats: list[dict]) -> None:
    expected = {
        "hbm_frames": 0,
        "host_pinned": 1,
        "is_host_pinned": True,
        "direct_slab_dma": True,
        "load_direct_slab_dma": True,
        "staging_fallback": False,
        "gdr_enabled": 0,
        "host_rdma": True,
        "frame_bytes": FRAME_BYTES,
        "num_kv_groups": 1,
        "block_size": 16,
    }
    require(len(stats) == 2, "Gate 1 requires both worker snapshots")
    for node, snapshot in enumerate(stats):
        mismatches = {k: (v, snapshot.get(k)) for k, v in expected.items() if snapshot.get(k) != v}
        require(not mismatches, f"Gate 1 node {node} storage layout mismatch (expected, actual): {mismatches}")
        require(snapshot["dram_frames"] > 0 and snapshot["nvme_frames"] > 0,
                f"Gate 1 node {node} is missing a DRAM or NVMe tier")


def verify_rdma(before: dict, after: dict, devices: list[str], payload_bytes: int) -> dict:
    donor, requester = devices
    tx = after[donor]["port_xmit_bytes"] - before[donor]["port_xmit_bytes"]
    rx = after[requester]["port_rcv_bytes"] - before[requester]["port_rcv_bytes"]
    require(payload_bytes > 0 and tx >= payload_bytes and rx >= payload_bytes,
            f"Gate 2 RDMA byte shortfall: donor TX={tx}, requester RX={rx}, payload={payload_bytes}")
    return {"donor_tx_bytes": tx, "requester_rx_bytes": rx, "expected_payload_bytes": payload_bytes}


class Requests:
    def __init__(self, vocabulary: list[int]):
        self.vocabulary = vocabulary
        self.rng = random.Random(42)
        self.index = 0

    def new(self, length: int = PROBE_TOKENS, output: int = OUTPUT_TOKENS) -> dict:
        result = {
            "req_idx": self.index,
            "prompt_token_ids": self.rng.choices(self.vocabulary, k=length),
            "input_tokens": length,
            "max_tokens": output,
        }
        self.index += 1
        return result


@contextmanager
def deadline(seconds: float, label: str):
    def expired(_signum, _frame):
        raise TimeoutError(f"{label} exceeded its wall-clock budget")

    handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


class SmokeClient:
    def __init__(self, session: aiohttp.ClientSession, cluster: ClusterContext):
        self.session = session
        self.cluster = cluster

    async def metrics(self, node: int) -> dict:
        return sum_metrics(await scrape_vllm_metrics(self.session, self.cluster.urls[node]))

    async def complete(self, req: dict, node: int, *, exact: bool = False) -> dict:
        result = await send_streaming_completion(
            self.session, self.cluster.urls[node], self.cluster.args.served_model_name,
            req, node, record_logprobs=exact, timeout_s=90,
        )
        require(result["error"] is None, f"{self.cluster.system} node {node} request failed: {result['error']}")
        return result

    async def batch(self, requests: list[dict], node: int) -> list[dict]:
        # One completions batch submits the target with distractors of different lengths.
        async with self.session.post(
            f"{self.cluster.urls[node]}/v1/completions",
            json={
                "model": self.cluster.args.served_model_name,
                "prompt": [req["prompt_token_ids"] for req in requests],
                "max_tokens": OUTPUT_TOKENS, "temperature": 0, "seed": 42,
                "ignore_eos": True, "echo": False, "logprobs": 1,
                "return_tokens_as_token_ids": True, "return_token_ids": True,
            },
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        choices = sorted(payload["choices"], key=lambda choice: choice["index"])
        require(len(choices) == len(requests), "Cold batch response cardinality mismatch")
        results = []
        for choice in choices:
            token_ids = choice["token_ids"]
            top1 = _top1_logprobs(choice, token_ids)
            require(len(token_ids) == OUTPUT_TOKENS and choice["finish_reason"] in ("length", "stop"),
                    "Cold batch token IDs or finish reason are missing")
            results.append({
                "output_tokens": len(token_ids), "token_ids": token_ids,
                "top1_token_ids": [token for token, _ in top1],
                "top1_logprobs": [value for _, value in top1],
                "error": None,
            })
        return results

    async def drain(self) -> list[dict]:
        await wait_idle(self.session, self.cluster.urls)
        return await asyncio.gather(*(asyncio.to_thread(
            read_vericache, path, drain=True, timeout=30,
        ) for path in self.cluster.stats_sockets))

    async def reset_hbm(self, node: int) -> None:
        reset_url = f"{self.cluster.urls[node]}/reset_prefix_cache?reset_external=false"
        limit = time.perf_counter() + 30
        await self.drain()
        while True:
            async with self.session.post(reset_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                if (await response.json())["success"]:
                    return
            if time.perf_counter() >= limit:
                raise TimeoutError(f"Failed to reset node {node} HBM prefix cache")
            await asyncio.sleep(0.1)

    async def wait_for_writes(self, node: int, *, min_puts: int | None = None) -> list[dict]:
        limit = time.perf_counter() + 25
        while True:
            stats = await self.drain()
            if min_puts is not None:
                if stats[node]["puts"] >= min_puts:
                    return stats
            else:
                nvme = read_nvme_usage(str(self.cluster.nvme_paths[node]))
                if nvme["files"] and nvme["allocated_bytes"] > 0:
                    return stats
            if time.perf_counter() >= limit:
                raise TimeoutError(f"{self.cluster.system} node {node} did not finish expected writes")
            await asyncio.sleep(0.1)

    async def port_evidence(self, before: dict, expected: int) -> dict:
        limit = time.perf_counter() + 3
        devices = self.cluster.hca_devices
        while True:
            after = read_hca(devices)
            try:
                return verify_rdma(before, after, devices, expected)
            except RuntimeError:
                if time.perf_counter() >= limit:
                    raise
                await asyncio.sleep(0.1)


async def cold_oracles(client: SmokeClient, prompts: dict, requests: Requests, report: dict) -> dict:
    before = [await client.metrics(node) for node in (0, 1)]
    oracle = {}
    comparisons = report["cold_to_cold"] = {}
    report["cold_batch_sizes"] = [1, 3]
    for path, req in prompts.items():
        oracle[path] = await client.complete(req, 0, exact=True)
        batch = [req, requests.new(1025), requests.new(4097)]
        other, *_ = await client.batch(batch, 1)
        comparisons[path] = exactness(oracle[path], other)
        require(comparisons[path]["exact"], f"Gate 3 cold→cold mismatch for {path}: {comparisons[path]}")
    metrics = [numeric_delta(before[node], await client.metrics(node)) for node in (0, 1)]
    require(all(m.get(GPU_HITS, 0) == m.get(EXTERNAL_HITS, 0) == 0 for m in metrics),
            "Gate 3 cold oracle unexpectedly reused cached tokens")
    return oracle


async def prime_vericache(client: SmokeClient, req: dict) -> list[dict]:
    before = await client.drain()
    await client.complete(req, 0)
    # The 2049-token prompt has 128 full blocks. Wait for the scheduler's async store.
    return await client.wait_for_writes(0, min_puts=before[0]["puts"] + req["input_tokens"] // 16)


async def vericache_gates(client: SmokeClient, prompts: dict, requests: Requests,
                         oracle: dict, report: dict) -> None:
    cluster = client.cluster
    startup = await client.drain()
    verify_storage_layout(startup)
    report["gate1"] = {"pass": True, "workers": startup}
    evidence, comparisons = {}, {}
    report["gate3"]["cold_to_cached"] = comparisons
    report["gate2"]["vericache"] = {"pass": False, "paths": evidence}

    async def evaluate(path: str, node: int, *, before: list[dict] | None = None,
                       hca_before: dict | None = None, disk_before: dict | None = None,
                       gpu_before: dict | None = None) -> None:
        answer = await client.complete(prompts[path], node, exact=True)
        comparisons[path] = exactness(oracle[path], answer)
        require(comparisons[path]["exact"], f"Gate 3 cold→{path} mismatch: {comparisons[path]}")
        after = await client.drain()
        if gpu_before is not None:
            delta = numeric_delta(gpu_before, await client.metrics(node))
            require(delta.get(GPU_HITS, 0) > 0, "Gate 3 HBM probe did not hit the vLLM prefix cache")
            evidence[path] = {"hbm_hit_tokens": delta[GPU_HITS]}
            return
        donor, requester = (numeric_delta(before[i], after[i]) for i in (0, node))
        if path == "dram":
            require(requester["dram_hits"] > 0, "Local pinned DRAM restore was not observed")
            evidence[path] = {"dram_hits": requester["dram_hits"]}
        elif path == "local_nvme":
            require(requester["nvme_hits"] > 0, "Local NVMe restore was not observed")
            evidence[path] = {"nvme_hits": requester["nvme_hits"]}
        else:
            require(requester["remote_hits"] > 0, f"Gate 2 {path} did not actually use a peer")
            warmed = requester["remote_nvme_warm_hits"]
            proof = await client.port_evidence(hca_before, requester["remote_hits"] * FRAME_BYTES)
            if path == "remote_dram":
                require(warmed == 0 and donor["donor_nvme_warms"] == 0,
                        "Gate 2 DRAM probe unexpectedly warmed from NVMe")
            else:
                disk = read_nvme_usage(str(cluster.nvme_paths[0]), include_usage=False)
                read_bytes = disk["device_read_bytes"] - disk_before["device_read_bytes"]
                require(warmed > 0 and donor["donor_nvme_warms"] > 0 and read_bytes >= FRAME_BYTES,
                        "Gate 2 donor NVMe warm, physical SSD read and successful remote restore did not agree")
                proof.update(donor_nvme_warms=donor["donor_nvme_warms"], donor_ssd_read_bytes=read_bytes)
            proof.update(remote_hits=requester["remote_hits"], remote_nvme_warm_hits=warmed)
            evidence[path] = proof

    await prime_vericache(client, prompts["hbm"])
    await evaluate("hbm", 0, gpu_before=await client.metrics(0))
    await prime_vericache(client, prompts["dram"])
    await client.reset_hbm(0)
    await evaluate("dram", 0, before=await client.drain())

    before = await prime_vericache(client, prompts["remote_dram"])
    await evaluate("remote_dram", 1, before=before, hca_before=read_hca(cluster.hca_devices))

    await prime_vericache(client, prompts["local_nvme"])
    await prime_vericache(client, prompts["remote_nvme"])
    _, dram_frames, _ = _vericache_frames(cluster.args, FRAME_BYTES)
    filler_count = dram_frames // (PROBE_TOKENS // 16) + 2
    for _ in range(filler_count):
        await prime_vericache(client, requests.new(output=1))
    await client.reset_hbm(0)

    before = await client.drain()
    require(before[0]["nvme_spills"] > 0, "Gate 2 pressure did not spill any donor frames")
    await evaluate("remote_nvme", 1, before=before, hca_before=read_hca(cluster.hca_devices),
                   disk_before=read_nvme_usage(str(cluster.nvme_paths[0]), include_usage=False))
    await evaluate("local_nvme", 0, before=await client.drain())
    report["gate2"]["vericache"]["pass"] = True


async def vendor_gate(client: SmokeClient, requests: Requests, evidence: dict) -> None:
    cluster = client.cluster
    remote = requests.new()
    await client.complete(remote, 0)
    await client.drain()
    # Give background controller/store publication a chance to finish before the first requester lookup.
    await asyncio.sleep(0.5)
    before = await client.metrics(1)
    hca = read_hca(cluster.hca_devices)
    await client.complete(remote, 1)
    await client.drain()
    received = numeric_delta(before, await client.metrics(1))
    external = received.get(EXTERNAL_HITS, 0)
    require(external > 0 and received.get(GPU_HITS, 0) == 0,
            f"Gate 2 {cluster.system} cross-node request did not report an external prefix hit")
    evidence["cross_node"] = await client.port_evidence(hca, int(external) * FRAME_BYTES // 16)
    evidence["cross_node"]["external_hit_tokens"] = external

    target = requests.new()
    await client.complete(target, 0)
    # Mooncake's master can allocate on either store. Pressure both for its SSD gate.
    pressure_nodes = (0, 1) if cluster.system == "mooncake_native" else (0,)
    byte_budget = int(cluster.args.dram_gb_per_node * GIB)
    pressure_count = byte_budget // (PROBE_TOKENS // 16 * FRAME_BYTES) + 3
    for _ in range(pressure_count):
        await asyncio.gather(*(client.complete(requests.new(output=1), node) for node in pressure_nodes))
    await client.wait_for_writes(0)
    await client.reset_hbm(0)
    if cluster.system == "lmcache":
        log = (cluster.work_dir / "vllm_node0.log").read_text(errors="replace")
        require(re.search(r"Using O_DIRECT for disk I/O:\s*True", log) is not None,
                "Gate 2 LMCache LocalDisk did not confirm O_DIRECT")
    paths = [str(path) for path in cluster.nvme_paths]
    ssd_before = [read_nvme_usage(path, include_usage=False) for path in paths]
    before = await client.metrics(0)
    await client.complete(target, 0)
    await client.drain()
    restored = numeric_delta(before, await client.metrics(0))
    ssd_reads = [read_nvme_usage(path, include_usage=False)["device_read_bytes"] - old["device_read_bytes"]
                 for path, old in zip(paths, ssd_before)]
    external = restored.get(EXTERNAL_HITS, 0)
    min_bytes = (cluster.args.lmcache_chunk_size * FRAME_BYTES // 16
                 if cluster.system == "lmcache" else FRAME_BYTES)
    relevant_ssd_reads = ssd_reads[0] if cluster.system == "lmcache" else sum(ssd_reads)
    require(external > 0 and restored.get(GPU_HITS, 0) == 0 and relevant_ssd_reads >= min_bytes,
            f"Gate 2 {cluster.system} SSD restore missing: external tokens={external}, SSD bytes={ssd_reads}")
    evidence["ssd"] = {"external_hit_tokens": external, "device_read_bytes_by_node": ssd_reads}


def run_cluster(system: str, args: argparse.Namespace, output: Path, seconds: float, operation):
    require(seconds > 0, f"No smoke budget left for {system}")
    cluster = ClusterContext(system, args, output / system)
    try:
        with deadline(seconds, system):
            cluster.start()

            async def execute():
                async with aiohttp.ClientSession() as session:
                    return await operation(SmokeClient(session, cluster))

            return asyncio.run(execute())
    finally:
        try:
            if not (cluster.work_dir / "manifest.json").exists():
                cluster.write_manifest()
        finally:
            cluster.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    parser.set_defaults(mode="correctness", dram_gb_per_node=0.75, nvme_gb_per_node=2)
    parser.add_argument("--deadline-s", type=float, default=900)
    parser.add_argument("--lmcache-timeout-s", type=float, default=180)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode != "correctness" or args.vericache_hbm_gb_per_node != 0 or not args.lmcache_p2p:
        parser.error("The three gates require BI=1 correctness mode, zero VeriCache HBM and LMCache P2P")
    if not 0 < args.dram_gb_per_node < args.nvme_gb_per_node:
        parser.error("Smoke requires positive DRAM and a larger NVMe budget")
    if min(args.deadline_s, args.lmcache_timeout_s) <= 0:
        parser.error("Smoke deadlines must be positive")

    output = args.output_dir or RESULTS_ROOT / datetime.datetime.now(datetime.timezone.utc).strftime(
        "step1_smoke_%Y%m%dT%H%M%S%fZ",
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict = {"gate1": {"pass": False}, "gate2": {"pass": False}, "gate3": {"pass": False}}
    overall = False

    def remaining() -> float:
        return args.deadline_s - (time.perf_counter() - started)

    try:
        requests = Requests(build_safe_vocab(args.model_path))
        prompts = {path: requests.new() for path in PATHS}
        oracle = run_cluster("baseline", args, output, remaining(),
                             lambda client: cold_oracles(client, prompts, requests, report["gate3"]))
        run_cluster("vericache", args, output, remaining(),
                    lambda client: vericache_gates(client, prompts, requests, oracle, report))
        report["gate3"]["pass"] = True
        for vendor in ("mooncake_native", "lmcache"):
            evidence = report["gate2"][vendor] = {"pass": False}
            budget = min(remaining(), args.lmcache_timeout_s) if vendor == "lmcache" else remaining()
            try:
                run_cluster(vendor, args, output, budget, lambda client: vendor_gate(client, requests, evidence))
                evidence["pass"] = True
            except Exception as exc:
                evidence["error"] = str(exc)
        report["gate2"]["pass"] = all(report["gate2"][system]["pass"]
                                     for system in ("vericache", "mooncake_native", "lmcache"))
        report["lmcache_four_tier_eligible"] = report["gate2"]["lmcache"]["pass"]
        overall = all(report[name]["pass"] for name in ("gate1", "gate2", "gate3"))
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        report["pass"] = overall
        report["elapsed_s"] = time.perf_counter() - started
        (output / "gates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
