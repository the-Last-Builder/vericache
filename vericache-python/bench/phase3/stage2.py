"""Stage 2 — Exp 2.2: Dual-Instance Cross-NUMA Trace Replay Comparison (Scheme A).

Compares 4 cross-instance systems on identical 2x H20 GPUs across NUMA 0 <-> NUMA 1,
identical 70% sticky / 30% drift router (`sticky_router.py`), identical frozen FAST'25
traces (`toolagent_routed`), and identical capacity budgets (19.5 GiB HBM + 36 GiB DRAM
per instance = 72 GiB pooled DRAM):
  1. `vericache_local`  : VeriCache (peer_fetch=off, isolated 2-instance baseline)
  2. `vericache_rdma`   : VeriCache (peer_fetch=on, Scheme A GlobalDirectory + Type-2B MW RDMA)
  3. `lmcache_p2p`      : LMCache (lmcache_controller + P2PBackend / NIXL UCX over RoCEv2)
  4. `mooncake_rdma`    : LMCache + shared MooncakeStore 0.3.14 over RoCEv2
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
from threading import Event, Thread
import time
from urllib.request import Request, urlopen

import psutil

from exp21_safety_gate import (
    BLOCK_TOKENS,
    FRAME_BYTES,
    GIB,
    GPU_TOPOLOGY,
    HBM_BLOCKS,
    KV_BLOCK_BYTES,
    LAYERS,
    hca_delta,
    read_hca_counters,
    reserve_port_block,
    resolve_endpoint,
    stop_proc,
    wait_server_ready,
)
from prepare_traces import read_rows
from stage0_correctness import MODEL, PHASE3
from stage0_layout import model_command, reserve_port
from stage1 import (
    DEFAULT_REQUEST_TIMEOUT_SECS,
    analyze,
    dump,
    file_sha256,
    metric_totals,
    parse_server_log_peaks,
    replay_options,
    run_replay,
    scrape,
    trace_source,
)

STAGE2_SYSTEMS = ("vericache_local", "vericache_rdma", "lmcache_p2p", "mooncake_rdma")
DRAM_GIB_PER_INSTANCE = 36


def sample_dual_cpu(done: Event, samples: list, pids_getter):
    while not done.is_set():
        usage = {}
        for root_pid in pids_getter():
            try:
                root = psutil.Process(root_pid)
                procs = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                continue
            for p in procs:
                try:
                    name = p.name()
                    if name in {"vllm", "python", "python3", "mooncake_master", "lmcache_controller"} or name.startswith("VLLM"):
                        cpu = p.cpu_times()
                        usage[p.pid] = {
                            "cpu_seconds": cpu.user + cpu.system,
                            "rss_bytes": p.memory_info().rss,
                        }
                except psutil.NoSuchProcess:
                    continue
        samples.append({"monotonic": time.monotonic(), "processes": usage})
        done.wait(0.5)


def wait_tcp_port(host: str, port: int, proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Service process exited early rc={proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for TCP {host}:{port}")


def read_vericache_dist_stats(stats_dir: Path) -> dict[str, dict]:
    out = {}
    for idx in (0, 1):
        p = stats_dir / f"vericache-dist-stats-inst{idx}.json"
        if p.is_file():
            out[f"inst{idx}"] = json.loads(p.read_text())
    return out


def start_vllm_instance(
    *,
    system: str,
    inst_idx: int,
    gpu_id: int,
    peer_ip: str,
    http_port: int,
    dram_blocks: int,
    out: Path,
    run: Path,
    aux: dict,
) -> subprocess.Popen:
    numa, rnic, local_ip = resolve_endpoint(gpu_id, inst_idx)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["VLLM_BATCH_INVARIANT"] = "0"
    env["PYTHONPATH"] = (
        f"{PHASE3 / 'vericache-python/bench/phase3'}:"
        f"{PHASE3 / 'vericache-python/python'}:"
        + env.get("PYTHONPATH", "")
    )
    transfer: dict = {"kv_role": "kv_both"}
    lmc_tmp = Path("/tmp") / f"vc_s2_{os.getpid()}_i{inst_idx}"

    if system in ("vericache_local", "vericache_rdma"):
        # mlx5_3 enforces a 4-MR limit for >=1 GiB MRs; 4 shards x 9 GiB = 36 GiB exact.
        num_shards = 4
        frames_per_shard = (dram_blocks * LAYERS) // num_shards
        peer_fetch = system == "vericache_rdma"
        cfg = {
            "distributed_mode": True,
            "node_id": inst_idx,
            "instance_id": inst_idx,
            "num_shards": num_shards,
            "host_frames_per_shard": frames_per_shard,
            "nvme_frames_per_shard": 0,
            "frame_size": FRAME_BYTES,
            "prefer_cuda_pinned": True,
            "lib_path": str(PHASE3 / "vericache-python/target/phase3/libvericache_python.so"),
            "io_workers": 4,
            "staging_pool_size": 16,
            "rdma_device": rnic if peer_fetch else None,
            "rdma_gid_index": 3,
            "rdma_windows_per_peer": 16,
            "peer_fetch_enabled": peer_fetch,
            "global_directory_endpoint": aux["gdir_endpoint"] if peer_fetch else None,
            "peers": (
                [
                    {
                        "peer_node_id": 1 - inst_idx,
                        "tcp_host": aux["inst0_ip"],
                        "base_port": aux["rdma_base_port"],
                        "is_listener": inst_idx == 0,
                        "psn_hint": 0x1000,
                    }
                ]
                if peer_fetch
                else []
            ),
        }
        transfer.update({
            "kv_connector": "Stage2VeriCacheV1Connector",
            "kv_connector_module_path": "stage2_connector",
            "kv_connector_extra_config": cfg,
        })
        env["PHASE3_STAGE2_STATS_DIR"] = str(run)
    elif system == "lmcache_p2p":
        transfer["kv_connector"] = "LMCacheConnectorV1"
        lmc_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(lmc_tmp)
        env["PYTHONHASHSEED"] = "0"
        env["UCX_NET_DEVICES"] = f"{rnic}:1"
        env["UCX_IB_GID_INDEX"] = "3"
        env["UCX_TLS"] = "rc,ud,sm,self"
        init_port = aux[f"p2p_init_port_{inst_idx}"]
        lookup_port = aux[f"p2p_lookup_port_{inst_idx}"]
        worker_port = aux[f"lmc_worker_port_{inst_idx}"]
        profile = {
            "chunk_size": BLOCK_TOKENS,
            "local_cpu": True,
            "max_local_cpu_size": float(DRAM_GIB_PER_INSTANCE),
            "enable_controller": True,
            "lmcache_instance_id": f"lmc_inst{inst_idx}",
            "controller_pull_url": f"127.0.0.1:{aux['ctrl_pull_port']}",
            "controller_reply_url": f"127.0.0.1:{aux['ctrl_reply_port']}",
            "lmcache_worker_ports": [worker_port],
            "enable_p2p": True,
            "p2p_host": local_ip,
            "p2p_init_ports": [init_port],
            "p2p_lookup_ports": [lookup_port],
            "transfer_channel": "nixl",
            "nixl_backends": ["UCX"],
            "extra_config": {
                "lookup_backoff_time": 0.001,
            },
        }
        cfg_file = out / f"lmcache_inst{inst_idx}.yaml"
        cfg_file.write_text(json.dumps(profile, indent=2) + "\n")
        env["LMCACHE_CONFIG_FILE"] = str(cfg_file)
    elif system == "mooncake_rdma":
        transfer["kv_connector"] = "LMCacheConnectorV1"
        lmc_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(lmc_tmp)
        env["PYTHONHASHSEED"] = "0"
        staging_gib = 4.0
        align_16m = 16 << 20
        seg_bytes = ((int((DRAM_GIB_PER_INSTANCE - staging_gib) * GIB)) // align_16m) * align_16m
        mc_extra = {
            "save_chunk_meta": False,
            "local_hostname": local_ip,
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": seg_bytes,
            "local_buffer_size": 64 << 20,
            "protocol": "rdma",
            "device_name": rnic,
            "master_server_address": f"127.0.0.1:{aux['mc_rpc_port']}",
            "transfer_timeout": 15,
            "mooncake_prefer_local_alloc": True,
        }
        profile = {
            "chunk_size": BLOCK_TOKENS,
            "local_cpu": False,
            "max_local_cpu_size": staging_gib,
            "remote_url": f"mooncakestore://127.0.0.1:{aux['mc_rpc_port']}/",
            "remote_serde": "naive",
            "extra_config": mc_extra,
        }
        cfg_file = out / f"lmcache_inst{inst_idx}.yaml"
        cfg_file.write_text(json.dumps(profile, indent=2) + "\n")
        env["LMCACHE_CONFIG_FILE"] = str(cfg_file)
        env["MC_STORE_MEMCPY"] = "1"
    else:
        raise ValueError(f"unknown system {system}")

    argv = model_command(
        MODEL, BLOCK_TOKENS, http_port, "vanilla", tensor_parallel_size=1, max_num_seqs=32
    )
    argv.extend(["--num-gpu-blocks-override", str(HBM_BLOCKS)])
    argv.extend(["--attention-backend", "FLASH_ATTN"])
    argv.extend(["--kv-transfer-config", json.dumps(transfer)])

    full_argv = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", *argv]
    log_path = out / f"server_inst{inst_idx}.log"
    return subprocess.Popen(
        full_argv,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def main() -> None:
    job = json.loads(Path(os.environ["PHASE3_STAGE2_JOB"]).read_text())
    system = job["system"]
    assert system in STAGE2_SYSTEMS
    trace = job.get("trace", "toolagent_routed")
    scale = float(job["scale"])
    sticky_pct = int(job.get("sticky_pct", 70))
    seed = int(job.get("seed", 20261124))
    warmup_end_ms = int(job.get("warmup_end_ms", 114441))
    request_timeout_secs = int(job.get("request_timeout_secs", DEFAULT_REQUEST_TIMEOUT_SECS))

    gpu0 = int(os.environ["STAGE2_GPU0"])
    gpu1 = int(os.environ["STAGE2_GPU1"])
    numa0, rnic0, ip0 = resolve_endpoint(gpu0, 0)
    numa1, rnic1, ip1 = resolve_endpoint(gpu1, 1)
    assert numa0 == 0 and numa1 == 1

    run = Path(os.environ["PHASE3_TEST_DIR"])
    out = Path(os.environ["PHASE3_RESULT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "job.json", job)

    _, frozen = trace_source(trace)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS", "--quiet"], cwd=frozen, check=True)

    dram_blocks = (DRAM_GIB_PER_INSTANCE * GIB) // KV_BLOCK_BYTES
    port0 = reserve_port()
    port1 = reserve_port()
    router_port = reserve_port()
    ep0 = f"http://127.0.0.1:{port0}"
    ep1 = f"http://127.0.0.1:{port1}"
    router_ep = f"http://127.0.0.1:{router_port}"

    aux: dict = {"inst0_ip": ip0, "inst1_ip": ip1}
    aux_procs: list[subprocess.Popen] = []
    proc0 = None
    proc1 = None
    router_proc = None

    try:
        # 1. Start system-specific coordinator/control-plane service
        if system == "vericache_rdma":
            gdir_port = reserve_port()
            aux["gdir_endpoint"] = f"tcp://127.0.0.1:{gdir_port}"
            aux["rdma_base_port"] = reserve_port_block(4)
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{PHASE3 / 'vericache-python/python'}:" + env.get("PYTHONPATH", "")
            p = subprocess.Popen(
                [sys.executable, "-m", "vericache.global_directory", "--endpoint", aux["gdir_endpoint"]],
                stdout=(out / "global_directory.log").open("w"),
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            aux_procs.append(p)
            wait_tcp_port("127.0.0.1", gdir_port, p)
        elif system == "lmcache_p2p":
            ctrl_http = reserve_port()
            ctrl_pull = reserve_port()
            ctrl_reply = reserve_port()
            aux.update({
                "ctrl_http_port": ctrl_http,
                "ctrl_pull_port": ctrl_pull,
                "ctrl_reply_port": ctrl_reply,
                "p2p_init_port_0": reserve_port(),
                "p2p_lookup_port_0": reserve_port(),
                "lmc_worker_port_0": reserve_port(),
                "p2p_init_port_1": reserve_port(),
                "p2p_lookup_port_1": reserve_port(),
                "lmc_worker_port_1": reserve_port(),
            })
            ctrl_env = os.environ.copy()
            ctrl_env["PYTHONHASHSEED"] = "0"
            p = subprocess.Popen(
                [
                    "lmcache_controller",
                    "--host", "127.0.0.1",
                    "--port", str(ctrl_http),
                    "--monitor-ports", json.dumps({"pull": ctrl_pull, "reply": ctrl_reply}),
                ],
                stdout=(out / "lmcache_controller.log").open("w"),
                stderr=subprocess.STDOUT,
                env=ctrl_env,
                start_new_session=True,
            )
            aux_procs.append(p)
            wait_tcp_port("127.0.0.1", ctrl_http, p)
        elif system == "mooncake_rdma":
            mc_rpc = reserve_port()
            mc_metrics = reserve_port()
            aux["mc_rpc_port"] = mc_rpc
            aux["mc_metrics_port"] = mc_metrics
            p = subprocess.Popen(
                [
                    "mooncake_master",
                    "--rpc_port", str(mc_rpc),
                    "--metrics_port", str(mc_metrics),
                    "--enable_http_metadata_server=false",
                    "--default_kv_lease_ttl=500",
                    "--eviction_high_watermark_ratio=0.9",
                    "--eviction_ratio=0.1",
                ],
                stdout=(out / "mooncake_master.log").open("w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            aux_procs.append(p)
            wait_tcp_port("127.0.0.1", mc_rpc, p)

        # 2. Launch Instance 0 (NUMA 0) and Instance 1 (NUMA 1).
        # For vericache_rdma, stagger startup so both OS processes do not simultaneously
        # pin & register 72 GiB of RDMA MRs on mlx5 at the exact same instant.
        proc0 = start_vllm_instance(
            system=system, inst_idx=0, gpu_id=gpu0, peer_ip=ip1,
            http_port=port0, dram_blocks=dram_blocks, out=out, run=run, aux=aux,
        )
        if system == "vericache_rdma":
            wait_server_ready(proc0, ep0, out / "server_inst0.log")
            proc1 = start_vllm_instance(
                system=system, inst_idx=1, gpu_id=gpu1, peer_ip=ip0,
                http_port=port1, dram_blocks=dram_blocks, out=out, run=run, aux=aux,
            )
            wait_server_ready(proc1, ep1, out / "server_inst1.log")
        else:
            proc1 = start_vllm_instance(
                system=system, inst_idx=1, gpu_id=gpu1, peer_ip=ip0,
                http_port=port1, dram_blocks=dram_blocks, out=out, run=run, aux=aux,
            )
            wait_server_ready(proc0, ep0, out / "server_inst0.log")
            wait_server_ready(proc1, ep1, out / "server_inst1.log")

        if system == "vericache_rdma":
            # Wait until all 4 RDMA shards are connected on both instances
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                st = read_vericache_dist_stats(run)
                if (
                    len(st) == 2
                    and st["inst0"]["rdma"]["peers_connected"] >= 4
                    and st["inst1"]["rdma"]["peers_connected"] >= 4
                ):
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(f"RDMA peers failed to connect: {read_vericache_dist_stats(run)}")

        # 3. Start deterministic sticky router
        router_proc = subprocess.Popen(
            [
                sys.executable,
                str(PHASE3 / "vericache-python/bench/phase3/sticky_router.py"),
                "--host", "127.0.0.1",
                "--port", str(router_port),
                "--backends", ep0, ep1,
                "--sticky-pct", str(sticky_pct),
                "--seed", str(seed),
            ],
            stdout=(out / "router.log").open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_tcp_port("127.0.0.1", router_port, router_proc)

        # 4. Prepare warmup & measurement request-sim plans pointing at router_ep
        warmup_argv, warmup_plan = replay_options(
            router_ep, out, trace, scale,
            warmup=True, warmup_end_ms=warmup_end_ms,
            request_timeout_secs=request_timeout_secs,
        )
        measure_argv, measure_plan = replay_options(
            router_ep, out, trace, scale,
            warmup=False, warmup_end_ms=warmup_end_ms,
            request_timeout_secs=request_timeout_secs,
        )

        dump(out / "manifest.json", {
            "stage": "stage2_exp22",
            "system": system,
            "trace": trace,
            "scale": scale,
            "sticky_pct": sticky_pct,
            "seed": seed,
            "gpu0": gpu0, "numa0": numa0, "rnic0": rnic0, "ip0": ip0,
            "gpu1": gpu1, "numa1": numa1, "rnic1": rnic1, "ip1": ip1,
            "hbm_blocks_per_instance": HBM_BLOCKS,
            "dram_gib_per_instance": DRAM_GIB_PER_INSTANCE,
            "dram_blocks_per_instance": dram_blocks,
            "request_sim_bin_sha256": file_sha256(PHASE3 / "bin/request-sim"),
            "warmup_plan": {k: v for k, v in warmup_plan.items() if k != "cohort_indices"},
            "measurement_plan": {k: v for k, v in measure_plan.items() if k != "cohort_indices"},
        })

        # 5. Warmup
        if warmup_end_ms > 0:
            run_replay(warmup_argv, out / "warmup.log", warmup_plan["watchdog_seconds"])

        # Reset router measurement stats (keeps seen_on prefix table from warmup!)
        with urlopen(Request(router_ep + "/router_reset_stats", data=b"{}", method="POST"), timeout=5):
            pass

        # 6. Snapshot BEFORE counters
        time.sleep(0.4)
        hca0_before = read_hca_counters(rnic0)
        hca1_before = read_hca_counters(rnic1)
        vc_before = read_vericache_dist_stats(run)
        scrape(ep0, out, "metrics-inst0-before.prom")
        scrape(ep1, out, "metrics-inst1-before.prom")
        if "mc_metrics_port" in aux:
            try:
                scrape(f"http://127.0.0.1:{aux['mc_metrics_port']}", out, "mooncake-before.prom")
            except Exception:
                pass

        # 7. Run measurement cohort with CPU/RSS sampling
        done, samples = Event(), []
        tracked_pids = lambda: [p.pid for p in (proc0, proc1, *aux_procs) if p is not None and p.poll() is None]
        sampler = Thread(target=sample_dual_cpu, args=(done, samples, tracked_pids), daemon=True)
        sampler.start()
        try:
            run_replay(measure_argv, out / "replay.log", measure_plan["watchdog_seconds"])
        finally:
            done.set()
            sampler.join(timeout=2)

        # 8. Snapshot AFTER counters
        time.sleep(0.5)
        hca0_after = read_hca_counters(rnic0)
        hca1_after = read_hca_counters(rnic1)
        vc_after = read_vericache_dist_stats(run)
        scrape(ep0, out, "metrics-inst0-after.prom")
        scrape(ep1, out, "metrics-inst1-after.prom")
        if "mc_metrics_port" in aux:
            try:
                scrape(f"http://127.0.0.1:{aux['mc_metrics_port']}", out, "mooncake-after.prom")
            except Exception:
                pass
        with urlopen(router_ep + "/router_stats", timeout=5) as r:
            router_stats = json.loads(r.read().decode("utf-8"))
    finally:
        stop_proc(router_proc)
        stop_proc(proc1)
        stop_proc(proc0)
        for p in reversed(aux_procs):
            stop_proc(p)
        for idx in (0, 1):
            shutil.rmtree(Path("/tmp") / f"vc_s2_{os.getpid()}_i{idx}", ignore_errors=True)

    # 9. Analyze & dump summary + server_metrics
    report = analyze(read_rows(out / "requests.jsonl"), plan=measure_plan)

    c0_b, c0_a = metric_totals(out / "metrics-inst0-before.prom"), metric_totals(out / "metrics-inst0-after.prom")
    c1_b, c1_a = metric_totals(out / "metrics-inst1-before.prom"), metric_totals(out / "metrics-inst1-after.prom")
    d0 = {k: v - c0_b.get(k, 0) for k, v in c0_a.items()}
    d1 = {k: v - c1_b.get(k, 0) for k, v in c1_a.items()}

    cpu_seconds = 0.0
    rss_peak = 0
    period = 0.0
    if samples:
        first, last = samples[0], samples[-1]
        period = last["monotonic"] - first["monotonic"]
        cpu_seconds = sum(
            row["cpu_seconds"] - first["processes"].get(pid, {}).get("cpu_seconds", 0)
            for pid, row in last["processes"].items()
        )
        rss_peak = max(sum(proc["rss_bytes"] for proc in row["processes"].values()) for row in samples)

    vc_deltas = {}
    if len(vc_before) == 2 and len(vc_after) == 2:
        for k in ("inst0", "inst1"):
            vc_deltas[k] = {
                m: vc_after[k]["rdma"][m] - vc_before[k]["rdma"][m]
                for m in vc_after[k]["rdma"]
            }

    server_metrics = {
        "system": system,
        "router_stats": router_stats,
        "inst0_counters": d0,
        "inst1_counters": d1,
        "total_hbm_apc_hit_tokens": (d0.get("vllm:prefix_cache_hits_total", 0) + d1.get("vllm:prefix_cache_hits_total", 0)),
        "total_external_prefix_hit_tokens": (
            d0.get("vllm:external_prefix_cache_hits_total", 0)
            + d1.get("vllm:external_prefix_cache_hits_total", 0)
        ),
        "total_preemptions": (d0.get("vllm:num_preemptions_total", 0) + d1.get("vllm:num_preemptions_total", 0)),
        "inst0_log_peaks": parse_server_log_peaks(out / "server_inst0.log"),
        "inst1_log_peaks": parse_server_log_peaks(out / "server_inst1.log"),
        "hca0_delta": hca_delta(hca0_before, hca0_after),
        "hca1_delta": hca_delta(hca1_before, hca1_after),
        "vericache_rdma_deltas": vc_deltas,
        "cluster_host_cpu_percent_of_one_core": 100.0 * cpu_seconds / period if period else None,
        "cluster_peak_rss_bytes": rss_peak,
    }
    dump(out / "summary.json", report)
    dump(out / "server_metrics.json", server_metrics)
    print(json.dumps({"summary": report, "metrics_brief": {
        "hbm_hits": server_metrics["total_hbm_apc_hit_tokens"],
        "ext_hits": server_metrics["total_external_prefix_hit_tokens"],
        "preempts": server_metrics["total_preemptions"],
        "hca0_xmit_gib": server_metrics["hca0_delta"].get("port_xmit_data", 0) / GIB,
        "hca1_xmit_gib": server_metrics["hca1_delta"].get("port_xmit_data", 0) / GIB,
        "router_remote_opp": router_stats.get("theoretical_remote_reuse_reqs"),
    }}), flush=True)


if __name__ == "__main__":
    main()
