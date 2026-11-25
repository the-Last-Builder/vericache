"""Stage 2 — Exp 2.1: Deterministic Cross-Instance Type-2B MW RDMA & Fault/Eviction Safety Gate.

Subtests (all with greedy temperature=0 on real Qwen3-30B-A3B across NUMA 0 <-> NUMA 1):
1. Forced A->B Hit: warm prompts on Instance 0, query Instance 1, verify 100% bit-exact
   outputs, RemoteRdmaHost hits, and real HCA byte counter increases on both RNICs.
2. Forced Concurrent Eviction & Type-2B MW Revocation: shrink Instance 0 DRAM while
   Instance 1 pulls concurrently over RDMA (both cooperative-first and unilateral fallback
   spin budgets), asserting 0 silent corruptions and logging exact revocation metrics.
3. Peer Disconnect Fault Tolerance & Automatic Reconnect: SIGKILL Instance 0 mid-stream,
   assert Instance 1 never deadlocks, GlobalDirectory revokes stale entries, local fallback
   stays 100% bit-exact, then restart Instance 0 and verify RDMA resumes cleanly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from stage0_correctness import MODEL, PHASE3
from stage0_layout import model_command, reserve_port

GIB = 1 << 30
BLOCK_TOKENS = 64
LAYERS = 48
FRAME_BYTES = 128 << 10
KV_BLOCK_BYTES = LAYERS * FRAME_BYTES
HBM_BLOCKS = 3328

RNIC_IP = {
    "mlx5_1": "26.114.8.146",
    "mlx5_2": "26.114.8.154",
    "mlx5_3": "26.114.10.146",
    "mlx5_4": "26.114.10.154",
}

GPU_TOPOLOGY = {
    0: (0, "mlx5_1", "26.114.8.146"),
    1: (0, "mlx5_1", "26.114.8.146"),
    2: (0, "mlx5_2", "26.114.8.154"),
    3: (0, "mlx5_2", "26.114.8.154"),
    4: (1, "mlx5_3", "26.114.10.146"),
    5: (1, "mlx5_3", "26.114.10.146"),
    6: (1, "mlx5_4", "26.114.10.154"),
    7: (1, "mlx5_4", "26.114.10.154"),
}


def resolve_endpoint(gpu_id: int, slot: int) -> tuple[int, str, str]:
    numa, default_rnic, default_ip = GPU_TOPOLOGY[gpu_id]
    override = os.environ.get(f"STAGE2_RNIC{slot}")
    if override:
        return numa, override, RNIC_IP[override]
    return numa, default_rnic, default_ip


def reserve_port_block(count: int = 4) -> int:
    import random

    for _ in range(200):
        base = random.randint(30000, 55000)
        socks = []
        ok = True
        for offset in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", base + offset))
                socks.append(s)
            except OSError:
                s.close()
                ok = False
                break
        for s in socks:
            s.close()
        if ok:
            return base
    raise RuntimeError(f"Could not reserve {count} consecutive ports")


def read_hca_counters(dev: str) -> dict[str, int]:
    base = Path(f"/sys/class/infiniband/{dev}/ports/1")
    out = {}
    for name in ("port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"):
        p = base / "counters" / name
        if p.is_file():
            val = int(p.read_text().strip())
            out[name] = val * 4 if name.endswith("_data") else val
    for name in ("rx_read_requests", "rx_write_requests", "req_remote_access_errors", "resp_remote_access_errors"):
        p = base / "hw_counters" / name
        if p.is_file():
            out[name] = int(p.read_text().strip())
    return out


def hca_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after.get(k, 0) - before.get(k, 0) for k in before}


def ctrl_rpc(sock_path: Path, payload: dict, timeout: float = 10.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode("utf-8"))


def query_completion(endpoint: str, prompt: str, max_tokens: int = 24, timeout: float = 120.0) -> dict:
    body = json.dumps({
        "model": "Qwen3-30B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = Request(
        endpoint + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_prompts(tag: str, count: int, approx_words: int = 480) -> list[str]:
    prompts = []
    for i in range(count):
        filler = " ".join(
            f"section_{tag}_{i}_clause_{j}_metric_{(i * 37 + j * 13) % 997}"
            for j in range(approx_words // 2)
        )
        prompts.append(
            f"System analysis context [{tag}-{i}]: {filler}. "
            f"Summarize the exact identifier {tag}-{i} and list three deterministic observations."
        )
    return prompts


def wait_server_ready(proc: subprocess.Popen, endpoint: str, log_path: Path, timeout_s: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-3000:] if log_path.is_file() else ""
            raise RuntimeError(f"vLLM exited early rc={proc.returncode} ({log_path}):\n{tail}")
        try:
            with urlopen(endpoint + "/health", timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {endpoint} ({log_path})")


def stop_proc(proc: subprocess.Popen | None, *, hard: bool = False) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL if hard else signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5 if hard else 30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def launch_instance(
    *,
    node_id: int,
    gpu_id: int,
    peer_node_id: int,
    peer_ip: str,
    rdma_base_port: int,
    is_listener: bool,
    gdir_endpoint: str,
    http_port: int,
    dram_blocks: int,
    num_shards: int,
    ctrl_sock: Path,
    stats_dir: Path,
    log_path: Path,
    peer_fetch_enabled: bool = True,
) -> subprocess.Popen:
    numa, rnic, _local_ip = resolve_endpoint(gpu_id, node_id)
    lib_path = str(PHASE3 / "vericache-python/target/phase3/libvericache_python.so")
    frames_per_shard = (dram_blocks * LAYERS) // num_shards
    cfg = {
        "distributed_mode": True,
        "node_id": node_id,
        "instance_id": node_id,
        "num_shards": num_shards,
        "host_frames_per_shard": frames_per_shard,
        "nvme_frames_per_shard": 0,
        "frame_size": FRAME_BYTES,
        "prefer_cuda_pinned": True,
        "lib_path": lib_path,
        "io_workers": 4,
        "staging_pool_size": 16,
        "rdma_device": rnic,
        "rdma_gid_index": 3,
        "rdma_windows_per_peer": 16,
        "peer_fetch_enabled": peer_fetch_enabled,
        "global_directory_endpoint": gdir_endpoint,
        "peers": [
            {
                "peer_node_id": peer_node_id,
                "tcp_host": peer_ip,
                "base_port": rdma_base_port,
                "is_listener": is_listener,
                "psn_hint": 0x1000,
            }
        ],
    }
    transfer = {
        "kv_role": "kv_both",
        "kv_connector": "Stage2VeriCacheV1Connector",
        "kv_connector_module_path": "stage2_connector",
        "kv_connector_extra_config": cfg,
    }
    argv = model_command(MODEL, BLOCK_TOKENS, http_port, "vanilla", tensor_parallel_size=1, max_num_seqs=16)
    argv.extend(["--num-gpu-blocks-override", str(HBM_BLOCKS)])
    argv.extend(["--attention-backend", "FLASH_ATTN"])
    argv.extend(["--kv-transfer-config", json.dumps(transfer)])

    full_argv = [
        "numactl",
        f"--cpunodebind={numa}",
        f"--membind={numa}",
        *argv,
    ]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["PHASE3_STAGE2_CTRL_SOCK"] = str(ctrl_sock)
    env["PHASE3_STAGE2_STATS_DIR"] = str(stats_dir)
    env["PYTHONPATH"] = (
        f"{PHASE3 / 'vericache-python/bench/phase3'}:"
        f"{PHASE3 / 'vericache-python/python'}:"
        + env.get("PYTHONPATH", "")
    )
    log_file = log_path.open("w")
    return subprocess.Popen(
        full_argv,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


def wait_peers_connected(ctrl0: Path, ctrl1: Path, expected_shards: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            s0 = ctrl_rpc(ctrl0, {"op": "stats"})["stats"]
            s1 = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]
            if s0["peers_connected"] >= expected_shards and s1["peers_connected"] >= expected_shards:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("Timed out waiting for cross-NUMA RDMA peers to connect on all shards")


def main() -> None:
    gpu0 = int(os.environ["STAGE2_GPU0"])
    gpu1 = int(os.environ["STAGE2_GPU1"])
    out = Path(os.environ["PHASE3_RESULT_DIR"])
    run = Path(os.environ["PHASE3_TEST_DIR"])
    out.mkdir(parents=True, exist_ok=True)

    numa0, rnic0, ip0 = resolve_endpoint(gpu0, 0)
    numa1, rnic1, ip1 = resolve_endpoint(gpu1, 1)
    assert numa0 == 0 and numa1 == 1, f"Expected cross-NUMA GPUs, got {gpu0} (NUMA {numa0}) and {gpu1} (NUMA {numa1})"

    num_shards = int(os.environ.get("STAGE2_NUM_SHARDS", "4"))
    # 8 GiB DRAM per instance is plenty for Exp 2.1 safety gate & fast startup (~1s MR reg)
    dram_blocks = (8 * GIB) // KV_BLOCK_BYTES
    gdir_port = reserve_port()
    gdir_endpoint = f"tcp://127.0.0.1:{gdir_port}"
    rdma_base_port = reserve_port_block(num_shards)
    port0 = reserve_port()
    port1 = reserve_port()
    ep0 = f"http://127.0.0.1:{port0}"
    ep1 = f"http://127.0.0.1:{port1}"

    ctrl0 = run / "ctrl-inst0.sock"
    ctrl1 = run / "ctrl-inst1.sock"

    gdir_proc = None
    proc0 = None
    proc1 = None

    report: dict = {
        "gpu0": gpu0, "numa0": numa0, "rnic0": rnic0, "ip0": ip0,
        "gpu1": gpu1, "numa1": numa1, "rnic1": rnic1, "ip1": ip1,
        "num_shards": num_shards,
        "dram_blocks_per_instance": dram_blocks,
        "subtests": {},
    }

    try:
        # Start decoupled Global Directory Server
        gdir_log = (out / "global_directory.log").open("w")
        gdir_env = os.environ.copy()
        gdir_env["PYTHONPATH"] = f"{PHASE3 / 'vericache-python/python'}:" + gdir_env.get("PYTHONPATH", "")
        gdir_proc = subprocess.Popen(
            [sys.executable, "-m", "vericache.global_directory", "--endpoint", gdir_endpoint],
            stdout=gdir_log,
            stderr=subprocess.STDOUT,
            env=gdir_env,
            start_new_session=True,
        )
        time.sleep(0.5)

        # Start both vLLM instances concurrently
        print(f"[Exp 2.1] Launching Inst0 (GPU {gpu0}/{rnic0}) and Inst1 (GPU {gpu1}/{rnic1})...", flush=True)
        proc0 = launch_instance(
            node_id=0, gpu_id=gpu0, peer_node_id=1, peer_ip=ip0,
            rdma_base_port=rdma_base_port, is_listener=True,
            gdir_endpoint=gdir_endpoint, http_port=port0,
            dram_blocks=dram_blocks, num_shards=num_shards,
            ctrl_sock=ctrl0, stats_dir=run, log_path=out / "inst0.log",
        )
        proc1 = launch_instance(
            node_id=1, gpu_id=gpu1, peer_node_id=0, peer_ip=ip0,
            rdma_base_port=rdma_base_port, is_listener=False,
            gdir_endpoint=gdir_endpoint, http_port=port1,
            dram_blocks=dram_blocks, num_shards=num_shards,
            ctrl_sock=ctrl1, stats_dir=run, log_path=out / "inst1.log",
        )
        wait_server_ready(proc0, ep0, out / "inst0.log")
        wait_server_ready(proc1, ep1, out / "inst1.log")
        wait_peers_connected(ctrl0, ctrl1, num_shards)
        print("[Exp 2.1] Both instances READY and Type-2B MW RDMA connected on all shards!", flush=True)

        # ------------------------------------------------------------------
        # Subtest 2.1.1: Forced A -> B Cross-Instance RDMA Hit & Bit-Exactness
        # ------------------------------------------------------------------
        print("[Exp 2.1.1] Running Forced A->B Hit & Bit-Exactness...", flush=True)
        prompts_a = make_prompts("ab", 6, approx_words=520)
        hca0_before = read_hca_counters(rnic0)
        hca1_before = read_hca_counters(rnic1)
        st0_before = ctrl_rpc(ctrl0, {"op": "stats"})["stats"]
        st1_before = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]

        golden_a = []
        for p in prompts_a:
            r = query_completion(ep0, p, max_tokens=24)
            golden_a.append(r["choices"][0]["message"]["content"])

        from vericache.global_directory import GlobalDirectoryClient

        with GlobalDirectoryClient(gdir_endpoint) as gclient:
            gdir_after_warm = gclient.stats()

        inst1_a = []
        cached_tokens_b = []
        for p in prompts_a:
            r = query_completion(ep1, p, max_tokens=24)
            inst1_a.append(r["choices"][0]["message"]["content"])
            details = (r.get("usage") or {}).get("prompt_tokens_details") or {}
            cached_tokens_b.append(int(details.get("cached_tokens", 0)))

        hca0_after = read_hca_counters(rnic0)
        hca1_after = read_hca_counters(rnic1)
        st0_after = ctrl_rpc(ctrl0, {"op": "stats"})["stats"]
        st1_after = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]

        mismatches_a = sum(1 for a, b in zip(golden_a, inst1_a) if a != b)
        remote_hits_delta = st1_after["remote_hits"] - st1_before["remote_hits"]
        rdma_bytes_delta = st1_after["rdma_fetch_bytes"] - st1_before["rdma_fetch_bytes"]
        grants_served_delta = st0_after["rdma_grants_served"] - st0_before["rdma_grants_served"]
        hca0_d = hca_delta(hca0_before, hca0_after)
        hca1_d = hca_delta(hca1_before, hca1_after)

        sub1 = {
            "prompts": len(prompts_a),
            "bit_exact_matches": len(prompts_a) - mismatches_a,
            "mismatches": mismatches_a,
            "inst1_cached_tokens": cached_tokens_b,
            "inst1_remote_hits": remote_hits_delta,
            "inst1_rdma_fetch_bytes": rdma_bytes_delta,
            "inst0_rdma_grants_served": grants_served_delta,
            "hca0_delta": hca0_d,
            "hca1_delta": hca1_d,
            "gdir_after_warm": gdir_after_warm,
            "passed": (
                mismatches_a == 0
                and remote_hits_delta > 0
                and rdma_bytes_delta > 0
                and hca0_d.get("port_xmit_data", 0) >= rdma_bytes_delta // 2
                and hca1_d.get("port_rcv_data", 0) >= rdma_bytes_delta // 2
            ),
        }
        report["subtests"]["2.1.1_forced_ab_hit"] = sub1
        print("[Exp 2.1.1] Result:", json.dumps(sub1), flush=True)
        assert sub1["passed"], f"Exp 2.1.1 failed: {sub1}"

        # ------------------------------------------------------------------
        # Subtest 2.1.2: Forced Concurrent Eviction & Type-2B MW Revocation
        # ------------------------------------------------------------------
        print("[Exp 2.1.2] Running Forced Concurrent Eviction & MW Revocation...", flush=True)
        prompts_ev = make_prompts("evict", 16, approx_words=520)
        golden_ev = [query_completion(ep0, p, max_tokens=20)["choices"][0]["message"]["content"] for p in prompts_ev]

        st0_ev_b = ctrl_rpc(ctrl0, {"op": "stats"})["stats"]
        st1_ev_b = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]

        # Phase A: Cooperative-first (spin_budget=512), Phase B: Unilateral fallback (spin_budget=0)
        ev_mismatches = 0
        ev_errors = 0
        ev_total = 0

        for phase_name, spin_budget, target_frames in (
            ("cooperative", 512, 64),
            ("unilateral", 0, 32),
        ):
            # Re-warm on Inst0 at full budget first
            ctrl_rpc(ctrl0, {"op": "set_dram_budget", "frames_per_shard": (dram_blocks * LAYERS) // num_shards})
            ctrl_rpc(ctrl0, {"op": "set_spin_budget", "spins": spin_budget})
            for p in prompts_ev:
                query_completion(ep0, p, max_tokens=8)

            # Simultaneously hammer Inst1 with RDMA pulls while shrinking Inst0 DRAM & writing churn
            churn_prompts = make_prompts(f"churn_{phase_name}", 12, approx_words=520)

            def _pull_worker(idx: int):
                r = query_completion(ep1, prompts_ev[idx], max_tokens=20)
                return idx, r["choices"][0]["message"]["content"]

            def _evict_driver():
                time.sleep(0.05)
                ctrl_rpc(ctrl0, {"op": "set_dram_budget", "frames_per_shard": target_frames})
                for cp in churn_prompts[:6]:
                    query_completion(ep0, cp, max_tokens=4)

            with ThreadPoolExecutor(max_workers=5) as ex:
                ev_fut = ex.submit(_evict_driver)
                pull_futs = [ex.submit(_pull_worker, i) for i in range(len(prompts_ev))]
                ev_fut.result()
                for fut in pull_futs:
                    ev_total += 1
                    try:
                        idx, text = fut.result()
                        if text != golden_ev[idx]:
                            ev_mismatches += 1
                    except Exception:
                        ev_errors += 1

        # Restore Inst0 DRAM budget and default spin budget
        ctrl_rpc(ctrl0, {"op": "set_dram_budget", "frames_per_shard": (dram_blocks * LAYERS) // num_shards})
        ctrl_rpc(ctrl0, {"op": "set_spin_budget", "spins": 512})

        st0_ev_a = ctrl_rpc(ctrl0, {"op": "stats"})["stats"]
        st1_ev_a = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]
        coop_delta = st0_ev_a["mw_revocations_cooperative"] - st0_ev_b["mw_revocations_cooperative"]
        uni_delta = st0_ev_a["mw_revocations_unilateral"] - st0_ev_b["mw_revocations_unilateral"]
        dram_ev_delta = st0_ev_a["dram_evictions"] - st0_ev_b["dram_evictions"]

        sub2 = {
            "total_concurrent_pulls": ev_total,
            "mismatches_silent_corruptions": ev_mismatches,
            "request_errors": ev_errors,
            "inst0_dram_evictions": dram_ev_delta,
            "inst0_mw_revocations_cooperative": coop_delta,
            "inst0_mw_revocations_unilateral": uni_delta,
            "inst0_mw_revocation_latency_us_max": st0_ev_a["mw_revocation_latency_us_max"],
            "inst1_rdma_fetches_ok": st1_ev_a["rdma_fetches_ok"] - st1_ev_b["rdma_fetches_ok"],
            "inst1_rdma_fetches_miss": st1_ev_a["rdma_fetches_miss"] - st1_ev_b["rdma_fetches_miss"],
            "passed": (ev_mismatches == 0 and ev_errors == 0 and dram_ev_delta > 0),
        }
        report["subtests"]["2.1.2_concurrent_eviction_mw_revoke"] = sub2
        print("[Exp 2.1.2] Result:", json.dumps(sub2), flush=True)
        assert sub2["passed"], f"Exp 2.1.2 failed: {sub2}"

        # ------------------------------------------------------------------
        # Subtest 2.1.3: Peer Disconnect Fault Tolerance & Automatic Reconnect
        # ------------------------------------------------------------------
        print("[Exp 2.1.3] Running Peer Disconnect (SIGKILL) & Auto-Reconnect...", flush=True)
        prompts_ft = make_prompts("fault", 6, approx_words=520)
        golden_ft = [query_completion(ep0, p, max_tokens=20)["choices"][0]["message"]["content"] for p in prompts_ft]

        st1_ft_b = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]
        with GlobalDirectoryClient(gdir_endpoint) as gclient:
            gdir_ft_b = gclient.stats()

        # Abruptly SIGKILL Instance 0 while Instance 1 starts pulling prompts_ft!
        def _kill_inst0():
            time.sleep(0.05)
            stop_proc(proc0, hard=True)

        ft_mismatches = 0
        ft_errors = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            kill_fut = ex.submit(_kill_inst0)
            futs = [ex.submit(query_completion, ep1, p, 20) for p in prompts_ft]
            kill_fut.result()
            for idx, f in enumerate(futs):
                try:
                    out_txt = f.result()["choices"][0]["message"]["content"]
                    if out_txt != golden_ft[idx]:
                        ft_mismatches += 1
                except Exception:
                    ft_errors += 1
        proc0 = None

        # Verify disconnect detected & GlobalDirectory revoked Inst0
        time.sleep(0.5)
        ctrl_rpc(ctrl1, {"op": "poll"})
        st1_after_kill = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]
        with GlobalDirectoryClient(gdir_endpoint) as gclient:
            gdir_after_kill = gclient.stats()

        # Restart Instance 0 and verify automatic RDMA reconnect + resumed remote hits!
        print("[Exp 2.1.3] Restarting Inst0 and verifying automatic RDMA reconnect...", flush=True)
        if ctrl0.exists():
            ctrl0.unlink()
        proc0 = launch_instance(
            node_id=0, gpu_id=gpu0, peer_node_id=1, peer_ip=ip0,
            rdma_base_port=rdma_base_port, is_listener=True,
            gdir_endpoint=gdir_endpoint, http_port=port0,
            dram_blocks=dram_blocks, num_shards=num_shards,
            ctrl_sock=ctrl0, stats_dir=run, log_path=out / "inst0_restarted.log",
        )
        wait_server_ready(proc0, ep0, out / "inst0_restarted.log")
        wait_peers_connected(ctrl0, ctrl1, num_shards, timeout_s=30.0)

        prompts_post = make_prompts("post_reconnect", 4, approx_words=520)
        golden_post = [query_completion(ep0, p, max_tokens=20)["choices"][0]["message"]["content"] for p in prompts_post]
        st1_pre_post = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]
        post_mismatches = 0
        for idx, p in enumerate(prompts_post):
            r = query_completion(ep1, p, max_tokens=20)
            if r["choices"][0]["message"]["content"] != golden_post[idx]:
                post_mismatches += 1
        st1_post = ctrl_rpc(ctrl1, {"op": "stats"})["stats"]

        sub3 = {
            "mid_kill_mismatches": ft_mismatches,
            "mid_kill_errors": ft_errors,
            "inst1_peer_disconnects_detected": st1_after_kill["peer_disconnects"] - st1_ft_b["peer_disconnects"],
            "inst1_peers_connected_while_dead": st1_after_kill["peers_connected"],
            "gdir_disconnect_revocations": (
                gdir_after_kill["revocations_disconnect"] - gdir_ft_b["revocations_disconnect"]
            ),
            "gdir_inst0_present_while_dead": "0" in gdir_after_kill["instances"],
            "inst1_peer_reconnects": st1_post["peer_reconnects"] - st1_ft_b["peer_reconnects"],
            "inst1_peers_connected_after_restart": st1_post["peers_connected"],
            "post_reconnect_remote_hits": st1_post["remote_hits"] - st1_pre_post["remote_hits"],
            "post_reconnect_mismatches": post_mismatches,
            "passed": (
                ft_mismatches == 0
                and ft_errors == 0
                and (st1_after_kill["peer_disconnects"] - st1_ft_b["peer_disconnects"]) >= num_shards
                and "0" not in gdir_after_kill["instances"]
                and st1_post["peers_connected"] >= num_shards
                and (st1_post["remote_hits"] - st1_pre_post["remote_hits"]) > 0
                and post_mismatches == 0
            ),
        }
        report["subtests"]["2.1.3_peer_disconnect_and_reconnect"] = sub3
        print("[Exp 2.1.3] Result:", json.dumps(sub3), flush=True)
        assert sub3["passed"], f"Exp 2.1.3 failed: {sub3}"

        report["all_passed"] = True
        (out / "exp21_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("[Exp 2.1] ALL 3 SAFETY GATE SUBTESTS PASSED!", flush=True)
    finally:
        stop_proc(proc1)
        stop_proc(proc0)
        stop_proc(gdir_proc)
        (out / "exp21_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
