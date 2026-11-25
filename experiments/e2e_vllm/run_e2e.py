#!/usr/bin/env python3
"""2-Node Dual-NUMA Peer-Shared (Morphology I) End-to-End vLLM Trace Replay Orchestrator.

Runs apples-to-apples comparisons across:
  - `baseline`  : 2x vLLM 0.34.0 (pure GPU prefix caching, no external offload)
  - `vericache` : 2x vLLM 0.34.0 + VeriCacheConnector (HBM + Pinned DRAM + NVMe io_uring + 2-GPU GDR)
  - `lmcache`   : 2x vLLM 0.34.0 + LMCacheConnectorV1 (LocalCPU + LocalDisk O_DIRECT + P2P NIXL/UCX)
  - `mooncake`  : 2x vLLM 0.34.0 + LMCacheConnectorV1 + MooncakeDistributedStore (RDMA + LocalDisk O_DIRECT)
  - `mooncake_native` : 2x vLLM 0.34.0 + native MooncakeStoreConnector (RDMA + LocalDisk)

Example 2-node dual-NUMA hardware mapping inside `image/dev.sh bench-hw`:
  - Node 0: container GPU 0, NUMA 0, RDMA device (e.g., mlx5_0), local NVMe scratch dir
  - Node 1: container GPU 1, NUMA 1, RDMA device (e.g., mlx5_1), local NVMe scratch dir
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import psutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
E2E_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "experiments" / "results"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_tcp(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for TCP {host}:{port}")


def _wait_vllm_health(
    url: str,
    proc: subprocess.Popen,
    log_path: pathlib.Path,
    timeout_s: float = 240.0,
) -> None:
    deadline = time.perf_counter() + timeout_s
    health_url = f"{url.rstrip('/')}/health"
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            tail = _tail_file(log_path, 60)
            raise RuntimeError(
                f"vLLM server exited early (rc={proc.returncode}) for {url}.\n"
                f"--- {log_path} (tail) ---\n{tail}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    tail = _tail_file(log_path, 60)
    raise RuntimeError(f"Timed out waiting for {health_url}.\n--- {log_path} ---\n{tail}")


def _tail_file(path: pathlib.Path, n: int = 40) -> str:
    if not path.exists():
        return "<no log>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _kill_proc_tree(proc: subprocess.Popen | None, grace_s: float = 8.0) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.perf_counter() + grace_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def _sample_tree_cpu_rss(pids: list[int]) -> tuple[float, float]:
    """Return (total_cpu_seconds, total_rss_mb) across process trees."""
    seen: set[int] = set()
    cpu_s = 0.0
    rss_b = 0
    for pid in pids:
        try:
            p = psutil.Process(pid)
            procs = [p] + p.children(recursive=True)
        except psutil.Error:
            continue
        for c in procs:
            if c.pid in seen:
                continue
            seen.add(c.pid)
            try:
                ct = c.cpu_times()
                cpu_s += ct.user + ct.system
                rss_b += c.memory_info().rss
            except psutil.Error:
                pass
    return cpu_s, rss_b / (1024.0 * 1024.0)


def _clean_dir(path: pathlib.Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _numa_cmd(numa_node: int) -> list[str]:
    if shutil.which("numactl") is None:
        return []
    probe = subprocess.run(
        ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}"]
    return []


def _bytes_per_vllm_block(model_path: str, block_size: int) -> int:
    cfg = json.loads((pathlib.Path(model_path) / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    num_layers = int(tc["num_hidden_layers"])
    num_kv_heads = int(tc.get("num_key_value_heads", tc["num_attention_heads"]))
    head_dim = int(tc.get("head_dim", int(tc["hidden_size"]) // int(tc["num_attention_heads"])))
    # bfloat16 = 2 bytes, K + V = 2 tensors
    raw = num_layers * 2 * block_size * num_kv_heads * head_dim * 2
    return ((raw + 4095) // 4096) * 4096


def _write_yaml(path: pathlib.Path, obj: dict[str, Any]) -> None:
    import yaml  # PyYAML is installed in bench-venv

    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


class ClusterContext:
    """Manages control-plane + 2 NUMA-pinned vLLM instances for one system."""

    def __init__(self, system: str, args: argparse.Namespace, work_dir: pathlib.Path) -> None:
        self.system = system
        self.args = args
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.port0 = _find_free_port()
        self.port1 = _find_free_port()
        self.url0 = f"http://127.0.0.1:{self.port0}"
        self.url1 = f"http://127.0.0.1:{self.port1}"

        self.aux_procs: list[subprocess.Popen] = []
        self.vllm_procs: list[subprocess.Popen] = []
        self.log_files: list[Any] = []

        self.nvme_n0 = pathlib.Path(args.nvme_dir_node0) / f"{system}_n0"
        self.nvme_n1 = pathlib.Path(args.nvme_dir_node1) / f"{system}_n1"

    def _spawn(
        self,
        cmd: list[str],
        env: dict[str, str],
        log_name: str,
        *,
        is_vllm: bool = False,
    ) -> tuple[subprocess.Popen, pathlib.Path]:
        lp = self.work_dir / log_name
        lf = lp.open("w", encoding="utf-8")
        self.log_files.append(lf)
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if is_vllm:
            self.vllm_procs.append(proc)
        else:
            self.aux_procs.append(proc)
        return proc, lp

    def start(self) -> None:
        args = self.args
        _clean_dir(self.nvme_n0)
        _clean_dir(self.nvme_n1)

        base_env = os.environ.copy()
        py_paths = [str(ROOT / "vericache-python" / "python"), str(E2E_DIR)]
        if base_env.get("PYTHONPATH"):
            py_paths.append(base_env["PYTHONPATH"])
        base_env["PYTHONPATH"] = ":".join(py_paths)
        # Ensure libvericache_python.so is discoverable
        lib_so = ROOT / "vericache-python" / "target" / "release" / "libvericache_python.so"
        if lib_so.is_file():
            base_env["VERICACHE_PYTHON_LIB"] = str(lib_so)
        # Critical for distributed prefix hashing consistency across Node 0 & Node 1
        # (LMCache's token_database falls back to Python builtin hash when vLLM uses xxhash).
        base_env["PYTHONHASHSEED"] = "0"
        base_env["VLLM_USAGE_STATS_DISABLED"] = "1"
        base_env["DO_NOT_TRACK"] = "1"
        # Avoid huggingface network probes for local model directories
        base_env["HF_HUB_OFFLINE"] = "1"

        frame_bytes = _bytes_per_vllm_block(args.model_path, args.block_size)
        dram_bytes = int(round(args.dram_gb_per_node * (1024**3)))
        nvme_bytes = int(round(args.nvme_gb_per_node * (1024**3)))
        gpu_kv_bytes = int(round(args.gpu_kv_gb_per_node * (1024**3)))

        # Per-system control plane & kv_transfer_config setup
        node_extra_envs: list[dict[str, str]] = [{}, {}]
        kv_configs: list[dict[str, Any] | None] = [None, None]
        node_gpu_kv_bytes = [gpu_kv_bytes, gpu_kv_bytes]

        if self.system == "baseline":
            pass

        elif self.system == "vericache":
            hbm_pool_bytes = int(round(args.vericache_hbm_gb_per_node * (1024**3)))
            # Apples-to-apples VRAM budget: subtract VeriCache's HBM pool from vLLM's KV cache!
            vc_vllm_kv_bytes = max(512 * 1024 * 1024, gpu_kv_bytes - hbm_pool_bytes)
            node_gpu_kv_bytes = [vc_vllm_kv_bytes, vc_vllm_kv_bytes]

            hbm_frames = max(32, hbm_pool_bytes // frame_bytes)
            dram_frames = max(64, dram_bytes // frame_bytes)
            nvme_frames = max(128, nvme_bytes // frame_bytes) if nvme_bytes > 0 else 0
            # Exclusive DRAM + NVMe capacity (when NVMe is enabled)
            max_cached_blocks = (
                dram_frames + nvme_frames if nvme_frames > 0 else dram_frames + hbm_frames
            )

            dir_port = _find_free_port()
            rdma_port = _find_free_port()
            # With peer NVMe on-demand warm-and-serve enabled in VeriCache,
            # directory tracks full local (DRAM + NVMe) capacity per node.
            self._spawn(
                [
                    sys.executable,
                    "-m",
                    "vericache.directory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(dir_port),
                    "--max-keys-per-node",
                    str(max(1024, max_cached_blocks)),
                ],
                base_env,
                "vericache_directory.log",
            )
            _wait_tcp("127.0.0.1", dir_port, timeout_s=15.0)

            rdma_devs = [args.rdma_dev_node0, args.rdma_dev_node1]
            nvme_dirs = [str(self.nvme_n0), str(self.nvme_n1)]
            for nid in (0, 1):
                peer_id = 1 - nid
                kv_configs[nid] = {
                    "kv_connector": "VeriCacheConnector",
                    "kv_connector_module_path": "vericache.vllm_connector",
                    "kv_role": "kv_both",
                    "kv_load_failure_policy": "recompute",
                    "kv_connector_extra_config": {
                        "node_id": nid,
                        "hbm_frames": hbm_frames,
                        "dram_frames": dram_frames,
                        "nvme_dir": nvme_dirs[nid] if nvme_frames > 0 else None,
                        "nvme_frames": nvme_frames,
                        "max_cached_blocks": max_cached_blocks,
                        "rdma_device": rdma_devs[nid],
                        "rdma_gid_index": args.rdma_gid_index,
                        "rdma_windows_per_peer": 16,
                        "directory_addr": f"127.0.0.1:{dir_port}",
                        "async_io": True,
                        "lazy_offload": bool(args.vericache_lazy_offload),
                        "store_chunk_size": int(args.vericache_store_chunk_size),
                        "store_direct_dram": bool(args.vericache_store_direct_dram),
                        "peers": [
                            {
                                "peer_node_id": peer_id,
                                "tcp_addr": f"127.0.0.1:{rdma_port}",
                                "is_listener": (nid == 0),
                                "psn_hint": 0x1000,
                            }
                        ],
                    },
                }

        elif self.system == "lmcache":
            ctrl_api_port = _find_free_port()
            ctrl_pull_port = _find_free_port()
            ctrl_reply_port = _find_free_port()
            self._spawn(
                [
                    "lmcache_controller",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ctrl_api_port),
                    "--monitor-ports",
                    json.dumps({"pull": ctrl_pull_port, "reply": ctrl_reply_port}),
                ],
                base_env,
                "lmcache_controller.log",
            )
            _wait_tcp("127.0.0.1", ctrl_api_port, timeout_s=20.0)

            p2p_hosts = [args.rdma_ip_node0, args.rdma_ip_node1]
            rdma_devs = [args.rdma_dev_node0, args.rdma_dev_node1]
            nvme_dirs = [str(self.nvme_n0), str(self.nvme_n1)]
            for nid in (0, 1):
                p2p_init = _find_free_port()
                p2p_lookup = _find_free_port()
                worker_port = _find_free_port()
                cfg_yaml = self.work_dir / f"lmcache_node{nid}.yaml"
                cfg_obj: dict[str, Any] = {
                    "chunk_size": args.lmcache_chunk_size,
                    "local_cpu": True,
                    "max_local_cpu_size": float(args.dram_gb_per_node),
                    "local_disk": f"file://{nvme_dirs[nid]}/" if args.nvme_gb_per_node > 0 else None,
                    "max_local_disk_size": float(args.nvme_gb_per_node),
                    "enable_controller": True,
                    "lmcache_instance_id": f"lmcache_node{nid}",
                    "controller_pull_url": f"127.0.0.1:{ctrl_pull_port}",
                    "controller_reply_url": f"127.0.0.1:{ctrl_reply_port}",
                    "lmcache_worker_ports": worker_port,
                    "enable_p2p": bool(args.lmcache_p2p),
                    "enable_async_loading": bool(args.lmcache_p2p),
                    "extra_config": {
                        "use_odirect": True,
                        "disk_io_threads": 4,
                        "lookup_backoff_time": 0.001,
                    },
                }
                if args.lmcache_p2p:
                    cfg_obj["p2p_host"] = p2p_hosts[nid]
                    cfg_obj["p2p_init_ports"] = [p2p_init]
                    cfg_obj["p2p_lookup_ports"] = [p2p_lookup]
                    cfg_obj["transfer_channel"] = "nixl"
                    cfg_obj["nixl_buffer_device"] = "cpu"
                _write_yaml(cfg_yaml, cfg_obj)
                node_extra_envs[nid] = {
                    "LMCACHE_CONFIG_FILE": str(cfg_yaml),
                    "LMCACHE_USE_EXPERIMENTAL": "True",
                    "UCX_NET_DEVICES": f"{rdma_devs[nid]}:1",
                    "UCX_IB_GID_INDEX": str(args.rdma_gid_index),
                    "UCX_IB_REG_MT_THRESH": "inf",
                }
                kv_configs[nid] = {
                    "kv_connector": "LMCacheConnectorV1Compat",
                    "kv_connector_module_path": "lmcache_compat",
                    "kv_role": "kv_both",
                }

        elif self.system == "mooncake":
            mc_master_port = _find_free_port()
            mc_metrics_port = _find_free_port()
            self._spawn(
                [
                    "mooncake_master",
                    "--rpc_port",
                    str(mc_master_port),
                    "--metrics_port",
                    str(mc_metrics_port),
                    "--enable_http_metadata_server=false",
                    "--default_kv_lease_ttl=5000",
                ],
                base_env,
                "mooncake_master.log",
            )
            _wait_tcp("127.0.0.1", mc_master_port, timeout_s=20.0)

            rdma_devs = [args.rdma_dev_node0, args.rdma_dev_node1]
            rdma_ips = [args.rdma_ip_node0, args.rdma_ip_node1]
            nvme_dirs = [str(self.nvme_n0), str(self.nvme_n1)]
            # Align global_segment_size to 16 MiB multiple.
            # For large DRAM budgets (>=16 GiB, e.g. 128 GiB with 32K contexts where
            # a single 32K request is ~1.74 GiB KV), size Mooncake local RDMA buffer
            # (4 GiB) + LMCache CPU staging pool (8 GiB) so concurrent 32K retrieves
            # never starve buffers, while deducting both from global_segment_size so
            # total per-node DRAM stays strictly equal to `dram_gb_per_node`.
            align = 16 * 1024 * 1024
            if dram_bytes >= 16 * (1024**3):
                staging_gb = 8.0
                local_buf_bytes = 4 * (1024**3)
                raw_seg = max(4 * (1024**3), dram_bytes - local_buf_bytes - int(staging_gb * (1024**3)))
                seg_bytes = ((raw_seg + align - 1) // align) * align
            else:
                staging_gb = 4.0
                seg_bytes = max(64 * 1024 * 1024, ((dram_bytes + align - 1) // align) * align)
                local_buf_bytes = min(1024 * 1024 * 1024, max(64 * 1024 * 1024, seg_bytes // 4))

            for nid in (0, 1):
                cfg_yaml = self.work_dir / f"mooncake_node{nid}.yaml"
                cfg_obj = {
                    "chunk_size": args.lmcache_chunk_size,
                    "local_cpu": False,
                    "max_local_cpu_size": staging_gb,
                    "local_disk": f"file://{nvme_dirs[nid]}/" if args.nvme_gb_per_node > 0 else None,
                    "max_local_disk_size": float(args.nvme_gb_per_node),
                    "remote_url": f"mooncakestore://127.0.0.1:{mc_master_port}",
                    "remote_serde": "naive",
                    "extra_config": {
                        "use_odirect": True,
                        "disk_io_threads": 4,
                        "mooncake_local_hostname": rdma_ips[nid],
                        "mooncake_metadata_server": "P2PHANDSHAKE",
                        "mooncake_master_server_addr": f"127.0.0.1:{mc_master_port}",
                        "mooncake_protocol": "rdma",
                        "mooncake_rdma_devices": rdma_devs[nid],
                        "mooncake_global_segment_size": seg_bytes,
                        "mooncake_local_buffer_size": local_buf_bytes,
                        "mooncake_prefer_local_alloc": True,
                        "transfer_timeout": 15,
                    },
                }
                _write_yaml(cfg_yaml, cfg_obj)
                node_extra_envs[nid] = {
                    "LMCACHE_CONFIG_FILE": str(cfg_yaml),
                    "LMCACHE_USE_EXPERIMENTAL": "True",
                    "MC_GID_INDEX": str(args.rdma_gid_index),
                }
                kv_configs[nid] = {
                    "kv_connector": "LMCacheConnectorV1Compat",
                    "kv_connector_module_path": "lmcache_compat",
                    "kv_role": "kv_both",
                    "kv_load_failure_policy": "recompute",
                }
        elif self.system == "mooncake_native":
            import re as _re

            mc_master_port = _find_free_port()
            mc_metrics_port = _find_free_port()
            has_nvme = args.nvme_gb_per_node > 0
            master_cmd = [
                "mooncake_master",
                "--rpc_port",
                str(mc_master_port),
                "--metrics_port",
                str(mc_metrics_port),
                "--enable_http_metadata_server=false",
                "--default_kv_lease_ttl=5000",
            ]
            if has_nvme:
                master_cmd.extend(
                    [
                        "--enable_offload=true",
                        "--offload_on_evict=true",
                        "--promotion_on_hit=true",
                        "--enable_disk_eviction=true",
                    ]
                )
            _, master_log = self._spawn(master_cmd, base_env, "mooncake_master.log")
            _wait_tcp("127.0.0.1", mc_master_port, timeout_s=20.0)

            rdma_devs = [args.rdma_dev_node0, args.rdma_dev_node1]
            rdma_ips = [args.rdma_ip_node0, args.rdma_ip_node1]
            numa_nodes_mc = [args.numa_node0, args.numa_node1]
            nvme_dirs = [str(self.nvme_n0), str(self.nvme_n1)]

            # Apples-to-apples per-node DRAM budget:
            # standalone-store splits DRAM across mooncake_client global_segment +
            # mooncake_client offload buffer + vLLM requester local_buffer == dram_bytes.
            align = 16 * 1024 * 1024
            if dram_bytes >= 16 * (1024**3):
                req_buf_bytes = 4 * (1024**3)
                client_buf_bytes = 4 * (1024**3) if has_nvme else 512 * (1024**2)
                raw_seg = max(4 * (1024**3), dram_bytes - req_buf_bytes - client_buf_bytes)
                seg_bytes = ((raw_seg + align - 1) // align) * align
            else:
                req_buf_bytes = min(1024**3, max(256 * 1024**2, dram_bytes // 4))
                client_buf_bytes = min(1024**3, max(256 * 1024**2, dram_bytes // 4)) if has_nvme else 128 * 1024**2
                raw_seg = max(256 * 1024**2, dram_bytes - req_buf_bytes - client_buf_bytes)
                seg_bytes = ((raw_seg + align - 1) // align) * align

            # Spawn one NUMA-pinned mooncake_client daemon per node owning DRAM + NVMe tier
            client_procs: list[tuple[int, subprocess.Popen, pathlib.Path]] = []
            for nid in (0, 1):
                cport = _find_free_port()
                cenv = base_env.copy()
                cenv["MC_GID_INDEX"] = str(args.rdma_gid_index)
                if has_nvme:
                    cenv["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] = nvme_dirs[nid]
                    cenv["MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES"] = str(nvme_bytes)
                    cenv["MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE"] = str(nvme_bytes)
                    cenv["MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES"] = str(nvme_bytes)
                    cenv["MOONCAKE_OFFLOAD_USE_URING"] = "1"
                    cenv["MOONCAKE_OFFLOAD_ENABLE_EVICTION"] = "1"
                    cenv["MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION"] = "1"
                ccmd = _numa_cmd(numa_nodes_mc[nid]) + [
                    "mooncake_client",
                    "--host",
                    rdma_ips[nid],
                    "--port",
                    str(cport),
                    "--metadata_server",
                    "P2PHANDSHAKE",
                    "--master_server_address",
                    f"127.0.0.1:{mc_master_port}",
                    "--protocol",
                    "rdma",
                    "--device_names",
                    rdma_devs[nid],
                    "--global_segment_size",
                    str(seg_bytes),
                    "--local_buffer_size",
                    str(client_buf_bytes),
                    f"--enable_offload={'true' if has_nvme else 'false'}",
                    f"--start_offload_rpc_server={'true' if has_nvme else 'false'}",
                    "--threads",
                    "4",
                ]
                cp, clp = self._spawn(ccmd, cenv, f"mooncake_client_node{nid}.log")
                client_procs.append((nid, cp, clp))

            # Wait for both mooncake_client daemons to mount their RDMA segments on master
            mounted_segs: dict[int, str] = {}
            mount_deadline = time.perf_counter() + args.startup_timeout_s
            while len(mounted_segs) < 2 and time.perf_counter() < mount_deadline:
                for nid, cp, clp in client_procs:
                    if cp.poll() is not None:
                        raise RuntimeError(
                            f"mooncake_client node{nid} exited early (rc={cp.returncode}).\n"
                            f"--- {clp} ---\n{_tail_file(clp, 50)}"
                        )
                if master_log.exists():
                    txt = master_log.read_text(encoding="utf-8", errors="replace")
                    for nid in (0, 1):
                        if nid in mounted_segs:
                            continue
                        m = _re.search(
                            rf"action=mount_segment,\s*segment_name=({_re.escape(rdma_ips[nid])}:\d+)",
                            txt,
                        )
                        if m:
                            mounted_segs[nid] = m.group(1)
                if len(mounted_segs) < 2:
                    time.sleep(1.0)
            if len(mounted_segs) < 2:
                raise RuntimeError(
                    f"Timed out waiting for mooncake_client mount_segment (got {mounted_segs}).\n"
                    f"--- {master_log} ---\n{_tail_file(master_log, 40)}"
                )
            print(
                f"[e2e] mooncake_native store daemons READY: node0={mounted_segs[0]} node1={mounted_segs[1]}",
                flush=True,
            )

            for nid in (0, 1):
                cfg_json = self.work_dir / f"mooncake_native_node{nid}.json"
                cfg_obj = {
                    "mode": "standalone-store",
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": f"127.0.0.1:{mc_master_port}",
                    "protocol": "rdma",
                    "device_name": rdma_devs[nid],
                    "global_segment_size": 0,
                    "local_buffer_size": req_buf_bytes,
                    "enable_offload": has_nvme,
                }
                cfg_json.write_text(json.dumps(cfg_obj, indent=2) + "\n", encoding="utf-8")
                node_extra_envs[nid] = {
                    "MOONCAKE_CONFIG_PATH": str(cfg_json),
                    "MOONCAKE_REQUESTER_LOCAL_HOSTNAME": rdma_ips[nid],
                    "MC_GID_INDEX": str(args.rdma_gid_index),
                }
                kv_configs[nid] = {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_load_failure_policy": "recompute",
                    "kv_connector_extra_config": {
                        "load_async": True,
                        "preferred_segment": mounted_segs[nid],
                    },
                }

        else:
            raise ValueError(f"Unknown system: {self.system}")

        # Launch Node 0 (NUMA 0, GPU 0) and Node 1 (NUMA 1, GPU 1) concurrently
        # Note: for VeriCache, Node 0 (TCP listener) starts slightly before Node 1 (connector)
        # so peer handshake succeeds cleanly during register_kv_caches.
        ports = [self.port0, self.port1]
        numa_nodes = [args.numa_node0, args.numa_node1]
        cuda_devs = [args.cuda_dev_node0, args.cuda_dev_node1]
        logs: list[pathlib.Path] = []

        for nid in (0, 1):
            env = base_env.copy()
            env.update(node_extra_envs[nid])
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devs[nid])

            cmd = _numa_cmd(numa_nodes[nid]) + [
                "vllm",
                "serve",
                args.model_path,
                "--served-model-name",
                args.served_model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(ports[nid]),
                "--dtype",
                "bfloat16",
                "--block-size",
                str(args.block_size),
                "--max-model-len",
                str(args.max_model_len),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--kv-cache-memory-bytes",
                str(node_gpu_kv_bytes[nid]),
                "--enable-prefix-caching",
                "--disable-log-requests",
            ]
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            if kv_configs[nid] is not None:
                cmd.extend(["--kv-transfer-config", json.dumps(kv_configs[nid])])

            _proc, lp = self._spawn(cmd, env, f"vllm_node{nid}.log", is_vllm=True)
            logs.append(lp)
            if self.system == "vericache" and nid == 0:
                time.sleep(1.0)

        print(
            f"[e2e] Waiting for {self.system} Node 0 ({self.url0}) & Node 1 ({self.url1})...",
            flush=True,
        )
        _wait_vllm_health(self.url0, self.vllm_procs[0], logs[0], timeout_s=args.startup_timeout_s)
        _wait_vllm_health(self.url1, self.vllm_procs[1], logs[1], timeout_s=args.startup_timeout_s)
        print(f"[e2e] {self.system} cluster READY.", flush=True)

    def stop(self) -> None:
        for p in self.vllm_procs:
            _kill_proc_tree(p)
        for p in self.aux_procs:
            _kill_proc_tree(p)
        for lf in self.log_files:
            try:
                lf.close()
            except Exception:
                pass
        self.vllm_procs.clear()
        self.aux_procs.clear()
        self.log_files.clear()
        shutil.rmtree(self.nvme_n0, ignore_errors=True)
        shutil.rmtree(self.nvme_n1, ignore_errors=True)


def _compare_greedy_outputs(
    baseline_jsonl: pathlib.Path, target_jsonl: pathlib.Path
) -> dict[str, Any]:
    if not baseline_jsonl.exists() or not target_jsonl.exists():
        return {"checked": 0, "matched": 0, "match_rate": 0.0, "first_char_match_rate": 0.0}
    base: dict[int, str] = {}
    for line in baseline_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not obj.get("error"):
                base[obj["req_idx"]] = obj.get("text", obj.get("text_prefix", ""))
    checked = 0
    matched = 0
    first_char_matched = 0
    for line in target_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if not obj.get("error") and obj["req_idx"] in base:
                checked += 1
                txt = obj.get("text", obj.get("text_prefix", ""))
                btxt = base[obj["req_idx"]]
                if txt == btxt:
                    matched += 1
                if txt[:1] == btxt[:1]:
                    first_char_matched += 1
    return {
        "checked": checked,
        "matched": matched,
        "match_rate": matched / max(1, checked),
        "first_char_match_rate": first_char_matched / max(1, checked),
    }


def _display_system(system: str) -> str:
    if system == "mooncake":
        return "mooncake (lmcache)"
    if system == "mooncake_native":
        return "mooncake (native)"
    return system


def _generate_markdown_report(run_dir: pathlib.Path, records: list[dict[str, Any]], args: argparse.Namespace) -> pathlib.Path:
    lines: list[str] = []
    lines.append(f"# End-to-End 2-Node Dual-NUMA vLLM Trace Replay (`{run_dir.name}`)")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Model**: `{args.model_path}` (`bfloat16`, `block_size={args.block_size}`, `max_model_len={args.max_model_len}`)")
    lines.append(
        f"- **Topology (Morphology I — Peer-Shared, {int(args.sticky_ratio*100)}/{int((1-args.sticky_ratio)*100)} sticky)**: "
        f"Node 0 (`NUMA {args.numa_node0}`, `GPU {args.cuda_dev_node0}`, `{args.rdma_dev_node0}`, `{args.nvme_dir_node0}`) ↔ "
        f"Node 1 (`NUMA {args.numa_node1}`, `GPU {args.cuda_dev_node1}`, `{args.rdma_dev_node1}`, `{args.nvme_dir_node1}`)"
    )
    lines.append(
        f"- **Per-Node Memory Budget (Apples-to-Apples)**: "
        f"Total GPU KV `{args.gpu_kv_gb_per_node:.1f} GiB` (VeriCache splits `{args.gpu_kv_gb_per_node - args.vericache_hbm_gb_per_node:.1f} GiB` vLLM + `{args.vericache_hbm_gb_per_node:.1f} GiB` GDR HBM pool) | "
        f"DRAM `{args.dram_gb_per_node:.1f} GiB` | NVMe `{args.nvme_gb_per_node:.1f} GiB`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Case | QPS | System | Err | Req/s | In tok/s | TTFT P50 (ms) | TTFT P99 (ms) | "
        "TPOT P50 (ms) | E2E P50 (ms) | E2E P99 All (ms) | SLO<1s (All) | GPU Hit% | Ext/All% | Tot Hit% | Server CPU% | RSS (GB) | Greedy Match |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(records, key=lambda x: (x["case"], x.get("qps_target") or 0.0, x["system"])):
        qps_val = r.get("qps_target")
        qps_s = f"{qps_val:.1f}" if (qps_val is not None and qps_val > 0) else "closed"
        gm = r.get("greedy_match", {})
        gm_s = f"{gm['match_rate']*100:.1f}%" if gm.get("checked") else "-"
        e2e_p99_all = r.get("e2e_p99_all_ms", r["e2e_p99_ms"])
        ext_all = r.get("vllm_ext_hit_total_rate", 0.0)
        tot_hit = r.get("vllm_total_hit_rate", r["vllm_gpu_hit_rate"] + ext_all)
        lines.append(
            f"| `{r['case']}` | {qps_s} | **{_display_system(r['system'])}** | {r.get('errors', 0)} | "
            f"{r['req_per_s']:.2f} | {r['input_tok_per_s']:.0f} | {r['ttft_p50_ms']:.1f} | "
            f"{r['ttft_p99_ms']:.1f} | {r['tpot_p50_ms']:.2f} | {r['e2e_p50_ms']:.1f} | "
            f"{e2e_p99_all:.1f} | {r['slo_ttft_1000ms']*100:.1f}% | {r['vllm_gpu_hit_rate']*100:.1f}% | "
            f"{ext_all*100:.1f}% | {tot_hit*100:.1f}% | {r.get('server_cpu_pct', 0.0):.1f} | "
            f"{r.get('server_rss_gb', 0.0):.1f} | {gm_s} |"
        )
    lines.append("")
    out = run_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="2-Node Dual-NUMA E2E vLLM Trace Replay Orchestrator")
    ap.add_argument(
        "--systems",
        nargs="+",
        default=["baseline", "vericache", "lmcache", "mooncake"],
        choices=["baseline", "vericache", "lmcache", "mooncake", "mooncake_native"],
    )
    ap.add_argument(
        "--traces",
        nargs="+",
        default=["toolagent"],
        choices=["toolagent", "conversation"],
    )
    ap.add_argument("--qps-list", nargs="+", type=float, default=[2.0])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--served-model-name", default="qwen2.5-7b")
    ap.add_argument("--trace-dir", default="/workspace/target/bench-traces")
    ap.add_argument("--max-requests", type=int, default=200)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--tokens-per-hash-id", type=int, default=512)
    ap.add_argument("--max-input-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=17408)
    ap.add_argument("--max-output-tokens", type=int, default=64)
    ap.add_argument("--min-output-tokens", type=int, default=16)
    ap.add_argument("--fixed-output-tokens", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--progress-interval-reqs", type=int, default=200)
    ap.add_argument("--progress-interval-s", type=float, default=30.0)
    ap.add_argument("--sticky-ratio", type=float, default=0.70)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--lmcache-chunk-size", type=int, default=256)
    ap.add_argument("--lmcache-p2p", action="store_true", default=True)
    ap.add_argument("--no-lmcache-p2p", dest="lmcache_p2p", action="store_false")
    ap.add_argument("--enforce-eager", action="store_true", default=False)

    # Equalized per-node memory tiers
    ap.add_argument("--gpu-kv-gb-per-node", type=float, default=8.0)
    ap.add_argument("--vericache-hbm-gb-per-node", type=float, default=0.125)
    ap.add_argument("--vericache-lazy-offload", action="store_true", default=False)
    ap.add_argument("--vericache-store-chunk-size", type=int, default=64)
    ap.add_argument("--vericache-store-direct-dram", action="store_true", default=True)
    ap.add_argument("--no-vericache-store-direct-dram", dest="vericache_store_direct_dram", action="store_false")
    ap.add_argument("--dram-gb-per-node", type=float, default=24.0)
    ap.add_argument("--nvme-gb-per-node", type=float, default=48.0)

    # Dual-NUMA 2-Node hardware mapping (override via CLI flags or VERICACHE_* env vars)
    ap.add_argument("--numa-node0", type=int, default=0)
    ap.add_argument("--numa-node1", type=int, default=1)
    ap.add_argument("--cuda-dev-node0", type=int, default=0)
    ap.add_argument("--cuda-dev-node1", type=int, default=1)
    ap.add_argument("--rdma-dev-node0", default=os.environ.get("VERICACHE_RDMA_DEV_NODE0", "mlx5_0"))
    ap.add_argument("--rdma-dev-node1", default=os.environ.get("VERICACHE_RDMA_DEV_NODE1", "mlx5_1"))
    ap.add_argument("--rdma-ip-node0", default=os.environ.get("VERICACHE_RDMA_IP_NODE0", "127.0.0.1"))
    ap.add_argument("--rdma-ip-node1", default=os.environ.get("VERICACHE_RDMA_IP_NODE1", "127.0.0.1"))
    ap.add_argument("--rdma-gid-index", type=int, default=3)
    ap.add_argument("--nvme-dir-node0", default=os.environ.get("VERICACHE_NVME_DIR_NODE0", "/tmp/vericache-e2e-nvme-n0"))
    ap.add_argument("--nvme-dir-node1", default=os.environ.get("VERICACHE_NVME_DIR_NODE1", "/tmp/vericache-e2e-nvme-n1"))
    ap.add_argument("--startup-timeout-s", type=float, default=240.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true", default=False)
    args = ap.parse_args()

    run_id = args.run_id or (
        "e2e_vllm_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (args.resume and (run_dir / "config.json").exists()):
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    trace_map = {
        "toolagent": pathlib.Path(args.trace_dir) / "toolagent_trace.jsonl",
        "conversation": pathlib.Path(args.trace_dir) / "conversation_trace.jsonl",
    }

    all_records: list[dict[str, Any]] = []
    raw_jsonl = run_dir / "results.jsonl"
    existing_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and raw_jsonl.exists():
        for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            q_val = r.get("qps_target")
            q_tag = f"qps{str(q_val).replace('.', 'p')}" if (q_val is not None and q_val > 0) else "closed"
            existing_by_key[(f"{r['case']}_{q_tag}", r["system"])] = r
            all_records.append(r)
    else:
        raw_jsonl.write_text("", encoding="utf-8")

    for trace_name in args.traces:
        trace_path = trace_map[trace_name]
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing trace file: {trace_path}")
        for qps in args.qps_list:
            qps_tag = f"qps{str(qps).replace('.', 'p')}" if qps > 0 else "closed"
            case_label = f"{trace_name}_{qps_tag}"
            for system in args.systems:
                if args.resume and (case_label, system) in existing_by_key:
                    print(f"[e2e] RESUME skip already completed {case_label} {system}", flush=True)
                    continue
                work_dir = run_dir / "logs" / f"{case_label}_{system}"
                cluster = ClusterContext(system, args, work_dir)
                try:
                    last_start_err: Exception | None = None
                    for attempt in range(1, 4):
                        try:
                            cluster.start()
                            last_start_err = None
                            break
                        except Exception as exc:
                            last_start_err = exc
                            print(
                                f"[e2e] WARN {case_label} {system} start attempt {attempt}/3 failed: {exc}",
                                flush=True,
                            )
                            cluster.stop()
                            if attempt < 3:
                                time.sleep(10.0)
                                cluster = ClusterContext(system, args, work_dir)
                    if last_start_err is not None:
                        raise last_start_err
                    pids = [p.pid for p in cluster.vllm_procs + cluster.aux_procs]
                    cpu0, _ = _sample_tree_cpu_rss(pids)
                    t0 = time.perf_counter()

                    out_json = run_dir / "raw" / f"{case_label}_{system}.json"
                    out_reqs = run_dir / "outputs" / f"{case_label}_{system}.jsonl"
                    cmd = [
                        sys.executable,
                        str(E2E_DIR / "replayer.py"),
                        "--system",
                        system,
                        "--case",
                        trace_name,
                        "--model-path",
                        args.model_path,
                        "--served-model-name",
                        args.served_model_name,
                        "--trace-path",
                        str(trace_path),
                        "--node0-url",
                        cluster.url0,
                        "--node1-url",
                        cluster.url1,
                        "--start-index",
                        str(args.start_index),
                        "--max-requests",
                        str(args.max_requests),
                        "--tokens-per-hash-id",
                        str(args.tokens_per_hash_id),
                        "--max-input-tokens",
                        str(args.max_input_tokens),
                        "--max-output-tokens",
                        str(args.max_output_tokens),
                        "--min-output-tokens",
                        str(args.min_output_tokens),
                        "--sticky-ratio",
                        str(args.sticky_ratio),
                        "--max-concurrency",
                        str(args.max_concurrency),
                        "--timeout-s",
                        str(args.timeout_s),
                        "--progress-interval-reqs",
                        str(args.progress_interval_reqs),
                        "--progress-interval-s",
                        str(args.progress_interval_s),
                        "--warmup",
                        "--output-json",
                        str(out_json),
                        "--save-outputs",
                        str(out_reqs),
                    ]
                    if qps > 0:
                        cmd.extend(["--qps", str(qps)])
                    if args.fixed_output_tokens is not None:
                        cmd.extend(["--fixed-output-tokens", str(args.fixed_output_tokens)])

                    replayer_log = work_dir / "replayer.log"
                    stderr_tail: list[str] = []
                    with replayer_log.open("w", encoding="utf-8") as rlf:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                        )
                        assert proc.stderr is not None and proc.stdout is not None
                        for line in proc.stderr:
                            rlf.write(line)
                            rlf.flush()
                            sys.stderr.write(line)
                            sys.stderr.flush()
                            stderr_tail.append(line)
                            if len(stderr_tail) > 50:
                                stderr_tail.pop(0)
                        stdout_text = proc.stdout.read()
                        rc = proc.wait()
                    dt = max(1e-6, time.perf_counter() - t0)
                    cpu1, rss_mb = _sample_tree_cpu_rss(pids)
                    if rc != 0:
                        raise RuntimeError(
                            f"Replayer FAILED ({system} {case_label}, rc={rc}):\n"
                            f"STDOUT:\n{stdout_text}\nSTDERR (tail):\n{''.join(stderr_tail)}"
                        )
                    rec = json.loads(out_json.read_text(encoding="utf-8"))
                    rec["server_cpu_pct"] = max(0.0, (cpu1 - cpu0) / dt * 100.0)
                    rec["server_rss_gb"] = rss_mb / 1024.0

                    if system != "baseline":
                        base_reqs = run_dir / "outputs" / f"{case_label}_baseline.jsonl"
                        rec["greedy_match"] = _compare_greedy_outputs(base_reqs, out_reqs)
                    else:
                        rec["greedy_match"] = {"checked": rec["completed"], "matched": rec["completed"], "match_rate": 1.0}

                    all_records.append(rec)
                    with raw_jsonl.open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    _generate_markdown_report(run_dir, all_records, args)
                    print(
                        f"[e2e] DONE {case_label} {system:<10} | err={rec.get('errors', 0)} "
                        f"req/s={rec['req_per_s']:.2f} "
                        f"TTFT p50={rec['ttft_p50_ms']:.1f}ms p99={rec['ttft_p99_ms']:.1f}ms "
                        f"E2E p99_all={rec.get('e2e_p99_all_ms', rec['e2e_p99_ms']):.1f}ms "
                        f"gpu_hit={rec['vllm_gpu_hit_rate']*100:.1f}% "
                        f"ext_all={rec.get('vllm_ext_hit_total_rate', 0.0)*100:.1f}% "
                        f"tot_hit={rec.get('vllm_total_hit_rate', 0.0)*100:.1f}% "
                        f"cpu={rec['server_cpu_pct']:.1f}% rss={rec['server_rss_gb']:.1f}GB "
                        f"greedy={rec['greedy_match']['match_rate']*100:.1f}%",
                        flush=True,
                    )
                finally:
                    cluster.stop()
                    # Short cooldown so CUDA/RDMA/sockets fully release
                    time.sleep(2.0)

    summary_path = _generate_markdown_report(run_dir, all_records, args)
    print(f"\n[e2e] COMPLETE: summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
