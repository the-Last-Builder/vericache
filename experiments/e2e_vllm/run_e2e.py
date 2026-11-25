#!/usr/bin/env python3
"""Launch a pair of NUMA-pinned vLLM servers and run the Step 1 replay."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import ipaddress
from importlib import metadata
import json
import math
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

import psutil

from step1_metrics import GIB, read_hca, read_nvme_usage, read_vericache
from trace_tokens import FRAME_BYTES

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


def _kill_proc_tree(proc: subprocess.Popen, grace_s: float = 8.0) -> None:
    # Each spawned server owns its process group even if its original parent exits.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.poll()
        return
    end = time.monotonic() + grace_s
    while time.monotonic() < end:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def _numa_cmd(numa_node: int) -> list[str]:
    executable = shutil.which("numactl")
    if executable is None:
        raise RuntimeError("numactl is required for Step 1")
    command = [executable, f"--cpunodebind={numa_node}", f"--membind={numa_node}"]
    probe = subprocess.run([*command, "true"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"NUMA binding to node {numa_node} failed: {probe.stderr.strip()}")
    return command


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def _versions() -> dict[str, str | None]:
    versions = {}
    for name in ("vllm", "torch", "transformers", "lmcache", "mooncake-transfer-engine",
                 "mooncake-store", "nixl", "aiohttp", "psutil", "vericache-python"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _vericache_frames(args: argparse.Namespace, frame_bytes: int) -> tuple[int, int, int]:
    return tuple(int(gib * GIB) // frame_bytes for gib in (
        args.vericache_hbm_gb_per_node, args.dram_gb_per_node, args.nvme_gb_per_node,
    ))


def _vllm_flags(args: argparse.Namespace, kv_bytes: int) -> list[str]:
    flags = [
        "--tensor-parallel-size", "1",
        "--distributed-executor-backend", "uni",
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--block-size", str(args.block_size),
        "--kv-cache-memory-bytes", str(kv_bytes),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", "8192",
        "--enable-prefix-caching", "--prefix-caching-hash-algo", "sha256",
        "--enable-chunked-prefill",
        "--attention-backend", "FLASH_ATTN",
        "--generation-config", "vllm", "--seed", "42",
        "--disable-log-requests",
    ]
    if args.mode == "correctness":
        flags.extend(["--enforce-eager", "--no-async-scheduling"])
    else:
        flags.extend(["--async-scheduling", "--compilation-config",
                      json.dumps({"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"})])
    return flags


def _make_nvme_dir(root: str, system: str, node: int) -> pathlib.Path:
    directory = pathlib.Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{system}-n{node}-", dir=directory))


def _validate_nvme(paths: list[pathlib.Path]) -> None:
    devices = []
    for path in paths:
        name = read_nvme_usage(str(path), include_usage=False).get("device_name", "")
        physical = re.match(r"^nvme\d+(?=n\d+)", name)
        if not physical:
            raise RuntimeError(f"{path} must be mounted on an independently measurable NVMe device")
        devices.append(physical.group())
    if len(set(devices)) != len(devices):
        raise RuntimeError(f"Step 1 requires independent NVMe devices, got {devices}")


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
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.port0 = _find_free_port()
        self.port1 = _find_free_port()
        self.url0 = f"http://127.0.0.1:{self.port0}"
        self.url1 = f"http://127.0.0.1:{self.port1}"

        self.aux_procs: list[subprocess.Popen] = []
        self.vllm_procs: list[subprocess.Popen] = []
        self.log_files: list[Any] = []

        self.nvme_paths: list[pathlib.Path] = []
        self.stats_sockets: list[str] = []
        self.aux_metrics_urls: list[str] = []
        self.spawned: list[dict] = []
        self.connector_configs = []
        self.backend_configs: list[dict] = []
        self.resolved_servers = []
        self.library: dict | None = None

    @property
    def urls(self) -> list[str]:
        return [self.url0, self.url1]

    @property
    def hca_devices(self) -> list[str]:
        return [self.args.rdma_dev_node0, self.args.rdma_dev_node1]

    def deployment_id(self) -> str:
        options = {
            k: v for k, v in vars(self.args).items()
            if k.startswith(("rdma_", "cuda_dev_", "numa_", "vericache_", "mooncake_", "lmcache_"))
            or k in {"mode", "gpu_kv_gb_per_node", "dram_gb_per_node", "nvme_gb_per_node",
                     "nvme_dir_node0", "nvme_dir_node1", "max_num_seqs", "max_model_len", "block_size"}
        }
        options["model_path"] = str(pathlib.Path(self.args.model_path).resolve())
        options["model_config_sha256"] = _sha256(pathlib.Path(self.args.model_path) / "config.json")
        options["packages"] = _versions()
        options["library"] = self.library["sha256"] if self.library else None
        return hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()

    def write_manifest(self, replay_argv: list[str] | None = None,
                       replay_env: dict[str, str] | None = None) -> None:
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
            git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            git_sha = os.environ.get("VERICACHE_GIT_SHA", "unknown")
            git_status = os.environ.get("VERICACHE_GIT_STATUS", "")
        manifest = {
            "system": self.system,
            "mode": self.args.mode,
            "git_sha": git_sha,
            "git_status": git_status,
            "python": sys.version,
            "packages": _versions(),
            "native_library": self.library,
            "orchestrator_argv": sys.argv,
            "options": vars(self.args),
            "commands": self.spawned,
            "connector_configs": self.connector_configs,
            "backend_configs": self.backend_configs,
            "vllm_resolved": self.resolved_servers,
            "hca_devices": self.hca_devices,
            "nvme_paths": [str(path) for path in self.nvme_paths],
        }
        if replay_argv is not None:
            manifest["replayer"] = {
                "argv": replay_argv,
                "env": replay_env,
            }
            window = getattr(self.args, "frozen_window", None)
            if window:
                manifest["frozen_window_sha256"] = _sha256(pathlib.Path(window))
        path = self.work_dir / "manifest.json"
        temp = self.work_dir / ".manifest.json.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, default=str)
            output.write("\n")
        temp.replace(path)

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
        self.spawned.append({"argv": cmd, "env": dict(env), "pid": proc.pid, "log": log_name})
        if is_vllm:
            self.vllm_procs.append(proc)
        else:
            self.aux_procs.append(proc)
        return proc, lp

    def start(self) -> None:
        args = self.args
        numa_commands = [_numa_cmd(args.numa_node0), _numa_cmd(args.numa_node1)]
        if len(set(self.hca_devices)) != 2:
            raise ValueError("Step 1 requires two distinct HCA devices")
        read_hca(self.hca_devices)
        if self.system == "mooncake_native" or (self.system == "lmcache" and args.lmcache_p2p):
            rdma_ips = [ipaddress.ip_address(args.rdma_ip_node0), ipaddress.ip_address(args.rdma_ip_node1)]
            if len(set(rdma_ips)) != 2 or any(ip.is_loopback or ip.is_unspecified for ip in rdma_ips):
                raise ValueError(f"{self.system} requires two distinct RoCE interface IPs")
        if self.system != "baseline" and args.nvme_gb_per_node > 0:
            for node, root in enumerate((args.nvme_dir_node0, args.nvme_dir_node1)):
                self.nvme_paths.append(_make_nvme_dir(root, self.system, node))
            _validate_nvme(self.nvme_paths)
        else:
            self.nvme_paths = []
        nvme_dirs = [str(path) for path in self.nvme_paths] or [None, None]

        base_env = os.environ.copy()
        base_env.pop("VLLM_MOONCAKE_STORE_TIER_LOG", None)
        base_env["VLLM_BATCH_INVARIANT"] = "1" if args.mode == "correctness" else "0"
        # Config and HBM reset endpoints are used only on localhost.
        base_env["VLLM_SERVER_DEV_MODE"] = "1"
        py_paths = [str(ROOT / "vericache-python" / "python"), str(E2E_DIR)]
        if base_env.get("PYTHONPATH"):
            py_paths.append(base_env["PYTHONPATH"])
        base_env["PYTHONPATH"] = ":".join(py_paths)
        if self.system == "vericache":
            package = ROOT / "vericache-python"
            explicit = base_env.get("VERICACHE_PYTHON_LIB")
            candidates = [pathlib.Path(explicit)] if explicit else [
                package / "python" / "vericache" / "libvericache_python.so",
                package / "target" / "release" / "libvericache_python.so",
                package / "target" / "debug" / "libvericache_python.so",
            ]
            lib_so = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if lib_so is None:
                raise FileNotFoundError(f"VeriCache native library missing: {candidates}")
            base_env["VERICACHE_PYTHON_LIB"] = str(lib_so)
            self.library = {"path": str(lib_so), "sha256": _sha256(lib_so)}
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
        if frame_bytes != FRAME_BYTES or args.block_size != 16:
            raise ValueError(f"Step 1 expects {FRAME_BYTES} bytes per 16-token frame, got {frame_bytes}")
        gpu_kv_bytes = int(args.gpu_kv_gb_per_node * GIB) // frame_bytes * frame_bytes

        # Per-system control plane & kv_transfer_config setup
        node_extra_envs: list[dict[str, str]] = [{}, {}]
        kv_configs: list[dict[str, Any] | None] = [None, None]
        node_gpu_kv_bytes = [gpu_kv_bytes, gpu_kv_bytes]

        if self.system == "baseline":
            pass

        elif self.system == "vericache":
            hbm_frames, dram_frames, nvme_frames = _vericache_frames(args, frame_bytes)
            vc_vllm_kv_bytes = gpu_kv_bytes - hbm_frames * frame_bytes
            if vc_vllm_kv_bytes <= 0 or dram_frames <= 0:
                raise ValueError("VeriCache requires positive vLLM KV and host DRAM capacity")
            node_gpu_kv_bytes = [vc_vllm_kv_bytes, vc_vllm_kv_bytes]
            max_cached_blocks = dram_frames + nvme_frames

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
                    str(max_cached_blocks),
                ],
                base_env,
                "vericache_directory.log",
            )
            _wait_tcp("127.0.0.1", dir_port, timeout_s=15.0)

            rdma_devs = [args.rdma_dev_node0, args.rdma_dev_node1]
            self.stats_sockets = [str(self.work_dir / f"vc{nid}.sock") for nid in (0, 1)]
            # AF_UNIX sun_path is 108 bytes on Linux, including the null terminator.
            if any(len(os.fsencode(path)) >= 108 for path in self.stats_sockets):
                tag = hashlib.sha256(os.fsencode(self.work_dir.resolve())).hexdigest()[:24]
                self.stats_sockets = [f"/tmp/vericache-{tag}-{node}.sock" for node in (0, 1)]
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
                        "local_lookup_socket": self.stats_sockets[nid],
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
                    cfg_obj["nixl_backends"] = ["UCX"]
                self.backend_configs.append(cfg_obj)
                _write_yaml(cfg_yaml, cfg_obj)
                node_extra_envs[nid] = {
                    "LMCACHE_CONFIG_FILE": str(cfg_yaml),
                    "LMCACHE_USE_EXPERIMENTAL": "True",
                    "UCX_NET_DEVICES": f"{rdma_devs[nid]}:1",
                    "UCX_TLS": "rc",
                    "UCX_IB_GID_INDEX": str(args.rdma_gid_index),
                    "UCX_IB_REG_MT_THRESH": "inf",
                }
                kv_configs[nid] = {
                    "kv_connector": "LMCacheConnectorV1Compat",
                    "kv_connector_module_path": "lmcache_compat",
                    "kv_role": "kv_both",
                }

        elif self.system == "mooncake_native":
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
                f"--default_kv_lease_ttl={args.mooncake_lease_ttl_ms}",
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
            self.aux_metrics_urls.append(f"http://127.0.0.1:{mc_metrics_port}")
            _wait_tcp("127.0.0.1", mc_master_port, timeout_s=20.0)

            rdma_devs = [args.rdma_dev_node0, args.rdma_dev_node1]
            rdma_ips = [args.rdma_ip_node0, args.rdma_ip_node1]

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
                ccmd = numa_commands[nid] + [
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
                        m = re.search(
                            rf"action=mount_segment,\s*segment_name=({re.escape(rdma_ips[nid])}:\d+)",
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
                self.backend_configs.append(cfg_obj)
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
        self.connector_configs = kv_configs
        cuda_devs = [args.cuda_dev_node0, args.cuda_dev_node1]
        logs: list[pathlib.Path] = []

        for nid in (0, 1):
            env = base_env.copy()
            env.update(node_extra_envs[nid])
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_devs[nid])

            cmd = numa_commands[nid] + [
                "vllm", "serve", args.model_path,
                "--served-model-name", args.served_model_name,
                "--host", "127.0.0.1", "--port", str(ports[nid]),
                *_vllm_flags(args, node_gpu_kv_bytes[nid]),
            ]
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
        for url in self.urls:
            version = _fetch_json(f"{url}/version")["version"]
            info = _fetch_json(f"{url}/server_info?config_format=json")
            self.resolved_servers.append({"url": url, "version": version, "server_info": info})
            if version.split("+", 1)[0] != "0.34.0":
                raise RuntimeError(f"Step 1 requires vLLM 0.34.0, {url} reports {version}")
        self.write_manifest()
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
        for path in self.nvme_paths:
            shutil.rmtree(path, ignore_errors=True)
        for path in self.stats_sockets:
            pathlib.Path(path).unlink(missing_ok=True)


def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mode", choices=("performance", "correctness"), default="performance")
    parser.add_argument("--enforce-eager", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--gpu-kv-gb-per-node", type=float, default=16)
    parser.add_argument("--vericache-hbm-gb-per-node", type=float, default=0)
    parser.add_argument("--dram-gb-per-node", type=float, default=32)
    parser.add_argument("--nvme-gb-per-node", type=float, default=64)
    parser.add_argument("--vericache-lazy-offload", action="store_true")
    parser.add_argument("--vericache-store-chunk-size", type=int, default=64)
    parser.add_argument("--mooncake-lease-ttl-ms", type=int, default=10000)
    parser.add_argument("--lmcache-chunk-size", type=int, choices=(64, 256), default=256)
    parser.add_argument("--lmcache-p2p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--numa-node0", type=int, default=0)
    parser.add_argument("--numa-node1", type=int, default=1)
    parser.add_argument("--cuda-dev-node0", type=int, default=0)
    parser.add_argument("--cuda-dev-node1", type=int, default=1)
    for node in (0, 1):
        parser.add_argument(f"--rdma-dev-node{node}",
                            default=os.environ.get(f"VERICACHE_RDMA_DEV_NODE{node}", f"mlx5_{node}"))
        parser.add_argument(f"--rdma-ip-node{node}",
                            default=os.environ.get(f"VERICACHE_RDMA_IP_NODE{node}", "127.0.0.1"))
        parser.add_argument(f"--nvme-dir-node{node}", default=os.environ.get(
            f"VERICACHE_NVME_DIR_NODE{node}", f"/tmp/vericache-e2e-nvme-n{node}",
        ))
    parser.add_argument("--rdma-gid-index", type=int, default=3)
    parser.add_argument("--startup-timeout-s", type=float, default=240)


def _fault_pid_file(cluster: ClusterContext, args: argparse.Namespace) -> pathlib.Path:
    node = args.fault_requester_node
    root = psutil.Process(cluster.vllm_procs[node].pid)
    allowed = {proc.pid for proc in root.children(recursive=True)}
    allowed.add(root.pid)
    if args.fault_pid_file:
        path = pathlib.Path(args.fault_pid_file)
        pid = int(path.read_text().strip())
    elif cluster.stats_sockets:
        pid = int(read_vericache(cluster.stats_sockets[node])["worker_pid"])
        path = cluster.work_dir / "fault_worker.pid"
    else:
        workers = [proc for proc in root.children(recursive=True)
                   if "enginecore" in proc.name().replace("_", "").lower()
                   or "enginecore" in " ".join(proc.cmdline()).replace("_", "").lower()]
        if len(workers) != 1:
            raise RuntimeError(f"Could not identify a unique requester EngineCore among {sorted(allowed)}")
        pid = workers[0].pid
        path = cluster.work_dir / "fault_worker.pid"
    if pid not in allowed:
        raise ValueError(f"Fault PID {pid} does not belong to requester node {node}")
    if not args.fault_pid_file:
        path.write_text(f"{pid}\n", encoding="ascii")
    return path


def replay_command(cluster: ClusterContext, args: argparse.Namespace, system: str, qps: float,
                   output: pathlib.Path, requests: pathlib.Path) -> list[str]:
    cmd = [
        sys.executable, str(E2E_DIR / "replayer.py"),
        "--system", system, "--case", "toolagent",
        "--model-path", args.model_path, "--served-model-name", args.served_model_name,
        "--node0-url", cluster.url0, "--node1-url", cluster.url1,
        "--warmup-requests", str(args.warmup_requests),
        "--measure-requests", str(args.measure_requests),
        "--max-model-len", str(args.max_model_len), "--max-input-tokens", "32768",
        "--max-concurrency", str(args.max_concurrency), "--timeout-s", str(args.timeout_s),
        "--deployment-id", cluster.deployment_id(),
        "--output-json", str(output), "--save-outputs", str(requests),
    ]
    if args.frozen_window:
        cmd.extend(["--frozen-window", args.frozen_window])
    else:
        cmd.extend(["--trace-path", args.trace_path, "--start-index", str(args.start_index)])
    if args.clamp_output_max is not None:
        cmd.extend(["--clamp-output-max", str(args.clamp_output_max)])
    if args.record_logprobs or args.mode == "correctness":
        cmd.append("--record-logprobs")
    if args.cold_oracle:
        cmd.extend(["--cold-oracle", args.cold_oracle])
    if qps > 0:
        cmd.extend(["--qps", str(qps)])
    if cluster.stats_sockets:
        cmd.extend(["--stats-socket", *cluster.stats_sockets])
    if cluster.nvme_paths:
        cmd.extend(["--nvme-dir", *(str(path) for path in cluster.nvme_paths)])
    for dev in cluster.hca_devices:
        cmd.extend(["--hca-device", dev])
    for proc in cluster.vllm_procs + cluster.aux_procs:
        cmd.extend(["--server-pid", str(proc.pid)])
    for gpu in (args.cuda_dev_node0, args.cuda_dev_node1):
        cmd.extend(["--gpu-id", str(gpu)])
    for url in cluster.aux_metrics_urls:
        cmd.extend(["--aux-metrics-url", url])
    if args.fault_offsets_s:
        cmd.extend([
            "--fault-pid-file", str(_fault_pid_file(cluster, args)),
            "--fault-requester-node", str(args.fault_requester_node),
            "--fault-duration-s", str(args.fault_duration_s),
            "--fault-offsets-s", *(str(offset) for offset in args.fault_offsets_s),
        ])
    if args.paired_baseline:
        cmd.extend(["--paired-baseline", args.paired_baseline])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    parser.add_argument("--systems", nargs="+", default=["baseline", "vericache", "mooncake_native"],
                        choices=["baseline", "vericache", "lmcache", "mooncake_native"])
    parser.add_argument("--qps-list", nargs="+", type=float, default=[1.0, 0],
                        help="Positive values are open-loop QPS; 0 runs fixed concurrency")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frozen-window")
    source.add_argument("--trace-path")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=500)
    parser.add_argument("--measure-requests", type=int, default=1500)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--clamp-output-max", type=int)
    parser.add_argument("--record-logprobs", action="store_true")
    parser.add_argument("--cold-oracle")
    parser.add_argument("--fault-pid-file")
    parser.add_argument("--fault-requester-node", type=int, choices=(0, 1), default=1)
    parser.add_argument("--fault-offsets-s", nargs="+", type=float)
    parser.add_argument("--fault-duration-s", type=float)
    parser.add_argument("--paired-baseline")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if any(not math.isfinite(qps) or qps < 0 for qps in args.qps_list):
        parser.error("--qps-list accepts positive rates and 0 for fixed concurrency")
    fault_opts = (args.fault_offsets_s, args.fault_duration_s, args.fault_pid_file, args.paired_baseline)
    if any(option is not None for option in fault_opts):
        if not args.fault_offsets_s or not args.fault_duration_s:
            parser.error("Fault runs require offsets and duration")
        if len(args.systems) != 1 or len(args.qps_list) != 1:
            parser.error("Use one system and one load point per paired fault run")
    if args.cold_oracle and args.mode != "correctness":
        parser.error("--cold-oracle requires --mode correctness")
    for attr in ("frozen_window", "trace_path", "cold_oracle", "paired_baseline", "fault_pid_file"):
        path = getattr(args, attr)
        if path is not None and not pathlib.Path(path).is_file():
            parser.error(f"{attr} file does not exist: {path}")

    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("step1_%Y%m%dT%H%M%S%fZ")
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    for qps in args.qps_list:
        case = f"qps{qps:g}" if qps else "closed"
        for system in args.systems:
            if system == "baseline" and qps == 0 and len(args.systems) > 1:
                continue
            work_dir = run_dir / f"{case}_{system}"
            summary = work_dir / "summary.json"
            if args.resume and summary.exists():
                saved = json.loads(summary.read_text())
                if (saved["after"] is not None and saved["faults_scheduled"] == saved["faults_applied"]
                        and not saved.get("cold_oracle_exactness", {}).get("mismatches")):
                    print(f"[e2e] Resume skipped {work_dir}", flush=True)
                    continue
            cluster = ClusterContext(system, args, work_dir)
            try:
                cluster.start()
                cmd = replay_command(cluster, args, system, qps, summary, work_dir / "requests.jsonl")
                replay_env = dict(os.environ)
                replay_env.pop("VLLM_MOONCAKE_STORE_TIER_LOG", None)
                cluster.write_manifest(cmd, replay_env)
                with (work_dir / "replayer.log").open("w", encoding="utf-8") as log:
                    subprocess.run(cmd, env=replay_env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(summary.read_text())
                print(f"[e2e] {case}/{system} {report['completed']}/{report['total_requests']} completed, "
                      f"joint SLO {report['joint_slo_attainment']:.1%}, "
                      f"wall goodput {report['wall_clock_goodput_req_s']:.3f} req/s", flush=True)
            finally:
                try:
                    if not (work_dir / "manifest.json").exists():
                        cluster.write_manifest()
                finally:
                    cluster.stop()
    print(f"[e2e] Results saved in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
