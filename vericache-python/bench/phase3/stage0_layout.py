"""Launch vLLM and collect the production connector's initialized KV layout."""

from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


def model_command(model, block_size, port, scenario="layout", *, tensor_parallel_size=1, max_num_seqs=16):
    frame_size = 2 * block_size * 4 * 128 * 2
    transfer = {
        "kv_connector": "VeriCacheV1Connector",
        "kv_connector_module_path": "stage0_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "frame_size": frame_size,
            "num_shards": 4,
            "host_frames_per_shard": (1 << 28) // (4 * frame_size),
        },
    }
    options = transfer["kv_connector_extra_config"]
    if scenario in {"dram", "nvme"}:
        options["host_frames_per_shard"] = (1 << 29) // (4 * frame_size)
    if scenario == "nvme":
        options["nvme_frames_per_shard"] = (64 << 30) // (4 * frame_size)
        options["nvme_path"] = str(Path(os.environ["PHASE3_TEST_DIR"]) / "cache")
        assert options["nvme_frames_per_shard"] * 4 * frame_size == 64 << 30
    args = [
        "vllm", "serve", str(model), "--served-model-name", "Qwen3-30B-A3B",
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--distributed-executor-backend", "uni" if tensor_parallel_size == 1 else "mp",
        "--dtype", "bfloat16", "--max-model-len", "32768",
        "--enforce-eager", "--no-async-scheduling",
        "--disable-hybrid-kv-cache-manager", "--enable-prompt-tokens-details",
        "--enable-prefix-caching", "--block-size", str(block_size),
        "--gpu-memory-utilization", "0.89", "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", "8192",
        "--generation-config", "vllm", "--seed", "42",
    ]
    if scenario not in {"vanilla", "hbm"}:
        args.extend(["--kv-transfer-config", json.dumps(transfer)])
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        args.extend(["--attention-backend", "FLASH_ATTN"])
    return args


def reserve_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_manifest(directory, model, argv):
    (directory / "manifest.json").write_text(json.dumps({
        "vericache_git_base": os.environ["PHASE3_VERICACHE_SHA"],
        "request_sim_git": "3605393955418a144559545434751adefdc7a78f",
        "package_versions": {
            name: version(name) for name in
            ("vllm", "torch", "lmcache", "mooncake-transfer-engine")
        },
        "physical_gpu_ids": os.environ["CUDA_VISIBLE_DEVICES"].split(","),
        "numa_node": int((directory / "numa-node").read_text()),
        "model": str(model),
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1",
        "argv": argv,
    }, indent=2) + "\n")


@contextmanager
def start_server(directory, argv, endpoint):
    with (directory / "server.log").open("w") as log:
        server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while server.poll() is None and time.monotonic() < deadline:
                try:
                    with urlopen(endpoint + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError, ConnectionResetError):
                    time.sleep(1)
            else:
                raise RuntimeError(f"vLLM did not reach /health; see {log.name}")
            yield
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)


def main():
    directory = Path(os.environ["PHASE3_RESULT_DIR"])
    model = Path("/data/nvme2/zxw/models/Qwen3-30B-A3B")
    block_size = int(os.environ["PHASE3_BLOCK_SIZE"])
    assert block_size in (16, 64)
    port = reserve_port()
    argv = model_command(model, block_size, port)
    write_manifest(directory, model, argv)
    endpoint = f"http://127.0.0.1:{port}"
    with start_server(directory, argv, endpoint):
        layout = json.loads((directory / "kv-layout.json").read_text())
        assert layout["vllm_scheduler_block_size"] == block_size
        assert layout["kv_cache_groups"] == 1
        assert layout["kv_cache_spec"] == "FullAttentionSpec"
        assert layout["is_slab_cuda_pinned"]
        assert layout["hbm_kv_blocks"] > 0
        with urlopen(endpoint + "/metrics", timeout=10) as response:
            (directory / "server-metrics.prom").write_bytes(response.read())
        print(f"KV layout verified for block size {block_size} across 48 layers", flush=True)


if __name__ == "__main__":
    main()
