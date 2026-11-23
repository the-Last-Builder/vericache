"""Real PyTorch + CUDA H20 GPU + NVMe + 2-Node ConnectX GPUDirect RDMA E2E tests
for `vericache-python` and `VeriCacheConnector` (vLLM v1 KVConnectorBase_V1).

Tests:
1. `test_python_client_4tier_pytorch_tensor`:
   Direct `VeriCacheClient.put_tensor` / `get_tensor` over PyTorch CUDA tensors
   hitting Local HBM, Local Pinned DRAM, and Local NVMe (zero host bounce for GPU).
2. `test_vllm_connector_single_node_multilayer`:
   Full vLLM v1 Scheduler + Worker lifecycle (`get_num_new_matched_tokens` ->
   `update_state_after_alloc` -> `build_connector_meta` -> `save_kv_layer` ->
   evict/spill -> new request `start_load_kv` -> `wait_for_layer_load`) on
   multi-layer PyTorch CUDA paged KV tensors `[2, num_blocks, block_size, heads, head_dim]`,
   verifying bit-exact `torch.equal` across HBM, DRAM, and NVMe hits.
3. `test_vllm_connector_2node_prefill_decode_gdr`:
   Two isolated OS processes (Prefill Node on GPU 0, Decode Node on GPU 1) connected
   over real ConnectX RDMA + GPUDirect RDMA. Prefill node runs `VeriCacheConnector`
   to save paged PyTorch CUDA KV blocks; Decode node runs `VeriCacheConnector` to
   load them directly into its own paged PyTorch CUDA KV tensors over 2-GPU GDR,
   asserting `HitLocation.REMOTE_GDR_GPU` and bit-exact `torch.equal`!
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import torch

from vericache import (
    HitLocation,
    PeerSpec,
    TierKind,
    VeriCacheClient,
    key_from_u64,
)
from vericache.vllm_connector import (
    KVConnectorRole,
    VeriCacheConnector,
    hash_prefix_blocks,
    layer_block_key,
)

# Paged KV block geometry: [2, num_blocks, block_size=16, num_kv_heads=4, head_dim=64] in fp16
# Each block = 2 * 16 * 4 * 64 * 2 bytes = 16,384 bytes (16 KiB, 4096-aligned for O_DIRECT NVMe)
BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_DIM = 64
DTYPE = torch.float16
FRAME_SIZE = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2  # 16,384 bytes
NUM_LAYERS = 2


@dataclass
class MockVllmRequest:
    request_id: str
    all_token_ids: list[int]


class MockKVCacheBlocks:
    def __init__(self, block_ids: list[int]) -> None:
        self._ids = block_ids

    def get_block_ids(self) -> list[list[int]]:
        return [self._ids]


def _make_nvme_dir(tag: str) -> Path:
    root = Path(__file__).resolve().parent.parent / "target" / "py-e2e-nvme"
    root.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=root))
    return d


def test_python_client_4tier_pytorch_tensor() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required"
    nvme_dir = _make_nvme_dir("client_tensor")
    try:
        with VeriCacheClient(
            node_id=0,
            frame_size=FRAME_SIZE,
            dram_frames=4,
            gpu_device_id=0,
            hbm_frames=4,
            nvme_dir=str(nvme_dir),
            nvme_frames=4,
        ) as cache:
            dev = torch.device("cuda:0")
            k_gpu = key_from_u64(1)
            k_dram = key_from_u64(2)
            k_nvme = key_from_u64(3)

            t_gpu = torch.randn(FRAME_SIZE // 2, dtype=DTYPE, device=dev)
            t_dram_cpu = torch.randn(FRAME_SIZE // 2, dtype=DTYPE)
            t_nvme_cpu = torch.randn(FRAME_SIZE // 2, dtype=DTYPE)

            v_gpu = cache.put_tensor(k_gpu, t_gpu)
            v_dram = cache.put_tensor(k_dram, t_dram_cpu)
            v_nvme = cache.put_tensor(k_nvme, t_nvme_cpu)
            assert cache.spill_to_nvme(k_nvme)

            # Same `cache.get_tensor(key, dst_cuda)` hits Local HBM, Local DRAM, and Local NVMe!
            dst = torch.empty_like(t_gpu)

            h1 = cache.get_tensor(k_gpu, dst)
            assert h1 is not None and h1.location == HitLocation.LOCAL_GPU
            assert h1.version == v_gpu
            assert torch.equal(dst, t_gpu)

            h2 = cache.get_tensor(k_dram, dst)
            assert h2 is not None and h2.location == HitLocation.LOCAL_DRAM
            assert h2.version == v_dram
            assert torch.equal(dst.cpu(), t_dram_cpu)

            h3 = cache.get_tensor(k_nvme, dst)
            assert h3 is not None and h3.location == HitLocation.LOCAL_NVME
            assert h3.version == v_nvme
            assert torch.equal(dst.cpu(), t_nvme_cpu)
    finally:
        shutil.rmtree(nvme_dir, ignore_errors=True)
    print("PASS test_python_client_4tier_pytorch_tensor")


def test_vllm_connector_single_node_multilayer() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required"
    nvme_dir = _make_nvme_dir("vllm_local")
    try:
        with VeriCacheClient(
            node_id=0,
            frame_size=FRAME_SIZE,
            dram_frames=8,
            gpu_device_id=0,
            hbm_frames=8,
            nvme_dir=str(nvme_dir),
            nvme_frames=8,
        ) as cache:
            dev = torch.device("cuda:0")
            num_vllm_blocks = 16
            kv_caches = {
                f"layer_{i}": torch.zeros(
                    (2, num_vllm_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
                    dtype=DTYPE,
                    device=dev,
                )
                for i in range(NUM_LAYERS)
            }

            conn = VeriCacheConnector(
                vllm_config=None,
                role=KVConnectorRole.WORKER,
                client=cache,
                block_size=BLOCK_SIZE,
                probe_layer_name="layer_0",
            )
            conn.register_kv_caches(kv_caches)

            # Request 1: cold prefill with 3 full blocks (48 tokens) in non-contiguous block_ids [5, 2, 11]
            tokens = list(range(1000, 1000 + 3 * BLOCK_SIZE))
            req1 = MockVllmRequest("req-1", tokens)
            req1_blocks = [5, 2, 11]

            matched, is_async = conn.get_num_new_matched_tokens(req1, num_computed_tokens=0)
            assert matched == 0 and not is_async

            conn.update_state_after_alloc(req1, MockKVCacheBlocks(req1_blocks), matched)
            meta1 = conn.build_connector_meta(scheduler_output=None)
            conn.bind_connector_metadata(meta1)

            # Simulate attention writing KV tensors into req1_blocks [5, 2, 11]
            expected_kv: dict[str, torch.Tensor] = {}
            for lname, kv in kv_caches.items():
                gt = torch.randn((2, 3, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE, device=dev)
                expected_kv[lname] = gt
                for i, bid in enumerate(req1_blocks):
                    kv[:, bid].copy_(gt[:, i])
                conn.save_kv_layer(lname, kv, attn_metadata=None)
            conn.wait_for_save()
            conn.clear_connector_metadata()

            # Force block 1 across all layers out of HBM -> DRAM -> NVMe to test multi-tier reload!
            digests = hash_prefix_blocks(tokens, BLOCK_SIZE)
            for i in range(NUM_LAYERS):
                k_blk1 = layer_block_key(digests[1], f"layer_{i}")
                # Demote to DRAM via get_host, evict from GPU, spill to NVMe
                assert cache.get_host(k_blk1) is not None
                assert cache.evict_from_tier(k_blk1, TierKind.GPU_VRAM)
                assert cache.spill_to_nvme(k_blk1)
                # Also evict block 2 from GPU only (leaving it in DRAM)
                k_blk2 = layer_block_key(digests[2], f"layer_{i}")
                assert cache.get_host(k_blk2) is not None
                assert cache.evict_from_tier(k_blk2, TierKind.GPU_VRAM)

            # Zero vLLM KV cache to guarantee reload comes strictly from VeriCache
            for kv in kv_caches.values():
                kv.zero_()

            # Request 2 shares the exact same 48-token prefix, allocated to brand-new block_ids [13, 0, 9]
            req2 = MockVllmRequest("req-2", tokens)
            req2_blocks = [13, 0, 9]

            matched2, is_async2 = conn.get_num_new_matched_tokens(req2, num_computed_tokens=0)
            assert matched2 == 3 * BLOCK_SIZE and not is_async2

            conn.update_state_after_alloc(req2, MockKVCacheBlocks(req2_blocks), matched2)
            meta2 = conn.build_connector_meta(scheduler_output=None)
            conn.bind_connector_metadata(meta2)

            conn.start_load_kv(forward_context=None)
            for lname, kv in kv_caches.items():
                conn.wait_for_layer_load(lname)
                hits = conn.last_load_hits[lname]
                assert len(hits) == 3
                assert hits[0] is not None and hits[0].location == HitLocation.LOCAL_GPU
                assert hits[1] is not None and hits[1].location == HitLocation.LOCAL_NVME
                assert hits[2] is not None and hits[2].location == HitLocation.LOCAL_DRAM
                for i, bid in enumerate(req2_blocks):
                    assert torch.equal(kv[:, bid], expected_kv[lname][:, i]), (
                        f"Mismatch on {lname} block {i}"
                    )
            conn.clear_connector_metadata()
    finally:
        shutil.rmtree(nvme_dir, ignore_errors=True)
    print("PASS test_vllm_connector_single_node_multilayer (HBM + NVMe + DRAM bit-exact)")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_prefill_or_decode_role(role: str, rdma_port: int, sync_port: int) -> None:
    rdma_dev = os.environ.get("VERICACHE_RDMA_DEVICE", "mlx5_0")
    gid_index = int(os.environ.get("VERICACHE_RDMA_GID_INDEX", "3"))
    nvme_dir = _make_nvme_dir(f"vllm_{role}")

    tokens = list(range(2000, 2000 + 2 * BLOCK_SIZE))
    digests = hash_prefix_blocks(tokens, BLOCK_SIZE)

    try:
        if role == "prefill":
            sync_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sync_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sync_srv.bind(("127.0.0.1", sync_port))
            sync_srv.listen(1)

            with VeriCacheClient(
                node_id=0,
                frame_size=FRAME_SIZE,
                dram_frames=8,
                gpu_device_id=0,
                hbm_frames=8,
                nvme_dir=str(nvme_dir),
                nvme_frames=4,
                rdma_device=rdma_dev,
                rdma_gid_index=gid_index,
                peers=[
                    PeerSpec(
                        peer_node_id=1,
                        tcp_addr=f"127.0.0.1:{rdma_port}",
                        is_listener=True,
                    )
                ],
            ) as cache:
                sync_conn, _ = sync_srv.accept()
                dev = torch.device("cuda:0")
                kv_caches = {
                    f"layer_{i}": torch.zeros(
                        (2, 8, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE, device=dev
                    )
                    for i in range(NUM_LAYERS)
                }
                conn = VeriCacheConnector(
                    vllm_config=None,
                    role=KVConnectorRole.WORKER,
                    client=cache,
                    block_size=BLOCK_SIZE,
                )
                conn.register_kv_caches(kv_caches)

                req = MockVllmRequest("pd-req-1", tokens)
                p_blocks = [3, 6]
                conn.update_state_after_alloc(req, MockKVCacheBlocks(p_blocks), 0)
                conn.bind_connector_metadata(conn.build_connector_meta(None))

                # Deterministic CPU-generated payload moved to GPU 0 so Decode can assert bit-exact equality
                torch.manual_seed(20261123)
                for lname, kv in kv_caches.items():
                    gt = torch.randn((2, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE).to(
                        dev
                    )
                    for i, bid in enumerate(p_blocks):
                        kv[:, bid].copy_(gt[:, i])
                    conn.save_kv_layer(lname, kv, attn_metadata=None)
                conn.wait_for_save()
                conn.clear_connector_metadata()

                # Signal Decode node that Prefill KV is published in GPU 0 HBM.
                # Note: VeriCache's dedicated Rust worker thread automatically polls & serves
                # Decode's GPUDirect RDMA FetchRequests while Python simply waits on sync_conn!
                sync_conn.sendall(b"R")
                done = sync_conn.recv(1)
                assert done == b"D"
                sync_conn.close()
                sync_srv.close()
        else:
            with VeriCacheClient(
                node_id=1,
                frame_size=FRAME_SIZE,
                dram_frames=8,
                gpu_device_id=1,
                hbm_frames=8,
                nvme_dir=str(nvme_dir),
                nvme_frames=4,
                rdma_device=rdma_dev,
                rdma_gid_index=gid_index,
                peers=[
                    PeerSpec(
                        peer_node_id=0,
                        tcp_addr=f"127.0.0.1:{rdma_port}",
                        is_listener=False,
                    )
                ],
            ) as cache:
                sync_conn = socket.create_connection(("127.0.0.1", sync_port), timeout=30)
                ready = sync_conn.recv(1)
                assert ready == b"R"

                dev = torch.device("cuda:1")
                kv_caches = {
                    f"layer_{i}": torch.zeros(
                        (2, 8, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE, device=dev
                    )
                    for i in range(NUM_LAYERS)
                }

                # Simulated cluster directory telling Decode Scheduler that Prefill holds `digests`
                known_remote = {layer_block_key(d, "layer_0") for d in digests}
                conn = VeriCacheConnector(
                    vllm_config=None,
                    role=KVConnectorRole.WORKER,
                    client=cache,
                    block_size=BLOCK_SIZE,
                    remote_lookup_fn=lambda keys: [k in known_remote for k in keys],
                )
                conn.register_kv_caches(kv_caches)

                req = MockVllmRequest("pd-req-1", tokens)
                d_blocks = [7, 1]
                matched, _ = conn.get_num_new_matched_tokens(req, num_computed_tokens=0)
                assert matched == 2 * BLOCK_SIZE

                conn.update_state_after_alloc(req, MockKVCacheBlocks(d_blocks), matched)
                conn.bind_connector_metadata(conn.build_connector_meta(None))

                # Load directly from Prefill GPU 0 HBM -> Decode GPU 1 HBM via 2-GPU GPUDirect RDMA!
                conn.start_load_kv(forward_context=None)

                # Reconstruct expected ground-truth tensors on CPU with the exact same seed
                torch.manual_seed(20261123)
                for lname, kv in kv_caches.items():
                    conn.wait_for_layer_load(lname)
                    hits = conn.last_load_hits[lname]
                    assert len(hits) == 2
                    for h in hits:
                        assert h is not None
                        assert h.location == HitLocation.REMOTE_GDR_GPU, f"expected GDR, got {h}"
                        assert h.peer_node_id == 0
                    gt_cpu = torch.randn((2, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE)
                    for i, bid in enumerate(d_blocks):
                        actual = kv[:, bid].cpu()
                        assert torch.equal(actual, gt_cpu[:, i]), (
                            f"2-GPU GDR KV mismatch on {lname} block {i}"
                        )
                conn.clear_connector_metadata()

                sync_conn.sendall(b"D")
                sync_conn.close()
    finally:
        shutil.rmtree(nvme_dir, ignore_errors=True)


def test_vllm_connector_2node_prefill_decode_gdr() -> None:
    if torch.cuda.device_count() < 2:
        if os.environ.get("VERICACHE_REQUIRE_HW"):
            raise RuntimeError("VERICACHE_REQUIRE_HW set, but fewer than 2 GPUs visible")
        print("SKIP test_vllm_connector_2node_prefill_decode_gdr: need 2 GPUs")
        return

    rdma_port = _find_free_port()
    sync_port = _find_free_port()

    env = os.environ.copy()
    env["VERICACHE_PY_E2E_ROLE"] = "decode"
    env["VERICACHE_PY_E2E_RDMA_PORT"] = str(rdma_port)
    env["VERICACHE_PY_E2E_SYNC_PORT"] = str(sync_port)

    child = subprocess.Popen([sys.executable, __file__], env=env)
    try:
        _run_prefill_or_decode_role("prefill", rdma_port, sync_port)
    finally:
        rc = child.wait(timeout=30)
    assert rc == 0, f"decode child exited with {rc}"
    print("PASS test_vllm_connector_2node_prefill_decode_gdr (Prefill GPU0 -> Decode GPU1 2-GPU GDR bit-exact!)")


if __name__ == "__main__":
    role = os.environ.get("VERICACHE_PY_E2E_ROLE")
    if role == "decode":
        _run_prefill_or_decode_role(
            "decode",
            int(os.environ["VERICACHE_PY_E2E_RDMA_PORT"]),
            int(os.environ["VERICACHE_PY_E2E_SYNC_PORT"]),
        )
    else:
        test_python_client_4tier_pytorch_tensor()
        test_vllm_connector_single_node_multilayer()
        test_vllm_connector_2node_prefill_decode_gdr()
        print("ALL VERICACHE-PYTHON + VLLM KVCONNECTOR E2E TESTS PASSED!")
