"""Unit tests for Scheme A Global Directory & DistributedVeriCacheEngine (CPU-only)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from vericache.global_directory import (
    GlobalDirectoryClient,
    GlobalDirectoryIndex,
    GlobalDirectoryServer,
)
from vericache.hashing import compute_prefix_block_hashes, derive_layer_block_key


def _make_hashes(seed: int, n: int) -> list[bytes]:
    tokens = list(range(seed, seed + n * 16))
    return compute_prefix_block_hashes(tokens, 16, "test-model")


def test_global_directory_local_first_and_remote_fallback():
    idx = GlobalDirectoryIndex()
    layers = ("layer_0", "layer_1")
    hashes = _make_hashes(100, 4)

    s0, s1 = object(), object()
    idx.register_worker(0, "inc-0", 0, 1, layers, session=s0)
    idx.register_worker(1, "inc-1", 0, 1, layers, session=s1)

    # Instance 0 has blocks 0, 1, 2 completely; block 3 only has layer_0 (incomplete)
    saved0 = [(hashes[i], l) for i in range(3) for l in layers]
    saved0.append((hashes[3], "layer_0"))
    idx.mark_saved_batch(0, "inc-0", 0, saved0, session=s0)

    # Instance 1 has block 0 locally
    idx.mark_saved_batch(1, "inc-1", 0, [(hashes[0], l) for l in layers], session=s1)

    # Query from Instance 1: block 0 should be local, blocks 1 & 2 remote on 0, block 3 stops (incomplete)
    plan = idx.query_prefix_plan(1, hashes, 1, layers, allow_remote=True)
    assert plan == [(1, False), (0, True), (0, True)]

    # With allow_remote=False: only block 0 matches
    local_plan = idx.query_prefix_plan(1, hashes, 1, layers, allow_remote=False)
    assert local_plan == [(1, False)]


def test_global_directory_eviction_and_disconnect_revocation():
    idx = GlobalDirectoryIndex()
    layers = ("layer_0", "layer_1")
    hashes = _make_hashes(200, 3)

    s0, s1 = object(), object()
    idx.register_worker(0, "inc-0", 0, 1, layers, session=s0)
    idx.register_worker(1, "inc-1", 0, 1, layers, session=s1)

    saved0 = [(hashes[i], l, derive_layer_block_key(hashes[i], l)) for i in range(3) for l in layers]
    idx.mark_saved_batch(0, "inc-0", 0, saved0, session=s0)

    assert idx.query_prefix_plan(1, hashes, 1, layers) == [(0, True), (0, True), (0, True)]

    # Evicting one layer of block 1 immediately truncates prefix at block 1
    ev_key = derive_layer_block_key(hashes[1], "layer_1")
    idx.mark_saved_batch(0, "inc-0", 0, (), evicted=(ev_key,), session=s0)
    assert idx.query_prefix_plan(1, hashes, 1, layers) == [(0, True)]

    # Re-registering instance 0 with new incarnation revokes all old blocks
    s0_new = object()
    idx.register_worker(0, "inc-0-v2", 0, 1, layers, session=s0_new)
    assert idx.query_prefix_plan(1, hashes, 1, layers) == []

    # Save again under new incarnation, then simulate abrupt socket disconnect
    idx.mark_saved_batch(0, "inc-0-v2", 0, saved0[:2], session=s0_new)
    assert idx.query_prefix_plan(1, hashes, 1, layers) == [(0, True)]
    idx.disconnect(s0_new)
    assert idx.query_prefix_plan(1, hashes, 1, layers) == []
    st = idx.stats()
    assert st["revocations_incarnation"] == 1
    assert st["revocations_disconnect"] == 1


def test_global_directory_tcp_roundtrip_and_disconnect():
    layers = ("l0", "l1")
    hashes = _make_hashes(300, 2)
    with GlobalDirectoryServer("tcp://127.0.0.1:0") as srv:
        c0 = GlobalDirectoryClient(srv.endpoint)
        c1 = GlobalDirectoryClient(srv.endpoint)
        try:
            assert c0.ping()
            c0.register_worker(0, "a1", 0, 1, layers)
            c1.register_worker(1, "b1", 0, 1, layers)
            c0.mark_saved_batch(
                0,
                "a1",
                0,
                [(h, l, derive_layer_block_key(h, l)) for h in hashes for l in layers],
            )
            plan = c1.query_prefix_plan(1, hashes, 1, layers)
            assert plan == [(0, True), (0, True)]

            # Abrupt close of c0 revokes instance 0 entries on server
            c0.close()
            # Give server thread a brief moment to process EOF
            import time

            for _ in range(50):
                if c1.query_prefix_plan(1, hashes, 1, layers) == []:
                    break
                time.sleep(0.01)
            assert c1.query_prefix_plan(1, hashes, 1, layers) == []
        finally:
            c0.close()
            c1.close()


def test_distributed_engine_local_put_lease_evict_subprocess():
    """Run DistributedVeriCacheEngine in an isolated subprocess (fresh NodeContext)."""
    code = r"""
import ctypes, tempfile
from vericache import DistributedVeriCacheEngine, VeriCacheConfig, key_from_u64

cfg = VeriCacheConfig(
    distributed_mode=True,
    node_id=0,
    num_shards=2,
    host_frames_per_shard=8,
    nvme_frames_per_shard=0,
    frame_size=4096,
    prefer_cuda_pinned=False,
)
with DistributedVeriCacheEngine(cfg) as eng:
    with eng.make_staging(4096 * 4) as stage:
        assert stage.size == 4096 * 4
    keys = [key_from_u64(i) for i in range(6)]
    bufs = [bytes([(i * 17 + j) % 251 for j in range(4096)]) for i in range(6)]
    c_bufs = [(ctypes.c_uint8 * 4096).from_buffer_copy(b) for b in bufs]
    statuses = eng.put_batch(keys, [ctypes.addressof(b) for b in c_bufs], [4096] * 6)
    assert statuses == [0] * 6
    assert eng.contains_batch(keys) == [True] * 6

    leases, lst = eng.acquire_read_lease_batch(keys, with_statuses=True)
    assert lst == [0] * 6
    for i, lease in enumerate(leases):
        assert lease is not None
        got = bytes(lease.memoryview)
        assert got == bufs[i]
    eng.release_read_lease_batch([l.lease_id for l in leases])

    # Shrink DRAM budget to 1 frame/shard -> forces eviction and reports evicted keys
    ev_n = eng.set_dram_budget_frames(1)
    assert ev_n >= 4
    evicted = eng.drain_evictions()
    assert len(evicted) >= 4
    st = eng.rdma_stats()
    assert st["puts"] == 6
    assert st["dram_hits"] == 6
    assert st["dram_evictions"] >= 4
print("DIST_ENGINE_OK")
"""
    lib_dir = Path(__file__).resolve().parent.parent / "target" / "release"
    cuda_dir = Path(__file__).resolve().parent.parent / "target" / "cuda-link" / "lib"
    env = os.environ.copy()
    env["VERICACHE_LIB"] = str(lib_dir / "libvericache_python.so")
    ld = env.get("LD_LIBRARY_PATH", "")
    extra = [
        str(cuda_dir),
        "/home/handshaker/miniconda3/lib/python3.13/site-packages/nvidia/cuda_runtime/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(extra + ([ld] if ld else []))
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "DIST_ENGINE_OK" in res.stdout
