#!/usr/bin/env python3
"""Official LMCache 0.5.7.post1 baseline runner for Suites 1 and 3."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import socket
import struct
import sys
import threading
import time
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    LatencyRecorder,
    LruTracker,
    MetricRecord,
    ProcSampler,
    bytes_per_sec_to_gbps,
    load_trace,
    route_request_scaleup,
    scaleup_config_from_job,
)


def _make_kv_shape_for_chunk(chunk_bytes: int, num_blocks: int = 16) -> tuple[int, int, int, int, int]:
    """Build a valid KV_2LTD shape (num_layers, 2, chunk_tokens, num_heads, head_size)
    with dtype=bfloat16 (2 bytes/elem) whose total byte size == chunk_bytes and
    chunk_tokens == num_blocks (so each token slot corresponds cleanly to 1 paged block!)."""
    # Total elements (bfloat16 = 2 bytes): chunk_bytes // 2
    # Shape: (num_layers=1, kv=2, num_tokens=num_blocks, num_heads=1, head_size=elems_per_slot)
    # 1 * 2 * num_blocks * 1 * head_size * 2 == chunk_bytes  =>  head_size = chunk_bytes // (4 * num_blocks)
    assert chunk_bytes % (4 * num_blocks) == 0, (
        f"chunk_bytes {chunk_bytes} not divisible by 4*num_blocks={4 * num_blocks}"
    )
    head_size = chunk_bytes // (4 * num_blocks)
    return (1, 2, num_blocks, 1, head_size)


def _create_local_cpu_backend(
    chunk_bytes: int,
    max_cpu_gb: float,
    num_blocks: int = 16,
    dst_device: str = "cuda:0",
):
    from lmcache.inprocess.v1.config import LMCacheEngineConfig
    from lmcache.inprocess.v1.metadata import LMCacheMetadata
    from lmcache.inprocess.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    kv_shape = _make_kv_shape_for_chunk(chunk_bytes, num_blocks)
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=num_blocks,
        local_cpu=True,
        max_local_cpu_size=max_cpu_gb,
    )
    meta = LMCacheMetadata(
        model_name="vericache-bench",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        chunk_size=num_blocks,
    )
    backend = LocalCPUBackend(config=cfg, metadata=meta, dst_device=dst_device)
    return cfg, meta, backend, kv_shape


# =========================================================================
# Suite 1A: LMCache cuda_ops.multi_layer_kv_transfer (16 non-contiguous GPU Paged KV blocks
#           <-> LocalCPUBackend pinned MemoryObj + torch.cuda.Event)
# =========================================================================
def _run_1a_gpu_scatter(job: dict[str, Any], is_d2h: bool) -> list[str]:
    import lmcache.cuda_ops as cuda_ops
    import lmcache.lmcache_native as lmcache_native
    from lmcache.v1.memory_management import MemoryFormat

    frame_size = int(job["chunk_bytes"])
    num_blocks = max(1, int(job.get("num_blocks", 16)))
    warmup = int(job.get("warmup_iters", 8))
    iters = int(job.get("iters", 48))
    gpu_id = int(job.get("donor_gpu", 0))

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Allocate pinned host MemoryObj via official LocalCPUBackend
    max_cpu_gb = max(0.25, (frame_size * 8) / (1024**3))
    _cfg, _meta, cpu_backend, kv_shape = _create_local_cpu_backend(
        frame_size, max_cpu_gb, num_blocks=num_blocks, dst_device=f"cuda:{gpu_id}"
    )
    # kv_shape = (1, 2, num_blocks, 1, head_size), KV_2LTD memory_obj shape = (2, num_layers=1, num_tokens=num_blocks, hidden=head_size)
    head_size = kv_shape[4]
    mem_shape = torch.Size([2, 1, num_blocks, head_size])
    mem_obj = cpu_backend.allocate(mem_shape, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
    if mem_obj is None:
        raise RuntimeError("LocalCPUBackend.allocate returned None")

    # Build a paged GPU KV cache with 4x total pages and pick 16 non-contiguous strided slots!
    # Standard vLLM paged layout per layer: [2, num_pages, block_size=1, num_heads=1, head_size]
    num_pages = num_blocks * 4
    kvcaches = [
        torch.randn((2, num_pages, 1, 1, head_size), dtype=torch.bfloat16, device=device)
    ]
    kv_ptrs = torch.tensor(
        [t.data_ptr() for t in kvcaches], dtype=torch.int64, device=device
    )
    # Non-contiguous strided slot_mapping across the page table (stride 4)
    slot_mapping = torch.tensor(
        [i * 4 + (i % 3) for i in range(num_blocks)], dtype=torch.int64, device=device
    )
    page_buffer_size = num_pages  # block_size = 1 => page_buffer_size = num_pages * 1
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    direction = (
        lmcache_native.TransferDirection.D2H
        if is_d2h
        else lmcache_native.TransferDirection.H2D
    )

    stream = torch.cuda.Stream(device=device)
    done_event = torch.cuda.Event(enable_timing=False)
    torch.cuda.synchronize(device)

    total_iters = warmup + iters
    lat = LatencyRecorder()
    sampler = None
    t_meas_start = 0.0

    for it in range(total_iters):
        if it == warmup:
            sampler = ProcSampler()
            t_meas_start = time.perf_counter()
        t0 = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            cuda_ops.multi_layer_kv_transfer(
                mem_obj.tensor,
                kv_ptrs,
                slot_mapping,
                device,
                page_buffer_size,
                direction,
                engine_kv_format,
                block_size=1,
                head_size=head_size,
            )
            done_event.record(stream)
        done_event.synchronize()
        dt_ns = time.perf_counter_ns() - t0
        if it >= warmup:
            lat.record_ns(dt_ns)

    elapsed_s = time.perf_counter() - t_meas_start
    cpu_pct, rss_mb = sampler.finish()
    gbps = bytes_per_sec_to_gbps(iters * frame_size, elapsed_s)
    p50, p95, p99 = lat.percentiles_us()

    mem_obj.ref_count_down()
    cpu_backend.close()

    rec = MetricRecord.from_job("lmcache", job)
    rec.throughput_gbps = gbps
    rec.goodput_gbps = gbps
    rec.p50_us = p50
    rec.p95_us = p95
    rec.p99_us = p99
    rec.hit_rate = 1.0
    rec.cpu_pct = cpu_pct
    rec.rss_mb = rss_mb
    dir_str = "D2H" if is_d2h else "H2D"
    rec.notes = (
        f"LMCache 0.5.7.post1 cuda_ops.multi_layer_kv_transfer ({dir_str}, {num_blocks} non-contiguous paged slots "
        f"<-> LocalCPUBackend pinned MemoryObj + torch.cuda.Event)"
    )
    return [rec.to_json()]


# =========================================================================
# Suite 1B: LMCache LocalDiskBackend on /workspace/target/bench-nvme (O_DIRECT)
# =========================================================================
def _run_1b_disk(job: dict[str, Any], is_write: bool) -> list[str]:
    from lmcache.inprocess.v1.config import LMCacheEngineConfig
    from lmcache.inprocess.v1.metadata import LMCacheMetadata
    from lmcache.inprocess.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.inprocess.v1.storage_backend.local_disk_backend import LocalDiskBackend
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryFormat

    frame_size = int(job["chunk_bytes"])
    qd = max(1, int(job.get("concurrency", 1)))
    warmup = int(job.get("warmup_iters", 4))
    iters = int(job.get("iters", 32))
    nvme_base = pathlib.Path(str(job.get("nvme_dir", "/workspace/target/bench-nvme")))
    disk_dir = nvme_base / f"lmcache_{os.getpid()}_{frame_size}"
    if disk_dir.exists():
        shutil.rmtree(disk_dir, ignore_errors=True)
    disk_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    try:
        kv_shape = _make_kv_shape_for_chunk(frame_size, num_blocks=16)
        head_size = kv_shape[4]
        mem_shape = torch.Size([2, 1, 16, head_size])
        max_cpu_gb = max(0.5, (frame_size * (qd + 8)) / (1024**3))
        max_disk_gb = max(2.0, (frame_size * (qd + 16)) / (1024**3))
        cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            local_cpu=True,
            max_local_cpu_size=max_cpu_gb,
            local_disk=str(disk_dir),
            max_local_disk_size=max_disk_gb,
            extra_config={"use_odirect": True, "disk_io_threads": max(1, qd)},
        )
        meta = LMCacheMetadata(
            model_name="vericache-bench",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=kv_shape,
            chunk_size=16,
        )
        cpu_backend = LocalCPUBackend(config=cfg, metadata=meta, dst_device="cpu")
        disk_backend = LocalDiskBackend(
            config=cfg,
            loop=loop,
            local_cpu_backend=cpu_backend,
            dst_device="cpu",
            metadata=meta,
        )

        # Pre-allocate `qd` MemoryObjs for writes (or seed `qd` keys on disk for reads)
        mem_objs = []
        keys = []
        for i in range(qd):
            mo = cpu_backend.allocate(mem_shape, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
            if mo is None:
                raise RuntimeError("LocalCPUBackend.allocate failed for disk bench")
            mo.tensor.fill_(1.0)
            mem_objs.append(mo)
            keys.append(CacheEngineKey("vericache-bench", 1, 0, i + 1, torch.bfloat16))

        def _put_batch_and_wait(b_keys, b_objs) -> None:
            remaining = [len(b_keys)]
            lock = threading.Lock()
            done_ev = threading.Event()

            def _on_done(_k):
                with lock:
                    remaining[0] -= 1
                    if remaining[0] <= 0:
                        done_ev.set()

            for k, mo in zip(b_keys, b_objs):
                disk_backend.submit_put_task(k, mo, on_complete_callback=_on_done)
            if not done_ev.wait(timeout=60.0):
                raise RuntimeError(
                    f"LocalDiskBackend.submit_put_task timed out (remaining={remaining[0]})"
                )

        if not is_write:
            # Seed `qd` keys to disk and wait for async PQ completion, then clear CPU cache
            _put_batch_and_wait(keys, mem_objs)
            for mo in mem_objs:
                mo.ref_count_down()
            mem_objs.clear()
            cpu_backend.clear()

        warmup_batches = (warmup + qd - 1) // qd
        meas_batches = (iters + qd - 1) // qd
        total_batches = warmup_batches + meas_batches

        lat = LatencyRecorder()
        sampler = None
        t_meas_start = 0.0
        total_ops = 0

        for b in range(total_batches):
            if b == warmup_batches:
                sampler = ProcSampler()
                t_meas_start = time.perf_counter()

            t0 = time.perf_counter_ns()
            if is_write:
                batch_keys = [
                    CacheEngineKey("vericache-bench", 1, 0, b * qd + i + 1000, torch.bfloat16)
                    for i in range(qd)
                ]
                _put_batch_and_wait(batch_keys, mem_objs)
                dt_ns = time.perf_counter_ns() - t0
                # Remove written keys so disk footprint stays bounded
                for k in batch_keys:
                    disk_backend.remove(k)
            else:
                if qd == 1:
                    mo = disk_backend.get_blocking(keys[0])
                    if mo is None:
                        raise RuntimeError("LocalDiskBackend.get_blocking missed seeded key")
                    mo.ref_count_down()
                else:
                    mos = disk_backend.batched_get_blocking(keys)
                    for mo in mos:
                        if mo is None:
                            raise RuntimeError("LocalDiskBackend.batched_get_blocking missed")
                        mo.ref_count_down()
                dt_ns = time.perf_counter_ns() - t0

            if b >= warmup_batches:
                lat.record_ns(dt_ns)
                total_ops += qd

        elapsed_s = time.perf_counter() - t_meas_start
        cpu_pct, rss_mb = sampler.finish()
        gbps = bytes_per_sec_to_gbps(total_ops * frame_size, elapsed_s)
        p50, p95, p99 = lat.percentiles_us()

        for mo in mem_objs:
            mo.ref_count_down()
        disk_backend.close()
        cpu_backend.close()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        shutil.rmtree(disk_dir, ignore_errors=True)

    rec = MetricRecord.from_job("lmcache", job)
    rec.throughput_gbps = gbps
    rec.goodput_gbps = gbps
    rec.p50_us = p50
    rec.p95_us = p95
    rec.p99_us = p99
    rec.hit_rate = 1.0
    rec.cpu_pct = cpu_pct
    rec.rss_mb = rss_mb
    op_desc = "submit_put_task (async PQ worker + O_DIRECT)" if is_write else "get_blocking (O_DIRECT)"
    rec.notes = f"LMCache 0.5.7.post1 LocalDiskBackend {op_desc} (QD={qd}, dir={nvme_base})"
    return [rec.to_json()]


# =========================================================================
# Suite 3: Official LMCache LocalCPUBackend Trace Replay (LRU + prefix match)
# =========================================================================
def run_suite3(job: dict[str, Any]) -> list[str]:
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryFormat

    trace_path = str(job["trace_path"])
    max_requests = job.get("max_requests")
    reqs = load_trace(trace_path, max_requests)
    unique_blocks = set()
    total_blocks_in_trace = 0
    for r in reqs:
        for h in r["hash_ids"]:
            unique_blocks.add(h)
        total_blocks_in_trace += len(r["hash_ids"])

    frame_size = int(job["chunk_bytes"])
    ratio = float(job.get("capacity_ratio", 0.30))
    if job.get("capacity_frames") is not None:
        cap_frames = int(job["capacity_frames"])
    else:
        target = int(round(len(unique_blocks) * ratio))
        max_by_mem = (2 * 1024 * 1024 * 1024) // frame_size
        cap_frames = max(16, min(target, max(16, max_by_mem)))

    # Size LocalCPUBackend to hold `cap_frames` chunks (+ small allocator overhead headroom)
    # and enforce exact `cap_frames` LRU budget via built-in LocalCPUBackend LRU + explicit count cap.
    alloc_bytes = max(frame_size * (cap_frames + 4), 16 * 1024 * 1024)
    max_cpu_gb = alloc_bytes / (1024**3)
    _cfg, _meta, cpu_backend, kv_shape = _create_local_cpu_backend(
        frame_size, max_cpu_gb, num_blocks=16, dst_device="cpu"
    )
    head_size = kv_shape[4]
    mem_shape = torch.Size([2, 1, 16, head_size])

    # Pre-allocated caller read/write scratch buffers
    read_dst = torch.empty(mem_shape, dtype=torch.bfloat16)
    write_src = torch.ones(mem_shape, dtype=torch.bfloat16)

    is_open_loop = str(job["mode"]) == "open_loop"
    time_scale = max(1.0, float(job.get("time_scale", 1000.0)))
    base_ts = int(reqs[0]["timestamp"])

    lat = LatencyRecorder()
    sampler = ProcSampler()
    t_start = time.perf_counter()
    hit_blocks = 0
    miss_blocks = 0

    # Track exact residency count matching cap_frames using LocalCPUBackend.hot_cache (OrderedDict LRU)
    for req in reqs:
        if is_open_loop:
            rel_s = max(0.0, (int(req["timestamp"]) - base_ts) / 1000.0) / time_scale
            now_off = time.perf_counter() - t_start
            if rel_s > now_off:
                wait = rel_s - now_off
                if wait > 0.0001:
                    time.sleep(wait)

        req_t0 = time.perf_counter_ns()
        prefix_open = True
        for hid in req["hash_ids"]:
            ckey = CacheEngineKey("vericache-bench", 1, 0, int(hid), torch.bfloat16)
            if prefix_open and cpu_backend.contains(ckey):
                mo = cpu_backend.get_blocking(ckey)
                if mo is not None:
                    read_dst.copy_(mo.tensor)
                    mo.ref_count_down()
                    hit_blocks += 1
                    continue
            prefix_open = False
            miss_blocks += 1
            if cpu_backend.contains(ckey):
                continue
            # Enforce exact cap_frames capacity before allocate+submit_put_task
            while len(cpu_backend.hot_cache) >= cap_frames:
                oldest_key = next(iter(cpu_backend.hot_cache))
                cpu_backend.remove(oldest_key)
            mo = cpu_backend.allocate(mem_shape, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
            if mo is None:
                # Evict one more if fragmentation triggered
                if cpu_backend.hot_cache:
                    cpu_backend.remove(next(iter(cpu_backend.hot_cache)))
                mo = cpu_backend.allocate(mem_shape, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
            if mo is not None:
                mo.tensor.copy_(write_src)
                cpu_backend.submit_put_task(ckey, mo)
                mo.ref_count_down()

        lat.record_ns(time.perf_counter_ns() - req_t0)

    elapsed_s = time.perf_counter() - t_start
    cpu_pct, rss_mb = sampler.finish()
    cpu_backend.clear()
    cpu_backend.close()

    total_blocks = hit_blocks + miss_blocks
    hit_rate = (hit_blocks / total_blocks) if total_blocks > 0 else 0.0
    total_bytes = total_blocks * frame_size
    hit_bytes = hit_blocks * frame_size
    gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s)
    hit_gbps = bytes_per_sec_to_gbps(hit_bytes, elapsed_s)
    p50, p95, p99 = lat.percentiles_us()

    rec = MetricRecord.from_job("lmcache", job)
    rec.num_blocks = cap_frames
    rec.throughput_gbps = gbps
    rec.goodput_gbps = gbps
    rec.p50_us = p50
    rec.p95_us = p95
    rec.p99_us = p99
    rec.hit_rate = hit_rate
    rec.cpu_pct = cpu_pct
    rec.rss_mb = rss_mb
    rec.notes = (
        f"LMCache 0.5.7.post1 LocalCPUBackend (pinned LRU), reqs={len(reqs)}, "
        f"unique_blks={len(unique_blocks)}, total_blks={total_blocks_in_trace}, "
        f"cap_frames={cap_frames} (ratio={ratio:.2f}), hit_only_gbps={hit_gbps:.2f}"
    )
    return [rec.to_json()]


# =========================================================================
# Suite 5 (Scale Wave 2): 600 GiB 2-Node 8-Shard 4-Tier LMCache Cluster Replay
# =========================================================================
def _lmc_recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(mv[got:], n - got)
        if r == 0:
            raise EOFError("peer socket closed")
        got += r
    return bytes(buf)


def _lmc_s5_node_proc(node_id: int, job: dict[str, Any], ctrl_conn) -> None:
    from lmcache.inprocess.v1.config import LMCacheEngineConfig
    from lmcache.inprocess.v1.metadata import LMCacheMetadata
    from lmcache.inprocess.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.inprocess.v1.storage_backend.local_disk_backend import LocalDiskBackend
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryFormat
    from mooncake.engine import TransferEngine

    cfg = scaleup_config_from_job(job)
    frame_size = cfg["frame_size"]
    shards = cfg["shards_per_node"]
    hbm_f = cfg["hbm_frames_per_shard"]
    dram_f = cfg["dram_frames_per_shard"]
    nvme_f = cfg["nvme_frames_per_shard"]
    sticky_pct = cfg["sticky_pct"]
    gpu_id = int(job.get("donor_gpu", 0) if node_id == 0 else job.get("initiator_gpu", 1))
    rdma_dev = str(job.get("rdma_dev", "mlx5_3"))
    nvme_key = f"nvme_dir_node{node_id}"
    nvme_base = pathlib.Path(str(job.get(nvme_key) or job.get("nvme_dir", "/workspace/target/bench-nvme")))
    nvme_base.mkdir(parents=True, exist_ok=True)
    node_disk_root = nvme_base / f"lmcache_s5_{os.getpid()}_n{node_id}"
    if node_disk_root.exists():
        shutil.rmtree(node_disk_root, ignore_errors=True)
    node_disk_root.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    kv_shape = _make_kv_shape_for_chunk(frame_size, num_blocks=16)
    head_size = kv_shape[4]
    mem_shape = torch.Size([2, 1, 16, head_size])

    reqs = load_trace(str(job["trace_path"]), job.get("max_requests"))
    per_shard_reqs: list[list[list[int]]] = [[] for _ in range(shards)]
    for req_idx, r in enumerate(reqs):
        hids = r["hash_ids"]
        t_node, sid = route_request_scaleup(req_idx, hids, shards, sticky_pct)
        if t_node == node_id:
            per_shard_reqs[sid].append(hids)

    req_socks: list[socket.socket] = [None] * shards  # type: ignore
    srv_socks: list[socket.socket] = [None] * shards  # type: ignore
    if node_id == 0:
        listeners = []
        for _ in range(shards):
            ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ls.bind(("127.0.0.1", 0))
            ls.listen(2)
            listeners.append(ls)
        ctrl_conn.send({"ports": [ls.getsockname()[1] for ls in listeners]})
        for s in range(shards):
            c1, _ = listeners[s].accept()
            c2, _ = listeners[s].accept()
            c1.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c2.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            tag1 = _lmc_recv_exact(c1, 1)
            _ = _lmc_recv_exact(c2, 1)
            if tag1 == b"A":
                req_socks[s], srv_socks[s] = c1, c2
            else:
                req_socks[s], srv_socks[s] = c2, c1
            listeners[s].close()
    else:
        ports = ctrl_conn.recv()["ports"]
        for s in range(shards):
            c1 = socket.create_connection(("127.0.0.1", ports[s]), timeout=60.0)
            c1.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c1.sendall(b"A")
            c2 = socket.create_connection(("127.0.0.1", ports[s]), timeout=60.0)
            c2.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c2.sendall(b"B")
            srv_socks[s], req_socks[s] = c1, c2

    init_done_barrier = threading.Barrier(shards + 1)
    start_go_barrier = threading.Barrier(shards + 1)
    shard_results: list[dict[str, Any]] = [{} for _ in range(shards)]
    shard_errors: list[BaseException | None] = [None] * shards

    def _shard_worker(s: int) -> None:
        te = None
        hbm_slab = None
        peer_stage = None
        hbm_ptr = 0
        stage_ptr = 0
        cpu_backend = None
        disk_backend = None
        loop = None
        loop_thread = None
        disk_dir = node_disk_root / f"shard_{s}"
        disk_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.cuda.set_device(gpu_id)
            t_init0 = time.perf_counter()

            # L1 GPU HBM slab + cross-node TransferEngine
            te = TransferEngine()
            rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", "rdma", rdma_dev)
            if rc != 0:
                raise RuntimeError(f"LMCache S5 node{node_id} shard{s} TE.initialize failed rc={rc}")
            seg_name = f"127.0.0.1:{te.get_rpc_port()}"

            hbm_bytes = hbm_f * frame_size
            hbm_slab = torch.empty((hbm_f, frame_size), dtype=torch.uint8, device=device)
            hbm_ptr = int(hbm_slab.data_ptr())
            rc_h = te.register_memory(hbm_ptr, hbm_bytes)
            if rc_h != 0:
                raise RuntimeError(
                    f"LMCache S5 node{node_id} shard{s} register HBM ({hbm_bytes} B) failed rc={rc_h}"
                )

            # Registered 2 MiB pinned staging buffer for serving LocalCPUBackend hits over RDMA
            peer_stage = torch.empty((frame_size,), dtype=torch.uint8, pin_memory=True)
            stage_ptr = int(peer_stage.data_ptr())
            rc_st = te.register_memory(stage_ptr, frame_size)
            if rc_st != 0:
                raise RuntimeError(f"LMCache S5 register peer_stage failed rc={rc_st}")

            # L2 Official LocalCPUBackend (25 GiB pinned DRAM per shard + small allocator headroom)
            # L3 Official LocalDiskBackend (6.25 GiB O_DIRECT NVMe per shard)
            max_cpu_gb = (frame_size * (dram_f + 4)) / (1024.0**3)
            max_disk_gb = (frame_size * (nvme_f + 4)) / (1024.0**3)
            lmc_cfg = LMCacheEngineConfig.from_defaults(
                chunk_size=16,
                local_cpu=True,
                max_local_cpu_size=max_cpu_gb,
                local_disk=str(disk_dir),
                max_local_disk_size=max_disk_gb,
                extra_config={"use_odirect": True, "disk_io_threads": 2},
            )
            meta = LMCacheMetadata(
                model_name=f"vericache-s5-n{node_id}-s{s}",
                world_size=1,
                local_world_size=1,
                worker_id=0,
                local_worker_id=0,
                kv_dtype=torch.bfloat16,
                kv_shape=kv_shape,
                chunk_size=16,
            )
            loop = asyncio.new_event_loop()
            loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
            loop_thread.start()

            cpu_backend = LocalCPUBackend(config=lmc_cfg, metadata=meta, dst_device=f"cuda:{gpu_id}")
            disk_backend = LocalDiskBackend(
                config=lmc_cfg,
                loop=loop,
                local_cpu_backend=cpu_backend,
                dst_device="cpu",
                metadata=meta,
            )

            # Exchange TransferEngine segment names with peer shard
            seg_b = seg_name.encode("utf-8")
            req_socks[s].sendall(struct.pack("!I", len(seg_b)) + seg_b)
            plen = struct.unpack("!I", _lmc_recv_exact(srv_socks[s], 4))[0]
            peer_seg = _lmc_recv_exact(srv_socks[s], plen).decode("utf-8")

            compute_slot = torch.empty((frame_size,), dtype=torch.uint8, device=device)
            cold_src = torch.empty((frame_size,), dtype=torch.uint8, pin_memory=True)
            cold_src.fill_(0x5A ^ (s & 0xFF))
            stream = torch.cuda.Stream(device=device)
            torch.cuda.synchronize(device)

            init_s = time.perf_counter() - t_init0

            lock = threading.Lock()
            cond = threading.Condition(lock)
            hbm_lru = LruTracker(hbm_f)
            dram_lru = LruTracker(dram_f)
            nvme_lru = LruTracker(nvme_f)
            free_hbm = list(range(hbm_f))
            pinned_hbm: dict[int, int] = {}

            def _mk_key(hid: int) -> CacheEngineKey:
                return CacheEngineKey(f"vericache-s5-n{node_id}-s{s}", 1, 0, int(hid), torch.bfloat16)

            def _disk_put_sync(ckey: CacheEngineKey, mo) -> None:
                done_ev = threading.Event()
                disk_backend.submit_put_task(ckey, mo, on_complete_callback=lambda _k: done_ev.set())
                if not done_ev.wait(timeout=60.0):
                    raise RuntimeError(f"LMCache S5 disk submit_put_task timed out (shard={s})")

            # Background peer responder thread serving HBM (direct GDR) or LocalCPUBackend (via stage)
            def _responder() -> None:
                rs = srv_socks[s]
                stage_view = peer_stage.view(torch.bfloat16).view(mem_shape)
                while True:
                    hdr = _lmc_recv_exact(rs, 9)
                    op = hdr[0]
                    hid = struct.unpack("!Q", hdr[1:9])[0]
                    if op == 1:  # QUERY
                        hbm_slot_found = None
                        in_dram = False
                        with lock:
                            if hid in hbm_lru:
                                hbm_slot_found = hbm_lru._od[hid]
                                pinned_hbm[hbm_slot_found] = pinned_hbm.get(hbm_slot_found, 0) + 1
                            elif hid in dram_lru:
                                in_dram = True
                        if hbm_slot_found is not None:
                            addr = hbm_ptr + hbm_slot_found * frame_size
                            rs.sendall(struct.pack("!BQQ", 1, addr, hbm_slot_found))
                        elif in_dram:
                            mo = cpu_backend.get_blocking(_mk_key(hid))
                            if mo is not None:
                                stage_view.copy_(mo.tensor)
                                mo.ref_count_down()
                                rs.sendall(struct.pack("!BQQ", 2, stage_ptr, 0))
                            else:
                                rs.sendall(struct.pack("!BQQ", 0, 0, 0))
                        else:
                            rs.sendall(struct.pack("!BQQ", 0, 0, 0))
                    elif op == 2:  # RELEASE
                        tier_tag = hid >> 56
                        slot = hid & 0x00FFFFFFFFFFFFFF
                        if tier_tag == 1:
                            with cond:
                                cnt = pinned_hbm.get(slot, 1) - 1
                                if cnt <= 0:
                                    pinned_hbm.pop(slot, None)
                                else:
                                    pinned_hbm[slot] = cnt
                                cond.notify_all()
                        rs.sendall(b"K")
                    elif op == 3:  # DONE
                        rs.sendall(b"D")
                        break

            resp_t = threading.Thread(target=_responder, daemon=True)
            resp_t.start()

            init_done_barrier.wait()
            start_go_barrier.wait()

            def _evict_dram_victim_to_nvme() -> None:
                with lock:
                    vic = dram_lru.pop_lru()
                if vic is None:
                    return
                dk, _ = vic
                dkey = _mk_key(dk)
                mo = cpu_backend.get_blocking(dkey)
                if mo is not None:
                    # Enforce NVMe capacity before spilling to disk
                    with lock:
                        if nvme_lru.is_full():
                            nvic = nvme_lru.pop_lru()
                            if nvic is not None:
                                disk_backend.remove(_mk_key(nvic[0]))
                    _disk_put_sync(dkey, mo)
                    mo.ref_count_down()
                    with lock:
                        nvme_lru.insert(dk, True)
                cpu_backend.remove(dkey)

            def _ensure_hbm_slot() -> int:
                with cond:
                    if free_hbm:
                        return free_hbm.pop()
                    vk, hslot = hbm_lru.pop_lru()
                    while pinned_hbm.get(hslot, 0) > 0:
                        cond.wait(timeout=0.001)
                # Demote vk from GPU HBM -> LocalCPUBackend (cascading DRAM -> LocalDiskBackend if full)
                vkey = _mk_key(vk)
                with lock:
                    if vk in dram_lru:
                        dram_lru.remove(vk)
                        cpu_backend.remove(vkey)
                while len(dram_lru) >= dram_f:
                    _evict_dram_victim_to_nvme()
                mo = cpu_backend.allocate(mem_shape, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
                if mo is None:
                    _evict_dram_victim_to_nvme()
                    mo = cpu_backend.allocate(mem_shape, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
                if mo is not None:
                    dst_u8 = mo.tensor.view(torch.uint8).view(-1)
                    with torch.cuda.stream(stream):
                        dst_u8.copy_(hbm_slab[hslot], non_blocking=True)
                    stream.synchronize()
                    cpu_backend.submit_put_task(vkey, mo)
                    mo.ref_count_down()
                    with lock:
                        dram_lru.insert(vk, True)
                return hslot

            lat_samples: list[int] = []
            hbm_hits = 0
            dram_hits = 0
            nvme_hits = 0
            remote_hits = 0
            cold_misses = 0
            rsock = req_socks[s]

            for hids in per_shard_reqs[s]:
                req_t0 = time.perf_counter_ns()
                prefix_open = True
                for hid in hids:
                    if prefix_open:
                        # Tier 1: L1 HBM hit
                        hslot_hit = None
                        with lock:
                            if hid in hbm_lru:
                                hslot_hit = hbm_lru.touch(hid)
                        if hslot_hit is not None:
                            with torch.cuda.stream(stream):
                                compute_slot.copy_(hbm_slab[hslot_hit], non_blocking=True)
                            stream.synchronize()
                            hbm_hits += 1
                            continue

                        # Tier 2: L2 LocalCPUBackend hit -> promote to HBM
                        in_dram = False
                        with lock:
                            if hid in dram_lru:
                                dram_lru.remove(hid)
                                in_dram = True
                        if in_dram:
                            ckey = _mk_key(hid)
                            mo = cpu_backend.get_blocking(ckey)
                            if mo is not None:
                                new_hslot = _ensure_hbm_slot()
                                src_u8 = mo.tensor.view(torch.uint8).view(-1)
                                with torch.cuda.stream(stream):
                                    hbm_slab[new_hslot].copy_(src_u8, non_blocking=True)
                                stream.synchronize()
                                mo.ref_count_down()
                                cpu_backend.remove(ckey)
                                with lock:
                                    hbm_lru.insert(hid, new_hslot)
                                dram_hits += 1
                                continue

                        # Tier 3: L3 LocalDiskBackend (O_DIRECT) hit -> warm -> promote to HBM
                        in_nvme = False
                        with lock:
                            if hid in nvme_lru:
                                nvme_lru.remove(hid)
                                in_nvme = True
                        if in_nvme:
                            ckey = _mk_key(hid)
                            mo = disk_backend.get_blocking(ckey)
                            if mo is not None:
                                new_hslot = _ensure_hbm_slot()
                                src_u8 = mo.tensor.view(torch.uint8).view(-1)
                                with torch.cuda.stream(stream):
                                    hbm_slab[new_hslot].copy_(src_u8, non_blocking=True)
                                stream.synchronize()
                                mo.ref_count_down()
                                disk_backend.remove(ckey)
                                with lock:
                                    hbm_lru.insert(hid, new_hslot)
                                nvme_hits += 1
                                continue

                        # Tier 4: Remote Peer hit -> TransferEngine RDMA into GPU HBM
                        rsock.sendall(struct.pack("!BQ", 1, hid))
                        found, remote_addr, remote_slot = struct.unpack(
                            "!BQQ", _lmc_recv_exact(rsock, 17)
                        )
                        if found != 0:
                            new_hslot = _ensure_hbm_slot()
                            local_gpu_addr = hbm_ptr + new_hslot * frame_size
                            rc_rd = te.transfer_sync_read(
                                peer_seg, local_gpu_addr, remote_addr, frame_size
                            )
                            rel_val = (found << 56) | remote_slot
                            rsock.sendall(struct.pack("!BQ", 2, rel_val))
                            _lmc_recv_exact(rsock, 1)
                            if rc_rd != 0:
                                raise RuntimeError(
                                    f"LMCache S5 transfer_sync_read failed rc={rc_rd}"
                                )
                            with lock:
                                hbm_lru.insert(hid, new_hslot)
                            remote_hits += 1
                            continue

                    # Tier 5: Cold Miss -> writeback into L1 HBM
                    prefix_open = False
                    cold_misses += 1
                    with lock:
                        if hid in hbm_lru:
                            hbm_lru.touch(hid)
                            continue
                        if hid in dram_lru:
                            dram_lru.remove(hid)
                            cpu_backend.remove(_mk_key(hid))
                        if hid in nvme_lru:
                            nvme_lru.remove(hid)
                            disk_backend.remove(_mk_key(hid))
                    new_hslot = _ensure_hbm_slot()
                    with torch.cuda.stream(stream):
                        hbm_slab[new_hslot].copy_(cold_src, non_blocking=True)
                    stream.synchronize()
                    with lock:
                        hbm_lru.insert(hid, new_hslot)

                lat_samples.append(max(1, time.perf_counter_ns() - req_t0))

            rsock.sendall(struct.pack("!BQ", 3, 0))
            _lmc_recv_exact(rsock, 1)
            resp_t.join(timeout=1800.0)

            shard_results[s] = {
                "init_time_s": init_s,
                "reqs": len(per_shard_reqs[s]),
                "hbm_hits": hbm_hits,
                "dram_hits": dram_hits,
                "nvme_hits": nvme_hits,
                "remote_hits": remote_hits,
                "cold_misses": cold_misses,
                "samples_ns": lat_samples,
            }
        except BaseException as exc:
            shard_errors[s] = exc
            try:
                init_done_barrier.abort()
                start_go_barrier.abort()
            except Exception:
                pass
        finally:
            if disk_backend is not None:
                try:
                    disk_backend.close()
                except Exception:
                    pass
            if cpu_backend is not None:
                try:
                    cpu_backend.clear()
                    cpu_backend.close()
                except Exception:
                    pass
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass
            if loop_thread is not None:
                loop_thread.join(timeout=5)
            if te is not None:
                if hbm_ptr:
                    try:
                        te.unregister_memory(hbm_ptr)
                    except Exception:
                        pass
                if stage_ptr:
                    try:
                        te.unregister_memory(stage_ptr)
                    except Exception:
                        pass
            shutil.rmtree(disk_dir, ignore_errors=True)
            del hbm_slab
            del peer_stage

    try:
        t_node_init0 = time.perf_counter()
        threads = [threading.Thread(target=_shard_worker, args=(s,)) for s in range(shards)]
        for t in threads:
            t.start()

        init_done_barrier.wait()
        node_init_s = time.perf_counter() - t_node_init0

        ctrl_conn.send("INIT_DONE")
        go_sig = ctrl_conn.recv()
        assert go_sig == "GO"

        sampler = ProcSampler()
        t_run0 = time.perf_counter()
        start_go_barrier.wait()

        for t in threads:
            t.join()

        elapsed_s = time.perf_counter() - t_run0
        cpu_pct, rss_mb = sampler.finish()
        import resource
        rss_mb = max(rss_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

        for s, err in enumerate(shard_errors):
            if err is not None:
                raise RuntimeError(f"LMCache S5 node{node_id} shard{s} failed: {err}") from err

        samples_ns: list[int] = []
        tot_reqs = hbm_h = dram_h = nvme_h = rem_h = cold_m = 0
        max_init = node_init_s
        for st in shard_results:
            tot_reqs += st["reqs"]
            hbm_h += st["hbm_hits"]
            dram_h += st["dram_hits"]
            nvme_h += st["nvme_hits"]
            rem_h += st["remote_hits"]
            cold_m += st["cold_misses"]
            max_init = max(max_init, st["init_time_s"])
            samples_ns.extend(st["samples_ns"])

        vram_mb = (hbm_f * shards * frame_size) / (1024.0 * 1024.0)
        ctrl_conn.send(
            {
                "node_id": node_id,
                "init_time_s": max_init,
                "elapsed_s": elapsed_s,
                "cpu_pct": cpu_pct,
                "rss_mb": rss_mb,
                "vram_mb": vram_mb,
                "reqs": tot_reqs,
                "hbm_hits": hbm_h,
                "dram_hits": dram_h,
                "nvme_hits": nvme_h,
                "remote_hits": rem_h,
                "cold_misses": cold_m,
                "samples_ns": samples_ns,
            }
        )
    finally:
        for sock in req_socks + srv_socks:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        shutil.rmtree(node_disk_root, ignore_errors=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_suite5(job: dict[str, Any]) -> list[str]:
    cfg = scaleup_config_from_job(job)
    nvme_base = pathlib.Path(str(job.get("nvme_dir", "/workspace/target/bench-nvme")))
    ctx = mp.get_context("spawn")
    p0_parent, p0_child = ctx.Pipe()
    p1_parent, p1_child = ctx.Pipe()

    proc0 = ctx.Process(target=_lmc_s5_node_proc, args=(0, job, p0_child))
    proc1 = ctx.Process(target=_lmc_s5_node_proc, args=(1, job, p1_child))
    try:
        proc0.start()
        proc1.start()

        ports_msg = p0_parent.recv()
        p1_parent.send(ports_msg)

        assert p0_parent.recv() == "INIT_DONE"
        assert p1_parent.recv() == "INIT_DONE"
        p0_parent.send("GO")
        p1_parent.send("GO")

        r0 = p0_parent.recv()
        r1 = p1_parent.recv()

        proc0.join(timeout=1800)
        proc1.join(timeout=1800)
        if proc0.exitcode != 0 or proc1.exitcode != 0:
            raise RuntimeError(
                f"LMCache Suite 5 failed: node0={proc0.exitcode}, node1={proc1.exitcode}"
            )
    finally:
        for p in (proc0, proc1):
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        # Belt-and-suspenders sweep for any lmcache_s5_* directories under all configured NVMe dirs
        nvme_dirs = {
            nvme_base,
            pathlib.Path(str(job.get("nvme_dir_node0") or nvme_base)),
            pathlib.Path(str(job.get("nvme_dir_node1") or nvme_base)),
        }
        for d in nvme_dirs:
            if d.exists():
                for child in d.glob("lmcache_s5_*"):
                    shutil.rmtree(child, ignore_errors=True)

    lat = LatencyRecorder()
    lat.extend_ns(r0["samples_ns"])
    lat.extend_ns(r1["samples_ns"])
    p50, p95, p99, p999 = lat.percentiles_us_full()

    init_time_s = max(r0["init_time_s"], r1["init_time_s"])
    elapsed_s = max(1e-9, max(r0["elapsed_s"], r1["elapsed_s"]))
    total_reqs = r0["reqs"] + r1["reqs"]
    hbm_hits = r0["hbm_hits"] + r1["hbm_hits"]
    dram_hits = r0["dram_hits"] + r1["dram_hits"]
    nvme_hits = r0["nvme_hits"] + r1["nvme_hits"]
    remote_hits = r0["remote_hits"] + r1["remote_hits"]
    cold_misses = r0["cold_misses"] + r1["cold_misses"]
    total_blocks = hbm_hits + dram_hits + nvme_hits + remote_hits + cold_misses
    total_hits = hbm_hits + dram_hits + nvme_hits + remote_hits

    pct = lambda n: (n * 100.0 / total_blocks) if total_blocks > 0 else 0.0
    frame_size = cfg["frame_size"]
    total_bytes = total_blocks * frame_size
    hit_bytes = total_hits * frame_size
    gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s)
    hit_gbps = bytes_per_sec_to_gbps(hit_bytes, elapsed_s)
    req_per_s = total_reqs / elapsed_s
    hit_rate = (total_hits / total_blocks) if total_blocks > 0 else 0.0
    total_cluster_frames = 2 * cfg["shards_per_node"] * (
        cfg["hbm_frames_per_shard"] + cfg["dram_frames_per_shard"] + cfg["nvme_frames_per_shard"]
    )

    notes_obj = {
        "init_time_s": round(init_time_s, 3),
        "elapsed_s": round(elapsed_s, 3),
        "req_per_s": round(req_per_s, 2),
        "reqs": total_reqs,
        "total_blocks": total_blocks,
        "hit_goodput_gbps": round(hit_gbps, 2),
        "hbm_hit_pct": round(pct(hbm_hits), 2),
        "dram_hit_pct": round(pct(dram_hits), 2),
        "nvme_hit_pct": round(pct(nvme_hits), 2),
        "remote_hit_pct": round(pct(remote_hits), 2),
        "cold_miss_pct": round(pct(cold_misses), 2),
        "hbm_hits": hbm_hits,
        "dram_hits": dram_hits,
        "nvme_hits": nvme_hits,
        "remote_hits": remote_hits,
        "cold_misses": cold_misses,
        "p999_us": round(p999, 1),
        "vram_mb": round(r0["vram_mb"] + r1["vram_mb"], 1),
        "hbm_gb_per_node": cfg["hbm_gb_per_node"],
        "dram_gb_per_node": cfg["dram_gb_per_node"],
        "nvme_gb_per_node": cfg["nvme_gb_per_node"],
        "shards_per_node": cfg["shards_per_node"],
    }

    rec = MetricRecord.from_job("lmcache", job)
    rec.num_blocks = total_cluster_frames
    rec.concurrency = cfg["shards_per_node"] * 2
    rec.throughput_gbps = gbps
    rec.goodput_gbps = gbps
    rec.p50_us = p50
    rec.p95_us = p95
    rec.p99_us = p99
    rec.hit_rate = hit_rate
    rec.cpu_pct = r0["cpu_pct"] + r1["cpu_pct"]
    rec.rss_mb = r0["rss_mb"] + r1["rss_mb"]
    rec.notes = json.dumps(notes_obj, separators=(",", ":"))
    return [rec.to_json()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-file", type=str)
    ap.add_argument("--job-json", type=str)
    args = ap.parse_args()
    if args.job_file:
        with open(args.job_file, "r", encoding="utf-8") as f:
            job = json.load(f)
    elif args.job_json:
        job = json.loads(args.job_json)
    else:
        raise SystemExit("must specify --job-file or --job-json")

    suite = str(job["suite"])
    case = str(job["case"])
    if suite == "s1_local_tiers":
        if case == "1a_gpu_to_host_scatter":
            out = _run_1a_gpu_scatter(job, is_d2h=True)
        elif case == "1a_host_to_gpu_scatter":
            out = _run_1a_gpu_scatter(job, is_d2h=False)
        elif case == "1b_nvme_write_spill":
            out = _run_1b_disk(job, is_write=True)
        elif case == "1b_nvme_read_cold":
            out = _run_1b_disk(job, is_write=False)
        else:
            raise SystemExit(f"unknown s1 case '{case}'")
    elif suite == "s3_trace_replay":
        out = run_suite3(job)
    elif suite == "s5_scaleup_cluster":
        out = run_suite5(job)
    else:
        raise SystemExit(f"lmcache_runner does not support suite '{suite}'")

    for line in out:
        print(line, flush=True)


if __name__ == "__main__":
    main()
