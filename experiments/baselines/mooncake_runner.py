#!/usr/bin/env python3
"""Official Mooncake 0.3.14 baseline runner for Suites 2, 3, and 4."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
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
    should_revoke,
    stamp_checksummed_array,
    verify_checksummed_array,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# =========================================================================
# Suite 2: 2-process Mooncake TransferEngine RDMA (2A Host->Host, 2B GDR GPU0->GPU1, 2C static MR)
# =========================================================================
def _s2_donor_proc(job: dict[str, Any], use_gpu: bool, conn) -> None:
    from mooncake.engine import TransferEngine

    frame_size = int(job["chunk_bytes"])
    qd = max(1, int(job.get("concurrency", 1)))
    rdma_dev = str(job.get("rdma_dev", "mlx5_3"))
    donor_gpu = int(job.get("donor_gpu", 0))

    te = TransferEngine()
    rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", "rdma", rdma_dev)
    if rc != 0:
        raise RuntimeError(f"Mooncake donor initialize failed rc={rc}")
    rpc_port = te.get_rpc_port()
    seg_name = f"127.0.0.1:{rpc_port}"

    total_bytes = frame_size * qd
    if use_gpu:
        torch.cuda.set_device(donor_gpu)
        buf = torch.empty((total_bytes,), dtype=torch.uint8, device=f"cuda:{donor_gpu}")
        buf.fill_(0x5A)
        torch.cuda.synchronize(donor_gpu)
    else:
        buf = torch.empty((total_bytes,), dtype=torch.uint8, pin_memory=True)
        buf.fill_(0x5A)

    ptr = int(buf.data_ptr())
    rc = te.register_memory(ptr, total_bytes)
    if rc != 0:
        raise RuntimeError(f"Mooncake donor register_memory failed rc={rc}")

    conn.send({"seg_name": seg_name, "ptr": ptr, "total_bytes": total_bytes})
    # Wait for initiator completion signal
    msg = conn.recv()
    assert msg == "done"
    te.unregister_memory(ptr)


def _s2_initiator_proc(job: dict[str, Any], use_gpu: bool, conn) -> None:
    from mooncake.engine import TransferEngine

    frame_size = int(job["chunk_bytes"])
    qd = max(1, int(job.get("concurrency", 1)))
    warmup = int(job.get("warmup_iters", 8))
    iters = int(job.get("iters", 64))
    rdma_dev = str(job.get("rdma_dev", "mlx5_3"))
    init_gpu = int(job.get("initiator_gpu", 1))

    peer_info = conn.recv()
    target_seg = peer_info["seg_name"]
    remote_base = int(peer_info["ptr"])

    te = TransferEngine()
    rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", "rdma", rdma_dev)
    if rc != 0:
        raise RuntimeError(f"Mooncake initiator initialize failed rc={rc}")

    total_bytes = frame_size * qd
    if use_gpu:
        torch.cuda.set_device(init_gpu)
        buf = torch.empty((total_bytes,), dtype=torch.uint8, device=f"cuda:{init_gpu}")
        buf.zero_()
        torch.cuda.synchronize(init_gpu)
    else:
        buf = torch.empty((total_bytes,), dtype=torch.uint8, pin_memory=True)
        buf.zero_()

    local_base = int(buf.data_ptr())
    rc = te.register_memory(local_base, total_bytes)
    if rc != 0:
        raise RuntimeError(f"Mooncake initiator register_memory failed rc={rc}")

    local_ptrs = [local_base + i * frame_size for i in range(qd)]
    remote_ptrs = [remote_base + i * frame_size for i in range(qd)]
    lengths = [frame_size] * qd

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
        if qd == 1:
            rc = te.transfer_sync_read(target_seg, local_base, remote_base, frame_size)
        else:
            rc = te.batch_transfer_sync_read(target_seg, local_ptrs, remote_ptrs, lengths)
        if rc != 0:
            raise RuntimeError(f"Mooncake RDMA read failed rc={rc} (qd={qd}, len={frame_size})")
        dt_ns = time.perf_counter_ns() - t0
        if b >= warmup_batches:
            lat.record_ns(dt_ns)
            total_ops += qd

    elapsed_s = time.perf_counter() - t_meas_start
    cpu_pct, rss_mb = sampler.finish()
    total_xfer = total_ops * frame_size
    gbps = bytes_per_sec_to_gbps(total_xfer, elapsed_s)
    p50, p95, p99 = lat.percentiles_us()

    te.unregister_memory(local_base)
    conn.send("done")

    rec = MetricRecord.from_job("mooncake", job)
    rec.throughput_gbps = gbps
    rec.goodput_gbps = gbps
    rec.p50_us = p50
    rec.p95_us = p95
    rec.p99_us = p99
    rec.hit_rate = 1.0
    rec.cpu_pct = cpu_pct
    rec.rss_mb = rss_mb
    api_name = "batch_transfer_sync_read" if qd > 1 else "transfer_sync_read"
    tier_desc = f"GPU{job.get('donor_gpu', 0)}->GPU{init_gpu} GDR" if use_gpu else "Host->Host RDMA"
    rec.notes = f"Mooncake 0.3.14 TransferEngine {api_name} ({tier_desc}, static MR rkey, QD={qd}, {rdma_dev})"
    conn.send(rec.to_json())


def run_suite2(job: dict[str, Any]) -> list[str]:
    case = str(job["case"])
    use_gpu = case in ("2b_gdr_gpu", "2c_ablation_raw_mw", "2c_ablation_dynamic_single", "2c_ablation_dynamic_pipelined")
    ctx = mp.get_context("spawn")
    parent_d, child_d = ctx.Pipe()
    parent_i, child_i = ctx.Pipe()

    p_donor = ctx.Process(target=_s2_donor_proc, args=(job, use_gpu, child_d))
    p_init = ctx.Process(target=_s2_initiator_proc, args=(job, use_gpu, child_i))
    p_donor.start()
    p_init.start()

    # Relay handshake info donor -> initiator, then done initiator -> donor
    peer_info = parent_d.recv()
    parent_i.send(peer_info)
    done_sig = parent_i.recv()
    parent_d.send(done_sig)
    rec_json = parent_i.recv()

    p_donor.join(timeout=60)
    p_init.join(timeout=60)
    if p_donor.exitcode != 0 or p_init.exitcode != 0:
        raise RuntimeError(
            f"Mooncake Suite 2 failed: donor={p_donor.exitcode}, initiator={p_init.exitcode}"
        )
    return [rec_json]


# =========================================================================
# Suite 3: Official MooncakeDistributedStore + mooncake_master trace replay
# =========================================================================
def run_suite3(job: dict[str, Any]) -> list[str]:
    from mooncake.store import MooncakeDistributedStore

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

    rdma_dev = str(job.get("rdma_dev", "mlx5_3"))
    master_port = _find_free_port()
    metrics_port = _find_free_port()
    master_addr = f"127.0.0.1:{master_port}"

    # Size global segment with headroom above cap_frames * frame_size (min 64 MiB for Mooncake allocator alignment),
    # and enforce exact `cap_frames` LRU capacity via explicit LRU remove() matching VeriCache & LMCache!
    raw_seg_bytes = max(64 * 1024 * 1024, int(cap_frames * frame_size * 1.5))
    # Round up to 16 MiB multiple
    align_16m = 16 * 1024 * 1024
    global_seg_size = ((raw_seg_bytes + align_16m - 1) // align_16m) * align_16m
    local_buf_size = max(16 * 1024 * 1024, frame_size * 4)

    master_proc = subprocess.Popen(
        [
            "mooncake_master",
            "--port",
            str(master_port),
            "--metrics_port",
            str(metrics_port),
            "--enable_http_metadata_server=false",
            "--default_kv_lease_ttl=50",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    store = MooncakeDistributedStore()
    try:
        # Wait for master port to accept TCP
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", master_port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)

        rc = store.setup(
            "127.0.0.1",
            "P2PHANDSHAKE",
            global_seg_size,
            local_buf_size,
            "rdma",
            rdma_dev,
            master_addr,
        )
        if rc != 0:
            raise RuntimeError(f"MooncakeDistributedStore.setup failed rc={rc}")

        io_buf = torch.empty((frame_size,), dtype=torch.uint8, pin_memory=True)
        io_buf.fill_(0x3C)
        read_buf = torch.empty((frame_size,), dtype=torch.uint8, pin_memory=True)
        rc_w = store.register_buffer(int(io_buf.data_ptr()), frame_size)
        rc_r = store.register_buffer(int(read_buf.data_ptr()), frame_size)
        if rc_w != 0 or rc_r != 0:
            raise RuntimeError(f"Mooncake register_buffer failed rc_w={rc_w}, rc_r={rc_r}")

        io_ptr = int(io_buf.data_ptr())
        read_ptr = int(read_buf.data_ptr())
        lru = LruTracker(cap_frames)

        is_open_loop = str(job["mode"]) == "open_loop"
        time_scale = max(1.0, float(job.get("time_scale", 1000.0)))
        base_ts = int(reqs[0]["timestamp"])

        lat = LatencyRecorder()
        sampler = ProcSampler()
        t_start = time.perf_counter()
        hit_blocks = 0
        miss_blocks = 0

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
                key_str = f"kv_{hid}"
                if prefix_open and hid in lru:
                    got = store.get_into(key_str, read_ptr, frame_size)
                    if got == frame_size:
                        lru.touch(hid)
                        hit_blocks += 1
                        continue
                prefix_open = False
                miss_blocks += 1
                if hid in lru:
                    lru.touch(hid)
                    continue
                if lru.is_full():
                    vic = lru.pop_lru()
                    if vic is not None:
                        store.remove(f"kv_{vic[0]}")
                put_rc = store.put_from(key_str, io_ptr, frame_size)
                if put_rc == 0:
                    lru.insert(hid, True)
                else:
                    # If allocator is tight, evict one more LRU victim and retry once
                    vic = lru.pop_lru()
                    if vic is not None:
                        store.remove(f"kv_{vic[0]}")
                    put_rc2 = store.put_from(key_str, io_ptr, frame_size)
                    if put_rc2 == 0:
                        lru.insert(hid, True)

            lat.record_ns(time.perf_counter_ns() - req_t0)

        elapsed_s = time.perf_counter() - t_start
        cpu_pct, rss_mb = sampler.finish()
        store.unregister_buffer(io_ptr)
        store.unregister_buffer(read_ptr)
        store.close()
    finally:
        master_proc.terminate()
        try:
            master_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            master_proc.kill()

    total_blocks = hit_blocks + miss_blocks
    hit_rate = (hit_blocks / total_blocks) if total_blocks > 0 else 0.0
    total_bytes = total_blocks * frame_size
    hit_bytes = hit_blocks * frame_size
    gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s)
    hit_gbps = bytes_per_sec_to_gbps(hit_bytes, elapsed_s)
    p50, p95, p99 = lat.percentiles_us()

    rec = MetricRecord.from_job("mooncake", job)
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
        f"MooncakeDistributedStore+mooncake_master (rdma/{rdma_dev}), reqs={len(reqs)}, "
        f"unique_blks={len(unique_blocks)}, total_blks={total_blocks_in_trace}, "
        f"cap_frames={cap_frames} (ratio={ratio:.2f}), hit_only_gbps={hit_gbps:.2f}"
    )
    return [rec.to_json()]


# =========================================================================
# Suite 4: Static-rkey contention & overwrite tearing check on Mooncake TransferEngine
# =========================================================================
def _build_s4_workload(job: dict[str, Any]) -> tuple[list[int], int]:
    trace_path = str(job.get("trace_path", "/workspace/target/bench-traces/toolagent_trace.jsonl"))
    reqs = load_trace(trace_path, int(job.get("max_requests", 128)))
    unique = set()
    seq = []
    for r in reqs:
        for h in r["hash_ids"]:
            unique.add(h)
            seq.append(int(h))
    ratio = float(job.get("capacity_ratio", 0.15))
    cap_frames = int(job.get("capacity_frames") or max(16, min(512, round(len(unique) * ratio))))
    total_ops = max(200, int(job.get("iters", 400)))
    ops = [seq[i % len(seq)] for i in range(total_ops)]
    return ops, cap_frames


def _s4_donor_proc(job: dict[str, Any], conn) -> None:
    from mooncake.engine import TransferEngine

    frame_size = int(job["chunk_bytes"])
    revoke_rate = float(job.get("revoke_rate", 0.01))
    rdma_dev = str(job.get("rdma_dev", "mlx5_3"))
    ops, donor_cap = _build_s4_workload(job)

    te = TransferEngine()
    rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", "rdma", rdma_dev)
    if rc != 0:
        raise RuntimeError(f"Mooncake S4 donor init failed rc={rc}")
    seg_name = f"127.0.0.1:{te.get_rpc_port()}"

    # Static MR slab registered once with static rkey (no per-lease Type-2B MW isolation).
    slab = torch.empty((donor_cap, frame_size), dtype=torch.uint8, pin_memory=True)
    slab_ptr = int(slab.data_ptr())
    rc = te.register_memory(slab_ptr, donor_cap * frame_size)
    if rc != 0:
        raise RuntimeError(f"Mooncake S4 donor register_memory failed rc={rc}")
    slab_u64 = slab.numpy().view(np.uint64)

    conn.send({"seg_name": seg_name, "slab_ptr": slab_ptr, "donor_cap": donor_cap})

    lru = LruTracker(donor_cap)
    free_slots = list(range(donor_cap))
    version_counter = 1

    for op_idx, hid in enumerate(ops):
        if hid in lru:
            slot = lru.touch(hid)
        else:
            if free_slots:
                slot = free_slots.pop()
            else:
                _vk, slot = lru.pop_lru()
            stamp_checksummed_array(slab_u64[slot], hid, version_counter)
            version_counter += 1
            lru.insert(hid, slot)

        remote_addr = slab_ptr + slot * frame_size
        will_evict = should_revoke(op_idx, revoke_rate)
        # Send static (addr, rkey-backed) grant to initiator
        conn.send({"addr": remote_addr, "hid": hid, "evict": will_evict})

        if will_evict:
            # Donor reclaims and immediately overwrites the frame with a conflicting hot key
            # (simulating tight 15% capacity eviction under static-MR RDMA with no hardware MW revoke!).
            conflict_hid = (hid + 0xDEADBEEF) & 0xFFFFFFFFFFFFFFFF
            lru.remove(hid)
            stamp_checksummed_array(slab_u64[slot], conflict_hid, version_counter)
            version_counter += 1
            lru.insert(conflict_hid, slot)

        ack = conn.recv()
        assert ack == "ok"

    te.unregister_memory(slab_ptr)


def _s4_initiator_proc(job: dict[str, Any], peer_conn, res_conn) -> None:
    from mooncake.engine import TransferEngine

    frame_size = int(job["chunk_bytes"])
    revoke_rate = float(job.get("revoke_rate", 0.01))
    rdma_dev = str(job.get("rdma_dev", "mlx5_3"))
    ops, donor_cap = _build_s4_workload(job)

    info = peer_conn.recv()
    target_seg = info["seg_name"]

    te = TransferEngine()
    rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", "rdma", rdma_dev)
    if rc != 0:
        raise RuntimeError(f"Mooncake S4 initiator init failed rc={rc}")

    local_buf = torch.empty((frame_size,), dtype=torch.uint8, pin_memory=True)
    local_ptr = int(local_buf.data_ptr())
    rc = te.register_memory(local_ptr, frame_size)
    if rc != 0:
        raise RuntimeError(f"Mooncake S4 initiator register_memory failed rc={rc}")
    local_u64 = local_buf.numpy().view(np.uint64)

    lat = LatencyRecorder()
    sampler = ProcSampler()
    t_start = time.perf_counter()

    corrupted = 0
    valid = 0
    forced_overwrites = 0

    for _op_idx, expected_hid in enumerate(ops):
        grant = peer_conn.recv()
        remote_addr = int(grant["addr"])
        if grant["evict"]:
            forced_overwrites += 1
            # Brief yield so donor's overwrite races / completes against static rkey
            time.sleep(0.00005)

        t0 = time.perf_counter_ns()
        rc = te.transfer_sync_read(target_seg, local_ptr, remote_addr, frame_size)
        if rc != 0:
            raise RuntimeError(f"Mooncake S4 RDMA read failed rc={rc}")
        dt_ns = time.perf_counter_ns() - t0
        lat.record_ns(dt_ns)

        ok, _reason = verify_checksummed_array(local_u64, expected_hid)
        if ok:
            valid += 1
        else:
            corrupted += 1
        peer_conn.send("ok")

    elapsed_s = time.perf_counter() - t_start
    cpu_pct, rss_mb = sampler.finish()
    te.unregister_memory(local_ptr)

    p50, p95, p99 = lat.percentiles_us()
    rec = MetricRecord.from_job("mooncake", job)
    rec.num_blocks = donor_cap
    rec.throughput_gbps = bytes_per_sec_to_gbps(len(ops) * frame_size, elapsed_s)
    rec.goodput_gbps = bytes_per_sec_to_gbps(valid * frame_size, elapsed_s)
    rec.p50_us = p50
    rec.p95_us = p95
    rec.p99_us = p99
    rec.hit_rate = valid / len(ops)
    rec.cpu_pct = cpu_pct
    rec.rss_mb = rss_mb
    rec.corrupted_frames = corrupted
    rec.recovery_us = 0.0
    rec.notes = (
        f"Static-MR rkey (no Type-2B MW revoke): forced_evict_overwrites={forced_overwrites}, "
        f"corrupted_silent_reads={corrupted}/{len(ops)} (rate={revoke_rate:.3f}, donor_cap_15pct={donor_cap})"
    )
    res_conn.send(rec.to_json())


def run_suite4(job: dict[str, Any]) -> list[str]:
    ctx = mp.get_context("spawn")
    d_end, i_end = ctx.Pipe()
    parent_res, child_res = ctx.Pipe()

    p_donor = ctx.Process(target=_s4_donor_proc, args=(job, d_end))
    p_init = ctx.Process(target=_s4_initiator_proc, args=(job, i_end, child_res))
    try:
        p_donor.start()
        p_init.start()
        rec_json = parent_res.recv()
        p_donor.join(timeout=60)
        p_init.join(timeout=60)
        if p_donor.exitcode != 0 or p_init.exitcode != 0:
            raise RuntimeError(
                f"Mooncake Suite 4 failed: donor={p_donor.exitcode}, initiator={p_init.exitcode}"
            )
        return [rec_json]
    finally:
        for p in (p_donor, p_init):
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


# =========================================================================
# Suite 5 (Scale Wave 2): 600 GiB 2-Node 8-Shard 4-Tier Cluster Replay
# =========================================================================
def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(mv[got:], n - got)
        if r == 0:
            raise EOFError("peer socket closed")
        got += r
    return bytes(buf)


def _get_libc_io():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
    libc.pread.restype = ctypes.c_ssize_t
    libc.pwrite.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
    libc.pwrite.restype = ctypes.c_ssize_t
    return libc


def _odirect_pwrite(libc, fd: int, ptr: int, size: int, offset: int) -> None:
    rc = libc.pwrite(fd, ctypes.c_void_p(ptr), size, offset)
    if rc != size:
        err = ctypes.get_errno()
        raise OSError(err, f"O_DIRECT pwrite failed rc={rc} expected={size} errno={err}")


def _odirect_pread(libc, fd: int, ptr: int, size: int, offset: int) -> None:
    rc = libc.pread(fd, ctypes.c_void_p(ptr), size, offset)
    if rc != size:
        err = ctypes.get_errno()
        raise OSError(err, f"O_DIRECT pread failed rc={rc} expected={size} errno={err}")


def _s5_node_proc(node_id: int, job: dict[str, Any], peer_ports: dict[str, Any], ctrl_conn) -> None:
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
    nvme_dir = str(job.get(nvme_key) or job.get("nvme_dir", "/workspace/target/bench-nvme"))
    os.makedirs(nvme_dir, exist_ok=True)

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    libc = _get_libc_io()

    reqs = load_trace(str(job["trace_path"]), job.get("max_requests"))
    per_shard_reqs: list[list[list[int]]] = [[] for _ in range(shards)]
    for req_idx, r in enumerate(reqs):
        hids = r["hash_ids"]
        t_node, sid = route_request_scaleup(req_idx, hids, shards, sticky_pct)
        if t_node == node_id:
            per_shard_reqs[sid].append(hids)

    # Establish 2 unidirectional TCP sockets per shard (0->1 and 1->0) for deadlock-free RPC.
    req_socks: list[socket.socket] = [None] * shards  # type: ignore
    srv_socks: list[socket.socket] = [None] * shards  # type: ignore
    listeners: list[socket.socket] = []
    if node_id == 0:
        for s in range(shards):
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
            tag1 = _recv_exact(c1, 1)
            tag2 = _recv_exact(c2, 1)
            if tag1 == b"A":
                req_socks[s], srv_socks[s] = c1, c2
            else:
                req_socks[s], srv_socks[s] = c2, c1
            listeners[s].close()
    else:
        ports = ctrl_conn.recv()["ports"]
        for s in range(shards):
            # c1 tagged 'A' connects to node0's req_socks[s], so it is node1's srv_socks[s]
            c1 = socket.create_connection(("127.0.0.1", ports[s]), timeout=60.0)
            c1.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c1.sendall(b"A")
            # c2 tagged 'B' connects to node0's srv_socks[s], so it is node1's req_socks[s]
            c2 = socket.create_connection(("127.0.0.1", ports[s]), timeout=60.0)
            c2.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c2.sendall(b"B")
            srv_socks[s], req_socks[s] = c1, c2

    # Parallel 8-shard pool allocation & TransferEngine registration.
    init_done_barrier = threading.Barrier(shards + 1)
    start_go_barrier = threading.Barrier(shards + 1)
    shard_results: list[dict[str, Any]] = [{} for _ in range(shards)]
    shard_errors: list[BaseException | None] = [None] * shards
    local_seg_names: list[str] = [""] * shards
    peer_seg_names: list[str] = [""] * shards

    def _shard_worker(s: int) -> None:
        te = None
        hbm_slab = None
        dram_slab = None
        nvme_fd = -1
        hbm_ptr = 0
        dram_ptr = 0
        nvme_path = os.path.join(nvme_dir, f"mooncake_s5_{os.getpid()}_n{node_id}_s{s}.bin")
        try:
            torch.cuda.set_device(gpu_id)
            t_init0 = time.perf_counter()

            te = TransferEngine()
            rc = te.initialize("127.0.0.1:0", "P2PHANDSHAKE", "rdma", rdma_dev)
            if rc != 0:
                raise RuntimeError(f"Mooncake S5 node{node_id} shard{s} TE.initialize failed rc={rc}")
            seg_name = f"127.0.0.1:{te.get_rpc_port()}"
            local_seg_names[s] = seg_name

            hbm_bytes = hbm_f * frame_size
            dram_bytes = dram_f * frame_size
            nvme_bytes = nvme_f * frame_size

            hbm_slab = torch.empty((hbm_f, frame_size), dtype=torch.uint8, device=device)
            dram_slab = torch.empty((dram_f, frame_size), dtype=torch.uint8, pin_memory=True)
            hbm_ptr = int(hbm_slab.data_ptr())
            dram_ptr = int(dram_slab.data_ptr())

            rc_h = te.register_memory(hbm_ptr, hbm_bytes)
            if rc_h != 0:
                raise RuntimeError(
                    f"Mooncake S5 node{node_id} shard{s} register HBM ({hbm_bytes} B) failed rc={rc_h}"
                )
            rc_d = te.register_memory(dram_ptr, dram_bytes)
            if rc_d != 0:
                raise RuntimeError(
                    f"Mooncake S5 node{node_id} shard{s} register DRAM ({dram_bytes} B) failed rc={rc_d}"
                )

            nvme_fd = os.open(nvme_path, os.O_CREAT | os.O_RDWR | os.O_DIRECT | os.O_TRUNC, 0o600)
            os.ftruncate(nvme_fd, nvme_bytes)
            # Unlink immediately so Linux guarantees 0 leaked files even on SIGKILL!
            os.unlink(nvme_path)

            # Exchange TransferEngine segment names over the shard's TCP sockets.
            seg_b = seg_name.encode("utf-8")
            req_socks[s].sendall(struct.pack("!I", len(seg_b)) + seg_b)
            plen = struct.unpack("!I", _recv_exact(srv_socks[s], 4))[0]
            peer_seg = _recv_exact(srv_socks[s], plen).decode("utf-8")
            peer_seg_names[s] = peer_seg

            compute_slot = torch.empty((frame_size,), dtype=torch.uint8, device=device)
            cold_src = torch.empty((frame_size,), dtype=torch.uint8, pin_memory=True)
            cold_src.fill_(0x3C ^ (s & 0xFF))
            stream = torch.cuda.Stream(device=device)
            torch.cuda.synchronize(device)

            init_s = time.perf_counter() - t_init0

            lock = threading.Lock()
            cond = threading.Condition(lock)
            hbm_lru = LruTracker(hbm_f)
            dram_lru = LruTracker(dram_f)
            nvme_lru = LruTracker(nvme_f)
            free_hbm = list(range(hbm_f))
            free_dram = list(range(dram_f))
            free_nvme = list(range(nvme_f))
            pinned_hbm: dict[int, int] = {}
            pinned_dram: dict[int, int] = {}

            # Background peer responder thread serving cross-node RDMA queries (HBM or DRAM).
            def _responder() -> None:
                rs = srv_socks[s]
                while True:
                    hdr = _recv_exact(rs, 9)
                    op = hdr[0]
                    hid = struct.unpack("!Q", hdr[1:9])[0]
                    if op == 1:  # QUERY
                        with lock:
                            if hid in hbm_lru:
                                slot = hbm_lru._od[hid]
                                pinned_hbm[slot] = pinned_hbm.get(slot, 0) + 1
                                addr = hbm_ptr + slot * frame_size
                                rs.sendall(struct.pack("!BQQ", 1, addr, slot))
                            elif hid in dram_lru:
                                slot = dram_lru._od[hid]
                                pinned_dram[slot] = pinned_dram.get(slot, 0) + 1
                                addr = dram_ptr + slot * frame_size
                                rs.sendall(struct.pack("!BQQ", 2, addr, slot))
                            else:
                                rs.sendall(struct.pack("!BQQ", 0, 0, 0))
                    elif op == 2:  # RELEASE (hid encodes tier<<56 | slot)
                        tier_tag = hid >> 56
                        slot = hid & 0x00FFFFFFFFFFFFFF
                        with cond:
                            if tier_tag == 1:
                                cnt = pinned_hbm.get(slot, 1) - 1
                                if cnt <= 0:
                                    pinned_hbm.pop(slot, None)
                                else:
                                    pinned_hbm[slot] = cnt
                            else:
                                cnt = pinned_dram.get(slot, 1) - 1
                                if cnt <= 0:
                                    pinned_dram.pop(slot, None)
                                else:
                                    pinned_dram[slot] = cnt
                            cond.notify_all()
                        rs.sendall(b"K")
                    elif op == 3:  # DONE
                        rs.sendall(b"D")
                        break

            resp_t = threading.Thread(target=_responder, daemon=True)
            resp_t.start()

            init_done_barrier.wait()
            start_go_barrier.wait()

            def _alloc_nvme_slot() -> int:
                if free_nvme:
                    return free_nvme.pop()
                _nk, nslot = nvme_lru.pop_lru()
                return nslot

            def _ensure_dram_slot() -> int:
                with cond:
                    if free_dram:
                        return free_dram.pop()
                    dk, dslot = dram_lru.pop_lru()
                    while pinned_dram.get(dslot, 0) > 0:
                        cond.wait(timeout=0.001)
                # Spill dk from dram_slab[dslot] -> NVMe O_DIRECT (outside lock!)
                nslot = _alloc_nvme_slot()
                _odirect_pwrite(
                    libc, nvme_fd, dram_ptr + dslot * frame_size, frame_size, nslot * frame_size
                )
                with lock:
                    old_n = nvme_lru.remove(dk)
                    if old_n is not None:
                        free_nvme.append(old_n)
                    nvme_lru.insert(dk, nslot)
                return dslot

            def _ensure_hbm_slot() -> int:
                with cond:
                    if free_hbm:
                        return free_hbm.pop()
                    vk, hslot = hbm_lru.pop_lru()
                    while pinned_hbm.get(hslot, 0) > 0:
                        cond.wait(timeout=0.001)
                # Demote vk: HBM -> DRAM (cascading DRAM -> NVMe if full)
                with lock:
                    old_d = dram_lru.remove(vk)
                    if old_d is not None:
                        free_dram.append(old_d)
                dslot = _ensure_dram_slot()
                with torch.cuda.stream(stream):
                    dram_slab[dslot].copy_(hbm_slab[hslot], non_blocking=True)
                stream.synchronize()
                with lock:
                    dram_lru.insert(vk, dslot)
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

                        # Tier 2: L2 Pinned DRAM hit -> promote to HBM
                        dslot_hit = None
                        with cond:
                            if hid in dram_lru:
                                dslot_hit = dram_lru.remove(hid)
                                while pinned_dram.get(dslot_hit, 0) > 0:
                                    cond.wait(timeout=0.001)
                        if dslot_hit is not None:
                            new_hslot = _ensure_hbm_slot()
                            with torch.cuda.stream(stream):
                                hbm_slab[new_hslot].copy_(dram_slab[dslot_hit], non_blocking=True)
                            stream.synchronize()
                            with lock:
                                free_dram.append(dslot_hit)
                                hbm_lru.insert(hid, new_hslot)
                            dram_hits += 1
                            continue

                        # Tier 3: L3 Local NVMe hit -> O_DIRECT read -> promote to HBM
                        nslot_hit = None
                        with lock:
                            if hid in nvme_lru:
                                nslot_hit = nvme_lru.remove(hid)
                        if nslot_hit is not None:
                            tmp_dslot = _ensure_dram_slot()
                            _odirect_pread(
                                libc,
                                nvme_fd,
                                dram_ptr + tmp_dslot * frame_size,
                                frame_size,
                                nslot_hit * frame_size,
                            )
                            new_hslot = _ensure_hbm_slot()
                            with torch.cuda.stream(stream):
                                hbm_slab[new_hslot].copy_(dram_slab[tmp_dslot], non_blocking=True)
                            stream.synchronize()
                            with lock:
                                free_nvme.append(nslot_hit)
                                free_dram.append(tmp_dslot)
                                hbm_lru.insert(hid, new_hslot)
                            nvme_hits += 1
                            continue

                        # Tier 4: Remote Peer hit -> Mooncake TransferEngine GPUDirect RDMA into GPU HBM
                        rsock.sendall(struct.pack("!BQ", 1, hid))
                        found, remote_addr, remote_slot = struct.unpack("!BQQ", _recv_exact(rsock, 17))
                        if found != 0:
                            new_hslot = _ensure_hbm_slot()
                            local_gpu_addr = hbm_ptr + new_hslot * frame_size
                            rc_rd = te.transfer_sync_read(
                                peer_seg, local_gpu_addr, remote_addr, frame_size
                            )
                            rel_val = (found << 56) | remote_slot
                            rsock.sendall(struct.pack("!BQ", 2, rel_val))
                            _recv_exact(rsock, 1)
                            if rc_rd != 0:
                                raise RuntimeError(
                                    f"Mooncake S5 transfer_sync_read failed rc={rc_rd} "
                                    f"(node={node_id}, shard={s})"
                                )
                            with lock:
                                hbm_lru.insert(hid, new_hslot)
                            remote_hits += 1
                            continue

                    # Tier 5: Cold Miss -> writeback new suffix frame into L1 HBM
                    prefix_open = False
                    cold_misses += 1
                    with cond:
                        if hid in hbm_lru:
                            hbm_lru.touch(hid)
                            continue
                        old_d = dram_lru.remove(hid)
                        if old_d is not None:
                            while pinned_dram.get(old_d, 0) > 0:
                                cond.wait(timeout=0.001)
                            free_dram.append(old_d)
                        old_n = nvme_lru.remove(hid)
                        if old_n is not None:
                            free_nvme.append(old_n)
                    new_hslot = _ensure_hbm_slot()
                    with torch.cuda.stream(stream):
                        hbm_slab[new_hslot].copy_(cold_src, non_blocking=True)
                    stream.synchronize()
                    with lock:
                        hbm_lru.insert(hid, new_hslot)

                lat_samples.append(max(1, time.perf_counter_ns() - req_t0))

            # Signal peer responder we are done sending queries and wait for peer to finish.
            rsock.sendall(struct.pack("!BQ", 3, 0))
            _recv_exact(rsock, 1)
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
            if te is not None:
                if hbm_ptr:
                    try:
                        te.unregister_memory(hbm_ptr)
                    except Exception:
                        pass
                if dram_ptr:
                    try:
                        te.unregister_memory(dram_ptr)
                    except Exception:
                        pass
            if nvme_fd >= 0:
                try:
                    os.close(nvme_fd)
                except OSError:
                    pass
            if os.path.exists(nvme_path):
                try:
                    os.unlink(nvme_path)
                except OSError:
                    pass
            del hbm_slab
            del dram_slab

    try:
        t_node_init0 = time.perf_counter()
        threads = [threading.Thread(target=_shard_worker, args=(s,)) for s in range(shards)]
        for t in threads:
            t.start()

        init_done_barrier.wait()
        node_init_s = time.perf_counter() - t_node_init0

        # Cross-node start barrier via parent coordinator pipe
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
                raise RuntimeError(f"Mooncake S5 node{node_id} shard{s} failed: {err}") from err

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
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_suite5(job: dict[str, Any]) -> list[str]:
    cfg = scaleup_config_from_job(job)
    ctx = mp.get_context("spawn")
    p0_parent, p0_child = ctx.Pipe()
    p1_parent, p1_child = ctx.Pipe()

    proc0 = ctx.Process(target=_s5_node_proc, args=(0, job, {}, p0_child))
    proc1 = ctx.Process(target=_s5_node_proc, args=(1, job, {}, p1_child))
    try:
        proc0.start()
        proc1.start()

        # Relay shard TCP listen ports from node0 -> node1
        ports_msg = p0_parent.recv()
        p1_parent.send(ports_msg)

        # Wait for both nodes to finish parallel 8-shard init, then release both together
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
                f"Mooncake Suite 5 failed: node0={proc0.exitcode}, node1={proc1.exitcode}"
            )
    finally:
        for p in (proc0, proc1):
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)

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

    rec = MetricRecord.from_job("mooncake", job)
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
    if suite == "s2_rdma_ablation":
        out = run_suite2(job)
    elif suite == "s3_trace_replay":
        out = run_suite3(job)
    elif suite == "s4_safety_revoke":
        out = run_suite4(job)
    elif suite == "s5_scaleup_cluster":
        out = run_suite5(job)
    else:
        raise SystemExit(f"mooncake_runner does not support suite '{suite}'")

    for line in out:
        print(line, flush=True)


if __name__ == "__main__":
    main()
