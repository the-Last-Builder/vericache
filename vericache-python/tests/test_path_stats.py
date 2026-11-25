from __future__ import annotations

import json
import socket
import sys
from types import SimpleNamespace

import pytest
import torch

from vericache import HitLocation, VeriCacheClient, key_from_u64
from vericache.vllm_connector import (
    _LOOKUP_HDR,
    _STATS_DRAIN,
    _VeriCacheHmaWorker,
    _LocalLookupServer,
    VeriCacheConnector,
)


def test_native_cpu_stats_and_udsocket_drain(tmp_path):
    try:
        client = VeriCacheClient(node_id=0, frame_size=4096, dram_frames=2)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    with client:
        for idx in range(3):
            client.put_host(key_from_u64(idx), bytes([idx]) * 4096)
        # Construct only telemetry's worker geometry. The native cache and UDS server are real.
        config = SimpleNamespace(kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)),
        ])
        worker = _VeriCacheHmaWorker(None, config, {})
        worker.client = client
        worker._frame_size = 4096
        server = _LocalLookupServer(str(tmp_path / "vc.sock"), client.contains_batch_local,
                                    stats_fn=worker._stats_snapshot)
        server.start()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                sock.connect(server.sock_path)
                sock.sendall(_LOOKUP_HDR.pack(_STATS_DRAIN))
                with sock.makefile("rb") as response:
                    size, = _LOOKUP_HDR.unpack(response.read(_LOOKUP_HDR.size))
                    stats = json.loads(response.read(size))
            assert stats["puts"] == stats["put_attempts"] == 3
            assert stats["dram_evictions"] == stats["dram_reclaims"] == 1
            assert stats["dram_frames"] == 2
            assert stats["hbm_frames"] == stats["host_pinned"] == 0
            assert not stats["direct_slab_dma"]
            assert stats["num_kv_groups"] == 1 and stats["block_size"] == 16
        finally:
            server.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for pinned DMA")
def test_zero_hbm_paged_dma_round_trip():
    with VeriCacheClient(node_id=0, frame_size=4096, dram_frames=4, gpu_device_id=0) as client:
        stats = client.stats()
        assert stats.hbm_frames == 0 and stats.host_pinned == 1
        buf = torch.arange(4096, device="cuda", dtype=torch.int16).to(torch.uint8).repeat(2, 1)
        buf[1].zero_()
        stream = torch.cuda.current_stream().cuda_stream
        args = ([key_from_u64(9)], [buf.data_ptr()], [4096], [0], [4096])
        assert client.put_gpu_paged_dram_batch(*args, [0], cuda_stream=stream) == 1
        flags = client.get_gpu_paged_batch_flags(*args, [1], cuda_stream=stream)
        assert list(flags) == [HitLocation.LOCAL_DRAM]
        assert torch.equal(buf[0], buf[1])



def test_hma_capacity_matches_dram_plus_nvme_even_with_padding(monkeypatch):
    class Scheduler:
        def __init__(self, _config, gpu_config, cpu_capacity_bytes, *, scheduler_block_size,
                     hash_block_size, lazy_offload, disk_capacity_bytes, use_page_cache,
                     num_cpu_blocks):
            assert scheduler_block_size == hash_block_size == 16
            assert cpu_capacity_bytes == disk_capacity_bytes
            allocation_bytes = gpu_config.kv_cache_tensors[0].size
            self.num_cpu_blocks = gpu_config.num_blocks * disk_capacity_bytes // allocation_bytes
            assert self.num_cpu_blocks == num_cpu_blocks == 6

    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", SimpleNamespace(
        resolve_kv_cache_block_sizes=lambda *_: (16, 16),
    ))
    monkeypatch.setitem(sys.modules, "vllm.v1.simple_kv_offload.manager", SimpleNamespace(
        SimpleCPUOffloadScheduler=Scheduler,
    ))
    connector = SimpleNamespace()
    VeriCacheConnector._init_hma_scheduler(
        connector,
        SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True)),
        SimpleNamespace(num_blocks=3, kv_cache_tensors=[SimpleNamespace(size=100)]),
        {"hbm_frames": 0, "dram_frames": 4, "nvme_frames": 2,
         "nvme_dir": "scratch", "enable_local_lookup": False},
    )
    assert connector._hma_scheduler.num_cpu_blocks == 6
