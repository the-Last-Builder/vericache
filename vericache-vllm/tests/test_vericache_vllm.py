from concurrent.futures import ThreadPoolExecutor
import ctypes
from types import SimpleNamespace

import pytest
import torch
import vllm
from vllm.config import CacheConfig, DeviceConfig, VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.forward_context import ForwardContext
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vericache_vllm import NativeVeriCacheEngine, PinnedStagingBuffer, VeriCacheConfig
from vericache_vllm._native import NativeVeriCacheError
from vericache_vllm.connector import VeriCacheConnectorMetadata, VeriCacheV1Connector
from vericache_vllm.hashing import compute_prefix_block_hashes, derive_layer_block_key


def test_aligned_staging_aliases_rust_and_survives_close():
    with PinnedStagingBuffer(4096, prefer_cuda_pinned=False) as stage:
        assert not stage.is_cuda_pinned and stage.ptr % 4096 == 0
        assert len(stage) == len(stage.memoryview) == 4096
        raw = stage.memoryview
        tensor = stage.as_tensor(torch.int32, (4, 4))
        assert tensor.data_ptr() == stage.ptr
        value = ctypes.c_int32.from_address(stage.ptr)
        value.value = -31
        assert tensor[0, 0].item() == -31
        tensor[0, 1] = 17
        assert int.from_bytes(raw[4:8], "little") == 17
    with pytest.raises(NativeVeriCacheError, match="closed"):
        stage.as_tensor(torch.int32, (1,))
    raw[4] = 19
    assert tensor[0, 1].item() == 19


def test_native_tensor_roundtrip_and_exact_stats():
    with (
        NativeVeriCacheEngine(
            num_shards=2,
            host_frames_per_shard=8,
            frame_size=128,
            prefer_cuda_pinned=False,
        ) as engine,
        engine.make_staging() as stage,
    ):
        assert engine.num_shards == 2
        assert engine.frame_size == len(stage) == 128
        src = torch.arange(30, dtype=torch.bfloat16).view(5, 6).T
        dst = torch.zeros((5, 6), dtype=src.dtype).T
        assert not engine.contains(7)
        assert not engine.get_tensor(7, dst, stage)
        engine.put_tensor(7, src, stage)
        assert engine.contains(7)
        assert engine.get_into_staging(7, stage) == 60
        assert torch.equal(stage.as_tensor(src.dtype, src.shape), src)
        assert engine.get_tensor(7, dst, stage)
        assert torch.equal(dst, src)
        engine.put(18, b"")
        assert engine.get(18) == b""
        assert engine.evict(7)
        assert not engine.evict(7)
        assert engine.stats() == {
            "puts": 2,
            "gets": 4,
            "hits": 3,
            "misses": 1,
            "evictions": 1,
            "bytes_written": 60,
            "bytes_read": 120,
        }
    with pytest.raises(NativeVeriCacheError, match="closed"):
        engine.contains(7)


def test_native_eviction_and_undersized_destination():
    with NativeVeriCacheEngine(
        num_shards=1, host_frames_per_shard=2, frame_size=8
    ) as engine:
        for key in range(20):
            engine.put(key, bytes([key]) * 5)
        assert engine.get(17) is None
        assert engine.get(18) == bytes([18]) * 5
        with engine.make_staging(4) as short:
            with pytest.raises(ValueError, match="5 bytes"):
                engine.get_into_staging(19, short)
        assert engine.get(19) == bytes([19]) * 5
        assert engine.stats()["evictions"] == 18


def test_native_dram_nvme_spill(tmp_path):
    backing = tmp_path / "cache.nvme"
    with NativeVeriCacheEngine(
        num_shards=1,
        host_frames_per_shard=1,
        nvme_frames_per_shard=2,
        frame_size=67,
        nvme_path=backing,
        prefer_cuda_pinned=False,
    ) as engine:
        assert backing.stat().st_size == 8192
        for key in range(3):
            engine.put(key, bytes([key]) * (key + 63))
        assert engine.stats()["evictions"] == 0
        for key in (0, 2, 1, 0):
            assert engine.get(key) == bytes([key]) * (key + 63)
        engine.put(3, b"last")
        assert engine.get(1) is None
        assert engine.get(0) == b"\x00" * 63
        assert engine.get(2) == b"\x02" * 65
        assert engine.get(3) == b"last"
        assert engine.stats()["evictions"] == 1


def test_native_multithreaded():
    with NativeVeriCacheEngine(
        num_shards=8,
        host_frames_per_shard=128,
        frame_size=32,
        prefer_cuda_pinned=False,
    ) as engine:

        def worker(worker_id):
            with engine.make_staging() as stage:
                for i in range(64):
                    key = (worker_id << 32) | i
                    data = key.to_bytes(8, "little") * 4
                    stage.memoryview[:] = data
                    engine.put_from_staging(key, stage)
                    stage.memoryview[:] = bytes(32)
                    assert engine.get_into_staging(key, stage) == 32
                    assert stage.memoryview == data

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))
        assert engine.stats()["hits"] == 512
        assert engine.stats()["evictions"] == 0


def test_prefix_chain_and_layer_keys():
    tokens = [1, 2, 3, 4, 5, 6, 7]
    base = compute_prefix_block_hashes(tokens, 2, "model")
    assert len(base) == 3
    assert all(0 < h < 2**64 for h in base)
    assert base == compute_prefix_block_hashes(tuple(tokens), 2, "model")
    assert base[:2] == compute_prefix_block_hashes(tokens[:4], 2, "model")
    assert base == compute_prefix_block_hashes(tokens[:-1] + [8], 2, "model")
    changed_prefix = compute_prefix_block_hashes([8, *tokens[1:]], 2, "model")
    assert all(a != b for a, b in zip(base, changed_prefix))
    assert base != compute_prefix_block_hashes(tokens, 2, "other-model")
    layer_keys = {
        derive_layer_block_key(h, layer) for h in base for layer in ("l0", "l1")
    }
    assert len(layer_keys) == 6 and 0 not in layer_keys
    assert derive_layer_block_key(base[0], "l0") == derive_layer_block_key(
        base[0], "l0"
    )
    assert compute_prefix_block_hashes([], 2) == []
    with pytest.raises(ValueError, match="positive"):
        compute_prefix_block_hashes(tokens, 0)


def test_config_reads_vllm_extras():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "num_shards": 5,
                "host_frames_per_shard": 13,
                "nvme_frames_per_shard": 2,
                "frame_size": 256,
                "nvme_path": "/example/cache",
                "prefer_cuda_pinned": False,
                "lib_path": "/example/libvericache_vllm.so",
                "model_tag": "model",
                "block_axis": 1,
            }
        )
    )
    assert VeriCacheConfig.from_vllm_config(vllm_config) == VeriCacheConfig(
        5,
        13,
        2,
        256,
        "/example/cache",
        False,
        "/example/libvericache_vllm.so",
        "model",
        1,
    )


BLOCK_SIZE = 16
NUM_BLOCKS = 16
LAYERS = ("model.layers.0.attn", "model.layers.1.attn")
CONNECTOR = "VeriCacheV1Connector"
MODULE = "vericache_vllm.connector"


def _connector_pair(monkeypatch, registration, *, prefer_pinned=False):
    assert vllm.__version__ == "0.13.0"
    registry = dict(KVConnectorFactory._registry)
    registry.pop(CONNECTOR, None)
    monkeypatch.setattr(KVConnectorFactory, "_registry", registry)
    if registration == "registry":
        KVConnectorFactory.register_connector(CONNECTOR, MODULE, CONNECTOR)
    extras = {
        "num_shards": 2,
        "host_frames_per_shard": 32,
        "frame_size": 1024,
        "prefer_cuda_pinned": prefer_pinned,
    }
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(block_size=BLOCK_SIZE),
        kv_transfer_config=KVTransferConfig(
            kv_connector=CONNECTOR,
            kv_role="kv_both",
            kv_connector_extra_config=extras,
            kv_connector_module_path=MODULE if registration == "module" else None,
        ),
    )
    for name, backend in zip(LAYERS, (FlashAttentionBackend, TritonAttentionBackend)):
        config.compilation_config.static_forward_context[name] = SimpleNamespace(
            get_attn_backend=lambda backend=backend: backend,
        )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=3, dtype=torch.float32
    )
    kv_cache_config = KVCacheConfig(
        NUM_BLOCKS, [], [KVCacheGroupSpec(list(LAYERS), spec)]
    )
    scheduler = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER, kv_cache_config
    )
    worker = KVConnectorFactory.create_connector(
        config, KVConnectorRole.WORKER, kv_cache_config
    )
    assert isinstance(scheduler, VeriCacheV1Connector)
    assert isinstance(worker, VeriCacheV1Connector)
    assert scheduler.engine is worker.engine
    return scheduler, worker


def _request(name, tokens, *, cache_salt=None):
    return Request(
        request_id=name,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=None,
        cache_salt=cache_salt,
    )


def _blocks(ids):
    return KVCacheBlocks(([KVCacheBlock(block_id=idx) for idx in ids],))


def _schedule_new(scheduler, request, ids, *, local=0, scheduled=None):
    external, asynchronous = scheduler.get_num_new_matched_tokens(request, local)
    assert asynchronous is False
    blocks = _blocks(ids)
    scheduler.update_state_after_alloc(request, blocks, external)
    request.num_computed_tokens = local + external
    count = (
        len(request.prompt_token_ids) - request.num_computed_tokens
        if scheduled is None
        else scheduled
    )
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData.from_request(request, blocks.get_block_ids())
    ]
    output.num_scheduled_tokens[request.request_id] = (
        output.total_num_scheduled_tokens
    ) = count
    output.kv_connector_metadata = scheduler.build_connector_meta(output)
    assert isinstance(output.kv_connector_metadata, VeriCacheConnectorMetadata)
    return external, output.kv_connector_metadata


def _register_worker(worker, device):
    # Alternate [K/V, blocks, ...] with [blocks, K/V, ...] and include a
    # non-contiguous layer view, as with vLLM's stride-order backends.
    dims = (NUM_BLOCKS, 2, BLOCK_SIZE, 2, 3)
    values = torch.arange(
        torch.Size(dims).numel(), dtype=torch.float32, device=device
    ).reshape(dims)
    first = values.transpose(0, 1)
    second = values.clone() + 1000
    kv = dict(zip(LAYERS, (first, second)))
    worker.register_kv_caches(kv)
    context = ForwardContext(
        {name: SimpleNamespace(kv_cache=[tensor]) for name, tensor in kv.items()},
        {name: object() for name in kv},
        0,
    )
    return kv, context


def _view(cache, layer, block_id):
    return cache.select(int(layer == LAYERS[0]), block_id)


def _save_or_load(worker, meta, kv, context, compute=None):
    worker.bind_connector_metadata(meta)
    worker.start_load_kv(context)
    for name, cache in kv.items():
        worker.wait_for_layer_load(name)
        if compute is not None:
            compute(name, cache)
        worker.save_kv_layer(name, cache, context.attn_metadata[name])
    worker.wait_for_save()
    assert worker.get_finished(set()) == (None, None)
    errors = worker.get_block_ids_with_load_errors()
    worker.clear_connector_metadata()
    return errors


@pytest.mark.parametrize("registration", ["module", "registry"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA device unavailable"
            ),
        ),
    ],
)
def test_factory_multilayer_cold_warm_and_partial_prefix(
    monkeypatch, registration, device
):
    scheduler, worker = _connector_pair(
        monkeypatch, registration, prefer_pinned=device == "cuda"
    )
    try:
        kv, context = _register_worker(worker, device)
        tokens = list(range(3 * BLOCK_SIZE + 1))
        cold = _request("cold", tokens)
        external, meta = _schedule_new(scheduler, cold, [1, 2, 3, 4])
        assert external == 0
        assert [block.block_id for block in meta.save] == [1, 2, 3]
        expected = {
            layer: [_view(cache, layer, i).clone() for i in (1, 2, 3)]
            for layer, cache in kv.items()
        }
        assert _save_or_load(worker, meta, kv, context) == set()
        assert scheduler.request_finished(cold, [1, 2, 3, 4]) == (False, None)

        for cache in kv.values():
            cache.zero_()
        warm = _request("warm", tokens)
        external, meta = _schedule_new(scheduler, warm, [5, 6, 7, 8])
        assert external == 3 * BLOCK_SIZE
        assert not meta.save
        assert _save_or_load(worker, meta, kv, context) == set()
        for layer, cache in kv.items():
            for i, source in enumerate(expected[layer], 5):
                assert torch.equal(_view(cache, layer, i), source)
            assert torch.count_nonzero(_view(cache, layer, 8)).item() == 0
        assert scheduler.get_num_new_matched_tokens(warm, BLOCK_SIZE) == (
            2 * BLOCK_SIZE,
            False,
        )
        assert scheduler.get_num_new_matched_tokens(
            _request("aligned", tokens[:-1]), 0
        ) == (2 * BLOCK_SIZE, False)
        assert scheduler.get_num_new_matched_tokens(
            _request("salted", tokens, cache_salt="other"), 0
        ) == (0, False)

        for cache in kv.values():
            cache.zero_()
        changed = tokens.copy()
        changed[2 * BLOCK_SIZE] = 987
        partial = _request("partial", changed)
        external, meta = _schedule_new(scheduler, partial, [9, 10, 11, 12])
        assert external == 2 * BLOCK_SIZE
        assert [block.block_id for block in meta.load] == [9, 10]
        assert [block.block_id for block in meta.save] == [11]

        def compute_suffix(layer, cache):
            _view(cache, layer, 11).fill_(21 if layer == LAYERS[0] else 41)

        assert _save_or_load(worker, meta, kv, context, compute_suffix) == set()
        for layer, cache in kv.items():
            for i, source in enumerate(expected[layer][:2], 9):
                assert torch.equal(_view(cache, layer, i), source)
            assert torch.count_nonzero(_view(cache, layer, 12)).item() == 0
        assert scheduler.get_num_new_matched_tokens(partial, 0) == (
            3 * BLOCK_SIZE,
            False,
        )
    finally:
        worker.shutdown()
        scheduler.shutdown()


def test_chunked_and_resumed_requests_with_local_prefix(monkeypatch):
    scheduler, worker = _connector_pair(monkeypatch, "module")
    try:
        kv, context = _register_worker(worker, "cpu")
        tokens = list(range(3 * BLOCK_SIZE + 1))
        request = _request("chunked", tokens)
        external, meta = _schedule_new(
            scheduler, request, [1], scheduled=BLOCK_SIZE - 1
        )
        assert external == 0 and not meta.save

        output = SchedulerOutput.make_empty()
        output.scheduled_cached_reqs = CachedRequestData(
            ["chunked"],
            set(),
            [],
            {},
            [([2],)],
            [BLOCK_SIZE - 1],
            [0],
        )
        output.num_scheduled_tokens = {"chunked": BLOCK_SIZE + 1}
        meta = scheduler.build_connector_meta(output)
        assert [block.block_id for block in meta.save] == [1, 2]
        assert _save_or_load(worker, meta, kv, context) == set()

        expected = {
            layer: _view(cache, layer, 2).clone() for layer, cache in kv.items()
        }
        external, is_async = scheduler.get_num_new_matched_tokens(request, BLOCK_SIZE)
        assert (external, is_async) == (BLOCK_SIZE, False)
        scheduler.update_state_after_alloc(request, _blocks([5, 6, 7]), external)
        output.scheduled_cached_reqs = CachedRequestData(
            ["chunked"],
            {"chunked"},
            [],
            {"chunked": tokens},
            [([5, 6, 7],)],
            [2 * BLOCK_SIZE],
            [0],
        )
        output.num_scheduled_tokens = {"chunked": BLOCK_SIZE}
        meta = scheduler.build_connector_meta(output)
        assert [block.block_id for block in meta.load] == [6]
        assert [block.block_id for block in meta.save] == [7]
        assert _save_or_load(worker, meta, kv, context) == set()
        for layer, cache in kv.items():
            assert torch.equal(_view(cache, layer, 6), expected[layer])
    finally:
        worker.shutdown()
        scheduler.shutdown()


def test_missing_layer_blocks_prefix_and_late_eviction_is_reported(monkeypatch):
    scheduler, worker = _connector_pair(monkeypatch, "module")
    try:
        kv, context = _register_worker(worker, "cpu")
        tokens = list(range(2 * BLOCK_SIZE + 1))
        request = _request("cold", tokens)
        _, cold = _schedule_new(scheduler, request, [1, 2, 3])
        assert _save_or_load(worker, cold, kv, context) == set()

        warm = _request("warm", tokens)
        external, meta = _schedule_new(scheduler, warm, [4, 5, 6])
        assert external == 2 * BLOCK_SIZE
        missing = derive_layer_block_key(cold.save[1].block_hash, LAYERS[1])
        assert worker.engine.evict(missing)
        assert scheduler.get_num_new_matched_tokens(warm, 0) == (BLOCK_SIZE, False)
        assert _save_or_load(worker, meta, kv, context) == {5}
        assert worker.get_block_ids_with_load_errors() == set()
    finally:
        worker.shutdown()
        scheduler.shutdown()


def test_read_lease_aliases_slab_and_defers_release_for_exported_views():
    engine = NativeVeriCacheEngine(
        num_shards=1, host_frames_per_shard=1, frame_size=4096, prefer_cuda_pinned=False
    )
    engine.put(1, bytes(range(16)))
    with engine.acquire_read_lease(1) as lease:
        view = lease.memoryview
        tensor = lease.as_tensor(torch.uint8, (4, 4))
        assert view.readonly
        assert tensor.data_ptr() == lease.ptr
        assert lease.ptr % 4096 == 0
        assert not lease.is_cuda_pinned
        assert tensor.flatten().tolist() == list(range(16))
        with engine.acquire_read_lease(1) as duplicate:
            assert duplicate.ptr == lease.ptr and duplicate.lease_id != lease.lease_id
        with pytest.raises(NativeVeriCacheError, match="status -4"):
            engine.put(2, b"cannot evict a leased frame")
        with pytest.raises(NativeVeriCacheError, match="status -4"):
            engine.evict(1)
    lease.release()
    assert engine.extended_stats()["active_leases"] == 1
    with pytest.raises(NativeVeriCacheError, match="released"):
        lease.as_tensor(torch.uint8, (16,))
    view.release()
    assert engine.extended_stats()["active_leases"] == 1
    del tensor
    assert engine.extended_stats()["active_leases"] == 0
    assert engine.evict(1)
    engine.put(2, b"data")
    lease = engine.acquire_read_lease(2)
    tensor = lease.as_tensor(torch.uint8, (4,))
    engine.close()
    with pytest.raises(NativeVeriCacheError, match="closed"):
        engine.contains(2)
    lease.release()
    assert bytes(tensor.tolist()) == b"data"
    del tensor
    assert engine._handle is None


def test_native_batches_preserve_order_misses_and_lengths():
    with NativeVeriCacheEngine(
        num_shards=4, host_frames_per_shard=32, frame_size=64, prefer_cuda_pinned=False
    ) as engine:
        source = [
            ctypes.create_string_buffer(payload) for payload in (b"first", b"two")
        ]
        assert engine.put_batch(
            [17, 39, 91], [*(ctypes.addressof(buf) for buf in source), None], [5, 3, 0]
        ) == [0, 0, 0]
        assert engine.contains_batch([39, 2, 17, 91]) == [True, False, True, True]
        first, miss, second, empty = engine.acquire_read_lease_batch([17, 2, 39, 91])
        assert miss is None and empty is not None and len(empty) == 0
        assert bytes(first.memoryview) == b"first"
        assert bytes(second.memoryview) == b"two"
        assert engine.extended_stats()["active_leases"] == 3
        engine.release_read_lease_batch(
            [second.lease_id, empty.lease_id, first.lease_id]
        )
        assert engine.extended_stats()["active_leases"] == 0
        target = ctypes.create_string_buffer(5)
        lengths, statuses = engine.get_batch(
            [17, 39, 2], [ctypes.addressof(target), None, None], [5, 0, 0]
        )
        assert (lengths, statuses, target.raw) == ([5, 3, 0], [0, -3, 1], b"first")
        assert engine.contains_batch([]) == []


def test_lru_and_eager_writeback_enable_clean_eviction_and_promotion(tmp_path):
    from time import monotonic, sleep

    with NativeVeriCacheEngine(
        num_shards=1, host_frames_per_shard=2, frame_size=32, prefer_cuda_pinned=False
    ) as engine:
        engine.put(1, b"first")
        engine.put(2, b"second")
        assert engine.get(1) == b"first"
        engine.put(3, b"third")
        assert engine.contains_batch([1, 2, 3]) == [True, False, True]
    with NativeVeriCacheEngine(
        num_shards=1,
        host_frames_per_shard=4,
        nvme_frames_per_shard=8,
        nvme_path=tmp_path / "phase2.nvme",
        frame_size=4096,
        prefer_cuda_pinned=False,
    ) as engine:
        engine.set_writeback_policy(75)
        for key in range(3):
            engine.put(key, bytes([key]) * 4096)
        deadline = monotonic() + 3
        settled = 0
        while engine.extended_stats()["async_writebacks"] < 3:
            settled += engine.poll_writeback()
            assert monotonic() < deadline
            sleep(0.001)
        assert settled > 0
        for key in (3, 4):
            engine.put(key, bytes([key]) * 4096)
        assert engine.extended_stats()["clean_dram_evictions"] > 0
        with engine.acquire_read_lease(0) as lease:
            assert bytes(lease.memoryview) == bytes(4096)
        promoted = engine.extended_stats()
        assert promoted["nvme_promotions"] == 1
        assert promoted["nvme_hits"] == 1
        assert engine.get(0) == bytes(4096)
        assert engine.extended_stats()["dram_hits"] == promoted["dram_hits"] + 1


def test_async_layer_load_waits_only_for_the_requested_layer(monkeypatch):
    from threading import Event

    scheduler, worker = _connector_pair(monkeypatch, "module")
    first_finished, second_started, continue_second = Event(), Event(), Event()
    try:
        assert worker.config.io_workers == 2 and worker.config.use_zero_copy_load
        with torch.inference_mode():
            kv, context = _register_worker(worker, "cpu")
        tokens = list(range(2 * BLOCK_SIZE + 1))
        _, cold = _schedule_new(scheduler, _request("cold-async", tokens), [1, 2, 3])
        expected = {
            (layer, index): _view(kv[layer], layer, block.block_id).clone()
            for index, block in enumerate(cold.save)
            for layer in LAYERS
        }
        assert _save_or_load(worker, cold, kv, context) == set()
        n, warm = _schedule_new(scheduler, _request("warm-async", tokens), [4, 5, 6])
        assert n == 2 * BLOCK_SIZE
        second_keys = {
            derive_layer_block_key(block.block_hash, LAYERS[1]) for block in warm.load
        }
        acquire = worker.engine.acquire_read_lease_batch

        def gated_acquire(keys, **kwargs):
            keys = list(keys)
            if set(keys) == second_keys:
                second_started.set()
                assert continue_second.wait(5)
            return acquire(keys, **kwargs)

        monkeypatch.setattr(worker.engine, "acquire_read_lease_batch", gated_acquire)
        worker.bind_connector_metadata(warm)
        worker.start_load_kv(context)
        assert second_started.wait(5)

        def wait_first():
            worker.wait_for_layer_load(LAYERS[0])
            first_finished.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wait_first)
            try:
                assert first_finished.wait(3)
                for index, block in enumerate(warm.load):
                    first = LAYERS[0]
                    assert torch.equal(
                        _view(kv[first], first, block.block_id), expected[first, index]
                    )
            finally:
                continue_second.set()
            future.result()
        worker.wait_for_save()
        assert worker.get_block_ids_with_load_errors() == set()
        assert worker.engine.extended_stats()["active_leases"] == 0
        for index, block in enumerate(warm.load):
            second = LAYERS[1]
            assert torch.equal(
                _view(kv[second], second, block.block_id), expected[second, index]
            )
    finally:
        continue_second.set()
        worker.shutdown()
        scheduler.shutdown()


def _ipc_connector(endpoint, nvme_path, rank=None):
    extras = dict(
        ipc_endpoint=endpoint,
        tp_rank=rank,
        world_size=2,
        num_shards=1,
        host_frames_per_shard=16,
        nvme_frames_per_shard=16,
        nvme_path=nvme_path,
        frame_size=1024,
        prefer_cuda_pinned=False,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp", world_size=2
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={
                layer: SimpleNamespace(get_attn_backend=lambda backend=backend: backend)
                for layer, backend in zip(
                    LAYERS, (FlashAttentionBackend, TritonAttentionBackend)
                )
            }
        ),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extras),
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=3, dtype=torch.float32
    )
    kv_cfg = KVCacheConfig(NUM_BLOCKS, [], [KVCacheGroupSpec(list(LAYERS), spec)])
    role = KVConnectorRole.SCHEDULER if rank is None else KVConnectorRole.WORKER
    return VeriCacheV1Connector(config, role, kv_cfg)


def _ipc_process(endpoint, nvme_path, pipe, rank):
    import os
    import traceback

    torch.set_num_threads(1)
    connector = None
    try:
        connector = _ipc_connector(endpoint, nvme_path, rank)
        if rank is not None:
            kv, context = _register_worker(connector, "cpu")
            for cache in kv.values():
                cache.add_(rank * 10_000)
            source = {name: cache.clone() for name, cache in kv.items()}
        pipe.send(("ok", os.getpid()))
        while True:
            op, value = pipe.recv()
            if op == "stop":
                break
            if op == "plan":
                request_id, ids = value
                request = _request(request_id, list(range(2 * BLOCK_SIZE + 1)))
                result = _schedule_new(connector, request, ids)
            elif op == "save":
                result = _save_or_load(connector, value, kv, context)
            elif op == "load":
                meta, cold = value
                for cache in kv.values():
                    cache.zero_()
                errors = _save_or_load(connector, meta, kv, context)
                result = (
                    errors,
                    all(
                        torch.equal(
                            _view(kv[layer], layer, target.block_id),
                            _view(source[layer], layer, original.block_id),
                        )
                        for layer in LAYERS
                        for target, original in zip(meta.load, cold.save)
                    ),
                    connector.engine.extended_stats()["active_leases"],
                )
            elif op == "evict":
                result = connector.engine.evict(connector._layer_key(value, LAYERS[1]))
                connector.wait_for_save()
            else:
                raise AssertionError(op)
            pipe.send(("ok", result))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        raise
    finally:
        if connector is not None:
            connector.shutdown()
        pipe.close()


def test_multiprocess_tp2_prefix_requires_both_ranks_and_restores_each_slice(tmp_path):
    import multiprocessing
    import os

    context = multiprocessing.get_context("spawn")
    endpoint = f"unix://{tmp_path / 'metadata.sock'}"
    backing = str(tmp_path / "kv")
    processes = []

    def launch(rank):
        parent, child = context.Pipe()
        process = context.Process(
            target=_ipc_process, args=(endpoint, backing, child, rank)
        )
        process.start()
        child.close()
        processes.append((process, parent))
        return parent

    def receive(pipe):
        assert pipe.poll(45), "VeriCache child timed out"
        status, data = pipe.recv()
        assert status == "ok", data
        return data

    def call(pipe, op, value=None):
        pipe.send((op, value))
        return receive(pipe)

    try:
        # vLLM sets up the workers' KV tensors before creating the scheduler.
        workers = [launch(0), launch(1)]
        worker_pids = [receive(worker) for worker in workers]
        scheduler = launch(None)
        scheduler_pid = receive(scheduler)
        assert len({os.getpid(), scheduler_pid, *worker_pids}) == 4
        hits, cold = call(scheduler, "plan", ("tp-cold", [1, 2, 3]))
        assert hits == 0 and len(cold.save) == 2
        assert call(workers[0], "save", cold) == set()
        assert call(scheduler, "plan", ("tp-half", [1, 2, 3]))[0] == 0
        assert call(workers[1], "save", cold) == set()
        hits, warm = call(scheduler, "plan", ("tp-warm", [4, 5, 6]))
        assert hits == 2 * BLOCK_SIZE
        for rank, worker in enumerate(workers):
            assert (tmp_path / f"kv.rank-{rank}").is_file()
            assert call(worker, "load", (warm, cold)) == (set(), True, 0)
        assert call(workers[1], "evict", cold.save[1].block_hash)
        assert call(scheduler, "plan", ("tp-eviction", [4, 5, 6]))[0] == BLOCK_SIZE
    finally:
        # Close the workers before their scheduler-owned metadata server.
        for process, pipe in reversed(processes):
            if process.is_alive():
                try:
                    pipe.send(("stop", None))
                except OSError:
                    pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()
    assert all(process.exitcode == 0 for process, _ in processes)
