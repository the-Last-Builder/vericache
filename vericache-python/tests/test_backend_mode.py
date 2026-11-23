"""Run each backend mode transition in its own process."""

from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
from textwrap import dedent

import pytest


_PRELUDE = """
from contextlib import contextmanager
from vericache import (
    NativeVeriCacheEngine, NativeVeriCacheError, PinnedStagingBuffer,
    VeriCacheClient, VeriCacheError, key_from_u64,
)

def native():
    return NativeVeriCacheEngine(
        num_shards=2, host_frames_per_shard=4, frame_size=16,
        prefer_cuda_pinned=False,
    )

def distributed():
    return VeriCacheClient(node_id=1, frame_size=16, dram_frames=4)

@contextmanager
def rejects(error, message):
    try:
        yield
    except error as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("initialization unexpectedly succeeded")
"""


def _run(code):
    result = subprocess.run(
        [sys.executable, "-c", dedent(_PRELUDE) + "\n" + dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "panicked at" not in result.stderr


@pytest.mark.parametrize("first", ["engine", "staging"])
def test_local_then_distributed(first):
    _run(f"""
        with rejects(VeriCacheError, "invalid frame_size or dram_frames"):
            VeriCacheClient(node_id=1, frame_size=0, dram_frames=4)
        with {"native()" if first == "engine" else "PinnedStagingBuffer(16, False)"}:
            with rejects(VeriCacheError, "Distributed VeriCache backend after Local backend"):
                distributed()
    """)


def test_distributed_then_local():
    _run("""
        with distributed() as cache:
            for create in (native, lambda: PinnedStagingBuffer(16, False)):
                with rejects(NativeVeriCacheError, "Local VeriCache backend after Distributed backend"):
                    create()
            assert cache.stats().puts == 0
    """)


def test_multiple_local_handles():
    _run("""
        with native() as one, native() as two, PinnedStagingBuffer(16, False) as stage:
            one.put((42, 1, 0, 0), b"first")
            two.put((42, 1, 0, 0), b"second")
            assert one.get_into_staging((42, 1, 0, 0), stage) == 5
            assert bytes(stage.memoryview[:5]) == b"first"
            assert two.get((42, 1, 0, 0)) == b"second"
    """)


def test_multiple_distributed_handles():
    _run("""
        from vericache._ffi import load_lib
        assert not load_lib().vericache_staging_alloc(0, 0)
        with distributed() as one, distributed() as two:
            key = key_from_u64(42)
            one.put_host(key, b"a" * 16)
            two.put_host(key, b"b" * 16)
            assert one.get_host(key)[1] == b"a" * 16
            assert two.get_host(key)[1] == b"b" * 16
    """)


def test_direct_distributed_connector_with_real_vllm():
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        pytest.skip("vLLM is unavailable")
    if installed != "0.13.0":
        pytest.skip("requires vLLM 0.13.0")
    _run("""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1, KVConnectorRole,
        )
        from vericache.vllm_connector import VeriCacheConnector, VeriCacheConnectorMetadata
        with distributed() as cache:
            connector = VeriCacheConnector(None, KVConnectorRole.WORKER, client=cache)
            assert isinstance(connector, KVConnectorBase_V1)
            assert connector.role is KVConnectorRole.WORKER and connector.client is cache
            meta = VeriCacheConnectorMetadata()
            connector.bind_connector_metadata(meta)
            assert connector._get_connector_metadata() is meta
            connector.clear_connector_metadata()
    """)
