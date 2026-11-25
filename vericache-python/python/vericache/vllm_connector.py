"""vLLM v1 `KVConnectorBase_V1` + `SupportsHMA` integration backed by VeriCache.

Supports two usage modes through the same `VeriCacheConnector` class:
1. **Real vLLM v1 serving mode** (`VeriCacheConnector(vllm_config, role, kv_cache_config)`):
   - Loaded via `--kv-transfer-config '{"kv_connector":"VeriCacheConnector",
     "kv_connector_module_path":"vericache.vllm_connector","kv_role":"kv_both",
     "kv_connector_extra_config":{...}}'`.
   - Implements `SupportsHMA` (`request_finished_all_groups`, lookahead block
     hashes, hybrid FullAttention + SlidingWindow + Mamba/short-conv groups,
     such as Thinking Machines' looped hybrid `Inkling-2-Mini`).
   - Works in both single-process and multiprocess vLLM v1: the Scheduler role
     uses vLLM's hybrid `SimpleCPUOffloadScheduler` to compute exact hybrid
     block hashes (`BlockHashWithGroupId`) and transfer schedules, while the
     Worker role owns a `VeriCacheClient` (Local HBM -> Pinned DRAM -> NVMe
     `io_uring` `O_DIRECT` -> Remote 2-GPU GPUDirect RDMA) and executes
     content-addressed GPU HBM-to-HBM `put_gpu_batch` / `get_gpu_batch`
     transfers keyed by SHA-256 of each block's `BlockHashWithGroupId`.
2. **Standalone direct-client mode** (`VeriCacheConnector(vllm_config, role, client=...)`):
   - Preserved for standalone PyTorch unit tests and disaggregated 2-GPU GDR
     Prefill/Decode tests without starting a full vLLM engine.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import logging
import os
import socket
import struct
import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence

from . import HitInfo, HitLocation, ObjectKey, PeerSpec, VeriCacheClient, key_from_bytes
from .directory import ClusterDirectoryClient

logger = logging.getLogger("vericache.vllm_connector")

_HAS_VLLM = False
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        KVConnectorWorkerMetadata,
        SupportsHMA,
    )
    from vllm.v1.metrics.cache_hit_source import CacheHitSource  # type: ignore
    from vllm.v1.simple_kv_offload.metadata import (  # type: ignore
        SimpleCPUOffloadWorkerMetadata,
    )

    _HAS_VLLM = True
except ImportError:

    @dataclass
    class SimpleCPUOffloadWorkerMetadata:  # type: ignore[no-redef]
        completed_store_events: dict[int, int] = field(default_factory=dict)

    class KVConnectorRole(Enum):  # type: ignore[no-redef]
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:  # type: ignore[no-redef]
        pass

    class KVConnectorWorkerMetadata:  # type: ignore[no-redef]
        def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KVConnectorWorkerMetadata":
            return self

    class SupportsHMA:  # type: ignore[no-redef]
        pass

    class CacheHitSource(Enum):  # type: ignore[no-redef]
        HOST = "host"
        DISK = "disk"

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        def __init__(
            self,
            vllm_config: Any,
            role: KVConnectorRole,
            kv_cache_config: Any = None,
        ) -> None:
            self._vllm_config = vllm_config
            self._role = role
            self._kv_cache_config = kv_cache_config
            self._connector_metadata: Optional[KVConnectorMetadata] = None

        def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self) -> KVConnectorMetadata:
            assert self._connector_metadata is not None
            return self._connector_metadata


# ---------------------------------------------------------------------------
# Prefix-chained deterministic hashing for (token_prefix, layer_name) -> ObjectKey
# ---------------------------------------------------------------------------


def hash_prefix_blocks(token_ids: Sequence[int], block_size: int) -> list[bytes]:
    """Compute chained SHA-256 prefix digests for every full `block_size` chunk."""
    num_full = len(token_ids) // block_size
    digests: list[bytes] = []
    parent = b"\x00" * 32
    for blk_idx in range(num_full):
        start = blk_idx * block_size
        chunk = token_ids[start : start + block_size]
        h = hashlib.sha256()
        h.update(parent)
        h.update(struct.pack(f"<{block_size}Q", *[t & 0xFFFFFFFFFFFFFFFF for t in chunk]))
        parent = h.digest()
        digests.append(parent)
    return digests


def layer_block_key(prefix_digest: bytes, layer_name: str) -> ObjectKey:
    """Derive a 256-bit VeriCache ObjectKey for `(prefix_digest, layer_name)`."""
    h = hashlib.sha256()
    h.update(prefix_digest)
    h.update(b"::")
    h.update(layer_name.encode("utf-8"))
    return struct.unpack("<4Q", h.digest())


_OBJECT_KEY_CACHE: dict[Any, ObjectKey] = {}
_OBJECT_KEY_CACHE_MAX = 1_000_000
_OBJECT_KEY_CACHE_EVICT = 200_000


def object_key_from_vllm_block_hash(block_hash_with_group_id: bytes | int) -> ObjectKey:
    """Derive a deterministic 256-bit VeriCache ObjectKey from a vLLM BlockHashWithGroupId."""
    cacheable = isinstance(block_hash_with_group_id, (bytes, int))
    if cacheable:
        cached = _OBJECT_KEY_CACHE.get(block_hash_with_group_id)
        if cached is not None:
            return cached
    if isinstance(block_hash_with_group_id, (bytes, bytearray, memoryview)):
        raw = bytes(block_hash_with_group_id)
    else:
        raw = str(block_hash_with_group_id).encode("utf-8")
    key = key_from_bytes(b"vllm-hma-v1::" + raw)
    if cacheable:
        if len(_OBJECT_KEY_CACHE) >= _OBJECT_KEY_CACHE_MAX:
            # Evict oldest ~20% in insertion order instead of wiping hot prefixes.
            it = iter(_OBJECT_KEY_CACHE)
            stale = [next(it) for _ in range(min(_OBJECT_KEY_CACHE_EVICT, len(_OBJECT_KEY_CACHE)))]
            for k in stale:
                _OBJECT_KEY_CACHE.pop(k, None)
        _OBJECT_KEY_CACHE[block_hash_with_group_id] = key
    return key


def _align_up(n: int, align: int = 4096) -> int:
    return ((n + align - 1) // align) * align


def _is_fatal_worker_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "worker reply channel closed" in msg
        or "worker thread exited" in msg
        or "worker thread disconnected" in msg
        or "vericacheclient is closed" in msg
    )


# ---------------------------------------------------------------------------
# Local Unix-domain exact lookup RPC (Scheduler <-> Worker in vLLM v1 MP mode)
# ---------------------------------------------------------------------------

_LOOKUP_HDR = struct.Struct("<I")
_LOOKUP_KEY_STRUCT = struct.Struct("<4Q")
_MAX_LOOKUP_BATCH = 1_000_000
_STATS_SNAPSHOT = 0xFFFFFFFF
_STATS_DRAIN = 0xFFFFFFFE


def _recv_exact_unix(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(mv[got:], n - got)
        if r == 0:
            raise EOFError("local lookup socket closed")
        got += r
    return bytes(buf)


def _local_lookup_sock_path(vllm_config: Any, extra_config: dict[str, Any]) -> str:
    """Return a deterministic per-engine Unix socket path shared by Scheduler & Worker."""
    explicit = extra_config.get("local_lookup_socket")
    if explicit:
        return str(explicit)
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    engine_id = getattr(kv_cfg, "engine_id", None) if kv_cfg is not None else None
    par_cfg = getattr(vllm_config, "parallel_config", None)
    dp_rank = int(getattr(par_cfg, "data_parallel_index", 0) or 0) if par_cfg is not None else 0
    if engine_id:
        tag = f"{engine_id}_dp{dp_rank}"
    else:
        node_id = int(extra_config.get("node_id", 1))
        tag = f"node{node_id}_dp{dp_rank}_{os.getpid()}"
    safe_tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(tag))
    if len(safe_tag) > 64:
        safe_tag = hashlib.sha256(safe_tag.encode("utf-8")).hexdigest()[:32]
    return f"/tmp/vericache_lookup_{safe_tag}.sock"


class _LocalLookupServer:
    """Zero-dependency Unix-domain RPC server exposing Worker's `contains_batch_local`.

    Wire protocol (little-endian over persistent AF_UNIX SOCK_STREAM):
      Request:  `<I` (count: u32) + `count * 32` bytes (`<4Q` ObjectKeys)
      Response: `<I` (count: u32) + `count` bytes (`HitLocation` u8; 0=MISS, >0=HIT)
    """

    def __init__(
        self,
        sock_path: str,
        contains_fn: Callable[[Sequence[ObjectKey]], Sequence[HitLocation]],
        contains_flags_fn: Optional[Callable[[Sequence[Any]], bytes]] = None,
        stats_fn: Optional[Callable[[bool], dict[str, Any]]] = None,
    ) -> None:
        self._sock_path = sock_path
        self._contains_fn = contains_fn
        self._contains_flags_fn = contains_flags_fn
        self._stats_fn = stats_fn
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._conns_lock = threading.Lock()
        self._conns: set[socket.socket] = set()
        self._handler_threads: list[threading.Thread] = []

    @property
    def sock_path(self) -> str:
        return self._sock_path

    def start(self) -> None:
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self._sock_path)
        srv.listen(8)
        srv.settimeout(0.2)
        self._sock = srv
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="vericache-local-lookup-srv",
            daemon=True,
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._conns_lock:
                if self._stop.is_set():
                    try:
                        conn.close()
                    except OSError:
                        pass
                    break
                self._conns.add(conn)
                t = threading.Thread(
                    target=self._handle_conn,
                    args=(conn,),
                    name="vericache-local-lookup-conn",
                    daemon=True,
                )
                self._handler_threads = [
                    worker for worker in self._handler_threads if worker.is_alive()
                ]
                self._handler_threads.append(t)
                t.start()

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                hdr = _recv_exact_unix(conn, _LOOKUP_HDR.size)
                (count,) = _LOOKUP_HDR.unpack(hdr)
                if count in (_STATS_SNAPSHOT, _STATS_DRAIN) and self._stats_fn is not None:
                    reply = json.dumps(self._stats_fn(count == _STATS_DRAIN)).encode()
                    conn.sendall(_LOOKUP_HDR.pack(len(reply)) + reply)
                    continue
                if count > _MAX_LOOKUP_BATCH:
                    break
                if count == 0:
                    conn.sendall(_LOOKUP_HDR.pack(0))
                    continue
                payload = _recv_exact_unix(conn, count * 32)
                if self._contains_flags_fn is not None:
                    raw_keys = [payload[i : i + 32] for i in range(0, len(payload), 32)]
                    reply = self._contains_flags_fn(raw_keys)
                else:
                    keys = list(_LOOKUP_KEY_STRUCT.iter_unpack(payload))
                    locs = self._contains_fn(keys)
                    reply = bytes(int(loc) for loc in locs)
                conn.sendall(_LOOKUP_HDR.pack(len(reply)) + reply)
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
            pass
        except Exception:
            if not self._stop.is_set():
                logger.exception("VeriCacheConnector local lookup handler error")
        finally:
            with self._conns_lock:
                self._conns.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        with self._conns_lock:
            conns = list(self._conns)
            threads = list(self._handler_threads)
            self._handler_threads.clear()
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                c.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        for t in threads:
            t.join(timeout=2.0)
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


class _LocalLookupClient:
    """Thread-safe persistent AF_UNIX client for `_LocalLookupServer`."""

    def __init__(self, sock_path: str, timeout_s: float = 2.0) -> None:
        self._sock_path = sock_path
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._warned_once = False

    def _ensure_conn(self) -> socket.socket:
        if self._sock is None:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self._timeout_s)
            s.connect(self._sock_path)
            self._sock = s
        return self._sock

    def _reset_conn(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def contains_batch_local(self, keys: Sequence[ObjectKey]) -> list[bool] | None:
        """Query Worker's real Rust LRU for `keys`. Returns `None` if RPC unavailable."""
        if not keys:
            return []
        count = len(keys)
        buf = bytearray(4 + count * 32)
        _LOOKUP_HDR.pack_into(buf, 0, count)
        for i, k in enumerate(keys):
            _LOOKUP_KEY_STRUCT.pack_into(buf, 4 + i * 32, k[0], k[1], k[2], k[3])
        with self._lock:
            for attempt in range(2):
                try:
                    s = self._ensure_conn()
                    s.sendall(buf)
                    (n,) = _LOOKUP_HDR.unpack(_recv_exact_unix(s, 4))
                    raw = _recv_exact_unix(s, n) if n > 0 else b""
                    if len(raw) != count:
                        raise OSError(
                            f"local lookup reply length mismatch: got {len(raw)}, expected {count}"
                        )
                    self._warned_once = False
                    return [b != 0 for b in raw]
                except (OSError, EOFError) as exc:
                    self._reset_conn()
                    if attempt == 1:
                        if not self._warned_once:
                            logger.warning(
                                "VeriCacheConnector local lookup RPC (%s) unavailable: %s",
                                self._sock_path,
                                exc,
                            )
                            self._warned_once = True
                        return None
        return None

    def close(self) -> None:
        with self._lock:
            self._reset_conn()


# ---------------------------------------------------------------------------
# Standalone Connector Metadata (Scheduler -> Worker)
# ---------------------------------------------------------------------------


@dataclass
class BlockTransferSpec:
    request_id: str
    block_ids: list[int]
    prefix_digests: list[bytes]


@dataclass
class VeriCacheConnectorMetadata(KVConnectorMetadata):
    loads: list[BlockTransferSpec] = field(default_factory=list)
    saves: list[BlockTransferSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Full vLLM v1 HMA Connector Metadata (Scheduler <-> Worker)
# ---------------------------------------------------------------------------


@dataclass
class VeriCacheLoadReqSpec:
    """Per-request LOAD specification for independent HoL-free completion."""

    req_id: str
    load_event: int
    gpu_blocks: list[int]
    cpu_blocks: list[int]
    keys: list[ObjectKey]


@dataclass
class VeriCacheV1Metadata(KVConnectorMetadata):
    """Per-step Scheduler -> Worker metadata for content-addressed VeriCache HMA transfers."""

    load_event: int = -1
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    load_keys: list[ObjectKey] = field(default_factory=list)
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)
    load_reqs: list[VeriCacheLoadReqSpec] = field(default_factory=list)

    store_event: int = -1
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_keys: list[ObjectKey] = field(default_factory=list)

    need_flush: bool = False


@dataclass
class VeriCacheWorkerMetadata(SimpleCPUOffloadWorkerMetadata):
    """Worker -> Scheduler metadata carrying exact store commits and load misses."""

    store_committed_counts: dict[int, int] = field(default_factory=dict)
    missed_local_cpu_blocks: set[int] = field(default_factory=set)
    missed_remote_keys: list[ObjectKey] = field(default_factory=list)

    def aggregate(self, other: Any) -> "VeriCacheWorkerMetadata":
        merged_events = dict(self.completed_store_events)
        for k, v in getattr(other, "completed_store_events", {}).items():
            merged_events[k] = merged_events.get(k, 0) + v
        merged_commits = dict(self.store_committed_counts)
        for k, v in getattr(other, "store_committed_counts", {}).items():
            # Conservative min across workers if multi-worker ever reports different counts
            merged_commits[k] = min(merged_commits[k], v) if k in merged_commits else v
        merged_missed = set(self.missed_local_cpu_blocks)
        merged_missed.update(getattr(other, "missed_local_cpu_blocks", ()))
        merged_missed_remote = list(self.missed_remote_keys)
        other_missed_remote = getattr(other, "missed_remote_keys", ())
        if other_missed_remote:
            merged_missed_remote.extend(other_missed_remote)
        return VeriCacheWorkerMetadata(
            completed_store_events=merged_events,
            store_committed_counts=merged_commits,
            missed_local_cpu_blocks=merged_missed,
            missed_remote_keys=merged_missed_remote,
        )


# ---------------------------------------------------------------------------
# Worker handler for real vLLM v1 HMA mode
# ---------------------------------------------------------------------------


class _VeriCacheHmaWorker:
    """Worker-side HMA KV cache handler backed by a live `VeriCacheClient`.

    Executes direct paged hardware Copy-Engine DMA loads and stores concurrently
    on dedicated `_load_thread` and `_store_thread` workers using separate CUDA
    streams (`_load_stream` at high priority, `_d2h_stream` at low priority) with
    zero GPU SM scatter/gather kernels and zero GPU staging VRAM.
    """

    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any,
        extra_config: dict[str, Any],
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.extra_config = dict(extra_config)

        self.client: VeriCacheClient | None = None
        self.device: Any = None
        self._regions: list[tuple[str, Any, int, int]] = []
        # Precomputed static paged-KV descriptors passed directly to Rust DMA FFI.
        self._region_base_ptrs: list[int] = []
        self._region_block_strides: list[int] = []
        self._region_offsets: list[int] = []
        self._region_sizes: list[int] = []
        self._total_bytes_per_block: int = 0
        self._frame_size: int = 0

        self._load_stream: Any = None
        self._d2h_stream: Any = None
        self._step_pre_forward_event: Any = None

        self._dir_client: ClusterDirectoryClient | None = None
        dir_addr = self.extra_config.get("directory_addr")
        if dir_addr:
            node_id = int(self.extra_config.get("node_id", 1))
            self._dir_client = ClusterDirectoryClient(str(dir_addr), node_id)
        self._local_lookup_server: _LocalLookupServer | None = None

        self._connector_metadata: VeriCacheV1Metadata | None = None
        self._async_io: bool = bool(self.extra_config.get("async_io", True))
        self._store_chunk_size: int = max(1, int(self.extra_config.get("store_chunk_size", 128)))
        env_store_cap = os.environ.get("VERICACHE_MAX_PENDING_STORE_BLOCKS")
        default_store_cap = int(env_store_cap) if env_store_cap is not None else 2048
        self._max_pending_store_blocks: int = int(
            self.extra_config.get("max_pending_store_blocks", default_store_cap)
        )
        self._pending_store_blocks: int = 0

        # Separate async LOAD and SAVE queues & worker threads; `_io_lock` protects
        # only fast metadata/queue/stats updates and is NEVER held across Rust FFI.
        self._io_lock = threading.Lock()
        self._load_cond = threading.Condition(self._io_lock)
        self._store_cond = threading.Condition(self._io_lock)
        self._flush_cond = threading.Condition(self._io_lock)
        self._load_queue: deque[
            tuple[int, list[int], list[int], list[ObjectKey], list[str], Any]
        ] = deque()
        self._store_queue: deque[
            tuple[int, list[int], list[ObjectKey], Any]
        ] = deque()
        self._in_flight_loads: int = 0
        self._in_flight_stores: int = 0
        self._io_stop: bool = False
        self._worker_dead: bool = False
        self._load_thread: threading.Thread | None = None
        self._store_thread: threading.Thread | None = None
        self._evict_stop = threading.Event()
        self._evict_sync_thread: threading.Thread | None = None

        self._completed_load_reqs: set[str] = set()
        # Keyed per req_id until `get_finished()` pops that req_id, keeping both
        # `failed_load_block_ids` and `missed_remote_keys` 100% step-aligned with
        # `finished_recving`!
        self._failed_blocks_by_req: dict[str, set[int]] = {}
        self._failed_blocks_to_report: set[int] = set()
        self._missed_local_cpu_blocks: set[int] = set()
        self._missed_remote_by_req: dict[str, list[ObjectKey]] = {}
        self._missed_remote_to_report: list[ObjectKey] = []
        self._completed_store_events: dict[int, int] = {}
        self._store_committed_counts: dict[int, int] = {}

        # Cumulative diagnostics for observability in vLLM logs.
        self._total_blocks_stored: int = 0
        self._total_blocks_store_offered: int = 0
        self._total_blocks_loaded: int = 0
        self._total_blocks_missed: int = 0
        self._total_blocks_store_skipped_pressure: int = 0
        self._total_evicted_keys_synced: int = 0
        self._prof_save_enqueue_ms: float = 0.0
        self._prof_store_put_ms: float = 0.0
        self._prof_load_get_ms: float = 0.0
        self._prof_max_store_q: int = 0
        self._prof_max_load_q: int = 0

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        import torch

        if not kv_caches:
            logger.warning("VeriCacheConnector: register_kv_caches received empty dict.")
            return

        self.device = next(iter(kv_caches.values())).device
        num_blocks = int(self.kv_cache_config.num_blocks)
        assert self.kv_cache_config.kv_cache_tensors

        # Mirror vLLM's SimpleCPUOffloadWorker region discovery so FullAttention,
        # SlidingWindow, Mamba/short-conv, and looped-pass layers are all captured
        # as raw [num_blocks, block_bytes] byte views in GPU HBM.
        unique_regions: list[tuple[str, Any, int, int]] = []
        seen: set[tuple[Any, int]] = set()
        offset = 0
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            name = next((n for n in kv_cache_tensor.layers if n in kv_caches), None)
            if name is None:
                continue
            num_layers = len(kv_cache_tensor.layers)
            block_bytes = int(kv_cache_tensor.block_stride)
            blocks_span = num_blocks * block_bytes
            layer_outermost = int(kv_cache_tensor.layer_stride) >= blocks_span
            if layer_outermost:
                start = int(kv_cache_tensor.offset)
                span = num_layers * int(kv_cache_tensor.layer_stride)
                layer_names = kv_cache_tensor.layers
            else:
                start = 0
                span = blocks_span
                layer_names = kv_cache_tensor.layers[:1]

            cache = kv_caches[name]
            raw = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(
                cache.untyped_storage()
            )
            assert raw.numel() >= start + span, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller than "
                f"{start + span} bytes required"
            )
            regions = raw[start : start + span].view(-1, num_blocks, block_bytes)
            groups_per_layer, remainder = divmod(len(regions), len(layer_names))
            assert remainder == 0
            for idx, region in enumerate(regions):
                key = (region.device, region.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                layer_name = layer_names[idx // groups_per_layer]
                key_name = (
                    layer_name
                    if groups_per_layer == 1
                    else f"{layer_name}.{idx % groups_per_layer}"
                )
                unique_regions.append((key_name, region, offset, block_bytes))
                offset += block_bytes

        self._regions = unique_regions
        self._region_base_ptrs = []
        self._region_block_strides = []
        self._region_offsets = []
        self._region_sizes = []
        for key_name, region, off, sz in unique_regions:
            assert region.is_contiguous(), f"Expected contiguous KV region {key_name!r}"
            self._region_base_ptrs.append(int(region.data_ptr()))
            self._region_block_strides.append(int(region.stride(0)) * int(region.element_size()))
            self._region_offsets.append(int(off))
            self._region_sizes.append(int(sz))

        self._total_bytes_per_block = offset
        # VeriCache requires 4096-byte aligned frames for O_DIRECT NVMe io_uring.
        self._frame_size = _align_up(offset, 4096)

        low_pri, high_pri = torch.cuda.Stream.priority_range()
        self._load_stream = torch.cuda.Stream(device=self.device, priority=high_pri)
        self._d2h_stream = torch.cuda.Stream(device=self.device, priority=low_pri)

        gpu_dev = self.device.index if self.device.index is not None else 0
        node_id = int(self.extra_config.get("node_id", 1))
        hbm_frames = int(self.extra_config.get("hbm_frames", 64))
        dram_frames = int(self.extra_config.get("dram_frames", 512))
        nvme_dir = self.extra_config.get("nvme_dir")
        nvme_frames = int(self.extra_config.get("nvme_frames", 0))
        if nvme_dir:
            rank_nvme_dir = os.path.join(str(nvme_dir), f"rank_{gpu_dev}")
            os.makedirs(rank_nvme_dir, exist_ok=True)
            if nvme_frames <= 0:
                nvme_frames = max(dram_frames * 2, 256)
        else:
            rank_nvme_dir = None
            nvme_frames = 0

        rdma_device = self.extra_config.get("rdma_device")
        rdma_gid_index = int(self.extra_config.get("rdma_gid_index", 0))
        rdma_windows = int(self.extra_config.get("rdma_windows_per_peer", 8))
        raw_peers = self.extra_config.get("peers") or ()
        peers = [
            PeerSpec(
                peer_node_id=int(p["peer_node_id"]),
                tcp_addr=str(p["tcp_addr"]),
                is_listener=bool(p.get("is_listener", False)),
                psn_hint=int(p.get("psn_hint", 0x1000)),
            )
            for p in raw_peers
        ]

        logger.warning(
            "VeriCacheConnector[WORKER]: initializing VeriCacheClient "
            "(regions=%d, paged_dma=True, block_bytes=%d, frame_size=%d, gpu_dev=%d, "
            "hbm_frames=%d [%.1f MiB], dram_frames=%d [%.1f MiB], "
            "nvme_dir=%s, nvme_frames=%d [%.1f MiB], peers=%d, async_io=%s, store_chunk=%d)",
            len(self._regions),
            self._total_bytes_per_block,
            self._frame_size,
            gpu_dev,
            hbm_frames,
            (hbm_frames * self._frame_size) / (1024 * 1024),
            dram_frames,
            (dram_frames * self._frame_size) / (1024 * 1024),
            rank_nvme_dir,
            nvme_frames,
            (nvme_frames * self._frame_size) / (1024 * 1024),
            len(peers),
            self._async_io,
            self._store_chunk_size,
        )

        self.client = VeriCacheClient(
            node_id=node_id,
            frame_size=self._frame_size,
            dram_frames=dram_frames,
            gpu_device_id=gpu_dev,
            hbm_frames=hbm_frames,
            nvme_dir=rank_nvme_dir,
            nvme_frames=nvme_frames,
            rdma_device=rdma_device,
            rdma_gid_index=rdma_gid_index,
            rdma_windows_per_peer=rdma_windows,
            peers=peers,
        )

        snapshot = self._stats_snapshot(False)
        logger.warning("VeriCacheConnector[WORKER] storage paths %s", json.dumps(snapshot))

        par_cfg = getattr(self.vllm_config, "parallel_config", None)
        worker_rank = int(getattr(par_cfg, "rank", 0) or 0) if par_cfg is not None else 0
        if bool(self.extra_config.get("enable_local_lookup", True)) and worker_rank == 0:
            sock_path = _local_lookup_sock_path(self.vllm_config, self.extra_config)
            client_ref = self.client
            has_fast_flags = hasattr(client_ref, "contains_batch_local_flags")

            def _safe_contains(keys: Sequence[ObjectKey]) -> Sequence[HitLocation]:
                if self._worker_dead:
                    return [HitLocation.MISS] * len(keys)
                return client_ref.contains_batch_local(keys)

            contains_flags_fn: Optional[Callable[[Sequence[Any]], bytes]] = None
            if has_fast_flags:
                def _safe_contains_flags(keys: Sequence[Any]) -> bytes:
                    if self._worker_dead:
                        return bytes(len(keys))
                    return client_ref.contains_batch_local_flags(keys)

                contains_flags_fn = _safe_contains_flags

            self._local_lookup_server = _LocalLookupServer(
                sock_path,
                _safe_contains,
                contains_flags_fn=contains_flags_fn,
                stats_fn=self._stats_snapshot,
            )
            self._local_lookup_server.start()
            logger.warning(
                "VeriCacheConnector[WORKER]: local lookup RPC server listening on %s (fast_flags=%s)",
                sock_path,
                contains_flags_fn is not None,
            )

        if self._dir_client is not None and hasattr(self.client, "take_evicted_keys"):
            self._evict_sync_thread = threading.Thread(
                target=self._evict_sync_loop,
                name=f"vericache-evict-sync-{node_id}",
                daemon=True,
            )
            self._evict_sync_thread.start()

        if self._async_io:
            self._load_thread = threading.Thread(
                target=self._load_worker_loop,
                name=f"vericache-vllm-load-{node_id}",
                daemon=True,
            )
            self._store_thread = threading.Thread(
                target=self._store_worker_loop,
                name=f"vericache-vllm-store-{node_id}",
                daemon=True,
            )
            self._load_thread.start()
            self._store_thread.start()

    def _stats_snapshot(self, drain: bool) -> dict[str, Any]:
        assert self.client is not None
        if drain:
            self._flush_all_io()
            self.client.drain()
        stats = asdict(self.client.stats())
        groups = self.kv_cache_config.kv_cache_groups
        block_sizes = [int(group.kv_cache_spec.block_size) for group in groups]
        with self._io_lock:
            stats.update(
                frame_bytes=self._frame_size,
                num_kv_groups=len(groups),
                block_size=block_sizes[0] if len(set(block_sizes)) == 1 else None,
                is_host_pinned=bool(stats["host_pinned"]),
                direct_slab_dma=bool(stats["host_pinned"]),
                load_direct_slab_dma=bool(stats["host_pinned"]),
                host_rdma=bool(self.extra_config.get("peers")),
                staging_fallback=not bool(stats["host_pinned"]),
                worker_pid=os.getpid(),
                store_committed_blocks=self._total_blocks_stored,
                store_offered_blocks=self._total_blocks_store_offered,
                store_failed_blocks=max(
                    0, self._total_blocks_store_offered - self._total_blocks_stored
                    - self._total_blocks_store_skipped_pressure - self._pending_store_blocks
                ),
                store_pressure_skipped_blocks=self._total_blocks_store_skipped_pressure,
                load_failed_blocks=self._total_blocks_missed,
                pending_store_blocks=self._pending_store_blocks,
                pending_loads=len(self._load_queue) + self._in_flight_loads,
                worker_dead=self._worker_dead,
            )
        return stats

    def _evict_sync_loop(self) -> None:
        """Periodically drain evicted local keys from Rust and unregister from directory."""
        while not self._evict_stop.wait(timeout=0.2):
            if self._worker_dead:
                break
            self._drain_evicted_keys_once()

    def _drain_evicted_keys_once(self) -> None:
        client = self.client
        dir_cli = self._dir_client
        if (
            self._worker_dead
            or client is None
            or dir_cli is None
            or not hasattr(client, "take_evicted_keys")
        ):
            return
        try:
            while not self._evict_stop.is_set() and not self._worker_dead:
                evicted = client.take_evicted_keys(4096)
                if not evicted:
                    break
                dir_cli.unregister_batch(evicted)
                with self._io_lock:
                    self._total_evicted_keys_synced += len(evicted)
                if len(evicted) < 4096:
                    break
        except Exception:
            if not self._evict_stop.is_set():
                logger.debug("VeriCacheConnector evicted-key sync failed", exc_info=True)

    def _fail_load_job_records(
        self,
        load_job: tuple[int, list[int], list[int], list[ObjectKey], list[str], Any],
    ) -> None:
        """Record terminal failure + completion for a LOAD job so vLLM always reschedules cleanly."""
        _ev, gpu_blocks, cpu_blocks, keys, req_ids, _compute_done = load_job
        n_local = len(cpu_blocks)
        failed_remote = keys[n_local:] if len(keys) > n_local else []
        with self._io_lock:
            if gpu_blocks:
                if req_ids:
                    for rid in req_ids:
                        self._failed_blocks_by_req.setdefault(rid, set()).update(gpu_blocks)
                else:
                    self._failed_blocks_to_report.update(gpu_blocks)
            if cpu_blocks:
                self._missed_local_cpu_blocks.update(cpu_blocks)
            if failed_remote:
                for rid in req_ids:
                    self._missed_remote_by_req.setdefault(rid, []).extend(failed_remote)
            self._completed_load_reqs.update(req_ids)
            self._total_blocks_missed += len(gpu_blocks)

    def _run_load_job_safe(
        self,
        load_job: tuple[int, list[int], list[int], list[ObjectKey], list[str], Any],
    ) -> None:
        """Execute a LOAD job guaranteeing terminal notification to vLLM even on error."""
        if self._worker_dead:
            self._fail_load_job_records(load_job)
            return
        try:
            self._execute_load_job(*load_job)
        except Exception as exc:
            if _is_fatal_worker_error(exc):
                if not self._worker_dead:
                    logger.error(
                        "VeriCacheConnector native worker died (%s); fast-failing future KV loads/saves",
                        exc,
                    )
                self._worker_dead = True
            else:
                logger.exception("VeriCacheConnector async LOAD job failed")
            if self._load_stream is not None:
                try:
                    self._load_stream.synchronize()
                except Exception:
                    pass
            self._fail_load_job_records(load_job)

    def _execute_load_job(
        self,
        load_event: int,
        load_gpu_blocks: list[int],
        load_cpu_blocks: list[int],
        load_keys: list[ObjectKey],
        req_ids: list[str],
        compute_done: Any = None,
    ) -> None:
        import torch

        assert self.client is not None and self.device is not None
        torch.cuda.set_device(self.device)
        n_blocks = len(load_gpu_blocks)
        n_local = len(load_cpu_blocks)
        stream = self._load_stream or torch.cuda.current_stream(self.device)

        # Wait on the blocking pre-forward CUDA event on `_load_thread` BEFORE
        # entering Rust FFI so prior steps' kernels / kv_block_zeroer / CoW copies
        # are 100% finished without ever stalling Rust's `worker_loop` inside
        # `cudaEventSynchronize`!
        if compute_done is not None:
            compute_done.synchronize()

        # Ensure non-overlapping destination GPU block IDs for CUDA 13.0 cudaMemcpyBatchAsync.
        if len(set(load_gpu_blocks)) == n_blocks:
            unique_keys = load_keys
            unique_gpu_blocks = load_gpu_blocks
            dup_map: list[int] | None = None
        else:
            seen_pos: dict[int, int] = {}
            unique_keys = []
            unique_gpu_blocks = []
            dup_map = []
            for key, gid in zip(load_keys, load_gpu_blocks):
                idx = seen_pos.get(gid)
                if idx is None:
                    idx = len(unique_gpu_blocks)
                    seen_pos[gid] = idx
                    unique_keys.append(key)
                    unique_gpu_blocks.append(gid)
                dup_map.append(idx)

        t_get0 = time.perf_counter()
        if hasattr(self.client, "get_gpu_paged_batch_flags"):
            raw_flags = self.client.get_gpu_paged_batch_flags(
                unique_keys,
                self._region_base_ptrs,
                self._region_block_strides,
                self._region_offsets,
                self._region_sizes,
                unique_gpu_blocks,
                cuda_stream=stream.cuda_stream,
            )
            dt_get_ms = (time.perf_counter() - t_get0) * 1000.0
            hit_flags = raw_flags if dup_map is None else bytes(raw_flags[i] for i in dup_map)
        else:
            raw_hits = self.client.get_gpu_paged_batch(
                unique_keys,
                self._region_base_ptrs,
                self._region_block_strides,
                self._region_offsets,
                self._region_sizes,
                unique_gpu_blocks,
                cuda_stream=stream.cuda_stream,
            )
            dt_get_ms = (time.perf_counter() - t_get0) * 1000.0
            hits = raw_hits if dup_map is None else [raw_hits[i] for i in dup_map]
            hit_flags = bytes(0 if h is None else int(h.location) for h in hits)

        n_hit = 0
        missed_local_cpu_ids: list[int] = []
        missed_remote_keys: list[ObjectKey] = []
        missed_gpu_ids: list[int] = []
        for pos, (gpu_bid, key, flag) in enumerate(zip(load_gpu_blocks, load_keys, hit_flags)):
            if flag == 0:
                missed_gpu_ids.append(gpu_bid)
                if pos < n_local:
                    missed_local_cpu_ids.append(load_cpu_blocks[pos])
                else:
                    missed_remote_keys.append(key)
            else:
                n_hit += 1

        if missed_remote_keys and self._dir_client is not None:
            # Purge stale peer directory entries without wiping this node's own copies.
            self._dir_client.unregister_batch(missed_remote_keys, peers_only=True)

        with self._io_lock:
            if missed_gpu_ids:
                if req_ids:
                    for rid in req_ids:
                        self._failed_blocks_by_req.setdefault(rid, set()).update(missed_gpu_ids)
                else:
                    self._failed_blocks_to_report.update(missed_gpu_ids)
            if missed_local_cpu_ids:
                self._missed_local_cpu_blocks.update(missed_local_cpu_ids)
            if missed_remote_keys:
                for rid in req_ids:
                    self._missed_remote_by_req.setdefault(rid, []).extend(missed_remote_keys)
            self._completed_load_reqs.update(req_ids)
            self._total_blocks_loaded += n_hit
            self._total_blocks_missed += len(missed_gpu_ids)
            self._prof_load_get_ms += dt_get_ms

        logger.debug(
            "VeriCacheConnector[LOAD] event=%d reqs=%s blocks=%d hits=%d misses=%d",
            load_event,
            req_ids,
            n_blocks,
            n_hit,
            len(missed_gpu_ids),
        )

    def _execute_store_job(
        self,
        store_event: int,
        store_gpu_blocks: list[int],
        store_keys: list[ObjectKey],
        compute_done: Any,
    ) -> None:
        import torch

        assert self.client is not None and self.device is not None
        torch.cuda.set_device(self.device)
        n_blocks = len(store_keys)
        committed = 0
        if self._worker_dead:
            with self._io_lock:
                self._completed_store_events[store_event] = 1
                self._store_committed_counts[store_event] = 0
            return
        try:
            stream = self._d2h_stream or torch.cuda.current_stream(self.device)
            # Wait on the blocking compute_done CUDA event on `_store_thread` (0% CPU
            # spin) BEFORE entering Rust FFI so an in-flight 50-300ms prefill step
            # NEVER freezes Rust's `worker_loop` inside `cudaEventSynchronize`!
            if compute_done is not None:
                compute_done.synchronize()

            chunk = self._store_chunk_size
            idx = 0
            while idx < n_blocks and not self._worker_dead:
                end = min(n_blocks, idx + chunk)
                sub_keys = store_keys[idx:end]
                sub_gids = store_gpu_blocks[idx:end]
                t_p0 = time.perf_counter()
                n_ok = int(
                    self.client.put_gpu_paged_dram_batch(
                        sub_keys,
                        self._region_base_ptrs,
                        self._region_block_strides,
                        self._region_offsets,
                        self._region_sizes,
                        sub_gids,
                        cuda_stream=stream.cuda_stream,
                    )
                )
                dt_put_ms = (time.perf_counter() - t_p0) * 1000.0
                with self._io_lock:
                    self._prof_store_put_ms += dt_put_ms
                committed += n_ok
                if n_ok < len(sub_keys):
                    logger.warning(
                        "VeriCacheConnector[SAVE] event=%d partial commit %d/%d",
                        store_event,
                        committed,
                        n_blocks,
                    )
                    break
                idx = end
        except Exception as exc:
            if _is_fatal_worker_error(exc):
                if not self._worker_dead:
                    logger.error(
                        "VeriCacheConnector native worker died during SAVE (%s); fast-completing future saves",
                        exc,
                    )
                self._worker_dead = True
            else:
                logger.exception(
                    "VeriCacheConnector async SAVE job failed (event=%d committed=%d/%d)",
                    store_event,
                    committed,
                    n_blocks,
                )
            if self._d2h_stream is not None:
                try:
                    self._d2h_stream.synchronize()
                except Exception:
                    pass
        finally:
            with self._io_lock:
                self._completed_store_events[store_event] = 1
                self._store_committed_counts[store_event] = committed
                self._total_blocks_stored += committed

        logger.debug(
            "VeriCacheConnector[SAVE] event=%d committed=%d/%d",
            store_event,
            committed,
            n_blocks,
        )

    def _load_worker_loop(self) -> None:
        while True:
            with self._io_lock:
                while not self._load_queue and not self._io_stop:
                    self._load_cond.wait(timeout=0.05)
                if self._load_queue:
                    load_job = self._load_queue.popleft()
                    self._in_flight_loads += 1
                elif self._io_stop:
                    break
                else:
                    continue
            try:
                self._run_load_job_safe(load_job)
            except Exception:
                logger.exception("VeriCacheConnector _load_worker_loop unexpected error")
                self._fail_load_job_records(load_job)
            finally:
                with self._io_lock:
                    self._in_flight_loads -= 1
                    self._flush_cond.notify_all()

    def _store_worker_loop(self) -> None:
        while True:
            with self._io_lock:
                while not self._store_queue and not self._io_stop:
                    self._store_cond.wait(timeout=0.05)
                if self._store_queue:
                    store_job = self._store_queue.popleft()
                    self._in_flight_stores += 1
                elif self._io_stop:
                    break
                else:
                    continue
            try:
                self._execute_store_job(*store_job)
            except Exception:
                logger.exception("VeriCacheConnector _store_worker_loop unexpected error")
                store_event = store_job[0]
                with self._io_lock:
                    self._completed_store_events.setdefault(store_event, 1)
                    self._store_committed_counts.setdefault(store_event, 0)
            finally:
                with self._io_lock:
                    self._in_flight_stores -= 1
                    self._pending_store_blocks = max(
                        0, self._pending_store_blocks - len(store_job[1])
                    )
                    self._flush_cond.notify_all()

    def _flush_all_io(self) -> None:
        if not self._async_io:
            return
        with self._io_lock:
            while (
                self._load_queue
                or self._store_queue
                or self._in_flight_loads > 0
                or self._in_flight_stores > 0
            ):
                self._flush_cond.wait(timeout=0.01)

    def handle_preemptions(self, metadata: VeriCacheV1Metadata) -> None:
        if metadata.need_flush:
            self._flush_all_io()

    def bind_connector_metadata(self, metadata: VeriCacheV1Metadata) -> None:
        self._connector_metadata = metadata
        self._step_pre_forward_event = None
        # Record a blocking GPU fence BEFORE the current step's forward pass (after
        # prior steps' kernels, kv_block_zeroer, and copy-on-write block copies).
        # `_load_thread` syncs this event with 0% CPU spin before entering Rust FFI.
        if (
            (metadata.load_reqs or (metadata.load_event >= 0 and metadata.load_gpu_blocks))
            and self.device is not None
        ):
            import torch

            ev = torch.cuda.Event(enable_timing=False, blocking=True)
            ev.record(torch.cuda.current_stream(self.device))
            self._step_pre_forward_event = ev

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None
        self._step_pre_forward_event = None

    def start_load_kv(self) -> None:
        import torch

        meta = self._connector_metadata
        if meta is None:
            return
        has_per_req = bool(meta.load_reqs)
        has_legacy = meta.load_event >= 0 and bool(meta.load_gpu_blocks)
        if not has_per_req and not has_legacy:
            return
        assert self.client is not None and self.device is not None

        compute_done = self._step_pre_forward_event
        if compute_done is None:
            compute_done = torch.cuda.Event(enable_timing=False, blocking=True)
            compute_done.record(torch.cuda.current_stream(self.device))
        self._step_pre_forward_event = None

        jobs: list[tuple[int, list[int], list[int], list[ObjectKey], list[str], Any]] = []
        if has_per_req:
            # Schedule local-only and shorter requests first so small DRAM hits
            # complete immediately without HoL blocking behind 32K NVMe/RDMA loads!
            ordered = sorted(
                meta.load_reqs,
                key=lambda r: (
                    0 if len(r.cpu_blocks) >= len(r.gpu_blocks) else 1,
                    len(r.gpu_blocks),
                ),
            )
            for spec in ordered:
                if not spec.gpu_blocks:
                    with self._io_lock:
                        self._completed_load_reqs.add(spec.req_id)
                    continue
                jobs.append(
                    (
                        spec.load_event,
                        list(spec.gpu_blocks),
                        list(spec.cpu_blocks),
                        list(spec.keys),
                        [spec.req_id],
                        compute_done,
                    )
                )
        else:
            req_ids = list(meta.load_event_to_reqs.get(meta.load_event, ()))
            jobs.append(
                (
                    meta.load_event,
                    list(meta.load_gpu_blocks),
                    list(meta.load_cpu_blocks),
                    list(meta.load_keys),
                    req_ids,
                    compute_done,
                )
            )

        if not jobs:
            return
        if self._worker_dead:
            for job in jobs:
                self._fail_load_job_records(job)
            return
        if self._async_io:
            with self._io_lock:
                self._load_queue.extend(jobs)
                if len(self._load_queue) > self._prof_max_load_q:
                    self._prof_max_load_q = len(self._load_queue)
                self._load_cond.notify_all()
        else:
            for job in jobs:
                self._run_load_job_safe(job)

    def wait_for_save(self) -> None:
        import torch

        meta = self._connector_metadata
        if meta is None or meta.store_event < 0:
            return
        n_new = len(meta.store_gpu_blocks)
        with self._io_lock:
            self._total_blocks_store_offered += n_new
        if not meta.store_gpu_blocks or self._worker_dead:
            # All blocks dedup-filtered on Scheduler (or native worker dead);
            # immediately mark complete so GPU block pins release on the next step.
            with self._io_lock:
                self._completed_store_events[meta.store_event] = 1
                self._store_committed_counts[meta.store_event] = 0
            return
        assert self.client is not None and self.device is not None

        if self._async_io and self._max_pending_store_blocks > 0:
            with self._io_lock:
                if (
                    self._pending_store_blocks > 0
                    and self._pending_store_blocks + n_new > self._max_pending_store_blocks
                ):
                    # Backpressure watermark: drop excess store immediately with
                    # committed=0 so SimpleCPUOffloadScheduler releases GPU block
                    # pins immediately instead of starving vLLM's GPU KV pool!
                    self._completed_store_events[meta.store_event] = 1
                    self._store_committed_counts[meta.store_event] = 0
                    self._total_blocks_store_skipped_pressure += n_new
                    return

        # `store_gpu_blocks` remain ref-counted/pinned in SimpleCPUOffloadScheduler's
        # `gpu_block_pool` until `completed_store_events` is reported back via
        # `build_connector_worker_meta()`. Record a blocking CUDA event here and
        # let `_store_thread` sync it before entering Rust FFI.
        t0 = time.perf_counter()
        compute_stream = torch.cuda.current_stream(self.device)
        compute_done = torch.cuda.Event(enable_timing=False, blocking=True)
        compute_done.record(compute_stream)
        dt_enq_ms = (time.perf_counter() - t0) * 1000.0

        job = (
            meta.store_event,
            list(meta.store_gpu_blocks),
            list(meta.store_keys),
            compute_done,
        )
        if self._async_io:
            with self._io_lock:
                self._prof_save_enqueue_ms += dt_enq_ms
                if (
                    self._max_pending_store_blocks > 0
                    and self._pending_store_blocks > 0
                    and self._pending_store_blocks + n_new > self._max_pending_store_blocks
                ):
                    self._completed_store_events[meta.store_event] = 1
                    self._store_committed_counts[meta.store_event] = 0
                    self._total_blocks_store_skipped_pressure += n_new
                    return
                self._pending_store_blocks += n_new
                self._store_queue.append(job)
                if len(self._store_queue) > self._prof_max_store_q:
                    self._prof_max_store_q = len(self._store_queue)
                self._store_cond.notify()
        else:
            with self._io_lock:
                self._prof_save_enqueue_ms += dt_enq_ms
            self._execute_store_job(*job)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        with self._io_lock:
            done_recv = self._completed_load_reqs
            self._completed_load_reqs = set()
            for rid in done_recv:
                fb = self._failed_blocks_by_req.pop(rid, None)
                if fb:
                    self._failed_blocks_to_report.update(fb)
                mk = self._missed_remote_by_req.pop(rid, None)
                if mk:
                    self._missed_remote_to_report.extend(mk)
        return None, (done_recv or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._io_lock:
            errs = self._failed_blocks_to_report
            self._failed_blocks_to_report = set()
        return errs

    def build_connector_worker_meta(self) -> Any:
        with self._io_lock:
            if (
                not self._completed_store_events
                and not self._missed_local_cpu_blocks
                and not self._missed_remote_to_report
            ):
                return None
            completed = self._completed_store_events
            self._completed_store_events = {}
            committed = self._store_committed_counts
            self._store_committed_counts = {}
            missed_cpu = self._missed_local_cpu_blocks
            self._missed_local_cpu_blocks = set()
            missed_remote = self._missed_remote_to_report
            self._missed_remote_to_report = []
        return VeriCacheWorkerMetadata(
            completed_store_events=completed,
            store_committed_counts=committed,
            missed_local_cpu_blocks=missed_cpu,
            missed_remote_keys=missed_remote,
        )

    def shutdown(self) -> None:
        if self._async_io and (
            self._load_thread is not None or self._store_thread is not None
        ):
            self._flush_all_io()
            with self._io_lock:
                self._io_stop = True
                self._load_cond.notify_all()
                self._store_cond.notify_all()
            if self._load_thread is not None:
                self._load_thread.join(timeout=5.0)
                self._load_thread = None
            if self._store_thread is not None:
                self._store_thread.join(timeout=5.0)
                self._store_thread = None

        if self._evict_sync_thread is not None:
            self._evict_stop.set()
            self._evict_sync_thread.join(timeout=2.0)
            self._evict_sync_thread = None

        if self._local_lookup_server is not None:
            self._local_lookup_server.close()
            self._local_lookup_server = None

        if self._dir_client is not None:
            self._dir_client.close()
            self._dir_client = None

        if self.client is not None:
            try:
                stats = self.client.stats()
                logger.warning(
                    "VeriCacheConnector[SHUTDOWN] stats: puts=%d gpu=%d dram=%d "
                    "nvme=%d remote=%d miss=%d (worker loaded=%d stored=%d missed=%d "
                    "pressure_skip=%d evict_sync=%d | prof_ms: save_enq=%.1f "
                    "store_put=%.1f load_get=%.1f max_sq=%d max_lq=%d)",
                    stats.puts,
                    stats.gpu_hits,
                    stats.dram_hits,
                    stats.nvme_hits,
                    stats.remote_hits,
                    stats.misses,
                    self._total_blocks_loaded,
                    self._total_blocks_stored,
                    self._total_blocks_missed,
                    self._total_blocks_store_skipped_pressure,
                    self._total_evicted_keys_synced,
                    self._prof_save_enqueue_ms,
                    self._prof_store_put_ms,
                    self._prof_load_get_ms,
                    self._prof_max_store_q,
                    self._prof_max_load_q,
                )
            except Exception:
                pass
            self.client.close()
            self.client = None


# ---------------------------------------------------------------------------
# VeriCacheConnector (vLLM v1 KVConnectorBase_V1 + SupportsHMA)
# ---------------------------------------------------------------------------


class VeriCacheConnector(KVConnectorBase_V1, SupportsHMA):
    """Location-transparent vLLM v1 KVConnector backed by `VeriCacheClient`.

    Data path (Route 1 — zero TCB expansion):
    - In real vLLM v1 serving mode (`VeriCacheConnector(vllm_config, role, kv_cache_config)`):
      supports HMA hybrid models (FullAttention + SlidingWindow + Mamba/sconv +
      looped layers), multiprocess Scheduler/Worker separation, and automatic
      4-tier VeriCache caching (Local HBM -> Pinned DRAM -> NVMe io_uring ->
      Remote GPUDirect RDMA).
    - In standalone mode (`VeriCacheConnector(vllm_config, role, client=...)`):
      exposes direct per-layer `start_load_kv` / `save_kv_layer` hooks over the
      caller-supplied `VeriCacheClient`.
    """

    @classmethod
    def supports_lookahead_block_hashes(cls, kv_transfer_config: Any) -> bool:
        return True

    @property
    def requires_kv_delivery(self) -> bool:
        # Runs as kv_both best-effort cache: a dropped save is just a future miss.
        return False

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
        *,
        client: Optional[VeriCacheClient] = None,
        block_size: int = 16,
        probe_layer_name: str = "layer_0",
        kv_layout: str = "2_N_...",
        remote_lookup_fn: Optional[Callable[[Sequence[ObjectKey]], Sequence[bool]]] = None,
        on_save_keys_fn: Optional[Callable[[Sequence[ObjectKey]], None]] = None,
    ) -> None:
        if _HAS_VLLM and vllm_config is not None and getattr(vllm_config, "kv_transfer_config", None) is not None:
            super().__init__(vllm_config, role, kv_cache_config)
        else:
            # Standalone test mode (no real VllmConfig / KVTransferConfig).
            self._vllm_config = vllm_config
            self._role = role
            self._kv_cache_config = kv_cache_config
            self._connector_metadata = None

        self._standalone_mode = client is not None
        self.client = client
        self.block_size = int(block_size)
        self.probe_layer_name = probe_layer_name
        self.kv_layout = kv_layout
        self.remote_lookup_fn = remote_lookup_fn
        self.on_save_keys_fn = on_save_keys_fn

        # Standalone Scheduler/Worker state
        self._matched_digests: dict[str, list[bytes]] = {}
        self._pending_loads: list[BlockTransferSpec] = []
        self._pending_saves: list[BlockTransferSpec] = []
        self._kv_caches: dict[str, Any] = {}
        self._last_load_hits: dict[str, list[HitInfo | None]] = {}
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

        # Real vLLM v1 HMA Scheduler / Worker handlers
        self._hma_scheduler: Any = None
        self._hma_worker: _VeriCacheHmaWorker | None = None
        self._sched_dir_client: ClusterDirectoryClient | None = None
        self._local_lookup_client: _LocalLookupClient | None = None
        # Cached last-seen ObjectKey per virtual cpu_block_id so concurrent
        # drift eviction never breaks in-flight build_connector_meta resolution.
        self._cpu_bid_to_key: dict[int, ObjectKey] = {}
        # Pending remote hits between get_num_new_matched_tokens and update_state_after_alloc:
        # req_id -> (local_matched_tokens, remote_keys, num_computed_tokens)
        self._pending_remote_hits: dict[str, tuple[int, list[ObjectKey], int]] = {}
        # Active remote loads waiting for build_connector_meta / completion:
        # req_id -> {"gpu_blocks": list[int], "keys": list[ObjectKey], "event": int | None, "finished": bool}
        self._remote_reqs_to_load: dict[str, dict[str, Any]] = {}
        self._remote_load_event_counter: int = 1_000_000_000
        # Store event -> ObjectKeys to publish to ClusterDirectoryClient on completion
        self._store_event_to_keys: dict[int, list[ObjectKey]] = {}
        # Store event -> (kept_cpu_ids, skipped_cpu_ids) when remote-hit store dedup filters blocks
        self._store_event_cpu_plan: dict[int, tuple[list[int], list[int]]] = {}
        # Bounded LRU of keys confirmed resident on remote peers (to skip 2x redundant re-store)
        self._dedup_remote_stores: bool = True
        self._remote_known_keys: OrderedDict[ObjectKey, None] = OrderedDict()
        self._remote_known_keys_cap: int = 262_144
        self._sched_remote_hit_blocks: int = 0
        self._sched_remote_store_deduped: int = 0
        self._sched_local_lookup_rpcs: int = 0
        self._sched_local_lookup_ms: float = 0.0
        self._sched_local_drift_evicted: int = 0

        if not self._standalone_mode:
            assert _HAS_VLLM, "Real vLLM mode requires vllm to be installed"
            extra_config = (
                getattr(self._kv_transfer_config, "kv_connector_extra_config", None) or {}
            )
            has_nvme = bool(extra_config.get("nvme_dir"))
            self._cache_hit_source = CacheHitSource.DISK if has_nvme else CacheHitSource.HOST

            if role == KVConnectorRole.SCHEDULER:
                self._init_hma_scheduler(vllm_config, kv_cache_config, extra_config)
            elif role == KVConnectorRole.WORKER:
                self._hma_worker = _VeriCacheHmaWorker(
                    vllm_config, kv_cache_config, extra_config
                )

    def _init_hma_scheduler(
        self,
        vllm_config: Any,
        kv_cache_config: Any,
        extra_config: dict[str, Any],
    ) -> None:
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes  # type: ignore
        from vllm.v1.simple_kv_offload.manager import (  # type: ignore
            SimpleCPUOffloadScheduler,
        )

        if not vllm_config.cache_config.enable_prefix_caching:
            logger.warning(
                "VeriCacheConnector[SCHEDULER]: prefix caching is disabled; "
                "external KV caching requires --enable-prefix-caching."
            )
            return

        assert kv_cache_config is not None
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hbm_frames = int(extra_config.get("hbm_frames", 64))
        dram_frames = int(extra_config.get("dram_frames", 512))
        nvme_dir = extra_config.get("nvme_dir")
        nvme_frames = int(extra_config.get("nvme_frames", 0))
        if nvme_dir and nvme_frames <= 0:
            nvme_frames = max(dram_frames * 2, 256)

        # The shadow pool indexes DRAM and NVMe together when the disk tier is enabled.
        default_cap = dram_frames + nvme_frames if nvme_frames else max(hbm_frames + dram_frames, 16)
        max_blocks = int(extra_config.get("max_cached_blocks", default_cap))
        max_blocks = max(max_blocks, 4)

        num_gpu_blocks = max(1, int(kv_cache_config.num_blocks))
        if kv_cache_config.kv_cache_tensors:
            # vLLM describes each placement against the same backing allocation.
            gpu_total = int(kv_cache_config.kv_cache_tensors[0].size)
        else:
            gpu_total = num_gpu_blocks * 65536
        gpu_bytes_per_block = max(1, gpu_total // num_gpu_blocks)
        # Invert vLLM's floor(gpu_blocks * offload_bytes / gpu_total_bytes)
        # so padding in the allocation cannot make the shadow pool one block short.
        nominal_capacity_bytes = (max_blocks * gpu_total + num_gpu_blocks - 1) // num_gpu_blocks

        dir_addr = extra_config.get("directory_addr")
        if dir_addr:
            node_id = int(extra_config.get("node_id", 1))
            self._sched_dir_client = ClusterDirectoryClient(str(dir_addr), node_id)
        self._dedup_remote_stores = bool(extra_config.get("dedup_remote_stores", True))

        lookup_sock_path: str | None = None
        if bool(extra_config.get("enable_local_lookup", True)):
            lookup_sock_path = _local_lookup_sock_path(vllm_config, extra_config)
            self._local_lookup_client = _LocalLookupClient(lookup_sock_path)

        logger.warning(
            "VeriCacheConnector[SCHEDULER]: initializing HMA scheduler "
            "(scheduler_block_size=%d, hash_block_size=%d, max_cached_blocks=%d, "
            "gpu_bytes_per_block=%d, directory_addr=%s, local_lookup_sock=%s)",
            scheduler_block_size,
            hash_block_size,
            max_blocks,
            gpu_bytes_per_block,
            dir_addr,
            lookup_sock_path,
        )

        sched_kwargs: dict[str, Any] = {
            "scheduler_block_size": scheduler_block_size,
            "hash_block_size": hash_block_size,
            "lazy_offload": bool(extra_config.get("lazy_offload", False)),
            "disk_capacity_bytes": nominal_capacity_bytes if nvme_dir else 0,
            "use_page_cache": False,
        }
        sig = inspect.signature(SimpleCPUOffloadScheduler.__init__)
        if "num_cpu_blocks" in sig.parameters:
            sched_kwargs["num_cpu_blocks"] = max_blocks

        self._hma_scheduler = SimpleCPUOffloadScheduler(
            vllm_config,
            kv_cache_config,
            nominal_capacity_bytes,
            **sched_kwargs,
        )

    # =======================================================================
    # Scheduler-side hooks
    # =======================================================================

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        if self._hma_scheduler is not None:
            self._hma_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def _cleanup_remote_load_req(self, req_id: str) -> list[ObjectKey]:
        st = self._remote_reqs_to_load.pop(req_id, None)
        if st is None:
            return []
        gpu_pool = getattr(self._hma_scheduler, "_gpu_block_pool", None)
        if gpu_pool is not None and st["gpu_blocks"]:
            gpu_pool.free_blocks(gpu_pool.blocks[bid] for bid in st["gpu_blocks"])
        return list(st.get("keys") or ())

    def _record_remote_known_keys(self, keys: Iterable[ObjectKey]) -> None:
        if not self._dedup_remote_stores:
            return
        known = self._remote_known_keys
        cap = self._remote_known_keys_cap
        for k in keys:
            if k in known:
                known.move_to_end(k)
            else:
                if len(known) >= cap:
                    known.popitem(last=False)
                known[k] = None

    def _snapshot_load_cpu_keys(self, req_id: str) -> None:
        """Cache ObjectKeys for newly scheduled load cpu_block_ids against concurrent eviction."""
        if self._hma_scheduler is None:
            return
        req_state = self._hma_scheduler._reqs_to_load.get(req_id)
        if req_state is None or req_state.transfer_meta is None:
            return
        cpu_pool = self._hma_scheduler.cpu_block_pool
        for cpu_bid in req_state.transfer_meta.cpu_block_ids:
            blk_hash = cpu_pool.blocks[cpu_bid].block_hash
            if blk_hash is not None:
                self._cpu_bid_to_key[cpu_bid] = object_key_from_vllm_block_hash(blk_hash)

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Return `(num_external_matched_tokens, load_async)` for `request`."""
        if not self._standalone_mode:
            if self._hma_scheduler is None:
                return 0, False
            self._pending_remote_hits.pop(request.request_id, None)
            local_matched, local_async = self._hma_scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
            local_matched = int(local_matched or 0)

            # Verify SimpleCPUOffloadScheduler's shadow virtual LRU hits against
            # Rust's real multi-tier LRU over the local Unix-domain RPC (<0.3ms).
            # Evicting drifted blocks before re-querying SimpleCPUOffloadScheduler
            # guarantees zero shadow-LRU drift and zero assertion/pin mismatch.
            if local_matched > 0 and self._local_lookup_client is not None:
                cpu_pool = self._hma_scheduler.cpu_block_pool
                verified_live_ids: set[int] = set()
                for _ in range(4):
                    pending = self._hma_scheduler._pending_cpu_hits.get(request.request_id)
                    if pending is None:
                        break
                    cpu_hit_blocks = pending[0]
                    to_check: list[tuple[Any, ObjectKey]] = []
                    for grp in cpu_hit_blocks:
                        for blk in grp:
                            if blk.is_null or blk.block_hash is None:
                                continue
                            if blk.block_id in verified_live_ids:
                                continue
                            to_check.append(
                                (blk, object_key_from_vllm_block_hash(blk.block_hash))
                            )
                    if not to_check:
                        break
                    t_rpc0 = time.perf_counter()
                    present = self._local_lookup_client.contains_batch_local(
                        [key for _blk, key in to_check]
                    )
                    self._sched_local_lookup_ms += (time.perf_counter() - t_rpc0) * 1000.0
                    self._sched_local_lookup_rpcs += 1
                    if present is None:
                        # Lookup RPC unavailable (e.g. startup/transient); fail open.
                        for blk, key in to_check:
                            self._cpu_bid_to_key[blk.block_id] = key
                        break
                    stale_ids: set[int] = set()
                    stale_keys: list[ObjectKey] = []
                    for (blk, key), ok in zip(to_check, present):
                        if ok:
                            verified_live_ids.add(blk.block_id)
                            self._cpu_bid_to_key[blk.block_id] = key
                        else:
                            stale_ids.add(blk.block_id)
                            stale_keys.append(key)
                    if not stale_ids:
                        break
                    cpu_pool.evict_blocks(stale_ids)
                    self._sched_local_drift_evicted += len(stale_ids)
                    if stale_keys and self._sched_dir_client is not None:
                        # Remove this node from peer directory for locally-evicted keys.
                        self._sched_dir_client.unregister_batch(stale_keys)
                    local_matched, local_async = (
                        self._hma_scheduler.get_num_new_matched_tokens(
                            request, num_computed_tokens
                        )
                    )
                    local_matched = int(local_matched or 0)
                    if local_matched <= 0:
                        break

            remote_matched = 0
            if (
                self._sched_dir_client is not None
                and not getattr(request, "skip_reading_external_kv", False)
                and not getattr(request, "skip_reading_prefix_cache", False)
                and len(self._hma_scheduler.cpu_kv_cache_config.kv_cache_groups) == 1
                and num_computed_tokens % self._hma_scheduler.block_size == 0
            ):
                from vllm.v1.core.kv_cache_utils import (  # type: ignore
                    make_block_hash_with_group_id,
                )

                blk_sz = self._hma_scheduler.fa_block_size
                hash_blk_sz = self._hma_scheduler.hash_block_size
                fa_gidx = self._hma_scheduler.fa_gidx
                if blk_sz == hash_blk_sz and local_matched % blk_sz == 0:
                    max_hit_len = int(request.num_tokens) - 1 - num_computed_tokens
                    start_idx = (num_computed_tokens + local_matched) // blk_sz
                    max_rem_blocks = max(0, (max_hit_len - local_matched) // blk_sz)
                    rem_hashes = request.block_hashes[start_idx : start_idx + max_rem_blocks]
                    if rem_hashes:
                        cand_keys = [
                            object_key_from_vllm_block_hash(
                                make_block_hash_with_group_id(h, fa_gidx)
                            )
                            for h in rem_hashes
                        ]
                        if hasattr(self._sched_dir_client, "exists_remote_prefix"):
                            contig = self._sched_dir_client.exists_remote_prefix(cand_keys)
                        else:
                            flags = self._sched_dir_client.exists_remote_batch(cand_keys)
                            contig = 0
                            for f in flags:
                                if f:
                                    contig += 1
                                else:
                                    break
                        if contig > 0:
                            remote_matched = contig * blk_sz
                            self._pending_remote_hits[request.request_id] = (
                                local_matched,
                                cand_keys[:contig],
                                num_computed_tokens,
                            )
                            self._sched_remote_hit_blocks += contig

            total_matched = local_matched + remote_matched
            is_async = bool(local_async or remote_matched > 0)
            if total_matched > 0:
                logger.debug(
                    "VeriCacheConnector[SCHEDULER]: EXTERNAL HIT req=%s "
                    "computed=%d local=%d remote=%d (async=%s)",
                    request.request_id,
                    num_computed_tokens,
                    local_matched,
                    remote_matched,
                    is_async,
                )
            return total_matched, is_async

        assert self.client is not None
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None:
            token_ids = request.prompt_token_ids
        all_digests = hash_prefix_blocks(token_ids, self.block_size)
        start_blk = num_computed_tokens // self.block_size
        candidate_digests = all_digests[start_blk:]
        if not candidate_digests:
            return 0, False

        probe_keys = [layer_block_key(d, self.probe_layer_name) for d in candidate_digests]
        local_locs = self.client.contains_batch_local(probe_keys)
        remote_hits: Sequence[bool] | None = None
        if self.remote_lookup_fn is not None:
            remote_hits = self.remote_lookup_fn(probe_keys)

        matched = 0
        for i, loc in enumerate(local_locs):
            hit_local = loc != HitLocation.MISS
            hit_remote = bool(remote_hits[i]) if remote_hits is not None else False
            if hit_local or hit_remote:
                matched += 1
            else:
                break

        if matched == 0:
            return 0, False

        self._matched_digests[request.request_id] = candidate_digests[:matched]
        return matched * self.block_size, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Record which paged `block_ids` to load from and/or save to VeriCache."""
        if not self._standalone_mode:
            if self._hma_scheduler is not None:
                pending_rem = self._pending_remote_hits.pop(request.request_id, None)
                if pending_rem is None:
                    self._hma_scheduler.update_state_after_alloc(
                        request, blocks, num_external_tokens
                    )
                    self._snapshot_load_cpu_keys(request.request_id)
                    return
                local_matched, remote_keys, num_computed_tokens = pending_rem
                local_part = min(num_external_tokens, local_matched)
                self._hma_scheduler.update_state_after_alloc(request, blocks, local_part)
                self._snapshot_load_cpu_keys(request.request_id)

                remote_tokens = max(0, num_external_tokens - local_part)
                if remote_tokens > 0:
                    blk_sz = self._hma_scheduler.fa_block_size
                    fa_gidx = self._hma_scheduler.fa_gidx
                    n_rem_blks = min(len(remote_keys), remote_tokens // blk_sz)
                    if n_rem_blks > 0:
                        group_gpu_ids = blocks.get_block_ids()[fa_gidx]
                        start_blk = (num_computed_tokens + local_part) // blk_sz
                        rem_gpu_ids = list(group_gpu_ids[start_blk : start_blk + n_rem_blks])
                        gpu_pool = self._hma_scheduler._gpu_block_pool
                        assert gpu_pool is not None
                        gpu_pool.touch([gpu_pool.blocks[bid] for bid in rem_gpu_ids])
                        self._remote_reqs_to_load[request.request_id] = {
                            "gpu_blocks": rem_gpu_ids,
                            "keys": remote_keys[:n_rem_blks],
                            "event": None,
                            "finished": False,
                        }
            return

        req_id = request.request_id
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None:
            token_ids = request.prompt_token_ids
        all_digests = hash_prefix_blocks(token_ids, self.block_size)

        if hasattr(blocks, "get_block_ids"):
            raw_ids = blocks.get_block_ids()
            block_ids = (
                list(raw_ids[0])
                if raw_ids and isinstance(raw_ids[0], (list, tuple))
                else list(raw_ids)
            )
        else:
            block_ids = list(blocks)

        num_load_blks = num_external_tokens // self.block_size
        matched = self._matched_digests.pop(req_id, None)
        if num_load_blks > 0:
            load_digests = (
                matched[:num_load_blks]
                if matched is not None
                else all_digests[:num_load_blks]
            )
            load_ids = block_ids[: len(load_digests)]
            if load_ids:
                self._pending_loads.append(
                    BlockTransferSpec(
                        request_id=req_id,
                        block_ids=load_ids,
                        prefix_digests=load_digests,
                    )
                )

        # Schedule newly computed suffix full blocks for save_kv_layer.
        num_full = min(len(all_digests), len(block_ids))
        if num_full > num_load_blks:
            self._pending_saves.append(
                BlockTransferSpec(
                    request_id=req_id,
                    block_ids=block_ids[num_load_blks:num_full],
                    prefix_digests=all_digests[num_load_blks:num_full],
                )
            )

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        if not self._standalone_mode:
            if self._hma_scheduler is None:
                return VeriCacheV1Metadata()

            raw = self._hma_scheduler.build_connector_meta(scheduler_output)
            gpu_pool = self._hma_scheduler._gpu_block_pool
            cpu_pool = self._hma_scheduler.cpu_block_pool

            load_event = raw.load_event
            load_gpu_blocks = list(raw.load_gpu_blocks)
            load_keys: list[ObjectKey] = []
            load_event_to_reqs: dict[int, list[str]] = {
                ev: list(rids) for ev, rids in raw.load_event_to_reqs.items()
            }

            def _resolve_cpu_key(cpu_bid: int) -> ObjectKey:
                blk_hash = cpu_pool.blocks[cpu_bid].block_hash
                if blk_hash is not None:
                    key = object_key_from_vllm_block_hash(blk_hash)
                    self._cpu_bid_to_key[cpu_bid] = key
                    return key
                cached_k = self._cpu_bid_to_key.get(cpu_bid)
                if cached_k is not None:
                    return cached_k
                # Evicted after allocation before build_connector_meta;
                # send sentinel miss key so worker triggers clean recompute.
                return (0, 0, 0, 0)

            if raw.load_cpu_blocks:
                for cpu_bid in raw.load_cpu_blocks:
                    load_keys.append(_resolve_cpu_key(cpu_bid))

            # Build per-request load specs so short/local requests complete immediately
            # on `_load_thread` without HoL blocking behind co-scheduled 32K NVMe/RDMA loads!
            reqs_to_load_map = getattr(self._hma_scheduler, "_reqs_to_load", None) or {}
            per_req_specs: dict[str, VeriCacheLoadReqSpec] = {}
            can_build_per_req = True
            if load_event >= 0:
                local_rids = load_event_to_reqs.get(load_event, [])
                if local_rids and all(rid in reqs_to_load_map for rid in local_rids):
                    for rid in local_rids:
                        tm = reqs_to_load_map[rid].transfer_meta
                        if tm is None:
                            continue
                        r_gids = list(tm.gpu_block_ids)
                        r_cids = list(tm.cpu_block_ids)
                        r_keys = [_resolve_cpu_key(cid) for cid in r_cids]
                        per_req_specs[rid] = VeriCacheLoadReqSpec(
                            req_id=rid,
                            load_event=load_event,
                            gpu_blocks=r_gids,
                            cpu_blocks=r_cids,
                            keys=r_keys,
                        )
                elif raw.load_gpu_blocks:
                    can_build_per_req = False

            # Merge queued peer-shared remote loads into both legacy flat lists
            # and per-request specs (merging local + remote for the same req_id).
            seen_load_bids: dict[int, ObjectKey] = {
                bid: key for bid, key in zip(load_gpu_blocks, load_keys)
            }
            pending_remote_rids: list[str] = []
            for rid, st in self._remote_reqs_to_load.items():
                if st["event"] is not None:
                    continue
                spec = per_req_specs.get(rid)
                req_seen: dict[int, ObjectKey] = (
                    {b: k for b, k in zip(spec.gpu_blocks, spec.keys)}
                    if spec is not None
                    else {}
                )
                rem_gids_for_spec: list[int] = []
                rem_keys_for_spec: list[ObjectKey] = []
                for bid, key in zip(st["gpu_blocks"], st["keys"]):
                    prev = seen_load_bids.get(bid)
                    if prev is None:
                        seen_load_bids[bid] = key
                        load_gpu_blocks.append(bid)
                        load_keys.append(key)
                    elif prev != key:
                        logger.warning(
                            "VeriCacheConnector[SCHEDULER]: skipping conflicting remote load "
                            "on gpu_bid=%d for req=%s",
                            bid,
                            rid,
                        )
                    if bid not in req_seen:
                        req_seen[bid] = key
                        rem_gids_for_spec.append(bid)
                        rem_keys_for_spec.append(key)
                if can_build_per_req:
                    if spec is not None:
                        spec.gpu_blocks.extend(rem_gids_for_spec)
                        spec.keys.extend(rem_keys_for_spec)
                    else:
                        per_req_specs[rid] = VeriCacheLoadReqSpec(
                            req_id=rid,
                            load_event=-1,  # filled right below once load_event assigned
                            gpu_blocks=rem_gids_for_spec,
                            cpu_blocks=[],
                            keys=rem_keys_for_spec,
                        )
                pending_remote_rids.append(rid)

            if pending_remote_rids:
                if load_event < 0:
                    load_event = self._remote_load_event_counter
                    self._remote_load_event_counter += 1
                ev_rids = load_event_to_reqs.setdefault(load_event, [])
                for rid in pending_remote_rids:
                    self._remote_reqs_to_load[rid]["event"] = load_event
                    if rid not in ev_rids:
                        ev_rids.append(rid)
                    if can_build_per_req and rid in per_req_specs and per_req_specs[rid].load_event < 0:
                        per_req_specs[rid].load_event = load_event

            load_reqs = list(per_req_specs.values()) if can_build_per_req else []

            store_gpu_blocks = list(raw.store_gpu_blocks) if raw.store_gpu_blocks else []
            store_keys: list[ObjectKey] = []
            if store_gpu_blocks:
                assert gpu_pool is not None
                for gpu_bid in store_gpu_blocks:
                    blk_hash = gpu_pool.blocks[gpu_bid].block_hash
                    assert blk_hash is not None, (
                        f"Missing block_hash on GPU block {gpu_bid}"
                    )
                    store_keys.append(object_key_from_vllm_block_hash(blk_hash))

                # Cross-node store dedup: skip re-storing blocks confirmed resident on
                # a remote peer (eliminates 2x cluster write amp on remote-hit prompts!).
                if (
                    self._dedup_remote_stores
                    and self._remote_known_keys
                    and raw.store_event >= 0
                ):
                    store_map = getattr(self._hma_scheduler, "_store_event_to_blocks", None) or {}
                    transfer = store_map.get(raw.store_event)
                    cpu_ids = getattr(transfer, "cpu_block_ids", ()) if transfer is not None else ()
                    if len(cpu_ids) == len(store_gpu_blocks):
                        kept_gids: list[int] = []
                        kept_keys: list[ObjectKey] = []
                        kept_cids: list[int] = []
                        skipped_cids: list[int] = []
                        known = self._remote_known_keys
                        for gid, cid, key in zip(store_gpu_blocks, cpu_ids, store_keys):
                            if key in known:
                                known.move_to_end(key)
                                skipped_cids.append(cid)
                            else:
                                kept_gids.append(gid)
                                kept_keys.append(key)
                                kept_cids.append(cid)
                        if skipped_cids:
                            self._store_event_cpu_plan[raw.store_event] = (kept_cids, skipped_cids)
                            self._sched_remote_store_deduped += len(skipped_cids)
                            store_gpu_blocks = kept_gids
                            store_keys = kept_keys

                if raw.store_event >= 0 and self._sched_dir_client is not None and store_keys:
                    self._store_event_to_keys[raw.store_event] = list(store_keys)

            return VeriCacheV1Metadata(
                load_event=load_event,
                load_gpu_blocks=load_gpu_blocks,
                load_cpu_blocks=list(raw.load_cpu_blocks or ()),
                load_keys=load_keys,
                load_event_to_reqs=load_event_to_reqs,
                load_reqs=load_reqs,
                store_event=raw.store_event,
                store_gpu_blocks=store_gpu_blocks,
                store_keys=store_keys,
                need_flush=raw.need_flush,
            )

        meta = VeriCacheConnectorMetadata(
            loads=self._pending_loads,
            saves=self._pending_saves,
        )
        self._pending_loads = []
        self._pending_saves = []
        return meta

    def update_connector_output(self, connector_output: Any) -> None:
        if not self._standalone_mode and self._hma_scheduler is not None:
            wmeta = getattr(connector_output, "kv_connector_worker_meta", None)
            completed = getattr(wmeta, "completed_store_events", None) or {}
            committed_counts = getattr(wmeta, "store_committed_counts", None) or {}
            missed_cpu = getattr(wmeta, "missed_local_cpu_blocks", None) or set()
            missed_remote = getattr(wmeta, "missed_remote_keys", None) or ()
            missed_remote_set = set(missed_remote)
            if missed_remote_set:
                for k in missed_remote_set:
                    self._remote_known_keys.pop(k, None)

            # Snapshot CPU block IDs for any skipped (remote-deduped) or partially
            # committed store events before SimpleCPUOffloadScheduler pops them from
            # `_store_event_to_blocks`.
            partial_evict_cpu_ids: set[int] = set()
            if completed:
                store_map = getattr(self._hma_scheduler, "_store_event_to_blocks", None) or {}
                for ev_idx in completed:
                    plan = self._store_event_cpu_plan.pop(ev_idx, None)
                    if plan is not None:
                        kept_cpu_ids, skipped_cpu_ids = plan
                        partial_evict_cpu_ids.update(skipped_cpu_ids)
                        if ev_idx in committed_counts:
                            n_ok = int(committed_counts[ev_idx])
                            if n_ok < len(kept_cpu_ids):
                                partial_evict_cpu_ids.update(kept_cpu_ids[n_ok:])
                        continue
                    if ev_idx not in committed_counts:
                        continue
                    transfer = store_map.get(ev_idx)
                    if transfer is None:
                        continue
                    n_ok = int(committed_counts[ev_idx])
                    cpu_ids = getattr(transfer, "cpu_block_ids", ())
                    if n_ok < len(cpu_ids):
                        partial_evict_cpu_ids.update(cpu_ids[n_ok:])

            self._hma_scheduler.update_connector_output(connector_output)

            # Evict stale/skipped/uncommitted virtual CPU block hashes so recompute
            # retries and local lookups never re-hit unstored blocks.
            to_evict = partial_evict_cpu_ids | set(missed_cpu)
            if to_evict:
                cpu_pool = getattr(self._hma_scheduler, "cpu_block_pool", None)
                if cpu_pool is not None and hasattr(cpu_pool, "evict_blocks"):
                    valid_ids = {
                        bid for bid in to_evict if 0 <= bid < len(cpu_pool.blocks)
                    }
                    if valid_ids:
                        cpu_pool.evict_blocks(valid_ids)

            for req_id in list(getattr(connector_output, "finished_recving", None) or []):
                rem_keys = self._cleanup_remote_load_req(req_id)
                if rem_keys:
                    if missed_remote_set:
                        rem_keys = [k for k in rem_keys if k not in missed_remote_set]
                    self._record_remote_known_keys(rem_keys)

            if self._sched_dir_client is not None and completed:
                keys_to_publish: list[ObjectKey] = []
                for ev_idx in completed:
                    ks = self._store_event_to_keys.pop(ev_idx, None)
                    if not ks:
                        continue
                    n_ok = int(committed_counts.get(ev_idx, len(ks)))
                    if n_ok > 0:
                        keys_to_publish.extend(ks[:n_ok])
                if keys_to_publish:
                    self._sched_dir_client.register_batch(keys_to_publish)

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self._standalone_mode and self._hma_scheduler is not None:
            req_id = request.request_id
            self._pending_remote_hits.pop(req_id, None)
            rem = self._remote_reqs_to_load.get(req_id)
            if rem is not None:
                if rem["event"] is not None:
                    rem["finished"] = True
                else:
                    self._cleanup_remote_load_req(req_id)
            return self._hma_scheduler.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self._standalone_mode and self._hma_scheduler is not None:
            req_id = request.request_id
            self._pending_remote_hits.pop(req_id, None)
            rem = self._remote_reqs_to_load.get(req_id)
            if rem is not None:
                if rem["event"] is not None:
                    rem["finished"] = True
                else:
                    self._cleanup_remote_load_req(req_id)
            return self._hma_scheduler.request_finished_all_groups(request, block_ids)
        return False, None

    def has_pending_push_work(self) -> bool:
        if not self._standalone_mode and self._hma_scheduler is not None:
            return self._hma_scheduler.has_pending_stores()
        return False

    def take_events(self) -> Iterable[Any]:
        if not self._standalone_mode and self._hma_scheduler is not None:
            return self._hma_scheduler.take_events()
        return []

    def reset_cache(self) -> bool | None:
        if not self._standalone_mode and self._hma_scheduler is not None:
            ok = self._hma_scheduler.reset()
            if ok:
                self._cpu_bid_to_key.clear()
                self._remote_known_keys.clear()
                self._store_event_cpu_plan.clear()
            return ok
        return None

    # =======================================================================
    # Worker-side hooks
    # =======================================================================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self._hma_worker is not None and isinstance(
            connector_metadata, VeriCacheV1Metadata
        ):
            self._hma_worker.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self._hma_worker is not None:
            self._hma_worker.clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        if self._hma_worker is not None and isinstance(
            kv_connector_metadata, VeriCacheV1Metadata
        ):
            self._hma_worker.handle_preemptions(kv_connector_metadata)

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register vLLM's per-layer paged KV cache GPU tensors."""
        if not self._standalone_mode:
            if self._hma_worker is not None:
                self._hma_worker.register_kv_caches(kv_caches)
            return

        assert self.client is not None
        self._kv_caches = dict(kv_caches)
        for layer_name, tensor in self._kv_caches.items():
            block_bytes = self._block_nbytes(tensor)
            if block_bytes != self.client.frame_size:
                raise ValueError(
                    f"Layer {layer_name} paged block size ({block_bytes} B) "
                    f"!= VeriCache frame_size ({self.client.frame_size} B)"
                )

    def _block_nbytes(self, kv_layer: Any) -> int:
        if self.kv_layout == "2_N_...":
            per_kv = 1
            for d in kv_layer.shape[2:]:
                per_kv *= int(d)
            return 2 * per_kv * int(kv_layer.element_size())
        per_blk = 1
        for d in kv_layer.shape[1:]:
            per_blk *= int(d)
        return per_blk * int(kv_layer.element_size())

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Location-transparently load all scheduled prefix blocks into GPU paged KV tensors."""
        if not self._standalone_mode:
            if self._hma_worker is not None:
                self._hma_worker.start_load_kv()
            return

        import torch

        assert self.client is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, VeriCacheConnectorMetadata)
        self._last_load_hits.clear()
        if not meta.loads:
            return

        flat_block_ids: list[int] = []
        flat_digests: list[bytes] = []
        req_ids: set[str] = set()
        for spec in meta.loads:
            req_ids.add(spec.request_id)
            flat_block_ids.extend(spec.block_ids)
            flat_digests.extend(spec.prefix_digests)

        if not flat_block_ids:
            return

        n_blocks = len(flat_block_ids)
        frame_size = self.client.frame_size

        for layer_name, kv_layer in self._kv_caches.items():
            device = kv_layer.device
            stream = (
                torch.cuda.current_stream(device).cuda_stream if kv_layer.is_cuda else 0
            )
            keys = [layer_block_key(d, layer_name) for d in flat_digests]

            if kv_layer.is_cuda:
                staging = torch.empty(
                    (n_blocks, frame_size), dtype=torch.uint8, device=device
                )
                base_ptr = staging.data_ptr()
                dst_ptrs = [base_ptr + i * frame_size for i in range(n_blocks)]
                hits = self.client.get_gpu_batch(keys, dst_ptrs, cuda_stream=stream)
                self._last_load_hits[layer_name] = hits

                if self.kv_layout == "2_N_...":
                    block_shape = (2,) + tuple(kv_layer.shape[2:])
                    typed = staging.view(kv_layer.dtype).reshape((n_blocks,) + block_shape)
                    idx_t = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
                    kv_layer[:, idx_t] = typed.transpose(0, 1)
                else:
                    block_shape = tuple(kv_layer.shape[1:])
                    typed = staging.view(kv_layer.dtype).reshape((n_blocks,) + block_shape)
                    idx_t = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
                    kv_layer[idx_t] = typed
            else:
                hits = []
                for blk_id, key in zip(flat_block_ids, keys):
                    res = self.client.get_host(key)
                    if res is None:
                        hits.append(None)
                        continue
                    hit, raw = res
                    hits.append(hit)
                    src = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(
                        kv_layer.dtype
                    )
                    if self.kv_layout == "2_N_...":
                        kv_layer[:, blk_id].copy_(
                            src.reshape((2,) + tuple(kv_layer.shape[2:]))
                        )
                    else:
                        kv_layer[blk_id].copy_(src.reshape(tuple(kv_layer.shape[1:])))
                self._last_load_hits[layer_name] = hits

        self._finished_recving.update(req_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """Gather newly computed paged KV blocks from `kv_layer` and publish into VeriCache."""
        if not self._standalone_mode:
            # In HMA v1 mode, saves execute across all layers/groups in wait_for_save().
            return

        import torch

        assert self.client is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, VeriCacheConnectorMetadata)
        if not meta.saves:
            return

        flat_block_ids: list[int] = []
        flat_digests: list[bytes] = []
        req_ids: set[str] = set()
        for spec in meta.saves:
            req_ids.add(spec.request_id)
            flat_block_ids.extend(spec.block_ids)
            flat_digests.extend(spec.prefix_digests)

        if not flat_block_ids:
            return

        keys = [layer_block_key(d, layer_name) for d in flat_digests]
        n_blocks = len(flat_block_ids)
        frame_size = self.client.frame_size

        if kv_layer.is_cuda:
            device = kv_layer.device
            stream = torch.cuda.current_stream(device).cuda_stream
            idx_t = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
            if self.kv_layout == "2_N_...":
                gathered = kv_layer[:, idx_t].transpose(0, 1).contiguous()
            else:
                gathered = kv_layer[idx_t].contiguous()
            raw_u8 = gathered.view(torch.uint8).reshape(n_blocks, frame_size)
            base_ptr = raw_u8.data_ptr()
            src_ptrs = [base_ptr + i * frame_size for i in range(n_blocks)]
            self.client.put_gpu_batch(keys, src_ptrs, cuda_stream=stream)
        else:
            for blk_id, key in zip(flat_block_ids, keys):
                if self.kv_layout == "2_N_...":
                    blk = kv_layer[:, blk_id].contiguous()
                else:
                    blk = kv_layer[blk_id].contiguous()
                self.client.put_host(key, blk.view(torch.uint8).numpy().tobytes())

        if self.on_save_keys_fn is not None and layer_name == self.probe_layer_name:
            self.on_save_keys_fn(keys)

        self._finished_sending.update(req_ids)

    def wait_for_save(self) -> None:
        if not self._standalone_mode and self._hma_worker is not None:
            self._hma_worker.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if not self._standalone_mode:
            if self._hma_worker is not None:
                return self._hma_worker.get_finished(finished_req_ids)
            return None, None

        done_send = self._finished_sending.intersection(finished_req_ids)
        done_recv = self._finished_recving.intersection(finished_req_ids)
        self._finished_sending.difference_update(done_send)
        self._finished_recving.difference_update(done_recv)
        return (done_send or None), (done_recv or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if not self._standalone_mode and self._hma_worker is not None:
            return self._hma_worker.get_block_ids_with_load_errors()
        return set()

    def build_connector_worker_meta(self) -> Any:
        if not self._standalone_mode and self._hma_worker is not None:
            return self._hma_worker.build_connector_worker_meta()
        return None

    def shutdown(self) -> None:
        if not self._standalone_mode:
            if self._hma_worker is not None:
                self._hma_worker.shutdown()
            if self._local_lookup_client is not None:
                logger.warning(
                    "VeriCacheConnector[SCHEDULER] local_lookup stats: "
                    "rpcs=%d total_ms=%.1f drift_evicted_blocks=%d "
                    "remote_hit_blocks=%d remote_store_deduped=%d",
                    self._sched_local_lookup_rpcs,
                    self._sched_local_lookup_ms,
                    self._sched_local_drift_evicted,
                    self._sched_remote_hit_blocks,
                    self._sched_remote_store_deduped,
                )
                self._local_lookup_client.close()
                self._local_lookup_client = None
            if self._sched_dir_client is not None:
                self._sched_dir_client.close()
                self._sched_dir_client = None

    @property
    def last_load_hits(self) -> dict[str, list[HitInfo | None]]:
        """Diagnostic hit locations from the most recent standalone `start_load_kv`."""
        return self._last_load_hits


__all__ = [
    "BlockTransferSpec",
    "KVConnectorBase_V1",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "SupportsHMA",
    "VeriCacheConnector",
    "VeriCacheConnectorMetadata",
    "VeriCacheLoadReqSpec",
    "VeriCacheV1Metadata",
    "hash_prefix_blocks",
    "layer_block_key",
    "object_key_from_vllm_block_hash",
]
