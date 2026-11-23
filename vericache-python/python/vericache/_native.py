"""ctypes handles for Rust-owned host frames and staging buffers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import ctypes as C
from functools import lru_cache
from math import prod
import os
from struct import Struct, error as StructError
import sys
from threading import Condition
from typing import TYPE_CHECKING

from ._ffi import last_error, load_lib
from .config import VeriCacheConfig

if TYPE_CHECKING:
    import torch

    from . import ObjectKey

_VOID = C.c_void_p
_SIZE = C.c_size_t
_U64 = C.c_uint64
_I32 = C.c_int32
_PVOID = C.POINTER(_VOID)
_PSIZE = C.POINTER(_SIZE)
_PU64 = C.POINTER(_U64)
_PI32 = C.POINTER(_I32)
_BUSY = -4
_KEY_WORDS = Struct("<4Q")
_U64_MASK = (1 << 64) - 1
_EXTENDED_STATS = (
    "dram_hits",
    "nvme_hits",
    "nvme_promotions",
    "async_writebacks",
    "clean_dram_evictions",
    "active_leases",
)
_STATS = ("puts", "gets", "hits", "misses", "evictions", "bytes_written", "bytes_read")


class NativeVeriCacheError(RuntimeError):
    """A failed native bridge operation."""


@lru_cache(maxsize=None)
def _bind_local(lib):
    signatures = {
        "engine_create": (_VOID, _SIZE, _SIZE, _SIZE, _SIZE, C.c_char_p, _SIZE),
        "engine_destroy": (None, _VOID),
        "engine_frame_size": (_SIZE, _VOID),
        "engine_num_shards": (_SIZE, _VOID),
        "engine_put_key256": (_I32, _VOID, _PU64, _VOID, _SIZE),
        "engine_get_key256": (_I32, _VOID, _PU64, _VOID, _SIZE, C.POINTER(_SIZE)),
        "engine_contains_key256": (_I32, _VOID, _PU64),
        "engine_has_cold_data": (_I32, _VOID),
        "engine_evict_key256": (_I32, _VOID, _PU64),
        "engine_stats": (_I32, _VOID, *[_PU64] * len(_STATS)),
        "engine_create_with_options": (
            _VOID,
            _SIZE,
            _SIZE,
            _SIZE,
            _SIZE,
            C.c_char_p,
            _SIZE,
            C.c_uint32,
        ),
        "engine_is_slab_cuda_pinned": (_I32, _VOID),
        "engine_acquire_read_lease_key256": (_I32, _VOID, _PU64, _PVOID, _PSIZE, _PU64),
        "engine_release_read_lease": (_I32, _VOID, _U64),
        "engine_acquire_read_lease_batch_key256": (
            _I32,
            _VOID,
            _PU64,
            _SIZE,
            _PVOID,
            _PSIZE,
            _PU64,
            _PI32,
        ),
        "engine_release_read_lease_batch": (_I32, _VOID, _PU64, _SIZE),
        "engine_contains_batch_key256": (
            _I32,
            _VOID,
            _PU64,
            _SIZE,
            C.POINTER(C.c_uint8),
        ),
        "engine_put_batch_key256": (_I32, _VOID, _PU64, _PVOID, _PSIZE, _SIZE, _PI32),
        "engine_get_batch_key256": (
            _I32,
            _VOID,
            _PU64,
            _PVOID,
            _PSIZE,
            _SIZE,
            _PSIZE,
            _PI32,
        ),
        "engine_poll_writeback": (_I32, _VOID),
        "engine_set_writeback_policy": (_I32, _VOID, _SIZE),
        "engine_extended_stats": (_I32, _VOID, *[_PU64] * len(_EXTENDED_STATS)),
        "engine_drain_evictions_key256": (C.c_ssize_t, _VOID, _PU64, _SIZE),
        "staging_alloc": (_VOID, _SIZE, _I32),
        "staging_ptr": (_VOID, _VOID),
        "staging_len": (_SIZE, _VOID),
        "staging_is_cuda_pinned": (_I32, _VOID),
        "staging_free": (None, _VOID),
    }
    for name, (result, *args) in signatures.items():
        function = getattr(lib, f"vericache_{name}")
        function.restype = result
        function.argtypes = args
    return lib


def _library(path=None):
    try:
        return _bind_local(load_lib(path))
    except OSError as error:
        raise NativeVeriCacheError(
            f"Could not load VeriCache library: {error}"
        ) from error


def _splitmix64(key: int) -> int:
    key = (key + 0x9E3779B97F4A7C15) & _U64_MASK
    key = ((key ^ (key >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    key = ((key ^ (key >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return key ^ (key >> 31)


def _key_words(key: ObjectKey | int) -> ObjectKey:
    if isinstance(key, int) and 0 <= key <= _U64_MASK:
        # Match Rust object_key(u64), including when batches mix tuples and ints.
        h1 = _splitmix64(key ^ 0x9E3779B97F4A7C15)
        h2 = _splitmix64(h1 ^ 0xBF58476D1CE4E5B9)
        h3 = _splitmix64(h2 ^ 0x94D049BB133111EB)
        return key, h1, h2, h3
    if isinstance(key, tuple) and len(key) == 4:
        return key
    raise ValueError("key must be four unsigned 64-bit words or a legacy uint64")


def _key(key):
    try:
        return (_U64 * 4).from_buffer_copy(_KEY_WORDS.pack(*_key_words(key)))
    except StructError as error:
        raise ValueError("key words must be unsigned 64-bit integers") from error


def _check(status, operation):
    if status < 0:
        raise NativeVeriCacheError(f"VeriCache {operation} failed with status {status}")
    return status


class _StagingAllocation:
    def __init__(self, lib, size, prefer_cuda_pinned):
        self.lib = lib
        self.handle = lib.vericache_staging_alloc(size, prefer_cuda_pinned)
        if not self.handle:
            raise NativeVeriCacheError(
                f"Could not allocate {size} bytes of host staging memory: {last_error(lib)}"
            )

    def __del__(self):
        if self.handle:
            self.lib.vericache_staging_free(self.handle)


class PinnedStagingBuffer:
    """The views keep the native allocation alive, including after close()."""

    def __init__(self, size, prefer_cuda_pinned=True, lib_path=None, *, _lib=None):
        if type(size) is not int or not 0 < size <= sys.maxsize:
            raise ValueError("staging size must be a positive addressable byte count")
        lib = _lib if _lib is not None else _library(lib_path)
        owner = _StagingAllocation(lib, size, prefer_cuda_pinned)
        handle = owner.handle
        self.size = lib.vericache_staging_len(handle)
        self.is_cuda_pinned = bool(lib.vericache_staging_is_cuda_pinned(handle))
        array = (C.c_uint8 * self.size).from_address(lib.vericache_staging_ptr(handle))
        array._owner = owner
        self._array = array

    def _storage(self):
        array = self._array
        if array is None:
            raise NativeVeriCacheError("staging buffer is closed")
        return array

    @property
    def ptr(self):
        """A borrowed pointer. Keep the buffer or one of its views alive."""
        return C.addressof(self._storage())

    @property
    def memoryview(self):
        return memoryview(self._storage()).cast("B")

    def __len__(self):
        return self.size

    def as_tensor(self, dtype: "torch.dtype", shape) -> "torch.Tensor":
        import torch

        shape = tuple(shape)
        count = prod(shape)
        if any(dim < 0 for dim in shape) or count <= 0:
            raise ValueError("staging tensor must have a nonempty, concrete shape")
        if count * dtype.itemsize > self.size:
            raise ValueError("tensor does not fit in the staging buffer")
        return torch.frombuffer(self.memoryview, dtype=dtype, count=count).view(shape)

    def close(self):
        self._array = None

    def __enter__(self):
        self._storage()
        return self

    def __exit__(self, *_):
        self.close()


def copy_tensor(dst, src, staging, *, record_event=True):
    """Copy on the current stream and return its event for an asynchronous hop."""
    import torch

    device = src.device if src.is_cuda else dst.device
    with torch.no_grad():
        if device.type != "cuda":
            dst.copy_(src)
            return None
        with torch.cuda.device(device):
            dst.copy_(src, non_blocking=staging.is_cuda_pinned)
            if staging.is_cuda_pinned and record_event:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                return event
    return None


class _LeaseOwner:
    __slots__ = ("engine", "id", "ptr", "length", "exports", "released")

    def __init__(self, engine, lease_id, ptr, length):
        self.engine = engine
        self.id = lease_id
        self.ptr = ptr
        self.length = length
        self.exports = 0
        self.released = False


class _LeaseExport:
    __slots__ = ("owner",)

    def __init__(self, owner):
        self.owner = owner

    def __del__(self):
        self.owner.engine._end_lease_export(self.owner)


class ReadLease:
    """Borrow immutable slab bytes. Torch tensors must only be read.

    Exported views retain the native lease until the last view goes away.
    A raw pointer is valid only while the caller keeps the lease open.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner):
        self._owner = owner

    @property
    def engine(self):
        return self._owner.engine

    @property
    def lease_id(self):
        return self._owner.id

    @property
    def ptr(self):
        with self.engine._cv:
            self._check_open()
            return self._owner.ptr

    @property
    def length(self):
        return self._owner.length

    def __len__(self):
        return self.length

    @property
    def is_cuda_pinned(self):
        return self.engine.is_slab_cuda_pinned

    def _check_open(self):
        if self._owner.released:
            raise NativeVeriCacheError("read lease is released")

    def _export(self):
        owner = self._owner
        with owner.engine._cv:
            self._check_open()
            array = (C.c_uint8 * owner.length).from_address(owner.ptr)
            owner.exports += 1
            array._lease = _LeaseExport(owner)
            return array

    @property
    def memoryview(self):
        return memoryview(self._export()).cast("B").toreadonly()

    def as_tensor(self, dtype: "torch.dtype", shape) -> "torch.Tensor":
        import torch

        shape = tuple(shape)
        count = prod(shape)
        if any(dim < 0 for dim in shape) or count <= 0:
            raise ValueError("read lease tensor must have a nonempty, concrete shape")
        if count * dtype.itemsize > self.length:
            raise ValueError("tensor does not fit in the read lease")
        # PyTorch has no read-only CPU tensor. The exporter keeps the slab
        # pinned for the tensor lifetime, and callers must treat it as immutable.
        tensor = torch.frombuffer(self._export(), dtype=dtype, count=count)
        return tensor if len(shape) == 1 else tensor.view(shape)

    def release(self):
        self.engine.release_read_lease_batch((self.lease_id,))

    def __enter__(self):
        with self.engine._cv:
            self._check_open()
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        if hasattr(self, "_owner") and not self._owner.released:
            self.release()


def _keys(keys):
    keys = tuple(keys)
    packed = bytearray(len(keys) * _KEY_WORDS.size)
    try:
        for i, key in enumerate(keys):
            _KEY_WORDS.pack_into(packed, i * _KEY_WORDS.size, *_key_words(key))
    except StructError as error:
        raise ValueError("key words must be unsigned 64-bit integers") from error
    return (_U64 * (len(keys) * 4)).from_buffer(packed), len(keys)


def _pointer_batch(ptrs, lengths, count):
    ptrs, lengths = tuple(ptrs), tuple(lengths)
    if len(ptrs) != count or len(lengths) != count:
        raise ValueError("keys, pointers and lengths must have the same size")
    if any(
        type(length) is not int or not 0 <= length <= sys.maxsize for length in lengths
    ):
        raise ValueError("buffer lengths must be addressable byte counts")
    return (_VOID * count)(*ptrs), (_SIZE * count)(*lengths)


class NativeVeriCacheEngine:
    def __init__(self, config: VeriCacheConfig | None = None, **options):
        if config is not None and options:
            raise TypeError("pass a VeriCacheConfig or keyword options")
        config = config if config is not None else VeriCacheConfig(**options)
        self.config = config
        self._cv = Condition()
        self._active = 0
        self._closing = False
        self._leases = {}
        self._handle = None
        self._lib = _library(config.lib_path)
        for value in (
            config.num_shards,
            config.host_frames_per_shard,
            config.nvme_frames_per_shard,
            config.frame_size,
        ):
            if value > sys.maxsize:
                raise ValueError("engine size exceeds the addressable range")
        path = os.fspath(config.nvme_path).encode() if config.nvme_path else b""
        self._handle = self._lib.vericache_engine_create_with_options(
            config.num_shards,
            config.host_frames_per_shard,
            config.nvme_frames_per_shard,
            config.frame_size,
            path,
            len(path),
            int(config.prefer_cuda_pinned) | (2 if config.ipc_endpoint else 0),
        )
        if not self._handle:
            kind = (
                f"DRAM and NVMe ({config.nvme_path})"
                if config.nvme_frames_per_shard
                else "DRAM"
            )
            raise NativeVeriCacheError(
                f"Could not create the VeriCache engine with {kind}: {last_error(self._lib)}"
            )
        self.frame_size = self._lib.vericache_engine_frame_size(self._handle)
        self.num_shards = self._lib.vericache_engine_num_shards(self._handle)
        self.is_slab_cuda_pinned = bool(
            self._lib.vericache_engine_is_slab_cuda_pinned(self._handle)
        )

    @contextmanager
    def _borrow(self, *, allow_closing=False):
        # ctypes releases the GIL. A shared borrow keeps the handle live without
        # serializing Rust calls that can use different shards.
        with self._cv:
            if (self._closing and not allow_closing) or not self._handle:
                raise NativeVeriCacheError("engine is closed")
            self._active += 1
            handle = self._handle
        try:
            yield handle
        finally:
            with self._cv:
                self._active -= 1
                self._destroy_if_idle()
                if not self._active:
                    self._cv.notify_all()

    def _destroy_if_idle(self):
        if self._closing and not self._active and not self._leases and self._handle:
            self._lib.vericache_engine_destroy(self._handle)
            self._handle = None

    def close(self):
        with self._cv:
            self._closing = True
            self._cv.wait_for(lambda: self._active == 0)
            # Open leases and exported tensors keep the handle alive. The last
            # release destroys it, even if close is called by a lease owner.
            self._destroy_if_idle()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def make_staging(self, size=None):
        return PinnedStagingBuffer(
            self.frame_size if size is None else size,
            self.config.prefer_cuda_pinned,
            _lib=self._lib,
        )

    def has_cold_data(self) -> bool:
        """Report whether any key resides exclusively on NVMe."""
        with self._borrow() as handle:
            return bool(self._lib.vericache_engine_has_cold_data(handle))

    def put(self, key: ObjectKey | int, data) -> None:
        payload = bytes(data)
        ptr = C.cast(C.c_char_p(payload), _VOID)
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_put_key256(
                    handle, _key(key), ptr, len(payload)
                ),
                "put",
            )

    def get(self, key: ObjectKey | int) -> bytes | None:
        data = (C.c_uint8 * self.frame_size)()
        length = self._get(key, data)
        return None if length is None else bytes(memoryview(data).cast("B")[:length])

    def contains(self, key: ObjectKey | int) -> bool:
        with self._borrow() as handle:
            return bool(
                _check(
                    self._lib.vericache_engine_contains_key256(handle, _key(key)),
                    "contains",
                )
            )

    def contains_batch(self, keys) -> list[bool]:
        keys, count = _keys(keys)
        present = (C.c_uint8 * count)()
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_contains_batch_key256(
                    handle,
                    keys,
                    count,
                    present,
                ),
                "contains batch",
            )
        return list(map(bool, present))

    def put_batch(self, keys, ptrs, lens) -> list[int]:
        """Write from borrowed addresses and return one native status per key."""
        keys, count = _keys(keys)
        ptrs, lens = _pointer_batch(ptrs, lens, count)
        statuses = (_I32 * count)()
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_put_batch_key256(
                    handle,
                    keys,
                    ptrs,
                    lens,
                    count,
                    statuses,
                ),
                "put batch",
            )
        return list(statuses)

    def get_batch(self, keys, ptrs, caps) -> tuple[list[int], list[int]]:
        """Read into disjoint borrowed buffers and return lengths and statuses."""
        keys, count = _keys(keys)
        ptrs, caps = _pointer_batch(ptrs, caps, count)
        lengths, statuses = (_SIZE * count)(), (_I32 * count)()
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_get_batch_key256(
                    handle,
                    keys,
                    ptrs,
                    caps,
                    count,
                    lengths,
                    statuses,
                ),
                "get batch",
            )
        return list(lengths), list(statuses)

    def _lease(self, lease_id, ptr, length) -> ReadLease:
        owner = _LeaseOwner(self, lease_id, ptr, length)
        with self._cv:
            self._leases[lease_id] = owner
        return ReadLease(owner)

    def acquire_read_lease(self, key: ObjectKey | int) -> ReadLease | None:
        ptr, length, lease_id = _VOID(), _SIZE(), _U64()
        with self._borrow() as handle:
            status = _check(
                self._lib.vericache_engine_acquire_read_lease_key256(
                    handle,
                    _key(key),
                    C.byref(ptr),
                    C.byref(length),
                    C.byref(lease_id),
                ),
                "acquire read lease",
            )
            return (
                None
                if status == 1
                else self._lease(lease_id.value, ptr.value, length.value)
            )

    def acquire_read_lease_batch(self, keys, *, with_statuses=False):
        """Return leases in key order. Missing keys have a None entry."""
        keys, count = _keys(keys)
        ptrs, lengths = (_VOID * count)(), (_SIZE * count)()
        ids, statuses = (_U64 * count)(), (_I32 * count)()
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_acquire_read_lease_batch_key256(
                    handle,
                    keys,
                    count,
                    ptrs,
                    lengths,
                    ids,
                    statuses,
                ),
                "acquire read lease batch",
            )
            owners = [
                _LeaseOwner(self, lease_id, ptr, length) if status == 0 else None
                for lease_id, ptr, length, status in zip(ids, ptrs, lengths, statuses)
            ]
            with self._cv:
                self._leases.update(
                    (owner.id, owner) for owner in owners if owner is not None
                )
            leases = [
                ReadLease(owner) if owner is not None else None for owner in owners
            ]
        if with_statuses:
            return leases, list(statuses)
        failure = next((status for status in statuses if status < 0), None)
        if failure is not None:
            self.release_read_lease_batch(
                lease.lease_id for lease in leases if lease is not None
            )
            _check(failure, "acquire read lease batch")
        return leases

    def _release_ready(self, ids):
        if not ids:
            return
        with self._borrow(allow_closing=True) as handle:
            if len(ids) == 1:
                status = self._lib.vericache_engine_release_read_lease(handle, ids[0])
            else:
                array = (_U64 * len(ids))(*ids)
                status = self._lib.vericache_engine_release_read_lease_batch(
                    handle,
                    array,
                    len(ids),
                )
            with self._cv:
                for lease_id in ids:
                    self._leases.pop(lease_id)
            _check(status, "release read lease")

    def _end_lease_export(self, owner):
        with self._cv:
            owner.exports -= 1
            ready = owner.exports == 0 and owner.released
        if ready:
            self._release_ready((owner.id,))

    def release_read_lease_batch(self, lease_ids) -> None:
        ids = tuple(lease_ids)
        if any(
            not isinstance(lease_id, int) or not 0 <= lease_id <= _U64_MASK
            for lease_id in ids
        ):
            raise ValueError("lease IDs must be unsigned 64-bit integers")
        ready = []
        with self._cv:
            for lease_id in ids:
                owner = self._leases.get(lease_id)
                if owner is not None and not owner.released:
                    owner.released = True
                    if not owner.exports:
                        ready.append(lease_id)
        self._release_ready(ready)

    def poll_writeback(self) -> int:
        with self._borrow() as handle:
            return _check(
                self._lib.vericache_engine_poll_writeback(handle), "poll writeback"
            )

    def set_writeback_policy(self, high_watermark_percent: int):
        if (
            type(high_watermark_percent) is not int
            or not 0 <= high_watermark_percent <= 100
        ):
            raise ValueError("writeback high watermark must be a percentage in 0..=100")
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_set_writeback_policy(
                    handle,
                    high_watermark_percent,
                ),
                "set writeback policy",
            )

    def drain_evictions(self) -> list[ObjectKey]:
        capacity = 256
        buf = (_U64 * (capacity * 4))()
        result = []
        with self._borrow() as handle:
            while True:
                count = _check(
                    self._lib.vericache_engine_drain_evictions_key256(
                        handle, buf, capacity
                    ),
                    "drain evictions",
                )
                result.extend(
                    _KEY_WORDS.iter_unpack(
                        memoryview(buf).cast("B")[: count * _KEY_WORDS.size]
                    )
                )
                if count < capacity:
                    return result

    def extended_stats(self) -> dict[str, int]:
        values = [_U64() for _ in _EXTENDED_STATS]
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_extended_stats(
                    handle,
                    *map(C.byref, values),
                ),
                "extended stats",
            )
        return {name: value.value for name, value in zip(_EXTENDED_STATS, values)}

    def evict(self, key: ObjectKey | int) -> bool:
        with self._borrow() as handle:
            return (
                _check(
                    self._lib.vericache_engine_evict_key256(handle, _key(key)), "evict"
                )
                == 0
            )

    def stats(self) -> dict[str, int]:
        values = [_U64() for _ in _STATS]
        with self._borrow() as handle:
            _check(
                self._lib.vericache_engine_stats(handle, *map(C.byref, values)), "stats"
            )
        return {name: value.value for name, value in zip(_STATS, values)}

    def put_from_staging(
        self, key: ObjectKey | int, staging: PinnedStagingBuffer, length=None
    ) -> None:
        data = staging._storage()
        length = len(data) if length is None else length
        if not 0 <= length <= len(data):
            raise ValueError("payload length exceeds staging capacity")
        with self._borrow() as handle:
            status = self._lib.vericache_engine_put_key256(
                handle, _key(key), C.addressof(data), length
            )
            _check(status, "put")

    def _get(self, key, data):
        length = _SIZE()
        with self._borrow() as handle:
            status = self._lib.vericache_engine_get_key256(
                handle,
                _key(key),
                C.addressof(data),
                len(data),
                C.byref(length),
            )
            if status == -3:
                raise ValueError(f"payload requires {length.value} bytes")
            return None if _check(status, "get") == 1 else length.value

    def get_into_staging(
        self, key: ObjectKey | int, staging: PinnedStagingBuffer
    ) -> int | None:
        return self._get(key, staging._storage())

    def put_tensor(
        self, key: ObjectKey | int, tensor: "torch.Tensor", staging=None
    ) -> None:
        length = tensor.numel() * tensor.element_size()
        if length > self.frame_size:
            raise ValueError("tensor exceeds engine frame size")
        if not length:
            self.put(key, b"")
            return
        with self.make_staging() if staging is None else nullcontext(staging) as stage:
            view = stage.as_tensor(tensor.dtype, tensor.shape)
            event = copy_tensor(view, tensor, stage)
            if event is not None:
                event.synchronize()
            self.put_from_staging(key, stage, length)

    def get_tensor(
        self, key: ObjectKey | int, tensor: "torch.Tensor", staging=None
    ) -> bool:
        expected = tensor.numel() * tensor.element_size()
        with self.make_staging() if staging is None else nullcontext(staging) as stage:
            size = self.get_into_staging(key, stage)
            if size is None:
                return False
            if size != expected:
                raise ValueError(
                    f"expected a {expected}-byte tensor, got a {size}-byte payload"
                )
            if size:
                view = stage.as_tensor(tensor.dtype, tensor.shape)
                event = copy_tensor(tensor, view, stage)
                if event is not None:
                    event.synchronize()
            return True
