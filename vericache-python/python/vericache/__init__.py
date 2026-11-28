"""Python client for VeriCache (formally verified, location-transparent KV cache)."""

from __future__ import annotations

import array
import ctypes
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import struct
from typing import Iterable, Sequence

from ._ffi import (
    VeriCacheConfigC,
    VeriCacheHitInfoC,
    VeriCachePeerSpecC,
    VeriCacheStatsC,
    last_error,
    load_lib,
)

ObjectKey = tuple[int, int, int, int]


class HitLocation(IntEnum):
    MISS = 0
    LOCAL_GPU = 1
    LOCAL_DRAM = 2
    LOCAL_NVME = 3
    REMOTE_RDMA_HOST = 4
    REMOTE_GDR_GPU = 5


class TierKind(IntEnum):
    GPU_VRAM = 1
    HOST_DRAM = 2
    LOCAL_NVME = 3


@dataclass(frozen=True)
class PeerSpec:
    peer_node_id: int
    tcp_addr: str
    is_listener: bool = False
    psn_hint: int = 0x1000


@dataclass(frozen=True)
class VersionId:
    origin_node: int
    seq: int


@dataclass(frozen=True)
class HitInfo:
    location: HitLocation
    version: VersionId
    peer_node_id: int = 0


@dataclass(frozen=True)
class CacheStats:
    puts: int
    gpu_hits: int
    dram_hits: int
    nvme_hits: int
    remote_hits: int
    misses: int
    policy_victims_rejected: int
    policy_victims_pinned: int
    remote_nvme_warm_hits: int = 0
    donor_nvme_warms: int = 0
    put_attempts: int = 0
    nvme_spills: int = 0
    nvme_evictions: int = 0
    dram_evictions: int = 0
    dram_reclaims: int = 0
    mw_coop_revokes: int = 0
    mw_unilateral_revokes: int = 0
    host_pinned: int = 0
    hbm_frames: int = 0
    dram_frames: int = 0
    nvme_frames: int = 0
    gdr_enabled: int = 0


def key_from_bytes(payload: bytes) -> ObjectKey:
    """Hash arbitrary bytes into a deterministic 256-bit VeriCache ObjectKey."""
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<4Q", digest)


def key_from_u64(val: int, salt: int = 0) -> ObjectKey:
    """Hash a `(val, salt)` integer pair into a deterministic 256-bit ObjectKey."""
    return key_from_bytes(struct.pack("<2Q", val & 0xFFFFFFFFFFFFFFFF, salt & 0xFFFFFFFFFFFFFFFF))


_HIT_LOC_BY_CODE = tuple(HitLocation(i) for i in range(6))
_SIZE_T_CODE = "Q" if ctypes.sizeof(ctypes.c_size_t) == 8 else "L"


def _pack_key(key: ObjectKey | bytes):
    if isinstance(key, (bytes, bytearray, memoryview)):
        mv = memoryview(key)
        if mv.nbytes != 32:
            raise ValueError(f"Expected 32-byte key, got {mv.nbytes} bytes")
        return (ctypes.c_uint64 * 4).from_buffer_copy(mv)
    return (ctypes.c_uint64 * 4)(*key)


def _pack_keys(keys: Sequence[ObjectKey | bytes]):
    n = len(keys)
    if n == 0:
        return (ctypes.c_uint64 * 0)()
    first = keys[0]
    if isinstance(first, (bytes, bytearray, memoryview)):
        raw = bytearray(n * 32)
        off = 0
        for i, k in enumerate(keys):
            if len(k) != 32:  # type: ignore[arg-type]
                raise ValueError(f"keys[{i}] must be 32 bytes, got {len(k)}")  # type: ignore[arg-type]
            raw[off : off + 32] = k  # type: ignore[assignment]
            off += 32
        return (ctypes.c_uint64 * (n * 4)).from_buffer(raw)
    arr = array.array("Q")
    extend = arr.extend
    for k in keys:
        extend(k)  # type: ignore[arg-type]
    if len(arr) != n * 4:
        raise ValueError("each ObjectKey must be a 4-tuple of uint64")
    return (ctypes.c_uint64 * (n * 4)).from_buffer(arr)


def _pack_size_array(vals: Sequence[int], name: str):
    n = len(vals)
    if n == 0:
        return (ctypes.c_size_t * 0)()
    try:
        arr = array.array(_SIZE_T_CODE, vals)
    except (OverflowError, TypeError):
        c_arr = (ctypes.c_size_t * n)()
        for i, v in enumerate(vals):
            iv = int(v)
            if iv < 0:
                raise ValueError(f"{name}[{i}] must be non-negative, got {iv}")
            c_arr[i] = iv
        return c_arr
    return (ctypes.c_size_t * n).from_buffer(arr)


class VeriCacheError(RuntimeError):
    """Error raised by the VeriCache native runtime."""


class VeriCacheClient:
    """Location-transparent Python client over a dedicated Rust VeriCache worker thread.

    Supports both host bytes/buffers and direct GPU HBM-to-HBM transfers for PyTorch
    CUDA tensors without bouncing through host DRAM.
    """

    def __init__(
        self,
        *,
        node_id: int,
        frame_size: int,
        dram_frames: int,
        gpu_device_id: int | None = None,
        hbm_frames: int = 0,
        nvme_dir: str | None = None,
        nvme_frames: int = 0,
        rdma_device: str | None = None,
        rdma_gid_index: int = 0,
        rdma_windows_per_peer: int = 32,
        rdma_max_rd_atomic: int = 16,
        peers: Sequence[PeerSpec] = (),
    ) -> None:
        if not 0 <= rdma_max_rd_atomic <= 255:
            raise ValueError("rdma_max_rd_atomic must be between 0 and 255")
        # A CUDA device also enables pinned-host DMA when hbm_frames is zero.
        self._lib = load_lib()
        self._frame_size = int(frame_size)
        self._node_id = int(node_id)
        self._handle: int | None = None

        nvme_bytes = nvme_dir.encode("utf-8") if nvme_dir else None
        rdma_bytes = rdma_device.encode("utf-8") if rdma_device else None
        peer_addr_refs = [p.tcp_addr.encode("utf-8") for p in peers]
        peers_arr = (VeriCachePeerSpecC * len(peers))() if peers else None
        for i, p in enumerate(peers):
            peers_arr[i] = VeriCachePeerSpecC(
                peer_node_id=p.peer_node_id,
                tcp_addr=peer_addr_refs[i],
                is_listener=1 if p.is_listener else 0,
                psn_hint=p.psn_hint,
            )

        cfg = VeriCacheConfigC(
            node_id=node_id,
            frame_size=frame_size,
            dram_frames=dram_frames,
            gpu_device_id=-1 if gpu_device_id is None else int(gpu_device_id),
            hbm_frames=hbm_frames,
            nvme_dir=nvme_bytes,
            nvme_frames=nvme_frames,
            rdma_device=rdma_bytes,
            rdma_gid_index=rdma_gid_index,
            rdma_max_rd_atomic=rdma_max_rd_atomic,
            rdma_windows_per_peer=rdma_windows_per_peer,
            peers=peers_arr if peers_arr is not None else ctypes.POINTER(VeriCachePeerSpecC)(),
            num_peers=len(peers),
        )

        handle = self._lib.vericache_create(ctypes.byref(cfg))
        if not handle:
            raise VeriCacheError(last_error(self._lib))
        self._handle = handle

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def node_id(self) -> int:
        return self._node_id

    def _require_handle(self) -> int:
        if not self._handle:
            raise VeriCacheError("VeriCacheClient is closed")
        return self._handle

    def close(self) -> None:
        if self._handle:
            self._lib.vericache_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "VeriCacheClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Host bytes API
    # ------------------------------------------------------------------

    def put_host(self, key: ObjectKey, data: bytes | bytearray | memoryview) -> VersionId:
        h = self._require_handle()
        mv = memoryview(data)
        if mv.nbytes != self._frame_size:
            raise ValueError(f"Expected {self._frame_size} bytes, got {mv.nbytes}")
        buf = (ctypes.c_uint8 * self._frame_size).from_buffer_copy(mv)
        k_arr = _pack_key(key)
        orig = ctypes.c_uint16(0)
        seq = ctypes.c_uint64(0)
        rc = self._lib.vericache_put_host(
            h, k_arr, buf, self._frame_size, ctypes.byref(orig), ctypes.byref(seq)
        )
        if rc != 0:
            raise VeriCacheError(last_error(self._lib))
        return VersionId(origin_node=orig.value, seq=seq.value)

    def get_host(self, key: ObjectKey) -> tuple[HitInfo, bytes] | None:
        h = self._require_handle()
        k_arr = _pack_key(key)
        out_buf = (ctypes.c_uint8 * self._frame_size)()
        info = VeriCacheHitInfoC()
        rc = self._lib.vericache_get_host(h, k_arr, out_buf, self._frame_size, ctypes.byref(info))
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        if rc == 0:
            return None
        hit = HitInfo(
            location=HitLocation(info.location),
            version=VersionId(info.version_origin, info.version_seq),
            peer_node_id=info.peer_node_id,
        )
        return hit, bytes(out_buf)

    # ------------------------------------------------------------------
    # Direct GPU pointer & PyTorch Tensor API (HBM-to-HBM, zero host bounce)
    # ------------------------------------------------------------------

    def put_gpu_ptr(
        self, key: ObjectKey, src_dev_ptr: int, *, cuda_stream: int = 0
    ) -> VersionId:
        h = self._require_handle()
        k_arr = _pack_key(key)
        orig = ctypes.c_uint16(0)
        seq = ctypes.c_uint64(0)
        rc = self._lib.vericache_put_gpu_ptr(
            h,
            k_arr,
            int(src_dev_ptr),
            self._frame_size,
            int(cuda_stream),
            ctypes.byref(orig),
            ctypes.byref(seq),
        )
        if rc != 0:
            raise VeriCacheError(last_error(self._lib))
        return VersionId(origin_node=orig.value, seq=seq.value)

    def get_gpu_ptr(
        self, key: ObjectKey, dst_dev_ptr: int, *, cuda_stream: int = 0
    ) -> HitInfo | None:
        h = self._require_handle()
        k_arr = _pack_key(key)
        info = VeriCacheHitInfoC()
        rc = self._lib.vericache_get_gpu_ptr(
            h,
            k_arr,
            int(dst_dev_ptr),
            self._frame_size,
            int(cuda_stream),
            ctypes.byref(info),
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        if rc == 0:
            return None
        return HitInfo(
            location=HitLocation(info.location),
            version=VersionId(info.version_origin, info.version_seq),
            peer_node_id=info.peer_node_id,
        )

    def put_gpu_batch(
        self,
        keys: Sequence[ObjectKey],
        src_dev_ptrs: Sequence[int],
        *,
        cuda_stream: int = 0,
    ) -> int:
        if len(keys) != len(src_dev_ptrs):
            raise ValueError("keys and src_dev_ptrs length mismatch")
        if not keys:
            return 0
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        p_arr = (ctypes.c_size_t * len(src_dev_ptrs))(*[int(p) for p in src_dev_ptrs])
        rc = self._lib.vericache_put_gpu_batch(
            h, k_arr, p_arr, len(keys), self._frame_size, int(cuda_stream)
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        return int(rc)

    def put_gpu_dram_batch(
        self,
        keys: Sequence[ObjectKey],
        src_dev_ptrs: Sequence[int],
        *,
        cuda_stream: int = 0,
    ) -> int:
        """Store GPU frames directly into Pinned Host DRAM via batched async D2H DMA."""
        if len(keys) != len(src_dev_ptrs):
            raise ValueError("keys and src_dev_ptrs length mismatch")
        if not keys:
            return 0
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        p_arr = (ctypes.c_size_t * len(src_dev_ptrs))(*[int(p) for p in src_dev_ptrs])
        rc = self._lib.vericache_put_gpu_dram_batch(
            h, k_arr, p_arr, len(keys), self._frame_size, int(cuda_stream)
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        return int(rc)

    def get_gpu_batch(
        self,
        keys: Sequence[ObjectKey],
        dst_dev_ptrs: Sequence[int],
        *,
        cuda_stream: int = 0,
    ) -> list[HitInfo | None]:
        if len(keys) != len(dst_dev_ptrs):
            raise ValueError("keys and dst_dev_ptrs length mismatch")
        if not keys:
            return []
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        p_arr = (ctypes.c_size_t * len(dst_dev_ptrs))(*[int(p) for p in dst_dev_ptrs])
        out_arr = (VeriCacheHitInfoC * len(keys))()
        rc = self._lib.vericache_get_gpu_batch(
            h, k_arr, p_arr, len(keys), self._frame_size, int(cuda_stream), out_arr
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        results: list[HitInfo | None] = []
        for i in range(len(keys)):
            info = out_arr[i]
            if info.found:
                results.append(
                    HitInfo(
                        location=_HIT_LOC_BY_CODE[info.location],
                        version=VersionId(info.version_origin, info.version_seq),
                        peer_node_id=info.peer_node_id,
                    )
                )
            else:
                results.append(None)
        return results

    def put_gpu_paged_dram_batch(
        self,
        keys: Sequence[ObjectKey],
        region_base_ptrs: Sequence[int],
        region_block_strides: Sequence[int],
        region_offsets: Sequence[int],
        region_sizes: Sequence[int],
        gpu_block_ids: Sequence[int],
        cuda_stream: int = 0,
    ) -> int:
        """Store paged multi-layer GPU KV blocks directly into Host DRAM via batched D2H DMA."""
        if len(keys) != len(gpu_block_ids):
            raise ValueError("keys and gpu_block_ids length mismatch")
        num_regions = len(region_base_ptrs)
        if (
            num_regions == 0
            or len(region_block_strides) != num_regions
            or len(region_offsets) != num_regions
            or len(region_sizes) != num_regions
        ):
            raise ValueError("region descriptor sequences must be non-empty and equal length")
        if not keys:
            return 0
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        b_arr = _pack_size_array(gpu_block_ids, "gpu_block_ids")
        base_arr = _pack_size_array(region_base_ptrs, "region_base_ptrs")
        stride_arr = _pack_size_array(region_block_strides, "region_block_strides")
        off_arr = _pack_size_array(region_offsets, "region_offsets")
        size_arr = _pack_size_array(region_sizes, "region_sizes")
        rc = self._lib.vericache_put_gpu_paged_dram_batch(
            h,
            k_arr,
            b_arr,
            len(keys),
            base_arr,
            stride_arr,
            off_arr,
            size_arr,
            num_regions,
            int(cuda_stream),
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        return int(rc)

    def get_gpu_paged_batch(
        self,
        keys: Sequence[ObjectKey],
        region_base_ptrs: Sequence[int],
        region_block_strides: Sequence[int],
        region_offsets: Sequence[int],
        region_sizes: Sequence[int],
        gpu_block_ids: Sequence[int],
        cuda_stream: int = 0,
    ) -> list[HitInfo | None]:
        """Load cached blocks directly into paged multi-layer GPU KV tensors via batched DMA."""
        if len(keys) != len(gpu_block_ids):
            raise ValueError("keys and gpu_block_ids length mismatch")
        num_regions = len(region_base_ptrs)
        if (
            num_regions == 0
            or len(region_block_strides) != num_regions
            or len(region_offsets) != num_regions
            or len(region_sizes) != num_regions
        ):
            raise ValueError("region descriptor sequences must be non-empty and equal length")
        if not keys:
            return []
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        b_arr = _pack_size_array(gpu_block_ids, "gpu_block_ids")
        base_arr = _pack_size_array(region_base_ptrs, "region_base_ptrs")
        stride_arr = _pack_size_array(region_block_strides, "region_block_strides")
        off_arr = _pack_size_array(region_offsets, "region_offsets")
        size_arr = _pack_size_array(region_sizes, "region_sizes")
        out_arr = (VeriCacheHitInfoC * len(keys))()
        rc = self._lib.vericache_get_gpu_paged_batch(
            h,
            k_arr,
            b_arr,
            len(keys),
            base_arr,
            stride_arr,
            off_arr,
            size_arr,
            num_regions,
            int(cuda_stream),
            out_arr,
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        results: list[HitInfo | None] = []
        for i in range(len(keys)):
            info = out_arr[i]
            if info.found:
                results.append(
                    HitInfo(
                        location=_HIT_LOC_BY_CODE[info.location],
                        version=VersionId(info.version_origin, info.version_seq),
                        peer_node_id=info.peer_node_id,
                    )
                )
            else:
                results.append(None)
        return results

    def get_gpu_paged_batch_flags(
        self,
        keys: Sequence[ObjectKey | bytes],
        region_base_ptrs: Sequence[int],
        region_block_strides: Sequence[int],
        region_offsets: Sequence[int],
        region_sizes: Sequence[int],
        gpu_block_ids: Sequence[int],
        cuda_stream: int = 0,
    ) -> bytes:
        """Fast zero-dataclass variant of `get_gpu_paged_batch` returning 1 byte/key.

        Each byte is a `HitLocation` code (`0=MISS, 1=LOCAL_GPU, 2=LOCAL_DRAM,
        3=LOCAL_NVME, 4=REMOTE_RDMA_HOST, 5=REMOTE_GDR_GPU`).
        """
        if len(keys) != len(gpu_block_ids):
            raise ValueError("keys and gpu_block_ids length mismatch")
        num_regions = len(region_base_ptrs)
        if (
            num_regions == 0
            or len(region_block_strides) != num_regions
            or len(region_offsets) != num_regions
            or len(region_sizes) != num_regions
        ):
            raise ValueError("region descriptor sequences must be non-empty and equal length")
        if not keys:
            return b""
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        b_arr = _pack_size_array(gpu_block_ids, "gpu_block_ids")
        base_arr = _pack_size_array(region_base_ptrs, "region_base_ptrs")
        stride_arr = _pack_size_array(region_block_strides, "region_block_strides")
        off_arr = _pack_size_array(region_offsets, "region_offsets")
        size_arr = _pack_size_array(region_sizes, "region_sizes")
        out_buf = bytearray(len(keys))
        out_arr = (ctypes.c_uint8 * len(keys)).from_buffer(out_buf)
        rc = self._lib.vericache_get_gpu_paged_batch_flags(
            h,
            k_arr,
            b_arr,
            len(keys),
            base_arr,
            stride_arr,
            off_arr,
            size_arr,
            num_regions,
            int(cuda_stream),
            out_arr,
        )
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        return bytes(out_buf)

    def put_tensor(self, key: ObjectKey, tensor) -> VersionId:
        """Store a contiguous CPU or CUDA `torch.Tensor` of size `frame_size` bytes."""
        import torch

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes != self._frame_size:
            raise ValueError(f"Tensor size {nbytes} != frame_size {self._frame_size}")
        if tensor.is_cuda:
            stream = torch.cuda.current_stream(tensor.device).cuda_stream
            return self.put_gpu_ptr(key, tensor.data_ptr(), cuda_stream=stream)
        return self.put_host(key, tensor.view(torch.uint8).numpy().tobytes())

    def get_tensor(self, key: ObjectKey, dst_tensor) -> HitInfo | None:
        """Location-transparently load `key` into a contiguous CPU or CUDA `torch.Tensor`."""
        import torch

        if not dst_tensor.is_contiguous():
            raise ValueError("dst_tensor must be contiguous")
        nbytes = dst_tensor.numel() * dst_tensor.element_size()
        if nbytes != self._frame_size:
            raise ValueError(f"Tensor size {nbytes} != frame_size {self._frame_size}")
        if dst_tensor.is_cuda:
            stream = torch.cuda.current_stream(dst_tensor.device).cuda_stream
            return self.get_gpu_ptr(key, dst_tensor.data_ptr(), cuda_stream=stream)
        res = self.get_host(key)
        if res is None:
            return None
        hit, raw = res
        dst_u8 = dst_tensor.view(torch.uint8).reshape(-1)
        dst_u8.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
        return hit

    # ------------------------------------------------------------------
    # Metadata, Tier Control & Progress
    # ------------------------------------------------------------------

    def contains_batch_local(self, keys: Sequence[ObjectKey | bytes]) -> list[HitLocation]:
        if not keys:
            return []
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        out_arr = (ctypes.c_int32 * len(keys))()
        rc = self._lib.vericache_contains_batch_local(h, k_arr, len(keys), out_arr)
        if rc != 0:
            raise VeriCacheError(last_error(self._lib))
        return [_HIT_LOC_BY_CODE[out_arr[i]] for i in range(len(keys))]

    def contains_batch_local_flags(self, keys: Sequence[ObjectKey | bytes]) -> bytes:
        """Lock-free local tier membership check returning 1 byte/key (`0=miss, 1=GPU, 2=DRAM, 3=NVMe`)."""
        if not keys:
            return b""
        h = self._require_handle()
        k_arr = _pack_keys(keys)
        out_buf = bytearray(len(keys))
        out_arr = (ctypes.c_uint8 * len(keys)).from_buffer(out_buf)
        rc = self._lib.vericache_contains_batch_local_flags(h, k_arr, len(keys), out_arr)
        if rc != 0:
            raise VeriCacheError(last_error(self._lib))
        return bytes(out_buf)

    def take_evicted_keys(self, max_keys: int = 4096) -> list[bytes]:
        """Drain up to `max_keys` locally evicted keys as 32-byte `<4Q` raw byte buffers."""
        if max_keys <= 0:
            return []
        h = self._require_handle()
        buf = bytearray(max_keys * 32)
        c_buf = (ctypes.c_uint8 * (max_keys * 32)).from_buffer(buf)
        n = int(self._lib.vericache_take_evicted_keys(h, c_buf, max_keys))
        if n <= 0:
            return []
        mv = memoryview(buf)
        return [bytes(mv[i * 32 : (i + 1) * 32]) for i in range(n)]

    def spill_to_nvme(self, key: ObjectKey) -> bool:
        h = self._require_handle()
        rc = self._lib.vericache_spill_to_nvme(h, _pack_key(key))
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        return rc != 0

    def evict_from_tier(self, key: ObjectKey, tier: TierKind) -> bool:
        h = self._require_handle()
        rc = self._lib.vericache_evict_tier(h, _pack_key(key), int(tier))
        if rc < 0:
            raise VeriCacheError(last_error(self._lib))
        return rc != 0

    def poll(self) -> None:
        h = self._require_handle()
        rc = self._lib.vericache_poll(h)
        if rc != 0:
            raise VeriCacheError(last_error(self._lib))

    def drain(self) -> None:
        """Wait for the worker's submitted background I/O."""
        if self._lib.vericache_drain(self._require_handle()) != 0:
            raise VeriCacheError(last_error(self._lib))

    def stats(self) -> CacheStats:
        h = self._require_handle()
        snapshot = VeriCacheStatsC()
        rc = self._lib.vericache_path_stats(h, ctypes.byref(snapshot))
        if rc != 0:
            raise VeriCacheError(last_error(self._lib))
        return CacheStats(**{name: getattr(snapshot, name) for name, _ in snapshot._fields_})


__all__ = [
    "CacheStats",
    "HitInfo",
    "HitLocation",
    "ObjectKey",
    "PeerSpec",
    "TierKind",
    "VeriCacheClient",
    "VeriCacheError",
    "VersionId",
    "key_from_bytes",
    "key_from_u64",
]
