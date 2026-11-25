"""Low-level ctypes bindings for `libvericache_python.so`."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


class VeriCachePeerSpecC(ctypes.Structure):
    _fields_ = [
        ("peer_node_id", ctypes.c_uint64),
        ("tcp_addr", ctypes.c_char_p),
        ("is_listener", ctypes.c_uint8),
        ("psn_hint", ctypes.c_uint32),
    ]


class VeriCacheConfigC(ctypes.Structure):
    _fields_ = [
        ("node_id", ctypes.c_uint64),
        ("frame_size", ctypes.c_size_t),
        ("dram_frames", ctypes.c_uint32),
        ("gpu_device_id", ctypes.c_int32),
        ("hbm_frames", ctypes.c_uint32),
        ("nvme_dir", ctypes.c_char_p),
        ("nvme_frames", ctypes.c_uint32),
        ("rdma_device", ctypes.c_char_p),
        ("rdma_gid_index", ctypes.c_uint8),
        ("rdma_windows_per_peer", ctypes.c_size_t),
        ("peers", ctypes.POINTER(VeriCachePeerSpecC)),
        ("num_peers", ctypes.c_size_t),
    ]


class VeriCacheHitInfoC(ctypes.Structure):
    _fields_ = [
        ("found", ctypes.c_uint8),
        ("location", ctypes.c_int32),
        ("peer_node_id", ctypes.c_uint64),
        ("version_origin", ctypes.c_uint16),
        ("version_seq", ctypes.c_uint64),
    ]


class VeriCacheStatsC(ctypes.Structure):
    _fields_ = [
        ("puts", ctypes.c_uint64),
        ("gpu_hits", ctypes.c_uint64),
        ("dram_hits", ctypes.c_uint64),
        ("nvme_hits", ctypes.c_uint64),
        ("remote_hits", ctypes.c_uint64),
        ("misses", ctypes.c_uint64),
        ("policy_victims_rejected", ctypes.c_uint64),
        ("policy_victims_pinned", ctypes.c_uint64),
        ("remote_nvme_warm_hits", ctypes.c_uint64),
        ("donor_nvme_warms", ctypes.c_uint64),
        ("put_attempts", ctypes.c_uint64),
        ("nvme_spills", ctypes.c_uint64),
        ("nvme_evictions", ctypes.c_uint64),
        ("dram_evictions", ctypes.c_uint64),
        ("dram_reclaims", ctypes.c_uint64),
        ("mw_coop_revokes", ctypes.c_uint64),
        ("mw_unilateral_revokes", ctypes.c_uint64),
        ("host_pinned", ctypes.c_uint8),
        ("hbm_frames", ctypes.c_uint32),
        ("dram_frames", ctypes.c_uint32),
        ("nvme_frames", ctypes.c_uint32),
        ("gdr_enabled", ctypes.c_uint8),
    ]


_LIB: ctypes.CDLL | None = None


def _find_library() -> Path:
    env_path = os.environ.get("VERICACHE_PYTHON_LIB")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VERICACHE_PYTHON_LIB={env_path} does not exist")

    pkg_dir = Path(__file__).resolve().parent
    crate_root = pkg_dir.parent.parent
    candidates = [
        pkg_dir / "libvericache_python.so",
        crate_root / "target" / "release" / "libvericache_python.so",
        crate_root / "target" / "debug" / "libvericache_python.so",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find libvericache_python.so (searched: {searched}). "
        "Build it with `cargo build --manifest-path vericache-python/Cargo.toml --release`."
    )


def load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    lib = ctypes.CDLL(str(_find_library()))

    lib.vericache_last_error.argtypes = []
    lib.vericache_last_error.restype = ctypes.c_char_p

    lib.vericache_create.argtypes = [ctypes.POINTER(VeriCacheConfigC)]
    lib.vericache_create.restype = ctypes.c_void_p

    lib.vericache_destroy.argtypes = [ctypes.c_void_p]
    lib.vericache_destroy.restype = None

    lib.vericache_put_host.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vericache_put_host.restype = ctypes.c_int32

    lib.vericache_get_host.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(VeriCacheHitInfoC),
    ]
    lib.vericache_get_host.restype = ctypes.c_int32

    lib.vericache_put_gpu_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vericache_put_gpu_ptr.restype = ctypes.c_int32

    lib.vericache_get_gpu_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(VeriCacheHitInfoC),
    ]
    lib.vericache_get_gpu_ptr.restype = ctypes.c_int32

    lib.vericache_put_gpu_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.vericache_put_gpu_batch.restype = ctypes.c_int32

    lib.vericache_put_gpu_dram_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.vericache_put_gpu_dram_batch.restype = ctypes.c_int32

    lib.vericache_put_gpu_paged_dram_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.vericache_put_gpu_paged_dram_batch.restype = ctypes.c_int32

    lib.vericache_get_gpu_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(VeriCacheHitInfoC),
    ]
    lib.vericache_get_gpu_batch.restype = ctypes.c_int32

    lib.vericache_get_gpu_paged_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(VeriCacheHitInfoC),
    ]
    lib.vericache_get_gpu_paged_batch.restype = ctypes.c_int32

    lib.vericache_get_gpu_paged_batch_flags.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.vericache_get_gpu_paged_batch_flags.restype = ctypes.c_int32

    lib.vericache_contains_batch_local.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.vericache_contains_batch_local.restype = ctypes.c_int32

    lib.vericache_contains_batch_local_flags.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.vericache_contains_batch_local_flags.restype = ctypes.c_int32

    lib.vericache_take_evicted_keys.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.vericache_take_evicted_keys.restype = ctypes.c_size_t

    lib.vericache_spill_to_nvme.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vericache_spill_to_nvme.restype = ctypes.c_int32

    lib.vericache_evict_tier.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32,
    ]
    lib.vericache_evict_tier.restype = ctypes.c_int32

    lib.vericache_poll.argtypes = [ctypes.c_void_p]
    lib.vericache_poll.restype = ctypes.c_int32
    lib.vericache_drain.argtypes = [ctypes.c_void_p]
    lib.vericache_drain.restype = ctypes.c_int32

    lib.vericache_path_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(VeriCacheStatsC),
    ]
    lib.vericache_path_stats.restype = ctypes.c_int32

    _LIB = lib
    return lib


def last_error(lib: ctypes.CDLL) -> str:
    raw = lib.vericache_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "unknown error"
