"""Deterministic 256-bit content keys for KV blocks and per-layer payloads."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ObjectKey


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
        h.update(
            struct.pack(f"<{block_size}Q", *[t & 0xFFFFFFFFFFFFFFFF for t in chunk])
        )
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


def compute_prefix_block_hashes(
    token_ids: Sequence[int], block_size: int, model_tag: str = ""
) -> list[bytes]:
    """Hash complete blocks in order, seeded by the block size and model tag."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    parent = hashlib.sha256(
        b"vericache-prefix-v1\0"
        + struct.pack("<Q", block_size)
        + model_tag.encode("utf-8")
    ).digest()
    block_words = struct.Struct(f"<{block_size}Q")
    digests = []
    for offset in range(0, len(token_ids) - block_size + 1, block_size):
        tokens = token_ids[offset : offset + block_size]
        payload = block_words.pack(*(token & 0xFFFFFFFFFFFFFFFF for token in tokens))
        parent = hashlib.sha256(parent + payload).digest()
        digests.append(parent)
    return digests


def derive_layer_block_key(block_hash: bytes, layer_name: str) -> ObjectKey:
    """Derive four little-endian uint64 words from a SHA-256 prefix digest."""
    if not isinstance(block_hash, bytes) or len(block_hash) != 32:
        raise ValueError("block_hash must be a 32-byte digest")
    return layer_block_key(block_hash, layer_name)
