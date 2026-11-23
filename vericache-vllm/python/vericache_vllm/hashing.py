"""Content keys for complete blocks and their per-layer payloads."""

from collections.abc import Sequence
from hashlib import blake2b
from struct import pack


def _hash64(data: bytes, domain: bytes) -> int:
    return (
        int.from_bytes(blake2b(data, digest_size=8, person=domain).digest(), "little")
        or 1
    )


def compute_prefix_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    model_tag: str = "",
) -> list[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    prefix = _hash64(pack("<Q", block_size) + model_tag.encode(), b"vc-prefix-root")
    hashes = []
    for offset in range(0, len(token_ids) - block_size + 1, block_size):
        prefix = _hash64(
            pack(
                f"<{block_size + 1}Q", prefix, *token_ids[offset : offset + block_size]
            ),
            b"vc-prefix-block",
        )
        hashes.append(prefix)
    return hashes


def derive_layer_block_key(block_hash: int, layer_name: str) -> int:
    return _hash64(pack("<Q", block_hash) + layer_name.encode(), b"vc-layer-block")
