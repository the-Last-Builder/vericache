"""VeriCache host staging and vLLM V1 integration."""

from typing import TYPE_CHECKING

from ._native import (
    NativeVeriCacheEngine,
    NativeVeriCacheError,
    PinnedStagingBuffer,
    ReadLease,
)
from .config import VeriCacheConfig
from .hashing import compute_prefix_block_hashes, derive_layer_block_key
from .ipc import MetadataClient, MetadataServer

if TYPE_CHECKING:
    from .connector import (
        VeriCacheConnectorMetadata,
        VeriCacheV1Connector,
        register_vericache_connector,
    )

__all__ = [
    "NativeVeriCacheEngine",
    "NativeVeriCacheError",
    "PinnedStagingBuffer",
    "ReadLease",
    "MetadataClient",
    "MetadataServer",
    "VeriCacheConfig",
    "VeriCacheV1Connector",
    "VeriCacheConnectorMetadata",
    "register_vericache_connector",
    "compute_prefix_block_hashes",
    "derive_layer_block_key",
]


def __getattr__(name: str):
    if name in (
        "VeriCacheConnectorMetadata",
        "VeriCacheV1Connector",
        "register_vericache_connector",
    ):
        from . import connector

        return getattr(connector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
