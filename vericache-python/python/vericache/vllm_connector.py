"""vLLM v1 `KVConnectorBase_V1` integration backed by VeriCache.

Implements the full vLLM v1 Scheduler + Worker KVConnector protocol:
- Scheduler side:
  - `get_num_new_matched_tokens(request, num_computed_tokens)`
  - `update_state_after_alloc(request, blocks, num_external_tokens)`
  - `build_connector_meta(scheduler_output)`
  - `request_finished(request, block_ids)`
- Worker side:
  - `register_kv_caches(kv_caches)`
  - `start_load_kv(forward_context)` / `wait_for_layer_load(layer_name)`
  - `save_kv_layer(layer_name, kv_layer, attn_metadata)` / `wait_for_save()`
  - `get_finished(finished_req_ids)`

When `vllm` is installed, `VeriCacheConnector` inherits directly from
`vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1`.
When running standalone PyTorch tests without `vllm`, a structurally identical
shim base class is used so the exact same connector code is testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from . import HitInfo, HitLocation, ObjectKey, VeriCacheClient
from .hashing import hash_prefix_blocks as hash_prefix_blocks
from .hashing import layer_block_key as layer_block_key

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
except ImportError:

    class KVConnectorRole(Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config: Any, role: KVConnectorRole) -> None:
            self._vllm_config = vllm_config
            self._role = role
            self._connector_metadata: Optional[KVConnectorMetadata] = None

        def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self) -> KVConnectorMetadata:
            assert self._connector_metadata is not None
            return self._connector_metadata


# ---------------------------------------------------------------------------
# Connector Metadata (Scheduler -> Worker)
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
# VeriCacheConnector (vLLM v1 KVConnectorBase_V1)
# ---------------------------------------------------------------------------


class VeriCacheConnector(KVConnectorBase_V1):
    """Location-transparent vLLM v1 KVConnector backed by `VeriCacheClient`.

    Data path (Route 1 — zero TCB expansion):
    - On `start_load_kv`: resolves each layer's requested blocks through
      `VeriCacheClient.get_gpu_batch` (transparently hitting Local HBM,
      Local Pinned DRAM, Local NVMe, or Remote 2-GPU GPUDirect RDMA) directly
      into a contiguous GPU staging buffer, then scatters into vLLM's paged
      `kv_caches[layer_name]` tensor in HBM on the active CUDA stream.
    - On `save_kv_layer`: gathers newly computed paged KV blocks from
      `kv_layer` into a contiguous GPU staging buffer on the active CUDA stream
      and publishes via `VeriCacheClient.put_gpu_batch` directly into HBM.
    """

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        *,
        client: VeriCacheClient,
        block_size: int = 16,
        probe_layer_name: str = "layer_0",
        kv_layout: str = "2_N_...",
        remote_lookup_fn: Optional[Callable[[Sequence[ObjectKey]], Sequence[bool]]] = None,
        on_save_keys_fn: Optional[Callable[[Sequence[ObjectKey]], None]] = None,
    ) -> None:
        if vllm_config is None:
            # Direct clients can use the connector without constructing vLLM.
            self._vllm_config = None
            self._kv_transfer_config = None
            self._kv_cache_config = None
            self._connector_metadata = None
            self._role = role
        else:
            super().__init__(vllm_config, role)
        self.client = client
        self.block_size = int(block_size)
        self.probe_layer_name = probe_layer_name
        self.kv_layout = kv_layout
        self.remote_lookup_fn = remote_lookup_fn
        self.on_save_keys_fn = on_save_keys_fn

        # Scheduler state
        self._matched_digests: dict[str, list[bytes]] = {}
        self._pending_loads: list[BlockTransferSpec] = []
        self._pending_saves: list[BlockTransferSpec] = []

        # Worker state
        self._kv_caches: dict[str, Any] = {}
        self._last_load_hits: dict[str, list[HitInfo | None]] = {}
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    # =======================================================================
    # Scheduler-side hooks
    # =======================================================================

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Return `(num_external_matched_tokens, load_async)` for `request`."""
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
        req_id = request.request_id
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None:
            token_ids = request.prompt_token_ids
        all_digests = hash_prefix_blocks(token_ids, self.block_size)

        if hasattr(blocks, "get_block_ids"):
            raw_ids = blocks.get_block_ids()
            block_ids = list(raw_ids[0]) if raw_ids and isinstance(raw_ids[0], (list, tuple)) else list(raw_ids)
        else:
            block_ids = list(blocks)

        num_load_blks = num_external_tokens // self.block_size
        matched = self._matched_digests.pop(req_id, None)
        if num_load_blks > 0:
            load_digests = matched[:num_load_blks] if matched is not None else all_digests[:num_load_blks]
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

    def build_connector_meta(self, scheduler_output: Any) -> VeriCacheConnectorMetadata:
        meta = VeriCacheConnectorMetadata(
            loads=self._pending_loads,
            saves=self._pending_saves,
        )
        self._pending_loads = []
        self._pending_saves = []
        return meta

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return False, None

    # =======================================================================
    # Worker-side hooks
    # =======================================================================

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register vLLM's per-layer paged KV cache GPU tensors."""
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
            # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            per_kv = 1
            for d in kv_layer.shape[2:]:
                per_kv *= int(d)
            return 2 * per_kv * int(kv_layer.element_size())
        # Shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        per_blk = 1
        for d in kv_layer.shape[1:]:
            per_blk *= int(d)
        return per_blk * int(kv_layer.element_size())

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Location-transparently load all scheduled prefix blocks into GPU paged KV tensors."""
        import torch

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
            stream = torch.cuda.current_stream(device).cuda_stream if kv_layer.is_cuda else 0
            keys = [layer_block_key(d, layer_name) for d in flat_digests]

            if kv_layer.is_cuda:
                staging = torch.empty((n_blocks, frame_size), dtype=torch.uint8, device=device)
                base_ptr = staging.data_ptr()
                dst_ptrs = [base_ptr + i * frame_size for i in range(n_blocks)]
                hits = self.client.get_gpu_batch(keys, dst_ptrs, cuda_stream=stream)
                self._last_load_hits[layer_name] = hits

                # Scatter loaded contiguous blocks into vLLM's paged KV tensor in HBM.
                if self.kv_layout == "2_N_...":
                    block_shape = (2,) + tuple(kv_layer.shape[2:])
                    typed = staging.view(kv_layer.dtype).reshape((n_blocks,) + block_shape)
                    idx_t = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
                    # typed: [n_blocks, 2, ...] -> transpose to [2, n_blocks, ...]
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
                    src = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(kv_layer.dtype)
                    if self.kv_layout == "2_N_...":
                        kv_layer[:, blk_id].copy_(src.reshape((2,) + tuple(kv_layer.shape[2:])))
                    else:
                        kv_layer[blk_id].copy_(src.reshape(tuple(kv_layer.shape[1:])))
                self._last_load_hits[layer_name] = hits

        self._finished_recving.update(req_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # `start_load_kv` already synchronized HBM D2D on the current CUDA stream.
        return None

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """Gather newly computed paged KV blocks from `kv_layer` and publish into VeriCache."""
        import torch

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
                # kv_layer[:, idx_t]: [2, n_blocks, ...] -> [n_blocks, 2, ...]
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
        return None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        done_send = self._finished_sending.intersection(finished_req_ids)
        done_recv = self._finished_recving.intersection(finished_req_ids)
        self._finished_sending.difference_update(done_send)
        self._finished_recving.difference_update(done_recv)
        return (done_send or None), (done_recv or None)

    @property
    def last_load_hits(self) -> dict[str, list[HitInfo | None]]:
        """Diagnostic hit locations from the most recent `start_load_kv`."""
        return self._last_load_hits


__all__ = [
    "BlockTransferSpec",
    "KVConnectorBase_V1",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "VeriCacheConnector",
    "VeriCacheConnectorMetadata",
    "hash_prefix_blocks",
    "layer_block_key",
]
