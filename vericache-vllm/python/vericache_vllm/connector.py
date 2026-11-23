"""vLLM 0.13.0 connector for layered KV prefix reuse."""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import json
import logging
from os import getpid
from queue import Empty, LifoQueue
from threading import Lock
from weakref import WeakValueDictionary

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from ._native import _BUSY, NativeVeriCacheEngine, NativeVeriCacheError, copy_tensor
from .config import VeriCacheConfig
from .hashing import compute_prefix_block_hashes, derive_layer_block_key
from .ipc import MetadataClient, MetadataServer

logger = logging.getLogger(__name__)
_ENGINES = WeakValueDictionary()
_ENGINES_LOCK = Lock()
_REGISTRATION_LOCK = Lock()


class _SharedEngine:
    def __init__(self, config):
        self.native = NativeVeriCacheEngine(config)
        self.promotion_lock = Lock()
        self.layers = ()


def _shared_engine(engine_id, config):
    key = (getpid(), engine_id)
    with _ENGINES_LOCK:
        store = _ENGINES.get(key)
        if store is None:
            store = _SharedEngine(config)
            _ENGINES[key] = store
        elif store.native.config != config:
            raise ValueError("VeriCache scheduler and worker settings must agree")
        return store


@dataclass(frozen=True)
class BlockTransfer:
    block_hash: int
    block_id: int


@dataclass
class VeriCacheConnectorMetadata(KVConnectorMetadata):
    load: list[BlockTransfer] = field(default_factory=list)
    save: list[BlockTransfer] = field(default_factory=list)


@dataclass
class _RequestState:
    hashes: list[int]
    block_ids: list[int]
    external_tokens: int = 0


def _block_ids(blocks):
    if hasattr(blocks, "get_block_ids"):
        blocks = blocks.get_block_ids()
    if not blocks:
        return []
    if isinstance(blocks[0], int):
        return list(blocks)
    if len(blocks) != 1:
        raise ValueError("VeriCache requires a single KV cache group")
    return list(blocks[0])


@dataclass(frozen=True)
class _BlockLayout:
    axis: int
    kernel_blocks: int

    def view(self, cache, block_id):
        if self.kernel_blocks == 1:
            return cache.select(self.axis, block_id)
        return cache.narrow(
            self.axis, block_id * self.kernel_blocks, self.kernel_blocks
        )


class _StagingSlot:
    def __init__(self, buffer):
        self.buffer = buffer
        self._tensor = None

    def view_as(self, template):
        tensor = self._tensor
        if (
            tensor is None
            or tensor.dtype != template.dtype
            or tensor.shape != template.shape
        ):
            tensor = self.buffer.as_tensor(template.dtype, template.shape)
            self._tensor = tensor
        return tensor

    def close(self):
        self._tensor = None
        self.buffer.close()


@dataclass
class _LayerLoad:
    errors: set[int] = field(default_factory=set)
    missing: list[int] = field(default_factory=list)
    retry: list[tuple[BlockTransfer, int]] = field(default_factory=list)
    leases: list = field(default_factory=list)
    event: object = None
    _lock: object = field(default_factory=Lock, repr=False)

    def finish(self, engine, compute_stream=None):
        with self._lock:
            if self.event is not None:
                if compute_stream is not None:
                    compute_stream.wait_event(self.event)
                self.event.synchronize()
                self.event = None
            self._release(engine)

    def finish_if_ready(self, engine):
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self.event is None or self.event.query():
                self.event = None
                self._release(engine)
        finally:
            self._lock.release()

    def _release(self, engine):
        if self.leases:
            engine.release_read_lease_batch(lease.lease_id for lease in self.leases)
            self.leases.clear()


def _rank(config, world_size):
    if config.tp_rank is not None:
        rank = config.tp_rank
    elif world_size == 1:
        rank = 0
    else:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    if not 0 <= rank < world_size:
        raise ValueError("VeriCache TP rank is outside the configured world size")
    return rank


def _metadata_scheduler(endpoint):
    with MetadataClient(endpoint, timeout=0.15) as probe:
        try:
            probe.ping()
            return None
        except (FileNotFoundError, ConnectionRefusedError):
            try:
                return MetadataServer(endpoint).start()
            except OSError:
                # Another scheduler may have bound the endpoint during startup.
                probe.ping()
                return None


class VeriCacheV1Connector(KVConnectorBase_V1):
    """Share prefixes across a full-attention KV cache group and its TP ranks."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.config = VeriCacheConfig.from_vllm_config(vllm_config)
        self._block_size = vllm_config.cache_config.block_size
        if self._block_size <= 0:
            raise ValueError("vLLM cache block size must be positive")
        parallel = getattr(vllm_config, "parallel_config", None)
        self._world_size = self.config.world_size or getattr(parallel, "world_size", 1)
        backend = getattr(parallel, "distributed_executor_backend", None)
        endpoint = self.config.ipc_endpoint
        if not endpoint and (self._world_size != 1 or backend not in (None, "uni")):
            raise ValueError("VeriCache multi-process executors require ipc_endpoint")
        groups = getattr(kv_cache_config, "kv_cache_groups", ())
        self._spec = None
        self._configured_layers = ()
        if groups:
            if len(groups) != 1 or not isinstance(
                groups[0].kv_cache_spec, FullAttentionSpec
            ):
                raise ValueError("VeriCache requires one full-attention KV cache group")
            self._spec = groups[0].kv_cache_spec
            if self._spec.block_size != self._block_size:
                raise ValueError(
                    "VeriCache requires matching scheduler and KV cache block sizes"
                )
            self._configured_layers = tuple(groups[0].layer_names)
        self._num_blocks = getattr(kv_cache_config, "num_blocks", None)
        model = getattr(vllm_config, "model_config", None)
        self._model_tag = (
            self.config.model_tag,
            getattr(model, "model", ""),
            getattr(model, "revision", ""),
            str(getattr(model, "dtype", "")),
            getattr(vllm_config.cache_config, "cache_dtype", ""),
        )
        transfer = self._kv_transfer_config
        engine_id = getattr(transfer, "engine_id", None) or id(vllm_config)
        self._metadata_server = None
        self._metadata = None
        self._metadata_registered = False
        self._metadata_lock = Lock()
        self._tp_rank = None
        self._layers = ()
        worker = role == KVConnectorRole.WORKER
        if endpoint:
            self._metadata = MetadataClient(endpoint)
            self._model_tag += (self._world_size,)
            if worker:
                self._tp_rank = _rank(self.config, self._world_size)
                path = self.config.nvme_path
                worker_config = replace(
                    self.config,
                    tp_rank=self._tp_rank,
                    world_size=self._world_size,
                    model_tag=json.dumps((*self._model_tag, self._tp_rank)),
                    nvme_path=f"{path}.rank-{self._tp_rank}" if path else None,
                )
                self._store = _SharedEngine(worker_config)
            else:
                self._metadata_server = _metadata_scheduler(endpoint)
                self._store = None
        else:
            self._store = _shared_engine(engine_id, self.config)
        self._producer = getattr(transfer, "is_kv_producer", True)
        self._consumer = getattr(transfer, "is_kv_consumer", True)
        self._requests: dict[str, _RequestState] = {}
        self._allocations: dict[str, _RequestState] = {}
        self._caches = {}
        self._layouts = {}
        self._staging = LifoQueue(self.config.staging_pool_size)
        self._staging_count = 0
        self._streams = {}
        self._layer_loads = {}
        self._save_futures = []
        self._load_errors: set[int] = set()
        self._pool = (
            ThreadPoolExecutor(
                max_workers=self.config.io_workers, thread_name_prefix="vericache"
            )
            if worker and self.config.enable_async_pipeline and self.config.io_workers
            else None
        )

    @property
    def engine(self):
        return self._store.native if self._store is not None else None

    def _layer_key(self, block_hash, layer):
        if self._tp_rank is not None:
            layer = f"{self._tp_rank}:{self._world_size}:{layer}"
        return derive_layer_block_key(block_hash, layer)

    def _hashes(self, request):
        # These inputs can change the computed KV independently of the token IDs.
        if (
            getattr(request, "mm_features", None)
            or getattr(request, "lora_request", None)
            or getattr(request, "prompt_embeds", None) is not None
        ):
            return []
        tokens = getattr(request, "prompt_token_ids", None) or []
        tag = json.dumps((*self._model_tag, getattr(request, "cache_salt", None)))
        return compute_prefix_block_hashes(tokens, self._block_size, tag)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # A final prompt token has to run through the model to produce logits.
        layers = self._configured_layers or (self._store.layers if self._store else ())
        if not self._consumer or num_computed_tokens % self._block_size:
            return 0, False
        tokens = getattr(request, "prompt_token_ids", None) or []
        stop = max(0, len(tokens) - 1) // self._block_size
        start = num_computed_tokens // self._block_size
        hashes = self._hashes(request)[start:stop]
        if not hashes:
            return 0, False
        hits = 0
        try:
            if self._metadata is not None:
                hits = self._metadata.query_prefix_hits(
                    hashes, self._world_size, layers
                )
            elif layers:
                present = self.engine.contains_batch(
                    self._layer_key(block, layer)
                    for block in hashes
                    for layer in layers
                )
                for start in range(0, len(present), len(layers)):
                    if not all(present[start : start + len(layers)]):
                        break
                    hits += 1
        except (NativeVeriCacheError, OSError, EOFError, ValueError):
            logger.warning("VeriCache prefix lookup failed", exc_info=True)
        return hits * self._block_size, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._allocations[request.request_id] = _RequestState(
            self._hashes(request),
            _block_ids(blocks),
            num_external_tokens,
        )

    def _plan(self, meta, state, computed, scheduled):
        block_size = self._block_size
        end_at = min(len(state.hashes), len(state.block_ids))
        if state.external_tokens:
            first = (computed - state.external_tokens) // block_size
            last = computed // block_size
            if (
                first < 0
                or last > end_at
                or computed % block_size
                or state.external_tokens % block_size
            ):
                raise ValueError("invalid external KV block allocation")
            meta.load.extend(
                BlockTransfer(state.hashes[i], state.block_ids[i])
                for i in range(first, last)
            )
            state.external_tokens = 0
        if self._producer:
            complete = min(end_at, (computed + scheduled) // block_size)
            meta.save.extend(
                BlockTransfer(state.hashes[i], state.block_ids[i])
                for i in range(computed // block_size, complete)
            )

    def build_connector_meta(self, scheduler_output):
        meta = VeriCacheConnectorMetadata()
        scheduled = scheduler_output.num_scheduled_tokens
        for req_id in getattr(scheduler_output, "finished_req_ids", ()):
            self._requests.pop(req_id, None)
            self._allocations.pop(req_id, None)
        for req in scheduler_output.scheduled_new_reqs:
            state = self._allocations.pop(req.req_id, None)
            ids = _block_ids(req.block_ids)
            if state is None:
                state = _RequestState(self._hashes(req), ids)
            else:
                state.block_ids = ids or state.block_ids
            self._requests[req.req_id] = state
            self._plan(
                meta, state, req.num_computed_tokens, scheduled.get(req.req_id, 0)
            )

        cached = scheduler_output.scheduled_cached_reqs
        if cached is not None:
            for i, req_id in enumerate(cached.req_ids):
                new_ids = _block_ids(cached.new_block_ids[i])
                if req_id in cached.resumed_req_ids:
                    state = self._allocations.pop(req_id, None) or self._requests.get(
                        req_id
                    )
                    if state is None:
                        continue
                    if new_ids:
                        state.block_ids = new_ids
                    self._requests[req_id] = state
                else:
                    state = self._requests.get(req_id)
                    if state is None:
                        continue
                    state.block_ids.extend(new_ids)
                self._plan(
                    meta, state, cached.num_computed_tokens[i], scheduled.get(req_id, 0)
                )
        self._allocations.clear()
        return meta

    def request_finished(self, request, block_ids):
        self._requests.pop(request.request_id, None)
        self._allocations.pop(request.request_id, None)
        return False, None

    def _infer_layout(self, layer_name, cache):
        num_blocks = self._num_blocks
        axis = self.config.block_axis
        if axis is None:
            compilation = getattr(self._vllm_config, "compilation_config", None)
            layer = getattr(compilation, "static_forward_context", {}).get(layer_name)
            if layer is not None and self._spec is not None:
                spec = self._spec
                shape = layer.get_attn_backend().get_kv_cache_shape
                args = (spec.block_size, spec.num_kv_heads, spec.head_size)
                cache_dtype = getattr(
                    self._vllm_config.cache_config, "cache_dtype", "auto"
                )
                small = shape(7, *args, cache_dtype_str=cache_dtype)
                large = shape(11, *args, cache_dtype_str=cache_dtype)
                (axis,) = (i for i, (a, b) in enumerate(zip(small, large)) if a != b)
            elif num_blocks:
                candidates = [
                    i for i, size in enumerate(cache.shape) if size == num_blocks
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"set block_axis for ambiguous layout in {layer_name}"
                    )
                axis = candidates[0]
            else:
                axis = 1 if cache.ndim == 5 and cache.shape[0] == 2 else 0
        if not -cache.ndim <= axis < cache.ndim:
            raise ValueError(f"invalid block_axis for {layer_name}")
        axis %= cache.ndim
        count = cache.shape[axis]
        if count == 0 or (num_blocks and count % num_blocks):
            raise ValueError(f"invalid paged KV cache shape for {layer_name}")
        return _BlockLayout(axis, count // num_blocks if num_blocks else 1)

    def register_kv_caches(self, kv_caches):
        if self._configured_layers and not set(self._configured_layers).issubset(
            kv_caches
        ):
            raise ValueError("worker is missing a configured VeriCache layer")
        layouts = {}
        layers = self._configured_layers or tuple(kv_caches)
        for name in layers:
            cache = kv_caches[name]
            layout = self._infer_layout(name, cache)
            view = layout.view(cache, 0)
            length = view.numel() * view.element_size()
            if not 0 < length <= self.config.frame_size:
                raise ValueError(
                    f"{name} has a {length}-byte KV block, exceeding frame_size"
                )
            layouts[name] = layout
        self._caches = {name: kv_caches[name] for name in layers}
        self._layouts = layouts
        self._store.layers = self._layers = layers
        if self._metadata is not None:
            with self._metadata_lock:
                try:
                    # vLLM registers worker caches before constructing its
                    # scheduler. Reconnect when there is a batch to publish.
                    self._metadata.register_worker(self._tp_rank, layers, wait=False)
                    self._metadata_registered = True
                except OSError:
                    self._metadata_registered = False

    def _submit(self, fn, *args):
        if self._pool is not None:
            return self._pool.submit(fn, *args)
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as error:
            future.set_exception(error)
        return future

    def _get_streams(self, tensor):
        if not tensor.is_cuda:
            return None, None
        if tensor.device not in self._streams:
            import torch

            self._streams[tensor.device] = (
                torch.cuda.Stream(device=tensor.device),
                torch.cuda.Stream(device=tensor.device),
            )
        return self._streams[tensor.device]

    def start_load_kv(self, forward_context, **kwargs):
        if self._layer_loads or self._save_futures:
            self.wait_for_save()
        self._load_errors.clear()
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not metadata.load:
            return
        if not self._caches:
            virtual_engine = getattr(forward_context, "virtual_engine", 0)
            self.register_kv_caches(
                {
                    name: layer.kv_cache[virtual_engine]
                    for name, layer in forward_context.no_compile_layers.items()
                    if getattr(layer, "kv_cache", None) is not None
                }
            )
        if not self._caches:
            self._load_errors.update(block.block_id for block in metadata.load)
            return
        prepared_streams = set()
        for layer, cache in self._caches.items():
            stream, _ = self._get_streams(cache)
            if stream is not None and cache.device not in prepared_streams:
                import torch

                stream.wait_stream(torch.cuda.current_stream(cache.device))
                prepared_streams.add(cache.device)
            items = [
                (block, self._layer_key(block.block_hash, layer))
                for block in metadata.load
            ]
            self._layer_loads[layer] = self._submit(
                self._load_layer, layer, items, stream
            )

    def _load_layer(self, layer, items, stream, *, defer_busy=True):
        cache, layout = self._caches[layer], self._layouts[layer]
        result = _LayerLoad()
        try:
            if self.config.use_zero_copy_load:
                if self.config.nvme_frames_per_shard and self.engine.has_cold_data():
                    # A cold batch can use every promotion frame. Release its
                    # leases after the copy so the next layer can reuse them.
                    with self._store.promotion_lock:
                        try:
                            self._load_leases(
                                cache, layout, items, stream, result, defer_busy
                            )
                        finally:
                            result.finish(self.engine)
                else:
                    self._load_leases(cache, layout, items, stream, result, defer_busy)
            else:
                self._load_staged(cache, layout, items, stream, result)
        except (NativeVeriCacheError, ValueError, RuntimeError):
            logger.warning("VeriCache could not load layer %s", layer, exc_info=True)
            result.errors.update(block.block_id for block, _ in items)
        return result

    def _load_leases(self, cache, layout, items, stream, result, defer_busy):
        import torch

        pending = items
        context = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with context, torch.inference_mode():
            while pending:
                leases, statuses = self.engine.acquire_read_lease_batch(
                    (key for _, key in pending),
                    with_statuses=True,
                )
                result.leases = [lease for lease in leases if lease is not None]
                busy = []
                destinations, sources = [], []
                try:
                    for (block, key), lease, status in zip(pending, leases, statuses):
                        if status == _BUSY:
                            busy.append((block, key))
                        elif status:
                            result.errors.add(block.block_id)
                            if status == 1:
                                result.missing.append(key)
                        else:
                            view = layout.view(cache, block.block_id)
                            if lease.length != view.numel() * view.element_size():
                                result.errors.add(block.block_id)
                                continue
                            source = lease.as_tensor(view.dtype, view.shape)
                            if stream is None:
                                destinations.append(view)
                                sources.append(source)
                            else:
                                copy_tensor(view, source, lease, record_event=False)
                            del source
                    if destinations:
                        torch._foreach_copy_(destinations, sources)
                finally:
                    sources.clear()
                    if stream is not None and result.leases:
                        result.event = stream.record_event()
                    # CPU copies have completed. Under pressure, finish this
                    # batch before retrying NVMe promotions that needed a frame.
                    if stream is None or busy:
                        result.finish(self.engine)
                if not busy:
                    break
                if len(busy) == len(pending):
                    if defer_busy:
                        result.retry = busy
                    else:
                        result.errors.update(block.block_id for block, _ in busy)
                    break
                pending = busy

    def _load_staged(self, cache, layout, items, stream, result):
        import torch

        context = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with self.engine.make_staging() as stage, context, torch.inference_mode():
            for block, key in items:
                view = layout.view(cache, block.block_id)
                size = self.engine.get_into_staging(key, stage)
                if size != view.numel() * view.element_size():
                    result.errors.add(block.block_id)
                    if size is None:
                        result.missing.append(key)
                    continue
                event = copy_tensor(
                    view, stage.as_tensor(view.dtype, view.shape), stage
                )
                if event is not None:
                    event.synchronize()

    def wait_for_layer_load(self, layer_name):
        future = self._layer_loads.get(layer_name)
        if future is None:
            return
        result = future.result()
        stream, _ = self._get_streams(self._caches[layer_name])
        compute = None
        if stream is not None:
            import torch

            compute = torch.cuda.current_stream(stream.device)
        result.finish(self.engine, compute)
        del self._layer_loads[layer_name]
        self._load_errors.update(result.errors)
        self._report_missing(result.missing)
        if result.retry:
            # Reap any already-complete DMA before retrying this layer.
            # This never waits for another layer's future or CUDA event.
            for pending in self._layer_loads.values():
                if pending.done():
                    pending.result().finish_if_ready(self.engine)
            retry = self._load_layer(layer_name, result.retry, stream, defer_busy=False)
            retry.finish(self.engine, compute)
            self._load_errors.update(retry.errors)
            self._report_missing(retry.missing)

    def _take_staging(self, *, wait):
        try:
            return self._staging.get_nowait()
        except Empty:
            if self._staging_count < self.config.staging_pool_size:
                stage = _StagingSlot(self.engine.make_staging())
                self._staging_count += 1
                return stage
            if wait:
                return self._staging.get()
            return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self._load_errors or not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not metadata.save:
            return
        layout = self._layouts.get(layer_name)
        if layout is None:
            if self._caches or (
                self._configured_layers and layer_name not in self._configured_layers
            ):
                return
            layout = self._infer_layout(layer_name, kv_layer)
        candidates = {
            self._layer_key(block.block_hash, layer_name): block
            for block in metadata.save
        }
        try:
            present = self.engine.contains_batch(candidates)
        except NativeVeriCacheError:
            logger.warning("VeriCache could not check save candidates", exc_info=True)
            return
        missing = [
            (key, block)
            for (key, block), hit in zip(candidates.items(), present)
            if not hit
        ]
        if not missing:
            return
        import torch

        _, stream = self._get_streams(kv_layer)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(kv_layer.device))
        load_futures = tuple(self._layer_loads.values())
        # Every view from this layout has the same shape and strides. vLLM
        # keeps the source blocks immutable until wait_for_save completes.
        if (
            stream is None
            and not (kv_layer.is_conj() or kv_layer.is_neg())
            and layout.view(kv_layer, missing[0][1].block_id).is_contiguous()
        ):
            with torch.inference_mode():
                batch = []
                for key, block in missing:
                    view = layout.view(kv_layer, block.block_id)
                    batch.append(
                        (
                            block.block_hash,
                            layer_name,
                            key,
                            view.data_ptr(),
                            view.numel() * view.element_size(),
                        )
                    )
            self._queue_save(
                self._submit(self._save_batch, batch, (), None, load_futures, kv_layer)
            )
            return
        offset = 0
        while offset < len(missing):
            stages = []
            batch = []
            try:
                context = (
                    torch.cuda.stream(stream) if stream is not None else nullcontext()
                )
                with context, torch.inference_mode():
                    destinations, sources = [], []
                    while offset < len(missing):
                        stage = self._take_staging(wait=not stages)
                        if stage is None:
                            break
                        stages.append(stage)
                        key, block = missing[offset]
                        offset += 1
                        view = layout.view(kv_layer, block.block_id)
                        target = stage.view_as(view)
                        if stream is None:
                            destinations.append(target)
                            sources.append(view)
                        else:
                            copy_tensor(target, view, stage.buffer, record_event=False)
                        batch.append(
                            (
                                block.block_hash,
                                layer_name,
                                key,
                                stage.buffer.ptr,
                                view.numel() * view.element_size(),
                            )
                        )
                    if destinations:
                        torch._foreach_copy_(destinations, sources)
                    event = stream.record_event() if stream is not None else None
                future = self._submit(
                    self._save_batch, batch, stages, event, load_futures
                )
            except BaseException:
                if stream is not None:
                    stream.synchronize()
                for stage in stages:
                    self._staging.put_nowait(stage)
                raise
            self._queue_save(future)

    def _queue_save(self, future):
        # Reap completed futures while a forward is producing new batches.
        self._save_futures = [
            f for f in self._save_futures if not self._finish_save_if_ready(f)
        ]
        self._save_futures.append(future)

    @staticmethod
    def _finish_save_if_ready(future):
        if future.done():
            future.result()
            return True
        return False

    def _save_batch(self, batch, stages, event, load_futures, _source_tensor=None):
        try:
            if event is not None:
                event.synchronize()
            for future in load_futures:
                load = future.result()
                if load.errors or load.retry:
                    return
            keys = [key for _, _, key, _, _ in batch]
            with self._metadata_lock if self._metadata is not None else nullcontext():
                statuses = self.engine.put_batch(
                    keys,
                    (ptr for _, _, _, ptr, _ in batch),
                    (length for _, _, _, _, length in batch),
                )
                if self._metadata is not None:
                    saved = [
                        (block, layer, key)
                        for (block, layer, key, _, _), status in zip(batch, statuses)
                        if not status
                    ]
                    self._publish_metadata(saved)
            for key, status in zip(keys, statuses):
                if status:
                    logger.warning(
                        "VeriCache could not save key %s (status %s)", key, status
                    )
        except (NativeVeriCacheError, OSError, EOFError, ValueError):
            logger.warning("VeriCache could not save KV batch", exc_info=True)
        finally:
            for stage in stages:
                self._staging.put_nowait(stage)

    def _publish_metadata(self, saved=(), missing=()):
        evicted = set(self.engine.drain_evictions())
        evicted.update(missing)
        if evicted:
            ordered = sorted(evicted)
            evicted = [
                key
                for key, present in zip(ordered, self.engine.contains_batch(ordered))
                if not present
            ]
        if saved:
            saved = [
                row
                for row, present in zip(
                    saved, self.engine.contains_batch(row[2] for row in saved)
                )
                if present
            ]
        if not saved and not evicted:
            return
        try:
            if not self._metadata_registered:
                self._metadata.register_worker(self._tp_rank, self._layers)
                self._metadata_registered = True
            self._metadata.mark_saved_batch(self._tp_rank, saved, evicted=evicted)
        except (OSError, EOFError, ValueError):
            self._metadata.close()
            self._metadata_registered = False
            raise

    def _report_missing(self, keys):
        if keys and self._metadata is not None:
            try:
                with self._metadata_lock:
                    self._publish_metadata(missing=keys)
            except (OSError, EOFError, ValueError):
                logger.warning(
                    "VeriCache could not publish missing KV keys", exc_info=True
                )

    def _drain_loads(self):
        for layer in tuple(self._layer_loads):
            self.wait_for_layer_load(layer)

    def wait_for_save(self):
        self._drain_loads()
        futures, self._save_futures = self._save_futures, []
        for future in futures:
            future.result()
        if self.engine is not None:
            try:
                with (
                    self._metadata_lock if self._metadata is not None else nullcontext()
                ):
                    self.engine.poll_writeback()
                    if self._metadata is not None:
                        self._publish_metadata()
            except (NativeVeriCacheError, OSError, EOFError, ValueError):
                logger.warning("VeriCache writeback progress failed", exc_info=True)

    def get_finished(self, finished_req_ids):
        return None, None

    def get_block_ids_with_load_errors(self):
        self._drain_loads()
        errors = self._load_errors
        self._load_errors = set()
        return errors

    def shutdown(self):
        try:
            self.wait_for_save()
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
            while not self._staging.empty():
                self._staging.get_nowait().close()
            self._staging_count = 0
            self._streams.clear()
            self._caches.clear()
            self._layouts.clear()
            self._requests.clear()
            self._allocations.clear()
            if self._metadata is not None:
                self._metadata.close()
                self._metadata = None
                if self._store is not None:
                    self._store.native.close()
            if self._metadata_server is not None:
                self._metadata_server.close()
                self._metadata_server = None
            self._store = None


def register_vericache_connector(name: str = "VeriCacheV1Connector") -> None:
    """Register VeriCache with vLLM under an available connector name."""
    with _REGISTRATION_LOCK:
        if name not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                name, __name__, VeriCacheV1Connector.__name__
            )
