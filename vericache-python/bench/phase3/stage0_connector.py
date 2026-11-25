"""Stage 0 measurements around the production VeriCache vLLM connector."""

import json
import os
from pathlib import Path

from vericache.vllm_v1_connector import VeriCacheV1Connector as ProductionConnector


class VeriCacheV1Connector(ProductionConnector):
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        layers = self._vllm_config.compilation_config.static_forward_context
        geometry = []
        for name, tensor in self._caches.items():
            layout = self._layouts[name]
            block = layout.view(tensor, 0)
            size = block.numel() * block.element_size()
            assert size == self.config.frame_size, (name, size, self.config.frame_size)
            geometry.append({
                "name": name,
                "backend": layers[name].get_attn_backend().get_name(),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "block_axis": layout.axis,
                "kernel_blocks_per_block": layout.kernel_blocks,
                "bytes_per_layer_per_block": size,
            })
        assert len(geometry) == 48
        assert self.engine.is_slab_cuda_pinned is True
        path = Path(os.environ["PHASE3_RESULT_DIR"]) / "kv-layout.json"
        path.write_text(json.dumps({
            "kv_cache_groups": 1,
            "kv_cache_spec": type(self._spec).__name__,
            "vllm_scheduler_block_size": self._block_size,
            "hbm_kv_blocks": self._num_blocks,
            "is_slab_cuda_pinned": self.engine.is_slab_cuda_pinned,
            "layers": geometry,
        }, indent=2) + "\n")

    def start_load_kv(self, forward_context, **kwargs):
        self._stage0_load_ids = ()
        if self.has_connector_metadata():
            loads = self._get_connector_metadata().load
            if loads:
                self._stage0_load_ids = tuple(block.block_id for block in loads)
                self._stage0_before = self.engine.extended_stats()
                flag = Path(os.environ["PHASE3_TEST_DIR"]) / "inject-late-eviction"
                if flag.is_file():
                    flag.unlink()
                    block, layer = loads[0], self._layers[0]
                    assert self.engine.evict(self._layer_key(block.block_hash, layer))
                    self._stage0_evicted = block.block_id
        super().start_load_kv(forward_context, **kwargs)

    def get_block_ids_with_load_errors(self):
        errors = super().get_block_ids_with_load_errors()
        if getattr(self, "_stage0_load_ids", ()):
            evicted = getattr(self, "_stage0_evicted", None)
            after = self.engine.extended_stats()
            event = {
                "load_ids": self._stage0_load_ids,
                "late_evicted_id": evicted,
                "load_errors": sorted(errors),
                "stats_delta": {key: value - self._stage0_before[key]
                                for key, value in after.items()},
            }
            path = Path(os.environ["PHASE3_RESULT_DIR"]) / "connector-loads.jsonl"
            with path.open("a") as output:
                output.write(json.dumps(event) + "\n")
            self._stage0_load_ids = ()
            self._stage0_evicted = None
        return errors
