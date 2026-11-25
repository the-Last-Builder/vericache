"""Sample local VeriCache counters during the Phase 3 serving benchmark."""

import json
import os
from pathlib import Path
from threading import Event, Thread

from vericache.vllm_v1_connector import VeriCacheV1Connector


class MeasuredVeriCacheV1Connector(VeriCacheV1Connector):
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        assert self.engine.is_slab_cuda_pinned is True
        assert self._num_blocks == 3328
        assert len(self._layers) == 48
        assert all(
            layout.view(self._caches[name], 0).numel() * self._caches[name].element_size()
            == self.config.frame_size
            for name, layout in self._layouts.items()
        )
        if not hasattr(self, "_stop_sampling"):
            policy = os.environ.get("PHASE3_NVME_POLICY")
            if policy:
                depth, probation, watermark = json.loads(policy)
                self.engine.set_nvme_policy(depth, probation)
                self.engine.set_writeback_policy(watermark)
            self._stop_sampling = Event()
            self._sampler = Thread(target=self._record_stats, daemon=True)
            self._sampler.start()

    def _record_stats(self):
        path = Path(os.environ["PHASE3_TEST_DIR"]) / f"vericache-stats-{os.getpid()}.json"
        next_sample = path.with_suffix(".next")
        while not self._stop_sampling.is_set():
            counters = {
                "base": self.engine.stats(),
                "extended": self.engine.extended_stats(),
            }
            if self.config.nvme_frames_per_shard:
                counters["nvme_io"] = self.engine.nvme_io_stats()
            next_sample.write_text(json.dumps(counters))
            next_sample.replace(path)
            self._stop_sampling.wait(0.5)

    def shutdown(self):
        if hasattr(self, "_stop_sampling"):
            self._stop_sampling.set()
            self._sampler.join(timeout=1)
        super().shutdown()
