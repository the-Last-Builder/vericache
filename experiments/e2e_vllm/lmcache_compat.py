"""Thin compatibility shim bridging LMCache 0.5.7.post1 to vLLM 0.34.0.

Bridges two concrete vLLM 0.34.0 API mismatches while keeping 100% of LMCache's
engine, storage backends, and transfer logic unmodified:
1. `CachedRequestData.resumed_req_ids`: removed in vLLM 0.34.0 (preempted
   requests are re-emitted via `scheduled_new_reqs`), whereas LMCache 0.5.7.post1
   expects `cached_reqs.resumed_req_ids`.
2. FULL CUDA Graph `start_load_kv(..., attn_metadata=...)`: in vLLM 0.34.0,
   FULL graph replay creates a `ForwardContext` with `attn_metadata=None` and
   passes the real `attn_metadata` via `kwargs["attn_metadata"]`, whereas
   LMCache 0.5.7.post1's `LMCacheConnectorV1Impl.start_load_kv` only checks
   `forward_context.attn_metadata` and silently returns without loading when
   it is `None`.
"""

from __future__ import annotations

from typing import Any

from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic


class LMCacheConnectorV1Compat(LMCacheConnectorV1Dynamic):
    def build_connector_meta(self, scheduler_output: Any) -> Any:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is not None and not hasattr(cached_reqs, "resumed_req_ids"):
            try:
                setattr(cached_reqs, "resumed_req_ids", set())
            except Exception:
                pass
        return super().build_connector_meta(scheduler_output)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        kw_attn = kwargs.get("attn_metadata")
        patched = False
        if (
            forward_context is not None
            and getattr(forward_context, "attn_metadata", None) is None
            and kw_attn is not None
        ):
            forward_context.attn_metadata = kw_attn
            patched = True
        try:
            super().start_load_kv(forward_context, **kwargs)
        finally:
            if patched:
                forward_context.attn_metadata = None
