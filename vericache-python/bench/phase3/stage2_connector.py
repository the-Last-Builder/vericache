"""Stage 2 measured + controllable distributed VeriCache vLLM v1 connector."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import socketserver
from threading import Event, Thread

from vericache.vllm_v1_connector import VeriCacheV1Connector


class _CtrlHandler(socketserver.StreamRequestHandler):
    def handle(self):
        conn = self.server.connector
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            req = json.loads(raw.decode("utf-8"))
            op = req.get("op")
            if op == "stats":
                resp = {"ok": True, "stats": conn.engine.rdma_stats()}
            elif op == "set_dram_budget":
                frames = int(req["frames_per_shard"])
                evicted = conn.engine.set_dram_budget_frames(frames)
                conn._publish_global_dir()
                resp = {"ok": True, "evicted": evicted, "stats": conn.engine.rdma_stats()}
            elif op == "set_spin_budget":
                spins = int(req["spins"])
                conn.engine.set_coop_revoke_spin_budget(spins)
                resp = {"ok": True, "spins": spins}
            elif op == "set_peer_fetch":
                enabled = bool(req["enabled"])
                conn.engine.set_peer_fetch_enabled(enabled)
                resp = {"ok": True, "enabled": enabled}
            elif op == "poll":
                conn.engine.poll_writeback()
                conn._publish_global_dir()
                resp = {"ok": True, "stats": conn.engine.rdma_stats()}
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:
            resp = {"ok": False, "error": repr(exc)}
        self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))


class _CtrlServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, connector):
        self.connector = connector
        super().__init__(path, _CtrlHandler)


class Stage2VeriCacheV1Connector(VeriCacheV1Connector):
    """Instrumented distributed VeriCache connector for Stage 2 cross-NUMA RDMA."""

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        if not hasattr(self, "_stop_sampling"):
            self._stop_sampling = Event()
            self._sampler = Thread(target=self._record_stats, daemon=True)
            self._sampler.start()
            ctrl_sock = os.environ.get("PHASE3_STAGE2_CTRL_SOCK")
            self._ctrl_server = None
            self._ctrl_thread = None
            self._ctrl_path = None
            if ctrl_sock:
                p = Path(ctrl_sock)
                if p.exists():
                    p.unlink()
                p.parent.mkdir(parents=True, exist_ok=True)
                self._ctrl_path = p
                self._ctrl_server = _CtrlServer(str(p), self)
                self._ctrl_thread = Thread(
                    target=self._ctrl_server.serve_forever,
                    kwargs={"poll_interval": 0.1},
                    daemon=True,
                )
                self._ctrl_thread.start()

    def _record_stats(self):
        out_dir = Path(os.environ.get("PHASE3_STAGE2_STATS_DIR", os.environ["PHASE3_TEST_DIR"]))
        node_id = self.config.node_id
        path = out_dir / f"vericache-dist-stats-inst{node_id}.json"
        next_sample = path.with_suffix(".next")
        while not self._stop_sampling.is_set():
            try:
                if self.engine is not None:
                    counters = {
                        "node_id": node_id,
                        "pid": os.getpid(),
                        "is_slab_cuda_pinned": self.engine.is_slab_cuda_pinned,
                        "rdma": self.engine.rdma_stats(),
                    }
                    next_sample.write_text(json.dumps(counters))
                    next_sample.replace(path)
            except Exception:
                pass
            self._stop_sampling.wait(0.25)

    def shutdown(self):
        if hasattr(self, "_stop_sampling"):
            self._stop_sampling.set()
            self._sampler.join(timeout=1)
        if getattr(self, "_ctrl_server", None) is not None:
            try:
                self._ctrl_server.shutdown()
                self._ctrl_server.server_close()
            except Exception:
                pass
            if self._ctrl_path is not None:
                try:
                    self._ctrl_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._ctrl_server = None
        super().shutdown()
