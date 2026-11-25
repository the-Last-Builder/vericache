"""Deterministic Session-Sticky HTTP Streaming Router for Stage 2 (Exp 2.2).

Routes OpenAI `/v1/chat/completions` requests across 2 vLLM instances with a
reproducible sticky/drift policy (default: 70% sticky to prefix home instance,
30% cross-instance drift to peer instance) and tracks theoretical local vs
cross-instance remote KV reuse opportunities.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import sys
from threading import Lock
import time
from urllib.parse import urlsplit


class RouterState:
    def __init__(self, backends: list[str], sticky_pct: int = 70, seed: int = 20261124):
        assert len(backends) == 2, "Stage 2 router expects exactly 2 backend URLs"
        self.backends = [urlsplit(u) for u in backends]
        self.backend_urls = list(backends)
        self.sticky_pct = int(sticky_pct)
        self.seed = int(seed)
        self._lock = Lock()
        self._seq = 0
        self._seen_on: list[set[str]] = [set(), set()]
        self.stats = {
            "total_requests": 0,
            "routed_to_inst0": 0,
            "routed_to_inst1": 0,
            "sticky_requests": 0,
            "drifted_requests": 0,
            "theoretical_local_reuse_reqs": 0,
            "theoretical_remote_reuse_reqs": 0,
            "cold_prefix_reqs": 0,
            "errors": 0,
        }

    @staticmethod
    def extract_prefix_key(body: bytes) -> str:
        try:
            payload = json.loads(body)
        except Exception:
            return hashlib.sha256(body[:256]).hexdigest()
        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs:
            parts = []
            for m in msgs:
                content = m.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            text = "\n".join(parts)
        else:
            text = str(payload.get("prompt", ""))
        # First 3584 chars span both the shared root block (~2528 chars in toolagent_routed)
        # and the per-session block (hash_ids[1]), cleanly identifying each conversation/session
        # while staying stable across multi-turn extensions.
        prefix = text[:3584]
        return hashlib.sha256(prefix.encode("utf-8", errors="ignore")).hexdigest()

    def choose_backend(self, body: bytes) -> tuple[int, str, bool]:
        pkey = self.extract_prefix_key(body)
        home = int(pkey[:8], 16) % 2
        with self._lock:
            seq = self._seq
            self._seq += 1
            bucket = int(
                hashlib.sha256(f"{self.seed}:{seq}:{pkey}".encode()).hexdigest()[:8], 16
            ) % 100
            if bucket < self.sticky_pct:
                target = home
                drifted = False
                self.stats["sticky_requests"] += 1
            else:
                target = 1 - home
                drifted = True
                self.stats["drifted_requests"] += 1

            self.stats["total_requests"] += 1
            if target == 0:
                self.stats["routed_to_inst0"] += 1
            else:
                self.stats["routed_to_inst1"] += 1

            seen_local = pkey in self._seen_on[target]
            seen_peer = pkey in self._seen_on[1 - target]
            if seen_local:
                self.stats["theoretical_local_reuse_reqs"] += 1
            elif seen_peer:
                self.stats["theoretical_remote_reuse_reqs"] += 1
            else:
                self.stats["cold_prefix_reqs"] += 1
            self._seen_on[target].add(pkey)
            return target, pkey, drifted

    def reset_stats(self) -> None:
        """Reset counters at warmup -> measurement boundary while preserving seen prefixes."""
        with self._lock:
            for k in self.stats:
                self.stats[k] = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **self.stats,
                "backends": self.backend_urls,
                "sticky_pct": self.sticky_pct,
                "seed": self.seed,
                "unique_prefixes_inst0": len(self._seen_on[0]),
                "unique_prefixes_inst1": len(self._seen_on[1]),
            }


class _RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def do_GET(self):
        state: RouterState = self.server.state
        if self.path == "/health":
            body = b'{"status":"ok"}\n'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/router_stats":
            body = (json.dumps(state.snapshot()) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        state: RouterState = self.server.state
        if self.path == "/router_reset_stats":
            state.reset_stats()
            body = b'{"ok":true}\n'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        target_idx, _pkey, _drifted = state.choose_backend(body)
        backend = state.backends[target_idx]

        conn = http.client.HTTPConnection(backend.hostname, backend.port, timeout=330)
        try:
            fwd_headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
                "Accept": self.headers.get("Accept", "*/*"),
                "Content-Length": str(len(body)),
                "Connection": "close",
            }
            if "Authorization" in self.headers:
                fwd_headers["Authorization"] = self.headers["Authorization"]

            conn.request("POST", self.path, body=body, headers=fwd_headers)
            resp = conn.getresponse()

            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type")
            if ctype:
                self.send_header("Content-Type", ctype)
            is_chunked = (resp.getheader("Transfer-Encoding") or "").lower() == "chunked"
            clen = resp.getheader("Content-Length")
            if is_chunked or clen is None:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read1(4096)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            # Client closed stream early (e.g. immediately after SSE [DONE]); not a backend error.
            self.close_connection = True
        except Exception as exc:
            with state._lock:
                state.stats["errors"] += 1
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(err)
            except OSError:
                pass
            self.close_connection = True
        finally:
            conn.close()


class StickyRouterServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, bind_addr, state: RouterState):
        self.state = state
        super().__init__(bind_addr, _RouterHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Deterministic Session-Sticky Router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backends", nargs=2, required=True, help="Two backend HTTP URLs")
    parser.add_argument("--sticky-pct", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    state = RouterState(args.backends, sticky_pct=args.sticky_pct, seed=args.seed)
    srv = StickyRouterServer((args.host, args.port), state)

    def _stop(_sig, _frame):
        srv.shutdown()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(
        f"StickyRouter listening on http://{args.host}:{args.port} -> {args.backends} "
        f"(sticky={args.sticky_pct}%, seed={args.seed})",
        flush=True,
    )
    try:
        srv.serve_forever(poll_interval=0.1)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
