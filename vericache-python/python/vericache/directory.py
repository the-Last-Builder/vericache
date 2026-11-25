"""Lightweight binary TCP Cluster Prefix Directory for peer-shared VeriCache clusters.

Each vLLM Scheduler registers newly stored 256-bit `ObjectKey`s with the directory
and batch-queries whether prefix blocks absent from its local tier exist on any
connected peer node. The directory is strictly metadata-only (zero KV data path
involvement and zero TCB expansion): all actual KV transfers flow through
VeriCache's verified 4-tier engine (Local HBM/DRAM/NVMe + GPUDirect RDMA).

Wire protocol (little-endian framing over persistent TCP_NODELAY connections):
  Request header: `<BQI` = (op: u8, node_id: u64, count: u32)
  Payload:        `count * 32` bytes (each key packed as `<4Q`)
  Ops:
    1 = EXISTS_REMOTE_BATCH  -> reply `<I` (count) + `count` bytes (1 if on any peer != node_id else 0)
    2 = REGISTER_BATCH       -> reply `<B` (0)
    3 = UNREGISTER_BATCH     -> reply `<B` (0)
    4 = STATS                -> reply `<I` (json_len) + UTF-8 JSON
    5 = UNREGISTER_PEERS     -> reply `<B` (0)
    6 = EXISTS_REMOTE_PREFIX -> reply `<I` (contiguous_hit_count)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import logging
import socket
import socketserver
import struct
import threading
from typing import Sequence

from . import ObjectKey

logger = logging.getLogger("vericache.directory")

_OP_EXISTS_REMOTE = 1
_OP_REGISTER = 2
_OP_UNREGISTER = 3
_OP_STATS = 4
_OP_UNREGISTER_PEERS = 5
_OP_EXISTS_REMOTE_PREFIX = 6

_REQ_HDR = struct.Struct("<BQI")
_KEY_STRUCT = struct.Struct("<4Q")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(mv[got:], n - got)
        if r == 0:
            raise EOFError("directory connection closed")
        got += r
    return bytes(buf)


def _pack_keys_bytes(keys: Sequence[ObjectKey | bytes] | bytes | bytearray | memoryview) -> tuple[bytes, int]:
    if isinstance(keys, (bytes, bytearray, memoryview)):
        raw = bytes(keys)
        if len(raw) % 32 != 0:
            raise ValueError(f"raw keys length {len(raw)} is not a multiple of 32")
        return raw, len(raw) // 32
    n = len(keys)
    if n == 0:
        return b"", 0
    first = keys[0]
    if isinstance(first, (bytes, bytearray, memoryview)):
        raw = b"".join(keys)  # type: ignore[arg-type]
        if len(raw) != n * 32:
            raise ValueError(f"expected {n * 32} raw key bytes, got {len(raw)}")
        return raw, n
    buf = bytearray(n * 32)
    for i, k in enumerate(keys):
        _KEY_STRUCT.pack_into(buf, i * 32, k[0], k[1], k[2], k[3])  # type: ignore[index]
    return bytes(buf), n


class _DirectoryState:
    def __init__(self, max_keys_per_node: int = 1_000_000) -> None:
        self._lock = threading.Lock()
        self._max_keys_per_node = max(16, int(max_keys_per_node))
        # node_id -> OrderedDict[bytes(32), None] (LRU order)
        self._node_keys: dict[int, OrderedDict[bytes, None]] = {}
        # key_bytes -> set[node_id]
        self._key_owners: dict[bytes, set[int]] = {}
        self._total_registers = 0
        self._total_unregisters = 0
        self._total_queries = 0
        self._total_remote_hits = 0

    def register_batch(self, node_id: int, raw_keys: bytes, count: int) -> None:
        with self._lock:
            lru = self._node_keys.get(node_id)
            if lru is None:
                lru = OrderedDict()
                self._node_keys[node_id] = lru
            for i in range(count):
                k = raw_keys[i * 32 : (i + 1) * 32]
                if k in lru:
                    lru.move_to_end(k)
                else:
                    if len(lru) >= self._max_keys_per_node:
                        old_k, _ = lru.popitem(last=False)
                        owners = self._key_owners.get(old_k)
                        if owners is not None:
                            owners.discard(node_id)
                            if not owners:
                                self._key_owners.pop(old_k, None)
                    lru[k] = None
                    owners = self._key_owners.get(k)
                    if owners is None:
                        self._key_owners[k] = {node_id}
                    else:
                        owners.add(node_id)
            self._total_registers += count

    def unregister_batch(self, node_id: int, raw_keys: bytes, count: int) -> None:
        with self._lock:
            for i in range(count):
                k = raw_keys[i * 32 : (i + 1) * 32]
                if node_id == 0xFFFFFFFFFFFFFFFF:
                    # Global invalidation (e.g., stale remote load miss feedback)
                    owners = self._key_owners.pop(k, None)
                    if owners:
                        for nid in owners:
                            lru = self._node_keys.get(nid)
                            if lru is not None:
                                lru.pop(k, None)
                else:
                    lru = self._node_keys.get(node_id)
                    if lru is not None:
                        lru.pop(k, None)
                    owners = self._key_owners.get(k)
                    if owners is not None:
                        owners.discard(node_id)
                        if not owners:
                            self._key_owners.pop(k, None)
            self._total_unregisters += count

    def unregister_peers_batch(self, self_node_id: int, raw_keys: bytes, count: int) -> None:
        """Remove `raw_keys` from all owners OTHER than `self_node_id`."""
        with self._lock:
            for i in range(count):
                k = raw_keys[i * 32 : (i + 1) * 32]
                owners = self._key_owners.get(k)
                if not owners:
                    continue
                to_drop = [nid for nid in owners if nid != self_node_id]
                for nid in to_drop:
                    owners.discard(nid)
                    lru = self._node_keys.get(nid)
                    if lru is not None:
                        lru.pop(k, None)
                if not owners:
                    self._key_owners.pop(k, None)
            self._total_unregisters += count

    def exists_remote_batch(self, self_node_id: int, raw_keys: bytes, count: int) -> bytes:
        out = bytearray(count)
        hits = 0
        with self._lock:
            for i in range(count):
                k = raw_keys[i * 32 : (i + 1) * 32]
                owners = self._key_owners.get(k)
                if owners and (len(owners) > 1 or self_node_id not in owners):
                    out[i] = 1
                    hits += 1
                    for nid in owners:
                        if nid != self_node_id:
                            lru = self._node_keys.get(nid)
                            if lru is not None and k in lru:
                                lru.move_to_end(k)
            self._total_queries += count
            self._total_remote_hits += hits
        return bytes(out)

    def exists_remote_prefix(self, self_node_id: int, raw_keys: bytes, count: int) -> int:
        """Return length of longest contiguous remote-present prefix (early-exit on first miss)."""
        contig = 0
        with self._lock:
            for i in range(count):
                k = raw_keys[i * 32 : (i + 1) * 32]
                owners = self._key_owners.get(k)
                if owners and (len(owners) > 1 or self_node_id not in owners):
                    contig += 1
                    for nid in owners:
                        if nid != self_node_id:
                            lru = self._node_keys.get(nid)
                            if lru is not None and k in lru:
                                lru.move_to_end(k)
                else:
                    break
            self._total_queries += contig + (1 if contig < count else 0)
            self._total_remote_hits += contig
        return contig

    def stats(self) -> dict:
        with self._lock:
            return {
                "unique_keys": len(self._key_owners),
                "per_node_keys": {str(nid): len(lru) for nid, lru in self._node_keys.items()},
                "total_registers": self._total_registers,
                "total_unregisters": self._total_unregisters,
                "total_queries": self._total_queries,
                "total_remote_hits": self._total_remote_hits,
            }


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ClusterDirectoryServer:
    """In-process or standalone TCP server for VeriCache cluster prefix metadata."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        max_keys_per_node: int = 1_000_000,
    ) -> None:
        state = _DirectoryState(max_keys_per_node=max_keys_per_node)
        self._state = state

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                sock: socket.socket = self.request
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    while True:
                        hdr = _recv_exact(sock, _REQ_HDR.size)
                        op, node_id, count = _REQ_HDR.unpack(hdr)
                        payload = _recv_exact(sock, count * 32) if count > 0 else b""
                        if op == _OP_EXISTS_REMOTE:
                            flags = state.exists_remote_batch(node_id, payload, count)
                            sock.sendall(struct.pack("<I", count) + flags)
                        elif op == _OP_EXISTS_REMOTE_PREFIX:
                            contig = state.exists_remote_prefix(node_id, payload, count)
                            sock.sendall(struct.pack("<I", contig))
                        elif op == _OP_REGISTER:
                            state.register_batch(node_id, payload, count)
                            sock.sendall(b"\x00")
                        elif op == _OP_UNREGISTER:
                            state.unregister_batch(node_id, payload, count)
                            sock.sendall(b"\x00")
                        elif op == _OP_UNREGISTER_PEERS:
                            state.unregister_peers_batch(node_id, payload, count)
                            sock.sendall(b"\x00")
                        elif op == _OP_STATS:
                            body = json.dumps(state.stats()).encode("utf-8")
                            sock.sendall(struct.pack("<I", len(body)) + body)
                        else:
                            break
                except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                    pass

        self._server = _ThreadedTCPServer((host, port), _Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        h, p = self._server.server_address
        return f"{h}:{p}"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="vericache-directory",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def stats(self) -> dict:
        return self._state.stats()


class ClusterDirectoryClient:
    """Thread-safe persistent TCP client for `ClusterDirectoryServer`."""

    def __init__(self, addr: str, node_id: int, timeout_s: float = 5.0) -> None:
        host, port_s = addr.rsplit(":", 1)
        self._host = host
        self._port = int(port_s)
        self._node_id = int(node_id)
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None

    def _ensure_conn(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
        return self._sock

    def _reset_conn(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _rpc(
        self,
        op: int,
        node_id: int,
        keys: Sequence[ObjectKey | bytes] | bytes | bytearray | memoryview,
    ) -> bytes:
        payload, count = _pack_keys_bytes(keys) if keys else (b"", 0)
        msg = _REQ_HDR.pack(op, node_id, count) + payload
        with self._lock:
            for attempt in range(2):
                try:
                    s = self._ensure_conn()
                    s.sendall(msg)
                    if op == _OP_EXISTS_REMOTE:
                        n = struct.unpack("<I", _recv_exact(s, 4))[0]
                        data = _recv_exact(s, n) if n > 0 else b""
                        if len(data) != count:
                            raise OSError(
                                f"directory reply length mismatch: got {len(data)}, expected {count}"
                            )
                        return data
                    if op == _OP_EXISTS_REMOTE_PREFIX:
                        return _recv_exact(s, 4)
                    if op in (_OP_REGISTER, _OP_UNREGISTER, _OP_UNREGISTER_PEERS):
                        ack = _recv_exact(s, 1)
                        if ack != b"\x00":
                            raise OSError(f"unexpected directory ACK: {ack!r}")
                        return b""
                    if op == _OP_STATS:
                        n = struct.unpack("<I", _recv_exact(s, 4))[0]
                        return _recv_exact(s, n) if n > 0 else b"{}"
                    raise ValueError(f"unknown op {op}")
                except (OSError, EOFError):
                    self._reset_conn()
                    if attempt == 1:
                        raise
        return b""

    def exists_remote_batch(self, keys: Sequence[ObjectKey | bytes]) -> list[bool]:
        if not keys:
            return []
        try:
            raw = self._rpc(_OP_EXISTS_REMOTE, self._node_id, keys)
            return [b != 0 for b in raw]
        except (OSError, EOFError) as exc:
            logger.warning("VeriCache directory exists_remote_batch failed: %s", exc)
            return [False] * len(keys)

    def exists_remote_prefix(self, keys: Sequence[ObjectKey | bytes]) -> int:
        """Query longest contiguous remote-present prefix count (early-exit on server)."""
        if not keys:
            return 0
        try:
            raw = self._rpc(_OP_EXISTS_REMOTE_PREFIX, self._node_id, keys)
            return int(struct.unpack("<I", raw)[0])
        except (OSError, EOFError) as exc:
            logger.warning("VeriCache directory exists_remote_prefix failed: %s", exc)
            return 0

    def register_batch(self, keys: Sequence[ObjectKey | bytes] | bytes) -> None:
        if not keys:
            return
        try:
            self._rpc(_OP_REGISTER, self._node_id, keys)
        except (OSError, EOFError) as exc:
            logger.warning("VeriCache directory register_batch failed: %s", exc)

    def unregister_batch(
        self,
        keys: Sequence[ObjectKey | bytes] | bytes,
        *,
        global_evict: bool = False,
        peers_only: bool = False,
    ) -> None:
        if not keys:
            return
        try:
            if peers_only:
                self._rpc(_OP_UNREGISTER_PEERS, self._node_id, keys)
            else:
                nid = 0xFFFFFFFFFFFFFFFF if global_evict else self._node_id
                self._rpc(_OP_UNREGISTER, nid, keys)
        except (OSError, EOFError) as exc:
            logger.warning("VeriCache directory unregister_batch failed: %s", exc)

    def stats(self) -> dict:
        raw = self._rpc(_OP_STATS, self._node_id, ())
        return json.loads(raw.decode("utf-8"))

    def close(self) -> None:
        with self._lock:
            self._reset_conn()


def main() -> None:
    ap = argparse.ArgumentParser(description="VeriCache Cluster Prefix Directory Server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18900)
    ap.add_argument("--max-keys-per-node", type=int, default=1_000_000)
    args = ap.parse_args()

    srv = ClusterDirectoryServer(
        host=args.host,
        port=args.port,
        max_keys_per_node=args.max_keys_per_node,
    )
    print(f"vericache-directory listening on {srv.address}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
