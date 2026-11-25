"""Decoupled Scheme A Global Directory service for cross-instance RDMA prefix routing.

Tracks per-(instance_id, incarnation, tp_rank) DRAM-resident (RDMA-servable) KV blocks
across independent vLLM instances. Automatically revokes all entries of an instance
upon disconnect, crash, or re-registration with a new incarnation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import signal
import socket
import socketserver
import struct
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep
from urllib.parse import urlsplit

from .hashing import derive_layer_block_key

_HEADER = struct.Struct("!I")
_KEY_WORDS = struct.Struct("<4Q")
_MAX_MESSAGE = 32 * 1024 * 1024


def _address(endpoint: str):
    url = urlsplit(endpoint)
    if (
        url.scheme == "unix"
        and not url.netloc
        and url.path
        and not url.query
        and not url.fragment
    ):
        return socket.AF_UNIX, url.path
    if (
        url.scheme == "tcp"
        and url.hostname
        and url.port is not None
        and not url.path
        and not url.query
        and not url.fragment
        and not url.username
    ):
        family = socket.AF_INET6 if ":" in url.hostname else socket.AF_INET
        return family, (url.hostname, url.port)
    raise ValueError("use unix:///path or tcp://host:port for global_directory_endpoint")


def _send(sock: socket.socket, value) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > _MAX_MESSAGE:
        raise ValueError("VeriCache directory message exceeds 32 MiB")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _read(stream):
    header = stream.read(_HEADER.size)
    if not header:
        raise EOFError
    if len(header) != _HEADER.size:
        raise ConnectionError("truncated VeriCache directory header")
    (length,) = _HEADER.unpack(header)
    if length > _MAX_MESSAGE:
        raise ValueError("VeriCache directory message exceeds 32 MiB")
    payload = stream.read(length)
    if len(payload) != length:
        raise ConnectionError("truncated VeriCache directory payload")
    return json.loads(payload)


def _encode_hash(digest: bytes) -> str:
    if not isinstance(digest, bytes) or len(digest) != _KEY_WORDS.size:
        raise ValueError("prefix hash must be a 32-byte digest")
    return digest.hex()


def _decode_hash(encoded: str) -> bytes:
    if not isinstance(encoded, str) or len(encoded) != 2 * _KEY_WORDS.size:
        raise ValueError("expected a 64-character lowercase hex digest")
    return bytes.fromhex(encoded)


def _encode_key(key: tuple[int, int, int, int]) -> str:
    return _KEY_WORDS.pack(*key).hex()


def _decode_key(encoded: str) -> tuple[int, int, int, int]:
    return _KEY_WORDS.unpack(_decode_hash(encoded))


class _WorkerRankState:
    def __init__(self, layers: frozenset[str], session: object) -> None:
        self.layers = layers
        self.session = session
        # block_hash -> set[layer_name]
        self.blocks: dict[bytes, set[str]] = {}
        # native_key -> (block_hash, layer_name)
        self.keys: dict[tuple[int, int, int, int], tuple[bytes, str]] = {}
        # block_hashes with all layers present on this rank
        self.complete_blocks: set[bytes] = set()

    def mark_saved(
        self, block: bytes, layer: str, key: tuple[int, int, int, int]
    ) -> bool:
        """Returns True if `block` just became complete on this rank."""
        prev = self.keys.get(key)
        if prev is not None and prev != (block, layer):
            self.evict_key(key)
        layer_set = self.blocks.setdefault(block, set())
        layer_set.add(layer)
        self.keys[key] = (block, layer)
        if len(layer_set) == len(self.layers) and block not in self.complete_blocks:
            self.complete_blocks.add(block)
            return True
        return False

    def evict_key(self, key: tuple[int, int, int, int]) -> bytes | None:
        """Returns `block` if `block` just lost completeness on this rank."""
        saved = self.keys.pop(key, None)
        if saved is None:
            return None
        block, layer = saved
        layer_set = self.blocks.get(block)
        if layer_set is not None:
            layer_set.discard(layer)
            if not layer_set:
                del self.blocks[block]
        if block in self.complete_blocks:
            self.complete_blocks.discard(block)
            return block
        return None


class _InstanceState:
    def __init__(self, instance_id: int, incarnation: str, world_size: int) -> None:
        self.instance_id = instance_id
        self.incarnation = incarnation
        self.world_size = world_size
        self.ranks: dict[int, _WorkerRankState] = {}
        # Blocks complete across ALL world_size ranks of this incarnation.
        self.complete_blocks: set[bytes] = set()

    def is_block_complete(self, block: bytes) -> bool:
        if len(self.ranks) < self.world_size:
            return False
        for r in range(self.world_size):
            st = self.ranks.get(r)
            if st is None or block not in st.complete_blocks:
                return False
        return True

    def recompute_block(self, block: bytes) -> bool:
        now_complete = self.is_block_complete(block)
        if now_complete:
            self.complete_blocks.add(block)
        else:
            self.complete_blocks.discard(block)
        return now_complete


@dataclass(frozen=True)
class PrefixPlanEntry:
    block_hash: bytes
    owner_instance: int
    is_remote: bool


class GlobalDirectoryIndex:
    """Thread-safe cluster-wide prefix presence index across vLLM instances."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._instances: dict[int, _InstanceState] = {}
        # Reverse index: block_hash -> set[instance_id] where complete in DRAM
        self._block_owners: dict[bytes, set[int]] = {}
        # Stats
        self._queries = 0
        self._local_hits = 0
        self._remote_hits = 0
        self._revocations_disconnect = 0
        self._revocations_incarnation = 0
        self._evictions_processed = 0

    def _drop_instance_locked(self, instance_id: int) -> None:
        inst = self._instances.pop(instance_id, None)
        if inst is None:
            return
        for blk in inst.complete_blocks:
            owners = self._block_owners.get(blk)
            if owners is not None:
                owners.discard(instance_id)
                if not owners:
                    del self._block_owners[blk]

    def register_worker(
        self,
        instance_id: int,
        incarnation: str,
        tp_rank: int,
        world_size: int,
        layers,
        *,
        session=None,
    ) -> None:
        if type(instance_id) is not int or instance_id < 0:
            raise ValueError("instance_id must be a nonnegative int")
        if not isinstance(incarnation, str) or not incarnation:
            raise ValueError("incarnation must be a nonempty string")
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive int")
        if type(tp_rank) is not int or not (0 <= tp_rank < world_size):
            raise ValueError("tp_rank out of range")
        layer_set = frozenset(layers)
        if not layer_set or any(not isinstance(l, str) for l in layer_set):
            raise ValueError("layers must be nonempty strings")

        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None or inst.incarnation != incarnation or inst.world_size != world_size:
                if inst is not None:
                    self._drop_instance_locked(instance_id)
                    self._revocations_incarnation += 1
                inst = _InstanceState(instance_id, incarnation, world_size)
                self._instances[instance_id] = inst
            prev_rank = inst.ranks.get(tp_rank)
            if prev_rank is not None:
                # Re-registering same rank clears its previous blocks.
                for blk in list(prev_rank.complete_blocks):
                    inst.complete_blocks.discard(blk)
                    owners = self._block_owners.get(blk)
                    if owners is not None:
                        owners.discard(instance_id)
                        if not owners:
                            del self._block_owners[blk]
            inst.ranks[tp_rank] = _WorkerRankState(layer_set, session)

    def disconnect(self, session) -> None:
        with self._lock:
            to_drop = []
            for iid, inst in self._instances.items():
                if any(r.session is session for r in inst.ranks.values()):
                    to_drop.append(iid)
            for iid in to_drop:
                self._drop_instance_locked(iid)
                self._revocations_disconnect += 1

    def revoke_instance(self, instance_id: int) -> None:
        with self._lock:
            if instance_id in self._instances:
                self._drop_instance_locked(instance_id)
                self._revocations_incarnation += 1

    def mark_saved_batch(
        self,
        instance_id: int,
        incarnation: str,
        tp_rank: int,
        saved,
        *,
        evicted=(),
        session=None,
    ) -> None:
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None or inst.incarnation != incarnation:
                raise ValueError(
                    f"instance {instance_id} incarnation {incarnation!r} not registered"
                )
            rank_state = inst.ranks.get(tp_rank)
            if rank_state is None or (session is not None and rank_state.session is not session):
                raise ValueError(
                    f"worker ({instance_id}, {tp_rank}) must register on this connection"
                )

            touched_complete: set[bytes] = set()
            touched_lost: set[bytes] = set()

            for key in evicted:
                lost_blk = rank_state.evict_key(key)
                self._evictions_processed += 1
                if lost_blk is not None:
                    touched_lost.add(lost_blk)

            for item in saved:
                if len(item) == 2:
                    block, layer = item
                    native_key = derive_layer_block_key(block, layer)
                else:
                    block, layer, native_key = item
                if layer not in rank_state.layers:
                    continue
                if rank_state.mark_saved(block, layer, native_key):
                    touched_complete.add(block)

            for blk in touched_lost:
                if not inst.recompute_block(blk):
                    owners = self._block_owners.get(blk)
                    if owners is not None:
                        owners.discard(instance_id)
                        if not owners:
                            del self._block_owners[blk]

            for blk in touched_complete:
                if inst.recompute_block(blk):
                    self._block_owners.setdefault(blk, set()).add(instance_id)

    def query_prefix_plan(
        self,
        local_instance_id: int,
        block_hashes,
        expected_world_size: int = 1,
        expected_layers=(),
        *,
        allow_remote: bool = True,
    ) -> list[tuple[int, bool]]:
        """Return contiguous prefix matches as `[(owner_instance_id, is_remote), ...]`.

        Local-first: if `local_instance_id` owns the block, returns `(local_instance_id, False)`.
        Otherwise, if `allow_remote`, picks the lowest live peer `owner_id` with matching
        `world_size` and `layers` and returns `(owner_id, True)`.
        Stops at the first miss to preserve contiguous KV prefix semantics.
        """
        req_layers = frozenset(expected_layers or ())
        plan: list[tuple[int, bool]] = []
        with self._lock:
            self._queries += 1
            for blk in block_hashes:
                owners = self._block_owners.get(blk)
                if not owners:
                    break
                if local_instance_id in owners:
                    inst = self._instances.get(local_instance_id)
                    if (
                        inst is not None
                        and inst.world_size == expected_world_size
                        and all(
                            req_layers.issubset(inst.ranks[r].layers)
                            for r in range(expected_world_size)
                            if r in inst.ranks
                        )
                    ):
                        plan.append((local_instance_id, False))
                        self._local_hits += 1
                        continue
                if not allow_remote:
                    break
                chosen = None
                for candidate in sorted(owners):
                    if candidate == local_instance_id:
                        continue
                    inst = self._instances.get(candidate)
                    if inst is None or inst.world_size != expected_world_size:
                        continue
                    if req_layers and any(
                        not req_layers.issubset(inst.ranks[r].layers)
                        for r in range(expected_world_size)
                        if r in inst.ranks
                    ):
                        continue
                    chosen = candidate
                    break
                if chosen is None:
                    break
                plan.append((chosen, True))
                self._remote_hits += 1
        return plan

    def stats(self) -> dict:
        with self._lock:
            return {
                "instances": {
                    str(iid): {
                        "incarnation": inst.incarnation,
                        "world_size": inst.world_size,
                        "ranks": sorted(inst.ranks.keys()),
                        "complete_blocks": len(inst.complete_blocks),
                    }
                    for iid, inst in self._instances.items()
                },
                "unique_blocks": len(self._block_owners),
                "queries": self._queries,
                "local_hits": self._local_hits,
                "remote_hits": self._remote_hits,
                "revocations_disconnect": self._revocations_disconnect,
                "revocations_incarnation": self._revocations_incarnation,
                "evictions_processed": self._evictions_processed,
            }


class _DirHandler(socketserver.StreamRequestHandler):
    def handle(self):
        owner = self.server.owner
        session = object()
        try:
            while True:
                request = _read(self.rfile)
                try:
                    result = owner._dispatch(request, session)
                    _send(self.connection, {"ok": result})
                except (ValueError, TypeError, KeyError) as error:
                    _send(self.connection, {"error": str(error)})
        except (EOFError, OSError, ValueError):
            pass
        finally:
            owner._index.disconnect(session)


class _DirSocketServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, family, address, owner):
        self.address_family = family
        self.owner = owner
        self._connections = set()
        self._connections_lock = Lock()
        super().__init__(address, _DirHandler)

    def get_request(self):
        sock, address = super().get_request()
        if self.address_family != socket.AF_UNIX:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._connections_lock:
            self._connections.add(sock)
        return sock, address

    def close_request(self, request):
        with self._connections_lock:
            self._connections.discard(request)
        super().close_request(request)

    def server_close(self):
        with self._connections_lock:
            for sock in self._connections:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        super().server_close()


class GlobalDirectoryServer:
    """Decoupled TCP/UNIX metadata directory serving multiple vLLM instances."""

    def __init__(self, endpoint: str) -> None:
        family, address = _address(endpoint)
        self._index = GlobalDirectoryIndex()
        self._server = _DirSocketServer(family, address, self)
        self._socket_path = Path(address) if family == socket.AF_UNIX else None
        self._inode = self._socket_path.stat().st_ino if self._socket_path else None
        if self._socket_path:
            os.chmod(self._socket_path, 0o600)
        if family == socket.AF_INET6:
            self.endpoint = f"tcp://[{address[0]}]:{self._server.server_address[1]}"
        elif family == socket.AF_INET:
            self.endpoint = f"tcp://{address[0]}:{self._server.server_address[1]}"
        else:
            self.endpoint = endpoint
        self._thread = None

    def start(self) -> "GlobalDirectoryServer":
        if self._thread is None:
            self._thread = Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="vericache-global-dir",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()
        if self._socket_path is not None:
            try:
                if self._socket_path.stat().st_ino == self._inode:
                    self._socket_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "GlobalDirectoryServer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.close()

    def stats(self) -> dict:
        return self._index.stats()

    def _dispatch(self, request, session):
        op, args = request["op"], request["args"]
        if op == "ping":
            return True
        if op == "stats":
            return self._index.stats()
        if op == "register":
            instance_id, incarnation, tp_rank, world_size, layers = args
            return self._index.register_worker(
                instance_id, incarnation, tp_rank, world_size, layers, session=session
            )
        if op == "saved":
            instance_id, incarnation, tp_rank, saved, evicted = args
            rows = [
                (_decode_hash(block), layer, _decode_key(key))
                for block, layer, key in saved
            ]
            return self._index.mark_saved_batch(
                instance_id,
                incarnation,
                tp_rank,
                rows,
                evicted=tuple(map(_decode_key, evicted)),
                session=session,
            )
        if op == "query_plan":
            local_instance_id, blocks, world_size, layers, allow_remote = args
            plan = self._index.query_prefix_plan(
                local_instance_id,
                tuple(map(_decode_hash, blocks)),
                world_size,
                layers,
                allow_remote=bool(allow_remote),
            )
            return [[owner, is_remote] for owner, is_remote in plan]
        if op == "revoke_instance":
            (instance_id,) = args
            return self._index.revoke_instance(instance_id)
        raise ValueError(f"unknown global directory operation {op!r}")


class GlobalDirectoryClient:
    """Persistent client connection to a `GlobalDirectoryServer`."""

    def __init__(self, endpoint: str, *, timeout: float = 5.0) -> None:
        self._family, self._address = _address(endpoint)
        self._timeout = timeout
        self._lock = Lock()
        self._socket = None
        self._stream = None

    def _connect(self, wait: bool) -> None:
        deadline = monotonic() + self._timeout
        while True:
            sock = socket.socket(self._family, socket.SOCK_STREAM)
            try:
                sock.settimeout(max(0.01, deadline - monotonic()) if wait else 0.1)
                sock.connect(self._address)
                sock.settimeout(self._timeout)
                if self._family != socket.AF_UNIX:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                if not wait or monotonic() >= deadline:
                    raise
                sleep(min(0.02, max(0, deadline - monotonic())))
            except BaseException:
                sock.close()
                raise
            else:
                self._socket, self._stream = sock, sock.makefile("rb")
                return

    def _disconnect(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            try:
                self._socket.close()
            except OSError:
                pass
            self._stream = self._socket = None

    def close(self) -> None:
        with self._lock:
            self._disconnect()

    def __enter__(self) -> "GlobalDirectoryClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _call(self, op: str, *args, wait: bool = True):
        with self._lock:
            try:
                if self._socket is None:
                    self._connect(wait)
                _send(self._socket, {"op": op, "args": args})
                response = _read(self._stream)
            except (EOFError, OSError, ValueError):
                self._disconnect()
                raise
        if "error" in response:
            raise ValueError(response["error"])
        return response["ok"]

    def ping(self) -> bool:
        return bool(self._call("ping"))

    def stats(self) -> dict:
        return self._call("stats")

    def register_worker(
        self,
        instance_id: int,
        incarnation: str,
        tp_rank: int,
        world_size: int,
        layers,
        *,
        wait: bool = True,
    ) -> None:
        self._call(
            "register",
            int(instance_id),
            str(incarnation),
            int(tp_rank),
            int(world_size),
            list(layers),
            wait=wait,
        )

    def mark_saved_batch(
        self,
        instance_id: int,
        incarnation: str,
        tp_rank: int,
        saved,
        *,
        evicted=(),
    ) -> None:
        rows = []
        for item in saved:
            if len(item) == 2:
                block, layer = item
                key = derive_layer_block_key(block, layer)
            else:
                block, layer, key = item
            rows.append((_encode_hash(block), layer, _encode_key(key)))
        self._call(
            "saved",
            int(instance_id),
            str(incarnation),
            int(tp_rank),
            rows,
            list(map(_encode_key, evicted)),
        )

    def query_prefix_plan(
        self,
        local_instance_id: int,
        block_hashes,
        expected_world_size: int = 1,
        expected_layers=(),
        *,
        allow_remote: bool = True,
    ) -> list[tuple[int, bool]]:
        raw = self._call(
            "query_plan",
            int(local_instance_id),
            list(map(_encode_hash, block_hashes)),
            int(expected_world_size),
            list(expected_layers or ()),
            bool(allow_remote),
        )
        return [(int(owner), bool(is_remote)) for owner, is_remote in raw]

    def revoke_instance(self, instance_id: int) -> None:
        self._call("revoke_instance", int(instance_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriCache Scheme A Global Directory Server")
    parser.add_argument("--endpoint", required=True, help="tcp://host:port or unix:///path")
    parser.add_argument("--ready-file", default=None, help="Write bound endpoint once ready")
    args = parser.parse_args()

    srv = GlobalDirectoryServer(args.endpoint).start()
    if args.ready_file:
        p = Path(args.ready_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(srv.endpoint + "\n")
        tmp.replace(p)
    print(f"GlobalDirectoryServer listening on {srv.endpoint}", flush=True)

    stop = False

    def _sig(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    try:
        while not stop:
            sleep(0.2)
    finally:
        srv.close()


if __name__ == "__main__":
    main()
