"""Length-prefixed JSON metadata IPC for workers with independent KV stores."""

import json
import os
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
_MAX_MESSAGE = 16 * 1024 * 1024


def _address(endpoint):
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
    raise ValueError("use unix:///path or tcp://host:port for ipc_endpoint")


def _send(sock, value):
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > _MAX_MESSAGE:
        raise ValueError("VeriCache metadata message exceeds 16 MiB")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _read(stream):
    header = stream.read(_HEADER.size)
    if not header:
        raise EOFError
    if len(header) != _HEADER.size:
        raise ConnectionError("truncated VeriCache metadata header")
    (length,) = _HEADER.unpack(header)
    if length > _MAX_MESSAGE:
        raise ValueError("VeriCache metadata message exceeds 16 MiB")
    payload = stream.read(length)
    if len(payload) != length:
        raise ConnectionError("truncated VeriCache metadata payload")
    return json.loads(payload)


def _encode_hash(digest):
    if not isinstance(digest, bytes) or len(digest) != _KEY_WORDS.size:
        raise ValueError("prefix hash must be a 32-byte digest")
    return digest.hex()


def _decode_hash(encoded):
    if not isinstance(encoded, str) or len(encoded) != 2 * _KEY_WORDS.size:
        raise ValueError("expected a 64-character lowercase hex digest")
    digest = bytes.fromhex(encoded)
    if digest.hex() != encoded:
        raise ValueError("expected a 64-character lowercase hex digest")
    return digest


def _encode_key(key):
    return _KEY_WORDS.pack(*key).hex()


def _decode_key(encoded):
    return _KEY_WORDS.unpack(_decode_hash(encoded))


class _Rank:
    def __init__(self, layers, session):
        self.layers = frozenset(layers)
        if not self.layers or any(not isinstance(layer, str) for layer in self.layers):
            raise ValueError("worker layers must be nonempty layer names")
        self.session = session
        self.blocks = {}
        self.keys = {}

    def evict(self, key):
        saved = self.keys.pop(key, None)
        if saved is not None:
            block, layer = saved
            self.blocks[block].discard(layer)
            if not self.blocks[block]:
                del self.blocks[block]


class _Presence:
    def __init__(self):
        self._lock = Lock()
        self._ranks = {}

    def register_worker(self, tp_rank, layers, *, session=None):
        if type(tp_rank) is not int or tp_rank < 0:
            raise ValueError("invalid TP rank")
        worker = _Rank(layers, session)
        with self._lock:
            previous = self._ranks.get(tp_rank)
            if (
                previous is None
                or previous.session is not session
                or previous.layers != worker.layers
            ):
                self._ranks[tp_rank] = worker

    def disconnect(self, session):
        with self._lock:
            for rank in [
                rank
                for rank, worker in self._ranks.items()
                if worker.session is session
            ]:
                del self._ranks[rank]

    def mark_saved(self, tp_rank, block_hash, layer_name, key=None):
        self.mark_saved_batch(tp_rank, [(block_hash, layer_name, key)])

    def mark_saved_batch(self, tp_rank, saved, *, evicted=(), session=None):
        with self._lock:
            worker = self._ranks.get(tp_rank)
            if worker is None or worker.session is not session:
                raise ValueError(
                    f"TP rank {tp_rank} needs to register on this connection"
                )
            rows = []
            for block, layer, *key in saved:
                if layer not in worker.layers or len(key) > 1:
                    raise ValueError(f"unexpected saved layer for rank {tp_rank}")
                native_key = (
                    key[0]
                    if key and key[0] is not None
                    else derive_layer_block_key(block, layer)
                )
                rows.append((block, layer, native_key))
            for key in evicted:
                worker.evict(key)
            for block, layer, key in rows:
                if key in worker.keys and worker.keys[key] != (block, layer):
                    worker.evict(key)
                worker.blocks.setdefault(block, set()).add(layer)
                worker.keys[key] = (block, layer)

    def mark_evicted(self, tp_rank, keys, *, session=None):
        self.mark_saved_batch(tp_rank, (), evicted=keys, session=session)

    def query_prefix_hits(self, block_hashes, expected_world_size, expected_layers=()):
        if type(expected_world_size) is not int or expected_world_size < 1:
            raise ValueError("expected_world_size must be positive")
        layers = frozenset(expected_layers or ())
        with self._lock:
            if expected_world_size > len(self._ranks):
                return 0
            ranks = [self._ranks.get(rank) for rank in range(expected_world_size)]
            if any(rank is None or not layers.issubset(rank.layers) for rank in ranks):
                return 0
            # A worker can register additional layers. Those layers also need to
            # be present because the worker will restore them on a prefix hit.
            for count, block in enumerate(block_hashes):
                if any(rank.blocks.get(block) != rank.layers for rank in ranks):
                    return count
            return len(block_hashes)


class _Handler(socketserver.StreamRequestHandler):
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


class _SocketServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, family, address, owner):
        self.address_family = family
        self.owner = owner
        self._connections = set()
        self._connections_lock = Lock()
        super().__init__(address, _Handler)

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


class MetadataServer:
    """Serve a thread-safe presence index for one vLLM engine."""

    def __init__(self, endpoint):
        family, address = _address(endpoint)
        self._index = _Presence()
        self._server = _SocketServer(family, address, self)
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

    def start(self):
        if self._thread is None:
            self._thread = Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="vericache-metadata",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self):
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join()
            self._thread = None
        self._server.server_close()
        if self._socket_path is not None:
            try:
                if self._socket_path.stat().st_ino == self._inode:
                    self._socket_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    def register_worker(self, tp_rank, layers):
        self._index.register_worker(tp_rank, layers)

    def mark_saved(self, tp_rank, block_hash, layer_name, key=None):
        self._index.mark_saved(tp_rank, block_hash, layer_name, key)

    def mark_saved_batch(self, tp_rank, saved, *, evicted=()):
        self._index.mark_saved_batch(tp_rank, saved, evicted=evicted)

    def mark_evicted(self, tp_rank, keys):
        self._index.mark_evicted(tp_rank, keys)

    def query_prefix_hits(self, block_hashes, expected_world_size, expected_layers=()):
        return self._index.query_prefix_hits(
            block_hashes, expected_world_size, expected_layers
        )

    def _dispatch(self, request, session):
        op, args = request["op"], request["args"]
        if op == "ping":
            return True
        if op == "register":
            return self._index.register_worker(*args, session=session)
        if op == "saved":
            rank, saved, evicted = args
            rows = [
                (_decode_hash(block), layer, _decode_key(key))
                for block, layer, key in saved
            ]
            return self._index.mark_saved_batch(
                rank, rows, evicted=tuple(map(_decode_key, evicted)), session=session
            )
        if op == "evicted":
            rank, keys = args
            return self._index.mark_evicted(
                rank, tuple(map(_decode_key, keys)), session=session
            )
        if op == "query":
            blocks, world_size, layers = args
            return self._index.query_prefix_hits(
                tuple(map(_decode_hash, blocks)), world_size, layers
            )
        raise ValueError(f"unknown metadata operation {op!r}")


class MetadataClient:
    """Keep a socket open for each scheduler or worker registration."""

    def __init__(self, endpoint, *, timeout=5.0):
        self._family, self._address = _address(endpoint)
        self._timeout = timeout
        self._lock = Lock()
        self._socket = None
        self._stream = None

    def _connect(self, wait):
        deadline = monotonic() + self._timeout
        while True:
            sock = socket.socket(self._family, socket.SOCK_STREAM)
            try:
                sock.settimeout(max(0.01, deadline - monotonic()) if wait else 0.05)
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

    def _disconnect(self):
        if self._stream is not None:
            self._stream.close()
            self._socket.close()
            self._stream = self._socket = None

    def close(self):
        with self._lock:
            self._disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _call(self, op, *args, wait=True):
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

    def ping(self):
        return self._call("ping")

    def register_worker(self, tp_rank, layers, *, wait=True):
        self._call("register", tp_rank, list(layers), wait=wait)

    def mark_saved(self, tp_rank, block_hash, layer_name, key=None):
        self.mark_saved_batch(tp_rank, [(block_hash, layer_name, key)])

    def mark_saved_batch(self, tp_rank, saved, *, evicted=()):
        rows = []
        for block, layer, *keys in saved:
            if len(keys) > 1:
                raise ValueError("expected a single key per saved layer")
            key = (
                keys[0]
                if keys and keys[0] is not None
                else derive_layer_block_key(block, layer)
            )
            rows.append((_encode_hash(block), layer, _encode_key(key)))
        self._call("saved", tp_rank, rows, list(map(_encode_key, evicted)))

    def mark_evicted(self, tp_rank, keys):
        self._call("evicted", tp_rank, list(map(_encode_key, keys)))

    def query_prefix_hits(self, block_hashes, expected_world_size, expected_layers=()):
        return self._call(
            "query",
            list(map(_encode_hash, block_hashes)),
            expected_world_size,
            list(expected_layers or ()),
        )
