"""Frame source that receives pre-decoded frames over a TCP socket.

This is the offline-driver mirror of the online ROS ingress: an external
publisher (e.g. ``scripts/habitat_online_sim.py`` running in a host
habitat-sim environment) renders frames while an agent moves and pushes them
here as they are produced, so ``scene_graph.offline.run`` maps them live.

Wire protocol (one stream, one publisher):

    [8-byte little-endian length][pickled frame dict]  ... repeated ...
    [8-byte zero]                                       # end-of-stream

Each pickled dict must already follow the pre-decoded contract of
``FrameSource`` (``rgb``, ``depth_f32``, ``T_world_cam`` in OpenCV
convention, ``rgb_instrinsics``, ...). The publisher is trusted — this is a
localhost bridge, not a hardened network service.
"""

from __future__ import annotations

import logging
import pickle
import socket
import struct
from typing import Iterator, Optional

from .base import FrameItem, FrameSource

LOGGER = logging.getLogger("scene_graph.offline.frame_sources.stream")

_LEN = struct.Struct("<Q")


class StreamFrameSource(FrameSource):
    """Listens on ``bind:port`` and yields frames as one publisher sends them.

    Args:
        port: TCP port to listen on.
        bind: Interface to bind (default loopback; the docker compose file
            uses host networking, so a host-side publisher reaches it there).
        accept_timeout_s: How long to wait for the publisher to connect.
        recv_timeout_s: Max seconds between frames before giving up.
    """

    def __init__(
        self,
        *,
        port: int = 5555,
        bind: str = "127.0.0.1",
        accept_timeout_s: float = 600.0,
        recv_timeout_s: float = 120.0,
    ) -> None:
        self._recv_timeout_s = float(recv_timeout_s)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((bind, int(port)))
        self._server.listen(1)
        self._server.settimeout(float(accept_timeout_s))
        self._conn: Optional[socket.socket] = None
        LOGGER.info("StreamFrameSource: waiting for publisher on %s:%d ...", bind, port)

    def _recv_exact(self, n: int) -> Optional[bytes]:
        assert self._conn is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def __iter__(self) -> Iterator[FrameItem]:
        conn, addr = self._server.accept()
        conn.settimeout(self._recv_timeout_s)
        self._conn = conn
        LOGGER.info("StreamFrameSource: publisher connected from %s", addr)
        n_frames = 0
        try:
            while True:
                header = self._recv_exact(_LEN.size)
                if header is None:
                    LOGGER.warning("StreamFrameSource: publisher disconnected mid-stream")
                    break
                (length,) = _LEN.unpack(header)
                if length == 0:
                    LOGGER.info("StreamFrameSource: end-of-stream after %d frames", n_frames)
                    break
                payload = self._recv_exact(int(length))
                if payload is None:
                    LOGGER.warning("StreamFrameSource: truncated frame payload")
                    break
                frame = pickle.loads(payload)
                n_frames += 1
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._conn = None
