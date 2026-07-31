"""Side-by-side frontier-exploration overlay for the offline viser view.

A stream publisher (e.g. FARM-Frontier's habitat_frontier_sim.py) may attach a
``frontier_viz`` dict to the frames it sends. The offline driver pops that key
and hands it here; the overlay draws a FrontierNet-style observer view — scene
mesh, agent frustum + trail, frontier frustums colored by information gain,
current goal — into the SAME viser server as the live scene graph, under a
root node translated by ``offset``. Both views therefore share one window,
one interactive sidebar, and one coordinate frame (identical orientation,
constant translation).

Payload schema (all poses 4x4 row-major lists, OpenCV cam-to-world, stream
world frame):
    offset:    [dx, dy, dz] translation of the overlay root (sent every time)
    glb_bytes: scene mesh as GLB bytes (sent once, first frame)
    agent:     current camera pose
    trail_pt:  [x, y, z] append-only agent trail point (optional)
    frontiers: [{"T": pose, "gain": float}, ...] current valid frontiers
    goal:      pose of the current goal frontier, or None
    fov/aspect: camera intrinsics for frustum rendering (optional)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

_ROOT = "/frontier_view"


class FrontierOverlay:
    def __init__(self) -> None:
        self._server = None
        self._glb_added = False
        self._trail: list = []
        self._frontier_handles: dict = {}
        self._goal_handle = None
        self._warned = False

    def _wxyz_pos(self, T: np.ndarray):
        from viser.transforms import SO3

        return SO3.from_matrix(np.asarray(T)[:3, :3]).wxyz, np.asarray(T)[:3, 3]

    @staticmethod
    def _gain_color(gain: float, max_gain: float):
        """Low gain -> dark red, high gain -> yellow (FrontierNet's convention)."""
        t = float(np.clip(gain / max(max_gain, 1e-6), 0.0, 1.0))
        return (int(120 + 135 * t), int(200 * t), 30)

    def handle(self, payload: dict, visualizer) -> None:
        try:
            self._handle(payload, visualizer)
        except Exception as exc:  # never break mapping because of the overlay
            if not self._warned:
                LOGGER.warning("frontier overlay disabled after error: %s", exc)
                self._warned = True

    def _handle(self, payload: dict, visualizer) -> None:
        server = getattr(visualizer, "_server", None) if visualizer is not None else None
        if server is None:
            return
        self._server = server
        scene = server.scene
        offset = np.asarray(payload.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)

        scene.add_frame(_ROOT, position=offset, show_axes=False)

        if not self._glb_added and payload.get("glb_bytes"):
            scene.add_glb(f"{_ROOT}/mesh", glb_data=payload["glb_bytes"], cast_shadow=False)
            self._glb_added = True
            LOGGER.info("frontier overlay: scene mesh added (%d bytes)", len(payload["glb_bytes"]))

        fov = float(payload.get("fov", 1.2))
        aspect = float(payload.get("aspect", 4.0 / 3.0))

        agent = payload.get("agent")
        if agent is not None:
            wxyz, pos = self._wxyz_pos(agent)
            scene.add_camera_frustum(
                f"{_ROOT}/agent", fov=fov, aspect=aspect, scale=0.35,
                line_width=2.5, color=(30, 100, 255), wxyz=wxyz, position=pos,
            )

        pt = payload.get("trail_pt")
        if pt is not None:
            self._trail.append(list(pt))
            if len(self._trail) >= 2:
                pts = np.asarray(self._trail, dtype=np.float32)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                scene.add_line_segments(
                    f"{_ROOT}/trail", points=segs, colors=(30, 100, 255), line_width=3.0,
                )

        frontiers = payload.get("frontiers")
        if frontiers is not None:
            max_gain = max([f["gain"] for f in frontiers], default=1.0)
            seen = set()
            for i, f in enumerate(frontiers):
                name = f"{_ROOT}/ft/{i}"
                seen.add(name)
                wxyz, pos = self._wxyz_pos(f["T"])
                self._frontier_handles[name] = scene.add_camera_frustum(
                    name, fov=fov, aspect=aspect, scale=0.25, line_width=2.0,
                    color=self._gain_color(f["gain"], max_gain), wxyz=wxyz, position=pos,
                )
            for name in list(self._frontier_handles):
                if name not in seen:
                    try:
                        self._frontier_handles.pop(name).remove()
                    except Exception:
                        pass

        goal = payload.get("goal")
        if goal is not None:
            _, pos = self._wxyz_pos(goal)
            self._goal_handle = scene.add_icosphere(
                f"{_ROOT}/goal", radius=0.16, color=(20, 60, 255), position=pos,
            )
        elif self._goal_handle is not None:
            try:
                self._goal_handle.remove()
            except Exception:
                pass
            self._goal_handle = None


def wrap_frame_iterator(src_iter, get_visualizer):
    """Pass-through frame iterator that pops ``frontier_viz`` payloads and
    forwards them to a FrontierOverlay. ``get_visualizer`` is a callable so the
    mapper's viser server can come up lazily."""
    overlay = FrontierOverlay()
    for item in src_iter:
        if isinstance(item, dict):
            payload = item.pop("frontier_viz", None)
            if payload is not None:
                overlay.handle(payload, get_visualizer())
        yield item
