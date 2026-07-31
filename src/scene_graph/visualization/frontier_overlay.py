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
world frame; all geometry in the SAME stream world frame — no glTF axis
conversion anywhere, so the overlay stays aligned with the scene graph):
    offset:    [dx, dy, dz] translation of the overlay root (sent every time)
    mesh:      {"vertices": Nx3 f32, "faces": Mx3 u32} GT mesh, sent once,
               drawn dark = the *unknown* scene (FrontierNet convention)
    occ_pts:   Kx3 f32 observed occupied voxel centers, periodic — drawn as
               bright points so the *explored* part lights up
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
        self._mesh_added = False
        self._up_set = False
        self._trail: list = []
        self._frontier_handles: dict = {}
        self._goal_handle = None
        self._rays_handle = None
        self._warned = False

    def _wxyz_pos(self, T: np.ndarray):
        from viser.transforms import SO3

        return SO3.from_matrix(np.asarray(T)[:3, :3]).wxyz, np.asarray(T)[:3, 3]

    @staticmethod
    def _gain_color(gain: float, max_gain: float):
        """FrontierNet gif convention: high gain -> orange, low gain -> yellow."""
        t = float(np.clip(gain / max(max_gain, 1e-6), 0.0, 1.0))
        return (255, int(225 - 120 * t), int(90 - 75 * t))

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

        if not self._up_set:
            # Stream data (habitat) is Y-up; make the whole viewer agree so
            # both the scene graph and this overlay stand upright.
            try:
                scene.set_up_direction("+y")
            except Exception:
                pass
            self._up_set = True

        if not self._mesh_added and payload.get("mesh") is not None:
            m = payload["mesh"]
            # Real textured GLB, pre-dimmed + unlit ("unknown" brightness).
            # viser's glb path applies no axis conversion; the wxyz node
            # rotation places the file-frame mesh into the stream world frame.
            scene.add_glb(
                f"{_ROOT}/mesh",
                glb_data=m["glb"],
                wxyz=tuple(float(v) for v in m["wxyz"]),
                cast_shadow=False,
            )
            self._mesh_added = True
            LOGGER.info(
                "frontier overlay: dimmed textured mesh added (%d bytes)", len(m["glb"])
            )

        occ = payload.get("occ_pts")
        if occ is not None and len(occ):
            pts = np.asarray(occ, dtype=np.float32)
            colors = payload.get("occ_colors")
            if colors is None:
                colors = np.full((len(pts), 3), (200, 200, 200), dtype=np.uint8)
            # seen = a little lighter than unknown: full-brightness texture
            # colors over the dimmed base mesh.
            scene.add_point_cloud(
                f"{_ROOT}/explored", points=pts, colors=np.asarray(colors, dtype=np.uint8),
                point_size=0.07, point_shape="circle",
            )

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
                # time-gradient trajectory (gif style): purple -> green
                t = np.linspace(0.0, 1.0, len(segs))[:, None]
                seg_col = ((1 - t) * np.array([150, 30, 200]) + t * np.array([40, 220, 120])).astype(np.uint8)
                scene.add_line_segments(
                    f"{_ROOT}/trail", points=segs,
                    colors=np.repeat(seg_col[:, None, :], 2, axis=1), line_width=4.0,
                )

        frontiers = payload.get("frontiers")
        agent_pos = None if payload.get("agent") is None else np.asarray(payload["agent"])[:3, 3]
        if frontiers is not None:
            max_gain = max([f["gain"] for f in frontiers], default=1.0)
            seen = set()
            ray_pts = []
            for i, f in enumerate(frontiers):
                name = f"{_ROOT}/ft/{i}"
                seen.add(name)
                wxyz, pos = self._wxyz_pos(f["T"])
                self._frontier_handles[name] = scene.add_camera_frustum(
                    name, fov=fov, aspect=aspect, scale=0.25, line_width=2.0,
                    color=self._gain_color(f["gain"], max_gain), wxyz=wxyz, position=pos,
                )
                if agent_pos is not None:
                    ray_pts.append([agent_pos, pos])
            for name in list(self._frontier_handles):
                if name not in seen:
                    try:
                        self._frontier_handles.pop(name).remove()
                    except Exception:
                        pass
            # thin light rays agent -> frontiers (the gif's graph edges)
            if ray_pts:
                self._rays_handle = scene.add_line_segments(
                    f"{_ROOT}/rays", points=np.asarray(ray_pts, dtype=np.float32),
                    colors=(225, 225, 225), line_width=1.0,
                )
            elif self._rays_handle is not None:
                try:
                    self._rays_handle.remove()
                except Exception:
                    pass

        goal = payload.get("goal")
        if goal is not None:
            # next-goal frontier: enlarged frustum (gif convention)
            wxyz, pos = self._wxyz_pos(goal)
            self._goal_handle = scene.add_camera_frustum(
                f"{_ROOT}/goal", fov=fov, aspect=aspect, scale=0.55,
                line_width=3.5, color=(255, 70, 0), wxyz=wxyz, position=pos,
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
