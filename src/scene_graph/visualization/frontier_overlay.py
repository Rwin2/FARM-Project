"""Dock a live frontier-exploration viewer inside the offline viser window.

A stream publisher (FARM-Frontier's habitat_frontier_sim.py) attaches a small
``frontier_viz`` dict to the frames it sends. The offline driver pops that key
and hands it here; the overlay then (1) aligns the viser up-axis with the
stream world (habitat is Y-up) and (2) embeds the publisher's exploration
viewer page (textured scene, agent trail, frontiers, explored-area lighting)
in viser's own sidebar via ``gui.add_html`` — one window, FARM's native
interface, the exploration view docked in it.

Payload schema: ``{"viewer_port": int}`` (the port serving the viewer page on
the same host the browser uses to reach viser).
"""

from __future__ import annotations

import json
import logging
import os
import time

LOGGER = logging.getLogger(__name__)


class FrontierOverlay:
    def __init__(self, query_sink_path: str | None = None,
                 graph_port: int = 8093) -> None:
        self._up_set = False
        self._gui_added = False
        self._warned = False
        self._query_sink_path = query_sink_path
        self._query_hooked = False
        self._match_folder = None
        # Full-frame JPEG cache, index-aligned with the mapper's image ids
        # (~25 KB per 320x320 frame; capped). Lets a query candidate resolve
        # its best-view FULL image, not just the object crop.
        self._frame_jpegs: list = []
        self._frame_cache_cap = 4000
        self._suppress_mirror = False
        self._graph_port = int(graph_port)
        self._graph_server_started = False

    def cache_frame(self, rgb) -> None:
        try:
            if len(self._frame_jpegs) >= self._frame_cache_cap:
                return
            import io

            import numpy as np
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(
                np.asarray(rgb)[..., :3]).astype(np.uint8)).save(
                buf, format="JPEG", quality=88)
            self._frame_jpegs.append(buf.getvalue())
        except Exception:
            self._frame_jpegs.append(b"")

    # Scene-state keys the explorer's graph client consumes (see FARM-Frontier
    # farm_frontier/reasoning/graph_client.py). Everything else stays private.
    _GRAPH_KEYS = (
        "means", "active", "cov6", "object_id",
        "object_caption", "object_category", "object_supercategory",
        "region_ids", "region_labels", "region_centroids",
    )

    def _start_graph_server(self, visualizer) -> None:
        """Serve the live in-memory scene graph as JSON on 127.0.0.1:<port>,
        so the explorer reads FARM's memory directly instead of waiting for a
        displacement-gated .pt checkpoint."""
        if self._graph_server_started or self._graph_port <= 0:
            return
        import http.server
        import threading

        overlay_self = self

        def _tolist(v):
            if hasattr(v, "detach"):
                v = v.detach().cpu()
            if hasattr(v, "tolist"):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [_tolist(x) for x in v]
            return v

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/refresh"):
                    # Re-run the last panel query (results list, match
                    # buttons, highlight) WITHOUT re-triggering a search.
                    try:
                        overlay_self._suppress_mirror = True
                        visualizer._handle_query_clicked()
                        body = b"ok"
                    except Exception as exc:
                        body = f"refresh failed: {exc}".encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not self.path.startswith("/graph.json"):
                    self.send_error(404)
                    return
                state = getattr(visualizer, "_latest_scene_state", None)
                if not state:
                    self.send_error(503)
                    return
                out = {"t": time.time()}
                for k in FrontierOverlay._GRAPH_KEYS:
                    v = state.get(k)
                    if v is not None:
                        out[k] = _tolist(v)
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self._graph_port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._graph_server_started = True
        LOGGER.info("frontier overlay: live graph endpoint at http://127.0.0.1:%d/graph.json",
                    self._graph_port)

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

        if not self._up_set:
            try:
                server.scene.set_up_direction("+y")
            except Exception:
                pass
            self._up_set = True

        viewer_port = payload.get("viewer_port")
        if not self._gui_added and viewer_port:
            with server.gui.add_folder("Frontier exploration"):
                server.gui.add_html(
                    f'<iframe src="http://localhost:{int(viewer_port)}/" '
                    'style="width:100%;height:460px;border:0;border-radius:4px;background:#111">'
                    "</iframe>"
                    f'<a href="http://localhost:{int(viewer_port)}/" target="_blank" '
                    'style="color:#8ab4ff;font-size:12px">open full size</a>'
                )
            self._gui_added = True
            LOGGER.info("frontier overlay: exploration viewer docked in sidebar (:%d)", viewer_port)

        self._hook_query_panel(visualizer)
        try:
            self._start_graph_server(visualizer)
        except Exception as exc:
            if not self._graph_server_started:
                LOGGER.warning("frontier overlay: graph endpoint failed: %s", exc)
                self._graph_server_started = True  # do not retry every frame

    def _hook_query_panel(self, visualizer) -> None:
        """Wrap the native Query panel handler so every submitted query is also
        written (atomically) to the sink file, together with how many objects
        the native retrieval matched. The explorer polls that file: matches
        found = classical FARM answered it; none = it starts exploring."""
        if self._query_hooked or not self._query_sink_path:
            return
        if getattr(visualizer, "_query_input", None) is None:
            return  # query GUI not built yet; retry on the next frame

        sink = self._query_sink_path
        orig_run = visualizer._run_relational_query
        orig_clicked = visualizer._handle_query_clicked
        last: dict = {}

        def run_wrapped(query: str):
            out = orig_run(query)
            try:
                import numpy as np

                results = out[0] or []
                last["query"], last["n_results"] = query, len(results)
                # Serialize the panel's own ranked candidates (object_id,
                # composite score, caption) plus their 3D positions, so the
                # explorer can verify them (SAM3 + VLM) and navigate. Note the
                # composite score is RELATIVE (sem /= sem.max(): top ~ 1.0).
                state = getattr(visualizer, "_latest_scene_state", None) or {}
                means = np.asarray(state.get("means", np.zeros((0, 3))),
                                   dtype=float).reshape(-1, 3)
                ids = [int(v) for v in np.asarray(
                    state.get("object_id", np.arange(len(means)))).reshape(-1)]
                rgb_obs = state.get("rgb_observations") or []
                vp_ids = state.get("viewpoint_image_ids") or []
                img_pos = state.get("image_positions")

                def _to_np(v):
                    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

                def _best_crop_b64(idx):
                    """The same stored crop the panel shows on click: largest
                    'image' entry of the object's rgb_observations row."""
                    row = rgb_obs[idx] if idx < len(rgb_obs) else None
                    if not isinstance(row, list) or not row:
                        return None
                    best, best_a = None, -1
                    for e in row:
                        img = e.get("image") if isinstance(e, dict) else None
                        if img is None:
                            continue
                        img = _to_np(img)
                        a = int(img.shape[0]) * int(img.shape[1])
                        if a > best_a:
                            best, best_a = img, a
                    if best is None:
                        return None
                    import base64
                    import io

                    from PIL import Image

                    buf = io.BytesIO()
                    Image.fromarray(np.ascontiguousarray(best[..., :3]).astype(np.uint8)).save(
                        buf, format="JPEG", quality=88)
                    return base64.b64encode(buf.getvalue()).decode()

                images_meta = state.get("images")

                def _best_view_pos(idx):
                    """Camera position of FARM's own best viewing frame."""
                    row = vp_ids[idx] if idx < len(vp_ids) else None
                    if not row or img_pos is None:
                        return None
                    fid = int(row[0])
                    if not (0 <= fid < len(img_pos)) or img_pos[fid] is None:
                        return None
                    p = _to_np(img_pos[fid]).reshape(-1)[:3]
                    return [round(float(v), 4) for v in p]

                def _best_view_frame_b64(idx):
                    """FULL image of FARM's best viewing frame, from the
                    overlay's frame cache."""
                    row = vp_ids[idx] if idx < len(vp_ids) else None
                    if not row:
                        return None
                    fid = int(row[0])
                    if not (0 <= fid < len(self._frame_jpegs)):
                        return None
                    raw = self._frame_jpegs[fid]
                    if not raw:
                        return None
                    import base64
                    return base64.b64encode(raw).decode()

                def _best_view_pose(idx):
                    """Full 4x4 camera pose of that frame (orientation
                    included), from the per-frame images metadata."""
                    row = vp_ids[idx] if idx < len(vp_ids) else None
                    if not row or not isinstance(images_meta, list):
                        return None
                    fid = int(row[0])
                    if not (0 <= fid < len(images_meta)):
                        return None
                    pose = getattr(images_meta[fid], "pose", None)
                    if pose is None:
                        return None
                    T = _to_np(pose).reshape(4, 4)
                    return [[round(float(v), 6) for v in r] for r in T]

                cands = []
                for oid, score, caption in list(results)[:10]:
                    try:
                        idx = ids.index(int(oid))
                    except ValueError:
                        continue
                    cands.append({
                        "object_id": int(oid),
                        "score": round(float(score), 4),
                        "caption": str(caption or ""),
                        "pos_hab": [round(float(v), 4) for v in means[idx]],
                        "view_pos_hab": _best_view_pos(idx),
                        "view_pose_hab": _best_view_pose(idx),
                        "crop_jpeg_b64": _best_crop_b64(idx),
                        "view_frame_jpeg_b64": _best_view_frame_b64(idx),
                    })
                last["candidates"] = cands
            except Exception:
                pass
            return out

        def clicked_wrapped() -> None:
            try:
                query = visualizer._gui_get_value(visualizer._query_input).strip()
            except Exception:
                query = ""
            last.clear()
            orig_clicked()
            if not query:
                return
            n = int(last.get("n_results", 0)) if last.get("query") == query else 0
            record = {"query": query, "n_results": n, "t": time.time()}
            if last.get("query") == query:
                record["candidates"] = last.get("candidates", [])
            try:
                self._update_match_buttons(visualizer)
            except Exception as exc:
                LOGGER.warning("frontier overlay: match buttons failed: %s", exc)
            if self._suppress_mirror:
                # Auto-refresh (e.g. after object_found): update the panel
                # only; do not write the sink or a new search would start.
                self._suppress_mirror = False
                LOGGER.info("frontier overlay: panel refreshed (no mirror)")
                return
            try:
                tmp = sink + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(record, f)
                os.replace(tmp, sink)
                LOGGER.info("frontier overlay: query %r forwarded (n_results=%d)", query, n)
            except OSError as exc:
                LOGGER.warning("frontier overlay: could not write query sink: %s", exc)

        visualizer._run_relational_query = run_wrapped
        visualizer._handle_query_clicked = clicked_wrapped
        self._query_hooked = True
        LOGGER.info("frontier overlay: Query panel wired to %s", sink)

        self._last_for_buttons = last

    def _update_match_buttons(self, visualizer) -> None:
        """One clickable button per top match of the last query: clicking it
        behaves exactly like clicking the object's 3D box (highlight + stored
        image), so the ranked list and the boxes can be cross-referenced."""
        server = getattr(visualizer, "_server", None)
        if server is None:
            return
        cands = (getattr(self, "_last_for_buttons", None) or {}).get("candidates") or []
        if self._match_folder is not None:
            try:
                self._match_folder.remove()
            except Exception:
                pass
            self._match_folder = None
        if not cands:
            return
        # Keep strong references to the button handles (as FARM does for its
        # 3D box handles) or their callbacks can be garbage-collected.
        self._match_btns = []
        with server.gui.add_folder("Query matches") as folder:
            self._match_folder = folder
            for c in cands[:8]:
                label = (f"#{c['object_id']} ({c['score']:.2f}) "
                         f"{(c['caption'] or 'uncaptioned')[:30]}")
                btn = server.gui.add_button(label)
                self._match_btns.append(btn)

                @btn.on_click
                def _(_, oid=int(c["object_id"])):
                    try:
                        LOGGER.info("frontier overlay: match button -> object %d", oid)
                        visualizer._handle_object_click(oid)
                    except Exception as exc:
                        LOGGER.warning("match button click failed: %s", exc)


def wrap_frame_iterator(src_iter, get_visualizer, query_sink_path: str | None = None):
    """Pass-through frame iterator that pops ``frontier_viz`` payloads and
    forwards them to a FrontierOverlay. ``get_visualizer`` is a callable so the
    mapper's viser server can come up lazily."""
    overlay = FrontierOverlay(query_sink_path=query_sink_path)
    for item in src_iter:
        if isinstance(item, dict):
            payload = item.pop("frontier_viz", None)
            if payload is not None:
                overlay.handle(payload, get_visualizer())
            if item.get("rgb") is not None:
                overlay.cache_frame(item["rgb"])
        yield item
