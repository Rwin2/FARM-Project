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
        self._graph_port = int(graph_port)
        self._graph_server_started = False

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
                last["query"], last["n_results"] = query, len(out[0] or [])
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
        yield item
