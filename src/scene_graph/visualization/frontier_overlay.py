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

import logging

LOGGER = logging.getLogger(__name__)


class FrontierOverlay:
    def __init__(self) -> None:
        self._up_set = False
        self._gui_added = False
        self._warned = False

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
