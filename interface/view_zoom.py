"""Zoom shared by every sensor view (the pseudo-frames of all tabs and the detection event-cluster image).

The zoom is the displayed part of the sensor, in normalised texture coordinates (uv_min, uv_max). Scrolling over a
registered view zooms around the cursor (the sensor point under the cursor stays under it) and every view follows,
so all pages show the same region. Scrolling back out returns to the whole frame.
"""
from typing import Callable, List, Sequence, Tuple

import dearpygui.dearpygui as dpg

ZOOM_STEP = 1.15  # zoom factor per wheel notch
MIN_SPAN = 0.02   # smallest displayed part of the sensor (2 % of each axis, ~25 x 15 pixels)


def zoom_uv(uv_min: Sequence[float], uv_max: Sequence[float], fx: float, fy: float, factor: float,
            min_span: float = MIN_SPAN) -> Tuple[List[float], List[float]]:
    """Region after zooming by `factor` (> 1: in) around the point at fraction (fx, fy) of the view: that point keeps
    its place in the view, the region stays inside [0, 1] and keeps its aspect ratio. Returns the region unchanged
    when it would get smaller than `min_span`."""
    new_min, new_max = [], []
    for lo, hi, f in ((uv_min[0], uv_max[0], fx), (uv_min[1], uv_max[1], fy)):
        span = hi - lo
        new_span = min(1.0, span / factor)
        if new_span < min_span:
            return list(uv_min), list(uv_max)
        f = min(max(f, 0.0), 1.0)
        start = lo + f * span - f * new_span  # anchor point (lo + f * span) stays at fraction f
        start = min(max(start, 0.0), 1.0 - new_span)
        new_min.append(start)
        new_max.append(start + new_span)
    return new_min, new_max


class SharedZoom:
    def __init__(self) -> None:
        self.uv_min = [0.0, 0.0]
        self.uv_max = [1.0, 1.0]
        self._views: List[Tuple[str, Callable]] = []  # (item hit-tested under the mouse, apply(uv_min, uv_max))

    def register(self, hit_item: str, apply: Callable) -> None:
        """Adds a view: `apply(uv_min, uv_max)` shows the region; scrolling over `hit_item` zooms."""
        self._views.append((hit_item, apply))
        self._install_wheel_handler()
        apply(self.uv_min, self.uv_max)

    def _install_wheel_handler(self) -> None:
        if not dpg.does_item_exist("shared_zoom_handlers"):
            with dpg.handler_registry(tag="shared_zoom_handlers"):
                dpg.add_mouse_wheel_handler(callback=self._on_wheel)

    def _on_wheel(self, sender, app_data, user_data=None) -> None:
        wheel = float(app_data)
        if wheel == 0:
            return
        mx, my = dpg.get_mouse_pos(local=False)
        for hit_item, _ in self._views:
            if not dpg.does_item_exist(hit_item) or not dpg.is_item_visible(hit_item):
                continue  # e.g. on another tab
            x0, y0 = dpg.get_item_rect_min(hit_item)
            x1, y1 = dpg.get_item_rect_max(hit_item)
            if x1 <= x0 or y1 <= y0 or not (x0 <= mx <= x1 and y0 <= my <= y1):
                continue
            factor = ZOOM_STEP if wheel > 0 else 1.0 / ZOOM_STEP
            self.set(*zoom_uv(self.uv_min, self.uv_max, (mx - x0) / (x1 - x0), (my - y0) / (y1 - y0), factor))
            return

    def set(self, uv_min: Sequence[float], uv_max: Sequence[float]) -> None:
        self.uv_min, self.uv_max = list(uv_min), list(uv_max)
        for _, apply in self._views:
            try:
                apply(self.uv_min, self.uv_max)
            except Exception as e:  # one broken view must not stop the others
                print(f"[ZOOM] view update failed: {e}")


def overlay_transform(uv_min: Sequence[float], uv_max: Sequence[float], width: float, height: float):
    """Matrix for a draw node whose items are drawn in drawlist coordinates of the whole frame (width x height), so
    that they follow the zoomed image: p' = (p - uv_min * size) / span."""
    su = uv_max[0] - uv_min[0]
    sv = uv_max[1] - uv_min[1]
    return (dpg.create_scale_matrix([1.0 / su, 1.0 / sv, 1.0])
            * dpg.create_translation_matrix([-uv_min[0] * width, -uv_min[1] * height, 0.0]))
