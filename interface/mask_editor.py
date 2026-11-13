"""Drawing and editing the polygon masks (utils/masks.py) on the Preprocessing pseudo-frame view.

Draw mode ("Draw new mask"): left click adds a point; clicking the first point again or right click closes the
polygon (3 points at least); Esc cancels.
Otherwise: left click inside a mask selects it; drag a vertex of the selected mask to move it; right click a vertex
to delete it (3 points are kept).
"""
import dearpygui.dearpygui as dpg

from utils.masks import Mask, contains

HIT_PX = 8  # screen distance for grabbing a vertex / closing a polygon

_COLORS = {"exclude": (255, 90, 90, 255), "keep": (90, 230, 120, 255)}
_DISABLED = (150, 150, 150, 200)
_DRAFT = (255, 220, 60, 255)


class PolygonMaskEditor:
    def __init__(self, drawlist, node, masks, geometry, zoom, on_change, on_message):
        """drawlist / node: the view and the draw node the masks are drawn in; masks: the list of Mask edited in place;
        geometry(): (sensor w, sensor h, display w, display h); zoom: the SharedZoom of the views;
        on_change(committed): masks changed (committed: an edit is finished, the events must be filtered again);
        on_message(text): user feedback."""
        self.drawlist, self.node = drawlist, node
        self.masks = masks
        self.geometry = geometry
        self.zoom = zoom
        self.on_change = on_change
        self.on_message = on_message
        self.drawing = False
        self.draft = []
        self.selected = -1
        self._drag = None  # (mask index, vertex index)
        self._cursor = None  # sensor position of the mouse while drawing
        if not dpg.does_item_exist("mask_editor_handlers"):
            with dpg.handler_registry(tag="mask_editor_handlers"):
                dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_left)
                dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self._on_right)
                dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_release)
                dpg.add_mouse_move_handler(callback=self._on_move)
                dpg.add_key_press_handler(key=dpg.mvKey_Escape, callback=self._on_escape)

    # ------------------------------------------------------------------ coordinates
    def _rect(self):
        if not dpg.does_item_exist(self.drawlist) or not dpg.is_item_visible(self.drawlist):
            return None
        (x0, y0), (x1, y1) = dpg.get_item_rect_min(self.drawlist), dpg.get_item_rect_max(self.drawlist)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def _mouse_sensor(self):
        """Mouse position in sensor pixels, or None when the mouse is not over the view."""
        rect = self._rect()
        if rect is None:
            return None
        mx, my = dpg.get_mouse_pos(local=False)
        x0, y0, x1, y1 = rect
        if not (x0 <= mx <= x1 and y0 <= my <= y1):
            return None
        sw, sh, _, _ = self.geometry()
        (u0, v0), (u1, v1) = self.zoom.uv_min, self.zoom.uv_max
        u = u0 + (mx - x0) / (x1 - x0) * (u1 - u0)
        v = v0 + (my - y0) / (y1 - y0) * (v1 - v0)
        return [u * sw, v * sh]

    def _to_view(self, p):
        """Sensor pixel -> drawlist coordinates (with the shared zoom)."""
        sw, sh, dw, dh = self.geometry()
        (u0, v0), (u1, v1) = self.zoom.uv_min, self.zoom.uv_max
        return [(p[0] / sw - u0) / (u1 - u0) * dw, (p[1] / sh - v0) / (v1 - v0) * dh]

    def _view_scale(self):
        """Screen pixels per sensor pixel (for the hit distances)."""
        sw, _, dw, _ = self.geometry()
        return dw / sw / (self.zoom.uv_max[0] - self.zoom.uv_min[0])

    def _vertex_at(self, mask_index, p):
        if not (0 <= mask_index < len(self.masks)):
            return -1
        r = HIT_PX / self._view_scale()
        best, best_d = -1, r * r
        for i, (x, y) in enumerate(self.masks[mask_index].points):
            d = (x - p[0]) ** 2 + (y - p[1]) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    # ------------------------------------------------------------------ actions
    def start_drawing(self):
        self.drawing, self.draft, self._drag = True, [], None
        self.on_message("Drawing a mask: left click adds points; click the first point or right click to close it; "
                        "Esc cancels.")
        self.redraw()

    def cancel_drawing(self):
        if self.drawing:
            self.drawing, self.draft = False, []
            self.on_message("Mask drawing cancelled.")
            self.redraw()

    def select(self, index):
        self.selected = index if 0 <= index < len(self.masks) else -1
        self.redraw()

    def _finish(self):
        if len(self.draft) >= 3:
            names = {m.name for m in self.masks}
            n = len(self.masks) + 1
            while f"Mask {n}" in names:
                n += 1
            self.masks.append(Mask(name=f"Mask {n}", points=[list(p) for p in self.draft]))
            self.selected = len(self.masks) - 1
            self.on_message(f"Mask {n} added ({len(self.draft)} points).")
            self.drawing, self.draft = False, []
            self.on_change(True)
        else:
            self.cancel_drawing()
        self.redraw()

    # ------------------------------------------------------------------ mouse / keyboard
    def _on_left(self, sender=None, app_data=None, user_data=None):
        p = self._mouse_sensor()
        if p is None:
            return
        if self.drawing:
            if len(self.draft) >= 3 and self._vertex_at_draft_start(p):
                self._finish()
                return
            self.draft.append(p)
            self.redraw()
            return
        v = self._vertex_at(self.selected, p)
        if v >= 0:
            self._drag = (self.selected, v)
            return
        hit = next((i for i in range(len(self.masks) - 1, -1, -1) if contains(self.masks[i].points, *p)), -1)
        if hit != self.selected:
            self.selected = hit
            self.on_change(False)
        self.redraw()

    def _vertex_at_draft_start(self, p):
        x, y = self.draft[0]
        r = HIT_PX / self._view_scale()
        return (x - p[0]) ** 2 + (y - p[1]) ** 2 <= r * r

    def _on_move(self, sender=None, app_data=None, user_data=None):
        if self._drag is not None:
            p = self._mouse_sensor()
            if p is not None:
                sw, sh, _, _ = self.geometry()
                mi, vi = self._drag
                self.masks[mi].points[vi] = [min(max(p[0], 0.0), sw), min(max(p[1], 0.0), sh)]
                self.redraw()
        elif self.drawing and self.draft:
            self._cursor = self._mouse_sensor()
            self.redraw()

    def _on_release(self, sender=None, app_data=None, user_data=None):
        if self._drag is not None:
            self._drag = None
            self.on_change(True)

    def _on_right(self, sender=None, app_data=None, user_data=None):
        p = self._mouse_sensor()
        if p is None:
            return
        if self.drawing:
            self._finish()
            return
        v = self._vertex_at(self.selected, p)
        if v >= 0:
            pts = self.masks[self.selected].points
            if len(pts) > 3:
                del pts[v]
                self.on_change(True)
            else:
                self.on_message("A mask keeps 3 points at least (delete the mask instead).")
            self.redraw()

    def _on_escape(self, sender=None, app_data=None, user_data=None):
        self.cancel_drawing()

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        if not dpg.does_item_exist(self.node):
            return
        dpg.delete_item(self.node, children_only=True)
        for i, m in enumerate(self.masks):
            if len(m.points) < 2:
                continue
            pts = [self._to_view(p) for p in m.points]
            color = _COLORS.get(m.mode, _COLORS["exclude"]) if m.enabled else _DISABLED
            selected = i == self.selected
            dpg.draw_polyline(pts + [pts[0]], color=color, thickness=3.0 if selected else 1.5, parent=self.node)
            dpg.draw_text(pts[0], m.name + ("" if m.enabled else " (off)"), color=color, size=14, parent=self.node)
            if selected:
                for x, y in pts:
                    dpg.draw_rectangle([x - 4, y - 4], [x + 4, y + 4], color=(255, 255, 255, 255),
                                       fill=color, parent=self.node)
        if self.drawing and self.draft:
            pts = [self._to_view(p) for p in self.draft]
            if len(pts) >= 2:
                dpg.draw_polyline(pts, color=_DRAFT, thickness=2.0, parent=self.node)
            if self._cursor is not None:
                dpg.draw_line(pts[-1], self._to_view(self._cursor), color=(255, 220, 60, 150), thickness=1.5,
                              parent=self.node)
            dpg.draw_circle(pts[0], 6.0 if len(pts) >= 3 else 4.0, color=_DRAFT, fill=_DRAFT, parent=self.node)
            for q in pts[1:]:
                dpg.draw_circle(q, 3.0, color=_DRAFT, fill=_DRAFT, parent=self.node)
