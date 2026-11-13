"""Timeline under the sensor view of every page: the displayed time step of the recording, shared by all pages.

The bar spans the recording (the events shown by Preprocessing, after the time crop), one accumulation step per
frame. Colours: blue where there is only input data, green where the current detection results cover the
recording, red where the current tracking results do (dark red: tracks of an earlier detection run): only the part
actually processed (a stopped run covers less than asked). Runs start at the start mark, or at the displayed step
when none is set, and go to the end mark, or to the end of the recording unless stopped. A yellow strip shows the
marks.

Mouse: click or drag on the bar. Keys (not while typing in a field): Left / Right one step (Shift: 10),
Home / End, Space play / pause, I / O set the start / end of the range at the displayed step.

Playback always draws PLAYBACK_FPS frames per second; the speed, in Hz, divides how fast recording time goes
(1 Hz: real time, 10000 Hz: one second of recording plays in 10000 s): each drawn frame, the shown window slides
by the real time since the previous one / speed (a fraction of a step when slow; the speed holds when a page
draws slower than PLAYBACK_FPS), and snaps back to the nearest step on pause.

Each page shows the bar and its status log in a tile of their own under the views (build_tile), so they stay
visible whichever view sub-tab is open.
"""
from __future__ import annotations

import math
import time
import weakref
from typing import Callable, Optional

import numpy as np
import dearpygui.dearpygui as dpg

BAR_HEIGHT = 30
_PAD = 6           # free space left and right of the bar (the handle stays visible at both ends)
_BAR_Y = (9, 23)   # top and bottom of the coloured bar
SPEED_TAG = "timeline_speed"
SPEED_DEFAULT = 20.0                # Hz: recording time / 20 (2 ms steps: 25 steps per second)
SPEED_MIN, SPEED_MAX = 1e-3, 1e9
PLAYBACK_FPS = 24
TILE_HEIGHT = 155                   # tile of the bar and the status log, under the views
VIEW_HEIGHT = 950 - 16 - TILE_HEIGHT - 6  # the views above it, in the 950 px high visualization column

COL_BG = (45, 45, 45, 255)
COL_DATA = (60, 120, 220, 255)
COL_DETECTION = (60, 190, 90, 255)
COL_TRACKING = (225, 60, 60, 255)
COL_TRACKING_OLD = (120, 55, 55, 255)
COL_TRACKING_WORK = (225, 60, 60, 150)  # batch being tracked
COL_PROGRESS_TEXT = (255, 200, 80)
COL_RANGE = (255, 210, 80, 255)
COL_HANDLE = (245, 245, 245, 255)


def _time_extent(tracks) -> Optional[tuple[float, float]]:
    """(first, last) finite time of a list of per-track time arrays (or PackedTracks), or None."""
    if tracks is None:
        return None
    if hasattr(tracks, "finite_min_max"):
        lo, hi = tracks.finite_min_max()
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        return (float(lo[ok].min()), float(hi[ok].max())) if ok.any() else None
    los, his = [], []
    for a in tracks:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            los.append(a.min())
            his.append(a.max())
    return (float(min(los)), float(max(his))) if los else None


def _field(out, name):
    v = getattr(out, name, None)
    if v is None and isinstance(out, dict):
        v = out.get(name)
    return v


class Timeline:
    def __init__(self, preprocessing_tab) -> None:
        self.pre = preprocessing_tab
        self.det = None
        self.trk = None
        self.start: Optional[float] = None  # range of the recording the steps run on (us); None: from the start
        self.end: Optional[float] = None    # None: to the end
        self.playing = False
        self._last_tick = 0.0
        self._due = 0.0                     # time the next playback frame is due
        self._drag: Optional[str] = None    # bar being dragged
        self._bars: list[str] = []
        self._range_texts: list[str] = []
        self._drawn: dict[str, tuple] = {}  # signature of what each bar shows
        self._extents: dict[tuple, Optional[tuple]] = {}
        self._input_items: dict[int, bool] = {}
        self._range_listeners: list[Callable[[], None]] = []
        self._text_sig = None
        # runs in progress (set from their threads): (start, done until, planned end, being processed until, text)
        self.live: dict[str, Optional[tuple]] = {"detection": None, "tracking": None}
        self._progress_texts: list[str] = []
        self._progress_sig = None

    def attach(self, detection_tab=None, tracking_tab=None) -> None:
        self.det = detection_tab or self.det
        self.trk = tracking_tab or self.trk

    def add_range_listener(self, callback: Callable[[], None]) -> None:
        self._range_listeners.append(callback)

    # ------------------------------------------------------------------ step grid
    def events(self):
        pre = self.pre
        return pre.filtered_events if pre.filtered_events is not None else pre.raw_events

    def grid(self) -> Optional[tuple[float, float, int]]:
        """(t0, step, K): frame k shows the events of (t0 + k * step, t0 + (k + 1) * step], k = 0 .. K - 1."""
        ev = self.events()
        if ev is None or ev[2] is None or len(ev[2]) == 0:
            return None
        step = float(self.pre.params.get("accumulation_time_ms", 0)) * 1000.0
        if not step > 0:
            return None
        T = ev[2]
        t0 = float(T[0])  # events are time sorted
        return t0, step, max(1, int(math.ceil((float(T[-1]) - t0) / step)))

    def index(self, grid=None) -> Optional[int]:
        g = grid or self.grid()
        if g is None:
            return None
        t0, step, K = g
        if self.pre.display_t0 is None:
            return 0
        return min(max(int(round((float(self.pre.display_t0) - t0) / step)), 0), K - 1)

    def seek(self, k: int) -> None:
        """Shows frame k (clamped) on every page."""
        g = self.grid()
        if g is None:
            return
        t0, step, K = g
        k = min(max(int(k), 0), K - 1)
        self.seek_time(t0 + k * step)

    def seek_time(self, t: float) -> None:
        """Shows the window (t, t + step] on every page (t need not be on the step grid: playback)."""
        if self.pre.display_t0 is not None and float(self.pre.display_t0) == t:
            return
        self.pre.display_t0 = float(t)
        self.pre.update_pseudoframe_from_data()

    def step(self, n: int) -> None:
        k = self.index()
        if k is not None:
            self.seek(k + n)

    # ------------------------------------------------------------------ range of the recording to process
    def run_steps(self, grid=None) -> Optional[tuple[float, float, int, bool]]:
        """(start, end, number of steps, whole recording?) of the next run: from the start mark, or the displayed
        step when no start is set, to the end mark, or the end of the recording when no end is set (the run can be
        stopped before). None without data, or when the displayed step is after the end mark."""
        g = grid or self.grid()
        if g is None:
            return None
        t0, step, K = g
        ks = self.index(g) if self.start is None else min(max(int(round((self.start - t0) / step)), 0), K - 1)
        ke = K if self.end is None else min(max(int(round((self.end - t0) / step)), 0), K)
        if ke <= ks:
            return None
        return t0 + ks * step, t0 + ke * step, ke - ks, ks == 0 and ke == K

    def set_start(self, *_):
        g, k = self.grid(), self.index()
        if g is None:
            return
        self.start = g[0] + k * g[1]
        if self.end is not None and self.end <= self.start:
            self.end = None
        self._range_changed()

    def set_end(self, *_):
        g, k = self.grid(), self.index()
        if g is None:
            return
        self.end = g[0] + (k + 1) * g[1]  # the displayed step is included
        if self.start is not None and self.start >= self.end:
            self.start = None
        self._range_changed()

    def clear_range(self, *_):
        if self.start is None and self.end is None:
            return
        self.start = self.end = None
        self._range_changed()

    def forget_range(self) -> None:
        """New recording: the range of the previous one does not apply."""
        self.start = self.end = None
        self.playing = False
        self._update_range_texts()

    def _range_changed(self) -> None:
        self._update_range_texts()
        self.pre.log_status(self.range_text())
        for cb in self._range_listeners:
            try:
                cb()
            except Exception as e:
                print(f"[TIMELINE] range listener failed: {e}")

    def range_text(self) -> str:
        r = self.run_steps()
        if r is None:
            if self.grid() is None:
                return "No data loaded."
            return "Nothing to run: the displayed step is after the end mark (go back, or set a start)."
        frm = "the start mark" if self.start is not None else "the displayed step"
        to = "the end mark" if self.end is not None else "the end of the recording (or Stop)"
        return f"Next run: from {frm} to {to}, t = [{r[0]:.0f} .. {r[1]:.0f}] us, {r[2]} time steps."

    def _update_range_texts(self) -> None:
        text = self.range_text()
        for tag in self._range_texts:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, text)

    # ------------------------------------------------------------------ results coverage
    def _extent(self, kind: str, out) -> Optional[tuple[float, float]]:
        r = _field(out, "time_range")
        if r is not None:
            return float(r[0]), float(r[1])
        key = (kind, id(out))
        if key not in self._extents:  # loaded results: from the first to the last point
            self._extents = {k: v for k, v in self._extents.items() if k[0] != kind}  # one result per kind
            try:
                self._extents[key] = _time_extent(_field(out, "t_clust" if kind == "detection" else "t_plot"))
            except Exception:
                self._extents[key] = None
        return self._extents[key]

    def detection_range(self, out=None):
        out = getattr(self.det, "last_result", None) if out is None else out
        return None if out is None else self._extent("detection", out)

    def tracking_range(self):
        """(start, end, from the current detection results?) of the tracking results, or None."""
        out = getattr(self.trk, "last_result", None)
        if out is None:
            return None
        r = self._extent("tracking", out)
        if r is None:
            return None
        src = getattr(self.trk, "_source_detection", None)
        current = True
        if src is not None:
            det_out = getattr(self.det, "last_result", None)
            current = (src() if isinstance(src, weakref.ref) else src) is det_out
        return r[0], r[1], current

    # ------------------------------------------------------------------ runs in progress
    def progress(self, kind: str, start: float, done: float, end: float, working: Optional[float] = None,
                 text: str = "") -> None:
        """A run of `kind` ("detection" / "tracking") in progress, called from its thread: [start, done] processed,
        [done, working] being processed, planned up to `end`. The bars show it until progress_end."""
        self.live[kind] = (float(start), float(done), float(end), None if working is None else float(working), text)

    def progress_end(self, kind: str) -> None:
        self.live[kind] = None

    def _update_progress_texts(self) -> None:
        text = "   ".join(v[4] for v in self.live.values() if v is not None and v[4])
        for tag in self._progress_texts:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, text)

    # ------------------------------------------------------------------ widgets
    def build_tile(self, name: str, time_info_tag: str, log_tag: str, status_tag: str = "") -> None:
        """The tile under the views of a page, in the current container: the bar, its controls and the page's
        status log (`log_tag`, a read-only multiline text). `status_tag`: the pseudo-frame status text."""
        with dpg.child_window(height=TILE_HEIGHT, border=True, no_scrollbar=True, tag=f"timeline_tile_{name}"):
            self._build_bar(name, time_info_tag, status_tag)
            dpg.add_input_text(tag=log_tag, multiline=True, readonly=True, width=-1, height=-1)

    def _build_bar(self, name: str, time_info_tag: str, status_tag: str) -> None:
        self._install_handlers()
        if not dpg.does_item_exist(SPEED_TAG):
            with dpg.value_registry():
                dpg.add_double_value(tag=SPEED_TAG, default_value=SPEED_DEFAULT)
        bar = f"timeline_bar_{name}"
        width = self.pre.display_size()[0]
        dpg.add_drawlist(width=width, height=BAR_HEIGHT, tag=bar)
        self._bars.append(bar)
        self.pre.register_geometry_listener(lambda w, h, dw, dh, bar=bar: self._resize(bar, dw))
        with dpg.group(horizontal=True):
            if time_info_tag:
                dpg.add_text("", tag=time_info_tag)
            dpg.add_text("  data", color=COL_DATA[:3])
            dpg.add_text("detected", color=COL_DETECTION[:3])
            dpg.add_text("tracked", color=COL_TRACKING[:3])
            dpg.add_text("processing range", color=COL_RANGE[:3])
            if status_tag:
                dpg.add_text("", tag=status_tag, color=(150, 150, 150))
            tag = f"timeline_progress_{name}"
            dpg.add_text("", tag=tag, color=COL_PROGRESS_TEXT)
            self._progress_texts.append(tag)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Set start (I)", callback=self.set_start)
            dpg.add_button(label="Set end (O)", callback=self.set_end)
            dpg.add_button(label="Clear marks", callback=self.clear_range)
            speed = dpg.add_input_double(label="Hz", source=SPEED_TAG, width=90, min_value=SPEED_MIN,
                                         max_value=SPEED_MAX, min_clamped=True, max_clamped=True, step=0,
                                         format="%.6g", tag=f"timeline_speed_{name}")
            with dpg.tooltip(speed):
                dpg.add_text(f"Playback speed: recording time divided by this value (1 Hz: real time, 10000 Hz:\n"
                             f"1 s of recording lasts 10000 s). Always drawn at {PLAYBACK_FPS} fps.\n\n"
                             "Space: play / pause   Left / Right: one step (Shift: 10)   Home / End\n"
                             "I / O: set start / end   click or drag the bar")
            tag = f"timeline_range_{name}"
            dpg.add_text(self.range_text(), tag=tag)
            self._range_texts.append(tag)

    def _resize(self, bar: str, width: int) -> None:
        if dpg.does_item_exist(bar):
            dpg.configure_item(bar, width=width)
            self._drawn.pop(bar, None)

    def _x(self, t: float, g, width: float) -> float:
        t0, step, K = g
        return _PAD + (t - t0) / (K * step) * (width - 2 * _PAD)

    def _draw(self, bar: str, g, width: float) -> None:
        dpg.delete_item(bar, children_only=True)
        y0, y1 = _BAR_Y
        if g is None:
            dpg.draw_rectangle((_PAD, y0), (width - _PAD, y1), fill=COL_BG, color=COL_BG, parent=bar)
            return
        t0, step, K = g
        t_end = t0 + K * step
        x = lambda t: self._x(min(max(t, t0), t_end), g, width)  # noqa: E731
        dpg.draw_rectangle((x(t0), y0), (x(t_end), y1), fill=COL_DATA, color=COL_DATA, parent=bar)
        trk = self.tracking_range()
        det = self.detection_range()
        live_det, live_trk = self.live["detection"], self.live["tracking"]
        if live_det is not None:  # running: the steps done so far (the results it replaces are not shown)
            self._draw_live(bar, x, live_det, COL_DETECTION, None)
        elif det is not None:
            dpg.draw_rectangle((x(det[0]), y0), (max(x(det[1]), x(det[0]) + 1), y1), fill=COL_DETECTION,
                               color=COL_DETECTION, parent=bar)
        if live_trk is not None:
            self._draw_live(bar, x, live_trk, COL_TRACKING, COL_TRACKING_WORK)
        elif trk is not None:  # tracks of an earlier detection run: thinner and dark
            col, (ya, yb) = (COL_TRACKING, (y0, y1)) if trk[2] else (COL_TRACKING_OLD, (y0 + 4, y1 - 4))
            dpg.draw_rectangle((x(trk[0]), ya), (max(x(trk[1]), x(trk[0]) + 1), yb), fill=col, color=col,
                               parent=bar)
        r = self.run_steps(g)
        if r is not None and (self.start is not None or self.end is not None):  # the marks set
            xs, xe = x(r[0]), x(r[1])
            dpg.draw_rectangle((xs, 2), (xe, 6), fill=COL_RANGE, color=COL_RANGE, parent=bar)
            for xx in (xs, xe):
                dpg.draw_line((xx, 2), (xx, y1 + 2), color=COL_RANGE, thickness=2, parent=bar)
        k = self.index(g)
        if k is not None:  # the window shown (between two steps while playing)
            ts = float(self.pre.display_t0) if self.pre.display_t0 is not None else t0
            a, b = x(ts), x(ts + step)
            if b - a < 4:
                a, b = (a + b) / 2 - 2, (a + b) / 2 + 2
            dpg.draw_rectangle((a, 0), (b, BAR_HEIGHT - 1), fill=COL_HANDLE, color=(20, 20, 20, 255), rounding=2,
                               parent=bar)

    @staticmethod
    def _draw_live(bar, x, live, col, col_work) -> None:
        """A run in progress: the planned interval outlined, filled up to what is done (and, lighter, what is
        being processed)."""
        start, done, end, working, _ = live
        y0, y1 = _BAR_Y
        planned = col[:3] + (70,)
        dpg.draw_rectangle((x(start), y0), (x(end), y1), fill=planned, color=planned, parent=bar)
        if working is not None and working > done:
            dpg.draw_rectangle((x(done), y0), (x(working), y1), fill=col_work, color=col_work, parent=bar)
        if done > start:
            dpg.draw_rectangle((x(start), y0), (max(x(done), x(start) + 1), y1), fill=col, color=col, parent=bar)
        dpg.draw_line((x(end), y0 - 2), (x(end), y1 + 2), color=col, thickness=2, parent=bar)  # planned end

    def _signature(self, g, width):
        det_out = getattr(self.det, "last_result", None)
        trk_out = getattr(self.trk, "last_result", None)
        return (g, width, self.pre.display_t0, self.start, self.end, id(det_out), self.detection_range(),
                id(trk_out), self.tracking_range(), self.live["detection"], self.live["tracking"])

    def refresh(self) -> None:
        """Redraws the shown bars whose content changed (cheap when nothing did: called every frame)."""
        live = (self.live["detection"], self.live["tracking"])
        if live != self._progress_sig:
            self._progress_sig = live
            self._update_progress_texts()
        g = None
        for bar in self._bars:
            if not dpg.does_item_exist(bar) or not dpg.is_item_visible(bar):
                continue
            if g is None:
                g = (self.grid(),)
            width = float(dpg.get_item_width(bar) or 0)
            sig = self._signature(g[0], width)
            if self._drawn.get(bar) != sig:
                self._drawn[bar] = sig
                self._draw(bar, g[0], width)
                if self._text_sig != (g[0],) + sig[2:5]:  # the next run starts at the displayed step
                    self._text_sig = (g[0],) + sig[2:5]
                    self._update_range_texts()

    # ------------------------------------------------------------------ input
    def _install_handlers(self) -> None:
        if dpg.does_item_exist("timeline_handlers"):
            return
        with dpg.handler_registry(tag="timeline_handlers"):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_click)
            for key in (dpg.mvKey_Left, dpg.mvKey_Right, dpg.mvKey_Home, dpg.mvKey_End, dpg.mvKey_Spacebar,
                        dpg.mvKey_I, dpg.mvKey_O):
                dpg.add_key_press_handler(key=key, callback=self._on_key, user_data=key)

    def _bar_under_mouse(self) -> Optional[str]:
        mx, my = dpg.get_mouse_pos(local=False)
        for bar in self._bars:
            if not dpg.does_item_exist(bar) or not dpg.is_item_visible(bar):
                continue
            (x0, y0), (x1, y1) = dpg.get_item_rect_min(bar), dpg.get_item_rect_max(bar)
            if x1 > x0 and y1 > y0 and x0 <= mx <= x1 and y0 <= my <= y1:
                return bar
        return None

    def _seek_mouse(self, bar: str) -> None:
        g = self.grid()
        if g is None or not dpg.does_item_exist(bar):
            return
        x0 = dpg.get_item_rect_min(bar)[0]
        width = float(dpg.get_item_rect_max(bar)[0] - x0)
        if width <= 2 * _PAD:
            return
        f = (dpg.get_mouse_pos(local=False)[0] - x0 - _PAD) / (width - 2 * _PAD)
        self.seek(int(math.floor(f * g[2])))

    def _on_click(self, sender=None, app_data=None, user_data=None) -> None:
        bar = self._bar_under_mouse()
        if bar is not None:
            self._drag = bar
            self._seek_mouse(bar)

    def _typing(self) -> bool:
        """True while a text or number field has the keyboard."""
        for item in dpg.get_all_items():
            is_input = self._input_items.get(item)
            if is_input is None:
                t = dpg.get_item_type(item)
                is_input = self._input_items[item] = ("mvInput" in t or "mvDrag" in t or "mvSlider" in t)
            if is_input and dpg.is_item_active(item):
                return True
        return False

    def _on_key(self, sender=None, app_data=None, user_data=None) -> None:
        if self._typing():
            return
        key = user_data
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        if key == dpg.mvKey_Spacebar:
            self.toggle_play()
        elif key == dpg.mvKey_Left:
            self.step(-10 if shift else -1)
        elif key == dpg.mvKey_Right:
            self.step(10 if shift else 1)
        elif key == dpg.mvKey_Home:
            self.seek(0)
        elif key == dpg.mvKey_End:
            g = self.grid()
            if g is not None:
                self.seek(g[2] - 1)
        elif key == dpg.mvKey_I:
            self.set_start()
        elif key == dpg.mvKey_O:
            self.set_end()

    def toggle_play(self) -> None:
        g = self.grid()
        if g is None:
            self.playing = False
            return
        if self.playing:
            self.pause()
            return
        self.playing = True
        if self.index(g) >= g[2] - 1:  # at the end: from the start again
            self.seek(0)
        self._last_tick = time.perf_counter()
        self._due = self._last_tick + 1.0 / PLAYBACK_FPS

    def pause(self) -> None:
        """Stops playback on the nearest step (arrows, marks and runs work on steps)."""
        self.playing = False
        k = self.index()
        if k is not None:
            self.seek(k)

    def speed(self) -> float:
        """Playback speed in Hz: recording time advances at real time / speed."""
        try:
            v = float(dpg.get_value(SPEED_TAG)) if dpg.does_item_exist(SPEED_TAG) else SPEED_DEFAULT
        except (TypeError, ValueError):
            v = SPEED_DEFAULT
        return min(max(v, SPEED_MIN), SPEED_MAX) if v > 0 else SPEED_DEFAULT

    def tick(self) -> None:
        """Every rendered frame: dragging, playback, bar redraw."""
        if self._drag is not None:
            if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                self._seek_mouse(self._drag)
            else:
                self._drag = None
        if self.playing:
            now = time.perf_counter()
            frame = 1.0 / PLAYBACK_FPS
            # playback frames are due on a PLAYBACK_FPS grid; one is drawn on the first rendered frame within half
            # a screen refresh of its due time (on a 60 Hz screen: 2 and 3 refreshes apart, 24 per second)
            if now >= self._due - 0.008:
                self._due += frame
                if self._due < now:  # drawing slower than PLAYBACK_FPS: no burst of frames to catch up
                    self._due = now + frame
                # recording time advances by the real time since the last playback frame / speed (the speed stays
                # right when a page draws slower)
                elapsed = min(now - self._last_tick, 0.25)  # a stall (dialog, heavy redraw) does not jump ahead
                self._last_tick = now
                g = self.grid()
                if g is None:
                    self.playing = False
                else:
                    t0, step, K = g
                    last = t0 + (K - 1) * step
                    cur = float(self.pre.display_t0) if self.pre.display_t0 is not None else t0
                    t = cur + 1e6 * elapsed / self.speed()  # us of recording
                    if t >= last:
                        self.playing = False
                        self.seek(K - 1)
                    else:
                        self.seek_time(t)
        self.refresh()
