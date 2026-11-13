import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import dearpygui.dearpygui as dpg
import numpy as np
from interface.view_zoom import SharedZoom
from config.constants import (
    DEFAULT_SENSOR_SIZE,
    POLARITY_OPTIONS,
    DEFAULT_PARAMS,
    PSEUDOFAME_DIMENSIONS,
    SUPPORTED_FORMATS,
)
from utils.paths import default_files_dir
from utils.masks import MASK_MODES, Mask, keep_map, load_masks, save_masks
from interface.mask_editor import PolygonMaskEditor
from utils.dataloader import build_preview_and_stats, load_data_file
from utils.helpers import EventDataHelper
from utils.plots import (
    figure_to_rgba_flat,
)

def _time_sorted(events):
    """Drops non-finite events and returns the event arrays sorted by time.

    Event streams from the camera are already sorted, in which case nothing is
    copied. Sorted times let every time window be taken with np.searchsorted.
    """
    if events is None:
        return None
    xs, ys, ts, ps = events
    n = min(len(xs), len(ys), len(ts))
    if ps is not None:
        n = min(n, len(ps))
        ps = ps[:n]
    xs, ys, ts = xs[:n], ys[:n], ts[:n]
    finite = np.isfinite(ts)
    for a in (xs, ys):
        if a.dtype.kind == "f":
            finite &= np.isfinite(a)
    if not finite.all():
        xs, ys, ts = xs[finite], ys[finite], ts[finite]
        ps = ps[finite] if ps is not None else None
    if ts.size > 1 and not bool(np.all(ts[1:] >= ts[:-1])):
        order = np.argsort(ts, kind="stable")
        xs, ys, ts = xs[order], ys[order], ts[order]
        ps = ps[order] if ps is not None else None
    return xs, ys, ts, ps


class PreprocessingTab:
    label: str = "Preprocessing"

    def __init__(self) -> None:
        self.view_zoom = SharedZoom()  # zoom of every sensor view, on all tabs
        self.params: dict[str, Any] = deepcopy(DEFAULT_PARAMS)
        self.data_file: str = ""
        self.output_dir: str = ""
        self.supported_formats: Sequence[str] = SUPPORTED_FORMATS
        self.pseudoframe_texture_tag: str = "preprocessing_pseudoframe_texture"
        self.pseudoframe_dimensions = PSEUDOFAME_DIMENSIONS
        self._geometry_listeners = []  # called with (sensor w, h, display w, h) when the sensor size changes
        self.masks = []  # polygon masks (utils/masks.py), applied to the events by apply_event_filters
        self._mask_editor = None
        # (mask, state) of the masks the events are filtered with: edits wait for the apply button
        self._applied_masks = []
        self.last_loaded_data: Any = None
        self.display_t0: float | None = None
        self.raw_events: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
        self.filtered_events: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
        self.last_loaded_file_ext: Optional[str] = None
        self.params = {
            "accumulation_time_ms": 2.0,
        }
        self.last_event_points: Optional[
                tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
            ] = None
        self.main_texture_tag: str = "preprocessing_main_plot_texture"
        self._pseudoframe_status_tags: set[str] = {"preprocessing_pseudoframe_status"}
        self._pseudoframe_time_info_tags: set[str] = {"pseudoframe_time_info"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, parent: int) -> None:
        """Build the tab UI."""
        self._ensure_textures()
        with dpg.group(parent=parent):
            with dpg.group(horizontal=True):
                with dpg.child_window(label="Data Management", width=500, height=950):
                    self._build_data_panel()
                with dpg.child_window(label="Visualization", width=1920-550, height=950):
                    dpg.add_text("VISUALIZATION", color=(100, 200, 255))
                    dpg.add_separator()
                    self._build_preprocessing_parameters_panel()
        self.reset_pseudoframe_visualization()

    def register_pseudoframe_mirror(self, *, status_tag: str, time_info_tag: str) -> None:
        if status_tag:
            self._pseudoframe_status_tags.add(status_tag)
        if time_info_tag:
            self._pseudoframe_time_info_tags.add(time_info_tag)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _ensure_textures(self) -> None:
        if not dpg.does_item_exist(self.main_texture_tag):
            dummy_data = np.zeros((580, 700, 4), dtype=np.float32)
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=500,
                    height=580,
                    default_value=dummy_data.flatten(),
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.main_texture_tag,
                )
                pseudo_width, pseudo_height = self.pseudoframe_dimensions
                pseudo_data = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
                dpg.add_raw_texture(
                    width=pseudo_width,
                    height=pseudo_height,
                    default_value=pseudo_data.flatten(),
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.pseudoframe_texture_tag,
                )

    def _build_data_panel(self) -> None:
        dpg.add_text("DATA MANAGEMENT", color=(100, 200, 255))
        dpg.add_separator()
        dpg.add_text("Select event stream file:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",
                tag="data_file_input",
                width=300,
                hint="Paste path and press Enter or click Browse...",
                callback=self.on_data_file_path_typed,
                on_enter=True,
            )
            dpg.add_button(label="Browse", callback=self.browse_data_file, width=100)
        formats_text = " | ".join(self.supported_formats)
        dpg.add_text("Supported File Types:", color=(150, 200, 150))
        dpg.add_text(formats_text, color=(180, 180, 180), wrap=700)
        dpg.add_text("", tag="validation_status", color=(100, 255, 100))
        dpg.add_separator()
        dpg.add_text("DATA PREVIEW", color=(100, 200, 255))
        dpg.add_separator()
        with dpg.child_window(label="Data Preview", height=150, width=480):
            dpg.add_text("", tag="data_preview_text", wrap=700)
        dpg.add_separator()
        dpg.add_text("PREPROCESSING PARAMETERS", color=(100, 200, 255))
        dpg.add_separator()

        dpg.add_text("Accumulation time:", color=(200, 200, 100))
        dpg.add_input_float(
            label="accumulation time (ms)",
            tag="bracket_preprocessing_accumulation_time",
            default_value=float(self.params["accumulation_time_ms"]),
            width=150,
            on_enter=True,
            min_value=0.0,
            min_clamped=True,
            callback=self.update_preprocessing_accumulation_time,
        )
        dpg.add_text("Events used:", color=(200, 200, 100))
        dpg.add_combo(
            items=list(POLARITY_OPTIONS),
            default_value=str(self.params.get("polarity", POLARITY_OPTIONS[0])),
            width=150,
            tag="preprocessing_polarity",
            callback=self.on_polarity_change,
        )
        dpg.add_text("Sensor size:", color=(200, 200, 100))
        w0, h0 = self.sensor_size()
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="width", tag="bracket_preprocessing_width", default_value=w0, width=120,
                              min_value=1, min_clamped=True, on_enter=True, callback=self.on_sensor_size_typed)
            dpg.add_input_int(label="height", tag="bracket_preprocessing_height", default_value=h0, width=120,
                              min_value=1, min_clamped=True, on_enter=True, callback=self.on_sensor_size_typed)
        dpg.add_text("Read from the file when it gives it (Enter to apply a typed value).",
                     tag="preprocessing_sensor_size_status", wrap=430, color=(180, 180, 180))
        dpg.add_text("Masks:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Draw new mask", callback=self.on_draw_mask, width=120)
            dpg.add_button(label="Load...", callback=lambda: dpg.show_item("masks_load_dialog"), width=70)
            dpg.add_button(label="Save...", callback=self.on_save_masks_clicked, width=70)
        dpg.add_text("Draw polygons on the pseudo-frame. 'exclude' removes the events inside, 'keep' keeps only the "
                     "events inside the keep masks. Enabled masks apply to the whole chain (detection, tracking, "
                     "validation). Click a mask to select it, drag its points, right click a point to delete it. "
                     "Changes (marked *) take effect when 'apply' (on the selected mask's row) is pressed.",
                     wrap=430, color=(160, 160, 160))
        dpg.add_group(tag="masks_list")
        dpg.add_text("", tag="masks_pending", wrap=430, color=(255, 190, 80))
        dpg.add_text("", tag="masks_status", wrap=430, color=(100, 255, 100))
        dpg.add_separator()
        dpg.add_text("Time range:", color=(200, 200, 100))
        dpg.add_checkbox(
                label="Enable time crop",
                tag="time_crop_enabled",
                default_value=bool(self.params.get("time_crop_enabled", False)),
                callback=self.on_time_crop_toggle,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="tmin",
                tag="time_tmin",
                default_value=float(self.params.get("tmin", 0.0)),
                width=170,
                on_enter=True,
                callback=self.update_time_param,
                user_data="tmin",
            )
            dpg.add_input_float(
                label="tmax",
                tag="time_tmax",
                default_value=float(self.params.get("tmax", 0.0)),
                width=170,
                on_enter=True,
                callback=self.update_time_param,
                user_data="tmax",
            )
        
        dpg.add_separator()
        dpg.add_text("EXPORT FILTERED EVENTS", color=(100, 200, 255))

        dpg.add_text("Save filtered events (X,Y,T,P):", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",
                tag="save_filtered_file_input",
                width=300,
                hint="Choose target file (.npz or .mat)",
                callback=self.on_save_filtered_path_typed,
                on_enter=True,
            )
            dpg.add_button(label="Browse", callback=self.browse_save_filtered_file, width=80)
            dpg.add_combo(
                items=[".npz", ".mat"],
                default_value=".npz",
                tag="save_filtered_format",
                width=70,
            )
        dpg.add_button(label="Save data", callback=self.save_filtered_events, width=120)
        dpg.add_text("", tag="save_filtered_status", color=(100, 255, 100))

        if not dpg.does_item_exist("save_filtered_file_dialog"):
            with dpg.file_dialog(
                tag="save_filtered_file_dialog",
                label="Save filtered events",
                directory_selector=False,
                default_path=default_files_dir(),
                show=False,
                callback=self.on_save_filtered_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("save_filtered_status", "Save cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
            
            
        for tag, label, cb in (("masks_load_dialog", "Load masks", self.on_load_masks_selected),
                               ("masks_save_dialog", "Save masks", self.on_save_masks_selected)):
            if not dpg.does_item_exist(tag):
                with dpg.file_dialog(tag=tag, label=label, directory_selector=False, default_path=default_files_dir(),
                                     show=False, callback=cb, width=800, height=500):
                    dpg.add_file_extension(".json", color=(150, 255, 150, 255))

        if not dpg.does_item_exist("data_file_dialog"):
            with dpg.file_dialog(
                tag="data_file_dialog",
                label="Select event-stream file",
                directory_selector=False,
                default_path=default_files_dir(),
                show=False,
                callback=self.on_data_file_selected,
                cancel_callback=self.on_data_file_cancelled,
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".raw", color=(150, 255, 150, 255))  # first: selected by default
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))

    def _build_preprocessing_parameters_panel(self) -> None:
        try:
            if dpg.does_item_exist("preprocessing_parameters_display"):
                dpg.delete_item("preprocessing_parameters_display")
        except Exception:
            pass
        dw, dh = self.display_size()
        with dpg.drawlist(width=dw, height=dh, tag="preprocessing_pf_drawlist"):
            dpg.draw_image(self.pseudoframe_texture_tag, pmin=[0, 0], pmax=[dw, dh], tag="preprocessing_pseudoframe_image")
            with dpg.draw_node(tag="preprocessing_masks_node"):
                pass
        self._mask_editor = PolygonMaskEditor(
            "preprocessing_pf_drawlist", "preprocessing_masks_node", self.masks,
            geometry=lambda: (*self.sensor_size(), *self.display_size()), zoom=self.view_zoom,
            on_change=self._on_masks_changed, on_message=self._mask_message, pending=self._mask_pending)
        self.view_zoom.register("preprocessing_pf_drawlist", self._apply_view_zoom)
        self.register_geometry_listener(self._on_own_geometry)
        dpg.add_text("Load data to generate pseudo-frame.", tag="preprocessing_pseudoframe_status", wrap=730)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="<< Prev", width=120, callback=self.on_prev_pseudoframe)
            dpg.add_button(label="Next >>", width=120, callback=self.on_next_pseudoframe)
            dpg.add_text("", tag="pseudoframe_time_info", wrap=600)

        dpg.add_separator()
        dpg.add_text("Status Log", color=(200, 200, 100))
        with dpg.child_window(label="Status Log", height=90, width=720):
            dpg.add_input_text(tag="status_log_text", multiline=True, readonly=True, width=695, height=75)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def browse_data_file(self, sender=None, app_data=None, user_data=None):
        dpg.show_item("data_file_dialog")
    
    def on_data_file_cancelled(self, sender, app_data):
        self.log_status("File selection cancelled")

    def on_data_file_selected(self, sender, app_data) -> None: 
        if app_data.get("file_path_name"):
            self.data_file = app_data["file_path_name"]
            dpg.set_value("data_file_input", self.data_file)
            if dpg.does_item_exist("preprocessing_file_display"):
                dpg.set_value("preprocessing_file_display", f"{Path(self.data_file).name}")
            self.log_status(f"File selected: {self.data_file}")
            self.update_preprocessing_parameters_display()
            self.validate_file()  # selecting a file loads it
            
    def on_data_file_path_typed(self, sender, app_data) -> None:
        path = (app_data or "").strip().strip('"').strip("'")
        if not path:
            self.data_file = ""
            if dpg.does_item_exist("validation_status"):
                dpg.set_value("validation_status", "No file selected")
            self.log_status("File path cleared")
            return
    
        self.data_file = path
        dpg.set_value("data_file_input", self.data_file)
    
        self.log_status(f"File path entered: {self.data_file}")
        self.update_preprocessing_parameters_display()
        self.validate_file()  # Enter on a path loads it

    def validate_file(self, sender=None, app_data=None, user_data=None) -> None:
        if not self.data_file:
            dpg.set_value("validation_status", "No file selected")
            return
        try:
            dpg.set_value("validation_status", "loading...")
            self.log_status(f"Loading file: {self.data_file}")
            loaded = load_data_file(self.data_file)

            if loaded.width is not None and loaded.height is not None:
                self._set_sensor_size(loaded.width, loaded.height,
                                      f"{loaded.width} x {loaded.height} read from the file.", warn=False)
            else:
                w, h = DEFAULT_SENSOR_SIZE
                self._set_sensor_size(w, h, f"The file gives no sensor size: {w} x {h} assumed. "
                                            "Enter the right size if it differs.", warn=True)

            self.last_loaded_data = loaded.data
            self.last_loaded_file_ext = loaded.file_ext
            self.load_data_preview(loaded)
            dpg.set_value("validation_status", loaded.status_message)
            self.log_status(f"File loaded: {self.data_file}")
        except Exception as exc:
            dpg.set_value("validation_status", f"Error: {exc}")
            self.log_status(f"Loading file failed: {exc}")

    def load_data_preview(self, loaded=None) -> None:
        if not self.data_file:
            return
        try:
            if loaded is None:  # validate_file passes the data it just loaded (no second load)
                loaded = load_data_file(self.data_file)
            data = loaded.data
            file_ext = loaded.file_ext

            self.update_pseudoframe_visualization(None, "Generating pseudo-frame preview...")

            preview_text = build_preview_and_stats(loaded)
            dpg.set_value("data_preview_text", preview_text)

            self.update_preprocessing_parameters_display()

            events = EventDataHelper.extract_event_points(file_ext, data, self.data_file)
            self.raw_events = _time_sorted(events)
            xs, ys, ts, ps = self.raw_events

            tmin = float(ts[0])
            tmax = float(ts[-1])
            
            self.params["tmin"] = tmin
            self.params["tmax"] = tmax
            
            if dpg.does_item_exist("time_tmin"):
                dpg.set_value("time_tmin", tmin)
            if dpg.does_item_exist("time_tmax"):
                dpg.set_value("time_tmax", tmax)
    
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Data loaded, masks applied.")
            
            xs, ys, ts, ps = self.filtered_events if self.filtered_events else self.raw_events
            self.display_t0 = float(ts[0]) if len(ts) else None
        except Exception as exc:
            dpg.set_value("data_preview_text", f"Error loading preview: {exc}")
            self.log_status(f"Error loading data preview: {exc}")

    def apply_event_filters(self) -> None:
        if not getattr(self, "raw_events", None):
            self.filtered_events = None
            return

        xs, ys, ts, ps = self.raw_events

        # raw events are time sorted (see _time_sorted): the time crop is a slice
        lo, hi = 0, len(ts)
        if bool(self.params.get("time_crop_enabled", False)):
            tmin = self.params.get("tmin", None)
            tmax = self.params.get("tmax", None)
            tmin_f = float(tmin) if tmin is not None and str(tmin) != "" else None
            tmax_f = float(tmax) if tmax is not None and str(tmax) != "" else None
            if tmin_f is not None and tmax_f is not None and tmax_f < tmin_f:
                tmin_f, tmax_f = tmax_f, tmin_f
            if tmin_f is not None:
                lo = int(np.searchsorted(ts, tmin_f, side="left"))
            if tmax_f is not None:
                hi = int(np.searchsorted(ts, tmax_f, side="right"))
            hi = max(hi, lo)
        X = xs[lo:hi]
        Y = ys[lo:hi]
        T = ts[lo:hi]
        P = ps[lo:hi] if ps is not None else None

        # only build the masks that can remove events
        mask = None
        def _and(m):
            nonlocal mask
            mask = m if mask is None else (mask & m)
        W, H = self.sensor_size()
        n_outside = 0
        if X.size and (X.min() < 0 or X.max() >= W or Y.min() < 0 or Y.max() >= H):  # outside the sensor
            inside = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
            n_outside = int(inside.size - np.count_nonzero(inside))
            _and(inside)
        keep = keep_map(self._applied_mask_list(), W, H)  # (not the edits waiting for 'apply')
        if keep is not None and X.size:
            _and(keep[np.clip(Y, 0, H - 1).astype(np.intp), np.clip(X, 0, W - 1).astype(np.intp)])
        polarity = self.params.get("polarity", POLARITY_OPTIONS[0])
        if P is not None and polarity != "ON and OFF":
            _and(P > 0 if polarity == "ON only" else P <= 0)

        if mask is not None:
            X, Y, T = X[mask], Y[mask], T[mask]
            P = P[mask] if P is not None else None
        # integer pixel coordinates; int32 halves the memory of int64 (values < 2^31)
        Xf = X.astype(np.int32)
        Yf = Y.astype(np.int32)
        Tf = np.asarray(T, dtype=np.float64)
        Pf = P
        self.filtered_events = (Xf, Yf, Tf, Pf)
        self.log_status(f"Filters applied: kept {Xf.size} / {len(xs)} events")
        if n_outside:
            self.log_status(f"{n_outside} events outside the {W} x {H} sensor removed: check the sensor size")
        if dpg.does_item_exist("masks_status"):
            applied = self._applied_mask_list()
            n_on = sum(1 for m in applied if m.enabled and len(m.points) >= 3)
            dpg.set_value("masks_status", f"{n_on} mask(s) applied: {Xf.size} of {len(xs)} events kept"
                                          if applied else "")

    # ------------------------------------------------------------------
    # Sensor size and polarity
    # ------------------------------------------------------------------
    def sensor_size(self) -> tuple[int, int]:
        """(width, height) of the sensor in pixels: read from the file, typed by the user, or the default."""
        return int(self.params.get("width", DEFAULT_SENSOR_SIZE[0])), int(self.params.get("height", DEFAULT_SENSOR_SIZE[1]))

    def display_size(self) -> tuple[int, int]:
        """Size of the sensor views: the sensor aspect ratio, fitted in the pseudo-frame area."""
        w, h = self.sensor_size()
        bw, bh = self.pseudoframe_dimensions
        scale = min(bw / w, bh / h)
        return max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    def register_geometry_listener(self, callback) -> None:
        """`callback(sensor_w, sensor_h, display_w, display_h)`, now and whenever the sensor size changes."""
        self._geometry_listeners.append(callback)
        callback(*self.sensor_size(), *self.display_size())

    def _set_sensor_size(self, width: int, height: int, message: str, warn: bool = False) -> None:
        self.params["width"], self.params["height"] = int(width), int(height)
        for tag, v in (("bracket_preprocessing_width", width), ("bracket_preprocessing_height", height)):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, int(v))
        if dpg.does_item_exist("preprocessing_sensor_size_status"):
            dpg.set_value("preprocessing_sensor_size_status", message)
            dpg.configure_item("preprocessing_sensor_size_status",
                               color=(255, 170, 80) if warn else (100, 255, 100))
        self.log_status(f"Sensor size: {message}")
        dims = (*self.sensor_size(), *self.display_size())
        for callback in self._geometry_listeners:
            try:
                callback(*dims)
            except Exception as e:  # one view must not stop the others
                self.log_status(f"View resize failed: {e}")
        self.update_preprocessing_parameters_display()

    def on_sensor_size_typed(self, sender=None, app_data=None, user_data=None) -> None:
        w = int(dpg.get_value("bracket_preprocessing_width"))
        h = int(dpg.get_value("bracket_preprocessing_height"))
        if (w, h) == self.sensor_size():
            return
        self._set_sensor_size(w, h, f"{w} x {h} entered.", warn=False)
        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Sensor size changed.")

    def on_polarity_change(self, sender=None, app_data=None, user_data=None) -> None:
        self.params["polarity"] = str(app_data)
        self.log_status(f"Events used: {app_data}")
        self.update_preprocessing_parameters_display()
        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Polarity changed.")

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def _apply_view_zoom(self, uv_min, uv_max) -> None:
        dpg.configure_item("preprocessing_pseudoframe_image", uv_min=uv_min, uv_max=uv_max)
        if self._mask_editor is not None:
            self._mask_editor.redraw()

    def _on_own_geometry(self, width, height, display_w, display_h) -> None:
        dpg.configure_item("preprocessing_pf_drawlist", width=display_w, height=display_h)
        dpg.configure_item("preprocessing_pseudoframe_image", pmax=[display_w, display_h])
        if self._mask_editor is not None:
            self._mask_editor.redraw()

    def _mask_message(self, text: str) -> None:
        if dpg.does_item_exist("masks_status"):
            dpg.set_value("masks_status", text)
        self.log_status(text)

    def on_draw_mask(self, sender=None, app_data=None, user_data=None) -> None:
        if self._mask_editor is not None:
            self._mask_editor.start_drawing()

    @staticmethod
    def _mask_state(m):
        """What of a mask acts on the events (not its name)."""
        return tuple(tuple(float(c) for c in p) for p in m.points), bool(m.enabled), str(m.mode)

    def _applied_mask_list(self):
        return [Mask(m.name, [list(p) for p in st[0]], st[1], st[2]) for m, st in self._applied_masks]

    def _mask_pending(self, m) -> bool:
        """Whether mask m has changes that are not applied yet."""
        st = next((st for a, st in self._applied_masks if a is m), None)
        if st is None:  # new mask: it changes the events once it is enabled and closed
            return bool(m.enabled and len(m.points) >= 3)
        return st != self._mask_state(m)

    def _masks_pending(self) -> bool:
        removed = any(st[1] and len(st[0]) >= 3 for a, st in self._applied_masks
                      if not any(a is m for m in self.masks))
        return removed or any(self._mask_pending(m) for m in self.masks)

    def _on_masks_changed(self, committed: bool) -> None:
        """committed: the masks were edited (waiting for 'apply'); otherwise only the selection changed."""
        self._rebuild_masks_list()
        if self._mask_editor is not None:
            self._mask_editor.redraw()

    def _on_masks_apply(self, sender=None, app_data=None, user_data=None) -> None:
        """Filters the events with the masks as they are now."""
        self._applied_masks = [(m, self._mask_state(m)) for m in self.masks]
        self._rebuild_masks_list()
        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Masks applied.")
        else:
            self._mask_message("Masks applied (they act on the events once data is loaded).")
        if self._mask_editor is not None:
            self._mask_editor.redraw()

    def _select_mask(self, index) -> None:
        if self._mask_editor is not None and self._mask_editor.selected != index:
            self._mask_editor.select(index)

    def _rebuild_masks_list(self) -> None:
        if not dpg.does_item_exist("masks_list"):
            return
        dpg.delete_item("masks_list", children_only=True)
        selected = self._mask_editor.selected if self._mask_editor is not None else -1
        pending = self._masks_pending()
        for i, m in enumerate(self.masks):
            with dpg.group(horizontal=True, parent="masks_list"):
                dpg.add_text("*" if self._mask_pending(m) else " ", color=(255, 190, 80))
                dpg.add_checkbox(default_value=m.enabled, user_data=i, callback=self._on_mask_enabled)
                dpg.add_input_text(default_value=m.name, width=130, user_data=i, callback=self._on_mask_renamed)
                dpg.add_combo(items=list(MASK_MODES), default_value=m.mode, width=90, user_data=i,
                              callback=self._on_mask_mode)
                if i == selected:
                    dpg.add_button(label="apply", width=70, tag="masks_apply_button", enabled=pending,
                                   callback=self._on_masks_apply)
                else:
                    dpg.add_button(label="select", width=70, user_data=i,
                                   callback=lambda s, a, u: (self._select_mask(u), self._rebuild_masks_list()))
                dpg.add_button(label="X", width=25, user_data=i, callback=self._on_mask_deleted)
        if dpg.does_item_exist("masks_pending"):
            dpg.set_value("masks_pending", "Changes not applied yet (*): press 'apply' on the selected mask."
                          if pending else "")

    def _on_mask_enabled(self, sender, app_data, user_data) -> None:
        self.masks[user_data].enabled = bool(app_data)
        self._select_mask(user_data)  # (its row gets the apply button)
        self._on_masks_changed(True)

    def _on_mask_renamed(self, sender, app_data, user_data) -> None:
        self.masks[user_data].name = str(app_data)
        if self._mask_editor is not None:
            self._mask_editor.redraw()

    def _on_mask_mode(self, sender, app_data, user_data) -> None:
        self.masks[user_data].mode = str(app_data)
        self._select_mask(user_data)
        self._on_masks_changed(True)

    def _after_list_change(self, select: int) -> None:
        """After masks were deleted or loaded: the apply button goes to mask `select`; with no mask left there is no
        button, so the change is applied at once."""
        if self.masks:
            self._select_mask(min(max(select, 0), len(self.masks) - 1))
            self._on_masks_changed(True)
        else:
            self._select_mask(-1)
            if self._masks_pending():
                self._on_masks_apply()
            else:
                self._on_masks_changed(True)

    def _on_mask_deleted(self, sender, app_data, user_data) -> None:
        name = self.masks[user_data].name
        del self.masks[user_data]
        if self._mask_editor is not None:
            self._mask_editor.selected = -1
        self._mask_message(f"{name} deleted." + ("" if self.masks else " No mask left: applied."))
        self._after_list_change(user_data)

    def on_save_masks_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        if not self.masks:
            self._mask_message("No masks to save.")
            return
        dpg.show_item("masks_save_dialog")

    def on_save_masks_selected(self, sender, app_data, user_data=None) -> None:
        path = Path(str(app_data.get("file_path_name", "")).strip())
        if not str(path):
            return
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            save_masks(path, self.masks)
            self._mask_message(f"Masks saved: {path}")
        except Exception as e:
            self._mask_message(f"Saving masks failed: {e}")

    def on_load_masks_selected(self, sender, app_data, user_data=None) -> None:
        path = str(app_data.get("file_path_name", "")).strip()
        if not path:
            return
        try:
            loaded = load_masks(path)
        except Exception as e:
            self._mask_message(f"Loading masks failed: {e}")
            return
        self.masks[:] = loaded  # (the editor holds this list)
        if self._mask_editor is not None:
            self._mask_editor.selected = -1
        self._mask_message(f"{len(loaded)} mask(s) loaded from {path}" + (": press 'apply' to use them." if loaded else ""))
        self._after_list_change(0)
            
    def update_time_param(self, sender, app_data, user_data):
        self.params[user_data] = float(app_data)
        if self.params.get("time_crop_enabled", False) and getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Time crop updated.")
            
    def on_time_crop_toggle(self, sender, app_data):
        self.params["time_crop_enabled"] = bool(app_data)
        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Time crop toggled.")

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def update_preprocessing_param(self, sender=None, app_data=None, user_data=None) -> None:  
        param_name = user_data
        if param_name in ("width", "height"):
            value = dpg.get_value(f"bracket_preprocessing_{param_name}")
            self.params[param_name] = value
            self.log_status(f"Preprocessing {param_name} updated to {value}")
            self.update_preprocessing_parameters_display()
            if self.last_event_points:
                self.update_pseudoframe_from_data(f"Pseudo-frame updated ({param_name} changed).")
                self.show_raw_events_in_visualization()
     
    def update_preprocessing_accumulation_time(self, sender, app_data, user_data=None):
        try:
            acc_ms = float(app_data)
        except Exception:
            return

        self.params["accumulation_time_ms"] = acc_ms
        step = acc_ms * 1000.0  
        self.log_status(f"Accumulation time set to {acc_ms:.3f} ms -> window step {step:.0f} ts-units")

        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Accumulation time changed.")

    def update_preprocessing_parameters_display(self) -> None:
        if self.data_file:
            file_display = f"{Path(self.data_file).name}"
            if dpg.does_item_exist("preprocessing_file_display"):
                dpg.set_value("preprocessing_file_display", file_display)
        else:
            if dpg.does_item_exist("preprocessing_file_display"):
                dpg.set_value("preprocessing_file_display", "No file selected")
        
        if not dpg.does_item_exist("preprocessing_parameters_display"):
            return
        params_text = "Current Settings:\n"
        w, h = self.sensor_size()
        params_text += f"  Sensor: {w} x {h}\n"
        acc = self.params.get("accumulation_time_ms", None)
        if acc is not None and acc != "":
            params_text += f"  Accumulation Time: {acc} ms\n"
        params_text += f"  Events used: {self.params.get('polarity', POLARITY_OPTIONS[0])}\n"
        dpg.set_value("preprocessing_parameters_display", params_text)

    # ------------------------------------------------------------------
    # Pseudo frame helpers
    # ------------------------------------------------------------------
    def reset_pseudoframe_visualization(self, message: str = "Load data to generate pseudo-frame.") -> None:
        pseudo_width, pseudo_height = self.pseudoframe_dimensions
        blank = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
        if dpg.does_item_exist(self.pseudoframe_texture_tag):
            dpg.set_value(self.pseudoframe_texture_tag, blank.flatten())
        for tag in list(self._pseudoframe_status_tags):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, message)

    def update_pseudoframe_visualization(self, frame: Optional[np.ndarray], status: Optional[str] = None) -> None:
        if frame is None:
            self.reset_pseudoframe_visualization(status or "Pseudo-frame preview not available for this format.")
            return
        self.render_pseudoframe_to_texture(frame)
        if status:
            for tag in list(self._pseudoframe_status_tags):
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, status)

    def render_pseudoframe_to_texture(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        frame_array = np.array(frame, dtype=np.float32)
        frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=0.0, neginf=0.0)
            
        if frame_array.ndim == 2:
            pseudo_width, pseudo_height = self.pseudoframe_dimensions
            src_height, src_width = frame_array.shape
            if src_height <= 0 or src_width <= 0:
                return
        
            y_indices = np.linspace(0, src_height - 1, pseudo_height).astype(np.int32)
            x_indices = np.linspace(0, src_width - 1, pseudo_width).astype(np.int32)
            resized = frame_array[y_indices][:, x_indices]

            cmap = np.array([
                [30,  37,  52],
                [64, 124, 198],
                [220, 226, 238],
            ], dtype=np.float32) / 255.0

            mask = resized > 0
            rgba = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
            rgba[:, :, :3] = cmap[0]          
            rgba[mask, :3] = cmap[2]          
            rgba[:, :, 3] = 1.0               
        elif frame_array.ndim == 3 and frame_array.shape[2] in (3, 4):
            pseudo_width, pseudo_height = self.pseudoframe_dimensions
            src_height, src_width = frame_array.shape[:2]
            if src_height <= 0 or src_width <= 0:
                return
            y_indices = np.linspace(0, src_height - 1, pseudo_height).astype(np.int32)
            x_indices = np.linspace(0, src_width - 1, pseudo_width).astype(np.int32)
            resized = frame_array[y_indices][:, x_indices]
            rgba = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
            rgba[:, :, :resized.shape[2]] = resized[:, :, :resized.shape[2]]
            if resized.shape[2] == 3:
                rgba[:, :, 3] = 1.0
            else:
                rgba[:, :, 3] = resized[:, :, 3]
        else:
            return
        if dpg.does_item_exist(self.pseudoframe_texture_tag):
            dpg.set_value(self.pseudoframe_texture_tag, rgba.flatten())

    def update_pseudoframe_from_data(self, message: Optional[str] = None) -> None:
        if not self.filtered_events:
            self.update_pseudoframe_visualization(None, "Pseudo-frame preview unavailable (no events, or all masked).")
            self.log_status("Cannot generate pseudo-frame: no filtered events available")
            return
        try:
            xs, ys, ts, polarities = self.filtered_events
            if xs is None or ys is None or len(xs) == 0 or len(ys) == 0:
                self.update_pseudoframe_visualization(None, "Pseudo-frame preview unavailable (empty coordinates).")
                self.log_status("Cannot generate pseudo-frame: empty coordinate arrays")
                return
            width, height = self.sensor_size()
            
            ts_arr, xs_arr, ys_arr = ts, xs, ys  # time sorted and finite (see _time_sorted)

            if xs_arr.size == 0:
                self.update_pseudoframe_visualization(None, "No valid events for accumulation.")
                return
            
            acc_ms = float(self.params.get("accumulation_time_ms", 2.0))
            step = acc_ms * 1000.0  
            
            if self.display_t0 is None:
                self.display_t0 = float(ts_arr[0])

            T_min = float(self.display_t0)
            T_max = T_min + step

            # window (T_min, T_max] by binary search instead of scanning all events
            lo, hi = np.searchsorted(ts_arr, [T_min, T_max], side="right")
            n_window = int(hi - lo)
            X = xs_arr[lo:hi].astype(np.int64)
            Y = ys_arr[lo:hi].astype(np.int64)

            valid = (X >= 0) & (X < width) & (Y >= 0) & (Y < height)
            X, Y = X[valid], Y[valid]
            
            I = np.zeros((height, width), dtype=np.float32)
            I[Y.astype(np.intp), X.astype(np.intp)] = 1.0

            self.update_pseudoframe_visualization(I, f"Accumulation preview ({n_window} events).")
            for tag in list(self._pseudoframe_time_info_tags):
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, f"t: [{T_min:.0f} .. {T_max:.0f}] (display window)")
        except Exception as exc:
            error_message = f"Pseudo-frame error: {exc}"
            self.log_status(error_message)
            self.update_pseudoframe_visualization(None, error_message)

    def _get_acc_step_ts_units(self) -> float:
        acc_ms = float(self.params.get("accumulation_time_ms", 2.0))
        return acc_ms * 1000.0
    
    def on_next_pseudoframe(self, sender=None, app_data=None, user_data=None):
        self._shift_time_window(+1)
    
    def on_prev_pseudoframe(self, sender=None, app_data=None, user_data=None):
        self._shift_time_window(-1)
    
    def _shift_time_window(self, direction: int) -> None:
        if not self.filtered_events:
            self.log_status("No filtered events available.")
            return
    
        xs, ys, ts, ps = self.filtered_events
        acc_ms = float(self.params.get("accumulation_time_ms", 2.0))
        step = acc_ms * 1000.0
    
        if len(ts) == 0:
            return
        if self.display_t0 is None:
            self.display_t0 = float(ts[0])

        new_t0 = float(self.display_t0) + direction * step

        data_min = float(ts[0])   # events are time sorted
        data_max = float(ts[-1])
        if new_t0 < data_min:
            new_t0 = data_min
        if new_t0 + step > data_max:
            new_t0 = data_max - step
    
        self.display_t0 = new_t0
        self.update_pseudoframe_from_data("Display window shifted.")

    def render_plot_to_texture(self, fig, texture_tag: str) -> None:  
        try:
            dpg.set_value(texture_tag, figure_to_rgba_flat(fig))
        except Exception as exc:
            self.log_status(f"Error rendering plot: {exc}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_status(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        if dpg.does_item_exist("status_log_text"):
            current_text = dpg.get_value("status_log_text")
            new_text = f"[{timestamp}] {message}\n{current_text}"
            dpg.set_value("status_log_text", new_text)

    # ------------------------------------------------------------------
    # Export / Import filtered events
    # ------------------------------------------------------------------
    def browse_save_filtered_file(self, sender=None, app_data=None, user_data=None):
        default_path = None
        try:
            if self.data_file:
                default_path = str(Path(self.data_file).resolve().parent)
        except Exception:
            default_path = None
        if default_path and dpg.does_item_exist("save_filtered_file_dialog"):
            dpg.configure_item("save_filtered_file_dialog", default_path=default_path)
        dpg.show_item("save_filtered_file_dialog")

    def on_save_filtered_file_selected(self, sender, app_data) -> None:
        file_path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
        if not file_path:
            dpg.set_value("save_filtered_status", "No file selected.")
            return
        file_path = str(file_path).strip().strip('"').strip("'")
        dpg.set_value("save_filtered_file_input", file_path)
        ext = Path(file_path).suffix.lower()
        if ext in (".npz", ".mat") and dpg.does_item_exist("save_filtered_format"):
            dpg.set_value("save_filtered_format", ext)
        dpg.set_value("save_filtered_status", f"Target set: {Path(file_path).name}")

    def on_save_filtered_path_typed(self, sender, app_data) -> None:
        path = (app_data or "").strip().strip('"').strip("'")
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext in (".npz", ".mat") and dpg.does_item_exist("save_filtered_format"):
            dpg.set_value("save_filtered_format", ext)

    def save_filtered_events(self, sender=None, app_data=None, user_data=None) -> None:
        if not getattr(self, "raw_events", None):
            dpg.set_value("save_filtered_status", "No data loaded.")
            return

        self.apply_event_filters()
        if not getattr(self, "filtered_events", None):
            dpg.set_value("save_filtered_status", "No filtered events available.")
            return

        X, Y, T, P = self.filtered_events
        target = ""
        if dpg.does_item_exist("save_filtered_file_input"):
            target = str(dpg.get_value("save_filtered_file_input") or "").strip()
        fmt = ".npz"
        if dpg.does_item_exist("save_filtered_format"):
            fmt = str(dpg.get_value("save_filtered_format") or ".npz").strip()

        if not target:
            base_dir = Path(self.data_file).resolve().parent if self.data_file else Path.cwd()
            base_name = Path(self.data_file).stem if self.data_file else "events"
            target = str(base_dir / f"{base_name}_filtered{fmt}")

        target_path = Path(target)
        if target_path.suffix.lower() not in (".npz", ".mat"):
            target_path = target_path.with_suffix(fmt)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.suffix.lower() == ".npz":
                payload = {"X": np.asarray(X), "Y": np.asarray(Y), "T": np.asarray(T)}
                if P is not None:
                    payload["P"] = np.asarray(P)
                np.savez_compressed(str(target_path), **payload, width=int(self.params.get("width", 0)), height=int(self.params.get("height", 0)))
            else:
                from scipy.io import savemat  
                payload = {"X": np.asarray(X), "Y": np.asarray(Y), "T": np.asarray(T)}
                if P is not None:
                    payload["P"] = np.asarray(P)
                payload["width"] = int(self.params.get("width", 0))
                payload["height"] = int(self.params.get("height", 0))
                savemat(str(target_path), payload)
            dpg.set_value("save_filtered_status", f"Saved: {target_path.name} ({int(np.asarray(X).size):,} events)")
            self.log_status(f"Filtered events saved to: {target_path}")
        except Exception as exc:
            dpg.set_value("save_filtered_status", f"Save failed: {exc}")
            self.log_status(f"Save filtered events failed: {exc}")