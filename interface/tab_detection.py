from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, TYPE_CHECKING

import dearpygui.dearpygui as dpg
import numpy as np
import pickle
import scipy.io as sio
from pathlib import Path
import math
import matplotlib.pyplot as plt
from utils.paths import default_files_dir
from utils.plots import figure_to_rgba_flat
from utils.raster import binary_rgba, draw_dots, resample_nearest, scalar_rgba
from utils.packed_tracks import PackedTracks
from utils import parallel
from interface.view_zoom import SharedZoom
from matplotlib.colors import ListedColormap
from datetime import datetime
import sys, io, contextlib, threading, queue

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.detection_engine import DetectionEngine

from config.constants import DEFAULT_PARAMS, PSEUDOFAME_DIMENSIONS
from config.settings import STORE
from interface.settings_ui import add_setting_field

if TYPE_CHECKING:
    from interface.tab_preprocessing import PreprocessingTab
    
    
PSEUDOFRAME_CMAP = ListedColormap(
    np.array([
        [30,  37,  52],
        [64, 124, 198],
        [220, 226, 238],
    ], dtype=np.float32) / 255.0
)

class QueueWriter(io.TextIOBase):
    def __init__(self, q):
        self.q = q
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(line.strip())
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.q.put(self._buf.strip())
        self._buf = ""
        
class DetectionCancelled(Exception):
    pass

def make_trace(stop_event):
    def trace(frame, event, arg):
        if stop_event.is_set():
            raise DetectionCancelled("Stop requested")
        return trace
    return trace


class DetectionTab:
    label: str = "Detection"

    def __init__(self, preprocessing_tab: Optional[PreprocessingTab] = None) -> None:
        self.params: dict[str, Any] = deepcopy(DEFAULT_PARAMS)
        self.preprocessing_tab: Optional[PreprocessingTab] = preprocessing_tab

        self.main_texture_tag = "detection_main_plot_texture"
        self._local_pseudoframe_texture_tag = "detection_pseudoframe_texture"
        self.pseudoframe_dimensions = PSEUDOFAME_DIMENSIONS
        self.cluster_texture_tag = "det_cluster_texture"

        self._status_tag = "detection_pseudoframe_status"
        self._time_info_tag = "detection_pseudoframe_time_info"
        self._det_method_combo_tag = "det_method_combo"
        self._det_method_options = ["pixelwise extension", "kd-tree", "DBSCAN", "pseudo-frame"]
        
        self._group_pixelwise = "det_group_pixelwise"
        self._group_kdtree = "det_group_kdtree"
        self._group_DBSCAN = "det_group_DBSCAN"
        self._group_pseudo = "det_group_pseudo"
        script_path = Path(__file__).resolve().parents[1] / "utils" / "detection.py"
        self.engine = DetectionEngine(script_path)
        
        self._run_status_tag = "det_run_status"
        self._run_button_tag = "det_run_button"
        self.params.setdefault("N_pixelwise", 5)
        self.params.setdefault("overlap", 0.4)
        self.params.setdefault("structure_size", 3)  # 3x3 dilation: joins the small gaps along fast streaks
        self.params.setdefault("min_steps", 3)    # a kept cluster spans at least this many time steps
        self.params.setdefault("min_events", 20)  # and holds at least this many events
        
        self.params.setdefault("epsilon", 7.5)  
        self.params.setdefault("minPts", 20)    
        self.params.setdefault("area", 5)       
        self.params.setdefault("filtersize", 3)
        
        self.params.setdefault("N", 1)
        self.params.setdefault("multirun", False)
        self.params.setdefault("multitimefactor", 2)
        self.params.setdefault("cpu_threads", 0)  # threads of the compiled kernels, 0 = automatic
        parallel.set_threads(self.params["cpu_threads"])
        
        self.params.setdefault("threshold", 7.5)
        self._group_matching_dbscan = "det_group_matching_dbscan"
        self._n_user_set = False
        self._stop_event = threading.Event()
        self._log_queue = queue.Queue()
        self._run_thread = None
        
        self._last_refresh_signature = None
        self._preview = None  # (test result, its clusters a full run would discard, summary)
        self._cluster_grey_mask = None  # pixels of those clusters in the event-cluster image
        self._refresh_in_progress = False
        self._plotted_signature = None  # what the plots show (see _plot_signature)
        self._overlay_pseudoframe_texture_tag = "detection_pseudoframe_overlay_texture"
        self._view_zoom = preprocessing_tab.view_zoom if preprocessing_tab is not None else SharedZoom()
        self._last_run_ctx = None
        if preprocessing_tab is not None:  # results shown on the timeline, N follows its range
            preprocessing_tab.timeline.attach(detection_tab=self)
            preprocessing_tab.timeline.add_range_listener(self._on_range_changed)
        self.duration_hist_texture_tag = "det_duration_hist_texture"
        self._texture_registry_tag = "det_texture_registry"        
        
        self._det_thread: Optional[threading.Thread] = None
        self.last_result = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, parent: int) -> None:
        self._ensure_textures()
        with dpg.group(parent=parent):
            with dpg.group(horizontal=True):
                with dpg.child_window(label="Detection", width=500, height=950):
                    self._build_data_panel()

                with dpg.child_window(label="Visualization", width=1920 - 550, height=950):
                    dpg.add_text("VISUALIZATION", color=(100, 200, 255))
                    dpg.add_separator()

                    with dpg.tab_bar(tag="det_vis_tabbar"):
                        with dpg.tab(label="Pseudo-frame", tag="det_vis_tab_detection"):
                            self._build_detection_plot_panel()
                            
                        with dpg.tab(label="Event cluster", tag="det_vis_tab_newplot"):
                            self._build_new_plot_panel()
                        with dpg.tab(label="Cluster duration", tag="det_vis_tab_duration_hist"):
                            self._build_duration_hist_panel()
                    
        handler_tag = "det_tab_handler_registry"

        if not dpg.does_item_exist(handler_tag):
            with dpg.item_handler_registry(tag=handler_tag):
                dpg.add_item_visible_handler(callback=self._on_tab_visible)
                dpg.add_item_activated_handler(callback=self._on_tab_visible)
    
        dpg.bind_item_handler_registry(parent, handler_tag)
                
        self._status_log_append("Ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_textures(self) -> None:
        """Ensure texture resources exist."""
        registry_tag = "texture_registry_main"
        if not dpg.does_item_exist(self._texture_registry_tag):
            dpg.add_texture_registry(tag=self._texture_registry_tag)

        if not dpg.does_item_exist(registry_tag):
            dpg.add_texture_registry(tag=registry_tag)

        if not dpg.does_item_exist(self.main_texture_tag):
            dummy = np.zeros((580, 700, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=700,
                height=580,
                default_value=dummy.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag=self.main_texture_tag,
                parent=registry_tag,
            )
            
        if not dpg.does_item_exist(self.cluster_texture_tag):
            dummy_data = np.zeros((2, 2, 4), dtype=np.float32)
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=2,
                    height=2,
                    default_value=dummy_data.flatten(),
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.cluster_texture_tag,
                )
                
        if not dpg.does_item_exist(self.duration_hist_texture_tag):
            dummy = np.zeros((580, 700, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=700,
                height=580,
                default_value=dummy.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag=self.duration_hist_texture_tag,
                parent=registry_tag,
            )

        if self.preprocessing_tab is None and not dpg.does_item_exist(self._local_pseudoframe_texture_tag):
            pseudo_w, pseudo_h = self.pseudoframe_dimensions
            pseudo = np.zeros((pseudo_h, pseudo_w, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=pseudo_w,
                height=pseudo_h,
                default_value=pseudo.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag=self._local_pseudoframe_texture_tag,
                parent=registry_tag,
            )
            
        if not dpg.does_item_exist(self._overlay_pseudoframe_texture_tag):
            pseudo_w, pseudo_h = self.pseudoframe_dimensions
            overlay = np.zeros((pseudo_h, pseudo_w, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=pseudo_w,
                height=pseudo_h,
                default_value=overlay.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag=self._overlay_pseudoframe_texture_tag,
                parent=registry_tag,
            )

    def _build_data_panel(self) -> None:
        with dpg.theme() as info_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 110, 160))
        
                dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 130, 180, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10) 
        
        dpg.add_text("DETECTION", color=(100, 200, 255))
        dpg.add_separator()
        
        dpg.add_text("Detection method:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_combo(
                items=self._det_method_options,
                default_value=self._det_method_options[0],
                tag=self._det_method_combo_tag,
                width=400,
                callback=self._on_det_method_change
            )
            
            dpg.add_button(label="?", width=22, height=22, tag="det_met_btn")
            dpg.bind_item_theme("det_met_btn", info_button_theme)
        
            with dpg.popup("det_met_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Direct processing: pixelwise extension, kd-Tree or DBSCAN.\n"
                    "Image-based detection: pseudo-frame."
                )
        
        with dpg.group(tag=self._group_pixelwise, show=False):
            dpg.add_input_int(
                label="minimum number of activated pixel per cluster",
                default_value=int(self.params.get("N_pixelwise", 5)),
                min_value=1,
                min_clamped=True,
                width=150,
                callback=lambda s, a, u: self.params.__setitem__("N_pixelwise", int(a)),
                tag="det_input_min_pixel",
            )
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="temporal overlap",
                    default_value=float(self.params.get("overlap", 0.4)),
                    min_value=0.0,
                    max_value=1.0,
                    min_clamped=True,
                    max_clamped=True,
                    width=150,
                    format="%.2f",
                    callback=lambda s, a, u: self.params.__setitem__("overlap", float(a)),
                    tag="det_input_pixelwise_overlap",
                )
            
                dpg.add_button(label="?", width=22, height=22, tag="overlap_info_btn")
                dpg.bind_item_theme("overlap_info_btn", info_button_theme)
            
                with dpg.popup("overlap_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Temporal overlap between consecutive timesteps\n"
                        "used to match found clusters to existing clusters.\n\n"
                        "0.0: no temporal overlap\n"
                        "1.0: full overlap"
                    )
                    
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="mask size",
                    default_value=int(self.params.get("structure_size", 3)),
                    min_value=1,
                    min_clamped=True,
                    width=150,
                    callback=lambda s, a, u: self.params.__setitem__("structure_size", int(a)),
                    tag="det_input_pixelwise_structure_size",
                )
            
                dpg.add_button(label="?", width=22, height=22, tag="structure_size_info_btn")
                dpg.bind_item_theme("structure_size_info_btn", info_button_theme)
            
                with dpg.popup("structure_size_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Defines the square binary dilation mask (NxN)\n"
                        "used in pixelwise extension before labeling.\n\n"
                        "3 applies a 3x3 mask.\n"
                        "Larger values connect pixels over larger gaps;\n"
                        "odd values are recommended for a centered mask."
                    )
            dpg.add_text("Cluster rules", color=(170, 170, 170))
            for key in ("new_cluster_events", "split_merge_distance", "split_min_events"):
                add_setting_field(key, width=150, info_theme=info_button_theme)
        
        with dpg.group(tag=self._group_kdtree, show=False):
        
            dpg.add_input_float(
                label="search radius",
                default_value=float(self.params.get("epsilon", 7.5)),
                min_value=0.0,
                min_clamped=True,
                width=150,
                format="%.3f",
                callback=lambda s, a, u: self.params.__setitem__("epsilon", float(a)),
                tag="det_input_kdtree_radius",
            )
        
            dpg.add_input_int(
                label="minimum number of events per cluster",
                default_value=int(self.params.get("minPts", 20)),
                min_value=1,
                min_clamped=True,
                width=150,
                callback=lambda s, a, u: self.params.__setitem__("minPts", int(a)),
                tag="det_input_kdtree_min_events",
            )

        with dpg.group(tag=self._group_DBSCAN, show=False):
            dpg.add_input_float(
                label="search radius",
                default_value=float(self.params.get("epsilon", 7.5)),
                min_value=0.0,
                min_clamped=True,
                width=150,
                format="%.3f",
                callback=lambda s, a, u: self.params.__setitem__("epsilon", float(a)),
                tag="det_input_dbscan_radius",
            )
        
            dpg.add_input_int(
                label="minimum number of events per cluster",
                default_value=int(self.params.get("minPts", 20)),
                min_value=1,
                min_clamped=True,
                width=150,
                callback=lambda s, a, u: self.params.__setitem__("minPts", int(a)),
                tag="det_input_dbscan_min_events",
            )
        
        with dpg.group(tag=self._group_pseudo, show=False):
            dpg.add_input_int(
                label="minimum pixel area",
                default_value=int(self.params.get("area", 5)),
                min_value=1,
                min_clamped=True,
                width=150,
                callback=lambda s, a, u: self.params.__setitem__("area", int(a)),
                tag="det_input_pseudo_min_area",
            )
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="filter size",
                    default_value=int(self.params.get("filtersize", 3)),
                    min_value=1,
                    min_clamped=True,
                    width=150,
                    callback=lambda s, a, u: self.params.__setitem__("filtersize", int(a)),
                    tag="det_input_pseudo_filtersize",
                )
                dpg.add_button(label="?", width=22, height=22, tag="filter_info_btn")
                dpg.bind_item_theme("filter_info_btn", info_button_theme)
            
                with dpg.popup("filter_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Defines the kernel size (NxN pixel) of a circular\n"
                        "averaging filter applied to the pseudo-frame."
                    )
        
        dpg.add_separator()
        dpg.add_button(
            label="Test run",
            width=200,
            callback=self._on_test_run_clicked,
            tag="det_button_test_run",
        )
        dpg.add_text("", tag="det_test_run_status", wrap=400, color=(100, 255, 100))
        
        dpg.add_text("Test run stats:", color=(200, 200, 100))

        with dpg.child_window(
            tag="det_test_run_stats_box",
            height=110,        
            autosize_x=True,
            border=True,
        ):
            dpg.add_text("No results yet.", tag="det_test_run_stats", wrap=430)
            
        dpg.add_checkbox(
            label="show found cluster centers",
            default_value=bool(self.params.get("show_scatter", False)),
            callback=self._on_show_scatter_toggle,
            tag="det_cb_show_scatter",
        )
        
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Clear test results",
                width=200,
                callback=lambda: self._on_clear_results("test"),
                tag="det_button_clear_test_results",
            )

        dpg.add_separator()
        dpg.add_text("Matching", color=(200, 200, 100))
        
        computed_N = self._compute_default_N()
        if not self._n_user_set:
            self.params["N"] = computed_N
        
        dpg.add_input_int(
            label="number of time steps",
            default_value=int(self.params.get("N", computed_N)),
            min_value=1,
            min_clamped=True,
            width=200,
            callback=self._on_N_change,
            tag="det_input_matching_N",
        )

        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="min. time steps per cluster",
                default_value=int(self.params.get("min_steps", 3)),
                min_value=1,
                min_clamped=True,
                width=150,
                callback=lambda s, a, u: self.params.__setitem__("min_steps", int(a)),
                tag="det_input_min_steps",
            )
            dpg.add_button(label="?", width=22, height=22, tag="min_steps_info_btn")
            dpg.bind_item_theme("min_steps_info_btn", info_button_theme)
            with dpg.popup("min_steps_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "A full run keeps a cluster only if it was followed\n"
                    "over at least this many consecutive time steps.\n"
                    "Shorter-lived clusters are discarded during the run.\n\n"
                    "Fast objects crossing the view in few steps need a\n"
                    "lower value (or a shorter accumulation time).\n"
                    "The test run (one step) shows which clusters a full\n"
                    "run would discard."
                )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="min. events per cluster",
                default_value=int(self.params.get("min_events", 20)),
                min_value=1,
                min_clamped=True,
                width=150,
                callback=lambda s, a, u: self.params.__setitem__("min_events", int(a)),
                tag="det_input_min_events",
            )
            dpg.add_button(label="?", width=22, height=22, tag="min_events_info_btn")
            dpg.bind_item_theme("min_events_info_btn", info_button_theme)
            with dpg.popup("min_events_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "A full run keeps a cluster only if it holds at least\n"
                    "this many events in total (over all its time steps)."
                )
        add_setting_field("retire_idle_steps", width=150, info_theme=info_button_theme)
        dpg.add_text("Track repair", color=(170, 170, 170))
        for key in ("repair_max_distance", "repair_max_gap"):
            add_setting_field(key, width=150, info_theme=info_button_theme)

        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="enable multirun",
                default_value=bool(self.params.get("multirun", False)),
                callback=self._on_multirun_change,
                tag="det_input_multirun",
            )
            dpg.add_button(label="?", width=22, height=22, tag="multi_info_btn")
            dpg.bind_item_theme("multi_info_btn", info_button_theme)
        
            with dpg.popup("multi_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "If activated, a second detection run\n"
                    "is performed on remaining events\n"
                    "that were not associated to clusters\n"
                    "in the first run."
                )

        with dpg.group(
            tag="det_group_multitimefactor",
            show=bool(self.params.get("multirun", False)), 
        ):
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="time factor",
                    default_value=int(self.params.get("multitimefactor", 2)),
                    min_value=1,
                    min_clamped=True,
                    width=150,
                    callback=lambda s, a, u: self.params.__setitem__("multitimefactor", int(a)),
                    tag="det_input_multitimefactor",
                )
                dpg.add_button(label="?", width=22, height=22, tag="time_info_btn")
                dpg.bind_item_theme("time_info_btn", info_button_theme)
            
                with dpg.popup("time_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "In the second run, the accumulation time\n"
                        "is multiplied by this factor to find \n"
                        "slower particles."
                    )
        
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="CPU threads (0 = auto)",
                default_value=int(self.params.get("cpu_threads", 0)),
                min_value=0,
                max_value=parallel.available(),
                min_clamped=True,
                max_clamped=True,
                width=150,
                callback=self._on_cpu_threads_change,
                tag="det_input_cpu_threads",
            )
            dpg.add_button(label="?", width=22, height=22, tag="cpu_threads_info_btn")
            dpg.bind_item_theme("cpu_threads_info_btn", info_button_theme)
            with dpg.popup("cpu_threads_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Threads used by the compiled detection code.\n"
                    f"0 = automatic: {parallel.auto()} of the {parallel.available()} logical CPUs; the others\n"
                    "prepare the next time steps and keep the\n"
                    "interface responsive."
                )

        with dpg.group(tag=self._group_matching_dbscan, show=False):
            dpg.add_input_float(
                label="allowed distance for cluster matching",
                default_value=float(self.params.get("threshold", 7.5)),
                min_value=0.0,
                min_clamped=True,
                width=200,
                format="%.3f",
                callback=lambda s, a, u: self.params.__setitem__("threshold", float(a)),
                tag="det_input_dbscan_threshold",
            )
        
        dpg.add_separator()
        dpg.add_button(
            label="Run detection",
            tag=self._run_button_tag,
            width=400,
            callback=self._on_run_detection_clicked,
        )
        
        dpg.add_button(
            label="Stop detection",
            tag="det_stop_button",
            enabled=False,
            callback=self._on_stop_detection_clicked,
        )
        
        dpg.add_text("", tag=self._run_status_tag, wrap=400, color=(100, 255, 100))
        
        dpg.add_text("Detection stats:", color=(200, 200, 100))
        with dpg.child_window(
            tag="det_run_stats_box",
            height=110,
            autosize_x=True,
            border=True,
        ):
            dpg.add_text("No results yet.", tag="det_run_stats", wrap=430)
            
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Clear results",
                width=200,
                callback=lambda: self._on_clear_results("run"),
                tag="det_button_clear_run_results",
            )
        
        dpg.add_separator()
        dpg.add_text("EXPORT / IMPORT RESULTS", color=(100, 200, 255))
        
        dpg.add_text("Save detection results:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",
                tag="det_save_results_file_input",
                width=300,
                hint="Choose target file (.npz or .mat)",
                callback=self.on_det_save_results_path_typed,
                on_enter=True,
            )
            dpg.add_button(label="Browse", callback=self.browse_det_save_results_file, width=80)
            dpg.add_combo(
                items=[".npz", ".mat"],
                default_value=".npz",
                tag="det_save_results_format",
                width=70,
            )
        dpg.add_button(label="Save results", callback=self.save_detection_results, width=120)
        dpg.add_text("", tag="det_save_results_status", color=(100, 255, 100), wrap=430)
        
        dpg.add_spacer(height=6)
        
        dpg.add_text("Load detection results:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",
                tag="det_load_results_file_input",
                width=300,
                hint="Select results file to load",
                callback=self.on_det_load_results_path_typed,
                on_enter=True,
            )
            dpg.add_button(label="Browse", callback=self.browse_det_load_results_file, width=80)
        dpg.add_text("", tag="det_load_results_status", color=(100, 255, 100), wrap=430)
        
        if not dpg.does_item_exist("det_save_results_file_dialog"):
            with dpg.file_dialog(
                tag="det_save_results_file_dialog",
                label="Save detection results",
                directory_selector=False,
                default_path=default_files_dir(),
                show=False,
                callback=self.on_det_save_results_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("det_save_results_status", "Save cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
        
        if not dpg.does_item_exist("det_load_results_file_dialog"):
            with dpg.file_dialog(
                tag="det_load_results_file_dialog",
                label="Load detection results",
                directory_selector=False,
                default_path=default_files_dir(),
                show=False,
                callback=self.on_det_load_results_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("det_load_results_status", "Load cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
        
        
        current = self.params.get("detection_method", self._det_method_options[0])
        self._on_det_method_change(None, current, None)
        
        dpg.configure_item(
            "det_input_multitimefactor",
            enabled=bool(self.params.get("multirun", False))
        )
        
     
    def _build_detection_plot_panel(self):
        pseudo_texture = (
            self.preprocessing_tab.pseudoframe_texture_tag
            if self.preprocessing_tab is not None
            else self._local_pseudoframe_texture_tag
        )
        
        w, h = self.pseudoframe_dimensions
        with dpg.drawlist(width=w, height=h, tag="det_pf_drawlist"):
            dpg.draw_image(
                pseudo_texture,
                pmin=[0, 0],
                pmax=[w, h],
                tag="det_pf_drawimage",
            )
        self._view_zoom.register("det_pf_drawlist", lambda uv_min, uv_max: dpg.configure_item(
            "det_pf_drawimage", uv_min=uv_min, uv_max=uv_max))
        dpg.add_text("Load data in Preprocessing to generate pseudo-frame.", tag=self._status_tag, wrap=730)

        if self.preprocessing_tab is not None:
            self.preprocessing_tab.timeline.build("detection", time_info_tag=self._time_info_tag)
        else:
            dpg.add_text("", tag=self._time_info_tag, wrap=600)

        if self.preprocessing_tab is not None:
            self.preprocessing_tab.register_pseudoframe_mirror(
                status_tag=self._status_tag,
                time_info_tag=self._time_info_tag,
            )
        
        dpg.add_separator()
        dpg.add_text("Status Log", color=(200, 200, 100))
        with dpg.child_window(label="Status Log", height=90, width=720):
            dpg.add_input_text(
                tag="det_status_log_text",
                multiline=True,
                readonly=True,
                width=695,
                height=75,
            )
    
    def _build_new_plot_panel(self):
        if not dpg.does_item_exist("det_cluster_plot_texture"):
            with dpg.texture_registry(show=False):
                dpg.add_static_texture(
                    width=1,
                    height=1,
                    default_value=[0, 0, 0, 255],
                    tag="det_cluster_plot_texture",
                )
        
        with dpg.child_window(
            tag="det_cluster_plot_container",
            autosize_x=True,
            autosize_y=True,
            border=False,
        ):
            dpg.add_text("", tag="det_cluster_time_info", wrap=600)
            dpg.add_image(
                texture_tag="det_cluster_plot_texture",
                tag="det_cluster_plot_image",
            )
        self._view_zoom.register("det_cluster_plot_image", lambda uv_min, uv_max: dpg.configure_item(
            "det_cluster_plot_image", uv_min=uv_min, uv_max=uv_max))
        if self.preprocessing_tab is not None:  # time label follows the displayed frame, like the pseudo-frame's
            self.preprocessing_tab.register_pseudoframe_mirror(status_tag="", time_info_tag="det_cluster_time_info")
            self.preprocessing_tab.register_geometry_listener(self._on_geometry)

    def _on_geometry(self, width, height, display_w, display_h):
        """Sensor size from Preprocessing (file, typed or default): detection runs and views use it."""
        self.params["width"], self.params["height"] = int(width), int(height)
        if dpg.does_item_exist("det_pf_drawlist"):
            dpg.configure_item("det_pf_drawlist", width=display_w, height=display_h)
        if dpg.does_item_exist("det_pf_drawimage"):
            dpg.configure_item("det_pf_drawimage", pmax=[display_w, display_h])
        if dpg.does_item_exist("det_cluster_plot_image"):
            dpg.configure_item("det_cluster_plot_image", width=display_w, height=display_h)
        
    def _build_duration_hist_panel(self):
        with dpg.child_window(
            tag="det_duration_hist_container",
            autosize_x=True,
            autosize_y=True,
            border=False,
        ):
            dpg.add_image(
                texture_tag=self.duration_hist_texture_tag,
                tag="det_duration_hist_image",
            )

    def _build_engine_context(self) -> dict[str, Any]:
        if self.preprocessing_tab is None:
            raise RuntimeError("PreprocessingTab ist nicht verbunden – keine Daten für Detection verfügbar.")

        method = self.params.get("detection_method", "pixelwise extension")
        pixelwise_extension = (method == "pixelwise extension")
        use_dbscan = (method == "DBSCAN")
        pseudo_images = (method == "pseudo-frame")
        
        if pseudo_images:
            lightweight_mode = True
        else:
            lightweight_mode = False
    
        db_clustering = use_dbscan
        
        pt = self.preprocessing_tab

        events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
        X_global, Y_global, T_global, _ = events
        
        dt = pt.params["accumulation_time_ms"]*1000
        
        if self.params.get("multirun", False):
            multiN = 2
        else:
            multiN = 1

        ctx = {
            "X_global": X_global,
            "Y_global": Y_global,
            "T_global": T_global,
    
            "pixelwise_extension": pixelwise_extension,
            "pseudo_images": pseudo_images,
            "db_clustering": db_clustering,
    
            "dt": dt,                 
            "N": int(self.params.get("N", 1)),
            "multiN": multiN,
            "overlap": float(self.params.get("overlap", 0.4)),
            "structure_size": int(self.params.get("structure_size", 3)),
            "min_steps": int(self.params.get("min_steps", 3)),
            "min_events": int(self.params.get("min_events", 20)),
    
            "height": self.params["width"],
            "width": self.params["height"],
    
            "lightweight_mode": lightweight_mode,
            "save_mode": True,
            "stop_event": self._stop_event,
    
            "apply_hdbscan": False,
            "epsilon": float(self.params.get("epsilon", 7.5)),
            "minPts": int(self.params.get("minPts", 20)),
    
            "Range": float(self.params.get("epsilon", 7.5)),
            "Lmin": int(self.params.get("minPts", 20)),
    
            "threshold": float(self.params.get("threshold", 7.5)),
    
            "area": int(self.params.get("area", 5)),
            "filtersize": int(self.params.get("filtersize", 3)),
    
            "N_pixelwise": int(self.params.get("N_pixelwise", 5)),
    
            "num_points": 1,
    
            "multirun": bool(self.params.get("multirun", False)),
            "multitimefactor": int(self.params.get("multitimefactor", 1)),
        }
        # settings of the page and of the Settings window (config/settings.py): cluster rules, track repair,
        # buffer, steps_rev, search_factor, max_tracks, max_length, ...
        ctx.update(STORE.values("Detection"))

        # part of the recording (timeline): from the start mark, or the displayed step, to the end mark or the
        # end of the recording. The steps start there and use no event before it (not even in the overlap of the
        # first step); at most its number of steps. From the first step: as before (no start time given).
        r = pt.timeline.run_steps()
        if r is not None and not r[3]:
            if r[0] > float(T_global[0]):
                ctx["t_start"] = ctx["t_floor"] = r[0]
            ctx["N"] = min(int(ctx["N"]), r[2])
        return ctx

    @staticmethod
    def _run_time_range(ctx: dict) -> tuple[float, float]:
        """Part of the recording a run with `ctx` covers (us): its steps, from the first event or t_start."""
        T = ctx["T_global"]
        lo = float(T[0]) if ctx.get("t_start") is None else max(float(T[0]), float(ctx["t_start"]))
        return lo, lo + int(ctx["N"]) * float(ctx["dt"])

    def _on_range_changed(self) -> None:
        """Range of the recording changed on the timeline: the number of steps follows it."""
        self._n_user_set = False
        n = self._compute_default_N()
        self.params["N"] = n
        if dpg.does_item_exist("det_input_matching_N"):
            dpg.set_value("det_input_matching_N", n)

    def _on_test_run_clicked(self, sender=None, app_data=None, user_data=None):
        self._status_log_append("Test run started for 1 time step...")
        if dpg.does_item_exist("det_button_test_run"):
            dpg.configure_item("det_button_test_run", enabled=False)
        if dpg.does_item_exist("det_test_run_status"):
            dpg.set_value("det_test_run_status", "Running test detection...")
    
        try:
            ctx = self._build_engine_context()

            ctx["N"] = 1
            ctx["test_mode"] = True
            ctx.pop("t_floor", None)  # like the step in a full run: with the overlap events before it
            ctx["multiN"] = 1
            # detect in the displayed frame: the plots only show clusters of that frame
            pt = self.preprocessing_tab
            if pt is not None and pt.display_t0 is not None:
                ctx["t_start"] = float(pt.display_t0)
    
        except Exception as e:
            dpg.set_value("det_test_run_status", f"Input error: {e}")
            dpg.configure_item("det_button_test_run", enabled=True)
            return
    
        def worker():
            try:
                dpg.set_value("det_test_run_stats", "-")
                out = self.engine.run(ctx)
                self._status_log_append("Test run done.")
                try:
                    self._preview = (out, *self._full_run_preview(ctx, out))
                    self._status_log_append(f"[DETECTION] {self._preview[2]}")
                except Exception as e:
                    self._preview = None
                    self._status_log_append(f"[DETECTION] Full-run preview failed: {e}")
                self.last_result = out

                pt = self.preprocessing_tab
                events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
                X_global, Y_global, T_global, _ = events
                
                self._status_log_append("Statistics calculated.")
                self._update_pseudoframe_plot()
                height = self.params["width"]
                width = self.params["height"]
                
                acc_ms = float(getattr(pt, "params", {}).get("accumulation_time_ms", 2.0))
                step_us = acc_ms * 1000.0

                t0 = pt.display_t0
                if t0 is None:
                    t0 = float(np.min(T_global))  
                
                T_min = float(t0)
                T_max = T_min + step_us
                
                I = self._build_cluster_image(
                    out.x_clust, out.y_clust, out.t_clust,
                    height=height, width=width,
                    T_min=T_min, T_max=T_max,
                    dropped=self._dropped_clusters(out),
                )
                
                W, H = self.pseudoframe_dimensions
                self._ensure_cluster_texture_size(W, H)
                self._set_cluster_texture(I, W, H)
                
                try:
                    self._on_refresh_plots()
                except Exception:
                    pass
                
                self._status_log_append("Plots are ready.")

                stats_text = self._compute_detection_stats_text(out, ctx, mode="test")
                if self._preview is not None and self._preview[0] is out:
                    stats_text += "\n" + self._preview[2]
                if dpg.does_item_exist("det_test_run_stats"):
                    dpg.set_value("det_test_run_stats", stats_text)

                if dpg.does_item_exist("det_test_run_status"):
                    dpg.set_value("det_test_run_status", "Test run done.")
            except Exception as e:
                dpg.set_value("det_test_run_status", f"Test run failed: {e}")
                self._status_log_append(f"[DETECTION] Test run failed: {e}")
            finally:
                dpg.configure_item("det_button_test_run", enabled=True)
        threading.Thread(target=worker, daemon=True).start()

    def _on_run_detection_clicked(self, sender=None, app_data=None, user_data=None):
        self._stop_event.clear()
        self._run_active = True
        dpg.configure_item("det_stop_button", enabled=True)
        dpg.configure_item(self._run_button_tag, enabled=False)
        if dpg.does_item_exist("det_input_matching_N"):
            self.params["N"] = int(dpg.get_value("det_input_matching_N"))
        
        if dpg.does_item_exist("det_input_multirun"):
            self.params["multirun"] = bool(dpg.get_value("det_input_multirun"))
        
        if dpg.does_item_exist("det_input_multitimefactor"):
            self.params["multitimefactor"] = int(dpg.get_value("det_input_multitimefactor"))
        
        dpg.configure_item(self._run_button_tag, enabled=False)
        dpg.set_value(self._run_status_tag, "Running detection...")
        self._status_log_append(f"Run detection started (N={self.params['N']})")
    
        try:
            ctx = self._build_engine_context()
            ctx["test_mode"] = False
            self._last_run_ctx = ctx
            if self.preprocessing_tab.timeline.run_steps() is None:
                raise ValueError("nothing to run: the displayed step is after the end mark")
            time_range = self._run_time_range(ctx)
            self._status_log_append(f"Steps t = [{time_range[0]:.0f} .. {time_range[1]:.0f}] us ({ctx['N']} steps); "
                                    "Stop keeps the steps done")
            if dpg.does_item_exist("det_run_stats"):
                dpg.set_value("det_run_stats", "-")
        except Exception as e:
            dpg.set_value(self._run_status_tag, f"Input error: {e}")
            self._run_active = False
            dpg.configure_item(self._run_button_tag, enabled=True)
            dpg.configure_item("det_stop_button", enabled=False)  # no run started: nothing to stop
            return

        timeline = self.preprocessing_tab.timeline
        n_steps, dt_run = int(ctx["N"]), float(ctx["dt"])
        timeline.progress("detection", time_range[0], time_range[0], time_range[1],
                          text=f"Detection 0 %: step 0 of {n_steps}")

        def progress(done_range):
            k = int(round((done_range[1] - done_range[0]) / dt_run))
            text = (f"Detection {100 * k / max(n_steps, 1):.0f} %: step {k} of {n_steps}" if k < n_steps
                    else "Detection: all steps done, finishing (track repair)")
            timeline.progress("detection", done_range[0], done_range[1], time_range[1], text=text)

        def worker():
            stdout_writer = QueueWriter(self._log_queue)
            stderr_writer = QueueWriter(self._log_queue)

            try:
                with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
                    # (the callback goes to the engine only: the saved run parameters stay plain values)
                    out = self.engine.run(dict(ctx, progress=progress))
                    self.last_result = out
                r = getattr(out, "time_range", None)
                if r is not None and r[1] < time_range[1] - 1e-6:
                    self._log_queue.put(f"[DETECTION] Stopped: t = [{r[0]:.0f} .. {r[1]:.0f}] us processed "
                                        f"({int(round((r[1] - r[0]) / float(ctx['dt'])))} of {ctx['N']} steps).")

                self._log_queue.put("[DETECTION] Done.")
                pt = self.preprocessing_tab
                events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
                X_global, Y_global, T_global, _ = events
                height = self.params["width"]
                width = self.params["height"]
                
                acc_ms = float(getattr(pt, "params", {}).get("accumulation_time_ms", 2.0))
                step_us = acc_ms * 1000.0

                t0 = pt.display_t0
                if t0 is None:
                    t0 = float(np.min(T_global))  
                
                T_min = float(t0)
                T_max = T_min + step_us
                
                I = self._build_cluster_image(
                    out.x_clust, out.y_clust, out.t_clust,
                    height=height, width=width,
                    T_min=T_min, T_max=T_max,
                    dropped=self._dropped_clusters(out),
                )
                
                W, H = self.pseudoframe_dimensions
                self._ensure_cluster_texture_size(W, H)
                self._set_cluster_texture(I, W, H)
                
                try:
                    self._update_duration_histogram()
                except Exception:
                    pass
                
                self._status_log_append("Plots are ready.")

            except DetectionCancelled:
                self._log_queue.put("[DETECTION] Run cancelled by user.")
            except Exception as e:
                self._log_queue.put(f"[DETECTION] ERROR: {e}")
            finally:
                timeline.progress_end("detection")
                self._log_queue.put("__DONE__")
    
        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()
    
    def _on_N_change(self, sender, app_data, user_data):
        self._n_user_set = True
        self.params["N"] = int(app_data)
    
    def _compute_default_N(self) -> int:
        pt = getattr(self, "preprocessing_tab", None)
        if pt is None:
            return 1

        events = None
        fe = getattr(pt, "filtered_events", None)
        re = getattr(pt, "raw_events", None)

        def _usable(ev: object) -> bool:
            if ev is None:
                return False
            try:
                _, _, T, _ = ev
                return T is not None and len(T) > 0
            except Exception:
                return False

        if _usable(fe):
            events = fe
        elif _usable(re):
            events = re

        if events is None:
            return 1

        r = pt.timeline.run_steps() if hasattr(pt, "timeline") else None
        if r is not None:  # from the start mark or the displayed step to the end mark or the end
            return int(r[2])

        _, _, T, _ = events

        acc_ms = None
        try:
            acc_ms = float(getattr(pt, "params", {}).get("accumulation_time_ms", 0))
        except Exception:
            acc_ms = 0

        acc_us = acc_ms * 1000.0
        if acc_us <= 0:
            return 1

        tmax = float(T[-1])   # preprocessing keeps the events time sorted
        tmin = float(T[0])
        N = int(math.ceil((tmax-tmin) / acc_us))
        N = max(1, N)
        return N
    
    def _on_det_method_change(self, sender, app_data, user_data):
        self.params["detection_method"] = app_data
    
        dpg.configure_item(self._group_pixelwise, show=(app_data == "pixelwise extension"))
        dpg.configure_item(self._group_kdtree,    show=(app_data == "kd-tree"))
        dpg.configure_item(self._group_DBSCAN,    show=(app_data == "DBSCAN"))
        dpg.configure_item(self._group_pseudo,    show=(app_data == "pseudo-frame"))
    
        dpg.configure_item(
            self._group_matching_dbscan,
            show=(app_data in ("DBSCAN", "pseudo-frame"))
        )
    
    def _on_cpu_threads_change(self, sender, app_data, user_data):
        self.params["cpu_threads"] = int(app_data)
        n = parallel.set_threads(int(app_data))
        self._status_log_append(f"CPU threads for detection: {n}")

    def _on_multirun_change(self, sender, app_data, user_data):
        enabled = bool(app_data)
        self.params["multirun"] = enabled
    
        if dpg.does_item_exist("det_group_multitimefactor"):
            dpg.configure_item("det_group_multitimefactor", show=enabled)
    
        if dpg.does_item_exist("det_input_multitimefactor"):
            dpg.configure_item("det_input_multitimefactor", enabled=enabled)
        
    def on_tab_selected(self) -> None:
        """Hook called by LayoutManager when user switches to this tab."""
        self.refresh_defaults_from_preprocessing()
            
    def _on_tab_visible(self, sender, app_data, user_data):
        if self._refresh_in_progress:
            return
    
        self._refresh_in_progress = True
        try:
            self.refresh_defaults_from_preprocessing()
            if self._plot_signature() != self._plotted_signature:
                self._on_refresh_plots()
        finally:
            self._refresh_in_progress = False

    def _plot_signature(self):
        """Inputs of the plots: the events shown by Preprocessing (filtered or raw), the displayed frame, the
        accumulation time and the detection results. Cheap: compared on every rendered frame of the tab, the
        plots are redrawn when it changes."""
        pt = self.preprocessing_tab
        if pt is None:
            return (id(self.last_result),)
        events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
        T = events[2] if events is not None else None
        return (id(T), 0 if T is None else len(T), pt.display_t0,
                float(pt.params.get("accumulation_time_ms", 0)), id(self.last_result))
            
            
    def refresh_defaults_from_preprocessing(self):
        pt = self.preprocessing_tab
        fe = getattr(pt, "filtered_events", None)
        re = getattr(pt, "raw_events", None)

        sig = None
        T = None
        for name, ev in (("filtered", fe), ("raw", re)):
            if ev is None:
                continue
            try:
                _, _, Ttmp, _ = ev
                if Ttmp is None or getattr(Ttmp, "size", len(Ttmp)) == 0:
                    continue
                # cheap signature: this handler runs on every rendered frame while the tab is shown
                sig = (name, id(Ttmp), int(len(Ttmp)), float(pt.params.get("accumulation_time_ms", 0)),
                       pt.timeline.start, pt.timeline.end, pt.timeline.index() if pt.timeline.start is None else 0)
                T = Ttmp
                break
            except Exception:
                continue
    
        if sig is None:
            if self._last_refresh_signature != "NO_EVENTS":
                self._last_refresh_signature = "NO_EVENTS"
            return
    
        if sig == self._last_refresh_signature:
            return
    
        self._last_refresh_signature = sig

        if getattr(self, "_n_user_set", False):
            return

        acc_ms = float(pt.params.get("accumulation_time_ms", 0))
        acc_us = acc_ms * 1000.0
        if acc_us <= 0:
            return
    
        new_N = self._compute_default_N()

        self.params["N"] = new_N
        if dpg.does_item_exist("det_input_matching_N"):
            dpg.set_value("det_input_matching_N", new_N)

    def _count_events_in_test_step(self, X_global, Y_global, T_global, dt_us: float, t_start=None) -> int:
        if T_global is None or len(T_global) == 0:
            return 0
        T = np.asarray(T_global)
        t_min = float(np.min(T)) if t_start is None else max(float(T[0]), float(t_start))
        mask = (T >= t_min) & (T <= (t_min + float(dt_us)))
        return int(np.count_nonzero(mask))
    
    
    def _count_events_in_steps(self, T_global, dt_us: float, N: int, t_start=None) -> int:
        if T_global is None or len(T_global) == 0:
            return 0
        T = np.asarray(T_global, dtype=float)
        t_min = float(np.min(T)) if t_start is None else max(float(T[0]), float(t_start))
        t_max = t_min + float(N) * float(dt_us)
        mask = (T >= t_min) & (T <= t_max)
        return int(np.count_nonzero(mask))

    def _compute_detection_stats_text(self, out, ctx: dict, *, mode: str) -> str:
        valid_idx = self._valid_cluster_indices(out.x_clust)
        n_clusters = len(valid_idx)

        if isinstance(out.ind_clust, PackedTracks):  # same numbers, computed on the flat array
            sel = out.ind_clust[np.asarray(valid_idx, dtype=np.int64)]
            events_per_cluster = sel.distinct_per_track()
            avg_events = float(np.mean(events_per_cluster)) if events_per_cluster.size else 0.0
            unique_clustered_events = sel.distinct_total()
        else:
            events_per_cluster = []
            all_inds = []

            for i in valid_idx:
                a = np.asarray(out.ind_clust[i]).ravel()
                if a.size == 0:
                    events_per_cluster.append(0)
                    continue
                a = a.astype(int, copy=False)
                au = np.unique(a)
                events_per_cluster.append(int(au.size))
                all_inds.append(au)

            avg_events = float(np.mean(events_per_cluster)) if events_per_cluster else 0.0

            unique_clustered_events = int(np.unique(np.concatenate(all_inds)).size) if all_inds else 0
        sum_clustered_events = unique_clustered_events

        pt = self.preprocessing_tab
        events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
        X_global, Y_global, T_global, _ = events

        dt_us = float(ctx.get("dt", 0))
        if mode == "test":
            n_total = self._count_events_in_test_step(X_global, Y_global, T_global, dt_us, ctx.get("t_start"))
        else:
            N = int(ctx.get("N", 1))
            n_total = self._count_events_in_steps(T_global, dt_us, N, ctx.get("t_start"))

        ratio = (sum_clustered_events / n_total) if n_total > 0 else 0.0

        stats_text = (
            f"clusters found: {n_clusters}\n"
            f"avg. events per cluster: {avg_events:.2f}\n"
            f"ratio of clustered events: {ratio:.3f}"
        )
        return stats_text
        
    def _count_non_nan(self, arr) -> int:
        """Zählt Elemente, ignoriert NaN (robust für list/np.array, object dtypes)."""
        if arr is None:
            return 0
        a = np.asarray(arr).ravel()
        if a.size == 0:
            return 0
        try:
            af = a.astype(float)
            return int(np.sum(~np.isnan(af)))
        except Exception:
            c = 0
            for v in a:
                if v is None:
                    continue
                if isinstance(v, str) and v.strip().lower() == "nan":
                    continue
                try:
                    if np.isnan(v):
                        continue
                except Exception:
                    pass
                c += 1
            return c
    
    def render_plot_to_texture(self, fig, texture_tag: str) -> None:
        dpg.set_value(texture_tag, figure_to_rgba_flat(fig))
    
    def _valid_cluster_indices(self, x_clust) -> list[int]:
        """Cluster gelten als valid, wenn x_clust mindestens einen Nicht-NaN enthält."""
        if isinstance(x_clust, PackedTracks):
            return np.flatnonzero(x_clust.count_finite() > 0).tolist()
        valid = []
        for i, arr in enumerate(x_clust):
            if self._count_non_nan(arr) > 0:
                valid.append(i)
        return valid   
    
    def _set_cluster_texture(self, I, W, H):
        # same picture as pcolor(I, cmap="viridis"): row 0 at the bottom, colours scaled to [min, max]
        I = np.asarray(I)
        rgba = np.zeros((H, W, 4), np.float32)
        img = resample_nearest(scalar_rgba(I[::-1]), H, W)  # sensor-sized image onto the texture
        h, w = min(H, img.shape[0]), min(W, img.shape[1])
        rgba[:h, :w] = img[:h, :w]
        grey = self._cluster_grey_mask
        if grey is not None and grey.shape == I.shape:  # clusters a full run would discard (test-run preview)
            rgba[:h, :w][resample_nearest(grey[::-1], H, W)[:h, :w]] = (0.55, 0.55, 0.55, 1.0)
        dpg.set_value(self.cluster_texture_tag, rgba.ravel())

    def _clusters_in_window(self, t_clust, T_min, T_max):
        """Indices of the clusters with events in [T_min, T_max] (per-cluster time ranges cached)."""
        cache = getattr(self, "_cluster_trange_cache", None)
        if cache is None or cache[0] is not t_clust or cache[1] != len(t_clust):
            n = len(t_clust)
            if isinstance(t_clust, PackedTracks):  # one compiled pass over the flat array
                tmin, tmax = t_clust.finite_min_max()
            else:
                tmin = np.full(n, np.inf)
                tmax = np.full(n, -np.inf)
                for i, tc in enumerate(t_clust):
                    if tc is None:
                        continue
                    t = np.asarray(tc, dtype=float).ravel()
                    if t.size == 0:
                        continue
                    t = t[np.isfinite(t)]
                    if t.size:
                        tmin[i] = t.min()
                        tmax[i] = t.max()
            cache = (t_clust, n, tmin, tmax)
            self._cluster_trange_cache = cache
        _, _, tmin, tmax = cache
        return np.flatnonzero((tmax >= T_min) & (tmin <= T_max))

    def _dropped_clusters(self, out):
        """Clusters of `out` that a full run would discard (test-run preview), or None."""
        if self._preview is not None and self._preview[0] is out:
            return self._preview[1]
        return None

    def _full_run_preview(self, test_ctx, test_out):
        """Which clusters of a test run (one time step) a full run would discard.

        A full run keeps a cluster only if it was followed over at least min_steps steps and holds at least
        min_events events: one step cannot tell. The full-run logic is run on the steps around the displayed one
        (min_steps + 1 on each side: any run of min_steps steps through the displayed one lies inside), and a test
        cluster counts as kept when most of its events belong to a cluster that this run keeps. Approximate: the
        clusters are cut at the window border (fewer events and steps than over the whole recording).
        Returns (indices of the discarded test clusters, summary text)."""
        min_steps = int(self.params.get("min_steps", 3))
        margin = min_steps + 1
        dt = float(test_ctx["dt"])
        ctx = self._build_engine_context()
        ctx["test_mode"] = False
        ctx["N"] = 2 * margin + 1
        ctx["t_start"] = float(test_ctx.get("t_start", 0.0)) - margin * dt
        full = self.engine.run(ctx)
        kept = [np.asarray(a).ravel() for a in full.ind_clust if len(a)]
        kept = np.unique(np.concatenate(kept)) if kept else np.zeros(0, dtype=np.int64)
        dropped = set()
        n = 0
        for i, ind in enumerate(test_out.ind_clust):
            ind = np.asarray(ind).ravel()
            if ind.size == 0:
                continue
            n += 1
            if np.count_nonzero(np.isin(ind, kept)) * 2 < ind.size:
                dropped.add(i)
        summary = (f"Full-run preview: {n - len(dropped)} of {n} clusters kept, {len(dropped)} discarded (grey) "
                   f"by the min. {min_steps} steps / {int(self.params.get('min_events', 20))} events rules")
        return dropped, summary

    def _build_cluster_image(self, x_clust, y_clust, t_clust, height, width, T_min, T_max, dropped=None):
        """Cluster labels image; the pixels of the `dropped` clusters go to self._cluster_grey_mask instead."""
        I = np.zeros((width, height), dtype=np.uint8)
        grey = np.zeros((width, height), dtype=bool) if dropped else None

        for ci in self._clusters_in_window(t_clust, T_min, T_max):
            xc, yc, tc = x_clust[ci], y_clust[ci], t_clust[ci]
            if xc is None or yc is None or tc is None:
                continue
    
            x = np.asarray(xc, dtype=float).ravel()
            y = np.asarray(yc, dtype=float).ravel()
            t = np.asarray(tc, dtype=float).ravel()
    
            if x.size == 0 or y.size == 0 or t.size == 0:
                continue

            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
            m &= (t >= T_min) & (t <= T_max)
            if not np.any(m):
                continue

            xi = x[m].astype(int) - 1
            yi = y[m].astype(int) - 1

            v = (xi >= 0) & (xi < height) & (yi >= 0) & (yi < width)
            xi = xi[v]
            yi = yi[v]
            if xi.size == 0:
                continue

            if grey is not None and ci in dropped:
                grey[-yi + (width - 1), xi] = True
                continue
            color = np.random.randint(1, 256)
            I[-yi + (width - 1), xi] = color

        self._cluster_grey_mask = grey
        return I
    
    def _ensure_cluster_texture_size(self, width: int, height: int) -> None:
        registry_tag = "texture_registry_main"
        if not dpg.does_item_exist(registry_tag):
            dpg.add_texture_registry(tag=registry_tag)
    
        if dpg.does_item_exist(self.cluster_texture_tag):
            cfg = dpg.get_item_configuration(self.cluster_texture_tag)
            if int(cfg.get("width", 0)) == int(width) and int(cfg.get("height", 0)) == int(height):
                return

            if dpg.does_item_exist("det_cluster_plot_image"):
                pass
    
            dpg.delete_item(self.cluster_texture_tag)
    
        data = np.zeros((height, width, 4), dtype=np.float32)
    
        dpg.add_raw_texture(
            width=width,
            height=height,
            default_value=data.flatten(),
            format=dpg.mvFormat_Float_rgba,
            tag=self.cluster_texture_tag,
            parent=registry_tag,
        )
    
        if dpg.does_item_exist("det_cluster_plot_image"):
            dpg.configure_item("det_cluster_plot_image", texture_tag=self.cluster_texture_tag)
       
    def _on_show_scatter_toggle(self, sender, app_data, user_data):
        self.params["show_scatter"] = bool(app_data)
        self._update_pseudoframe_plot()
            
    def _update_pseudoframe_plot(self):
        if not dpg.does_item_exist("det_pf_drawimage"):
            return
    
        base_texture = (
            self.preprocessing_tab.pseudoframe_texture_tag
            if self.preprocessing_tab is not None
            else self._local_pseudoframe_texture_tag
        )
    
        show = bool(self.params.get("show_scatter", False))
    
        if not show or self.last_result is None or self.preprocessing_tab is None:
            dpg.configure_item("det_pf_drawimage", texture_tag=base_texture)
            return
    
        pt = self.preprocessing_tab

        events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
        if events is None:
            dpg.configure_item("det_pf_drawimage", texture_tag=base_texture)
            return
        X_all, Y_all, T_all, _ = events

        acc_ms = float(getattr(pt, "params", {}).get("accumulation_time_ms", 2.0))
        step_us = acc_ms * 1000.0
        
        t0 = pt.display_t0
        if t0 is None:
            t0 = float(np.min(T_all)) 
        
        T_min = float(t0)
        T_max = T_min + step_us
        
        lo, hi = np.searchsorted(T_all, [T_min, T_max], side="right")  # events are time sorted
        X = X_all[lo:hi].astype(np.int64)
        Y = Y_all[lo:hi].astype(np.int64)
    
        W, H = int(self.params["width"]), int(self.params["height"])  # sensor size
        valid = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        X, Y = X[valid], Y[valid]
        
        I = np.zeros((H, W), dtype=np.float32)
        I[Y.astype(np.intp), X.astype(np.intp)] = 1.0
    
        method = self.params.get("detection_method", "pixelwise extension")
        out = self.last_result
        
        if method in ("pixelwise extension", "DBSCAN", "kd-tree"):
            xs, ys = self._cluster_means_in_window(out, T_min, T_max)
        else:
            xs, ys = self._flatten_xy_m(self.last_result.x_m, self.last_result.y_m)
            xs, ys = self._collect_centers_in_window(out, T_min, T_max)
        
    
        # pcolor(I, PSEUDOFRAME_CMAP) + inverted y axis == the image itself, row 0 on top
        rgba = binary_rgba(I, PSEUDOFRAME_CMAP.colors[0], PSEUDOFRAME_CMAP.colors[-1])
        if bool(self.params.get("show_scatter", False)) and self.last_result is not None:
            if len(xs) > 0:
                draw_dots(rgba, xs, ys, radius=2.0, rgb=(1.0, 0.0, 0.0), alpha=0.8)

        tw, th = self.pseudoframe_dimensions  # the overlay texture has the pseudo-frame texture size
        dpg.set_value(self._overlay_pseudoframe_texture_tag, resample_nearest(rgba, th, tw).ravel())

        dpg.configure_item("det_pf_drawimage", texture_tag=self._overlay_pseudoframe_texture_tag)
        
    def _flatten_xy_m(self, x_m, y_m):
        xs = []
        ys = []
    
        if isinstance(x_m, np.ndarray) and isinstance(y_m, np.ndarray):
            mask = (~np.isnan(x_m)) & (~np.isnan(y_m))
            xs = x_m[mask].astype(float).tolist()
            ys = y_m[mask].astype(float).tolist()
            return xs, ys
    
        for xm, ym in zip(x_m, y_m):
            if xm is None or ym is None:
                continue
            xm = np.asarray(xm, dtype=float).ravel()
            ym = np.asarray(ym, dtype=float).ravel()
            if xm.size == 0 or ym.size == 0:
                continue
            m = (~np.isnan(xm)) & (~np.isnan(ym))
            if np.any(m):
                xs.extend(xm[m].tolist())
                ys.extend(ym[m].tolist())
    
        return xs, ys
     
        
    def _collect_centers_in_window(self, out, T_min, T_max):
        xs, ys = [], []
    
        x_m, y_m, t_m = out.x_m, out.y_m, out.t_m
        t_m = self._tm_to_us(t_m)

        if isinstance(x_m, np.ndarray) and isinstance(t_m, np.ndarray):
            m = np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(t_m)
            m &= (t_m >= T_min) & (t_m <= T_max)
            return x_m[m].astype(float).tolist(), y_m[m].astype(float).tolist()
    
        for xm, ym, tm in zip(x_m, y_m, t_m):
            if xm is None or ym is None or tm is None:
                continue
            xm = np.asarray(xm, dtype=float).ravel()
            ym = np.asarray(ym, dtype=float).ravel()
            tm = np.asarray(tm, dtype=float).ravel()
            if xm.size == 0 or tm.size == 0:
                continue
    
            m = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(tm)
            m &= (tm >= T_min) & (tm <= T_max)
            if np.any(m):
                xs.extend(xm[m].tolist())
                ys.extend(ym[m].tolist())
    
        return xs, ys
   
    def _tm_to_us(self, t_m):
        if t_m is None:
            return None
    
        def _conv(arr):
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                return a
            m = np.nanmax(a)
            return a * 1e6 if m < 1e4 else a
    
        if isinstance(t_m, np.ndarray):
            return _conv(t_m)
    
        if isinstance(t_m, (list, tuple)):
            return [_conv(a) if a is not None else np.array([], dtype=float) for a in t_m]
    
        try:
            v = float(t_m)
            return v * 1e6 if v < 1e4 else v
        except Exception:
            return t_m 
        
    def _cluster_means_in_window(self, out, T_min: float, T_max: float):
        xs, ys = [], []
    
        x_clust = out.x_clust
        y_clust = out.y_clust
        t_clust = out.t_clust

        for ci in self._clusters_in_window(t_clust, T_min, T_max):
            xc, yc, tc = x_clust[ci], y_clust[ci], t_clust[ci]
            if xc is None or yc is None or tc is None:
                continue
    
            x = np.asarray(xc, dtype=float).ravel()
            y = np.asarray(yc, dtype=float).ravel()
            t = np.asarray(tc, dtype=float).ravel()
    
            if x.size == 0 or y.size == 0 or t.size == 0:
                continue
    
            m = (t >= T_min) & (t <= T_max)
            m &= np.isfinite(x) & np.isfinite(y)
    
            if not np.any(m):
                continue
    
            xs.append(float(np.mean(x[m])))
            ys.append(float(np.mean(y[m])))
    
        return xs, ys
      
    def _mouse_inside_item(self, item_tag: str) -> bool:
        if not dpg.does_item_exist(item_tag):
            return False
        mx, my = dpg.get_mouse_pos(local=False)
        x0, y0 = dpg.get_item_rect_min(item_tag)
        x1, y1 = dpg.get_item_rect_max(item_tag)
        return (x0 <= mx <= x1) and (y0 <= my <= y1)
    
    def _status_log_set(self, text: str) -> None:
        if dpg.does_item_exist("det_status_log_text"):
            dpg.set_value("det_status_log_text", str(text))
        
    def _status_log_append(self, line: str) -> None:
        if not dpg.does_item_exist("det_status_log_text"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {line}"
        current = dpg.get_value("det_status_log_text") or ""
        dpg.set_value("det_status_log_text", (line + ("\n" if current else "") + current))
        
    def _on_stop_detection_clicked(self, sender=None, app_data=None, user_data=None):
        self._status_log_append("[DETECTION] Stop requested.")
        self._stop_event.set()
        
    def process_background_tasks(self):
        while not self._log_queue.empty():
            line = self._log_queue.get_nowait()
    
            if line == "__DONE__":
                if dpg.does_item_exist(self._run_status_tag):
                    dpg.set_value(self._run_status_tag, "Detection done.")
                self._run_active = False
                dpg.configure_item(self._run_button_tag, enabled=True)
                dpg.configure_item("det_stop_button", enabled=False)
                
                try:
                    self._on_refresh_plots()
                except Exception as e:
                    self._status_log_append(f"[DETECTION] Refresh plots after run failed: {e}")
                
                try:
                    if self.last_result is not None and self._last_run_ctx is not None:
                        stats_text = self._compute_detection_stats_text(
                            self.last_result, self._last_run_ctx, mode="run"
                        )
                        if dpg.does_item_exist("det_run_stats"):
                            dpg.set_value("det_run_stats", stats_text)
                except Exception as e:
                    self._status_log_append(f"[DETECTION] Stats failed: {e}")

                continue
            self._status_log_append(line)

    def _on_clear_results(self, mode: str):
        if mode == "test":
            self.last_result = None

            if dpg.does_item_exist("det_test_run_stats"):
                dpg.set_value("det_test_run_stats", "No results yet.")

        elif mode == "run":
            self.last_result = None
            self._last_run_ctx = None

            if dpg.does_item_exist("det_run_stats"):
                dpg.set_value("det_run_stats", "No results yet.")

            if dpg.does_item_exist(self._run_status_tag):
                dpg.set_value(self._run_status_tag, "Done.")

        else:
            return  

        self.last_result = None

        try:
            self._clear_cluster_plot()
        except Exception:
            pass
        
        try:
            self._update_duration_histogram()
        except Exception:
            pass

        try:
            self._update_detection_overlay()
        except Exception:
            pass

        self._status_log_append(
            f"[DETECTION] Cleared {mode} results."
        )
        
    def _clear_cluster_plot(self):
        if dpg.does_item_exist("det_cluster_plot_image"):
            dpg.configure_item("det_cluster_plot_image", texture_tag=self.cluster_texture_tag)
    
        if dpg.does_item_exist(self.cluster_texture_tag):
            cfg = dpg.get_item_configuration(self.cluster_texture_tag)
            w = int(cfg.get("width", 2))
            h = int(cfg.get("height", 2))
            empty = np.zeros((h, w, 4), dtype=np.float32)  
            dpg.set_value(self.cluster_texture_tag, empty.flatten())
          
    def _on_refresh_plots(self, sender=None, app_data=None, user_data=None):
        self._plotted_signature = self._plot_signature()  # first: a failing refresh is not retried every frame
        try:
            self._update_pseudoframe_plot()
        except Exception as e:
            self._status_log_append(f"[DETECTION] Refresh pseudoframe failed: {e}")
    
        try:
            out = getattr(self, "last_result", None)
    
            height = int(self.params["width"])
            width  = int(self.params["height"])
    
            I = np.zeros((width, height), dtype=np.uint8)
    
            pt = self.preprocessing_tab
            events = None
            if pt is not None:
                events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
    
            if (
                out is not None
                and events is not None
                and hasattr(out, "x_clust") and hasattr(out, "y_clust") and hasattr(out, "t_clust")
            ):
                Xg, Yg, Tg, _ = events
                acc_ms = float(getattr(pt, "params", {}).get("accumulation_time_ms", 2.0))
                step_us = acc_ms * 1000.0
                t0 = pt.display_t0 if pt.display_t0 is not None else float(np.min(Tg))
                T_min = float(t0)
                T_max = T_min + step_us
    
                I = self._build_cluster_image(
                    out.x_clust, out.y_clust, out.t_clust,
                    height=height, width=width,
                    T_min=T_min, T_max=T_max,
                    dropped=self._dropped_clusters(out),
                )
    
            W, H = self.pseudoframe_dimensions
            self._ensure_cluster_texture_size(W, H)
            self._set_cluster_texture(I, W, H)
    
        except Exception as e:
            self._status_log_append(f"[DETECTION] Refresh cluster plot failed: {e}")

        try:
            self._update_duration_histogram()
        except Exception as e:
            self._status_log_append(f"[DETECTION] Refresh histogram failed: {e}")
    
        self._log_clusters_in_frame()

    def _log_clusters_in_frame(self):
        # tells why the views can be empty: they only show clusters of the displayed frame
        out = getattr(self, "last_result", None)
        pt = self.preprocessing_tab
        if out is None or pt is None or pt.display_t0 is None or not hasattr(out, "t_clust"):
            return
        t0 = float(pt.display_t0)
        t1 = t0 + float(pt.params.get("accumulation_time_ms", 2.0)) * 1000.0
        n = len(self._clusters_in_window(out.t_clust, t0, t1))
        msg = f"[DETECTION] {n} clusters in the displayed frame t = [{t0:.0f} .. {t1:.0f}] us"
        if n == 0:
            msg += " (none detected here: use Test run on this frame, or run detection over it)"
        self._status_log_append(msg)  
        
    def _update_duration_histogram(self):
        if not dpg.does_item_exist(self.duration_hist_texture_tag):
            return
    
        out = getattr(self, "last_result", None)
        sig = (id(out), self.params.get("detection_method"))
        if getattr(self, "_duration_hist_sig", None) == sig:
            return  # the histogram does not depend on the displayed frame
        self._duration_hist_sig = sig

        durations_us = []

        if out is not None:
            method = self.params.get("detection_method", "pixelwise extension")

            t_src = getattr(out, "t_clust", None)

            if isinstance(t_src, PackedTracks):  # durations of the tracks with at least 2 time stamps
                t_lo, t_hi = t_src.finite_min_max()
                durations_us = (t_hi - t_lo)[t_src.count_finite() >= 2].tolist()
            elif t_src is not None:
                if isinstance(t_src, (list, tuple)):
                    clusters = t_src
                else:
                    clusters = list(t_src)
    
                for t in clusters:
                    if t is None:
                        continue
                    tt = np.asarray(t, dtype=float)
                    tt = tt[~np.isnan(tt)]
                    if tt.size < 2:
                        continue
                    durations_us.append(float(np.max(tt) - np.min(tt)))
    
            if (len(durations_us) == 0) and (method == "pseudo-frame"):
                t_m = getattr(out, "t_m", None)
                if t_m is not None:
                    t_m_us = self._tm_to_us(t_m)
    
                    if isinstance(t_m_us, (list, tuple)):
                        clusters = t_m_us
                    else:
                        clusters = [t_m_us]
    
                    for t in clusters:
                        if t is None:
                            continue
                        tt = np.asarray(t, dtype=float)
                        tt = tt[~np.isnan(tt)]
                        if tt.size < 2:
                            continue
                        durations_us.append(float(np.max(tt) - np.min(tt)))
    
        W, H = 700, 580
        dpi = 100
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        
        fig.patch.set_facecolor((0.12, 0.12, 0.12))
        ax.set_facecolor((0.12, 0.12, 0.12))
        
        for spine in ax.spines.values():
            spine.set_color((0.7, 0.7, 0.7))
        
        ax.tick_params(colors=(0.85, 0.85, 0.85))
        ax.xaxis.label.set_color((0.9, 0.9, 0.9))
        ax.yaxis.label.set_color((0.9, 0.9, 0.9))
        ax.title.set_color((0.95, 0.95, 0.95))
        
        ax.grid(True, alpha=0.2)
    
        if len(durations_us) == 0:
            ax.text(0.5, 0.5, "No duration data.", ha="center", va="center", color=(0.9, 0.9, 0.9))
            ax.set_axis_off()
        else:
            d_ms = np.asarray(durations_us, dtype=float) / 1000.0
            ax.hist(d_ms, bins=100, edgecolor=(0.85, 0.85, 0.85))
            ax.set_xlabel("Cluster duration [ms]")
            ax.set_ylabel("Count")
            ax.set_title("Cluster duration histogram")
    
        fig.tight_layout(pad=0.5)
    
        dpg.set_value(self.duration_hist_texture_tag, figure_to_rgba_flat(fig))
        plt.close(fig)
        
    def on_det_save_results_path_typed(self, sender, app_data, user_data=None):
        try:
            dpg.set_value("det_save_results_file_input", str(app_data).strip())
        except Exception:
            pass

    def on_det_load_results_path_typed(self, sender, app_data, user_data=None):
        path = str(app_data).strip()
        try:
            dpg.set_value("det_load_results_file_input", path)
        except Exception:
            pass
        if path:  # Enter on a path loads it
            self.load_detection_results()

    def browse_det_save_results_file(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("det_save_results_file_dialog"):
            dpg.show_item("det_save_results_file_dialog")

    def browse_det_load_results_file(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("det_load_results_file_dialog"):
            dpg.show_item("det_load_results_file_dialog")

    def on_det_save_results_file_selected(self, sender, app_data, user_data=None):
        path = str(app_data.get("file_path_name", "")).strip()
        if not path:
            dpg.set_value("det_save_results_status", "No file selected.")
            return
        dpg.set_value("det_save_results_file_input", path)

    def on_det_load_results_file_selected(self, sender, app_data, user_data=None):
        path = str(app_data.get("file_path_name", "")).strip()
        if not path:
            dpg.set_value("det_load_results_status", "No file selected.")
            return
        dpg.set_value("det_load_results_file_input", path)
        self.load_detection_results()  # selecting a file loads it

    def _build_results_payload_bytes(self):
        if self.last_result is None:
            raise ValueError("No detection results available (last_result is None).")
        ctx = getattr(self, "_last_run_ctx", None)
        if ctx is None:
            raise ValueError("No ctx available (_last_run_ctx is None). Run detection first or load results that include ctx.")
        result_bytes = pickle.dumps(self.last_result, protocol=pickle.HIGHEST_PROTOCOL)
        ctx_params = self._extract_ctx_params(ctx)  
        ctx_bytes = pickle.dumps(ctx_params, protocol=pickle.HIGHEST_PROTOCOL)
        return result_bytes, ctx_bytes

    def save_detection_results(self, sender=None, app_data=None, user_data=None):
        try:
            path = str(dpg.get_value("det_save_results_file_input") or "").strip()
            ext = str(dpg.get_value("det_save_results_format") or ".npz").strip().lower()

            if not path:
                self.browse_det_save_results_file()
                return

            p = Path(path)
            if p.suffix.lower() not in [".npz", ".mat"]:
                p = p.with_suffix(ext)

            result_bytes, ctx_bytes = self._build_results_payload_bytes()
            meta = {"format_version": 1, "saved_at": datetime.now().isoformat(timespec="seconds")}

            if p.suffix.lower() == ".npz":
                np.savez_compressed(
                    p,
                    result_pickle=np.frombuffer(result_bytes, dtype=np.uint8),
                    ctx_pickle=np.frombuffer(ctx_bytes, dtype=np.uint8),
                    meta=np.array([meta], dtype=object),
                )
            else:
                sio.savemat(
                    p,
                    {
                        "result_pickle": np.frombuffer(result_bytes, dtype=np.uint8),
                        "ctx_pickle": np.frombuffer(ctx_bytes, dtype=np.uint8),
                        "meta": np.array([meta], dtype=object),
                    },
                    do_compression=True,
                )

            dpg.set_value("det_save_results_status", f"Saved: {p}")
        except Exception as e:
            dpg.set_value("det_save_results_status", f"Save failed: {e}")

    def load_detection_results(self, sender=None, app_data=None, user_data=None):
        try:
            path = str(dpg.get_value("det_load_results_file_input") or "").strip()
            if not path:
                self.browse_det_load_results_file()
                return

            p = Path(path)
            self._status_log_append(f"[DETECTION] Loading results: {p}")
            if not p.exists():
                raise FileNotFoundError(p)

            if p.suffix.lower() == ".npz":
                data = np.load(p, allow_pickle=True)
                result_bytes = bytes(np.array(data["result_pickle"], dtype=np.uint8).tobytes())
                ctx_bytes = bytes(np.array(data["ctx_pickle"], dtype=np.uint8).tobytes())
            elif p.suffix.lower() == ".mat":
                data = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
                result_bytes = bytes(np.array(data["result_pickle"], dtype=np.uint8).tobytes())
                ctx_bytes = bytes(np.array(data["ctx_pickle"], dtype=np.uint8).tobytes())
            else:
                raise ValueError("Unsupported file type. Use .npz or .mat.")

            self.last_result = pickle.loads(result_bytes)
            self._last_run_ctx = pickle.loads(ctx_bytes)
            ctx_params = self._last_run_ctx if isinstance(self._last_run_ctx, dict) else {}
            self.apply_loaded_ctx_to_gui(ctx_params)
            ctx_params = self._last_run_ctx
            if isinstance(ctx_params, dict):
                self.params.update(ctx_params)

            try:
                ctx_params = self._last_run_ctx if isinstance(self._last_run_ctx, dict) else {}
                if "N" in ctx_params:
                    self.params["N"] = int(ctx_params["N"])
                    if dpg.does_item_exist("det_input_matching_N"):
                        dpg.set_value("det_input_matching_N", int(ctx_params["N"]))
                if "overlap" in ctx_params:
                    self.params["overlap"] = float(ctx_params["overlap"])
                    if dpg.does_item_exist("det_input_matching_overlap"):
                        dpg.set_value("det_input_matching_overlap", float(ctx_params["overlap"]))
            except Exception:
                pass

            try:
                if dpg.does_item_exist("det_run_stats"):
                    dpg.set_value("det_run_stats", f"Loaded results from: {p.name}")
            except Exception:
                pass

            self._on_refresh_plots()

            dpg.set_value("det_load_results_status", f"Loaded: {p}")
            self._status_log_append(f"[DETECTION] Loaded results: {p.name}")
        except Exception as e:
            dpg.set_value("det_load_results_status", f"Load failed: {e}")
            self._status_log_append(f"[DETECTION] Loading results failed: {e}")
            
    def _is_basic_serializable(self, v):
        import numpy as np
        return v is None or isinstance(v, (bool, int, float, str, np.ndarray))
    
    def _make_pickleable(self, obj, max_depth=6):
        import pickle
    
        if max_depth <= 0:
            return None
    
        if self._is_basic_serializable(obj):
            return obj
    
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                key = k if isinstance(k, str) else str(k)
                vv = self._make_pickleable(v, max_depth=max_depth - 1)
                if vv is not None:
                    out[key] = vv
            return out

        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                vv = self._make_pickleable(v, max_depth=max_depth - 1)
                if vv is not None:
                    out.append(vv)
            return out

        try:
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return obj
        except Exception:
            return None
    
    def _extract_ctx_params(self, ctx):
        if ctx is None:
            return {}
    
        preferred_keys = [
            "N",
            "overlap",
            "structure_size",
            "min_steps",
            "min_events",
            *STORE.values("Detection"),  # settings of the page / Settings window (config/settings.py)
            "threshold",
            "min_cluster_size",
            "max_cluster_size",
            "dt",
            "radius",
            "sigma",
        ]
    
        if isinstance(ctx, dict):
            params = {}
            for k in preferred_keys:
                if k in ctx and self._is_basic_serializable(ctx[k]):
                    params[k] = ctx[k]
            if len(params) == 0:
                return self._make_pickleable(ctx)
            return params

        return self._make_pickleable(ctx)
    
    def apply_loaded_ctx_to_gui(self, ctx_params: dict):
        if not isinstance(ctx_params, dict):
            return

        px = bool(ctx_params.get("pixelwise_extension", False))
        ps = bool(ctx_params.get("pseudo_images", False))
        db = bool(ctx_params.get("db_clustering", False))
    
        if px:
            method = "pixelwise extension"
        elif ps:
            method = "pseudo-frame"
        elif db:
            method = "DBSCAN"
        else:
            method = "kd-tree"
    
        self.params["detection_method"] = method
        if dpg.does_item_exist(self._det_method_combo_tag):
            dpg.set_value(self._det_method_combo_tag, method)
        self._on_det_method_change(None, method, None)

        if "N" in ctx_params and dpg.does_item_exist("det_input_matching_N"):
            try:
                v = int(ctx_params["N"])
                self.params["N"] = v
                dpg.set_value("det_input_matching_N", v)
                self._n_user_set = True  
            except Exception:
                pass
    
        if "multirun" in ctx_params and dpg.does_item_exist("det_input_multirun"):
            try:
                v = bool(ctx_params["multirun"])
                self.params["multirun"] = v
                dpg.set_value("det_input_multirun", v)
                self._on_multirun_change(None, v, None)
            except Exception:
                pass
    
        if "multitimefactor" in ctx_params and dpg.does_item_exist("det_input_multitimefactor"):
            try:
                v = int(ctx_params["multitimefactor"])
                self.params["multitimefactor"] = v
                dpg.set_value("det_input_multitimefactor", v)
            except Exception:
                pass

        if "overlap" in ctx_params and dpg.does_item_exist("det_input_pixelwise_overlap"):
            try:
                v = float(ctx_params["overlap"])
                self.params["overlap"] = v
                dpg.set_value("det_input_pixelwise_overlap", v)
            except Exception:
                pass
    
        if "N_pixelwise" in ctx_params and dpg.does_item_exist("det_input_min_pixel"):
            try:
                v = int(ctx_params["N_pixelwise"])
                self.params["N_pixelwise"] = v
                dpg.set_value("det_input_min_pixel", v)
            except Exception:
                pass
            
        restored = STORE.update({k: v for k, v in ctx_params.items() if k in STORE.values("Detection")})
        if restored:
            self._status_log_append(f"Detection settings restored from the results file: "
                                    + ", ".join(f"{k}={STORE.get(k)}" for k in restored))
        for key, tag in (("min_steps", "det_input_min_steps"), ("min_events", "det_input_min_events")):
            if key in ctx_params:
                try:
                    v = int(ctx_params[key])
                    self.params[key] = v
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, v)
                except Exception:
                    pass

        if "structure_size" in ctx_params and dpg.does_item_exist("det_input_pixelwise_structure_size"):
            try:
                v = int(ctx_params["structure_size"])
                self.params["structure_size"] = v
                dpg.set_value("det_input_pixelwise_structure_size", v)
            except Exception:
                pass
    
        if "epsilon" in ctx_params:
            try:
                v = float(ctx_params["epsilon"])
                self.params["epsilon"] = v
                if dpg.does_item_exist("det_input_kdtree_radius"):
                    dpg.set_value("det_input_kdtree_radius", v)
                if dpg.does_item_exist("det_input_dbscan_radius"):
                    dpg.set_value("det_input_dbscan_radius", v)
            except Exception:
                pass
    
        if "minPts" in ctx_params:
            try:
                v = int(ctx_params["minPts"])
                self.params["minPts"] = v
                if dpg.does_item_exist("det_input_kdtree_min_events"):
                    dpg.set_value("det_input_kdtree_min_events", v)
                if dpg.does_item_exist("det_input_dbscan_min_events"):
                    dpg.set_value("det_input_dbscan_min_events", v)
            except Exception:
                pass
    
        if "threshold" in ctx_params and dpg.does_item_exist("det_input_dbscan_threshold"):
            try:
                v = float(ctx_params["threshold"])
                self.params["threshold"] = v
                dpg.set_value("det_input_dbscan_threshold", v)
            except Exception:
                pass
    
        if "area" in ctx_params and dpg.does_item_exist("det_input_pseudo_min_area"):
            try:
                v = int(ctx_params["area"])
                self.params["area"] = v
                dpg.set_value("det_input_pseudo_min_area", v)
            except Exception:
                pass
    
        if "filtersize" in ctx_params and dpg.does_item_exist("det_input_pseudo_filtersize"):
            try:
                v = int(ctx_params["filtersize"])
                self.params["filtersize"] = v
                dpg.set_value("det_input_pseudo_filtersize", v)
            except Exception:
                pass