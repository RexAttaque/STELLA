"""Algorithm settings that used to be hard-coded.

Every setting has a key (the name the detection / tracking scripts read from their context), a default that
reproduces the former hard-coded value exactly, and a page. The main ones (main=True) have a field on their page;
the others are in the Settings window (top menu bar: Settings > Algorithm settings...).

STORE holds the current values. It is saved to the user's settings file when the application closes and loaded
when it starts (load_user_settings / save_user_settings); detection results files also carry the detection values.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

Number = Union[int, float]


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    default: Number
    page: str  # "Detection", "Tracking", "Validation", "Loading"
    group: str  # sub-heading in the Settings window (and the methods it applies to)
    help: str
    main: bool = False  # True: a field on its page; False: in the Settings window
    minimum: Optional[Number] = None
    maximum: Optional[Number] = None
    fmt: str = "%.3g"  # display format of float fields

    @property
    def is_int(self) -> bool:
        return isinstance(self.default, int) and not isinstance(self.default, bool)


SETTINGS: Sequence[Setting] = (
    # ---------------------------------------------------------------- detection, on the page
    Setting("new_cluster_events", "events to start a cluster", 6, "Detection", "Cluster rules (pixelwise extension)",
            "A region of a time step that matches no cluster starts a new one only if it holds at least this many "
            "events of that step.", main=True, minimum=1),
    Setting("retire_idle_steps", "idle steps before a young cluster is dropped", 2, "Detection", "Cluster rules",
            "A cluster followed over fewer than 'min. steps' steps is dropped (its slot reused) when it was not "
            "extended for more than this many steps.", main=True, minimum=0),
    Setting("split_merge_distance", "split: merge centres closer than (px)", 4.0, "Detection",
            "Cluster rules (pixelwise extension)",
            "When a region touches several clusters, its events go to the nearest cluster centre. Centres closer "
            "than this to another centre are left out (those clusters take no events).", main=True, minimum=0.0,
            fmt="%.2f"),
    Setting("split_min_events", "split: min. events per share", 4, "Detection", "Cluster rules (pixelwise extension)",
            "When a region is split between clusters, a cluster takes its share only if it holds at least this many "
            "events.", main=True, minimum=1),
    Setting("repair_max_distance", "track repair: max. distance (px)", 5.0, "Detection", "Track repair",
            "After detection, a track that starts within this distance of where another one ends is joined to it.",
            main=True, minimum=0.0, fmt="%.2f"),
    Setting("repair_max_gap", "track repair: max. gap (time steps)", 1.0, "Detection", "Track repair",
            "... if it starts at most this many time steps after the other one ends.", main=True, minimum=0.0,
            fmt="%.2f"),
    # ---------------------------------------------------------------- detection, Settings window
    Setting("buffer", "buffer (steps)", 5, "Detection", "Engine",
            "Time steps a cluster keeps in its history for matching.", minimum=1),
    Setting("steps_rev", "steps_rev", 10000, "Detection", "Engine",
            "Steps after which a cluster's history is trimmed.", minimum=1),
    Setting("search_factor", "sub-grid search levels", 5, "Detection", "Engine (kd-tree, DBSCAN)",
            "Number of nested sub-grids searched for new clusters.", minimum=1),
    Setting("max_tracks", "max. clusters", 100000, "Detection", "Engine",
            "Cluster slots allocated at the start (they grow when needed).", minimum=1),
    Setting("max_length", "max. track length", 100000, "Detection", "Engine", "Initial track length.", minimum=1),
    Setting("kd_radius_margin", "kd-tree: radius margin (px)", 4.0, "Detection", "kd-tree",
            "Events within (radius + margin) of a cluster's centre are matched to it.", minimum=0.0, fmt="%.2f"),
    Setting("subgrid_factor", "sub-grid: event threshold factor", 5.0, "Detection", "Engine (kd-tree, DBSCAN)",
            "A sub-grid is searched when it holds more than (free slots x min. points x factor) events.",
            minimum=0.0, fmt="%.2f"),
    Setting("pseudo_min_track_points", "pseudo-frame: min. track points", 3, "Detection", "Pseudo-frame",
            "Tracks with fewer points are dropped.", minimum=0),
    Setting("pseudo_event_factor", "pseudo-frame: min. events factor", 2.0, "Detection", "Pseudo-frame",
            "A pseudo-frame track is kept with at least (factor x min. events) events.", minimum=0.0, fmt="%.2f"),
    Setting("pseudo_repair_max_distance", "pseudo-frame repair: max. distance (px)", 4.0, "Detection",
            "Pseudo-frame", "Track repair distance for pseudo-frame tracks.", minimum=0.0, fmt="%.2f"),
    Setting("pseudo_repair_max_gap", "pseudo-frame repair: max. gap (time steps)", 1.5, "Detection", "Pseudo-frame",
            "Track repair gap for pseudo-frame tracks.", minimum=0.0, fmt="%.2f"),
    # ---------------------------------------------------------------- tracking, on the page
    Setting("kalman_init_steps", "initial state: first time steps", 10, "Tracking", "Kalman filter",
            "The initial position and velocity of a track are fitted on its points within this many time steps of "
            "its start.", main=True, minimum=1),
    Setting("kalman_p_scale", "initial covariance factor (x P)", 10.0, "Tracking", "Kalman filter",
            "The filter starts with the covariance P times this factor.", main=True, minimum=0.0, fmt="%.3g"),
    Setting("spline_timeout", "fit timeout per track (s)", 5.0, "Tracking", "B-spline",
            "A spline fit taking longer stops the search; the best smoothing found so far is used.", main=True,
            minimum=0.1, fmt="%.1f"),
    Setting("spline_s_max", "largest smoothing factor", 9e7, "Tracking", "B-spline",
            "The search tries smoothing factors from this one down (fixed schedule 9e7 ... 0.01).", main=True,
            minimum=0.0, fmt="%.3g"),
    Setting("spline_s_min", "smallest smoothing factor", 0.01, "Tracking", "B-spline",
            "... down to this one.", main=True, minimum=0.0, fmt="%.3g"),
    # ---------------------------------------------------------------- tracking, Settings window
    Setting("smooth_window", "smoothing window (points)", 3, "Tracking", "Smoothing (Kalman, hybrids)",
            "Moving average applied to positions and velocities after the Kalman filter.", minimum=1),
    Setting("smooth_passes", "smoothing passes", 2, "Tracking", "Smoothing (Kalman, hybrids)",
            "Times the moving average is applied (2 passes of 3 points = weights 1 2 3 2 1 / 9).", minimum=0),
    Setting("min_track_points", "min. track points (hybrids)", 3, "Tracking", "Hybrid methods",
            "Kalman tracks with fewer points are not spline fitted.", minimum=0),
    Setting("pseudo_kalman_init_steps", "pseudo-frame: initial state steps", 4, "Tracking", "Kalman filter (pseudo-frame)",
            "As 'initial state: first time steps', for pseudo-frame tracks (end point included).", minimum=1),
    Setting("pseudo_kalman_p_scale", "pseudo-frame: covariance factor", 50.0, "Tracking", "Kalman filter (pseudo-frame)",
            "As 'initial covariance factor', for pseudo-frame tracks.", minimum=0.0, fmt="%.3g"),
    Setting("spline_start_rms", "starting RMS", 100.0, "Tracking", "B-spline search",
            "RMS the search starts from (a fit must improve on it).", minimum=0.0, fmt="%.3g"),
    Setting("spline_bounds_margin", "bounds margin (px)", 100.0, "Tracking", "B-spline search",
            "A fit leaving the track's extent plus this margin is rejected.", minimum=0.0, fmt="%.3g"),
    Setting("spline_stop_s", "stop below smoothing factor", 200.0, "Tracking", "B-spline search",
            "Below this smoothing factor, a rejected or worse fit stops the search.", minimum=0.0, fmt="%.3g"),
    Setting("spline_accept_ratio", "accept ratio", 1.02, "Tracking", "B-spline search",
            "A fit is taken when its RMS is below (ratio x best RMS so far).", minimum=0.0, fmt="%.3f"),
    Setting("spline_tolerance_ratio", "tolerance ratio", 1.15, "Tracking", "B-spline search",
            "A fit up to (ratio x best RMS) does not stop the search.", minimum=0.0, fmt="%.3f"),
    Setting("spline_min_points", "min. track points", 3, "Tracking", "B-spline search",
            "Tracks with fewer points are not fitted.", minimum=1),
    # ---------------------------------------------------------------- validation, Settings window
    Setting("track_quality_quantile", "quality normalisation quantile", 0.95, "Validation", "Track quality",
            "Each quality measure is divided by this quantile over all tracks.", minimum=0.01, maximum=1.0,
            fmt="%.3f"),
    Setting("nb_sigma", "neighbour weight width (px, 0 = radius)", 0.0, "Validation", "Neighbourhood filter",
            "Width of the Gaussian weight of the neighbours; 0 uses the neighbourhood radius.", minimum=0.0,
            fmt="%.2f"),
    # ---------------------------------------------------------------- loading
    Setting("raw_max_events", "max. events read from a .raw file", 700_000_000, "Loading", "Loading",
            "Events after this count are not read.", minimum=1),
)

BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}
PAGES: Sequence[str] = ("Detection", "Tracking", "Validation", "Loading")


def coerce(setting: Setting, value) -> Number:
    v = int(round(float(value))) if setting.is_int else float(value)
    if setting.minimum is not None and v < setting.minimum:
        v = type(v)(setting.minimum)
    if setting.maximum is not None and v > setting.maximum:
        v = type(v)(setting.maximum)
    return v


class SettingsStore:
    def __init__(self):
        self._values: Dict[str, Number] = {s.key: s.default for s in SETTINGS}
        self._listeners: List[Callable[[str, Number], None]] = []

    def get(self, key: str) -> Number:
        return self._values[key]

    def set(self, key: str, value) -> Number:
        v = coerce(BY_KEY[key], value)
        if v != self._values[key] or type(v) is not type(self._values[key]):
            self._values[key] = v
            for cb in list(self._listeners):
                cb(key, v)
        return v

    def update(self, values: Dict[str, object]) -> List[str]:
        """Sets the known keys of values; returns the keys set."""
        done = []
        for k, v in values.items():
            if k in BY_KEY:
                try:
                    self.set(k, v)
                    done.append(k)
                except (TypeError, ValueError):
                    pass
        return done

    def values(self, page: Optional[str] = None) -> Dict[str, Number]:
        return {s.key: self._values[s.key] for s in SETTINGS if page is None or s.page == page}

    def changed(self) -> Dict[str, Number]:
        """The values that differ from their default."""
        return {s.key: self._values[s.key] for s in SETTINGS if self._values[s.key] != s.default}

    def reset(self, page: Optional[str] = None) -> None:
        for s in SETTINGS:
            if page is None or s.page == page:
                self.set(s.key, s.default)

    def add_listener(self, cb: Callable[[str, Number], None]) -> None:
        self._listeners.append(cb)


STORE = SettingsStore()


def user_settings_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "STELLA" / "settings.json"


def load_user_settings(path: Optional[Path] = None) -> List[str]:
    """Loads the saved values (missing file: defaults); returns the keys that differ from their default."""
    path = Path(path) if path else user_settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    STORE.update(data.get("settings", {}) if isinstance(data, dict) else {})
    return list(STORE.changed())


def save_user_settings(path: Optional[Path] = None) -> None:
    path = Path(path) if path else user_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stella_settings": 1, "settings": STORE.changed()}, indent=1), encoding="utf-8")
