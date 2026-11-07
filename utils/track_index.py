"""Per-track time ranges, cached per result, so the pseudo-frame overlays only visit the
tracks that overlap the displayed time window instead of all (tens of thousands of) tracks."""
from __future__ import annotations

import numpy as np

_cache: dict[int, tuple[object, int, np.ndarray, np.ndarray]] = {}


def _ranges(tracks) -> tuple[np.ndarray, np.ndarray]:
    key = id(tracks)
    hit = _cache.get(key)
    if hit is not None and hit[0] is tracks and hit[1] == len(tracks):
        return hit[2], hit[3]
    n = len(tracks)
    tmin = np.full(n, np.inf)
    tmax = np.full(n, -np.inf)
    for i, t in enumerate(tracks):
        if t is None:
            continue
        a = np.asarray(t, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            tmin[i] = a.min()
            tmax[i] = a.max()
    if len(_cache) > 8:
        _cache.clear()
    _cache[key] = (tracks, n, tmin, tmax)
    return tmin, tmax


def tracks_in_window(tracks, t_lo: float, t_hi: float, n: int | None = None) -> np.ndarray:
    """Indices (< n) of the tracks with at least one time in [t_lo, t_hi]."""
    tmin, tmax = _ranges(tracks)
    idx = np.flatnonzero((tmax >= t_lo) & (tmin <= t_hi))
    if n is not None:
        idx = idx[idx < n]
    return idx
