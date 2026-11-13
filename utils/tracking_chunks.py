"""Tracking in batches of clusters, in the order of their start time, so that a run can start at a time step of
the recording and be stopped between batches.

Every cluster is tracked on its own (Kalman filter or spline fit of its points), so tracking a set of clusters in
batches gives the tracks of a single run. The merged tracks are put back in the order of the clusters (the order
of a single run). After k batches, all the clusters starting before the first cluster of batch k + 1 are tracked:
that is the part of the recording the results cover.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

MIN_BATCH = 500   # clusters per batch at least (fewer batches: less overhead per engine run)
N_BATCHES = 20    # batches of a long run (how often Stop is checked)


def cluster_start_times(t_clust) -> np.ndarray:
    """First finite time of every cluster (us), +inf for a cluster without points."""
    if hasattr(t_clust, "finite_min_max"):  # PackedTracks
        lo = np.asarray(t_clust.finite_min_max()[0], dtype=float)
        return np.where(np.isfinite(lo), lo, np.inf)
    out = np.full(len(t_clust), np.inf)
    for i, a in enumerate(t_clust):
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            out[i] = a.min()
    return out


def plan(starts: np.ndarray, t_lo: float, t_hi: float, limit: Optional[int] = None,
         min_batch: int = MIN_BATCH, n_batches: int = N_BATCHES):
    """Batches (cluster index arrays, in start-time order) of the clusters starting in [t_lo, t_hi), and the
    start time of the first cluster of each batch. `limit`: only the clusters 0 .. limit - 1."""
    idx = np.flatnonzero((starts >= t_lo) & (starts < t_hi))
    if limit is not None:
        idx = idx[idx < limit]
    idx = idx[np.argsort(starts[idx], kind="stable")]
    size = max(min_batch, int(math.ceil(len(idx) / n_batches))) if len(idx) else 1
    batches = [idx[i:i + size] for i in range(0, len(idx), size)]
    # a batch boundary between clusters starting at the same time would split them: move it after them
    merged = []
    for b in batches:
        if merged and starts[b[0]] == starts[merged[-1][-1]]:
            merged[-1] = np.concatenate([merged[-1], b])
        else:
            merged.append(b)
    return merged, [float(starts[b[0]]) for b in merged]


def slice_ctx(ctx: dict, idx: np.ndarray) -> dict:
    """Engine context with only the clusters `idx` (in that order)."""
    ctx = dict(ctx)
    n = None
    for key in ("t_clust", "x_clust", "y_clust", "t_m", "x_m", "y_m"):
        if key in ctx and ctx[key] is not None:
            src = ctx[key]
            ctx[key] = [src[int(i)] for i in idx]
            n = len(idx)
    if n is not None:
        ctx["max_tracks"] = max(1, n)
    return ctx


def merge(outs: list, batches: list, make: Callable[..., Any]):
    """One result from the batch results: tracks in the order of their clusters, track_to_cluster of the whole
    detection result. `make(**fields)` builds the result object."""
    fields = ("x_plot", "y_plot", "x_plotv", "y_plotv", "t_plot")
    lists = {f: [] for f in fields}
    ttc = []
    for out, idx in zip(outs, batches):
        m = getattr(out, "track_to_cluster", None)
        n = len(out.t_plot)
        local = np.arange(n) if m is None else np.asarray(m, dtype=int)
        ttc.append(np.where(local >= 0, np.asarray(idx)[np.clip(local, 0, max(len(idx) - 1, 0))], -1)
                   if len(idx) else np.full(n, -1, dtype=int))
        for f in fields:
            lists[f].extend(list(getattr(out, f)))
    ttc = np.concatenate(ttc).astype(int) if ttc else np.zeros(0, dtype=int)
    order = np.argsort(ttc, kind="stable")  # the order of a single run (tracks follow their clusters)
    return make(**{f: [lists[f][i] for i in order] for f in fields}, track_to_cluster=ttc[order])
