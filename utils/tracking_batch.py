"""Batched versions of the per-track steps of utils/tracking.py ("Kalman afterwards" and its post-processing).

The tracking script mapped utils_v2.process_index_afterwards over every cluster with a thread pool. Only the
compiled Kalman filter releases the GIL: the numpy calls around it (sort, two least-squares fits, two means) ran
one track at a time under the GIL, so the threads mostly waited for each other. Here the same calls are grouped:

- the least-squares fits of all tracks with the same number of points go to numpy's lstsq gufunc in one call
  (the same LAPACK routine, track by track: identical results), the means likewise;
- the Kalman filter runs over chunks of tracks in the worker threads (one task per chunk, not per track);
- the post-processing of the results (NaN removal, smoothing, truncation) is one parallel compiled pass over
  all tracks instead of ~10 numpy / scipy calls per track. The smoothing reproduces
  scipy.ndimage.uniform_filter1d(size=3, mode='nearest') operation for operation (running sum, divided at
  output), so the results are bit-identical.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, prange
from numpy.linalg import _umath_linalg

try:  # thread count of the parallel kernels (utils/parallel.py)
    from utils import parallel as _parallel  # noqa: F401
except ImportError:  # run with only utils/ on the path
    import parallel as _parallel  # noqa: F401

try:  # numpy's error handler for a non-converging lstsq (raises LinAlgError)
    from numpy.linalg._linalg import _raise_linalgerror_lstsq
except ImportError:  # numpy < 2
    from numpy.linalg.linalg import _raise_linalgerror_lstsq


def _lstsq_gufunc(m, n):
    """numpy's lstsq gufunc for (m, n) matrices: numpy >= 2 has a single one, numpy < 2 one per shape case
    (lstsq_m when m <= n, lstsq_n otherwise), chosen as np.linalg.lstsq does."""
    if hasattr(_umath_linalg, 'lstsq'):
        return _umath_linalg.lstsq
    return _umath_linalg.lstsq_m if m <= n else _umath_linalg.lstsq_n


@njit(cache=True, nogil=True, parallel=True)
def _fit_points(t, off, st_max, dtt, inclusive):
    """Points of every (time-sorted) track used for its initial state, as process_index_afterwards selects them:
    st = floor(min(st_max, (max(t) - min(t)) / dtt)), then t < min(t) + st * dtt (<= when inclusive)."""
    n = off.size - 1
    m = np.zeros(n, np.int64)
    for i in prange(n):
        a = off[i]
        b = off[i + 1]
        if a == b:
            continue
        q = (t[b - 1] - t[a]) / dtt
        st = np.floor(q if q < st_max else st_max)  # min(st_max, q), as Python's min
        thr = t[a] + st * dtt
        c = 0
        for k in range(a, b):
            if (t[k] <= thr) if inclusive else (t[k] < thr):
                c += 1
        m[i] = c
    return m


def _lstsq_slopes(T, V):
    """Slope of the least-squares line through (T * 1e-6, V) for every row: np.linalg.lstsq(np.vstack([t * 1e-6,
    np.ones_like(t)]).T, v, rcond=None)[0][0], all rows in one gufunc call."""
    k, m = T.shape
    if m == 0:
        return np.zeros(k)  # (lstsq gives 0 for no points)
    A = np.empty((k, m, 2))
    A[:, :, 0] = T * 1e-6
    A[:, :, 1] = 1.0
    rcond = np.finfo(np.float64).eps * max(m, 2)
    with np.errstate(call=_raise_linalgerror_lstsq, invalid='call', over='ignore', divide='ignore', under='ignore'):
        sol = _lstsq_gufunc(m, 2)(A, V[:, :, None], rcond, signature='ddd->ddid')[0]
    return sol[:, 0, 0]


def kalman_afterwards(t_clust, x_clust, y_clust, P, C, Q, R, dtt, factor, kalman, num_worker,
                      st_max=10, inclusive=False, p_scale=10, start_id=1, progress=None):
    """Results of mapping utils_v2.process_index_afterwards over all clusters (the lightweight variant with
    st_max=4, inclusive=True, p_scale=50, start_id=0): a list of (i, result dict or None), and the clusters
    sorted by time in place, as that function leaves them.

    progress: optional callable(n_done) called after every chunk of tracks.
    """
    n = len(t_clust)
    results = [(i, None) for i in range(n)]
    lens = np.fromiter((len(a) for a in t_clust), np.int64, n)
    todo = np.flatnonzero(lens)
    if todo.size == 0:
        return results
    # time order of every cluster (np.argsort as before: the same order for equal times)
    for i in todo.tolist():
        o = np.argsort(t_clust[i])
        t_clust[i] = t_clust[i][o]
        x_clust[i] = x_clust[i][o]
        y_clust[i] = y_clust[i][o]
    sel = todo.tolist()
    off = np.zeros(todo.size + 1, np.int64)
    np.cumsum(lens[todo], out=off[1:])
    T = np.concatenate([np.asarray(t_clust[i], dtype=np.float64) for i in sel])
    X = np.concatenate([np.asarray(x_clust[i]) for i in sel])
    Y = np.concatenate([np.asarray(y_clust[i]) for i in sel])
    m = _fit_points(T, off, float(st_max), float(dtt), bool(inclusive))
    # initial states: slope of the least-squares line and mean of the selected points, grouped by point count
    xe = np.zeros((todo.size, 3))
    ye = np.zeros((todo.size, 3))
    order = np.argsort(m, kind="stable")
    ms = m[order]
    cuts = np.flatnonzero(np.diff(ms)) + 1
    for g in np.split(np.arange(todo.size), cuts):
        if g.size == 0:
            continue
        rows = order[g]
        mm = int(ms[g[0]])
        idx = off[rows][:, None] + np.arange(mm)
        Tg = T[idx]
        Xg = X[idx]
        Yg = Y[idx]
        xe[rows, 1] = _lstsq_slopes(Tg, Xg.astype(np.float64))
        ye[rows, 1] = _lstsq_slopes(Tg, Yg.astype(np.float64))
        with np.errstate(invalid='ignore', divide='ignore'):
            if mm == 0:  # np.mean of no points: NaN (as before, without the warning)
                xe[rows, 0] = np.nan
                ye[rows, 0] = np.nan
            else:
                xe[rows, 0] = Xg.mean(axis=1)
                ye[rows, 0] = Yg.mean(axis=1)
    Ps = P * p_scale  # (read only by the filter with rev = 0)

    def run(lo, hi):
        out = []
        for j in range(lo, hi):
            i = sel[j]
            P1, xe1, Pz, ye1, t, x_p, y_p, xv_p, yv_p, xa_p, ya_p = kalman(
                t_clust[i], x_clust[i], y_clust[i], start_id, C, Q, R, Ps, xe[j], Ps, ye[j], 0, dtt, factor)
            out.append((i, {"P": P1, "xe": xe1, "Pz": Pz, "ye": ye1, "t": t, "x": x_p, "y": y_p,
                            "u": xv_p, "v": yv_p, "ax": xa_p, "ay": ya_p}))
        return out

    chunk = max(64, todo.size // (8 * max(1, num_worker)))
    bounds = [(lo, min(lo + chunk, todo.size)) for lo in range(0, todo.size, chunk)]
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, num_worker)) as ex:
        for part in ex.map(lambda b: run(*b), bounds):
            for i, r in part:
                results[i] = (i, r)
            done += len(part)
            if progress is not None:
                progress(done)
    return results


def _pack(arrays):
    lens = np.fromiter((len(a) for a in arrays), np.int64, len(arrays))
    off = np.zeros(len(arrays) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    data = np.concatenate(arrays) if len(arrays) else np.zeros(0)
    return data.astype(np.float64, copy=False), off


@njit(cache=True, nogil=True)
def _smooth3(a, out):
    """scipy.ndimage.uniform_filter1d(a, size=3, mode='nearest'): running sum over the edge-extended line,
    divided at output (the same operations as scipy, so the same bits)."""
    n = a.size
    if n == 0:
        return
    ext = np.empty(n + 2)
    ext[0] = a[0]
    ext[1:n + 1] = a
    ext[n + 1] = a[n - 1]
    tmp = 0.0
    for q in range(3):
        tmp += ext[q]
    out[0] = tmp / 3.0
    for q in range(1, n):
        tmp += ext[q + 2] - ext[q - 1]
        out[q] = tmp / 3.0


@njit(cache=True, nogil=True)
def _finite(src, a, b, scale, out):
    """Non-NaN values of src[a:b] (times scale) into out; returns their number."""
    c = 0
    for k in range(a, b):
        v = src[k]
        if v == v:
            out[c] = v * scale if scale != 1.0 else v
            c += 1
    return c


@njit(cache=True, nogil=True, parallel=True)
def _post(t, x, xv, xa, y, yv, ya, off_t, off_x, off_xv, off_xa, off_y, off_yv, off_ya, v_scale):
    """Post-processing of the Kalman results of every track (see postprocess_afterwards). Returns the kept tracks
    and, for each of the 7 quantities, its output values and offsets (over the kept tracks)."""
    n = off_x.size - 1
    keep = np.zeros(n, np.uint8)
    L = np.zeros((7, n), np.int64)
    for i in prange(n):
        c = np.zeros(7, np.int64)
        srcs = (t, x, xv, xa, y, yv, ya)
        offs = (off_t, off_x, off_xv, off_xa, off_y, off_yv, off_ya)
        for q in range(7):
            src = srcs[q]
            o = offs[q]
            cnt = 0
            for k in range(o[i], o[i + 1]):
                if src[k] == src[k]:
                    cnt += 1
            c[q] = cnt
        if c[1] == 0:
            continue  # x all NaN (or empty): the track is dropped
        keep[i] = 1
        lx = c[1]
        L[0, i] = c[0]
        L[1, i] = lx
        L[4, i] = c[4]
        trunc = lx < c[2]  # velocities / accelerations cut to the length of x when x is shorter than xv
        for q in range(2, 7):
            if q != 4:
                L[q, i] = min(c[q], lx) if trunc else c[q]
    kept = np.flatnonzero(keep)
    nk = kept.size
    oo = np.zeros((7, nk + 1), np.int64)
    for q in range(7):
        for j in range(nk):
            oo[q, j + 1] = oo[q, j] + L[q, kept[j]]
    outs = (np.empty(oo[0, nk]), np.empty(oo[1, nk]), np.empty(oo[2, nk]), np.empty(oo[3, nk]),
            np.empty(oo[4, nk]), np.empty(oo[5, nk]), np.empty(oo[6, nk]))
    for j in prange(nk):
        i = kept[j]
        srcs = (t, x, xv, xa, y, yv, ya)
        offs = (off_t, off_x, off_xv, off_xa, off_y, off_yv, off_ya)
        for q in range(7):
            o = offs[q]
            a = o[i]
            b = o[i + 1]
            buf = np.empty(b - a)
            cnt = _finite(srcs[q], a, b, v_scale if (q == 2 or q == 5) else 1.0, buf)
            buf = buf[:cnt]
            if q == 1 or q == 2 or q == 4 or q == 5:  # x, y and their velocities: smoothed twice
                tmp = np.empty(cnt)
                _smooth3(buf, tmp)
                _smooth3(tmp, buf)
            out = outs[q]
            p = oo[q, j]
            for k in range(L[q, i]):
                out[p + k] = buf[k]
    return kept, outs, oo


def postprocess_afterwards(t_plot, x_plot, x_plotv, x_plota, y_plot, y_plotv, y_plota, v_scale):
    """The post-processing of the Kalman results in utils/tracking.py, for all tracks in one parallel pass:

        keep the tracks whose x is not all NaN; velocities times v_scale; NaN removed from every quantity; x, xv,
        y, yv smoothed twice (uniform_filter1d, size 3, 'nearest'); xv, xa, yv, ya cut to the length of x when x
        is shorter than xv. (The script also empties t where x is empty: never the case for a kept track.)

    Returns (kept track indices, t, x, xv, xa, y, yv, ya as lists of arrays), or None if some array is not
    float64 (the script then runs its per-track code).
    """
    groups = (t_plot, x_plot, x_plotv, x_plota, y_plot, y_plotv, y_plota)
    for g in groups:
        for a in g:
            if not (isinstance(a, np.ndarray) and a.dtype == np.float64 and a.ndim == 1):
                return None
    packed = [_pack(g) for g in groups]
    kept, outs, oo = _post(*[p[0] for p in packed], *[p[1] for p in packed], float(v_scale))
    lists = []
    for q in range(7):
        data = outs[q]
        o = oo[q]
        lists.append([data[o[j]:o[j + 1]] for j in range(kept.size)])
    return (kept, *lists)
