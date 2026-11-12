"""Per-track event arrays stored back to back in one flat array.

The detection used to return x_clust, y_clust, t_clust and ind_clust as Python lists of millions of small numpy
arrays (one per track). Every pass over all tracks then paid Python's per-object cost, the array headers alone took
GBs, and pickling them for saving was slow.

`PackedTracks` holds the same data as one flat array plus the [start, stop) range of every track. It behaves like
the list it replaces: ``len(x)``, ``x[i]`` (a view of track i, no copy), ``for a in x`` and ``x[i] = sorted_a``
(same length, written in place) all work, so code written for the lists keeps working. Passes over all tracks use
the methods below (lengths, per-track min / max / counts) instead of Python loops.

Ranges may leave holes (a track dropped by a filter keeps its events in `data` until the next compaction) and
never overlap. Pickling stores the compact form.
"""
from collections.abc import Sequence

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------------------------------- kernels

@njit(cache=True, nogil=True, parallel=True)
def _gather(data, start, stop, off, out):
    """out[off[i]:off[i + 1]] = data[start[i]:stop[i]] for every track i."""
    for i in prange(start.size):
        p = off[i]
        for j in range(start[i], stop[i]):
            out[p] = data[j]
            p += 1


@njit(cache=True, nogil=True, parallel=True)
def _gather_chains(data, start, stop, head, nxt, off, out):
    """Track i of the output is the concatenation of the input tracks head[i], nxt[head[i]], ... (-1 ends)."""
    for i in prange(head.size):
        p = off[i]
        s = head[i]
        while s != -1:
            for j in range(start[s], stop[s]):
                out[p] = data[j]
                p += 1
            s = nxt[s]


@njit(cache=True, nogil=True, parallel=True)
def _count_finite(data, start, stop):
    """Number of non-NaN values of every track."""
    n = np.zeros(start.size, np.int64)
    for i in prange(start.size):
        c = 0
        for j in range(start[i], stop[i]):
            v = data[j]
            if v == v:
                c += 1
        n[i] = c
    return n


@njit(cache=True, nogil=True, parallel=True)
def _finite_min_max(data, start, stop):
    """(min, max) of the finite values of every track as float64; (inf, -inf) when it has none."""
    lo = np.full(start.size, np.inf)
    hi = np.full(start.size, -np.inf)
    for i in prange(start.size):
        a = np.inf
        b = -np.inf
        for j in range(start[i], stop[i]):
            v = np.float64(data[j])
            if np.isfinite(v):
                if v < a:
                    a = v
                if v > b:
                    b = v
        lo[i] = a
        hi[i] = b
    return lo, hi


@njit(cache=True, nogil=True, parallel=True)
def _distinct_per_track(data, start, stop):
    """np.unique(track.astype(int)).size of every track."""
    n = np.zeros(start.size, np.int64)
    for i in prange(start.size):
        m = stop[i] - start[i]
        if m == 0:
            continue
        b = np.empty(m, np.int64)
        for j in range(m):
            b[j] = np.int64(data[start[i] + j])
        b.sort()
        c = 1
        for j in range(1, m):
            if b[j] != b[j - 1]:
                c += 1
        n[i] = c
    return n


@njit(cache=True, nogil=True)
def _distinct_total(data, start, stop):
    """np.unique(np.concatenate(tracks).astype(int)).size for non-negative values."""
    top = -1
    for i in range(start.size):
        for j in range(start[i], stop[i]):
            top = max(top, np.int64(data[j]))
    if top < 0:
        return 0
    seen = np.zeros(top + 1, np.bool_)
    c = 0
    for i in range(start.size):
        for j in range(start[i], stop[i]):
            v = np.int64(data[j])
            if not seen[v]:
                seen[v] = True
                c += 1
    return c


@njit(cache=True, nogil=True)
def _clear_mask(mask, data, start, stop):
    """mask[np.concatenate(tracks)] = False."""
    for i in range(start.size):
        for j in range(start[i], stop[i]):
            mask[data[j]] = False


# ---------------------------------------------------------------------------------------------------- class

def _rebuild(data, off):
    return PackedTracks(data, off[:-1], off[1:])


class PackedTracks(Sequence):
    """Tracks as one flat array: track i is ``data[start[i]:stop[i]]``."""

    __slots__ = ("data", "start", "stop")

    def __init__(self, data, start, stop):
        self.data = data
        self.start = np.asarray(start, dtype=np.int64)
        self.stop = np.asarray(stop, dtype=np.int64)

    # ---- construction
    @classmethod
    def from_offsets(cls, data, off):
        off = np.asarray(off, dtype=np.int64)
        return cls(data, off[:-1], off[1:])

    @classmethod
    def from_list(cls, arrays, dtype=None):
        """Packs a list of 1-D arrays. The dtype is that of the non-empty ones (empty placeholders such as
        np.array([]) do not promote it)."""
        arrays = [np.asarray(a).ravel() for a in arrays]
        lens = np.fromiter((a.size for a in arrays), np.int64, len(arrays))
        if dtype is None:
            filled = [a for a in arrays if a.size]
            dtype = np.result_type(*filled) if filled else np.float64
        off = np.zeros(lens.size + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        filled = [a for a in arrays if a.size]
        data = np.concatenate(filled).astype(dtype, copy=False) if filled else np.zeros(0, dtype)
        return cls.from_offsets(data, off)

    @staticmethod
    def concat(parts):
        """The tracks of all parts, in order (like adding lists)."""
        parts = [p for p in parts if len(p)]
        if not parts:
            return PackedTracks(np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64))
        if len(parts) == 1:
            return parts[0]
        shift = np.cumsum([0] + [p.data.size for p in parts[:-1]])
        data = np.concatenate([p.data for p in parts])
        return PackedTracks(data, np.concatenate([p.start + s for p, s in zip(parts, shift)]),
                            np.concatenate([p.stop + s for p, s in zip(parts, shift)]))

    # ---- list behaviour
    def __len__(self):
        return self.start.size

    def __getitem__(self, i):
        if i.__class__ is int or isinstance(i, np.integer):  # (numpy handles negative / out-of-range indices)
            return self.data[self.start[i]:self.stop[i]]
        if isinstance(i, slice):
            return PackedTracks(self.data, self.start[i], self.stop[i])
        idx = np.asarray(i)
        if idx.dtype != np.bool_:
            idx = idx.astype(np.int64)
        return PackedTracks(self.data, self.start[idx], self.stop[idx])

    def __setitem__(self, i, value):
        """Replaces the values of track i in place (same length only, e.g. the track re-sorted by time)."""
        view = self[int(i)]
        value = np.asarray(value).ravel()
        if value.size != view.size:
            raise ValueError("PackedTracks: a track can only be replaced by values of the same length")
        view[...] = value

    def __iter__(self):
        d = self.data
        for a, b in zip(self.start.tolist(), self.stop.tolist()):
            yield d[a:b]

    def __reduce__(self):
        c = self.compact()
        return _rebuild, (c.data, np.concatenate((np.zeros(1, np.int64), c.stop)))

    def __repr__(self):
        return f"PackedTracks({len(self)} tracks, {int(self.lengths().sum())} values, dtype {self.data.dtype})"

    def to_list(self):
        return list(self)

    @property
    def dtype(self):
        return self.data.dtype

    # ---- whole-collection operations
    def lengths(self):
        return self.stop - self.start

    def nonempty(self):
        """Only the tracks with at least one value (``[a for a in x if a.size > 0]``)."""
        return self[self.stop > self.start]

    def keep(self, mask):
        """Same tracks, with those where mask is False emptied (``x[i] = np.array([])``)."""
        mask = np.asarray(mask, dtype=bool)
        return PackedTracks(self.data, self.start, np.where(mask, self.stop, self.start))

    def is_compact(self):
        n = self.start.size
        return (n == 0 or (self.start[0] == 0 and bool(np.all(self.start[1:] == self.stop[:-1])))) \
            and (self.stop[-1] if n else 0) == self.data.size

    def compact(self):
        """Same tracks, stored without holes."""
        if self.is_compact():
            return self
        off = np.zeros(self.start.size + 1, np.int64)
        np.cumsum(self.lengths(), out=off[1:])
        out = np.empty(off[-1], self.data.dtype)
        _gather(self.data, self.start, self.stop, off, out)
        return PackedTracks.from_offsets(out, off)

    def flat(self):
        """All values, track after track (``np.concatenate(list(x))``)."""
        return self.compact().data

    def count_finite(self):
        """Number of non-NaN values per track."""
        if self.data.dtype.kind in "iub":
            return self.lengths()
        return _count_finite(self.data, self.start, self.stop)

    def finite_min_max(self):
        """Per-track (min, max) of the finite values, (inf, -inf) for tracks without any."""
        return _finite_min_max(self.data, self.start, self.stop)

    def distinct_per_track(self):
        return _distinct_per_track(self.data, self.start, self.stop)

    def distinct_total(self):
        return int(_distinct_total(self.data, self.start, self.stop))

    def clear_in_mask(self, mask):
        """mask[all indices held by the tracks] = False (the tracks hold event indices)."""
        _clear_mask(mask, self.data, self.start, self.stop)


def as_packed(tracks):
    """`tracks` as PackedTracks (a list of arrays is packed once)."""
    if isinstance(tracks, PackedTracks):
        return tracks
    return PackedTracks.from_list(tracks)
