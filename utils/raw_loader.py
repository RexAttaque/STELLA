"""Fast reader for Prophesee EVT 3.0 .raw files.

Numba port of the decoder used by pyEBIV (EBIV/pyebiv/src/ebi_events.cpp,
_loadRawEventData + the monotonic-time consistency check), so the events are
identical to pyEBIV.loadRaw() + time()/x()/y()/p(), but decoded straight into
numpy arrays: no Python lists, about a second for 500 M events.

The decoding is split into chunks of words decoded in parallel. The decoder state
(time base and loop count, last time, y, x base, polarity) at the start of each
chunk is found first: every chunk summarises what it does to the state (in
parallel), the summaries are chained (sequential, one step per chunk), then the
kept events are counted per chunk and written at the resulting offsets.

Same conventions as pyEBIV:
- times are relative to the first CD event; events at exactly that time are
  dropped (pyEBIV keeps only t - t_first > 0);
- x is taken modulo the sensor width, y modulo the sensor height;
- events whose time exceeds the last event time get the previous event's time.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

try:  # thread count of the parallel kernels (utils/parallel.py)
    from utils import parallel as _parallel  # noqa: F401
except ImportError:  # run with only utils/ on the path
    import parallel as _parallel  # noqa: F401


@dataclass
class RawEvents:
    x: np.ndarray      # uint16
    y: np.ndarray      # uint16
    t: np.ndarray      # int64 (or the t_dtype asked), microseconds relative to the first event
    p: np.ndarray      # int8, 0 / 1
    width: int
    height: int
    time_stamp: int    # absolute time of the first event (us)
    geometry_found: bool = True  # False: the header gave no sensor size (width, height = 1280 x 720)


def _parse_geometry(key: str, val: str):
    """(width, height) from a 'geometry WxH' or 'format EVT3;height=H;width=W' header field, else None."""
    if key == "geometry":
        m = re.fullmatch(r"(\d+)x(\d+)", val)
        if m:
            return int(m.group(1)), int(m.group(2))
    elif key == "format":
        opts = dict(kv.split("=", 1) for kv in val.split(";")[1:] if "=" in kv)
        if opts.get("width", "").isdigit() and opts.get("height", "").isdigit():
            return int(opts["width"]), int(opts["height"])
    return None


def _read_header(f) -> tuple[int, int, bool, int, bool]:
    """Parses the '%' header lines. Returns (width, height, is_evt3, data offset, geometry found in the file);
    without geometry the size is 1280 x 720."""
    width = height = 0
    integrator = plugin = ""
    is_evt3 = False
    have_geometry = False
    while True:
        pos = f.tell()
        first = f.read(1)
        if first != b"%":
            f.seek(pos)
            break
        line = (first + f.readline()).decode("latin-1").rstrip("\r\n")
        if line == "% end":
            break
        vals = line.split(" ")
        if len(vals) == 3:
            key, val = vals[1], vals[2]
            if key == "integrator_name":
                integrator = val
            elif key == "plugin_name":
                plugin = val
            elif key == "generation" and val == "4.2":
                is_evt3 = True
            elif key in ("geometry", "format"):
                if key == "format" and val.split(";")[0] == "EVT3":
                    is_evt3 = True
                geometry = _parse_geometry(key, val)
                if geometry is not None and (key == "geometry" or not have_geometry):
                    (width, height), have_geometry = geometry, True
        if line == "% evt 3.0":
            is_evt3 = True
    if integrator == "Prophesee" and plugin in ("hal_plugin_gen41_evk2", "hal_plugin_imx636_evk4"):
        width, height, have_geometry = 1280, 720, True
    elif integrator == "Prophesee" and plugin == "evc3a_plugin_gen31":
        width, height, have_geometry = 640, 480, True
    elif integrator == "CenturyArks" and (plugin == "evc4a_plugin_imx636" or not have_geometry):
        width, height, have_geometry = 1280, 720, True
    if not have_geometry:
        width, height = 1280, 720
    return width, height, is_evt3, f.tell(), have_geometry


MAX_TIMESTAMP_BASE = ((1 << 12) - 1) << 12
TIME_LOOP = MAX_TIMESTAMP_BASE + (1 << 12)
LOOP_THRESHOLD = 10 << 12
CHUNK_WORDS = 1 << 21


@njit(cache=True, nogil=True)
def _first_time_high(words):
    """Index of the first EVT_TIME_HIGH (the words before it are skipped: time base unknown)."""
    for i in range(words.shape[0]):
        if (words[i] >> 12) == 0x8:
            return i
    return words.shape[0]


@njit(cache=True, nogil=True, parallel=True)
def _summarize(words, starts, height, summ):
    """What each chunk words[starts[c]:starts[c + 1]] does to the decoder state, whatever the state at its start.

    summ[c] = (has_y, y, has_base_x, pol, x_base, x_advance, n_time_high, first, last, wraps, last_time_kind,
    time_low): the last EVT_ADDR_Y; the last VECT_BASE_X and the x base after the vector words that follow it
    (else the total advance of the vector words); the first / last EVT_TIME_HIGH values and the time loops
    between them; the kind of the last time word (0 none, 1 high, 2 low) and the last EVT_TIME_LOW value.
    """
    for c in prange(starts.size - 1):
        has_y = 0
        y = 0
        has_bx = 0
        pol = 0
        xb = 0
        adv = 0
        nth = 0
        first = 0
        prev = 0
        wraps = 0
        kind = 0
        low = 0
        for i in range(starts[c], starts[c + 1]):
            w = words[i]
            typ = w >> 12
            if typ == 0x4 or typ == 0x5:
                nb = 12 if typ == 0x4 else 8
                if has_bx:
                    xb = (xb + nb) & 0xFFFF
                else:
                    adv = (adv + nb) & 0xFFFF
            elif typ == 0x0:
                has_y = 1
                y = (w & 0x7FF) % height
            elif typ == 0x3:
                has_bx = 1
                pol = (w >> 11) & 0x1
                xb = w & 0x7FF
            elif typ == 0x8:
                v = np.int64(w & 0xFFF)
                if nth == 0:
                    first = v
                elif prev > v and (prev - v) << 12 >= MAX_TIMESTAMP_BASE - LOOP_THRESHOLD:
                    wraps += 1  # (the decoder compares time bases: both carry the same loop offset)
                prev = v
                nth += 1
                kind = 1
            elif typ == 0x6:
                kind = 2
                low = w & 0xFFF
        summ[c, 0] = has_y
        summ[c, 1] = y
        summ[c, 2] = has_bx
        summ[c, 3] = pol
        summ[c, 4] = xb
        summ[c, 5] = adv
        summ[c, 6] = nth
        summ[c, 7] = first
        summ[c, 8] = prev
        summ[c, 9] = wraps
        summ[c, 10] = kind
        summ[c, 11] = low


@njit(cache=True, nogil=True)
def _chain(summ, first_time_base, state):
    """Decoder state at the start of every chunk (state[c] = time base, loop count, time, y, x base, polarity),
    from the state of the sequential decoder at the first EVT_TIME_HIGH."""
    base = first_time_base
    nl = np.int64(0)
    t = np.int64(0)
    y = 0
    xb = 0
    pol = 0
    for c in range(summ.shape[0]):
        state[c, 0] = base
        state[c, 1] = nl
        state[c, 2] = t
        state[c, 3] = y
        state[c, 4] = xb
        state[c, 5] = pol
        if summ[c, 0]:
            y = summ[c, 1]
        if summ[c, 2]:
            pol = summ[c, 3]
            xb = summ[c, 4]
        else:
            xb = (xb + summ[c, 5]) & 0xFFFF
        if summ[c, 6]:
            nb = (summ[c, 7] << 12) + nl * TIME_LOOP
            if base > nb and base - nb >= MAX_TIMESTAMP_BASE - LOOP_THRESHOLD:
                nl += 1
            nl += summ[c, 9]
            base = (summ[c, 8] << 12) + nl * TIME_LOOP
        if summ[c, 10] == 1:
            t = base
        elif summ[c, 10] == 2:
            t = base + summ[c, 11]


@njit(cache=True, nogil=True)
def _scan(words, a, b, st, width, height, time_stamp, mode, out0, limit, xs, ys, ts, ps):
    """The decoder loop on words[a:b] from state st. mode 0: time of the first CD event (-1 if none);
    mode 1: number of kept events (time > time_stamp); mode 2: writes the kept events from index out0, below limit.
    """
    current_time_base = st[0]
    n_loop = st[1]
    current_time = st[2]
    current_y = st[3]
    current_x_base = st[4]
    current_pol = st[5]
    n = out0
    for i in range(a, b):
        w = words[i]
        typ = w >> 12
        if typ == 0x2:                        # EVT_ADDR_X
            if mode == 0:
                return current_time
            if current_time > time_stamp:
                if mode == 2 and n < limit:
                    xs[n] = (w & 0x7FF) % width
                    ys[n] = current_y
                    ps[n] = (w >> 11) & 0x1
                    ts[n] = current_time - time_stamp
                n += 1
        elif typ == 0x4 or typ == 0x5:        # VECT_12 / VECT_8
            nbits = 12 if typ == 0x4 else 8
            end = (current_x_base + nbits) & 0xFFFF
            valid = w & 0xFFF if typ == 0x4 else w & 0xFF
            if mode == 0 and valid:
                return current_time
            if current_time > time_stamp:
                xi = current_x_base
                while xi != end:
                    if valid & 0x1:
                        if mode == 2 and n < limit:
                            xs[n] = xi % width
                            ys[n] = current_y
                            ps[n] = current_pol
                            ts[n] = current_time - time_stamp
                        n += 1
                    valid >>= 1
                    xi = (xi + 1) & 0xFFFF
            current_x_base = end
        elif typ == 0x0:                      # EVT_ADDR_Y
            current_y = (w & 0x7FF) % height
        elif typ == 0x3:                      # VECT_BASE_X
            current_pol = (w >> 11) & 0x1
            current_x_base = w & 0x7FF
        elif typ == 0x8:                      # EVT_TIME_HIGH
            new_base = (np.int64(w & 0xFFF) << 12) + n_loop * TIME_LOOP
            if current_time_base > new_base and current_time_base - new_base >= MAX_TIMESTAMP_BASE - LOOP_THRESHOLD:
                new_base += TIME_LOOP
                n_loop += 1
            current_time_base = new_base
            current_time = current_time_base
        elif typ == 0x6:                      # EVT_TIME_LOW
            current_time = current_time_base + (w & 0xFFF)
    if mode == 0:
        return -1
    return n - out0


@njit(cache=True, nogil=True, parallel=True)
def _count(words, starts, state, width, height, time_stamp, counts, xs, ys, ts, ps):
    for c in prange(starts.size - 1):
        counts[c] = _scan(words, starts[c], starts[c + 1], state[c], width, height, time_stamp, 1, 0, 0,
                          xs, ys, ts, ps)


@njit(cache=True, nogil=True, parallel=True)
def _write(words, starts, state, width, height, time_stamp, offsets, limit, xs, ys, ts, ps):
    for c in prange(starts.size - 1):
        if offsets[c] < limit:
            _scan(words, starts[c], starts[c + 1], state[c], width, height, time_stamp, 2, offsets[c], limit,
                  xs, ys, ts, ps)


@njit(cache=True, nogil=True, parallel=True)
def _beyond(ts, n, t_end):
    """Whether any of ts[:n] exceeds t_end."""
    hit = 0
    for k in prange(n):
        if ts[k] > t_end:
            hit += 1
    return hit > 0


@njit(cache=True, nogil=True)
def _clamp(ts, n):
    """pyEBIV consistency check: times beyond the last event time are clamped to the previous event's time."""
    t_end = ts[n - 1]
    if ts[0] > t_end:
        ts[0] = 0
    t_prev = ts[0]
    for k in range(n):
        if ts[k] > t_end:
            ts[k] = t_prev
        t_prev = ts[k]


def _decode(words, width, height, max_events, t_dtype=np.int64):
    """Events of the EVT 3.0 words (x, y, t, p, time_stamp), as the sequential pyEBIV decoder gives them."""
    n_words = words.shape[0]
    xs = np.zeros(0, np.uint16)
    ys = np.zeros(0, np.uint16)
    ts = np.zeros(0, t_dtype)
    ps = np.zeros(0, np.int8)
    i0 = int(_first_time_high(words))
    if i0 >= n_words:
        return xs, ys, ts, ps, 0
    n_chunks = max(1, (n_words - i0) // CHUNK_WORDS)
    starts = np.linspace(i0, n_words, n_chunks + 1).astype(np.int64)
    summ = np.zeros((n_chunks, 12), np.int64)
    _summarize(words, starts, height, summ)
    state = np.zeros((n_chunks, 6), np.int64)
    _chain(summ, np.int64(words[i0] & 0xFFF) << 12, state)
    # time of the first CD event: every event at or before it is dropped (pyEBIV keeps t - t_first > 0)
    time_stamp = -1
    for c in range(n_chunks):
        time_stamp = _scan(words, starts[c], starts[c + 1], state[c], width, height, 0, 0, 0, 0, xs, ys, ts, ps)
        if time_stamp >= 0:
            break
    if time_stamp < 0:
        return xs, ys, ts, ps, 0
    counts = np.zeros(n_chunks, np.int64)
    _count(words, starts, state, width, height, time_stamp, counts, xs, ys, ts, ps)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    n = int(min(offsets[-1], max_events))
    xs = np.empty(n, np.uint16)
    ys = np.empty(n, np.uint16)
    ts = np.empty(n, t_dtype)
    ps = np.empty(n, np.int8)
    _write(words, starts, state, width, height, time_stamp, offsets, n, xs, ys, ts, ps)
    if n > 0 and (ts[0] > ts[n - 1] or _beyond(ts, n, ts[n - 1])):
        _clamp(ts, n)
    return xs, ys, ts, ps, int(time_stamp)


def load_raw(file_path: str, max_events: int = 700_000_000, t_dtype=np.int64) -> RawEvents:
    """Events of an EVT 3.0 .raw file; t_dtype: dtype of the times (exact integers in float64 too)."""
    with open(file_path, "rb") as f:
        width, height, is_evt3, offset, geometry_found = _read_header(f)
    if not is_evt3:
        raise ValueError(f"not an EVT 3.0 event data file: {file_path}")
    size = os.path.getsize(file_path) - offset
    words = np.memmap(file_path, dtype="<u2", mode="r", offset=offset, shape=(size // 2,))
    xs, ys, ts, ps, time_stamp = _decode(words, width, height, max_events, t_dtype)
    del words
    return RawEvents(xs, ys, ts, ps, width, height, time_stamp, geometry_found)
