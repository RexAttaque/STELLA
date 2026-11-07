"""Fast reader for Prophesee EVT 3.0 .raw files.

Numba port of the decoder used by pyEBIV (EBIV/pyebiv/src/ebi_events.cpp,
_loadRawEventData + the monotonic-time consistency check), so the events are
identical to pyEBIV.loadRaw() + time()/x()/y()/p(), but decoded straight into
numpy arrays: no Python lists, no second pass, a few seconds instead of
minutes for 100 M events.

Same conventions as pyEBIV:
- times are relative to the first CD event; events at exactly that time are
  dropped (pyEBIV keeps only t - t_first > 0);
- x is taken modulo the sensor width, y modulo the sensor height;
- events whose time exceeds the last event time get the previous event's time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class RawEvents:
    x: np.ndarray      # uint16
    y: np.ndarray      # uint16
    t: np.ndarray      # int64, microseconds relative to the first event
    p: np.ndarray      # int8, 0 / 1
    width: int
    height: int
    time_stamp: int    # absolute time of the first event (us)


def _read_header(f) -> tuple[int, int, bool, int]:
    """Parses the '%' header lines. Returns (width, height, is_evt3, data offset)."""
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
            elif key == "geometry":
                if val == "640x480":
                    width, height, have_geometry = 640, 480, True
                elif val == "1280x720":
                    width, height, have_geometry = 1280, 720, True
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
    return width, height, is_evt3, f.tell()


@njit(cache=True, nogil=True)
def _count_upper_bound(words):
    # every CD event needs at least one EVT_ADDR_X or one valid bit of a vector word
    n = 0
    for w in words:
        typ = w >> 12
        if typ == 0x2:
            n += 1
        elif typ == 0x4:
            v = w & 0xFFF
            while v:
                v &= v - 1
                n += 1
        elif typ == 0x5:
            v = w & 0xFF
            while v:
                v &= v - 1
                n += 1
    return n


@njit(cache=True, nogil=True)
def _decode(words, width, height, max_events, xs, ys, ts, ps):
    MaxTimestampBase = ((1 << 12) - 1) << 12
    TimeLoop = MaxTimestampBase + (1 << 12)
    LoopThreshold = 10 << 12

    n_words = words.shape[0]
    i = 0
    first_time_base = -1
    # skip words until the first EVT_TIME_HIGH (time base unknown before it)
    while i < n_words:
        w = words[i]
        if (w >> 12) == 0x8:
            first_time_base = np.int64(w & 0xFFF) << 12
            break
        i += 1
    current_time_base = max(first_time_base, 0)
    current_time = np.int64(0)
    current_y = 0
    current_x_base = 0
    current_pol = 0
    n_loop = 0
    is_cd = True
    ev_count = 0          # CD events seen (pyEBIV evCount)
    time_stamp = np.int64(0)
    n = 0                 # stored events
    while i < n_words and n < max_events:
        w = words[i]
        typ = w >> 12
        if typ == 0x2:                        # EVT_ADDR_X
            if is_cd:
                if ev_count == 0:
                    time_stamp = current_time
                if current_time - time_stamp > 0:
                    xs[n] = (w & 0x7FF) % width
                    ys[n] = current_y
                    ps[n] = (w >> 11) & 0x1
                    ts[n] = current_time - time_stamp
                    n += 1
                ev_count += 1
        elif typ == 0x4 or typ == 0x5:        # VECT_12 / VECT_8
            nbits = 12 if typ == 0x4 else 8
            end = (current_x_base + nbits) & 0xFFFF
            if is_cd:
                valid = w & 0xFFF if typ == 0x4 else w & 0xFF
                xi = current_x_base
                while xi != end:
                    if valid & 0x1:
                        if ev_count == 0:
                            time_stamp = current_time
                        if current_time - time_stamp > 0:
                            xs[n] = xi % width
                            ys[n] = current_y
                            ps[n] = current_pol
                            ts[n] = current_time - time_stamp
                            n += 1
                        ev_count += 1
                    valid >>= 1
                    xi = (xi + 1) & 0xFFFF
            current_x_base = end
        elif typ == 0x0:                      # EVT_ADDR_Y
            is_cd = True
            current_y = (w & 0x7FF) % height
        elif typ == 0x3:                      # VECT_BASE_X
            current_pol = (w >> 11) & 0x1
            current_x_base = w & 0x7FF
        elif typ == 0x8:                      # EVT_TIME_HIGH
            new_base = (np.int64(w & 0xFFF) << 12) + n_loop * TimeLoop
            if current_time_base > new_base and current_time_base - new_base >= MaxTimestampBase - LoopThreshold:
                new_base += TimeLoop
                n_loop += 1
            current_time_base = new_base
            current_time = current_time_base
        elif typ == 0x6:                      # EVT_TIME_LOW
            current_time = current_time_base + (w & 0xFFF)
        i += 1

    # pyEBIV consistency check: times beyond the last event time are clamped
    if n > 0:
        t_end = ts[n - 1]
        if ts[0] > t_end:
            ts[0] = 0
        t_prev = ts[0]
        for k in range(n):
            if ts[k] > t_end:
                ts[k] = t_prev
            t_prev = ts[k]
    return n, time_stamp


def load_raw(file_path: str, max_events: int = 700_000_000) -> RawEvents:
    with open(file_path, "rb") as f:
        width, height, is_evt3, offset = _read_header(f)
    if not is_evt3:
        raise ValueError(f"not an EVT 3.0 event data file: {file_path}")
    size = os.path.getsize(file_path) - offset
    words = np.memmap(file_path, dtype="<u2", mode="r", offset=offset, shape=(size // 2,))
    upper = min(_count_upper_bound(words), max_events)
    xs = np.empty(upper, np.uint16)
    ys = np.empty(upper, np.uint16)
    ts = np.empty(upper, np.int64)
    ps = np.empty(upper, np.int8)
    n, time_stamp = _decode(words, width, height, upper, xs, ys, ts, ps)
    del words
    if n < upper:   # shrink without keeping the over-allocation alive
        xs, ys, ts, ps = xs[:n].copy(), ys[:n].copy(), ts[:n].copy(), ps[:n].copy()
    return RawEvents(xs, ys, ts, ps, width, height, int(time_stamp))
