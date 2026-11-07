"""Direct numpy rendering of the pseudo-frame / cluster views into Dear PyGui textures.

These replace matplotlib pcolor() figures: pcolor builds one polygon per pixel
(921 600 for a 1280 x 720 frame) and took 1-3 s per refresh; filling the RGBA
array directly takes a few milliseconds and gives the same picture.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps


def _lut(name: str, n: int = 256) -> np.ndarray:
    return colormaps[name](np.linspace(0.0, 1.0, n)).astype(np.float32)


_LUTS: dict[str, np.ndarray] = {}


def lut(name: str) -> np.ndarray:
    if name not in _LUTS:
        _LUTS[name] = _lut(name)
    return _LUTS[name]


def binary_rgba(I: np.ndarray, off_rgb, on_rgb) -> np.ndarray:
    """(H, W) image -> (H, W, 4) float RGBA: off colour where I <= 0, on colour elsewhere."""
    h, w = I.shape
    rgba = np.empty((h, w, 4), np.float32)
    rgba[:, :, :3] = off_rgb
    rgba[I > 0, :3] = on_rgb
    rgba[:, :, 3] = 1.0
    return rgba


def scalar_rgba(I: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """(H, W) values -> RGBA with the colormap normalised to [min, max] (as pcolor does)."""
    vmin = float(I.min()) if I.size else 0.0
    vmax = float(I.max()) if I.size else 0.0
    table = lut(cmap)
    if vmax > vmin:
        idx = ((I.astype(np.float32) - vmin) * ((len(table) - 1) / (vmax - vmin))).astype(np.intp)
    else:
        idx = np.zeros(I.shape, np.intp)
    return table[idx]


def draw_dots(rgba: np.ndarray, xs, ys, radius: float = 2.0, rgb=(1.0, 0.0, 0.0), alpha: float = 0.8) -> None:
    """Blends filled discs centred on the (x, y) pixel positions into an RGBA image."""
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(ok):
        return
    h, w = rgba.shape[:2]
    r = int(np.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (ox * ox + oy * oy) <= radius * radius
    ox, oy = ox[disc], oy[disc]
    px = (np.floor(xs[ok]).astype(np.int64)[:, None] + ox[None, :]).ravel()
    py = (np.floor(ys[ok]).astype(np.int64)[:, None] + oy[None, :]).ravel()
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py = px[inside], py[inside]
    if px.size == 0:
        return
    flat = np.unique(py * w + px)          # each pixel blended once even where discs overlap
    view = rgba.reshape(-1, 4)
    view[flat, :3] = view[flat, :3] * (1.0 - alpha) + np.asarray(rgb, np.float32) * alpha
