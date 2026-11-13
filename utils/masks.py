"""Polygon masks of the events, drawn on the Preprocessing pseudo-frame and applied to the events before detection
(so they act on the whole chain).

An "exclude" mask removes the events inside it; "keep" masks keep only the events inside them (inside one of them
when there are several). A pixel is inside a polygon when its centre is (even-odd rule); coordinates are 0-based
sensor pixels, as the events.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

import numpy as np

MASK_MODES = ("exclude", "keep")


@dataclass
class Mask:
    name: str
    points: List[List[float]] = field(default_factory=list)  # [[x, y], ...] in sensor pixels
    enabled: bool = True
    mode: str = "exclude"  # one of MASK_MODES


def rasterize(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """(height, width) boolean image of the pixels whose centre is inside the polygon (scanline fill)."""
    out = np.zeros((height, width), dtype=bool)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return out
    xa, ya = pts[:, 0], pts[:, 1]
    xb, yb = np.roll(xa, 1), np.roll(ya, 1)
    y0 = max(0, int(np.floor(ya.min())))
    y1 = min(height - 1, int(np.ceil(ya.max())))
    for y in range(y0, y1 + 1):
        cy = y + 0.5
        crossing = (ya > cy) != (yb > cy)
        if not crossing.any():
            continue
        xs = np.sort(xa[crossing] + (cy - ya[crossing]) * (xb[crossing] - xa[crossing]) / (yb[crossing] - ya[crossing]))
        for k in range(0, len(xs) - 1, 2):  # pixel centres x + 0.5 in [xs[k], xs[k + 1])
            a = max(int(np.ceil(xs[k] - 0.5)), 0)
            b = min(int(np.ceil(xs[k + 1] - 0.5)) - 1, width - 1)
            if b >= a:
                out[y, a:b + 1] = True
    return out


def contains(points: Sequence[Sequence[float]], x: float, y: float) -> bool:
    """Whether the point is inside the polygon (even-odd rule)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return False
    inside = False
    for (xa, ya), (xb, yb) in zip(pts, np.roll(pts, 1, axis=0)):
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            inside = not inside
    return inside


def keep_map(masks: Sequence[Mask], width: int, height: int) -> Optional[np.ndarray]:
    """(height, width) boolean image of the pixels whose events are kept by the enabled masks; None when no enabled
    mask applies (everything kept)."""
    active = [m for m in masks if m.enabled and len(m.points) >= 3]
    if not active:
        return None
    keeps = [m for m in active if m.mode == "keep"]
    if keeps:
        keep = np.zeros((height, width), dtype=bool)
        for m in keeps:
            keep |= rasterize(m.points, width, height)
    else:
        keep = np.ones((height, width), dtype=bool)
    for m in active:
        if m.mode == "exclude":
            keep &= ~rasterize(m.points, width, height)
    return keep


def save_masks(path, masks: Sequence[Mask]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stella_masks": 1, "masks": [asdict(m) for m in masks]}, f, indent=1)


def load_masks(path) -> List[Mask]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    masks = []
    for d in data.get("masks", []):
        mode = d.get("mode", "exclude")
        masks.append(Mask(name=str(d.get("name", f"Mask {len(masks) + 1}")),
                          points=[[float(x), float(y)] for x, y in d.get("points", [])],
                          enabled=bool(d.get("enabled", True)),
                          mode=mode if mode in MASK_MODES else "exclude"))
    return masks
