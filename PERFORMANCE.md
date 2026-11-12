# Performance work in this fork (`performance` and `numba_convert` branches)

STELLA was very slow on real recordings: stepping through the pseudo-frames of a few-seconds recording took
minutes, and a full detection run could take hours or fail. This branch profiles and optimises it without changing
its features, its GUI or its results. The original algorithms are kept; only how they are computed changed.

## Results

Measured on a 96 M-event recording (`.raw` EVT 3.0, 340 MB, 1280 x 720), Python 3.12, numpy 2.5, numba 0.68,
16-thread CPU.

| Operation | Before | After |
|---|---|---|
| Load the `.raw` file | 37.5 s | 2.3 s |
| Next / previous pseudo-frame, Preprocessing tab | 0.9 s | 0.012 s |
| Next / previous pseudo-frame, Detection tab | 3.4 s | 0.04 s |
| Detection, 200 time steps (pixelwise extension) | 49 s | 5.7 s |
| Detection, whole recording | ~40 min, then crash (NameError, 41 GB of memory) | 220 s, ~5 GB, 286 081 clusters |

### `numba_convert` branch: compiled detection

The pixelwise-extension detection (the default method) spent ~98% of its time in Python bookkeeping: matching the
~5000 regions of every 2 ms step to the active clusters with numpy calls per region, re-concatenating every cluster
history at every step, then post-processing millions of track slots one by one. That work is now compiled.
Measured with Python 3.12, numpy 2.5, numba 0.68, scipy 1.18 on an 8-core / 16-thread Ryzen 7 7800X3D, 64 GB.

| Detection, whole recording (pixelwise extension, GUI defaults) | `performance` | `numba_convert` |
|---|---|---|
| 96 M events, 15.4 s, 7682 steps (`_1.raw`, 340 MB) | 257 s | 18.6 s, 3.7 GB peak |
| 553 M events, 12.0 s, 5988 steps (1.5 GB) | 50 min, 31.6 GB peak (*) | 95 s, 21.1 GB peak |

(*) partly measured while other runs shared the CPU; alone it runs at ~374 ms per step, ~40 min in total.

Per 2 ms step on the 553 M-event recording (~130 000 events, ~5000 regions): 374 ms before, ~10 ms now. Multi-run
and larger mask sizes (dilated regions) are 3-14x faster; the DBSCAN and pseudo-frame methods are unchanged.

## Verification

- `.raw` decoding gives exactly the same events as pyEBIV (all 96 170 404 events compared).
- Every detection method (including multi-run) gives results identical to the original code on the provided
  datasets and on the recording above.
- `numba_convert`: every detection output (`x_clust`, `y_clust`, `t_clust`, `ind_clust`, `steps`, `x_m`, `y_m`,
  `t_m`: dtype, length and bytes of every array) is identical to the `performance` branch on both whole recordings
  above, with multi-run, with mask sizes 2 and 3, and with the DBSCAN and pseudo-frame methods. The compiled
  labelling, median and track-linking kernels were also checked against scipy / numpy on thousands of random cases.
- Tracking: Kalman filter identical within 1e-8 relative, B-spline identical, hybrid within 2e-7 relative
  (floating-point summation order).

## What was changed

**Loading** (`utils/raw_loader.py`, new; `utils/dataloader.py`, `utils/helpers.py`)
- Numba port of the EVT 3.0 decoder of pyEBIV, reading the file through a memory map: ~10x faster than pyEBIV
  and pyEBIV is no longer needed to open `.raw` files.
- Event times are kept in float64. float32 times lose microsecond precision after 16.7 s of recording.
- The file was loaded twice when validated; it is now loaded once.

**Navigation** (`interface/tab_preprocessing.py`, `interface/tab_detection.py`, `utils/raster.py`, new)
- Events are time-sorted once; every time window is then found by binary search (`np.searchsorted`) instead of
  masks over the whole recording.
- Pseudo-frames and cluster images are drawn as numpy RGBA textures instead of matplotlib `pcolor`
  (921 600 polygons per frame), with the same colour scaling.
- The cluster overlays and statistics only visit the clusters / tracks inside the displayed time window
  (`utils/track_index.py`, new; cached time ranges).

**Detection** (`utils/detection.py`, `utils/utils_v2.py`)
- Python loops over pixels and labels replaced by vectorised numpy (grouping by label with one stable sort,
  label clearing without full-frame comparisons); an O(n²) lookup in the pseudo-frame method replaced by slices.
- `np.median` in the cluster update replaced by a numba median that is bit-identical to it.
- Bug fixes: extended track slots were allocated with `max_length` NaNs each (the cause of the 41 GB), now empty
  placeholders; a leftover reference to undefined `P_new` / `P_newz` that crashed long runs was removed.
- `hdbscan` is imported only when the HDBSCAN clustering is used.

**Compiled detection, `numba_convert` branch** (`utils/pixelwise_numba.py`, new; `utils/detection.py`,
`utils/utils_v2.py`)
- `PixelwiseTracker` runs the per-step association of the pixelwise extension in numba (steps 1 to N of each
  pass; step 0 and the other methods keep the Python code, which `numba_matching=False` in the context forces).
  Same algorithm, same tie-breaking and same quirks (slot numbering and reuse, the `max(label)` slot extension,
  slot 0 never taking part in a split, medians including the overlap events). Distances in the region split
  are squared with the C `pow`, as numpy's scalar `** 2` does, since it is not always the correctly rounded `x * x`.
- Cluster histories are an append-only log of (track, event index), assembled once at the end of the pass from the
  global event arrays, instead of being re-concatenated at every step (quadratic in the track length).
- Connected regions of the events (no dilation, the default mask size 1) come from a two-pass union-find labelling
  that numbers regions like `scipy.ndimage.label`; with dilation scipy is still used.
- Medians by quickselect (identical values), free slots kept in a min-heap, only the slots that can change are
  visited each step.
- `repair_tracks`: start / end medians computed in compiled chunks (in parallel), and the link candidates found
  through a time / space grid instead of ball queries that returned ~200 neighbours each. The first acceptable
  neighbour is the one `cKDTree` would list first: its traversal returns points in the order of `tree.indices`.
- End of pass: the clustered events are removed from the global arrays only when a second pass follows, the slot
  lists are reset with a shared empty array and handed over without copying every cluster (millions of slots on
  long recordings).

**Tracking** (`utils/kalman_filter_v2.py`, `interface/tab_tracking.py`, `interface/tab_validation.py`)
- The Kalman filter measures position only (C = [1, 0, 0]), so the innovation covariance is a scalar: a scalar
  numba step replaces the matrix inversion (the general path is kept for other models).
- The Kalman functions release the GIL (`nogil=True`), so STELLA's thread pool really runs in parallel.
- `matplotlib.cm.get_cmap` (removed in matplotlib 3.9) replaced by `matplotlib.colormaps`: the tracking
  pseudo-frame overlay crashed with current matplotlib.

**Usability** (`interface/tab_detection.py`)
- *Test run* now detects in the displayed pseudo-frame instead of always the first one. Previously, after
  stepping forward, the event-cluster view and "show found cluster centers" showed nothing because the test
  clusters belonged to the first frame.
- After each refresh the status log says how many clusters the displayed frame contains (with a hint when none).

## What was considered and not done

- A GPU port of the detection: the per-frame image work (mask, labelling, grouping) runs 12 ms -> 2.1 ms per frame
  with CuPy on an RTX 5090, but it was <3% of the detection time; the association is branchy per-region work that
  suits compiled CPU code better. With the association compiled, a GPU could still save a few ms per step.
- GPU (CUDA) for the Kalman filter: all tracks of the recording above batch on the CPU in ~5 ms, so a GPU cannot
  help.
- A cross-correlation (optical flow) detection method, which would profit from a GPU (~20x measured with CuPy
  for pseudo-frame cross-correlation), was left for later.

## Installation and use

Unchanged from the main README: Python 3.12, `pip install -r requirements.txt`, `python main.py`. numba was already
a dependency. The first start compiles the numba functions once (cached in `__pycache__`), so it is slower than the
following ones.

To update an existing STELLA clone to one of these branches:

```
git remote add fork https://github.com/RexAttaque/STELLA.git
git fetch fork
git checkout -b performance fork/performance       # or: git checkout -b numba_convert fork/numba_convert
```
