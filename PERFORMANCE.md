# Performance work in this fork (`performance` branch)

STELLA was very slow on real recordings: stepping through the pseudo-frames of a few-seconds recording took
minutes, and a full detection run could take hours or fail. This branch profiles and optimises it without changing
its features, its GUI or its results. The original algorithms are kept; only how they are computed changed.

## Results

Measured on a 96 M-event recording (`.raw` EVT 3.0, 1.5 GB, 1280 x 720), Python 3.12, numpy 2.5, numba 0.68,
16-thread CPU.

| Operation | Before | After |
|---|---|---|
| Load the `.raw` file | 37.5 s | 2.3 s |
| Next / previous pseudo-frame, Preprocessing tab | 0.9 s | 0.012 s |
| Next / previous pseudo-frame, Detection tab | 3.4 s | 0.04 s |
| Detection, 200 time steps (pixelwise extension) | 49 s | 5.7 s |
| Detection, whole recording | ~40 min, then crash (NameError, 41 GB of memory) | 220 s, ~5 GB, 286 081 clusters |

## Verification

- `.raw` decoding gives exactly the same events as pyEBIV (all 96 170 404 events compared).
- Every detection method (including multi-run) gives results identical to the original code on the provided
  datasets and on the recording above.
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

- GPU (CUDA) for the Kalman filter: all tracks of the recording above batch on the CPU in ~5 ms, so a GPU cannot
  help.
- A cross-correlation (optical flow) detection method, which would profit from a GPU (~20x measured with CuPy
  for pseudo-frame cross-correlation), was left for later.

## Installation and use

Unchanged from the main README: Python 3.12, `pip install -r requirements.txt`, `python main.py`. numba was already
a dependency. The first start compiles the numba functions once (cached in `__pycache__`), so it is slower than the
following ones.

To update an existing STELLA clone to this branch:

```
git remote add fork https://github.com/RexAttaque/STELLA.git
git fetch fork
git checkout -b performance fork/performance
```
