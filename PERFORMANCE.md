# Performance work in this fork (`performance`, `numba_convert`, `flat-array_and_optim`, `KD_tree_replace` and `parallel` branches)

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

### `flat-array_and_optim` branch: flat track arrays

The detection result held each quantity (`x_clust`, `y_clust`, `t_clust`, `ind_clust`) as a Python list of one
numpy array per track: 3.2 million slots and 2.3 million final tracks on the large recording. Every pass over all
tracks (end of pass, `repair_tracks`, the GUI statistics, saving) paid Python's per-object cost. The tracks are now
stored back to back in one flat array per quantity (`PackedTracks`), and the labelling of the next time step runs
in a second thread while the current one is matched. Both columns measured back to back on the same machine:

| Detection, whole recording (pixelwise extension, GUI defaults) | `numba_convert` | `flat-array_and_optim` |
|---|---|---|
| 96 M events, 7682 steps (`_1.raw`) | 12.3 s, 2.5 GB peak | 7.3 s, 2.5 GB peak |
| 553 M events, 5988 steps (1.5 GB) | 87.8 s, 21.1 GB peak | 52.0 s, 17.0 GB peak |

On the large recording: the time steps went from 8.7 to 6.4 ms each (labelling overlapped, ~14 s) and the end of
the pass from ~50 s to ~17 s. GUI work on a result with 57 900 tracks (the factor grows with the number of tracks):

| | lists | flat arrays |
|---|---|---|
| Detection statistics (Run detection / Test run) | 1.3 s | 0.14 s |
| Clusters of the displayed frame (first call, builds the time index) | 0.13 s | 0.013 s |
| Cluster order of the tracking tab | 0.2 s | 0.002 s |
| Pickling the result for *Save results* | 0.42 s | 0.01 s |

### `KD_tree_replace` branch: track-repair links without a KD-tree

`repair_tracks` joins a track that starts where, and right after, another one ends. When several track ends
qualify for the same start, the original takes the first one returned by a `cKDTree` ball query. That order is the
tree's internal point order, so the choice between them is arbitrary. This branch links every start to the
**nearest** qualifying end instead (then the smaller time gap, then the lower track index), and searches the
candidates of all starts in parallel. The tree is no longer built.

| Large recording (553 M events) | `flat-array_and_optim` | `KD_tree_replace` |
|---|---|---|
| Link search (end of pass + final repair) | 3.4 s + 2.5 s | 0.75 s + 0.56 s |
| Whole detection (3 runs each; full runs vary by several seconds on this machine) | 52.0-58.7 s | 47.6 s |

The tracks change where a start had several candidates: 0.03 % of the tracks on the 96 M-event recording and
1.1 % on the 553 M-event one (26 000 of 2.3 million tracks, 1.9 % of the events). Where the two rules pick
different ends, the joined start and end are 3.1 px apart on average instead of 4.3 px.

The first commit of the branch (`ff0de96`, "Search track-repair links in parallel") keeps the tree order and
identical results, with the link search at 1.3 s + 0.9 s.

### `parallel` branch: all CPU cores, same results

Everything that took noticeable time now runs on all CPU cores. Results are identical to `KD_tree_replace`.
Times on an 8-core / 16-thread CPU (Ryzen 7 7800X3D), measured back to back:

| | `KD_tree_replace` | `parallel` |
|---|---|---|
| Loading the 553 M-event `.raw` | ~10 s | 1.8 s |
| Detection, 553 M events (5988 steps) | 47.6 s | 20.8 s |
| Detection, 96 M events (7682 steps) | 8.1 s | 5.1 s |
| Detection, 96 M events, mask size 3, 400 steps | 5.7 s | 1.8 s |
| Detection, 96 M events, mask size 2 + multi-run, 800 steps | 11.5 s | 2.7 s |
| Kalman tracking, 286 000 tracks (whole 96 M-event recording) | 47.0 s | 9.8 s |
| Validation, neighbourhood filter, 58 000 tracks | 109 s | 5.5 s |

- **Matching** (`_match`): a step's regions were matched one after the other because a cluster taken by a region
  leaves the label map for the next ones. That is the only link between regions, so the step now runs in passes:
  the clusters under every region and the split of shared regions in parallel; then one short sequential pass
  in the original order that removes the clusters already taken (and recomputes a split only when one of its
  clusters was taken); then the matched clusters (event copies, medians, history) in parallel, each written
  where the sequential loop wrote it. 6.5 ms -> 1.4 ms per step on the large recording.
- **Labelling** is compiled for every mask size (the dilation too, same labels as scipy), and three threads
  prepare the next steps (time window, labelling, grouping by region) while the current one is matched. Mask
  sizes 2 and 3 no longer go through scipy (10-13 ms per step -> 2.5-3.5 ms).
- **End of pass**: the history log stores one track id per cluster and step instead of one per event
  (1.4 GB less on the large recording) and is assembled in parallel (1.8 s -> 0.4 s).
- **Loading**: the EVT 3.0 decoder runs on chunks of the file in parallel. The decoder state at the start of each
  chunk (time base, last time, y, x base, polarity) comes from per-chunk summaries chained in order; the events
  are then counted and written per chunk into arrays of the exact size, times directly as float64.
- **Tracking** (Kalman and the hybrids): the least-squares start estimates of all tracks with the same number of
  points go to numpy's `lstsq` in one call; the compiled Kalman filter runs over chunks of tracks in the worker
  threads (only it releases the GIL, so one task per track mostly waited); the post-processing (NaN removal,
  smoothing, truncation) is one parallel compiled pass.
- **Validation**: the neighbourhood filter scanned every track for every time bin; the points of all bins are
  now gathered in one pass.
- **CPU threads** (`utils/parallel.py`, detection tab, "CPU threads (0 = auto)"): automatic uses every logical CPU
  but four, left to the three step-preparing threads and the GUI. numba's OpenMP threads spin ~0.1 s after each
  parallel loop before sleeping; with every CPU taken, the preparing threads would wait for a core (16 kernel
  threads 32 s vs 12 threads 20.8 s). Steps with fewer than 20 000 events run on one thread: waking the threads
  costs more than they save (the whole 96 M-event recording: 8.8 s parallel vs 6.3 s serial before the third
  preparing thread).

## Verification

- `.raw` decoding gives exactly the same events as pyEBIV (all 96 170 404 events compared).
- Every detection method (including multi-run) gives results identical to the original code on the provided
  datasets and on the recording above.
- `numba_convert`: every detection output (`x_clust`, `y_clust`, `t_clust`, `ind_clust`, `steps`, `x_m`, `y_m`,
  `t_m`: dtype, length and bytes of every array) is identical to the `performance` branch on both whole recordings
  above, with multi-run, with mask sizes 2 and 3, and with the DBSCAN and pseudo-frame methods. The compiled
  labelling, median and track-linking kernels were also checked against scipy / numpy on thousands of random cases.
- `flat-array_and_optim`: every track of every detection output holds the same values as on `numba_convert`
  (both whole recordings, multi-run, mask sizes 2 and 3; DBSCAN, pseudo-frame and `numba_matching=False`, which
  keep lists, are identical including dtypes). The only difference: tracks that the list version turned into
  float64 by concatenating them with an emptied `np.array([])` (2625 of 2.3 million on the large recording) keep
  the dtype of the others (uint16 coordinates, int64 indices). The GUI statistics, frame index, duration
  histogram and cluster order, and the Kalman tracking, give identical results from flat arrays and from lists.
- `KD_tree_replace`: the link search gives the same links as a brute-force nearest-first search on 300 random
  cases (149 000 links), and the list and flat-array versions of `repair_tracks` give identical tracks. Its first
  commit (tree order kept) gives outputs identical to `flat-array_and_optim` in every mode listed above.
- `parallel`: every detection output identical (dtype, length and bytes of every array) to `KD_tree_replace` on
  both whole recordings (the large one loaded with the new decoder), with multi-run, mask sizes 2 and 3 and the
  Python matching path; the compiled matching also on captured real steps with one thread and with 16. The
  parallel decoder gives the same events as the sequential one on both recordings and on 400 random word streams
  cut into chunks of 3-200 words; the compiled labelling matches scipy (dilation included) on 6000 random cases
  and on real steps. Tracking outputs are identical in every mode using the Kalman stage (Kalman, with
  correction, the three hybrids, lightweight); the validation flags and deviations are identical on all 58 000
  tracks.
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

**Flat track arrays, `flat-array_and_optim` branch** (`utils/packed_tracks.py`, new; `utils/detection.py`,
`utils/pixelwise_numba.py`, `utils/utils_v2.py`, `interface/tab_detection.py`, `interface/tab_tracking.py`)
- `PackedTracks`: one flat array plus the [start, stop) range of every track. It behaves like the list it
  replaces (`len`, `x[i]` returns a view, iteration, replacing a track by values of the same length), so code
  written for lists keeps working; whole-collection operations (lengths, filters, per-track min / max / counts,
  distinct values) are vectorised or compiled. Pickling stores it compactly.
- The pixelwise-extension detection produces `PackedTracks` directly from the tracker: the end-of-pass filters
  only change ranges, `repair_tracks` works on the flat arrays (track ends computed in place, every linked track
  written once, same links and contents as the list version) and the final result is compact. The other methods
  still return lists.
- The time window and labelling of step k + 1 are prepared in a worker thread while step k is matched (the
  compiled labelling releases the GIL).
- GUI: the detection statistics, the per-frame cluster index, the duration histogram and the tracking-tab cluster
  order use the flat arrays; lists (results of the other methods, files saved before) take the previous code.
  The tracking script gets lists of track views (it indexes every track many times from several threads, which is
  ~20x faster on a list), so tracking no longer re-sorts the detection result in place as a side effect.
- Results saved with *Save results* store `PackedTracks` and need this branch to load (`utils/packed_tracks.py`);
  older files load as before.

**Track-repair links, `KD_tree_replace` branch** (`utils/utils_v2.py`)
- `_link_candidates` lists, for every track start in parallel, the track ends within 5 px whose timing allows a
  link, through the same time / space grid as before. `_pick_links` then takes the starts in ascending order and
  links each one to its nearest end that is not used yet. `cKDTree` is no longer imported.

**All cores, `parallel` branch** (`utils/pixelwise_numba.py`, `utils/detection.py`, `utils/parallel.py`, new;
`utils/raw_loader.py`, `utils/dataloader.py`, `utils/tracking_batch.py`, new; `utils/tracking.py`,
`interface/tab_detection.py`, `interface/tab_validation.py`)
- `_match` and `_create` are parallel count / resolve / fill passes (see Results); `label_events` takes the
  dilation offsets; `_group` runs in the preparing threads; the history log is stored as runs.
- `utils/parallel.py` holds the thread count; every module with parallel kernels imports it, and the detection
  script applies it to its thread (numba's count is per thread).
- `raw_loader._decode` is the chunked parallel decoder. Its `max_events` limit is now exact: the sequential decoder
  checked it only between words, so a vector word could write up to 11 events past its buffers (only for files
  above the limit, 700 M events by default).
- `tracking_batch.kalman_afterwards` / `postprocess_afterwards` replace the per-track thread pool and the
  per-track post-processing; the smoothing reproduces `scipy.ndimage.uniform_filter1d(size=3, mode='nearest')`
  operation for operation (running sum divided at output), hence identical bits. Arrays of another dtype take the
  per-track code.
- `ValidationTab._points_by_time_bin` gathers the points of all time bins; the per-point statistics are unchanged.

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

### GPU assessment (`parallel` branch)

Where the 20.8 s of the large recording go: matching ~8 s on the main thread (1.4 ms per step); labelling and
grouping ~20 s of thread time on the three preparing threads, in parallel with the matching; end of pass ~6 s
(track repair 4 s, export 1.6 s); step 0 and setup ~1.5 s.

| Candidate | Fit for a GPU | Expected gain |
|---|---|---|
| Labelling + grouping of many steps at once | Good: per-pixel union-find, steps independent | 1-2 s (frees the preparing threads' cores for the matching) |
| Matching | The parallel passes are count / prefix-sum / fill kernels over flat arrays and port directly; the resolution pass is sequential in every step (one GPU thread or a CPU round trip per step); the whole tracker state must live on the GPU | 3-5 s, large port |
| End-of-pass gathers (350 M events) | Memory-bound, fast on a GPU, but ~7 GB of events cross PCIe each way unless they already live there | ~3 s |
| Loading | Chunked decoding ports, file read stays | <= 1 s |

A GPU path for the first three could bring the detection to roughly 12-14 s. Not recommended yet: on large
recordings the remaining costs are now after the detection (next section) and can be compiled on the CPU first. The
kernels were written so that a port stays possible: flat arrays, counts then prefix sums then independent writes.

### Remaining CPU work, not done to keep identical results

- Tracking still spends ~30 us of Python per track (numpy's sort per track, which fixes the order of equal times,
  and the per-track result objects): about 80 s for the 2.3 million tracks of the large recording. Fully compiled,
  it would take a few seconds, with results equal within floating-point precision but not bit for bit (sort order
  of equal times, LAPACK vs closed-form least squares).
- Validation: the per-point neighbourhood statistics (5.5 s for 58 000 tracks) and the quality filter (3.5 s) are
  per-point / per-track numpy. Compiled, ~100x faster, but `np.exp` and the sort order of ties could differ in the
  last bit and flip a flag in rare cases.
- Step 0 of the detection still runs the original Python code (~1 s).

## Installation and use

Unchanged from the main README: Python 3.12, `pip install -r requirements.txt`, `python main.py`. numba was already
a dependency. The first start compiles the numba functions once (cached in `__pycache__`), so it is slower than the
following ones.

To update an existing STELLA clone to one of these branches:

```
git remote add fork https://github.com/RexAttaque/STELLA.git
git fetch fork
git checkout -b performance fork/performance       # or numba_convert, flat-array_and_optim, KD_tree_replace, parallel
```
