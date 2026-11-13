import numpy as np
import time
import warnings
from utils_v2 import * 
from scipy.ndimage import label
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import binary_dilation
import collections
from pixelwise_numba import PixelwiseTracker, label_events, dilation_offsets, _group
try:
    from utils.parallel import apply as _apply_threads
except ImportError:
    from parallel import apply as _apply_threads
_apply_threads()  # CPU-threads setting of the detection tab, for this (worker) thread
from concurrent.futures import ThreadPoolExecutor
import threading
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    test_mode
except NameError:
    test_mode = False
    
try:
    structure_size
except NameError:
    structure_size = 1

# A cluster is kept only if it was followed over at least min_steps time steps and holds at least min_events events.
# During the run, a cluster younger than min_steps that was not extended for 2 steps is retired (its slot reused).
min_steps = max(1, int(globals().get("min_steps", 3)))
min_events = max(1, int(globals().get("min_events", 20)))

structure_size = max(1, int(structure_size))
structure = np.ones((structure_size, structure_size), dtype=bool)

# %% init
stop_event = globals().get("stop_event", None)
x_clust = [[] for _ in range(max_tracks)]
y_clust = [[] for _ in range(max_tracks)]
t_clust = [[] for _ in range(max_tracks)]
ind_clust = [[] for _ in range(max_tracks)]
active_pixels = [[] for _ in range(max_tracks)]
x_est = [[] for _ in range(max_tracks)]
y_est = [[] for _ in range(max_tracks)]
x_temp = np.zeros(max_tracks)
y_temp = np.zeros(max_tracks)
inactive_k = np.ones(max_tracks, dtype=np.int32)
del_ = np.zeros(max_tracks, dtype=np.int32)
steps = np.zeros(max_tracks, dtype=np.int32)
valid = np.ones(max_tracks, dtype=np.int32)
inactive = np.ones(max_tracks, dtype=np.int32)
L_before = np.zeros(max_tracks, dtype=np.int32)
num_involved = np.zeros(max_tracks, dtype=np.int32)
if lightweight_mode:
    x_m = [np.array([]) for _ in range(max_tracks)]
    y_m = [np.array([]) for _ in range(max_tracks)]
    t_m = [np.array([]) for _ in range(max_tracks)]
else:
    x_m = np.zeros(max_tracks)
    y_m = np.zeros(max_tracks)
    t_m = np.zeros(max_tracks)
 
T_min = np.min(T_global)
T_max = np.max(T_global)
# optional start time (the test run detects in the frame shown in the GUI)
if globals().get("t_start", None) is not None:
    T_min = max(T_min, float(t_start))


def _is_time_sorted(T):
    return T.size < 2 or bool(np.all(T[1:] >= T[:-1]))


def _time_window(t_lo, t_hi):
    """Events with t_lo <= T_global <= t_hi as (X + 1, Y + 1, T copy, global indices).

    The event stream is time sorted, so the window is found by binary search
    (O(log n)) instead of comparing every event of the recording at every step.
    """
    if _T_sorted:
        lo = int(np.searchsorted(T_global, t_lo, side="left"))
        hi = max(int(np.searchsorted(T_global, t_hi, side="right")), lo)
        return X_global[lo:hi] + 1, Y_global[lo:hi] + 1, T_global[lo:hi].copy(), np.arange(lo, hi)
    mask = np.logical_and(T_global >= t_lo, T_global <= t_hi)
    return X_global[mask] + 1, Y_global[mask] + 1, T_global[mask], np.where(mask)[0]


def _group_by_label(labels):
    """(label, ascending positions) for every non-zero label, labels in ascending order."""
    sel = np.flatnonzero(labels)
    if sel.size == 0:
        return []
    order = sel[np.argsort(labels[sel], kind="stable")]
    labs = labels[order]
    cuts = np.flatnonzero(labs[1:] != labs[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [order.size]))
    return [(labs[a], order[a:b]) for a, b in zip(starts, ends)]


def _dilate(mask):
    # a 1x1 structuring element (the default mask size) leaves the mask unchanged
    if structure.size == 1:
        return mask.copy()
    return binary_dilation(mask, structure=structure, iterations=1)


def _clear_label(labelmap, i_best):
    # same as labelmap[labelmap == i_best] = -1: the label was only written on active_pixels[i_best]
    reg = active_pixels[i_best]
    if len(reg) == 0:
        return
    rx = reg[:, 0]
    ry = reg[:, 1]
    hit = labelmap[rx, ry] == i_best
    labelmap[rx[hit], ry[hit]] = -1

# compiled association for the pixelwise extension (utils/pixelwise_numba.py, same results as the Python
# branch below, which still handles the other modes; numba_matching=False forces it)
_fast_pixelwise = bool(pixelwise_extension and not pseudo_images and not lightweight_mode
                       and not (max_tracks > 0 and save_mode == False)
                       and globals().get("numba_matching", True))
_tracker = None
_pending = collections.deque()  # steps being prepared (in order) while the current one is matched
_next_k = 1  # next step to submit
if _fast_pixelwise:
    _n_prep = 3  # preparing threads: labelling + grouping a step take longer than matching it
    _pool = ThreadPoolExecutor(max_workers=_n_prep)
    _tls = threading.local()  # one label_events scratch map per preparing thread
    _label_offs = dilation_offsets(structure)  # pixels the dilation sets around one pixel ([[0, 0]] for 1x1)


def _prepare_fast_step(k):
    """Events (X, Y, T, ind_global), 0-based pixels and the events grouped by kept region for step k > 0 of the
    compiled branch. Runs in a worker thread one step ahead: the compiled labelling (dilation included, same
    labels as scipy's binary_dilation + label) and grouping release the GIL, so they overlap the matching of
    step k - 1."""
    X, Y, T, ind_global = _time_window(T_min + (k-overlap) * dt, T_min + (k+1) * dt)
    T[T<T_min+k*dt] = -1
    xs = (X - 1).astype(np.int32)
    ys = (Y - 1).astype(np.int32)
    pixmap = getattr(_tls, "pixmap", None)
    if pixmap is None:
        pixmap = _tls.pixmap = np.zeros((height, width), dtype=np.int32)
    labels_xy, num_regions = label_events(xs, ys, pixmap, N_pixelwise, _label_offs)
    order, off = _group(labels_xy, int(num_regions))
    return X, Y, T, ind_global, xs, ys, order, off


tic_start_ges = time.time()
#%% main loop
for multi in range(multiN):
    _T_sorted = _is_time_sorted(T_global)
    _next_k = 1
    for k in range(N):
        tic_start = time.time()
        _fast_step = _fast_pixelwise and k > 0
        if not _fast_step:  # (the compiled branch gets its events from _prepare_fast_step)
            if pixelwise_extension:
                if k>0:
                    X, Y, T, ind_global = _time_window(T_min + (k-overlap) * dt, T_min + (k+1) * dt)
                else:
                    X, Y, T, ind_global = _time_window(T_min + k * dt, T_min + (k+1) * dt)
            else:
                X, Y, T, ind_global = _time_window(T_min + k * dt, T_min + (k+1) * dt)
            if pixelwise_extension:
                T[T<T_min+k*dt] = -1
            if pseudo_images:
                X_flat = X.ravel()
                Y_flat = Y.ravel()
                T_flat = T.ravel()

        if _fast_step:
            while len(_pending) <= _n_prep and _next_k < N:  # keep the preparing threads busy, in step order
                _pending.append(_pool.submit(_prepare_fast_step, _next_k))
                _next_k += 1
            X, Y, T, ind_global, xs, ys, _order, _off = _pending.popleft().result()
            if _tracker is None:  # takes over the clusters started at step 0
                _tracker = PixelwiseTracker(height, width, structure, multi == 1, buffer, len(X_global),
                                            steps, inactive, inactive_k, valid, x_m, y_m, t_m,
                                            active_pixels, ind_clust, min_steps=min_steps)
            _tracker.step(k, xs, ys, _order, _off, X, Y, T, ind_global)
        elif k == 0:
            if pseudo_images:
                centroids, coords = analyze_pseudo_image(X,Y,height,width,area,filtersize)            
                lookup = defaultdict(list)
                for i, (x, y) in enumerate(zip(X.ravel(), Y.ravel())):
                    lookup[(x, y)].append(i)
                X_r, Y_r, T_r = X.ravel(), Y.ravel(), T.ravel()
                for i, arr in enumerate(coords):
                    for x, y in arr:
                        if (x, y) in lookup:
                            # ind_global is unique: X[ind_global == idx] is the single element j
                            for j in lookup[(x+1, y+1)]:
                                ind_clust[i].append(ind_global[j])
                                x_clust[i].append(X_r[j:j+1])
                                y_clust[i].append(Y_r[j:j+1])
                                t_clust[i].append(T_r[j:j+1])
                ind_clust = [np.array(a, dtype=int) if len(a) > 0 else np.array([]) for a in ind_clust]
                x_clust   = [np.array(a) if len(a) > 0 else np.array([]) for a in x_clust]
                y_clust   = [np.array(a) if len(a) > 0 else np.array([]) for a in y_clust]
                
                for i in range(len(centroids)):
                    x_m[i] = np.append(x_m[i],centroids[i,0])
                    y_m[i] = np.append(y_m[i],centroids[i,1])
                    t_m[i] = np.append(t_m[i],(T_min+dt/2+(k)*dt)*1e-6)
                    del_[i] = 0
                    inactive[i] = 0
                    steps[i] = 1
                    num_involved[i] = num_points
                    
                valid = np.ones(len(x_clust))
                
            elif pixelwise_extension:
                structure = np.ones((structure_size, structure_size), dtype=bool)    
  
                XY = np.column_stack((X-1,Y-1))
                new_mask = np.zeros((height, width), dtype=bool)
                new_mask[X-1,Y-1] = True
                new_mask = _dilate(new_mask)
                labeled, num_regions = label(new_mask)
                
                region_sizes = np.bincount(labeled.ravel())
                keep = region_sizes >= N_pixelwise
                labeled[~keep[labeled]] = 0
                # every event gets the label of its (kept) region; same int16 storage as before
                idx = labeled[XY[:, 0], XY[:, 1]].astype(np.int16)

                idx[idx==0]=-1        
                    
                if max_tracks > 0 and save_mode==False:
                    idx[idx > max_tracks] = -1   
                    
                valid_clusters = idx != -1
                X = X[valid_clusters]
                Y = Y[valid_clusters]
                T = T[valid_clusters]
                ind_global = ind_global[valid_clusters]
                idx = idx[valid_clusters]
        
                max_cluster = idx.max() if len(idx) > 0 else -1

                # events of each cluster in their original order (stable sort), clusters ascending
                for i, mask in _group_by_label(np.where(idx > 0, idx, 0)):
                    x_clust[i] = X[mask]
                    y_clust[i] = Y[mask]
                    t_clust[i] = T[mask] 
                    ind_clust[i] = ind_global[mask]
                    active_pixels[i] = np.unique(np.column_stack((x_clust[i]-1,y_clust[i]-1)),axis=0)
                    if lightweight_mode:
                        t_min, t_max = t_clust[i].min(), t_clust[i].max()
                        bins = np.linspace(t_min, t_max, num=num_points+1)
                        for j in range(num_points):
                            mask = np.where((t_clust[i]>=bins[j]) & (t_clust[i]<=bins[j+1]))
                            if mask[0].size > 0:
                                x_m[i] = np.append(x_m[i], fast_median(x_clust[i][mask]))
                                y_m[i] = np.append(y_m[i], fast_median(y_clust[i][mask]))
                                t_m[i] = np.append(t_m[i], fast_median(t_clust[i][mask])* 1e-6)
                                num_involved[i] +=1
                    else:
                        x_m[i] = fast_median(x_clust[i])
                        y_m[i] = fast_median(y_clust[i])
                        t_m[i] = fast_median(t_clust[i])* 1e-6
        
                    del_[i] = 0
                    inactive[i] = 0
                    steps[i] = 1
                    
                valid = np.ones(len(x_clust))
            else:
                idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                if max_tracks > 0 and save_mode==False:
                    idx[idx > max_tracks] = -1  
        
                valid_clusters = idx != -1
                X = X[valid_clusters]
                Y = Y[valid_clusters]
                T = T[valid_clusters]
                ind_global = ind_global[valid_clusters]
                idx = idx[valid_clusters]
        
                max_cluster = idx.max() if len(idx) > 0 else -1
                
                for i in range(max_cluster):
                    mask = idx == i
                    x_clust[i] = X[mask]
                    y_clust[i] = Y[mask]
                    t_clust[i] = T[mask] 
                    ind_clust[i] = ind_global[mask]
                    if lightweight_mode:
                        t_min, t_max = t_clust[i].min(), t_clust[i].max()
                        bins = np.linspace(t_min, t_max, num=num_points+1)
                        for j in range(num_points):
                            mask = np.where((t_clust[i]>=bins[j]) & (t_clust[i]<=bins[j+1]))
                            if mask[0].size > 0:
                                x_m[i] = np.append(x_m[i], fast_median(x_clust[i][mask]))
                                y_m[i] = np.append(y_m[i], fast_median(y_clust[i][mask]))
                                t_m[i] = np.append(t_m[i], fast_median(t_clust[i][mask])* 1e-6)
                                num_involved[i] +=1
                    else:
                        x_m[i] = fast_median(x_clust[i])
                        y_m[i] = fast_median(y_clust[i])
                        t_m[i] = fast_median(t_clust[i])* 1e-6
        
                    del_[i] = 0
                    inactive[i] = 0
                    steps[i] = 1
                    
                valid = np.ones(len(x_clust))
        else:
            if pseudo_images:
                centroids, coords = analyze_pseudo_image(X,Y,height,width,area,filtersize)
                idx = []
                points_a = []
                for i in range(len(x_m)):
                    if len(x_m[i])<1 or valid[i]==0:
                        continue
                    if t_m[i][-1]==(T_min+dt/2+(k-1)*dt)*1e-6:
                        idx = np.append(idx,i)
                        points_a.append([x_m[i][-1], y_m[i][-1]])
                points_a = np.array(points_a)
                row_ind, col_ind, distances = match_nearest_neighbors_partial_threshold(points_a, centroids,threshold)
                matched_b = set(col_ind)
                
                valid_indices = [i for i in range(len(x_m)) if len(x_m[i]) > 0]
                valid_set = set(valid_indices)- set(idx[row_ind])
                for i in valid_set:
                    valid[i] = 0
                    
                lookup = defaultdict(list)
                for i, (x, y) in enumerate(zip(X_flat, Y_flat)):
                    lookup[(x, y)].append(i)
            
                for i in range(len(col_ind)):
                    track_id = int(idx[row_ind[i]])
                    x_m[track_id] = np.append(x_m[track_id], centroids[col_ind[i]][0])
                    y_m[track_id] = np.append(y_m[track_id], centroids[col_ind[i]][1])
                    t_m[track_id] = np.append(t_m[track_id], (T_min+dt/2+(k)*dt)*1e-6)
                    steps[track_id] +=1
                    inactive_k[track_id] =k
                    num_involved[track_id] = num_points
                    
                    tmp_ind = []
                    tmp_x   = []
                    tmp_y   = []
                    tmp_t   = []
                    for x, y in coords[col_ind[i]]:
                        for idxx in lookup.get((x+1, y+1), []):
                            tmp_ind.append(ind_global[idxx])
                            tmp_x.append(X_flat[idxx])
                            tmp_y.append(Y_flat[idxx])
                            tmp_t.append(T_flat[idxx])
                    ind_clust[track_id] = np.append(ind_clust[track_id],np.array(tmp_ind))
                    x_clust[track_id]   = np.append(x_clust[track_id], np.array(tmp_x))
                    y_clust[track_id]   = np.append(y_clust[track_id], np.array(tmp_y))
                    t_clust[track_id]   = np.append(t_clust[track_id], np.array(tmp_t))
                
            elif pixelwise_extension:
                XY = np.column_stack((X - 1, Y - 1)).astype(np.int32)
                orig_mask = np.zeros((height, width), dtype=bool)
                orig_mask[XY[:, 0], XY[:, 1]] = True
                
                dilated_mask = _dilate(orig_mask)
                labeled, num_regions = label(dilated_mask)
                labels_xy = labeled[XY[:, 0], XY[:, 1]]
                region_sizes = np.bincount(labels_xy, minlength=labeled.max() + 1)
                
                keep = region_sizes >= N_pixelwise
                labels_xy[~keep[labels_xy]] = 0
                
                new_labels = np.unique(labels_xy)
                new_labels = new_labels[new_labels != 0]
                
                # event coordinates / indices of every region, in event order
                region_coords = defaultdict(list)
                region_indices = defaultdict(list)
                for label_0, members in _group_by_label(labels_xy):
                    region_coords[label_0] = XY[members].astype(np.uint16)
                    region_indices[label_0] = members.astype(np.uint32)

                active_labelmap = np.zeros_like(orig_mask, dtype=np.int32)
                active_labelmap[:, :] = -1

                # only the valid clusters (the others are skipped anyway)
                for i in np.flatnonzero(np.asarray(valid[:len(active_pixels)]) != 0):
                    region = active_pixels[i]
                    if len(region) == 0:
                        continue
                    active_labelmap[region[:, 0], region[:, 1]] = i
                    
                active_temp = defaultdict(list)  # only matched clusters get entries
                ind_temp = []
                
                assigned_region_labels = set()
                    
                for label_0 in new_labels:
                    coords = region_coords[label_0]
                    if coords.shape[0] == 0:
                        continue    

                    if multi==1:
                        max_x, max_y = active_labelmap.shape
                        # max_x = 1280
                        # max_y = 720
                        
                        coords_i = coords.astype(np.int32, copy=False)  
                        
                        min_x = max(int(coords_i[:, 0].min()) - 1, 0)
                        max_x_roi = min(int(coords_i[:, 0].max()) + 2, max_x)
                        min_y = max(int(coords_i[:, 1].min()) - 1, 0)
                        max_y_roi = min(int(coords_i[:, 1].max()) + 2, max_y)

                        if max_x_roi <= min_x or max_y_roi <= min_y:
                            continue
                        
                        mask = np.zeros((max_x_roi - min_x, max_y_roi - min_y), dtype=bool)
                        
                        coords_roi = coords_i.copy()
                        coords_roi[:, 0] -= min_x
                        coords_roi[:, 1] -= min_y
                        
                        mask[coords_roi[:, 0], coords_roi[:, 1]] = True
                        mask_dilated = binary_dilation(mask, structure=structure, iterations=1)
                        # mask_dilated = mask
                        
                        coords_exp = np.argwhere(mask_dilated).astype(np.int32)
                        coords_exp[:, 0] += min_x
                        coords_exp[:, 1] += min_y
                        
                        i_labels = active_labelmap[coords_exp[:, 0], coords_exp[:, 1]]
                    else:
                        i_labels = active_labelmap[coords[:, 0], coords[:, 1]]
                    i_labels = i_labels[i_labels >= 0] 
                    
                    if len(i_labels) == 0:
                        continue
                    
                    unique_clusters, counts = np.unique(i_labels, return_counts=True)
                    
                    if len(unique_clusters) == 1:
                        i_best = unique_clusters[0]
                        xx = region_coords[label_0]
                        idx = np.array(region_indices[label_0])
                    
                        x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                            x_m, y_m, t_m, i_best, X, Y, T, x_clust, y_clust, t_clust, idx,
                            del_, steps, buffer, steps_rev, inactive_k, k, lightweight_mode, num_points, ind_clust, ind_global
                        )
                    
                        mask = t_clust[i_best] >= 0
                        x_clust[i_best] = x_clust[i_best][mask]
                        y_clust[i_best] = y_clust[i_best][mask]
                        t_clust[i_best] = t_clust[i_best][mask]
                        ind_clust[i_best] = ind_clust[i_best][mask]
                        
                        _clear_label(active_labelmap, i_best)

                        active_temp[i_best].append(xx)
                        ind_temp.append(i_best)
                        labeled[xx[:, 0], xx[:, 1]] = 0
                        
                        assigned_region_labels.add(label_0)
                    else:
                        cluster_centers = {
                                i: np.mean(active_pixels[i], axis=0)
                                for i in unique_clusters
                            }
                        
                        keys = list(cluster_centers.keys())
                        centers = np.array(list(cluster_centers.values()))
                        dist_matrix = squareform(pdist(centers))
                        for i in range(len(keys)):
                            for j in range(len(keys)):
                                if i != j and dist_matrix[i, j] < 4:
                                    centers[j] = np.array([-1000, -1000])
                        cluster_centers = dict(zip(keys, centers))
                        
                        assigned_labels = []
                        for x_var, y_var in coords:
                            dists = {
                                i: (x_var - cx)**2 + (y_var - cy)**2
                                for i, (cx, cy) in cluster_centers.items()
                            }
                            assigned_label = unique_clusters[np.argmin(list(dists.values()))]
                            assigned_labels.append((x_var,y_var, assigned_label))
                            
                        label_counts = collections.Counter([label for (_, _, label) in assigned_labels])
                        assigned_labels = [
                            (x, y, label if label_counts[label] >= 4 else 0)
                            for (x, y, label) in assigned_labels
                        ]
                        unique_labels = list(set(label for (_, _, label) in assigned_labels))
                            
                        if len(unique_labels)>1:
                            k
                        for assignit in range(len(unique_labels)):
                            if unique_labels[assignit]==0:
                                continue
                            xx = region_coords[label_0]
                            idx = np.array(region_indices[label_0])
                            indices = [i for i, (_, _, label) in enumerate(assigned_labels) if label == unique_labels[assignit]]
                            
                            xx = xx[indices]
                            idx = idx[indices]
                            i_best = unique_labels[assignit]
                            x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                                x_m, y_m, t_m, i_best, X, Y, T, x_clust, y_clust, t_clust, idx,
                                del_, steps, buffer, steps_rev, inactive_k, k, lightweight_mode, num_points, ind_clust, ind_global
                            )
                            
                            mask = t_clust[i_best] >= 0
                            x_clust[i_best] = x_clust[i_best][mask]
                            y_clust[i_best] = y_clust[i_best][mask]
                            t_clust[i_best] = t_clust[i_best][mask]
                            ind_clust[i_best] = ind_clust[i_best][mask]
                            
                            _clear_label(active_labelmap, i_best)
                            active_temp[i_best].append(xx)
                            ind_temp.append(i_best)
                            labeled[xx[:, 0], xx[:, 1]] = 0
                        assigned_region_labels.add(label_0)
                        
                ind_temp = np.unique(ind_temp)
                cand = np.flatnonzero(np.asarray(valid[:len(active_pixels)]) != 0)
                matched = np.isin(cand, ind_temp)
                for ii in cand[matched]:
                    if len(active_temp[ii])>1:
                        active_pixels[ii] = np.concatenate(active_temp[ii], axis=0)
                    else:
                        active_pixels[ii] = active_temp[ii][0]
                valid[cand[~matched]] = 0
                inactive_k[cand[~matched]] = k
                    
            elif db_clustering:
                if len(X)<1:
                    idx = []
                else:
                    idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                
                    valid_clusters = idx != -1
                    X = X[valid_clusters]
                    Y = Y[valid_clusters]
                    T = T[valid_clusters]
                    ind_global = ind_global[valid_clusters]
                    idx = idx[valid_clusters]
    
                max_cluster = idx.max() if len(idx) > 0 else -1
                for i in range(max_cluster):
                    mask = idx == i
                    x_temp[i] = fast_median(X[mask])
                    y_temp[i] = fast_median(Y[mask])
                    
                nearest_indices, x_clust, y_clust, x_m, y_m, t_m, steps, inactive_k, valid, t_clust, num_involved, ind_clust = db_matching(
                    valid,x_m,y_m,t_m,x_temp,y_temp,max_tracks,threshold,idx,del_,steps,buffer,steps_rev,X,Y,T,x_clust,y_clust,t_clust,k,inactive_k,lightweight_mode,num_points,num_involved,ind_clust, ind_global)
                
                x_temp = np.zeros(max_tracks)
                y_temp = np.zeros(max_tracks)
    
            else:  
                
                x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, valid, X, Y, T, num_involved,ind_clust,ind_global = kdtree_clustering(
                    X,Y,T,valid,x_m,y_m,t_m,Range,Lmin,steps,del_,buffer,steps_rev,inactive_k,k,x_clust,y_clust,t_clust,lightweight_mode,num_points,num_involved,ind_clust,ind_global)
            
                        
            if max_tracks > 0 and save_mode==False:
                if X.size > 0 and np.sum(inactive) > 0:
                    idx = subgrid_search(height,width,search_factor,minPts,inactive,X,Y)
    
                    if np.any(idx):
                        X = X[idx]
                        Y = Y[idx]
                        T = T[idx]
                        idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                        idx[idx > np.sum(inactive)] = -1
                    else:
                        idx = np.array([])
                else:
                    idx = np.array([])
            else:
                if pseudo_images:
                    unmatched_b = set(range(len(centroids))) - matched_b
                    idx = []
                    for idx_unmatched in unmatched_b:
                        idx.append(idx_unmatched)
                    idx = np.array(idx)
                elif pixelwise_extension==True:
                    remaining_labels = [
                        region_id
                        for region_id in region_indices.keys()
                        if region_id not in assigned_region_labels
                    ]
                    
                    idx = np.full(len(X), -1, dtype=np.int32)
                    
                    m = 0
                    
                    for region_id in remaining_labels:
                        all_indices = region_indices[region_id]
                    
                        if len(all_indices) == 0:
                            continue
                    
                        if np.count_nonzero(T[all_indices] > 0) > 5:
                            idx[all_indices] = region_id
                            m += 1
                    
                    if m == 0:
                        idx = np.array([], dtype=np.int32)
                    # if np.max(labeled) > 0:
                    #     new_labels = np.unique(labeled)
                    #     new_labels = new_labels[new_labels != 0]
                    #     idx = np.full(len(X), -1)

                    #     m=0
                    #     for region_id in new_labels:
                    #         all_indices = []
                    #         for coord in region_coords[region_id]:
                    #             all_indices.extend(pixel_to_indices.get(tuple(coord), []))
                    #         if np.count_nonzero(T[all_indices] > 0) > 5:
                    #             idx[all_indices] = region_id
                    #             m+=1
                    #     if m==0:
                    #         idx = np.array([])
                    # else:
                    #     idx = np.array([])
    
                elif X.size > 0 and db_clustering==False:
                    idx = clustering(apply_hdbscan=apply_hdbscan,epsilon=epsilon,minPts=minPts,X=X,Y=Y)
                elif X.size > 0 and db_clustering==True:
                    unmatched_indices = np.where(nearest_indices == -1)[0]
                    idx[~np.isin(idx, unmatched_indices)] = -1
                else:
                    idx = np.array([])
    
            if idx.size > 0 and pseudo_images==False:
                valid_idx = idx != -1
                X = X[valid_idx]
                Y = Y[valid_idx]
                T = T[valid_idx]
                ind_global = ind_global[valid_idx]
                idx = idx[valid_idx]
                
    
            if idx.size > 0:
                refil = np.where(inactive == 1)[0]
                if pseudo_images:
                    needed = len(idx)
                else:
                    needed = max(idx)
                missing = needed - len(refil)
                
                if missing > 0:
                    extend_size = missing
                    # new slots start empty (they are filled below); np.full((max_length,), nan)
                    # placeholders cost 4 MB per slot and stayed in the output as all-NaN clusters
                    x_clust = extend_list_of_arrays(x_clust, extend_size, (0,))
                    y_clust = extend_list_of_arrays(y_clust, extend_size, (0,))
                    t_clust = extend_list_of_arrays(t_clust, extend_size, (0,))
                    ind_clust = extend_list_of_arrays(ind_clust, extend_size, (0,))
                    active_pixels = extend_list_of_arrays(active_pixels, extend_size, (0,))
                    # (P_new / P_newz only exist in the tracking script: extending them raised NameError)
                    x_est = extend_list_of_arrays(x_est, extend_size, (3,))
                    y_est = extend_list_of_arrays(y_est, extend_size, (3,))
                
                    if lightweight_mode or pseudo_images:
                        x_m = extend_list_of_arrays(x_m, extend_size, (0,))
                        y_m = extend_list_of_arrays(y_m, extend_size, (0,))
                        t_m = extend_list_of_arrays(t_m, extend_size, (0,))
                    else:
                        x_m = extend_1d(x_m, extend_size)
                        y_m = extend_1d(y_m, extend_size)
                        t_m = extend_1d(t_m, extend_size)
                    valid = extend_1d(valid, extend_size, 0)
                    steps = extend_1d(steps, extend_size, 0)
                    num_involved = extend_1d(num_involved, extend_size, 0)
                    L_before = extend_1d(L_before, extend_size, 0)
                    del_ = extend_1d(del_, extend_size, 0)
                    inactive = extend_1d(inactive, extend_size, 1)
                    inactive_k = extend_1d(inactive_k, extend_size, -1)
                    max_tracks = max_tracks+int(extend_size)

                    refil = np.where(inactive == 1)[0]
                
                m=0
                if pseudo_images:
                    lookup = defaultdict(list)
                    for j, (x, y) in enumerate(zip(X.ravel(), Y.ravel())):
                        lookup[(x, y)].append(j)
                if not pseudo_images:
                    # event positions of every cluster label (ascending, like idx == i)
                    label_groups = dict(_group_by_label(np.where(idx > 0, idx, 0)))
                for i in np.unique(idx):
                    if i < 1:   # labels start at 1 here, as before
                        continue
                    if pseudo_images:
                        clust_idx = i
                    else:
                        clust_idx = label_groups[i]
                    r = refil[m]
                    if pseudo_images:
                        tmp_ind = []
                        tmp_x   = []
                        tmp_y   = []
                        tmp_t   = []
                        for x, y in coords[clust_idx]:
                            for idxx in lookup.get((x+1, y+1), []):
                                tmp_ind.append(ind_global[idxx])
                                tmp_x.append(X_flat[idxx])
                                tmp_y.append(Y_flat[idxx])
                                tmp_t.append(T_flat[idxx])
                        ind_clust[r] = np.array(tmp_ind)
                        x_clust[r]   = np.array(tmp_x)
                        y_clust[r]   = np.array(tmp_y)
                        t_clust[r]   = np.array(tmp_t)
                    else:
                        x_clust[r] = X[clust_idx]
                        y_clust[r] = Y[clust_idx]
                        t_clust[r] = T[clust_idx]
                        ind_clust[r] = ind_global[clust_idx]
                    if pseudo_images:
                        x_m[r]=np.array([centroids[clust_idx][0]])
                        y_m[r]=np.array([centroids[clust_idx][1]])
                        t_m[r]=np.array([(T_min+dt/2+(k)*dt)*1e-6])
                        num_involved[r] = len(x_m[r])
                    elif lightweight_mode:
                        t_min, t_max = t_clust[r].min(), t_clust[r].max()
                        bins = np.linspace(t_min, t_max, num=num_points+1)
                        x_m[r]=[]
                        y_m[r]=[]
                        t_m[r]=[]
                        for j in range(num_points):
                            mask = np.where((t_clust[r]>=bins[j]) & (t_clust[r]<=bins[j+1]))
                            if mask[0].size > 0:
                                x_m[r] = np.append(x_m[r], fast_median(x_clust[r][mask]))
                                y_m[r] = np.append(y_m[r], fast_median(y_clust[r][mask]))
                                t_m[r] = np.append(t_m[r], fast_median(t_clust[r][mask])* 1e-6)
                        num_involved[r] = len(x_m[r])
                    elif pixelwise_extension:
                        mask = t_clust[r] >= 0
                        x_clust[r] = x_clust[r][mask]
                        y_clust[r] = y_clust[r][mask]
                        t_clust[r] = t_clust[r][mask]
                        ind_clust[r] = ind_clust[r][mask]
                        active_pixels[r] = region_coords[np.int32(i)]
                        x_m[r] = fast_median(x_clust[r])
                        y_m[r] = fast_median(y_clust[r])
                        t_m[r] = fast_median(t_clust[r])* 1e-6
                    else:
                        x_m[r] = fast_median(x_clust[r])
                        y_m[r] = fast_median(y_clust[r])
                        t_m[r] = fast_median(t_clust[r])* 1e-6
                    valid[r] = 1
                    steps[r] = 1
                    del_[r] = 0
                    inactive[r] = 0
                    inactive_k[r] = k
                    m+=1        
    
            # (the former L_before bookkeeping here was never read)
            _n = int(max_tracks)
            inactive[:_n][(steps[:_n] < buffer) & (steps[:_n] < min_steps) & ((inactive_k[:_n] + 2) < k)] = 1

        del_[:] = 1
        
        tic_end = time.time()
        if k % 10 == 0:
            print(f"Loop {k} completed in {tic_end - tic_start} seconds")
        if stop_event is not None and stop_event.is_set():
            break

    while _pending:  # steps prepared before a stop request
        _pending.popleft().result()
    if _fast_pixelwise:
        # the compiled branch keeps the clusters as PackedTracks (utils/packed_tracks.py): one flat array per
        # quantity instead of lists of millions of small arrays; same values, track by track
        if _tracker is not None:
            _out = _tracker.export(X_global, Y_global, T_global)
            x_clust, y_clust, t_clust, ind_clust = _out["x_clust"], _out["y_clust"], _out["t_clust"], _out["ind_clust"]
            steps, x_m, y_m, t_m, max_tracks = _out["steps"], _out["x_m"], _out["y_m"], _out["t_m"], _out["max_tracks"]
            _tracker = _out = None
        else:  # no step after k = 0
            x_clust, y_clust, t_clust, ind_clust = (PackedTracks.from_list(c) for c in (x_clust, y_clust, t_clust, ind_clust))

    if lightweight_mode==False:
        # for i in range(3):
            x_clust, y_clust, t_clust, steps, ind_clust = repair_tracks(x_clust, y_clust, t_clust, ind_clust, steps,dt)
    else:
        # for i in range(3):
            x_clust, y_clust, t_clust, steps, ind_clust, x_m, y_m, t_m = repair_tracks_pseudoframes(x_clust, y_clust, t_clust, ind_clust, steps, dt,x_m,y_m,t_m)


    if multi==1:
        n_events = [] if isinstance(ind_clust, PackedTracks) else [len(a) for a in ind_clust]
        if lightweight_mode or pseudo_images:
            for i in range(len(ind_clust)):
                if n_events[i]<min_events:
                    x_clust[i] = np.array([], dtype=int)
                    y_clust[i] = np.array([], dtype=int)
                    t_clust[i] = np.array([], dtype=int)
                    ind_clust[i] = np.array([], dtype=int)
                    x_m[i] = np.array([])
                    y_m[i] = np.array([])
                    t_m[i] = np.array([])
        elif isinstance(ind_clust, PackedTracks):
            _keep = ind_clust.lengths() >= min_events
            x_clust, y_clust, t_clust, ind_clust = (c.keep(_keep) for c in (x_clust, y_clust, t_clust, ind_clust))
        else:
            for i in range(len(ind_clust)):
                if n_events[i]<min_events:
                    x_clust[i] = np.array([], dtype=int)
                    y_clust[i] = np.array([], dtype=int)
                    t_clust[i] = np.array([], dtype=int)
                    ind_clust[i] = np.array([], dtype=int)

    if multi==0:
        if lightweight_mode == False:
            if isinstance(x_clust, PackedTracks):
                if not test_mode:
                    _keep = ~((steps[:len(x_clust)] < min_steps) | (x_clust.lengths() < min_events))
                    x_clust, y_clust, t_clust, ind_clust = (c.keep(_keep) for c in (x_clust, y_clust, t_clust, ind_clust))
            else:
                n_events = [len(a) for a in x_clust]
                if not test_mode:
                    for i in range(len(ind_clust)):
                        if steps[i] < min_steps or n_events[i]<min_events:
                            x_clust[i] = np.array([], dtype=int)
                            y_clust[i] = np.array([], dtype=int)
                            t_clust[i] = np.array([], dtype=int)
                            ind_clust[i] = np.array([], dtype=int)
            if multi < multiN - 1:  # the next pass gets the events left (X_global is not used after the last)
                mask = np.ones(len(X_global), dtype=bool)
                if isinstance(ind_clust, PackedTracks):
                    ind_clust.clear_in_mask(mask)
                else:
                    ind_all = np.concatenate(ind_clust)
                    ind_all = ind_all.astype(int)
                    mask[ind_all] = False
                X_global = X_global[mask]
                Y_global = Y_global[mask]
                T_global = T_global[mask]
            
            # the lists are handed over as they are (they were copied array by array: same values, but the
            # whole detection result twice in memory); x_clust & co. get new lists just below
            x_clust_fast = x_clust
            y_clust_fast = y_clust
            t_clust_fast = t_clust
            ind_clust_fast = ind_clust
            
            if multirun:
                dt = multitimefactor*dt
                N = int(N//multitimefactor)
                minPts = minPts*multitimefactor
            
            # one shared empty array per list: a size-0 array cannot be changed in place, slots are reassigned
            x_clust = [np.array([])] * max_tracks
            y_clust = [np.array([])] * max_tracks
            t_clust = [np.array([])] * max_tracks
            ind_clust = [np.array([])] * max_tracks
            active_pixels = [[] for _ in range(max_tracks)]
            x_est = [[] for _ in range(max_tracks)]
            y_est = [[] for _ in range(max_tracks)]
            x_temp = np.zeros(max_tracks)
            y_temp = np.zeros(max_tracks)
            inactive_k = np.ones(max_tracks, dtype=np.int32)
            del_ = np.zeros(max_tracks, dtype=np.int32)
            steps = np.zeros(max_tracks, dtype=np.int32)
            valid = np.ones(max_tracks, dtype=np.int32)
            inactive = np.ones(max_tracks, dtype=np.int32)
            L_before = np.zeros(max_tracks, dtype=np.int32)
            num_involved = np.zeros(max_tracks, dtype=np.int32)
            x_m = np.zeros(max_tracks)
            y_m = np.zeros(max_tracks)
            t_m = np.zeros(max_tracks)
        else:
            n_events = [len(a) for a in ind_clust]
            if not test_mode:
                for i in range(len(ind_clust)):
                    if len(x_m[i]) < min_steps or n_events[i]<2 * min_events:  # (twice the minimum, as before)
                        x_clust[i] = np.array([], dtype=int)
                        y_clust[i] = np.array([], dtype=int)
                        t_clust[i] = np.array([], dtype=int)
                        ind_clust[i] = np.array([], dtype=int)
                        x_m[i] = np.array([])
                        y_m[i] = np.array([])
                        t_m[i] = np.array([])
            ind_all = np.concatenate(ind_clust)
            ind_all = ind_all.astype(int)
            mask = np.ones(len(X_global), dtype=bool)
            mask[ind_all] = False
            X_global = X_global[mask]
            Y_global = Y_global[mask]
            T_global = T_global[mask]
            
            x_m_fast = [arr.copy() for arr in x_m]
            y_m_fast = [arr.copy() for arr in y_m]
            t_m_fast = [arr.copy() for arr in t_m]
            x_clust_fast = [arr.copy() for arr in x_clust]
            y_clust_fast = [arr.copy() for arr in y_clust]
            t_clust_fast = [arr.copy() for arr in t_clust]
            ind_clust_fast = [arr.copy() for arr in ind_clust]
            
            if multirun:
                dt = multitimefactor*dt
                N = int(N//multitimefactor)
            
            x_clust = [[] for _ in range(max_tracks)]
            y_clust = [[] for _ in range(max_tracks)]
            t_clust = [[] for _ in range(max_tracks)]
            ind_clust = [[] for _ in range(max_tracks)]
            active_pixels = [[] for _ in range(max_tracks)]
            x_est = [[] for _ in range(max_tracks)]
            y_est = [[] for _ in range(max_tracks)]
            x_temp = np.zeros(max_tracks)
            y_temp = np.zeros(max_tracks)
            inactive_k = np.ones(max_tracks, dtype=np.int32)
            del_ = np.zeros(max_tracks, dtype=np.int32)
            steps = np.zeros(max_tracks, dtype=np.int32)
            valid = np.ones(max_tracks, dtype=np.int32)
            inactive = np.ones(max_tracks, dtype=np.int32)
            L_before = np.zeros(max_tracks, dtype=np.int32)
            num_involved = np.zeros(max_tracks, dtype=np.int32)
            x_m = [np.array([]) for _ in range(max_tracks)]
            y_m = [np.array([]) for _ in range(max_tracks)]
            t_m = [np.array([]) for _ in range(max_tracks)]

if _fast_pixelwise:
    _pool.shutdown()
else:
    x_clust = [np.asarray(a) for a in x_clust]
    y_clust = [np.asarray(a) for a in y_clust]
    t_clust = [np.asarray(a) for a in t_clust]
    ind_clust = [np.asarray(a) for a in ind_clust]

    if "x_clust_fast" in locals():
        x_clust_fast = [np.asarray(a) for a in x_clust_fast]
        y_clust_fast = [np.asarray(a) for a in y_clust_fast]
        t_clust_fast = [np.asarray(a) for a in t_clust_fast]
        ind_clust_fast = [np.asarray(a) for a in ind_clust_fast]


def _nonempty_packed(c):
    """The non-empty tracks of c (PackedTracks, or the lists reset after a pass) as PackedTracks."""
    if isinstance(c, PackedTracks):
        return c.nonempty()
    if not any(map(len, c)):  # (the reset lists: millions of empty slots)
        return PackedTracks.from_list([])
    return PackedTracks.from_list([a for a in c if len(a) > 0])


if _fast_pixelwise:
    # same as the list version below: non-empty tracks of the last pass, then those of the first pass
    x_clust, y_clust, t_clust, ind_clust = (
        PackedTracks.concat([_nonempty_packed(a), _nonempty_packed(b)])
        for a, b in ((x_clust, x_clust_fast), (y_clust, y_clust_fast),
                     (t_clust, t_clust_fast), (ind_clust, ind_clust_fast)))
    x_clust_fast = y_clust_fast = t_clust_fast = ind_clust_fast = None  # frees the flat arrays of the pass
elif lightweight_mode == False:
    x_clust = [arr for arr in x_clust if arr.size > 0] + \
             [arr for arr in x_clust_fast if arr.size > 0]
    y_clust = [arr for arr in y_clust if arr.size > 0] + \
             [arr for arr in y_clust_fast if arr.size > 0]
    t_clust = [arr for arr in t_clust if arr.size > 0] + \
             [arr for arr in t_clust_fast if arr.size > 0]
    ind_clust = [arr for arr in ind_clust if arr.size > 0] + \
             [arr for arr in ind_clust_fast if arr.size > 0]
else:
    LLmin = 0  
    x_comb = x_clust + x_clust_fast
    y_comb = y_clust + y_clust_fast
    t_comb = t_clust + t_clust_fast
    x_m_comb = x_m + x_m_fast
    y_m_comb = y_m + y_m_fast
    t_m_comb = t_m + t_m_fast
    
    x_clust, y_clust, t_clust, x_m, y_m, t_m = zip(*[
        (xc, yc, tc, xm, ym, tm)
        for xc, yc, tc, xm, ym, tm in zip(x_comb, y_comb, t_comb, x_m_comb, y_m_comb, t_m_comb)
        if len(xm) > LLmin
    ])

    x_clust, y_clust, t_clust = map(list, (x_clust, y_clust, t_clust))
    x_m, y_m, t_m = map(list, (x_m, y_m, t_m))

if multirun == True:
    dt = int(dt/multitimefactor)
if lightweight_mode==False:
    # for i in range(3):
        x_clust, y_clust, t_clust, steps, ind_clust = repair_tracks(x_clust, y_clust, t_clust, ind_clust, steps,dt)
else:
    # for i in range(3):
        x_clust, y_clust, t_clust, steps, ind_clust, x_m, y_m, t_m = repair_tracks_pseudoframes(x_clust, y_clust, t_clust, ind_clust, steps, dt,x_m,y_m,t_m)

if not test_mode:
    if isinstance(x_clust, PackedTracks):
        x_clust, y_clust, t_clust = x_clust.nonempty(), y_clust.nonempty(), t_clust.nonempty()
    elif lightweight_mode == False:
        x_clust = [arr for arr in x_clust if arr.size > 0]
        y_clust = [arr for arr in y_clust if arr.size > 0]
        t_clust = [arr for arr in t_clust if arr.size > 0]
    else:
        LLmin = 2
        x_clust, y_clust, t_clust, x_m, y_m, t_m = zip(*[
            (xc, yc, tc, xm, ym, tm)
            for xc, yc, tc, xm, ym, tm in zip(x_clust, y_clust, t_clust, x_m, y_m, t_m)
            if len(xm) > LLmin
        ])
    
        x_clust, y_clust, t_clust = map(list, (x_clust, y_clust, t_clust))
        x_m, y_m, t_m = map(list, (x_m, y_m, t_m))
        for i in range(len(x_m)):
            x_m[i] = x_m[i]+1
            y_m[i] = y_m[i]+1
            t_m[i] = t_m[i]*1e6

if pseudo_images:
    t_m = [arr[~np.isnan(arr)] for arr in t_m]
    x_m = [arr[~np.isnan(arr)] for arr in x_m]
    y_m = [arr[~np.isnan(arr)] for arr in y_m]    
