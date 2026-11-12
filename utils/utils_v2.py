import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree, cKDTree
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from contextlib import contextmanager
from joblib.parallel import BatchCompletionCallBack
from joblib import Parallel
from numba import njit, prange
try:
    from utils.packed_tracks import PackedTracks, _gather_chains
except ImportError:  # run with only utils/ on the path
    from packed_tracks import PackedTracks, _gather_chains


@njit(cache=True, nogil=True)
def _median_sorted(a):
    n = a.size
    if n == 0:
        return np.nan
    for v in a:
        if np.isnan(v):
            return np.nan
    b = np.sort(a)
    h = n // 2
    if n % 2:
        return b[h]
    return (b[h - 1] + b[h]) / 2.0


def fast_median(a):
    """np.median of a 1-D array, same value (incl. NaN propagation), without numpy's
    ~15 us per-call overhead (detection takes hundreds of thousands of small medians)."""
    a = np.asarray(a)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "iuf":
        return np.median(a)
    return float(_median_sorted(a.astype(np.float64)))
        
def compute_perpendicular_spread(events, velocity):
    if np.linalg.norm(velocity) == 0 or len(events) == 0:
        return 0.0

    v = velocity / np.linalg.norm(velocity)
    n = np.array([-v[1], v[0]])  
    events = events.T
    centered = events - np.mean(events, axis=0)
    projections = np.dot(centered, n)
    return np.std(projections)

def correct_measurement(z, velocity, events, beta=0.8):
    sigma_perp = compute_perpendicular_spread(events, velocity)
    if np.linalg.norm(velocity) == 0:
        return z  

    direction = velocity / np.linalg.norm(velocity)
    offset = beta * sigma_perp * direction
    return z - offset

def extend_1d(arr, extend_size, val=np.nan):
    return np.concatenate((arr, np.full(extend_size, val)))

def extend_list_of_arrays(lst, pad_length, array_shape):
    for _ in range(pad_length):
        lst.append(np.full(array_shape, np.nan))
    return lst
    
def process_index_afterwards(i, P, t_clust, x_clust, y_clust, 
                  C, Q, R, P_new, x_est, P_newz, y_est,
                  Kalman_v2_both_optimized, Kalman_v2_both_buffer_optimized,dtt, factor):
    if len(t_clust[i])==0:
        return i, None  
    sort_idx = np.argsort(t_clust[i])
    t_clust[i] = t_clust[i][sort_idx]
    x_clust[i] = x_clust[i][sort_idx]
    y_clust[i] = y_clust[i][sort_idx]
    
    st = np.floor(min(10,(np.max(t_clust[i])-np.min(t_clust[i]))/dtt))
    mask = t_clust[i]<np.min(t_clust[i])+st*dtt

    A = np.vstack([t_clust[i][mask]* 1e-6, np.ones_like(t_clust[i][mask])]).T
    a, b = np.linalg.lstsq(A, x_clust[i][mask], rcond=None)[0]
    px = a
    a, b = np.linalg.lstsq(A, y_clust[i][mask], rcond=None)[0]
    py = a
    xe = np.array([np.mean(x_clust[i][mask]), px, 0.0], dtype=np.float64)
    ye = np.array([np.mean(y_clust[i][mask]), py, 0.0], dtype=np.float64)
    P, xe, Pz, ye, t, x_p, y_p, xv_p, yv_p, xa_p, ya_p = Kalman_v2_both_buffer_optimized(
        t_clust[i], x_clust[i], y_clust[i], 1, C, Q, R, P*10, xe, P*10, ye, 0, dtt, factor
    )
    return i, {
        "P": P, "xe": xe, "Pz": Pz, "ye": ye,
        "t": t, "x": x_p, "y": y_p,
        "u": xv_p, "v": yv_p, "ax": xa_p, "ay": ya_p
    }

def process_index_lightweight_afterwards(i, P, t_m, x_m, y_m, 
                  C, Q, R, P_new, x_est, P_newz, y_est,
                  Kalman_v2_both_optimized, Kalman_v2_both_buffer_optimized,dtt,factor):
    if len(t_m[i])==0:
        return i, None  
    
    sort_idx = np.argsort(t_m[i])
    t_m[i] = t_m[i][sort_idx]
    x_m[i] = x_m[i][sort_idx]
    y_m[i] = y_m[i][sort_idx]
    
    st = np.floor(min(4,(np.max(t_m[i])-np.min(t_m[i]))/(dtt)))
    mask = t_m[i]<=np.min(t_m[i])+st*dtt
    
    A = np.vstack([t_m[i][mask]*1e-6, np.ones_like(t_m[i][mask])]).T
    a, b = np.linalg.lstsq(A, x_m[i][mask], rcond=None)[0]
    px = a
    a, b = np.linalg.lstsq(A, y_m[i][mask], rcond=None)[0]
    py = a
    xe = np.array([np.mean(x_m[i][mask]), px, 0.0], dtype=np.float64)
    ye = np.array([np.mean(y_m[i][mask]), py, 0.0], dtype=np.float64)
    P, xe, Pz, ye, t, x_p, y_p, xv_p, yv_p, xa_p, ya_p = Kalman_v2_both_buffer_optimized(
        t_m[i], x_m[i], y_m[i], 0, C, Q, R, P*50, xe, P*50, ye, 0, dtt,factor
    )
    return i, {
        "P": P, "xe": xe, "Pz": Pz, "ye": ye,
        "t": t, "x": x_p, "y": y_p,
        "u": xv_p, "v": yv_p, "ax": xa_p, "ay": ya_p
    }
    
def clustering(apply_hdbscan,epsilon,minPts,X,Y):
    if apply_hdbscan:
        import hdbscan  # imported on demand: the import alone takes ~3 s
        db = hdbscan.HDBSCAN(min_cluster_size=minPts, min_samples=minPts)
        idx = db.fit_predict(np.stack((X, Y), axis=-1))
    else:
        db = DBSCAN(eps=epsilon, min_samples=minPts)
        idx = db.fit_predict(np.stack((X, Y), axis=-1))
    return idx

def update_cluster(x_m,y_m,t_m,i,X,Y,T,x_clust,y_clust,t_clust,idx,del_,steps,buffer,steps_rev,inactive_k,k,lightweight_mode,num_points,ind_clust,ind_global):
        if lightweight_mode:
            t_min, t_max = T[idx].min(), T[idx].max()
            bins = np.linspace(t_min, t_max, num=num_points+1)
            for j in range(num_points):
                mask = np.where((T[idx]>=bins[j]) & (T[idx]<=bins[j+1]))
                if mask[0].size > 0:
                    x_m[i] = np.append(x_m[i], np.median(X[idx][mask]))
                    y_m[i] = np.append(y_m[i], np.median(Y[idx][mask]))
                    t_m[i] = np.append(t_m[i], np.median(T[idx][mask])* 1e-6)
        else:
            x_m[i] = fast_median(X[idx])
            y_m[i] = fast_median(Y[idx])
            t_m[i] = fast_median(T[idx]) * 1e-6

        x_clust[i] = np.concatenate([x_clust[i], X[idx]])
        y_clust[i] = np.concatenate([y_clust[i], Y[idx]])
        t_clust[i] = np.concatenate([t_clust[i], T[idx]])
        ind_clust[i] = np.concatenate([ind_clust[i], ind_global[idx]])
        steps[i] += 1
        inactive_k[i] = k
        return x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust

def db_matching(valid,x_m,y_m,t_m,x_temp,y_temp,max_tracks,threshold,idx,del_,steps,buffer,steps_rev,X,Y,T,x_clust,y_clust,t_clust,k,inactive_k,lightweight_mode,num_points,num_involved,ind_clust, ind_global):
        valid_indices = np.where(valid == 1)[0]
        pp_temp = np.full(len(valid_indices), 0, dtype=np.float64)
        pp_temp2 = np.full(len(valid_indices), 0, dtype=np.float64)

        if lightweight_mode:
            for j in range(len(valid_indices)):
                if len(x_m[valid_indices[j]])>0:
                    pp_temp[j] = np.mean(x_m[valid_indices[j]][-num_involved[valid_indices[j]]::])
                    pp_temp2[j] = np.mean(y_m[valid_indices[j]][-num_involved[valid_indices[j]]::])
            raw_points_m = np.column_stack((pp_temp, pp_temp2))
        else:   
            raw_points_m = np.column_stack((x_m[valid_indices], y_m[valid_indices]))
        nonzero_mask = ~np.all(raw_points_m == 0, axis=1)
        points_m = raw_points_m[nonzero_mask]
        original_m_indices = valid_indices[nonzero_mask]
        points_temp = np.column_stack((x_temp, y_temp))[~np.all(np.column_stack((x_temp, y_temp)) == 0, axis=1)]

        dist_matrix = cdist(points_temp, points_m)

        nearest_indices = np.full(len(points_temp), -1)
        nearest_distances = np.full(len(points_temp), np.inf)
        
        for i in range(dist_matrix.shape[0]):
            valid_indices = np.where(dist_matrix[i] < threshold)[0]
            if valid_indices.size > 0:
                min_idx = valid_indices[np.argmin(dist_matrix[i][valid_indices])]
                nearest_indices[i] = min_idx
                nearest_distances[i] = dist_matrix[i][min_idx]

        for m_index in np.unique(nearest_indices[nearest_indices != -1]):
            temp_matches = np.where(nearest_indices == m_index)[0]
            if len(temp_matches) > 1:
                best = temp_matches[np.argmin(nearest_distances[temp_matches])]
                for i in temp_matches:
                    if i != best:
                        nearest_indices[i] = -1 
                        nearest_distances[i] = np.inf

        for i in range(len(nearest_indices)):
            if nearest_indices[i]>-1:
                mask = idx == i
                if lightweight_mode:
                    l_temp = len(x_m[original_m_indices[nearest_indices[i]]])
                x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                    x_m,y_m,t_m,original_m_indices[nearest_indices[i]],X,Y,T,x_clust,y_clust,t_clust,mask,del_,steps,buffer,steps_rev,inactive_k,k,lightweight_mode,num_points,ind_clust, ind_global)
                if lightweight_mode:
                    num_involved[original_m_indices[nearest_indices[i]]] = len(x_m[original_m_indices[nearest_indices[i]]]) - l_temp  

        missing_m_indices = np.setdiff1d(np.arange(len(points_m)), nearest_indices[nearest_indices != -1])
        for i in range(len(missing_m_indices)):
            idx_real = original_m_indices[missing_m_indices[i]]
            valid[idx_real] = 0
            inactive_k[idx_real] = k
        return nearest_indices, x_clust, y_clust, x_m, y_m, t_m, steps, inactive_k, valid, t_clust, num_involved,ind_clust
    
    
def kdtree_clustering(X,Y,T,valid,x_m,y_m,t_m,Range,Lmin,steps,del_,buffer,steps_rev,inactive_k,k,x_clust,y_clust,t_clust,lightweight_mode,num_points,num_involved,ind_clust,ind_global):
        points = np.column_stack((X, Y))
        valid_indices = np.where(valid == 1)[0]
        
        tree = KDTree(points)
        all_points_to_delete = []
        for i in valid_indices:
            if lightweight_mode:
                if len(x_m[i])==0:
                    continue
                
                center = [np.mean(x_m[i][-num_involved[i]::]), np.mean(y_m[i][-num_involved[i]::])]
            else:   
                if len(x_clust[i])==0:
                    continue
                center = [x_m[i], y_m[i]]
            idx = tree.query_ball_point(center, r=Range+4)

            if len(idx) > Lmin:
                if lightweight_mode:
                    l_temp = len(x_m[i])
                x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, ind_clust = update_cluster(
                    x_m,y_m,t_m,i,X,Y,T,x_clust,y_clust,t_clust,idx,del_,steps,buffer,steps_rev,inactive_k,k,lightweight_mode,num_points, ind_clust, ind_global)
                if lightweight_mode:
                    num_involved[i] = len(x_m[i]) - l_temp  
                all_points_to_delete.extend(idx)
            else:
                valid[i] = 0
                inactive_k[i] = k

        if all_points_to_delete:
            unique_indices_to_delete = np.unique(all_points_to_delete)
            mask = np.ones(len(X), dtype=bool)
            mask[unique_indices_to_delete] = False
    
            X = X[mask]
            Y = Y[mask]
            T = T[mask]
            ind_global = ind_global[mask]
        return x_m, y_m, t_m, x_clust, y_clust, t_clust, steps, inactive_k, valid, X, Y, T, num_involved, ind_clust, ind_global
    
    
def subgrid_search(height,width,search_factor,minPts,inactive,X,Y):
    g = 0
    startx = height / 2 + (np.random.rand() - 0.5) * height / 1
    starty = width / 2 + (np.random.rand() - 0.5) * width / 1
    factor_array = 4 * (search_factor - np.arange(1, search_factor+1))+1
    threshold_sub = np.sum(inactive) * minPts * 5 

    while g < search_factor:
        range_x = height / factor_array[g]
        range_y = width / factor_array[g]

        x_min, x_max = startx - range_x, startx + range_x
        y_min, y_max = starty - range_y, starty + range_y

        idx = (X > x_min) & (X < x_max) & (Y > y_min) & (Y < y_max)

        if np.sum(idx) >= threshold_sub:
            break

        g += 1
    return idx

def add_results(result,buffer,lightweight_mode):
    if buffer:
        return result["P"], result["xe"], result["Pz"], result["ye"], result["t"].ravel(), result["x"], result["u"], result["ax"], result["y"], result["v"], result["ay"]
    else:
        if lightweight_mode:
            return result["P"], result["xe"], result["Pz"], result["ye"], result["t"], result["x"], result["u"], result["ax"], result["y"], result["v"], result["ay"]
        else:
            return result["P"], result["xe"], result["Pz"], result["ye"], result["t"][-1], result["x"][-1], result["u"][-1], result["ax"][-1], result["y"][-1], result["v"][-1], result["ay"][-1]
                          
def circular_average_filter(image, radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(float)
    kernel /= kernel.sum()  
    return convolve(image, kernel, mode='reflect')               
                
def analyze_pseudo_image(X,Y,height,width,area,filtersize):
    I = np.zeros((height, width))
    I[X-1,Y-1] = 1
    
    I = circular_average_filter(I, filtersize)
    thresh = threshold_otsu(I)
    I = I>thresh
    label_image = label(I)
    props = regionprops(label_image)
    centroids = np.array([prop.centroid for prop in props if prop.area>area])
    coords = [prop.coords for prop in props if prop.area>area]
    return centroids, coords
                
def match_nearest_neighbors_partial_threshold(points_a, points_b, threshold=10.0):
    if len(points_a)<1 or len(points_b)<1:
        return [], [], []
    cost_matrix = cdist(points_a, points_b)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_distances = cost_matrix[row_ind, col_ind]

    valid_mask = matched_distances < threshold
    row_ind_valid = row_ind[valid_mask]
    col_ind_valid = col_ind[valid_mask]
    distances_valid = matched_distances[valid_mask]
    return row_ind_valid, col_ind_valid, distances_valid                
                
def extend_length(array_list,L):
    array_list = [np.pad(arr, (0, L), mode='constant', constant_values=np.nan) for arr in array_list]
    return array_list                
                
def deriv(t_val,coefs):
    return sum(i * coefs[i] * t_val**(i - 1) for i in range(1, len(coefs)))

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = Parallel.__init__.__globals__['BatchCompletionCallBack']
    Parallel.__init__.__globals__['BatchCompletionCallBack'] = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        Parallel.__init__.__globals__['BatchCompletionCallBack'] = old_callback
        tqdm_object.close()           

@njit(cache=True, nogil=True)
def _nanmedian_where(v, t, thr, below, buf):
    """np.nanmedian(v[t < thr]) (below) or np.nanmedian(v[t > thr]); NaN when nothing is selected."""
    m = 0
    for i in range(t.size):
        if (t[i] < thr) if below else (t[i] > thr):
            if v[i] == v[i]:  # not NaN
                buf[m] = v[i]
                m += 1
    if m == 0:
        return np.nan
    b = buf[:m]
    b.sort()
    h = m // 2
    if m % 2:
        return b[h]
    return (b[h - 1] + b[h]) / 2.0


@njit(cache=True, nogil=True, parallel=True)
def _track_ends(x, y, t, lo, hi, ids, dt, t_s, t_e, x_s, y_s, x_e, y_e):
    """t_s, t_e, x_s, y_s, x_e, y_e of repair_tracks for clusters ids (events x[lo[c]:hi[c]], ...)."""
    for c in prange(ids.size):
        a = lo[c]
        b = hi[c]
        buf = np.empty(b - a)
        xc = x[a:b]
        allnan = True  # the original skips clusters whose x are all NaN
        for v in xc:
            if v == v:
                allnan = False
                break
        if allnan:
            continue
        i = ids[c]
        tc = t[a:b]
        t_s[i] = np.min(tc) + dt
        t_e[i] = np.max(tc) - dt
        x_s[i] = _nanmedian_where(xc, tc, t_s[i], True, buf)
        y_s[i] = _nanmedian_where(y[a:b], tc, t_s[i], True, buf)
        x_e[i] = _nanmedian_where(xc, tc, t_e[i], False, buf)
        y_e[i] = _nanmedian_where(y[a:b], tc, t_e[i], False, buf)


@njit(cache=True, nogil=True)
def _link_tracks(targets, x_s, y_s, t_s, ends, x_e, y_e, t_e, rank, dt, max_dtt, r, n):
    """Links of repair_tracks: for every start (ascending), the first end within r (in cKDTree order = ascending
    rank) that is not the track itself, not used yet and 0 < |(t_s - dt) - (t_e + dt)| <= max_dtt.

    Ends are found through a (time, x, y) grid instead of the ball query: an end can only link if
    t_e is in [t_s - 3 dt, t_s - dt], so only a few grid rows are searched.
    """
    ne = ends.size
    lt = np.empty(targets.size, np.int64)
    ls = np.empty(targets.size, np.int64)
    if ne == 0 or targets.size == 0:
        return lt[:0], ls[:0]
    tb = np.empty(ne, np.int64)
    cx = np.empty(ne, np.int64)
    cy = np.empty(ne, np.int64)
    for j in range(ne):
        e = ends[j]
        tb[j] = np.int64(np.floor(t_e[e] / dt))
        cx[j] = np.int64(np.floor(x_e[e] / r))
        cy[j] = np.int64(np.floor(y_e[e] / r))
    tb0 = tb.min()
    tb1 = tb.max()
    cx0 = cx.min()
    cy0 = cy.min()
    ncx = cx.max() - cx0 + 1
    ncy = cy.max() - cy0 + 1
    key = np.empty(ne, np.int64)
    for j in range(ne):
        key[j] = ((tb[j] - tb0) * ncx + (cx[j] - cx0)) * ncy + (cy[j] - cy0)
    order = np.argsort(key)
    skey = key[order]
    used = np.zeros(n, np.bool_)
    r2 = r * r
    nl = 0
    for q in range(targets.size):
        tgt = targets[q]
        xs = x_s[tgt]
        ys = y_s[tgt]
        # generous bounds (one row / cell more on each side): the exact tests below decide
        b_lo = max(np.int64(np.floor((t_s[tgt] - 3 * dt) / dt)) - 1, tb0)
        b_hi = min(np.int64(np.floor((t_s[tgt] - dt) / dt)) + 1, tb1)
        qx = np.int64(np.floor(xs / r))
        qy = np.int64(np.floor(ys / r))
        y_lo = max(qy - 1, cy0) - cy0
        y_hi = min(qy + 1, cy0 + ncy - 1) - cy0
        best = -1
        best_rank = np.iinfo(np.int64).max
        if y_lo <= y_hi:
            for b in range(b_lo, b_hi + 1):
                for gx in range(max(qx - 1, cx0), min(qx + 1, cx0 + ncx - 1) + 1):
                    row = ((b - tb0) * ncx + (gx - cx0)) * ncy
                    p = np.searchsorted(skey, row + y_lo)
                    while p < ne and skey[p] <= row + y_hi:
                        j = order[p]
                        p += 1
                        if rank[j] >= best_rank:
                            continue
                        src = ends[j]
                        if src == tgt or used[src]:
                            continue
                        dx = xs - x_e[src]
                        dy = ys - y_e[src]
                        if dx * dx + dy * dy > r2:
                            continue
                        time_diff = abs((t_s[tgt] - dt) - (t_e[src] + dt))
                        if not (0 < time_diff <= max_dtt):
                            continue
                        best = src
                        best_rank = rank[j]
        if best >= 0:
            lt[nl] = tgt
            ls[nl] = best
            nl += 1
            used[best] = True
    return lt[:nl], ls[:nl]


def _find_links(n, x_s, y_s, t_s, x_e, y_e, t_e, dt):
    """(targets, sources) of repair_tracks, in the order the links are applied."""
    max_dtt  = dt
    max_dist = 5

    valid_end = ~np.isnan(x_e) & ~np.isnan(y_e)
    end_points = np.column_stack((x_e[valid_end], y_e[valid_end]))
    end_indices = np.where(valid_end)[0]

    targets = np.flatnonzero(~np.isnan(x_s) & ~np.isnan(y_s))
    if targets.size == 0 or end_indices.size == 0 or not dt > 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    tree = cKDTree(end_points)  # built as before: its point order decides between candidates
    rank = np.empty(end_indices.size, np.int64)
    rank[tree.indices] = np.arange(end_indices.size)
    return _link_tracks(targets, x_s, y_s, t_s, end_indices, x_e, y_e, t_e, rank,
                        float(dt), float(max_dtt), float(max_dist), n)


@njit(cache=True, nogil=True)
def _apply_links(links_t, links_s, lens, steps):
    """The link loop of repair_tracks on chains of tracks instead of concatenated arrays.

    Output track i holds input tracks head[i], nxt[head[i]], ... in order (head -1: emptied). Same order as the
    list version: a target gets the current content of its source appended, the source is emptied; a link whose
    source is already empty is skipped. steps is updated in place, as before.
    """
    n = lens.size
    head = np.arange(n)
    tail = np.arange(n)
    nxt = np.full(n, -1, np.int64)
    clen = lens.copy()
    for q in range(links_t.size):
        tg = links_t[q]
        sc = links_s[q]
        if clen[sc] == 0:
            continue
        if head[tg] == -1:
            head[tg] = head[sc]
        else:
            nxt[tail[tg]] = head[sc]
        tail[tg] = tail[sc]
        clen[tg] += clen[sc]
        head[sc] = -1
        tail[sc] = -1
        clen[sc] = 0
        steps[tg] += steps[sc]
        steps[sc] = 0
    return head, nxt, clen


def repair_tracks_packed(x_clust, y_clust, t_clust, ind_clust, steps, dt):
    """repair_tracks for PackedTracks (utils/packed_tracks.py): same links and same track contents.

    The four collections are updated in place, like the lists, and returned. The track ends are computed directly
    on the flat arrays and every linked track is written once (no repeated concatenation). Tracks keep the dtype
    of their collection: the list version turned a few of them into float64 (concatenation with an emptied
    np.array([])), with the same values.
    """
    fields = (x_clust, y_clust, t_clust, ind_clust)
    n = len(x_clust)
    lens = x_clust.lengths()
    for f in fields[1:]:
        if len(f) != n or not np.array_equal(f.lengths(), lens):
            raise ValueError("repair_tracks_packed: x, y, t and ind must hold tracks of the same lengths")

    x_s = np.full(n, np.nan)
    y_s = np.full(n, np.nan)
    x_e = np.full(n, np.nan)
    y_e = np.full(n, np.nan)
    t_s = np.zeros(n)
    t_e = np.zeros(n)
    ids = np.flatnonzero(lens)
    if ids.size:
        if not (x_clust.start is y_clust.start is t_clust.start and x_clust.stop is y_clust.stop is t_clust.stop):
            if not (np.array_equal(x_clust.start, y_clust.start) and np.array_equal(x_clust.start, t_clust.start)):
                raise ValueError("repair_tracks_packed: x, y and t must share their track ranges")
        _track_ends(x_clust.data, y_clust.data, t_clust.data, x_clust.start[ids], x_clust.stop[ids], ids,
                    float(dt), t_s, t_e, x_s, y_s, x_e, y_e)

    links_t, links_s = _find_links(n, x_s, y_s, t_s, x_e, y_e, t_e, dt)
    if links_t.size == 0:
        return x_clust, y_clust, t_clust, steps, ind_clust
    if max(int(links_t.max()), int(links_s.max())) >= len(steps):
        raise IndexError("repair_tracks: track index out of range of steps")

    head, nxt, clen = _apply_links(links_t, links_s, lens, steps)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(clen, out=off[1:])
    start, stop = off[:-1], off[1:]
    for f in fields:  # one collection at a time: its old flat array is freed before the next one is built
        out = np.empty(off[-1], f.data.dtype)
        _gather_chains(f.data, f.start, f.stop, head, nxt, off, out)
        f.data, f.start, f.stop = out, start, stop
    return x_clust, y_clust, t_clust, steps, ind_clust


def repair_tracks(x_clust, y_clust, t_clust, ind_clust, steps, dt):
    """Joins a track that starts where (and right after) another one ends.

    Same links and results as the original per-cluster numpy version: the start / end medians are compiled
    (clusters taken in chunks), and the first acceptable neighbour of the ball query is found through a time / space
    grid, in the order cKDTree would list it (its traversal returns points in the order of tree.indices).
    """
    if isinstance(x_clust, PackedTracks):
        return repair_tracks_packed(x_clust, y_clust, t_clust, ind_clust, steps, dt)
    n = len(x_clust)

    x_s = np.full(n, np.nan)
    y_s = np.full(n, np.nan)
    x_e = np.full(n, np.nan)
    y_e = np.full(n, np.nan)
    t_s = np.zeros(n)
    t_e = np.zeros(n)

    lens = np.fromiter((len(a) for a in x_clust), np.int64, n)
    ids = np.flatnonzero(lens)
    a = 0
    while a < ids.size:  # chunks of ~20 M events: bounded memory, one compiled call each
        b = a + int(np.searchsorted(np.cumsum(lens[ids[a:a + 1_000_000]]), 20_000_000, side="right"))
        b = min(max(b, a + 1), ids.size)
        sel = ids[a:b]
        off = np.concatenate(([0], np.cumsum(lens[sel])))
        xs = np.concatenate([np.asarray(x_clust[i]) for i in sel.tolist()])
        ys = np.concatenate([np.asarray(y_clust[i]) for i in sel.tolist()])
        ts = np.concatenate([np.asarray(t_clust[i]) for i in sel.tolist()])
        _track_ends(xs, ys, ts, off[:-1], off[1:], sel, float(dt), t_s, t_e, x_s, y_s, x_e, y_e)
        a = b

    links_t, links_s = _find_links(n, x_s, y_s, t_s, x_e, y_e, t_e, dt)

    for idx_target, idx_source in zip(links_t.tolist(), links_s.tolist()):
        if len(x_clust[idx_source]) == 0:
            continue

        x_clust[idx_target] = np.concatenate((x_clust[idx_target], x_clust[idx_source]))
        y_clust[idx_target] = np.concatenate((y_clust[idx_target], y_clust[idx_source]))
        t_clust[idx_target] = np.concatenate((t_clust[idx_target], t_clust[idx_source]))
        ind_clust[idx_target] = np.concatenate((ind_clust[idx_target], ind_clust[idx_source]))

        x_clust[idx_source] = np.array([])
        y_clust[idx_source] = np.array([])
        t_clust[idx_source] = np.array([])
        ind_clust[idx_source] = np.array([])
        steps[idx_target] += steps[idx_source]
        steps[idx_source] = 0
    return x_clust, y_clust, t_clust, steps, ind_clust

def repair_tracks_pseudoframes(
    x_clust, y_clust, t_clust, ind_clust,
    steps, dt,
    x_m, y_m, t_m
):
    t_start = np.array([arr[0] * 1e6 for arr in t_m if len(arr) > 0])
    t_end   = np.array([arr[-1] * 1e6 for arr in t_m if len(arr) > 0])

    x_start = np.array([arr[0] for arr in x_m if len(arr) > 0])
    x_end   = np.array([arr[-1] for arr in x_m if len(arr) > 0])

    y_start = np.array([arr[0] for arr in y_m if len(arr) > 0])
    y_end   = np.array([arr[-1] for arr in y_m if len(arr) > 0])

    max_dtt  = 1.5 * dt
    max_dist = 4.0
    max_dist2 = max_dist * max_dist

    order = np.argsort(t_end)
    t_end_s = t_end[order]

    links = []
    used_sources = set()
    used_targets = set()

    for tgt in range(len(t_start)):
        if x_start[tgt] == 0:
            continue

        ts = t_start[tgt]

        left  = np.searchsorted(t_end_s, ts - max_dtt, side="left")
        right = np.searchsorted(t_end_s, ts + max_dtt, side="right")

        for k in range(left, right):
            src = order[k]

            if src == tgt:
                continue
            if src in used_sources or tgt in used_targets:
                continue
            if x_end[src] == 0:
                continue
            if t_end[src] == ts:
                continue

            dx = x_start[tgt] - x_end[src]
            dy = y_start[tgt] - y_end[src]
            if dx*dx + dy*dy > max_dist2:
                continue

            links.append((tgt, src))
            used_sources.add(src)
            used_targets.add(tgt)
            break  
    for idx_target, idx_source in links:
        if len(x_clust[idx_source]) == 0:
            continue

        x_m[idx_target] = np.concatenate((x_m[idx_target], x_m[idx_source]))
        y_m[idx_target] = np.concatenate((y_m[idx_target], y_m[idx_source]))
        t_m[idx_target] = np.concatenate((t_m[idx_target], t_m[idx_source]))

        x_clust[idx_target] = np.concatenate((x_clust[idx_target], x_clust[idx_source]))
        y_clust[idx_target] = np.concatenate((y_clust[idx_target], y_clust[idx_source]))
        t_clust[idx_target] = np.concatenate((t_clust[idx_target], t_clust[idx_source]))
        ind_clust[idx_target] = np.concatenate((ind_clust[idx_target], ind_clust[idx_source]))

        x_m[idx_source] = np.array([])
        y_m[idx_source] = np.array([])
        t_m[idx_source] = np.array([])

        x_clust[idx_source] = np.array([])
        y_clust[idx_source] = np.array([])
        t_clust[idx_source] = np.array([])
        ind_clust[idx_source] = np.array([])

        steps[idx_target] += steps[idx_source]
        steps[idx_source] = 0

    return x_clust, y_clust, t_clust, steps, ind_clust, x_m, y_m, t_m

def process_correction(x_i,y_i,t_i,xv_i,yv_i,x_clust_i,y_clust_i,t_clust_i,dt,beta,lightweight_mode):
    if lightweight_mode:
        delta_t = np.diff(t_i[~np.isnan(t_i)])
        dt_nonzero = delta_t[delta_t != 0]
        if len(dt_nonzero) > 0:
            dt = np.unique(dt_nonzero)[0]*4
    for j in range(len(x_i)):
        velocity = np.array([xv_i[j], yv_i[j]]) * 1e6
        mask = np.logical_and(
            t_clust_i > (t_i[j] - dt/2),
            t_clust_i < (t_i[j] + dt/2)
        )
        events = np.array([x_clust_i[mask], y_clust_i[mask]])
        x_i[j], y_i[j] = correct_measurement([x_i[j], y_i[j]], velocity, events, beta)
        if np.isnan(x_i[j]):
            t_i[j]=np.nan

    return x_i, y_i, t_i
