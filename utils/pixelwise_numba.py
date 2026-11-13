"""Compiled per-step cluster association of the pixelwise-extension detection (utils/detection.py).

The original loop matched the ~5000 regions of every time step to the active clusters in Python: several numpy
calls per region, a Python loop over the pixels of ambiguous regions, and every cluster history re-concatenated at
every step (quadratic in the track length). That was ~98% of the detection time.

`PixelwiseTracker` does the same work in numba loops. Cluster histories are kept as an append-only log of
(track id, event index) and assembled once, at the end of the pass, into flat arrays (utils/packed_tracks.py).
The algorithm is unchanged, including its tie-breaking and quirks (slot numbering and reuse, slot 0 never taking
part in a split, medians that include the overlap events, the ``max(label)`` slot extension), so the outputs are
identical to the Python loop, which is still used for the other detection modes (and with
``numba_matching=False``).
"""
import numpy as np
try:  # thread count of the parallel kernels (utils/parallel.py)
    from utils import parallel as _parallel  # noqa: F401
except ImportError:  # run with only utils/ on the path
    import parallel as _parallel  # noqa: F401
import numba
from numba import njit, prange
from scipy.ndimage import binary_dilation

try:
    from utils.packed_tracks import PackedTracks
except ImportError:  # run with only utils/ on the path
    from packed_tracks import PackedTracks


def dilation_offsets(structure):
    """(n, 2) offsets of the pixels that scipy's binary_dilation with `structure` sets around one pixel.

    Even sizes are not centred (size 2 covers -1..0): taking them from scipy keeps exactly its convention.
    """
    sx, sy = structure.shape
    m = np.zeros((2 * sx + 1, 2 * sy + 1), dtype=bool)
    m[sx, sy] = True
    return (np.argwhere(binary_dilation(m, structure=structure, iterations=1)) - (sx, sy)).astype(np.int64)


SERIAL_EVENTS = 20_000  # steps with fewer events run their kernels on one thread

# ---------------------------------------------------------------------------------------------------- kernels

@njit(cache=True, nogil=True)
def _find(parent, j):
    r = j
    while parent[r] != r:
        r = parent[r]
    while parent[j] != r:  # path compression
        parent[j], j = r, parent[j]
    return r


@njit(cache=True, nogil=True)
def label_events(xs, ys, pixmap, n_min, offs):
    """Region of every event, as detection.py computes it with scipy:

        mask[xs, ys] = True; mask = binary_dilation(mask, structure)   # (no dilation for a 1x1 structure)
        labeled, num = scipy.ndimage.label(mask); labels = labeled[xs, ys]
        labels[np.bincount(labels)[labels] < n_min] = 0

    `offs` are the pixels the dilation sets around one pixel (dilation_offsets; [[0, 0]] for 1x1). Two-pass
    labelling like scipy's: provisional labels in raster order, merged with the smallest label as root, so the
    roots taken in increasing order number the regions in raster order of their first pixel, as scipy does.
    4-connectivity (scipy's default structure). (A raster scan of the whole map is faster than visiting only the
    event pixels: sequential memory access.) `pixmap` is an all-zero int32 (H, W) scratch array, left all zero.
    Returns (labels int32, num).
    """
    H, W = pixmap.shape
    n = xs.size
    m = offs.shape[0]
    for i in range(n):
        for o in range(m):
            x = xs[i] + offs[o, 0]
            y = ys[i] + offs[o, 1]
            if 0 <= x < H and 0 <= y < W:
                pixmap[x, y] = -1
    parent = np.empty(n * m + 1, np.int64)  # at most one provisional label per set pixel
    parent[0] = 0
    nprov = 0
    for x in range(H):
        for y in range(W):
            if pixmap[x, y] == 0:
                continue
            up = pixmap[x - 1, y] if x > 0 else 0
            left = pixmap[x, y - 1] if y > 0 else 0
            if up > 0 and left > 0:
                a = _find(parent, up)
                b = _find(parent, left)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                pixmap[x, y] = min(a, b)
            elif up > 0:
                pixmap[x, y] = up
            elif left > 0:
                pixmap[x, y] = left
            else:
                nprov += 1
                parent[nprov] = nprov
                pixmap[x, y] = nprov
    final = np.zeros(nprov + 1, np.int32)
    num = 0
    for p in range(1, nprov + 1):
        r = _find(parent, p)
        if r == p:
            num += 1
            final[p] = num
        else:
            final[p] = final[r]  # r < p: already numbered
    labels = np.empty(n, np.int32)
    size = np.zeros(num + 1, np.int64)
    for i in range(n):
        l = final[pixmap[xs[i], ys[i]]]  # (0 if the dilation does not cover the event's own pixel)
        labels[i] = l
        size[l] += 1
    if m == 1:
        for i in range(n):
            pixmap[xs[i], ys[i]] = 0
    else:
        pixmap[:] = 0
    for i in range(n):
        if size[labels[i]] < n_min:
            labels[i] = 0
    return labels, num


@njit(cache=True, nogil=True)
def _group(labels, n_labels):
    """Event positions grouped by label: order[off[l]:off[l + 1]] are the events of label l, ascending."""
    off = np.zeros(n_labels + 2, np.int64)
    for i in range(labels.size):
        off[labels[i] + 1] += 1
    for l in range(n_labels + 1):
        off[l + 1] += off[l]
    pos = off[:-1].copy()
    order = np.empty(labels.size, np.int64)
    for i in range(labels.size):
        l = labels[i]
        order[pos[l]] = i
        pos[l] += 1
    return order, off


@njit(cache=True, nogil=True)
def _select(b, n, kth):
    """Puts the kth smallest of b[:n] at b[kth], smaller-or-equal values before it (quickselect, in place)."""
    lo = 0
    hi = n - 1
    while hi > lo:
        # median of three as pivot; three-way partition (pixel coordinates and times repeat a lot)
        mid = (lo + hi) >> 1
        p0 = b[lo]
        p1 = b[mid]
        p2 = b[hi]
        if p1 < p0:
            p0, p1 = p1, p0
        if p2 < p1:
            p1 = p2
            if p1 < p0:
                p1 = p0
        pivot = p1
        lt = lo  # b[lo:lt] < pivot, b[lt:i] == pivot, b[gt + 1:hi + 1] > pivot
        i = lo
        gt = hi
        while i <= gt:
            v = b[i]
            if v < pivot:
                b[lt], b[i] = v, b[lt]
                lt += 1
                i += 1
            elif v > pivot:
                b[gt], b[i] = v, b[gt]
                gt -= 1
            else:
                i += 1
        if kth < lt:
            hi = lt - 1
        elif kth > gt:
            lo = gt + 1
        else:
            return


@njit(cache=True, nogil=True)
def _median(buf, n):
    """np.median of buf[:n] (utils_v2.fast_median): NaN if any value is NaN, mean of the two middle values.

    Same value as sorting: the middle element(s) are found by selection, in place in the scratch buffer.
    """
    if n == 0:
        return np.nan
    for i in range(n):
        if np.isnan(buf[i]):
            return np.nan
    h = n // 2
    _select(buf, n, h)
    if n % 2:
        return buf[h]
    lower = buf[0]  # the (h - 1)th smallest: the largest of the h values left of the hth
    for i in range(1, h):
        if buf[i] > lower:
            lower = buf[i]
    return (lower + buf[h]) / 2.0


@njit(cache=True, nogil=True)
def _heap_push(heap, n, v):
    i = n
    heap[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if heap[p] <= heap[i]:
            break
        heap[p], heap[i] = heap[i], heap[p]
        i = p
    return n + 1


@njit(cache=True, nogil=True)
def _heap_pop(heap, n):
    top = heap[0]
    n -= 1
    heap[0] = heap[n]
    i = 0
    while True:
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n and heap[c + 1] < heap[c]:
            c += 1
        if heap[i] <= heap[c]:
            break
        heap[i], heap[c] = heap[c], heap[i]
        i = c
    return top, n


@njit(cache=True, nogil=True)
def _touched(a, b, dilate, offs, order, xs, ys, labelmap, out):
    """Clusters under region events order[a:b] (under its dilation within its bounding box + 1 when `dilate`),
    as written in the label map, into out (ascending); returns their number."""
    H, W = labelmap.shape
    nt = 0
    last = -1
    if dilate:
        x0 = xs[order[a]]
        x1 = x0
        y0 = ys[order[a]]
        y1 = y0
        for e in range(a, b):
            i = order[e]
            x0 = min(x0, xs[i])
            x1 = max(x1, xs[i])
            y0 = min(y0, ys[i])
            y1 = max(y1, ys[i])
        rx0 = max(x0 - 1, 0)
        rx1 = min(x1 + 2, H)
        ry0 = max(y0 - 1, 0)
        ry1 = min(y1 + 2, W)
        for e in range(a, b):
            i = order[e]
            for o in range(offs.shape[0]):
                px = xs[i] + offs[o, 0]
                py = ys[i] + offs[o, 1]
                if px < rx0 or px >= rx1 or py < ry0 or py >= ry1:
                    continue
                l = labelmap[px, py]
                if l < 0 or l == last:
                    continue
                last = l
                new = True
                for q in range(nt):
                    if out[q] == l:
                        new = False
                        break
                if new:
                    out[nt] = l
                    nt += 1
    else:
        for e in range(a, b):
            i = order[e]
            l = labelmap[xs[i], ys[i]]
            if l < 0 or l == last:
                continue
            last = l
            new = True
            for q in range(nt):
                if out[q] == l:
                    new = False
                    break
            if new:
                out[nt] = l
                nt += 1
    for q in range(1, nt):  # insertion sort: a handful of clusters
        v = out[q]
        r = q - 1
        while r >= 0 and out[r] > v:
            out[r + 1] = out[r]
            r -= 1
        out[r + 1] = v
    return nt


@njit(cache=True, nogil=True)
def _split(cl, nt, a, b, order, xs, ys, Tw, ap_off, ap_len, ap_x, ap_y, cx, cy, cnt, clg, asg, two):
    """Split of a region touching the clusters cl[:nt] (ascending) between their centres (mean of their active
    pixels): asg[e] = index in cl of the cluster event order[e] goes to, cnt[c] = its number of events, clg[c] the
    number of them with T >= 0 (its history entries).

    Centres closer than 4 px to another centre are moved out of reach (pdist / squareform, d < 4). `two` is 2.0
    passed at run time: the original squares distances with np.float64 ** 2 (the C pow), which is not always the
    correctly rounded x * x; a run-time exponent keeps that call.
    """
    for c in range(nt):
        s = cl[c]
        sx = 0.0
        sy = 0.0
        for p in range(ap_off[s], ap_off[s] + ap_len[s]):
            sx += ap_x[p]
            sy += ap_y[p]
        cx[c] = sx / ap_len[s]
        cy[c] = sy / ap_len[s]
    for c in range(nt):
        cnt[c] = 0  # "too close" flags first, then the event counts
    for c in range(nt):
        for d in range(c + 1, nt):
            dx = cx[c] - cx[d]
            dy = cy[c] - cy[d]
            if np.sqrt(dx * dx + dy * dy) < 4:
                cnt[c] = 1
                cnt[d] = 1
    for c in range(nt):
        if cnt[c]:
            cx[c] = -1000.0
            cy[c] = -1000.0
        cnt[c] = 0
        clg[c] = 0
    for e in range(a, b):
        i = order[e]
        best = 0
        bd = 0.0
        for c in range(nt):
            dd = (xs[i] - cx[c]) ** two + (ys[i] - cy[c]) ** two
            if c == 0 or dd < bd:  # np.argmin: first minimum
                bd = dd
                best = c
        asg[e] = best
        cnt[best] += 1
        if Tw[i] >= 0:
            clg[best] += 1


@njit(cache=True, nogil=True, parallel=True)
def _match(k, dilate, offs, xs, ys, Xw, Yw, Tw, indw, order, off, labelmap, vlist,
           ap_off, ap_len, ap_x, ap_y, steps, inactive_k, x_m, y_m, t_m, mark,
           nap_x, nap_y, ent_slot, ent_off, ent_len, ent_lg, log_ind, newlabs, reg_lg, two):
    """Match the regions of one step to the valid clusters (the region loop of the pixelwise branch).

    The original takes the regions in ascending label order; a cluster that matches a region leaves the label map,
    so later regions cannot match it. That is the only dependence between regions, so the work is split into
    passes, each one parallel over regions except the short resolution pass:

    1. the label map of the active clusters (sequential: shared pixels keep the highest cluster);
    2. the clusters under every region, in the label map as painted (parallel);
    3. the split of every region touching several clusters, as if all of them were still free (parallel);
    4. resolution in ascending label order (sequential): the clusters already taken leave the region; a region
       left with one cluster gives it all its events, with several the split is used (computed again only if some
       of its clusters were taken); every cluster that takes events is one "job";
    5. the jobs (event copies, medians, active pixels, history log) in parallel, each one writing at offsets
       computed from the job sizes in resolution order, so every output is where the sequential loop put it.

    Every job is one entry: its cluster (ent_slot), its new active pixels (ent_off, ent_len into nap_x, nap_y)
    and its history entries (ent_lg consecutive event indices in log_ind, in entry order).
    Returns (new active-pixel entries, log entries, unmatched regions to start as new clusters).
    `mark` (all zero on entry and on return) flags the clusters that took events. reg_lg[lab]: events of region
    lab with T >= 0, for _create.
    """
    nv = vlist.size
    n_lab = off.size - 2
    n_ev = xs.size
    n_off = offs.shape[0] if dilate else 1
    for j in range(nv):
        s = vlist[j]
        for p in range(ap_off[s], ap_off[s] + ap_len[s]):
            labelmap[ap_x[p], ap_y[p]] = s

    # 2. + 3. (one parallel loop: every loop start costs ~50 us) clusters under every region (at most
    # min(probes, nv) per region), speculative splits (all touched clusters free), events with T >= 0 (history
    # entries, kept for _create in reg_lg) and with T > 0 (new clusters) of every region
    tbase = np.zeros(n_lab + 2, np.int64)
    for lab in range(1, n_lab + 1):
        tbase[lab + 1] = tbase[lab] + min((off[lab + 1] - off[lab]) * n_off, max(nv, 1))
    tl = np.empty(tbase[n_lab + 1], np.int64)
    tn = np.zeros(n_lab + 1, np.int64)
    cx = np.empty(tl.size)
    cy = np.empty(tl.size)
    cnt = np.zeros(tl.size, np.int64)
    clg = np.zeros(tl.size, np.int64)
    asg = np.empty(n_ev, np.int64)
    reg_new = np.zeros(n_lab + 1, np.int64)
    for lab in prange(1, n_lab + 1):
        a = off[lab]
        b = off[lab + 1]
        lg = 0
        nw = 0
        for e in range(a, b):
            t = Tw[order[e]]
            if t >= 0:
                lg += 1
                if t > 0:
                    nw += 1
        reg_lg[lab] = lg
        reg_new[lab] = nw
        if a == b:
            continue
        t0 = tbase[lab]
        nt = _touched(a, b, dilate, offs, order, xs, ys, labelmap, tl[t0:tbase[lab + 1]])
        tn[lab] = nt
        if nt >= 2:
            _split(tl[t0:t0 + nt], nt, a, b, order, xs, ys, Tw, ap_off, ap_len, ap_x, ap_y,
                   cx[t0:t0 + nt], cy[t0:t0 + nt], cnt[t0:t0 + nt], clg[t0:t0 + nt], asg, two)

    # 4. resolution, ascending labels
    assigned = np.zeros(n_lab + 1, np.uint8)
    j_lab = np.empty(tl.size, np.int64)
    j_s = np.empty(tl.size, np.int64)
    j_c = np.empty(tl.size, np.int64)  # -1: all events of the region, else the events with asg == c
    j_ap = np.zeros(tl.size + 1, np.int64)  # offsets of the active-pixel entries of every job
    j_lg = np.zeros(tl.size + 1, np.int64)  # offsets of the history entries
    nj = 0
    for lab in range(1, n_lab + 1):
        nt = tn[lab]
        if nt == 0:
            continue
        t0 = tbase[lab]
        ne = 0
        for q in range(nt):
            if mark[tl[t0 + q]] == 0:
                tl[t0 + ne] = tl[t0 + q]  # free clusters to the front, still ascending
                ne += 1
        if ne == 0:
            continue
        assigned[lab] = 1
        a = off[lab]
        b = off[lab + 1]
        if ne == 1:
            s = tl[t0]
            mark[s] = 1
            j_lab[nj] = lab
            j_s[nj] = s
            j_c[nj] = -1
            j_ap[nj + 1] = j_ap[nj] + b - a
            j_lg[nj + 1] = j_lg[nj] + reg_lg[lab]
            nj += 1
            continue
        if ne < nt:
            _split(tl[t0:t0 + ne], ne, a, b, order, xs, ys, Tw, ap_off, ap_len, ap_x, ap_y,
                   cx[t0:t0 + ne], cy[t0:t0 + ne], cnt[t0:t0 + ne], clg[t0:t0 + ne], asg, two)
        # clusters with at least 4 events take them; label 0 means "unassigned", so slot 0 never takes part
        for c in range(ne):
            s = tl[t0 + c]
            if s == 0 or cnt[t0 + c] < 4:
                continue
            mark[s] = 1
            j_lab[nj] = lab
            j_s[nj] = s
            j_c[nj] = c
            j_ap[nj + 1] = j_ap[nj] + cnt[t0 + c]
            j_lg[nj + 1] = j_lg[nj] + clg[t0 + c]
            nj += 1

    # 5. jobs
    fx = np.empty(j_ap[nj])
    fy = np.empty(j_ap[nj])
    ft = np.empty(j_ap[nj])
    for j in prange(nj):
        lab = j_lab[j]
        s = j_s[j]
        c = j_c[j]
        p0 = j_ap[j]
        p = p0
        g = j_lg[j]
        for e in range(off[lab], off[lab + 1]):
            if c >= 0 and asg[e] != c:
                continue
            i = order[e]
            fx[p] = Xw[i]
            fy[p] = Yw[i]
            ft[p] = Tw[i]
            if Tw[i] >= 0:
                log_ind[g] = indw[i]
                g += 1
            nap_x[p] = xs[i]
            nap_y[p] = ys[i]
            p += 1
        ns = p - p0
        # the medians include the overlap events (T = -1), as before
        x_m[s] = _median(fx[p0:p], ns)
        y_m[s] = _median(fy[p0:p], ns)
        t_m[s] = _median(ft[p0:p], ns) * 1e-6
        ent_slot[j] = s
        ent_off[j] = p0
        ent_len[j] = ns
        ent_lg[j] = g - j_lg[j]
        steps[s] += 1
        inactive_k[s] = k

    # valid clusters that matched no region stop (valid = 0, inactive_k = k); the label map is cleared. Every
    # cluster that took events is in the label map, so a valid one: its mark is cleared here
    for j in prange(nv):
        s = vlist[j]
        if mark[s] == 0:
            inactive_k[s] = k
        mark[s] = 0
        for p in range(ap_off[s], ap_off[s] + ap_len[s]):
            labelmap[ap_x[p], ap_y[p]] = -1

    # unmatched regions with more than 5 events of this step (T > 0) start new clusters, ascending labels
    nn = 0
    for lab in range(1, n_lab + 1):
        if assigned[lab] == 0 and reg_new[lab] > 5:
            newlabs[nn] = lab
            nn += 1
    return j_ap[nj], nj, j_lg[nj], nn


@njit(cache=True, nogil=True, parallel=True)
def _create(k, newlabs, nn, order, off, xs, ys, Xw, Yw, Tw, indw, heap, heap_n,
            steps, inactive, inactive_k, x_m, y_m, t_m, tid, next_tid,
            nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, ent_lg, log_ind, nlog, cand, cand_n, reg_lg):
    """New clusters in the lowest free slots (refil[m]), ascending region label.

    The slots come from the free-slot heap (sequential); the clusters are then filled in parallel, each at offsets
    computed from the region sizes, so the entries are where the sequential loop put them.
    """
    slot = np.empty(nn, np.int64)
    for q in range(nn):
        slot[q], heap_n = _heap_pop(heap, heap_n)
    q_ap = np.zeros(nn + 1, np.int64)
    q_lg = np.zeros(nn + 1, np.int64)
    for q in range(nn):
        lab = newlabs[q]
        q_ap[q + 1] = q_ap[q] + off[lab + 1] - off[lab]
        q_lg[q + 1] = q_lg[q] + reg_lg[lab]  # events with T >= 0 (counted by _match)
    fx = np.empty(q_ap[nn])
    fy = np.empty(q_ap[nn])
    ft = np.empty(q_ap[nn])
    for q in prange(nn):
        lab = newlabs[q]
        r = slot[q]
        t = next_tid + q
        tid[r] = t
        f0 = q_ap[q]
        m = 0
        g = nlog + q_lg[q]
        p = nap + q_ap[q]
        for e in range(off[lab], off[lab + 1]):
            i = order[e]
            if Tw[i] >= 0:  # history and medians without the overlap events
                fx[f0 + m] = Xw[i]
                fy[f0 + m] = Yw[i]
                ft[f0 + m] = Tw[i]
                m += 1
                log_ind[g] = indw[i]
                g += 1
            nap_x[p] = xs[i]  # active pixels: all events of the region
            nap_y[p] = ys[i]
            p += 1
        ent_slot[nent + q] = r
        ent_off[nent + q] = nap + q_ap[q]
        ent_len[nent + q] = q_ap[q + 1] - q_ap[q]
        ent_lg[nent + q] = m
        x_m[r] = _median(fx[f0:f0 + m], m)
        y_m[r] = _median(fy[f0:f0 + m], m)
        t_m[r] = _median(ft[f0:f0 + m], m) * 1e-6
        steps[r] = 1
        inactive[r] = 0
        inactive_k[r] = k
    for q in range(nn):
        cand[cand_n + q] = slot[q]
    return heap_n, next_tid + nn, nap + q_ap[nn], nent + nn, nlog + q_lg[nn], cand_n + nn


@njit(cache=True, nogil=True)
def _release(k, thresh, cand, cand_n, steps, inactive, inactive_k, heap, heap_n):
    """inactive[(steps < buffer) & (steps < min_steps) & (inactive_k + 2 < k)] = 1, over the only slots it can
    change (thresh = min(buffer, min_steps)).

    `cand` holds every slot with inactive == 0 and steps < thresh: slots enter it when they start a
    cluster and never come back until they start another one.
    """
    w = 0
    for j in range(cand_n):
        s = cand[j]
        if steps[s] >= thresh:
            continue
        if inactive_k[s] + 2 < k:
            inactive[s] = 1
            heap_n = _heap_push(heap, heap_n, s)
            continue
        cand[w] = s
        w += 1
    return w, heap_n


@njit(cache=True, nogil=True, parallel=True)
def _assemble(log_ind, run_tid, run_len, tid):
    """Indices of the live tracks grouped by slot, in log (= append) order: ind[off[s]:off[s + 1]] for slot s.

    The log is a sequence of runs: run r is run_len[r] consecutive entries of log_ind, of track run_tid[r]. The
    runs are placed sequentially (a few per cluster and step), their entries copied in parallel.
    """
    n_slots = tid.size
    n_tid = 0
    for s in range(n_slots):
        n_tid = max(n_tid, tid[s] + 1)
    slot_of = np.full(n_tid, -1, np.int64)
    for s in range(n_slots):
        if tid[s] >= 0:
            slot_of[tid[s]] = s
    nr = run_tid.size
    src = np.empty(nr + 1, np.int64)
    src[0] = 0
    for r in range(nr):
        src[r + 1] = src[r] + run_len[r]
    rslot = np.empty(nr, np.int64)
    off = np.zeros(n_slots + 1, np.int64)
    for r in range(nr):
        t = run_tid[r]
        sl = slot_of[t] if t < n_tid else -1
        rslot[r] = sl
        if sl >= 0:
            off[sl + 1] += run_len[r]
    for s in range(n_slots):
        off[s + 1] += off[s]
    dest = np.empty(nr, np.int64)
    pos = off[:-1].copy()
    for r in range(nr):
        sl = rslot[r]
        if sl >= 0:
            dest[r] = pos[sl]
            pos[sl] += run_len[r]
    ind = np.empty(off[n_slots], np.int64)
    for r in prange(nr):
        if rslot[r] >= 0:
            d = dest[r]
            a = src[r]
            for q in range(run_len[r]):
                ind[d + q] = log_ind[a + q]
    return ind, off


@njit(cache=True, nogil=True, parallel=True)
def _take(src, ind, plus):
    """src[ind] + plus (plus = 1) or src[ind] (plus = 0), in src's dtype, in parallel."""
    out = np.empty(ind.size, src.dtype)
    if plus:
        for i in prange(ind.size):
            out[i] = src[ind[i]] + 1
    else:
        for i in prange(ind.size):
            out[i] = src[ind[i]]
    return out


# ---------------------------------------------------------------------------------------------------- tracker

class PixelwiseTracker:
    """State of the pixelwise-extension association from step 1 to the end of one pass.

    Built from the step-0 state of utils/detection.py; `export` gives back what the Python loop would have produced
    (x_clust, y_clust, t_clust, ind_clust as PackedTracks with the same tracks; steps, x_m, y_m, t_m, max_tracks).
    """

    def __init__(self, height, width, structure, second_pass, buffer, n_global,
                 steps, inactive, inactive_k, valid, x_m, y_m, t_m, active_pixels, ind_clust, min_steps=3):
        n = len(steps)
        self.n = n
        self.cap = n
        self.steps_dtype = np.asarray(steps).dtype
        # extend_1d(steps, n, 0) concatenates an int array: the dtype the original ends with once it grew
        self.steps_dtype_extended = np.concatenate((np.asarray(steps)[:0], np.full(1, 0))).dtype
        self.extended = False
        self.steps = np.asarray(steps, dtype=np.int64).copy()
        self.inactive = np.asarray(inactive).astype(np.int8)
        self.inactive_k = np.asarray(inactive_k, dtype=np.int64).copy()
        self.x_m = np.asarray(x_m, dtype=np.float64).copy()
        self.y_m = np.asarray(y_m, dtype=np.float64).copy()
        self.t_m = np.asarray(t_m, dtype=np.float64).copy()
        self.tid = np.full(n, -1, np.int64)
        self.ap_off = np.zeros(n, np.int64)
        self.ap_len = np.zeros(n, np.int64)
        self.mark = np.zeros(n, np.uint8)
        self.ind_dtype = np.int32 if n_global < 2 ** 31 else np.int64

        # clusters started at step 0 (inactive == 0): their history and active pixels
        filled = np.flatnonzero(self.inactive == 0)
        self.tid[filled] = np.arange(filled.size)
        self.next_tid = int(filled.size)
        lens = np.array([len(ind_clust[s]) for s in filled], dtype=np.int64)
        # history log as runs: run_len[r] consecutive entries of log_ind belong to track run_tid[r]
        self.run_tid = [np.arange(filled.size, dtype=np.int32)]
        self.run_len = [lens.astype(np.int32)]
        self.log_ind = [np.concatenate([np.asarray(ind_clust[s], dtype=self.ind_dtype) for s in filled])
                        if filled.size else np.zeros(0, self.ind_dtype)]
        aps = [np.asarray(active_pixels[s]).reshape(-1, 2) for s in filled]
        ap = np.concatenate(aps).astype(np.int32) if aps else np.zeros((0, 2), np.int32)
        self.ap_x = np.ascontiguousarray(ap[:, 0])
        self.ap_y = np.ascontiguousarray(ap[:, 1])
        ap_lens = np.array([len(a) for a in aps], dtype=np.int64)
        self.ap_len[filled] = ap_lens
        self.ap_off[filled] = np.cumsum(ap_lens) - ap_lens

        self.vlist = np.flatnonzero(np.asarray(valid) != 0).astype(np.int64)
        self.thresh = float(min(min_steps, buffer))
        self.cand = np.zeros(n, np.int64)
        c = np.flatnonzero((self.inactive == 0) & (self.steps < self.thresh))
        self.cand[:c.size] = c
        self.cand_n = int(c.size)
        self.heap = np.zeros(n, np.int64)
        h = np.flatnonzero(self.inactive == 1)  # ascending: already a min-heap
        self.heap[:h.size] = h
        self.heap_n = int(h.size)

        self.labelmap = np.full((height, width), -1, dtype=np.int32)
        self.dilate = bool(second_pass)
        self.offs = dilation_offsets(np.asarray(structure, dtype=bool))

    def _grow(self, missing):
        """extend_1d / extend_list_of_arrays of the original: `missing` new free slots at the end."""
        new_n = self.n + missing
        if new_n > self.cap:
            cap = max(new_n, 2 * self.cap)

            def grown(a, fill):
                b = np.full(cap, fill, dtype=a.dtype)
                b[:self.n] = a[:self.n]
                return b

            self.steps = grown(self.steps, 0)
            self.inactive = grown(self.inactive, 1)
            self.inactive_k = grown(self.inactive_k, -1)
            self.x_m = grown(self.x_m, np.nan)
            self.y_m = grown(self.y_m, np.nan)
            self.t_m = grown(self.t_m, np.nan)
            self.tid = grown(self.tid, -1)
            self.ap_off = grown(self.ap_off, 0)
            self.ap_len = grown(self.ap_len, 0)
            self.mark = grown(self.mark, 0)
            heap = np.zeros(cap, np.int64)
            heap[:self.heap_n] = self.heap[:self.heap_n]
            self.heap = heap
            cand = np.zeros(cap, np.int64)
            cand[:self.cand_n] = self.cand[:self.cand_n]
            self.cand = cand
            self.cap = cap
        # the new slots are larger than every free slot: appended in order, the heap stays a heap
        self.heap[self.heap_n:self.heap_n + missing] = np.arange(self.n, new_n)
        self.heap_n += missing
        self.n = new_n
        self.extended = True

    def step(self, k, xs, ys, order, off, X, Y, T, ind_global):
        if xs.size < SERIAL_EVENTS:  # waking the kernel threads (~50 us per parallel loop) costs more than it saves
            numba.set_num_threads(1)
            try:
                self._step(k, xs, ys, order, off, X, Y, T, ind_global)
            finally:
                _parallel.apply()
        else:
            self._step(k, xs, ys, order, off, X, Y, T, ind_global)

    def _step(self, k, xs, ys, order, off, X, Y, T, ind_global):
        """One time step k > 0. xs, ys: 0-based event pixels; order, off: the events grouped by kept region
        (_group of label_events: order[off[l]:off[l + 1]] are the events of region l, region 0 = none)."""
        n = xs.size
        nap_x = np.empty(n, np.int32)
        nap_y = np.empty(n, np.int32)
        ent_slot = np.empty(n, np.int64)
        ent_off = np.empty(n, np.int64)
        ent_len = np.empty(n, np.int64)
        ent_lg = np.empty(n, np.int32)
        log_ind = np.empty(n, self.ind_dtype)
        newlabs = np.empty(off.size - 1, np.int64)
        reg_lg = np.empty(off.size - 1, np.int64)
        nap, nent, nlog, nn = _match(
            k, self.dilate, self.offs, xs, ys, X, Y, T, ind_global, order, off, self.labelmap, self.vlist,
            self.ap_off, self.ap_len, self.ap_x, self.ap_y, self.steps, self.inactive_k,
            self.x_m, self.y_m, self.t_m, self.mark,
            nap_x, nap_y, ent_slot, ent_off, ent_len, ent_lg, log_ind, newlabs, reg_lg, 2.0)
        if nn > 0:
            missing = int(newlabs[nn - 1]) - self.heap_n  # needed = max(idx), the largest new label
            if missing > 0:
                self._grow(missing)
            self.heap_n, self.next_tid, nap, nent, nlog, self.cand_n = _create(
                k, newlabs, nn, order, off, xs, ys, X, Y, T, ind_global, self.heap, self.heap_n,
                self.steps, self.inactive, self.inactive_k, self.x_m, self.y_m, self.t_m, self.tid,
                self.next_tid, nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, ent_lg, log_ind, nlog,
                self.cand, self.cand_n, reg_lg)
        self.cand_n, self.heap_n = _release(k, self.thresh, self.cand, self.cand_n, self.steps, self.inactive,
                                            self.inactive_k, self.heap, self.heap_n)
        # matched and new clusters are the valid ones of the next step, with their new active pixels
        slots = ent_slot[:nent]
        self.ap_off[slots] = ent_off[:nent]
        self.ap_len[slots] = ent_len[:nent]
        self.ap_x = nap_x
        self.ap_y = nap_y
        self.vlist = np.sort(slots)
        if nent:
            self.run_tid.append(self.tid[slots].astype(np.int32))  # (a slot's track id does not change in a step)
            self.run_len.append(ent_lg[:nent].copy())
        if nlog:
            self.log_ind.append(log_ind[:nlog].copy())

    def export(self, X_global, Y_global, T_global):
        """Cluster histories of the pass as PackedTracks (one track per slot, empty for never used slots) and the
        per-slot arrays, as the Python loop leaves them at the end of the pass."""
        n = self.n
        log_ind = np.concatenate(self.log_ind)
        run_tid = np.concatenate(self.run_tid)
        run_len = np.concatenate(self.run_len)
        self.log_ind = self.run_tid = self.run_len = None
        ind, off = _assemble(log_ind, run_tid, run_len, self.tid[:n])
        del log_ind, run_tid, run_len
        start, stop = off[:-1], off[1:]
        dtype = self.steps_dtype_extended if self.extended else self.steps_dtype
        return dict(x_clust=PackedTracks(_take(X_global, ind, 1), start, stop),  # X_global[ind] + 1
                    y_clust=PackedTracks(_take(Y_global, ind, 1), start, stop),
                    t_clust=PackedTracks(_take(T_global, ind, 0), start, stop),
                    ind_clust=PackedTracks(ind, start, stop),
                    steps=self.steps[:n].astype(dtype), x_m=self.x_m[:n].copy(), y_m=self.y_m[:n].copy(),
                    t_m=self.t_m[:n].copy(), max_tracks=n)
