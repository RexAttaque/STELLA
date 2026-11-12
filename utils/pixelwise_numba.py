"""Compiled per-step cluster association of the pixelwise-extension detection (utils/detection.py).

The original loop matched the ~5000 regions of every time step to the active clusters in Python: several numpy
calls per region, a Python loop over the pixels of ambiguous regions, and every cluster history re-concatenated at
every step (quadratic in the track length). That was ~98% of the detection time.

`PixelwiseTracker` does the same work in numba loops. Cluster histories are kept as an append-only log of
(track id, event index) and assembled once, at the end of the pass, from the global event arrays. The algorithm is
unchanged, including its tie-breaking and quirks (slot numbering and reuse, slot 0 never taking part in a split,
medians that include the overlap events, the ``max(label)`` slot extension), so the outputs are identical to the
Python loop, which is still used for the other detection modes (and with ``numba_matching=False``).
"""
import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation

_EMPTY = np.array([])


def dilation_offsets(structure):
    """(n, 2) offsets of the pixels that scipy's binary_dilation with `structure` sets around one pixel.

    Even sizes are not centred (size 2 covers -1..0): taking them from scipy keeps exactly its convention.
    """
    sx, sy = structure.shape
    m = np.zeros((2 * sx + 1, 2 * sy + 1), dtype=bool)
    m[sx, sy] = True
    return (np.argwhere(binary_dilation(m, structure=structure, iterations=1)) - (sx, sy)).astype(np.int64)


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
def label_events(xs, ys, pixmap, n_min):
    """Region of every event, as detection.py computes it without dilation (1x1 structure):

        mask[xs, ys] = True; labeled, num = scipy.ndimage.label(mask); labels = labeled[xs, ys]
        labels[np.bincount(labels)[labels] < n_min] = 0

    Two-pass labelling like scipy's: provisional labels in raster order, merged with the smallest label as root,
    so the roots taken in increasing order number the regions in raster order of their first pixel, as scipy
    does. 4-connectivity (scipy's default structure). `pixmap` is an all-zero int32 (H, W) scratch array, left
    all zero. Returns (labels int32, num).
    """
    H, W = pixmap.shape
    n = xs.size
    for i in range(n):
        pixmap[xs[i], ys[i]] = -1
    parent = np.empty(n + 1, np.int64)  # at most one provisional label per distinct pixel
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
        l = final[pixmap[xs[i], ys[i]]]
        labels[i] = l
        size[l] += 1
    for i in range(n):
        pixmap[xs[i], ys[i]] = 0
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
def _update(s, sel, ns, k, xs, ys, Xw, Yw, Tw, indw, labelmap, ap_off, ap_len, ap_x, ap_y,
            steps, inactive_k, x_m, y_m, t_m, tid, mark, fx, fy, ft,
            nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, log_tid, log_ind, nlog):
    """utils_v2.update_cluster + the t >= 0 filter + _clear_label + active_temp, for events sel[:ns] of cluster s."""
    for q in range(ns):
        i = sel[q]
        fx[q] = Xw[i]
        fy[q] = Yw[i]
        ft[q] = Tw[i]
    # the medians include the overlap events (T = -1), as before
    x_m[s] = _median(fx, ns)
    y_m[s] = _median(fy, ns)
    t_m[s] = _median(ft, ns) * 1e-6
    ent_slot[nent] = s
    ent_off[nent] = nap
    for q in range(ns):
        i = sel[q]
        if Tw[i] >= 0:
            log_tid[nlog] = tid[s]
            log_ind[nlog] = indw[i]
            nlog += 1
        nap_x[nap] = xs[i]
        nap_y[nap] = ys[i]
        nap += 1
    ent_len[nent] = ns
    nent += 1
    steps[s] += 1
    inactive_k[s] = k
    mark[s] = 1
    # the cluster leaves the label map: later regions of this step cannot match it again
    for p in range(ap_off[s], ap_off[s] + ap_len[s]):
        if labelmap[ap_x[p], ap_y[p]] == s:
            labelmap[ap_x[p], ap_y[p]] = -1
    return nap, nent, nlog


@njit(cache=True, nogil=True)
def _match(k, dilate, offs, xs, ys, Xw, Yw, Tw, indw, order, off, labelmap, vlist,
           ap_off, ap_len, ap_x, ap_y, steps, inactive_k, x_m, y_m, t_m, tid, mark, touch,
           nap_x, nap_y, ent_slot, ent_off, ent_len, log_tid, log_ind, newlabs, two):
    """Match the regions of one step to the valid clusters (the region loop of the pixelwise branch).

    Returns (new active-pixel entries, log entries, unmatched regions to start as new clusters).
    `two` is 2.0 passed at run time: the original squares distances with np.float64 ** 2 (the C pow), which is
    not always the correctly rounded x * x; a run-time exponent keeps that call.
    """
    H, W = labelmap.shape
    nv = vlist.size
    # label map of the active clusters, written in ascending order: shared pixels keep the highest cluster
    for j in range(nv):
        s = vlist[j]
        for p in range(ap_off[s], ap_off[s] + ap_len[s]):
            labelmap[ap_x[p], ap_y[p]] = s

    n_lab = off.size - 2
    nmax = 1
    for lab in range(1, n_lab + 1):
        nmax = max(nmax, off[lab + 1] - off[lab])
    fx = np.empty(nmax)
    fy = np.empty(nmax)
    ft = np.empty(nmax)
    sel = np.empty(nmax, np.int64)
    asg = np.empty(nmax, np.int64)
    tl = np.empty(max(nv, 1), np.int64)
    cx = np.empty(max(nv, 1))
    cy = np.empty(max(nv, 1))
    far = np.zeros(max(nv, 1), np.bool_)
    cnt = np.zeros(max(nv, 1), np.int64)
    assigned = np.zeros(n_lab + 1, np.uint8)
    nap = 0
    nent = 0
    nlog = 0

    for lab in range(1, n_lab + 1):
        a = off[lab]
        b = off[lab + 1]
        if a == b:
            continue
        # clusters under the region (under its dilation within its bounding box + 1 on the second pass)
        nt = 0
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
                    if l >= 0 and touch[l] == 0:
                        touch[l] = 1
                        tl[nt] = l
                        nt += 1
        else:
            for e in range(a, b):
                i = order[e]
                l = labelmap[xs[i], ys[i]]
                if l >= 0 and touch[l] == 0:
                    touch[l] = 1
                    tl[nt] = l
                    nt += 1
        for q in range(nt):
            touch[tl[q]] = 0
        if nt == 0:
            continue
        assigned[lab] = 1

        if nt == 1:
            ns = b - a
            for e in range(a, b):
                sel[e - a] = order[e]
            nap, nent, nlog = _update(tl[0], sel, ns, k, xs, ys, Xw, Yw, Tw, indw, labelmap, ap_off, ap_len,
                                      ap_x, ap_y, steps, inactive_k, x_m, y_m, t_m, tid, mark, fx, fy, ft,
                                      nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, log_tid, log_ind, nlog)
            continue

        # several clusters: split the region between their centres (mean of their active pixels)
        cl = tl[:nt]
        cl.sort()
        for c in range(nt):
            s = cl[c]
            sx = 0.0
            sy = 0.0
            for p in range(ap_off[s], ap_off[s] + ap_len[s]):
                sx += ap_x[p]
                sy += ap_y[p]
            cx[c] = sx / ap_len[s]
            cy[c] = sy / ap_len[s]
        # centres closer than 4 px to another centre are moved out of reach (pdist / squareform, d < 4)
        for c in range(nt):
            far[c] = False
            cnt[c] = 0
        for c in range(nt):
            for d in range(c + 1, nt):
                dx = cx[c] - cx[d]
                dy = cy[c] - cy[d]
                if np.sqrt(dx * dx + dy * dy) < 4:
                    far[c] = True
                    far[d] = True
        for c in range(nt):
            if far[c]:
                cx[c] = -1000.0
                cy[c] = -1000.0
        for e in range(a, b):
            i = order[e]
            best = 0
            bd = 0.0
            for c in range(nt):
                dd = (xs[i] - cx[c]) ** two + (ys[i] - cy[c]) ** two
                if c == 0 or dd < bd:  # np.argmin: first minimum
                    bd = dd
                    best = c
            asg[e - a] = best
            cnt[best] += 1
        # clusters with at least 4 events take them; label 0 means "unassigned", so slot 0 never takes part
        for c in range(nt):
            s = cl[c]
            if s == 0 or cnt[c] < 4:
                continue
            ns = 0
            for e in range(a, b):
                if asg[e - a] == c:
                    sel[ns] = order[e]
                    ns += 1
            nap, nent, nlog = _update(s, sel, ns, k, xs, ys, Xw, Yw, Tw, indw, labelmap, ap_off, ap_len,
                                      ap_x, ap_y, steps, inactive_k, x_m, y_m, t_m, tid, mark, fx, fy, ft,
                                      nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, log_tid, log_ind, nlog)

    # valid clusters that matched no region stop (valid = 0, inactive_k = k)
    for j in range(nv):
        s = vlist[j]
        if mark[s] == 0:
            inactive_k[s] = k
        for p in range(ap_off[s], ap_off[s] + ap_len[s]):
            labelmap[ap_x[p], ap_y[p]] = -1
    for e in range(nent):
        mark[ent_slot[e]] = 0

    # unmatched regions with more than 5 events of this step (T > 0) start new clusters, ascending labels
    nn = 0
    for lab in range(1, n_lab + 1):
        a = off[lab]
        b = off[lab + 1]
        if assigned[lab] or a == b:
            continue
        c = 0
        for e in range(a, b):
            if Tw[order[e]] > 0:
                c += 1
        if c > 5:
            newlabs[nn] = lab
            nn += 1
    return nap, nent, nlog, nn


@njit(cache=True, nogil=True)
def _create(k, newlabs, nn, order, off, xs, ys, Xw, Yw, Tw, indw, heap, heap_n,
            steps, inactive, inactive_k, x_m, y_m, t_m, tid, next_tid,
            nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, log_tid, log_ind, nlog, cand, cand_n):
    """New clusters in the lowest free slots (refil[m]), ascending region label."""
    nmax = 1
    for q in range(nn):
        nmax = max(nmax, off[newlabs[q] + 1] - off[newlabs[q]])
    fx = np.empty(nmax)
    fy = np.empty(nmax)
    ft = np.empty(nmax)
    for q in range(nn):
        lab = newlabs[q]
        r, heap_n = _heap_pop(heap, heap_n)
        tid[r] = next_tid
        next_tid += 1
        ent_slot[nent] = r
        ent_off[nent] = nap
        m = 0
        for e in range(off[lab], off[lab + 1]):
            i = order[e]
            if Tw[i] >= 0:  # history and medians without the overlap events
                fx[m] = Xw[i]
                fy[m] = Yw[i]
                ft[m] = Tw[i]
                m += 1
                log_tid[nlog] = next_tid - 1
                log_ind[nlog] = indw[i]
                nlog += 1
            nap_x[nap] = xs[i]  # active pixels: all events of the region
            nap_y[nap] = ys[i]
            nap += 1
        ent_len[nent] = nap - ent_off[nent]
        nent += 1
        x_m[r] = _median(fx, m)
        y_m[r] = _median(fy, m)
        t_m[r] = _median(ft, m) * 1e-6
        steps[r] = 1
        inactive[r] = 0
        inactive_k[r] = k
        cand[cand_n] = r
        cand_n += 1
    return heap_n, next_tid, nap, nent, nlog, cand_n


@njit(cache=True, nogil=True)
def _release(k, thresh, cand, cand_n, steps, inactive, inactive_k, heap, heap_n):
    """inactive[(steps < buffer) & (steps < 3) & (inactive_k + 2 < k)] = 1, over the only slots it can change.

    `cand` holds every slot with inactive == 0 and steps < min(buffer, 3): slots enter it when they start a
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


@njit(cache=True, nogil=True)
def _assemble(log_tid, log_ind, tid):
    """Indices of the live tracks grouped by slot, in log (= append) order: ind[off[s]:off[s + 1]] for slot s."""
    n_slots = tid.size
    n_tid = 0
    for s in range(n_slots):
        n_tid = max(n_tid, tid[s] + 1)
    slot_of = np.full(n_tid, -1, np.int64)
    for s in range(n_slots):
        if tid[s] >= 0:
            slot_of[tid[s]] = s
    off = np.zeros(n_slots + 1, np.int64)
    for j in range(log_tid.size):
        t = log_tid[j]
        if t < n_tid and slot_of[t] >= 0:
            off[slot_of[t] + 1] += 1
    for s in range(n_slots):
        off[s + 1] += off[s]
    pos = off[:-1].copy()
    ind = np.empty(off[n_slots], np.int64)
    for j in range(log_tid.size):
        t = log_tid[j]
        if t < n_tid and slot_of[t] >= 0:
            s = slot_of[t]
            ind[pos[s]] = log_ind[j]
            pos[s] += 1
    return ind, off


# ---------------------------------------------------------------------------------------------------- tracker

class PixelwiseTracker:
    """State of the pixelwise-extension association from step 1 to the end of one pass.

    Built from the step-0 state of utils/detection.py; `export` gives back the same lists and arrays the Python loop
    would have produced (x_clust, y_clust, t_clust, ind_clust, steps, x_m, y_m, t_m, max_tracks).
    """

    def __init__(self, height, width, structure, second_pass, buffer, n_global,
                 steps, inactive, inactive_k, valid, x_m, y_m, t_m, active_pixels, ind_clust):
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
        self.touch = np.zeros(n, np.uint8)
        self.ind_dtype = np.int32 if n_global < 2 ** 31 else np.int64

        # clusters started at step 0 (inactive == 0): their history and active pixels
        filled = np.flatnonzero(self.inactive == 0)
        self.tid[filled] = np.arange(filled.size)
        self.next_tid = int(filled.size)
        lens = np.array([len(ind_clust[s]) for s in filled], dtype=np.int64)
        self.log_tid = [np.repeat(np.arange(filled.size, dtype=np.int32), lens)]
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
        self.thresh = float(min(3, buffer))
        self.cand = np.zeros(n, np.int64)
        c = np.flatnonzero((self.inactive == 0) & (self.steps < self.thresh))
        self.cand[:c.size] = c
        self.cand_n = int(c.size)
        self.heap = np.zeros(n, np.int64)
        h = np.flatnonzero(self.inactive == 1)  # ascending: already a min-heap
        self.heap[:h.size] = h
        self.heap_n = int(h.size)

        self.labelmap = np.full((height, width), -1, dtype=np.int32)
        self.pixmap = np.zeros((height, width), dtype=np.int32)  # scratch of label_events
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
            self.touch = grown(self.touch, 0)
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

    def step(self, k, xs, ys, labels_xy, num_regions, X, Y, T, ind_global):
        """One time step k > 0. xs, ys: 0-based event pixels; labels_xy: kept region of every event (0 = none)."""
        order, off = _group(labels_xy, int(num_regions))
        n = xs.size
        nap_x = np.empty(n, np.int32)
        nap_y = np.empty(n, np.int32)
        ent_slot = np.empty(n, np.int64)
        ent_off = np.empty(n, np.int64)
        ent_len = np.empty(n, np.int64)
        log_tid = np.empty(n, np.int32)
        log_ind = np.empty(n, self.ind_dtype)
        newlabs = np.empty(int(num_regions) + 1, np.int64)
        nap, nent, nlog, nn = _match(
            k, self.dilate, self.offs, xs, ys, X, Y, T, ind_global, order, off, self.labelmap, self.vlist,
            self.ap_off, self.ap_len, self.ap_x, self.ap_y, self.steps, self.inactive_k,
            self.x_m, self.y_m, self.t_m, self.tid, self.mark, self.touch,
            nap_x, nap_y, ent_slot, ent_off, ent_len, log_tid, log_ind, newlabs, 2.0)
        if nn > 0:
            missing = int(newlabs[nn - 1]) - self.heap_n  # needed = max(idx), the largest new label
            if missing > 0:
                self._grow(missing)
            self.heap_n, self.next_tid, nap, nent, nlog, self.cand_n = _create(
                k, newlabs, nn, order, off, xs, ys, X, Y, T, ind_global, self.heap, self.heap_n,
                self.steps, self.inactive, self.inactive_k, self.x_m, self.y_m, self.t_m, self.tid,
                self.next_tid, nap_x, nap_y, nap, ent_slot, ent_off, ent_len, nent, log_tid, log_ind, nlog,
                self.cand, self.cand_n)
        self.cand_n, self.heap_n = _release(k, self.thresh, self.cand, self.cand_n, self.steps, self.inactive,
                                            self.inactive_k, self.heap, self.heap_n)
        # matched and new clusters are the valid ones of the next step, with their new active pixels
        slots = ent_slot[:nent]
        self.ap_off[slots] = ent_off[:nent]
        self.ap_len[slots] = ent_len[:nent]
        self.ap_x = nap_x
        self.ap_y = nap_y
        self.vlist = np.sort(slots)
        if nlog:
            self.log_tid.append(log_tid[:nlog].copy())
            self.log_ind.append(log_ind[:nlog].copy())

    def export(self, X_global, Y_global, T_global):
        """Per-slot cluster lists and arrays, as the Python loop leaves them at the end of the pass."""
        n = self.n
        log_tid = np.concatenate(self.log_tid)
        log_ind = np.concatenate(self.log_ind)
        self.log_tid = self.log_ind = None
        ind, off = _assemble(log_tid, log_ind, self.tid[:n])
        del log_tid, log_ind
        x_all = X_global[ind] + 1
        y_all = Y_global[ind] + 1
        t_all = T_global[ind]
        x_clust, y_clust, t_clust, ind_clust = [], [], [], []
        live = self.tid[:n] >= 0
        for s in range(n):
            if live[s]:
                a, b = off[s], off[s + 1]
                x_clust.append(x_all[a:b])
                y_clust.append(y_all[a:b])
                t_clust.append(t_all[a:b])
                ind_clust.append(ind[a:b])
            else:  # never used slot (the original holds [] or an empty float array there)
                x_clust.append(_EMPTY)
                y_clust.append(_EMPTY)
                t_clust.append(_EMPTY)
                ind_clust.append(_EMPTY)
        dtype = self.steps_dtype_extended if self.extended else self.steps_dtype
        return dict(x_clust=x_clust, y_clust=y_clust, t_clust=t_clust, ind_clust=ind_clust,
                    steps=self.steps[:n].astype(dtype), x_m=self.x_m[:n].copy(), y_m=self.y_m[:n].copy(),
                    t_m=self.t_m[:n].copy(), max_tracks=n)
