"""Classical wall / room / walkable extraction and geodesic distance. `u` = pixels per door width (0.9 m), the only scale input."""
import cv2
import numpy as np
cv2.setNumThreads(1)
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

SQ = lambda k: np.ones((max(1, int(k)), max(1, int(k))), np.uint8)
HL = lambda k: np.ones((1, max(1, int(k))), np.uint8)
VL = lambda k: np.ones((max(1, int(k)), 1), np.uint8)


def ink_mask(img, floor=135):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (g < min(max(t, 60), floor)).astype(np.uint8)  # floor keeps pale room tints out on coloured plans


def keep_big(m, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def walls(img, u, thick_k=0.09, line_k=1.25, net_k=1.2, floor=135, dbl_k=0.0):
    """thick strokes plus long straight strokes that touch the thick network."""
    ink = ink_mask(img, floor)
    if dbl_k:  # bridge double-line and hatched walls before the thickness test
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, SQ(dbl_k * u))
    k = max(3, int(round(thick_k * u)))
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, SQ(k))
    net = keep_big(thick, (net_k * u) * k * 1.0)
    L = int(line_k * u)
    # long strokes after removing the thick network: a sill line across a door gap is too short to survive, a thin wall is not
    rest = (cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8)) > 0) & ~(cv2.dilate(net.astype(np.uint8), SQ(5)) > 0)
    r8 = rest.astype(np.uint8)
    lines = cv2.morphologyEx(r8, cv2.MORPH_OPEN, HL(L)) | cv2.morphologyEx(r8, cv2.MORPH_OPEN, VL(L))
    both = (net | (lines > 0)).astype(np.uint8)
    n, lab = cv2.connectedComponents(cv2.dilate(both, SQ(7)), connectivity=8)
    touch = np.unique(lab[net])
    return np.isin(lab, touch[touch > 0]) & (both > 0)


def blockers(img, u, line_k=0.4, min_k=0.6, floor=170, thick_k=0.09):
    """Everything that stops a person, not just walls: straight strokes and thick ink, minus text and arcs.
    Furniture stays as an obstacle, which a route simply goes around."""
    ink = ink_mask(img, floor)
    L = max(5, int(line_k * u))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, HL(L)) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, VL(L))
    k = max(3, int(round(thick_k * u)))
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, SQ(k))
    m = lines | thick
    # text is many small components; walls are few big ones
    return keep_big(m, (min_k * u) ** 2 * 0.25)


def close_doors(w, u, max_gap=1.35):
    """Bridge collinear wall ends across door-sized gaps. Long runs only, so corridors between parallel walls stay open."""
    w8 = w.astype(np.uint8)
    run = int(0.35 * u)
    h = cv2.morphologyEx(w8, cv2.MORPH_OPEN, HL(run))
    v = cv2.morphologyEx(w8, cv2.MORPH_OPEN, VL(run))
    g = int(max_gap * u)
    hc = cv2.morphologyEx(h, cv2.MORPH_CLOSE, HL(g))
    vc = cv2.morphologyEx(v, cv2.MORPH_CLOSE, VL(g))
    return (w8 | hc | vc) > 0


def footprint(closed, u, fp_gap=2.6):
    """Everything not reachable from the sheet edge once windows and doors are bridged wide."""
    p = int(3 * u)  # padding: cv2 closing treats the image edge as ink otherwise
    c0 = cv2.copyMakeBorder(closed.astype(np.uint8), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    c = cv2.morphologyEx(c0, cv2.MORPH_CLOSE, HL(fp_gap * u)) | cv2.morphologyEx(c0, cv2.MORPH_CLOSE, VL(fp_gap * u))
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, SQ(0.12 * u))
    n, lab = cv2.connectedComponents(1 - c, connectivity=4)
    return (lab != lab[0, 0])[p:-p, p:-p]


def rooms(closed, fp, u):
    """Labels of free regions inside the footprint; openings are bridged so each room is its own component."""
    free = (fp & ~closed).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, SQ(0.1 * u))
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    keep = st[:, cv2.CC_STAT_AREA] >= (0.8 * u) ** 2
    keep[0] = False
    lab[~keep[lab]] = 0
    return lab


def walkable(w, fp, u, clear=0.25):
    """Free space inside the footprint, eroded by half a body so pinholes in a wall do not leak."""
    r = max(1, int(clear * u * 0.5))
    free = (fp & ~w).astype(np.uint8)
    return cv2.erode(free, SQ(2 * r + 1)) > 0


def geodesic(walk, seeds_xy, scale=0.5):
    """Distance in original pixels from seed points over `walk`. Seeds snap to the nearest walkable pixel."""
    small = cv2.resize(walk.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
    if not small.any() or not len(seeds_xy):
        return np.full(walk.shape, np.inf)
    idx = ndi.distance_transform_edt(~small, return_distances=False, return_indices=True)
    starts = set()
    for x, y in seeds_xy:
        yy, xx = min(int(y * scale), small.shape[0] - 1), min(int(x * scale), small.shape[1] - 1)
        starts.add((int(idx[0, yy, xx]), int(idx[1, yy, xx])))
    cost = np.where(small, 1.0, -1.0)
    d, _ = MCP_Geometric(cost, fully_connected=True).find_costs(list(starts))
    d = np.where(small & np.isfinite(d), d / scale, np.inf)
    return cv2.resize(d, (walk.shape[1], walk.shape[0]), interpolation=cv2.INTER_NEAREST)


def _near(wd, x, y, _):
    y, x = int(y), int(x)
    return 0 <= y < wd.shape[0] and 0 <= x < wd.shape[1] and wd[y, x] > 0


def _door_templates(r):
    """8 variants of a door symbol anchored at the hinge: quarter arc plus leaf. Returns (arc kernel, leaf kernel, leaf dir, open dir)."""
    out = []
    n = int(r) + 3
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i in range(4):
        for turn in (1, -1):
            a = np.array(dirs[i], float)                 # leaf direction
            b = np.array(dirs[(i + turn) % 4], float)    # opening direction
            arc = np.zeros((2 * n + 1, 2 * n + 1), np.float32)
            leaf = arc.copy()
            for t in np.linspace(0.06, 0.94, 60) * np.pi / 2:
                p = np.round(n + r * (np.cos(t) * a + np.sin(t) * b)).astype(int)
                arc[p[1], p[0]] = 1
            for t in np.linspace(0.15, 0.95, 30):
                p = np.round(n + r * t * a).astype(int)
                leaf[p[1], p[0]] = 1
            fill = np.zeros_like(arc)
            for ri in np.linspace(0.25, 0.8, 8):
                for t in np.linspace(0.1, 0.9, 12) * np.pi / 2:
                    p = np.round(n + r * ri * (np.cos(t) * a + np.sin(t) * b)).astype(int)
                    fill[p[1], p[0]] = 1
            ring = np.zeros_like(arc)
            for ri in (0.8, 1.18):  # an arc is an isolated curve: both sides of it are clean paper
                for t in np.linspace(0.12, 0.88, 40) * np.pi / 2:
                    p = np.round(n + r * ri * (np.cos(t) * a + np.sin(t) * b)).astype(int)
                    if abs(p - n).max() <= n:
                        ring[p[1], p[0]] = 1
            out.append((arc / arc.sum(), leaf / leaf.sum(), fill / fill.sum(), ring / ring.sum(), a, b, n))
    return out


def find_doors(img, u, radii=(0.6, 0.72, 0.85, 1.0, 1.15, 1.3), arc_min=0.8, leaf_min=0.55, floor=135, near=1.6, fill_max=0.25, ring_max=0.3, wall=None, wall_near=0.3):
    """Chamfer match of quarter-arc door symbols on ink. Returns [(hinge xy, open-end xy, leaf-end xy, score)] in input pixels."""
    sc = min(1.0, 40.0 / u)  # work at about 40 px per door
    ink = ink_mask(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), max(floor, 200))
    nearby = (cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3) <= near).astype(np.float32)
    # distance to wall, used to require hinge and open end on a wall
    wd = None
    if wall is not None:
        ws = cv2.resize(wall.astype(np.uint8), (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)
        wd = cv2.dilate(ws, SQ(2 * wall_near * u * sc + 1))
    hits = []
    for rr in radii:
        r = rr * u * sc
        for arc, leaf, fill, ring, a, b, n in _door_templates(r):
            # filter2D correlates with the kernel centred on the anchor, so the response peaks at the hinge
            A = cv2.filter2D(nearby, cv2.CV_32F, arc, borderType=cv2.BORDER_CONSTANT)
            if A.max() < arc_min:
                continue
            L = cv2.filter2D(nearby, cv2.CV_32F, leaf, borderType=cv2.BORDER_CONSTANT)
            m = ((A >= arc_min) & (L >= leaf_min)).astype(np.uint8)
            if not m.any():
                continue
            # one candidate per blob: the blob's best pixel, and hatching or text inside the swing rejects it
            F = cv2.filter2D(nearby, cv2.CV_32F, fill, borderType=cv2.BORDER_CONSTANT)
            R = cv2.filter2D(nearby, cv2.CV_32F, ring, borderType=cv2.BORDER_CONSTANT)
            nb, lab = cv2.connectedComponents(m, connectivity=8)
            S = (A + L) * m
            best = ndi.maximum_position(S, lab, range(1, nb + 1))
            for y, x in best:
                if F[y, x] <= fill_max and R[y, x] <= ring_max and (wd is None or (_near(wd, x, y, 0) and _near(wd, x + r * b[0], y + r * b[1], 0))):
                    hits.append((float(S[y, x]), int(x), int(y), r, a, b))
    hits.sort(key=lambda h: -h[0])
    keep = []
    for s, x, y, r, a, b in hits:  # non-maximum suppression on the arc middle, so the two arcs of a double door both live
        m = np.array([x, y]) + r * 0.7 * (a + b)
        if all(np.hypot(*(m - k[0])) > 0.35 * r for k in keep):
            keep.append((m, x, y, r, a, b, s))
    return [(np.array([x, y]) / sc, (np.array([x, y]) + r * b) / sc, (np.array([x, y]) + r * a) / sc, s) for m, x, y, r, a, b, s in keep]


def carve(w, doors, u, half=0.32):
    """Open each detected door through the blocking mask. Returns (blocked mask with doors open, door centre lines)."""
    out = w.astype(np.uint8).copy()
    bar = np.zeros(w.shape, np.uint8)
    for H, E, Lf, s in doors:
        d = (E - H) / max(1e-6, np.hypot(*(E - H)))
        n = (Lf - H) / max(1e-6, np.hypot(*(Lf - H)))
        t = half * u
        q = np.array([H + d * 3 - n * t, H + d * 3 + n * t, E - d * 3 + n * t, E - d * 3 - n * t]).round().astype(np.int32)
        cv2.fillConvexPoly(out, q, 0)
        cv2.line(out, tuple(np.round(H + n * t).astype(int).tolist()), tuple(np.round(Lf).astype(int).tolist()), 0, 5)  # the leaf itself
        cv2.line(bar, tuple(np.round(H).astype(int).tolist()), tuple(np.round(E).astype(int).tolist()), 1, 3)
    return out > 0, bar > 0


def join_rooms(w, fp, u, cand, clear=0.25, strict=2.2):
    """Accept a door candidate only if it joins two separate free regions (maximum spanning forest by score).
    A false arc on furniture joins nothing, so it does no harm. Returns accepted candidates."""
    wk = walkable(w, fp, u, clear)
    n, lab = cv2.connectedComponents(wk.astype(np.uint8), connectivity=4)
    idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def side(p):
        x, y = int(np.clip(p[0], 0, w.shape[1] - 1)), int(np.clip(p[1], 0, w.shape[0] - 1))
        yy, xx = idx[0, y, x], idx[1, y, x]
        return int(lab[yy, xx]) if np.hypot(yy - y, xx - x) < 0.3 * u else 0

    acc = []
    for q in sorted(cand, key=lambda q: -q[3]):
        H, E, Lf, s = q
        n_ = (Lf - H) / max(1e-6, np.hypot(*(Lf - H)))
        mid = (H + E) / 2
        a, b = side(mid + n_ * 0.45 * u), side(mid - n_ * 0.45 * u)
        if a and b and find(a) != find(b):
            parent[find(a)] = find(b)
            acc.append(q)
    return acc


if __name__ == '__main__':
    # self-check: two rooms joined by a door gap, path must go through it
    im = np.full((300, 500, 3), 255, np.uint8)
    cv2.rectangle(im, (20, 20), (480, 280), (0, 0, 0), 8)
    cv2.line(im, (250, 20), (250, 120), (0, 0, 0), 8)
    cv2.line(im, (250, 190), (250, 280), (0, 0, 0), 8)
    u = 70
    w = walls(im, u)
    c = close_doors(w, u)
    fp = footprint(c, u)
    lab = rooms(c, fp, u)
    assert len(np.unique(lab)) == 3, np.unique(lab)
    wk = walkable(w, fp, u)
    d = geodesic(wk, [(100, 150)])
    assert np.isfinite(d[150, 400]) and 290 < d[150, 400] < 340, d[150, 400]
    # door symbol found at the right place
    im2 = im.copy()
    cv2.line(im2, (250, 120), (320, 120), (0, 0, 0), 2)
    cv2.ellipse(im2, (250, 120), (70, 70), 0, 0, 90, (0, 0, 0), 2)
    ds = find_doors(im2, 70)
    assert len(ds) >= 1 and np.hypot(*(ds[0][0] - (250, 120))) < 6 and np.hypot(*(ds[0][1] - (250, 190))) < 6, ds
    print('ok')
