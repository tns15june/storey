"""Clinic routes: same router as lab/walls, plus a check of a polyline against the IFC truth mask. No API."""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/walls")
import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
import clinic_const as K
from truepath_lib import taut

SHEET = {1: K.SHEETS[1][0], 2: K.SHEETS[2][0]}


def load(fl):
    z = np.load(K.O + f"clinic_f{fl}.npz")
    return {k: z[k] for k in ("walk", "gt", "solid", "lab")}


def rooms_of(lab, walk):
    """Room label -> deepest walkable point (x, y). Room names sit at room centres and are solid in the walk mask."""
    out = {}
    for i in np.unique(lab):
        if i == 0:
            continue
        mk = ((lab == i) & walk).astype(np.uint8)
        if mk.sum() < 0.5 * K.PXM ** 2:
            continue
        ys, xs = np.nonzero(mk)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        dt = cv2.distanceTransform(mk[y0:y1, x0:x1], cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(dt.argmax()), dt.shape)
        out[int(i)] = (int(x + x0), int(y + y0))
    return out


def snapper(walk):
    idx = ndi.distance_transform_edt(~walk, return_distances=False, return_indices=True)
    return lambda p: (int(idx[0, int(p[1]), int(p[0])]), int(idx[1, int(p[1]), int(p[0])]))


def path_to(walk, snap, s, ends):
    """Shortest route from (x, y) to the nearest of `ends` [(x, y)]. Returns (metres, [(y, x), ...]) or None."""
    a = snap(s)
    bs = list({snap(e) for e in ends})
    mcp = MCP_Geometric(np.where(walk, 1.0, -1.0), fully_connected=True)
    c, _ = mcp.find_costs([a], bs)
    b = min(bs, key=lambda q: c[q])
    if not np.isfinite(c[b]):
        return None
    pts, t = taut(walk, mcp.traceback(b))
    return t / K.PXM, pts


def sample(pts, step=1.0):
    out = []
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        n = max(int(np.hypot(y1 - y0, x1 - x0) / step), 1)
        out += [(y0 + (y1 - y0) * t / n, x0 + (x1 - x0) * t / n) for t in range(n)]
    out.append(pts[-1])
    return np.round(np.array(out)).astype(int)


def crossings(pts, gt, tol=3, min_run=8):
    """Runs of the polyline that leave the IFC walkable mask by more than `tol` px for at least `min_run` px (2.5 cm per px). Returns [(y, x, run_px)]."""
    free = cv2.dilate(gt.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    s = sample(pts)
    off = ~free[s[:, 0], s[:, 1]]
    lab, n = ndi.label(off)
    out = []
    for i in range(1, n + 1):
        k = np.nonzero(lab == i)[0]
        if len(k) >= min_run:
            m = k[len(k) // 2]
            out.append((int(s[m, 0]), int(s[m, 1]), len(k)))
    return out


if __name__ == "__main__":
    g = np.ones((40, 40), bool); g[:, 20:23] = False
    p = [(5, 5), (5, 35)]
    assert crossings(p, g, tol=0, min_run=2) and not crossings([(5, 5), (30, 5)], g, tol=0, min_run=2)
    print("ok")
