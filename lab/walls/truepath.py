"""Architect's drawn travel paths on the life safety sheets against geodesic routes on the extracted walkable mask."""
import json, sys
sys.path.insert(0, 'lab/walls')
import cv2, fitz, numpy as np, vec, clinic_const as K
from skimage.graph import MCP_Geometric
from scipy import ndimage as ndi
O = K.O


def taut(walk, path):
    """Any-angle length: greedy line-of-sight shortcut along the pixel path."""
    pts = [path[0]]; i = 0; n = len(path)
    def los(a, b):
        L = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) + 1
        ys = np.linspace(a[0], b[0], L).round().astype(int); xs = np.linspace(a[1], b[1], L).round().astype(int)
        return walk[ys, xs].all()
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not los(path[i], path[j]):
            j = max(i + 1, i + (j - i) // 2) if (j - i) > 64 else j - 1
        pts.append(path[j]); i = j
    return pts, sum(np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1))


def route(walk, s, e):
    def snap(p):
        idx = ndi.distance_transform_edt(~walk, return_distances=False, return_indices=True)
        y, x = int(p[1]), int(p[0])
        return (int(idx[0, y, x]), int(idx[1, y, x]))
    a, b = snap(s), snap(e)
    mcp = MCP_Geometric(np.where(walk, 1.0, -1.0), fully_connected=True)
    c, _ = mcp.find_costs([a], [b])
    if not np.isfinite(c[b]):
        return None, None, None, (a, b)
    p = mcp.traceback(b)
    pts, t = taut(walk, p)
    return float(c[b]), float(t), pts, (a, b)


def offs(f1, f2):
    a = np.array([x[0] for x in vec.door_arcs(fitz.open(K.D + f1)[0], K.PPM)]); b = np.array([x[0] for x in vec.door_arcs(fitz.open(K.D + f2)[0], K.PPM)])
    d = np.hypot(*(a[:, None] - b[None]).transpose(2, 0, 1))
    return np.median(b[d.argmin(1)] - a, axis=0)


CASES = {1: dict(stated=41.75, pts=[(1552, 989), (1465, 923), (1463, 762), (1946, 747)]),
         2: dict(stated=46.65, pts=[(1870, 850), (1793, 752), (1663, 730), (1625, 765), (1600, 823), (1560, 833), (1510, 1047), (1523, 1253)])}
out = {}
for fl, c in CASES.items():
    f, fls = K.SHEETS[fl]
    off = offs(f, fls)
    z = np.load(O + f'clinic_f{fl}.npz')
    walk = z['walk']
    p = (np.array(c['pts'], float) - off - np.array(K.BOX[:2])) * K.SC
    drawn = float(np.hypot(*np.diff(np.array(c['pts'], float), axis=0).T).sum() / K.PPM)
    g8, t, pts, snaps = route(walk, p[0], p[-1])
    gt_g8, gt_t, gt_pts, _ = route(z['gt'], p[0], p[-1])
    r = dict(offset=off.tolist(), stated_m=c['stated'], drawn_polyline_m=round(drawn, 2), mine_m=None if t is None else round(t / K.PXM, 2), mine_8conn_m=None if g8 is None else round(g8 / K.PXM, 2),
             ifc_truth_mask_m=None if gt_t is None else round(gt_t / K.PXM, 2))
    if t is not None:
        r['err_vs_drawn'] = round((t / K.PXM - drawn) / drawn, 3); r['err_vs_stated'] = round((t / K.PXM - c['stated']) / c['stated'], 3)
    if gt_t is not None and t is not None:
        r['err_vs_ifc'] = round((t - gt_t) / gt_t, 3)
    out[fl] = r
    print(fl, json.dumps(r))
    v = np.full(walk.shape + (3,), 255, np.uint8); v[walk] = (205, 240, 205); v[z['solid']] = (50, 50, 50)
    cv2.polylines(v, [np.round(p).astype(np.int32)], False, (0, 0, 255), 3)
    if pts is not None:
        cv2.polylines(v, [np.array([(q[1], q[0]) for q in pts], np.int32)], False, (255, 0, 0), 3)
    if gt_pts is not None:
        cv2.polylines(v, [np.array([(q[1], q[0]) for q in gt_pts], np.int32)], False, (0, 180, 0), 2)
    ys, xs = np.nonzero(z['solid']); v2 = v[ys.min():ys.max(), xs.min():xs.max()]
    cv2.imwrite(O + f'truepath_f{fl}.jpg', cv2.resize(v2, None, fx=.55, fy=.55, interpolation=cv2.INTER_AREA))
json.dump(out, open(O + 'truepath.json', 'w'), indent=1)
