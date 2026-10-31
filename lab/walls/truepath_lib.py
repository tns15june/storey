import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric


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


