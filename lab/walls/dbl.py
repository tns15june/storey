"""Wall segments from any pen: long segments with a parallel partner at wall thickness."""
import sys
sys.path.insert(0, '/Users/tsharma/Github/Meta/lab/walls')
import numpy as np


def wall_segments(segs, ppm, tmin=0.07, tmax=0.45, lmin=0.5, overlap=0.5):
    """segs: Nx4 in points. Returns a bool per segment: long and paired with a parallel overlapping segment tmin..tmax metres away."""
    d = segs[:, 2:] - segs[:, :2]
    L = np.hypot(*d.T)
    i = np.nonzero(L >= lmin * ppm)[0]
    s = segs[i]; dd = d[i] / L[i, None]; LL = L[i]
    n = len(s)
    ok = np.zeros(n, bool)
    th = np.round(np.degrees(np.arctan2(dd[:, 1], dd[:, 0])) % 180, 0) % 180
    for t in np.unique(th):
        J = np.nonzero(np.abs((th - t + 90) % 180 - 90) <= 1)[0]
        if len(J) < 2:
            continue
        u = np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]); v = np.array([-u[1], u[0]])
        off = s[J, :2] @ v
        a0, a1 = s[J, :2] @ u, s[J, 2:] @ u
        lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
        order = np.argsort(off)
        off, lo, hi, JJ = off[order], lo[order], hi[order], J[order]
        for k in range(len(JJ)):
            r = np.searchsorted(off, [off[k] + tmin * ppm, off[k] + tmax * ppm])
            if r[1] <= r[0]:
                continue
            ov = np.minimum(hi[k], hi[r[0]:r[1]]) - np.maximum(lo[k], lo[r[0]:r[1]])
            good = ov >= overlap * np.minimum(hi[k] - lo[k], hi[r[0]:r[1]] - lo[r[0]:r[1]])
            if good.any():
                ok[JJ[k]] = True
                ok[JJ[r[0]:r[1]][good]] = True
    out = np.zeros(len(segs), bool)
    out[i[ok]] = True
    return out


if __name__ == '__main__':
    a = np.array([[0, 0, 100, 0], [0, 8, 100, 8], [0, 200, 100, 200], [0, 0, 0, 3.]])
    r = wall_segments(a, 40)
    assert r.tolist() == [True, True, False, False], r
    print('ok')
