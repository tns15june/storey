"""Register IFC metres to sheet points: vote a translation for each 90 degree rotation and flip, then fit an affine."""
import numpy as np
from scipy.optimize import linear_sum_assignment

SCALE = 72 / 25.4 * 1000 / 150  # pt per metre at 1:150


def fit(ifc_xy, sheet_xy):
    best = (0, None)
    for k in range(4):
        th = k * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * SCALE
        for F in ([[1, 0], [0, -1]], [[1, 0], [0, 1]]):
            M = R @ np.array(F, float)
            pi = ifc_xy @ M.T
            diff = (sheet_xy[None, :, :] - pi[:, None, :]).reshape(-1, 2)
            u, c = np.unique(np.round(diff / 2).astype(int), axis=0, return_counts=True)
            i = c.argmax()
            if c[i] > best[0]:
                best = (c[i], (M.copy(), u[i] * 2.0))
    M, t = best[1]
    A = np.c_[M, t]
    for tol in (6, 4, 3, 2.5):
        p = ifc_xy @ A[:2, :2].T + A[:2, 2]
        D = np.linalg.norm(p[:, None] - sheet_xy[None], axis=2)
        r, c = linear_sum_assignment(np.minimum(D, 1e3))
        ok = D[r, c] < tol
        X = np.c_[ifc_xy[r[ok]], np.ones(ok.sum())]
        A = np.linalg.lstsq(X, sheet_xy[c[ok]], rcond=None)[0].T
    return A, int(ok.sum()), int(best[0]), float(np.median(D[r, c][ok]))


def apply(A, xy):
    return np.asarray(xy) @ A[:2, :2].T + A[:2, 2]
