"""ticks_on from lab/modern/clinic.py, copied because importing that module opens a metalab client and this code needs no API."""
import numpy as np


def ticks_on(a, L, horiz):
    """Positions along a dimension line where a short slash crosses it."""
    d = a[:, 2:4] - a[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    s = a[(ln > 4) & (ln < 16) & (np.abs(d[:, 0]) > 1) & (np.abs(d[:, 1]) > 1)]
    if horiz:
        y = L[1]
        t = s[(np.minimum(s[:, 1], s[:, 3]) <= y) & (y <= np.maximum(s[:, 1], s[:, 3]))]
        return np.unique(np.round((t[:, 0] + t[:, 2]) / 2, 1))
    x = L[0]
    t = s[(np.minimum(s[:, 0], s[:, 2]) <= x) & (x <= np.maximum(s[:, 0], s[:, 2]))]
    return np.unique(np.round((t[:, 1] + t[:, 3]) / 2, 1))
