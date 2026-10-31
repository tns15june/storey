import numpy as np


def cluster(arcs, pt_per_m):
    """Drop arcs in title text: a real door has other doors within 12 m."""
    c = np.array([[a["cx"], a["cy"]] for a in arcs])
    D = np.linalg.norm(c[:, None] - c[None], axis=2)
    return [a for a, n in zip(arcs, (D < 12 * pt_per_m).sum(1)) if n >= 4]
