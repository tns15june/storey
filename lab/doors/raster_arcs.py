"""Door arcs on any raster plan: ring-quadrant template matching on the thin ink.
shortcut: axis-aligned quadrants only, so rotated wings need a rotated template; upgrade by adding angles."""
import numpy as np, cv2
from scipy.signal import fftconvolve


def thin_ink(bgr):
    """Thin dark strokes on any background: black-hat, so hairlines and colour fills both work."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    return (bh > 35).astype(np.uint8)


def ring_quadrant(r, q, w=1.5):
    s = int(r + 3)
    y, x = np.mgrid[-s:s + 1, -s:s + 1]
    d = np.hypot(x, y)
    sx, sy = (1, 1) if q == 0 else (-1, 1) if q == 1 else (-1, -1) if q == 2 else (1, -1)
    m = (np.abs(d - r) <= w) & (x * sx > r * 0.12) & (y * sy > r * 0.12)
    return m.astype(np.float32)


def line_cover(ink, x, y, dx, dy, r):
    """Fraction of the axis line from (x, y) in direction (dx, dy) with ink within 2 px across it."""
    H, W = ink.shape
    n = hit = 0
    for t in range(int(r * 0.15), int(r * 0.95)):
        px, py = int(x + dx * t), int(y + dy * t)
        n += 1
        y0, y1, x0, x1 = max(py - 3, 0), py + 4, max(px - 3, 0), px + 4
        hit += ink[y0:y1, x0:x1].any()
    return hit / max(n, 1)


def features(a, ink):
    """Angular coverage of the arc, leaf and gap on the two radii, and how empty the swept sector is."""
    sx, sy = ((1, 1), (-1, 1), (-1, -1), (1, -1))[a["q"]]
    x, y, r = a["x"], a["y"], a["r"]
    H, W = ink.shape
    hit = 0
    n = 32
    for t in np.linspace(0.05, 1.5, n):
        px, py = int(x + sx * r * np.cos(t)), int(y + sy * r * np.sin(t))
        hit += ink[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3].any()
    a["arc_cov"] = hit / n
    a["leaf_h"], a["leaf_v"] = line_cover(ink, x, y, sx, 0, r), line_cover(ink, x, y, 0, sy, r)
    i0 = int(r * 0.25)
    i1 = int(r * 0.62)  # inside the arc on the diagonal square
    xs = sorted((int(x + sx * i0), int(x + sx * i1)))
    ys = sorted((int(y + sy * i0), int(y + sy * i1)))
    box = ink[max(ys[0], 0):max(ys[1], 1), max(xs[0], 0):max(xs[1], 1)]
    a["sector_ink"] = float(box.mean()) if box.size else 1.0
    return a


def door_like(a):
    h, v = a["leaf_h"], a["leaf_v"]
    return a["arc_cov"] >= 0.8 and max(h, v) >= 0.7 and min(h, v) <= 0.5 and a["sector_ink"] <= 0.12


def find(bgr, rmin, rmax, thr=0.62, step=1.0, gate=True):
    """Returns dicts with centre (x, y), radius r, quadrant q and score, in pixels. Quadrant gives the swept corner."""
    base = thin_ink(bgr)
    ink = cv2.dilate(base, np.ones((2, 2), np.uint8)).astype(np.float32)
    H, W = ink.shape
    hits = []
    for r in np.arange(rmin, rmax + step, step):
        for q in range(4):
            t = ring_quadrant(r, q)
            sc = fftconvolve(ink, t[::-1, ::-1], mode="same") / t.sum()
            ys, xs = np.where(sc > thr)
            for x, y in zip(xs, ys):
                hits.append((float(sc[y, x]), int(x), int(y), float(r), q))
    hits.sort(reverse=True)
    grid, kept = {}, []
    cell = max(4, int(rmin * 0.3))
    for s, x, y, r, q in hits:
        k = (x // cell, y // cell, q)
        if any(((k[0] + i, k[1] + j, q) in grid) for i in (-1, 0, 1) for j in (-1, 0, 1)):
            continue
        grid[k] = 1
        kept.append({"x": x, "y": y, "r": r, "q": q, "score": s})
    kept = [features(a, base) for a in kept]
    return [a for a in kept if not gate or door_like(a)]


def arc_mid(a):
    d = a["r"] * 0.7071
    sx, sy = ((1, 1), (-1, 1), (-1, -1), (1, -1))[a["q"]]
    return a["x"] + sx * d, a["y"] + sy * d


if __name__ == "__main__":
    t = np.zeros((101, 101, 3), np.uint8) + 255
    cv2.ellipse(t, (20, 20), (40, 40), 0, 0, 90, (0, 0, 0), 1)
    f = find(t, 38, 42, thr=0.6, gate=False)
    assert f and f[0]["q"] == 0 and abs(f[0]["x"] - 20) <= 2, f
    print("ok", f[0])
