"""Door swing arcs from a vector PDF page: chain strokes, split at corners, fit circles, find the leaf line."""
import math, sys
from collections import defaultdict
import numpy as np


def page_segments(page):
    """All stroke segments, Beziers sampled, with the source path index and clip off."""
    polys = []
    for di, d in enumerate(page.get_drawings()):
        cur = []
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                polys.append((a.x, a.y, b.x, b.y, di))
            elif it[0] == "c":
                p = [np.array([q.x, q.y]) for q in it[1:5]]
                ts = np.linspace(0, 1, 9)
                pts = [(1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3] for t in ts]
                for u, v in zip(pts, pts[1:]):
                    polys.append((u[0], u[1], v[0], v[1], di))
    return np.array(polys, float)


def chains(segs, tol=0.13):
    """Link segments that share endpoints into polylines. Junctions (degree not 2) end a chain."""
    key = lambda x, y: (round(x / tol), round(y / tol))
    adj = defaultdict(list)
    for i, s in enumerate(segs):
        if (s[0], s[1]) == (s[2], s[3]):
            continue
        adj[key(s[0], s[1])].append((i, 0))
        adj[key(s[2], s[3])].append((i, 1))
    used = np.zeros(len(segs), bool)
    out = []

    def walk(i, end):
        pts = [segs[i][:2] if end == 1 else segs[i][2:4], segs[i][2:4] if end == 1 else segs[i][:2]]
        used[i] = True
        while True:
            k = key(*pts[-1])
            nx = [(j, e) for j, e in adj[k] if not used[j]]
            if len(adj[k]) != 2 or not nx:
                return pts
            j, e = nx[0]
            used[j] = True
            pts.append(segs[j][2:4] if e == 0 else segs[j][:2])

    # start from chain ends so a chain is walked once
    for i in np.argsort([-(len(adj[key(s[0], s[1])]) != 2) for s in segs]):
        if used[i]:
            continue
        p = walk(i, 1)
        q = walk_back = None
        if len(adj[key(segs[i][0], segs[i][1])]) == 2:
            q = walk_prepend(i, segs, adj, used, key)
            p = q[::-1] + p
        out.append(np.array(p))
    return out


def walk_prepend(i, segs, adj, used, key):
    pts = []
    k = key(segs[i][0], segs[i][1])
    while True:
        nx = [(j, e) for j, e in adj[k] if not used[j]]
        if len(adj[k]) != 2 or not nx:
            return pts
        j, e = nx[0]
        used[j] = True
        p = segs[j][2:4] if e == 0 else segs[j][:2]
        pts.append(p)
        k = key(*p)


def split_corners(pl, deg=40):
    if len(pl) < 3:
        return [pl]
    v = np.diff(pl, axis=0)
    a = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn = np.abs(np.diff(a))
    cuts = [0] + [i + 1 for i, t in enumerate(turn) if t > math.radians(deg)] + [len(pl) - 1]
    return [pl[a_:b_ + 1] for a_, b_ in zip(cuts, cuts[1:]) if b_ > a_]


def fit_circle(p):
    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = sol[:2]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0))
    res = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
    return cx, cy, r, res


def find_arcs(segs, rmin, rmax, smin=55, smax=115):
    """Quarter-ish circle arcs with radius in [rmin, rmax]. Arcs get cut at every line touching them, so
    fragments on one circle are merged by centre and radius before the sweep is judged."""
    frags = []
    for pl in chains(segs):
        for part in split_corners(pl):
            if len(part) < 3:
                continue
            cx, cy, r, res = fit_circle(part)
            if not (rmin * 0.9 <= r <= rmax * 1.1) or res > max(0.25, 0.015 * r):
                continue
            ang = np.unwrap(np.arctan2(part[:, 1] - cy, part[:, 0] - cx))
            if abs(ang[-1] - ang[0]) < math.radians(9):
                continue
            frags.append((cx, cy, r, res, part))
    groups = []
    for f in sorted(frags, key=lambda q: -len(q[4])):
        for g in groups:
            if math.hypot(f[0] - g[0][0], f[1] - g[0][1]) < max(1.2, 0.1 * f[2]) and abs(f[2] - g[0][2]) < max(0.8, 0.06 * f[2]):
                g.append(f)
                break
        else:
            groups.append([f])
    arcs = []
    for g in groups:
        pts = np.vstack([q[4] for q in g])
        cx, cy, r, res = fit_circle(pts)
        if not (rmin <= r <= rmax) or res > max(0.4, 0.025 * r):
            continue
        ang = np.sort(np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx), 2 * math.pi))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
        i = gaps.argmax()
        sweep = 2 * math.pi - gaps[i]
        if not (math.radians(smin) <= sweep <= math.radians(smax)):
            continue
        a0, a1 = ang[(i + 1) % len(ang)], ang[i]
        end = lambda t: [cx + r * math.cos(t), cy + r * math.sin(t)]
        arcs.append({"cx": cx, "cy": cy, "r": r, "sweep": math.degrees(sweep), "res": res, "a": end(a0), "b": end(a1), "frags": len(g)})
    return arcs


def cover(s, c, d, length, t):
    """Length of strokes running along c + u*d for u in [0, length], within t of that line."""
    u = d / np.linalg.norm(d)
    n = np.array([-u[1], u[0]])
    r0, r1 = s[:, :2] - c, s[:, 2:4] - c
    off = np.maximum(np.abs(r0 @ n), np.abs(r1 @ n))
    t0, t1 = r0 @ u, r1 @ u
    lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
    ok = (off < t) & (hi > 0) & (lo < length)
    return float((np.minimum(hi[ok], length) - np.maximum(lo[ok], 0)).sum()) if ok.any() else 0.0


def attach_leaf(arcs, segs):
    """Leaf: a stroke from the arc centre to one end. Shut end: the other end, whose chord lies in the wall line,
    so the wall continues behind the hinge. A threshold line across the opening must not read as a leaf."""
    s = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) > 0]
    for A in arcs:
        c = np.array([A["cx"], A["cy"]])
        t = max(1.2, 0.14 * A["r"])
        a, b = np.array(A["a"]), np.array(A["b"])
        la, lb = cover(s, c, a - c, A["r"], t) / A["r"], cover(s, c, b - c, A["r"], t) / A["r"]
        wa, wb = cover(s, c, c - b, A["r"] * .6, t) / A["r"], cover(s, c, c - a, A["r"] * .6, t) / A["r"]
        A["leaf_cover"] = max(la, lb)
        A["wall_cover"] = max(wa, wb)
        A["leaf"] = None
        if A["leaf_cover"] >= 0.6:
            A["leaf"] = "a" if la + wa >= lb + wb else "b"
        A["open_end"], A["shut_end"] = (A["a"], A["b"]) if (la + wa >= lb + wb) else (A["b"], A["a"])
    return arcs
