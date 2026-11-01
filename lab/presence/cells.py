"""Steps 3b and 4 without any model call: turn 3 runs into rows and cells.
A row is a place, not a name. Every present point from every run is mapped to the reference floor and clustered by distance, so a run that glues two places under one name is split, and a run that says nothing at a place is a no vote there.
Points on an unaligned floor cannot be placed, so they follow the row that holds the rest of their entity. An entity seen only on unaligned floors becomes its own row."""
import json
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, "lab/presence")
import pres
from pres import OUT, SH, reg, apply_pt

R = 80  # px on a 2000 px sheet, cluster radius: stairs sit 90 px or more apart, Spark's points scatter less than that on the dev set
HALF = 50  # half side of the fixed ink box
FAM = {"stairs": "stair", "shafts": "shaft"}


def floor_index(name, k, n, order):
    m_ = re.search(r"(\d+)", str(name))
    if m_ and 1 <= int(m_.group(1)) <= n and str(name).strip().upper().startswith("F"):
        return order[int(m_.group(1)) - 1]
    return order[k] if k < n else None  # shortcut: falls back to array position, only reached if a run ignores the codes


def points(b, key, runs=(0, 1, 2)):
    """One row per present claim: run, family, entity id, sheet floor, raw px, ref px or None."""
    A = pres.alignment(b)
    n = pres.nfloors(b)
    pts, usd, secs = [], 0.0, []
    for r in runs:
        d = json.load(open(OUT / f"spark/{key}_r{r}.json"))
        usd += d["info"]["usd"]
        secs.append(d["info"]["seconds"])
        for lst, f_ in FAM.items():
            for e, it in enumerate(d["out"][lst]):
                for k, fl in enumerate(it["floors"]):
                    f = floor_index(fl["floor"], k, n, d["order"])
                    if f is None or not fl["present"]:
                        continue
                    w, h = pres.size(b, f)
                    raw = (fl["x"] * w / 1000, fl["y"] * h / 1000)
                    ok = A[str(f)]["aligned"]
                    pts.append(dict(run=r, fam=f_, ent=f"{r}/{lst}/{e}", name=it["name"], kind=it.get("kind", "stair"), floor=f, raw=raw, ref=apply_pt(np.array(A[str(f)]["H"]), raw) if ok else None))
    return pts, usd, secs


def cluster(pts):
    """Leader clustering on ref position with two recentring passes. Points without a ref position join through their entity."""
    placed = [p for p in pts if p["ref"] is not None]
    cs = []
    for _ in range(3):
        for c in cs:
            c["pts"] = []
        for p in placed:
            best, bd = None, 1e9
            for c in cs:
                d = float(np.hypot(*(p["ref"] - c["pos"])))
                if d < bd:
                    best, bd = c, d
            if best is not None and bd <= R:
                best["pts"].append(p)
            else:
                cs.append(dict(pos=p["ref"].copy(), pts=[p]))
        cs = [c for c in cs if c["pts"]]
        for c in cs:
            c["pos"] = np.median([p["ref"] for p in c["pts"]], axis=0)
    home = {}
    for i, c in enumerate(cs):
        for p in c["pts"]:
            home.setdefault(p["ent"], []).append(i)
    # a stray point rejoins its own entity's row when that row is close: same name from Spark, and near enough to be the same stair
    maj = {e: max(set(h), key=h.count) for e, h in home.items()}
    for i, c in enumerate(cs):
        for p in list(c["pts"]):
            j = maj[p["ent"]]
            if j != i and float(np.hypot(*(p["ref"] - cs[j]["pos"]))) <= 2 * R:
                c["pts"].remove(p)
                cs[j]["pts"].append(p)
    cs = [c for c in cs if c["pts"]]
    home = {}
    for i, c in enumerate(cs):
        for p in c["pts"]:
            home.setdefault(p["ent"], []).append(i)
    orphans = {}
    for p in pts:
        if p["ref"] is None:
            h = home.get(p["ent"])
            if h:
                cs[max(set(h), key=h.count)]["pts"].append(p)
            else:
                orphans.setdefault(p["ent"], []).append(p)
    for v in orphans.values():
        cs.append(dict(pos=None, pts=v))
    return cs


def rows(b, key, runs=(0, 1, 2)):
    """Rows with per-floor votes. loc is the marker: Spark's median point when a run claimed the floor, else the row position through the floor's H. chk is where local checks look."""
    A = pres.alignment(b)
    pts, usd, secs = points(b, key, runs)
    out = []
    for fam in ("stair", "shaft"):
        for c in cluster([p for p in pts if p["fam"] == fam]):
            runs_seen = sorted({p["run"] for p in c["pts"]})
            cells = {}
            for f in range(pres.nfloors(b)):
                here = [p for p in c["pts"] if p["floor"] == f]
                v = len({p["run"] for p in here})
                reg_ = apply_pt(np.linalg.inv(np.array(A[str(f)]["H"])), c["pos"]) if c["pos"] is not None and A[str(f)]["aligned"] else None
                mark = np.median([p["raw"] for p in here], axis=0) if here else reg_
                chk = reg_ if reg_ is not None else mark  # the registered place: a run that imagines a stair on an erased sheet also drifts its point
                cells[f] = dict(v=v, loc=None if mark is None else [float(mark[0]), float(mark[1])], chk=None if chk is None else [float(chk[0]), float(chk[1])], aligned=A[str(f)]["aligned"])
            out.append(dict(fam=fam, pos=None if c["pos"] is None else [float(c["pos"][0]), float(c["pos"][1])], runs=runs_seen, names=sorted({p["name"] for p in c["pts"]})[:4],
                            kinds=sorted({p["kind"] for p in c["pts"]}), npts=len(c["pts"]), cells=cells))
    return out, usd, secs


_ink = {}


def inkmap(path):
    k = str(path)
    if k not in _ink:
        _ink[k] = reg.ink(cv2.imread(k))
    return _ink[k]


def density(path, loc, half=HALF):
    im = inkmap(path)
    h, w = im.shape
    x, y = int(loc[0]), int(loc[1])
    c = im[max(y - half, 0):min(y + half, h), max(x - half, 0):min(x + half, w)]
    return float(c.mean()) if c.size else 0.0


def ink_ratios(row, paths, half=HALF):
    """paths: floor -> sheet path. Ratio of ink in the box on this floor to the median over the other floors at least two runs called present."""
    out = {}
    d = {f: density(paths[f], c["chk"], half) for f, c in row["cells"].items() if c["chk"] is not None}
    for f, c in row["cells"].items():
        if f not in d:
            out[f] = None
            continue
        others = [d[g] for g, cc in row["cells"].items() if g != f and g in d and cc["v"] >= 2]
        out[f] = round(d[f] / (float(np.median(others)) + 1e-6), 3) if others else None
    return out, {f: round(v, 4) for f, v in d.items()}


if __name__ == "__main__":
    # self-check: three runs agree on one place, a fourth place only one run saw; a glued entity is split into two rows
    P = lambda r, e, f, x, y: dict(run=r, fam="stair", ent=f"{r}/s/{e}", name="s", kind="stair", floor=f, raw=(x, y), ref=np.array([x, y], float))
    pts = [P(r, 0, f, 100 + r, 100) for r in range(3) for f in range(2)] + [P(0, 1, 0, 900, 900)] + [P(1, 2, 0, 500, 500), P(1, 2, 1, 700, 500)]
    cs = cluster(pts)
    sizes = sorted(len(c["pts"]) for c in cs)
    assert sizes == [1, 1, 1, 6], sizes
    u = dict(run=2, fam="stair", ent="2/s/0", name="s", kind="stair", floor=3, raw=(1, 1), ref=None)
    v = dict(run=1, fam="stair", ent="1/s/9", name="s", kind="stair", floor=3, raw=(1, 1), ref=None)
    cs = cluster(pts + [u, v])
    assert max(len(c["pts"]) for c in cs) == 7 and sum(c["pos"] is None for c in cs) == 1
    print("cells self-check ok")
