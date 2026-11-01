"""Score cells against truth. No API. A cell is judged by where it sits, not by a row-to-stair match, so no choice made with truth can flatter a row.
Found: correct if a truth object of the family is drawn on that floor within D px of the marker, else a wrong pass. Missing: a false finding if such an object is there. Check: counted only.
An object marked unsure on that floor excuses the cell. Coverage: truth present cells with a Found, a Check, a Missing or nothing near them.
Baselines: always-present over the truth object list, and run 0 alone."""
import json
import sys

sys.path.insert(0, "lab/presence")
import numpy as np

import pres
from pres import T, box_dist, fam, apply_pt, OUT

D = 40


def near(b, f, loc, family, fam_of=fam):
    """Objects of the family within D of this place on floor f: (present names, unsure names)."""
    pr, un = [], []
    for o, v in T[b]["objects"].items():
        if fam_of(v["kind"]) != family:
            continue
        us = {int(x) for x in v.get("unsure", [])}
        bx = v["boxes"].get(str(f))
        if bx is None or box_dist(loc, bx) > D:
            continue
        (un if f in us else pr).append(o)
    return pr, un


def ref_pos(b, v):
    """Median place of a truth object on the reference floor, through aligned floors only."""
    A = pres.alignment(b)
    pts = [apply_pt(np.array(A[f]["H"]), ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)) for f, bx in v["boxes"].items() if A[f]["aligned"]]
    return np.median(pts, axis=0) if pts else None


def absent_at(b, f, loc, family):
    """Truth object absent on floor f whose registered place is within D of loc. Tells a wrong pass on a real stair from a phantom row."""
    A = pres.alignment(b)
    if not A[str(f)]["aligned"]:
        return None
    H = np.array(A[str(f)]["H"])
    q = apply_pt(H, loc)
    best = None
    for o, v in T[b]["objects"].items():
        if fam(v["kind"]) != family or str(f) in v["boxes"]:
            continue
        p = ref_pos(b, v)
        if p is None:
            continue
        d = float(np.hypot(*(q - p)))
        if d <= D + 40 and (best is None or d < best[1]):
            best = (o, d)
    return best and best[0]


def score(b, decided, family="stair"):
    """decided: list of dict(f, loc, state, row). Returns counts and the lists behind them."""
    r = dict(found=0, ok=0, wrong=[], phantom=[], excused=0, missing=0, false=[], check=0, check_true=0)
    seen = {}
    for c in decided:
        if c["state"] is None or c["loc"] is None:
            continue
        pr, un = near(b, c["f"], c["loc"], family)
        if c["state"] == "Found":
            r["found"] += 1
            if pr:
                r["ok"] += 1
            elif un:
                r["excused"] += 1
            else:
                o = absent_at(b, c["f"], c["loc"], family)
                (r["wrong"] if o else r["phantom"]).append(dict(row=c["row"], f=c["f"], loc=[round(x) for x in c["loc"]], obj=o))
        elif c["state"] == "Missing":
            r["missing"] += 1
            if pr:
                r["false"].append(dict(row=c["row"], f=c["f"], loc=[round(x) for x in c["loc"]], obj=pr[0]))
        else:
            r["check"] += 1
            r["check_true"] += bool(pr)
        for o in pr:
            k = (o, c["f"])
            rank = {"Found": 3, "Check": 2, "Missing": 1}[c["state"]]
            seen[k] = max(seen.get(k, 0), rank)
    # coverage of truth cells
    cov = dict(found=0, check=0, missing=0, none=0)
    for o, v in T[b]["objects"].items():
        if fam(v["kind"]) != family:
            continue
        us = {int(x) for x in v.get("unsure", [])}
        for f in map(int, v["boxes"]):
            if f in us:
                continue
            cov[{3: "found", 2: "check", 1: "missing", 0: "none"}[seen.get((o, f), 0)]] += 1
    r["cover"] = cov
    return r


def baseline(b, family="stair", cls=None):
    """Always-present over the truth object list: every cell Found."""
    pres_, abs_ = 0, 0
    n = pres.nfloors(b)
    for v in T[b]["objects"].values():
        if fam(v["kind"]) != family or (cls and v["cls"] not in cls):
            continue
        us = {int(x) for x in v.get("unsure", [])}
        p = [f for f in map(int, v["boxes"]) if f not in us]
        pres_ += len(p)
        abs_ += n - len(p) - len(us)
    return dict(present=pres_, absent=abs_, wrong=abs_)


def single_run(key, b, r=0, family="stair"):
    """Run r alone, entity by entity: present is Found at its point, absent is Missing at the entity's median place."""
    d = json.load(open(OUT / f"spark/{key}_r{r}.json"))
    A = pres.alignment(b)
    out = []
    lst = "stairs" if family == "stair" else "shafts"
    for e, it in enumerate(d["out"][lst]):
        pts = {}
        import cells
        n = pres.nfloors(b)
        for k, fl in enumerate(it["floors"]):
            f = cells.floor_index(fl["floor"], k, n, d["order"])
            if f is None:
                continue
            w, h = pres.size(b, f)
            pts[f] = (fl["present"], (fl["x"] * w / 1000, fl["y"] * h / 1000))
        refs = [apply_pt(np.array(A[str(f)]["H"]), p) for f, (on, p) in pts.items() if on and A[str(f)]["aligned"]]
        pos = np.median(refs, axis=0) if refs else None
        for f, (on, p) in pts.items():
            if on:
                out.append(dict(f=f, loc=list(p), state="Found", row=f"r{r}e{e}"))
            elif pos is not None and A[str(f)]["aligned"]:
                out.append(dict(f=f, loc=list(apply_pt(np.linalg.inv(np.array(A[str(f)]["H"])), pos)), state="Missing", row=f"r{r}e{e}"))
    return out


if __name__ == "__main__":
    b = "nibs_clinic"
    fake = [dict(f=0, loc=[800, 200], state="Found", row=0), dict(f=1, loc=[100, 100], state="Found", row=1), dict(f=1, loc=[745, 350], state="Missing", row=2)]
    s = score(b, fake)
    assert s["ok"] == 1 and len(s["phantom"]) == 1 and len(s["false"]) == 1, s
    assert baseline("bme_f", cls=("main",))["absent"] == 2
    print("score self-check ok")
