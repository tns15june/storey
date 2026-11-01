"""Independent rescore of tightseed from truth files and cached tracks. No API, imports nothing from tightseed.
Run: .venv/bin/python -I lab/audit2/rescore_tightseed.py"""
import json
import statistics as st_
from pathlib import Path

import numpy as np

R = Path("data/lab/tightseed")
OUT = Path("data/lab/audit2")
T = {k: v for k, v in json.load(open(R / "truth.json")).items() if not k.startswith("_")}
FRESH = {k: v for k, v in json.load(open(R / "truth_fresh.json")).items() if not k.startswith("_")}
ALL = {**T, **FRESH}
HELD = [k for k, v in T.items() if v["split"] == "heldout"]
SC = ("main", "short")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def hit(box, t, grow=.25):
    gx, gy = (t[2] - t[0]) * grow, (t[3] - t[1]) * grow
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    inside = t[0] - gx <= cx <= t[2] + gx and t[1] - gy <= cy <= t[3] + gy
    big = (box[2] - box[0]) * (box[3] - box[1]) > 4 * (t[2] - t[0]) * (t[3] - t[1])
    return iou(box, t) >= .25 or (inside and not big)


def load(st, me):
    p = R / "sam" / f"{st}__{me}.json"
    return json.load(open(p)) if p.exists() else None


def owners(st, tr, grow=.25):
    """Stair -> set of track ids whose seed hits any floor's box of that stair."""
    own = {}
    for o, f, b in tr["seeds"]:
        for k, s in ALL[st]["stairs"].items():
            if any(hit(b, x, grow) for x in s["boxes"].values()):
                own.setdefault(k, set()).add(str(o))
    return own


def rows_of(st, tr, conf=.5, grow=.25):
    own = owners(st, tr, grow)
    n = ALL[st]["n"]
    rows = []
    for k, s in ALL[st]["stairs"].items():
        if s["cls"] not in SC:
            continue
        ex = set(s.get("exclude") or ())
        boxes = {int(f): b for f, b in s["boxes"].items()}
        for f in range(n):
            if f in ex:
                continue
            masks = []
            for o in own.get(k, ()):
                mk = tr["tracks"].get(o, {}).get(str(f))
                masks.append(mk)
            claimed = [m for m in masks if m and m["area"] > 0 and (m["conf"] or 0) >= conf]
            present = f in boxes
            if present:
                ok = bool(own.get(k)) and len(claimed) == len(masks) and all(hit(m["box"], boxes[f], grow) for m in claimed)
            else:
                ok = not claimed
            rows.append(dict(stack=st, stair=k, floor=f, present=present, seeded=bool(own.get(k)), ok=ok, claimed=len(claimed), confs=[m["conf"] for m in masks if m and m["area"] > 0], cls=s["cls"]))
    return rows


def tally(rows):
    pr = [r for r in rows if r["present"]]
    ab = [r for r in rows if not r["present"]]
    sab = [r for r in ab if r["seeded"]]
    stairs = {}
    for r in rows:
        stairs.setdefault((r["stack"], r["stair"]), []).append(r)
    return dict(pairs=f"{sum(r['ok'] for r in rows)}/{len(rows)}", present=f"{sum(r['ok'] for r in pr)}/{len(pr)}", absent=f"{sum(r['ok'] for r in ab)}/{len(ab)}", seeded_absent=f"{sum(r['ok'] for r in sab)}/{len(sab)}",
                ghosts=sum(r["claimed"] > 0 for r in ab), stairs=f"{sum(all(x['ok'] for x in v) for v in stairs.values())}/{len(stairs)}", unseeded=sum(not v[0]["seeded"] for v in stairs.values()))


def one_call(st):
    """Round 1 one-call list matched by the 120 px rule, reimplemented. Returns {(stair, floor): list of present votes}."""
    p = R / "spark" / f"{st}_floors.json"
    ls = json.load(open(p))["out"]["stairs"]
    t = T[st]
    n = t["n"]
    x1, y1, x2, y2 = t["crop"]
    w, h = x2 - x1, y2 - y1
    say = {}
    for k, s in t["stairs"].items():
        if s["cls"] not in SC:
            continue
        bx = {int(f): b for f, b in s["boxes"].items()}
        near = []
        for l in ls:
            if len(l["floors"]) != n:
                continue
            d = [np.hypot(l["floors"][f]["x"] * w / 1000 + x1 - (bx[f][0] + bx[f][2]) / 2, l["floors"][f]["y"] * h / 1000 + y1 - (bx[f][1] + bx[f][3]) / 2) for f in bx if l["floors"][f]["present"]]
            if d and np.median(d) < 120:
                near.append(l)
        for f in range(n):
            say[(k, f)] = [bool(l["floors"][f]["present"]) for l in near]
    return say


def main():
    res = {}
    print("baseline always-present from truth only")
    rows = []
    for st in HELD:
        for k, s in T[st]["stairs"].items():
            if s["cls"] in SC:
                ex = set(s.get("exclude") or ())
                for f in range(T[st]["n"]):
                    if f not in ex:
                        rows.append((st, k, f, str(f) in s["boxes"]))
    pres = sum(r[3] for r in rows)
    by = {}
    for r in rows:
        by.setdefault((r[0], r[1]), []).append(r[3])
    print(" ", pres, "/", len(rows), "absent", len(rows) - pres, "stairs all present", sum(all(v) for v in by.values()), "/", len(by))
    for st in HELD:
        q = [r for r in rows if r[0] == st]
        print("   ", st, sum(r[3] for r in q), "/", len(q))
    print("\nstrict per method, held-out, conf .5, grow .25")
    for me in ("hand_960", "hand_2000", "loose_960", "snap_960", "sparkcrop_960", "d_960", "P_960", "Ppin_960"):
        r = []
        for st in HELD:
            tr = load(st, me)
            r += rows_of(st, tr)
        res[me] = tally(r)
        print(f" {me:15s}", res[me])
        print("     by stack", {st: f"{sum(x['ok'] for x in r if x['stack'] == st)}/{sum(x['stack'] == st for x in r)}" for st in HELD})
    print("\nvector arms")
    for st in ("wfu_huffman", "mml_abbot"):
        print(" ", st, tally(rows_of(st, load(st, "vector_960"))))
    print("\nfresh royce_east")
    for me in ("hand_960", "loose_960", "P_960", "Ppin_960"):
        print(f" {me:15s}", tally(rows_of("royce_east", load("royce_east", me))))
    print("\nsensitivity: grow .15 (the common.py default) and conf .4 .45 .55 .6, held-out")
    for me in ("hand_960", "P_960", "Ppin_960"):
        for conf, grow in ((.5, .15), (.4, .25), (.45, .25), (.55, .25), (.6, .25)):
            r = []
            for st in HELD:
                r += rows_of(st, load(st, me), conf, grow)
            print(f" {me:10s} conf {conf} grow {grow}", tally(r))
    print("\nconfidences on seeded-stair truth-absent floors (any mask)")
    for me in ("hand_960", "P_960", "Ppin_960"):
        a = []
        for st in HELD + ["royce_east"]:
            for x in rows_of(st, load(st, me)):
                if not x["present"] and x["confs"]:
                    a.append((st, x["stair"], x["floor"], x["confs"]))
        print(" ", me, a)
    print("\nending flag on seeded stairs, Spark one-call says absent; veto by SAM mask at conf .5. The report ran P only; the recipe uses Ppin.")
    for me in ("P_960", "Ppin_960"):
        tp = fp = vetoed_true = vetoed_false = fn = 0
        rowsv = []
        lost = []
        for st in HELD:
            say = one_call(st)
            for r in rows_of(st, load(st, me)):
                if not r["seeded"]:
                    continue
                v = say.get((r["stair"], r["floor"]), [])
                spark_absent = (r["floor"] is not None) and ((not v) or (not all(v) if r["present"] else not any(v)))
                # same reading as final2: strict row wrong on a present floor, or strict row right on an absent floor
                spark_ok = bool(v) and all(x == r["present"] for x in v)
                spark_absent = spark_ok != r["present"]
                mask = r["claimed"] > 0
                if spark_absent and not r["present"]:
                    tp += not mask
                    vetoed_true += mask
                    if mask:
                        lost.append((st, r["stair"], r["floor"], r["confs"]))
                if spark_absent and r["present"]:
                    fp += not mask
                    vetoed_false += mask
                if (not spark_absent) and not r["present"]:
                    fn += 1
        print(f" {me}: flags kept true {tp} false {fp}; cancelled by SAM true {vetoed_true} false {vetoed_false}; true endings Spark never flagged {fn}; true flags lost {lost}")
    json.dump(res, open(OUT / "tightseed_rescore.json", "w"), indent=1)


if __name__ == "__main__":
    main()
