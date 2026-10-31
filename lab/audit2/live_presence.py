"""L3, L4, L4b from data/lab/audit2/rule.json. Usage: live_presence.py hard | soft | compare. All calls go through metalab under LAB_NAME=audit2. Nothing is written outside data/lab/audit2."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/audit2")
import pres
from pres import OUT, SH, T, TRUTH, box_dist
import cells as C
import compare
import crop
import plants
import rule
import metalab as m

AUD = pres.ROOT / "data/lab/audit2"
TARGETS = json.load(open(AUD / "plant_targets.json"))["targets"]


def ask_once(path, loc):
    _, x, y = crop.key_of(path, loc, "stair")
    r, info = m.spark(crop.Q["stair"], [crop.cut(path, x, y)], schema=crop.ANS, effort="low", max_tokens=4000)
    return dict(answer=r["answer"], seen=r["seen"], usd=info["usd"])


def many(jobs):
    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(lambda j: ask_once(*j), jobs))


def hard():
    rows = []
    for b in pres.HELD:
        for kind, o, f in TRUTH["plant"][b]:
            key = f"{b}__{kind}"
            E = json.load(open(OUT / f"cells/{key}.json"))
            bx = T[b]["objects"][o]["boxes"][str(f)]
            near = sorted([(box_dist(r["cells"][str(f)]["chk"], bx), r) for r in E["rows"] if r["fam"] == "stair" and r["cells"][str(f)]["chk"]], key=lambda q: q[0])
            r = near[0][1]
            c = r["cells"][str(f)]
            row = dict(key=key, obj=o, f=f, v=c["v"], ink=r["ink30"].get(str(f)), blank=c.get("blank"), frozen_crop=c.get("crop"), aligned=c["aligned"], established=r["established"])
            if not c.get("blank"):
                row["planted"] = [a["answer"] for a in many([(OUT / f"planted/{key}/{f}.jpg", c["chk"])] * 3)]
                row["seen"] = many([(OUT / f"planted/{key}/{f}.jpg", c["chk"])])[0]["seen"]
                row["planted"].append(None)
                row["planted"].pop()
                row["clean"] = [a["answer"] for a in many([(SH / f"{b}/{f}.jpg", c["chk"])] * 2)]
            rows.append(row)
            print(row, flush=True)
    json.dump(rows, open(AUD / "L3_hard.json", "w"), indent=1)
    print("spent", round(m.spent("audit2"), 4))


def soft():
    rows = []
    jobs = []
    for t in TARGETS:
        b, o, f = t["b"], t["obj"], t["f"]
        E = json.load(open(OUT / f"cells/{b}.json"))
        r = E["rows"][t["row"]]
        assert r["id"] == t["row"]
        plant = AUD / f"plants/{b}__{o}/{f}.jpg"
        others = [C.density(SH / f"{b}/{g}.jpg", c["chk"], 30) for g, c in r["cells"].items() if int(g) != f and c["chk"] is not None and c["v"] >= 2]
        ink = C.density(plant, t["chk"], 30) / (float(np.median(others)) + 1e-6) if others else None
        blank = C.density(plant, t["chk"], 50) < .002
        rows.append(dict(b=b, obj=o, f=f, ink=None if ink is None else round(ink, 3), blank=blank, v_clean=t["v_clean"], share_removed=t.get("share_removed"), others=len(others)))
        jobs.append((plant, SH / f"{b}/{f}.jpg", t["chk"]))
    with ThreadPoolExecutor(6) as ex:
        pl = list(ex.map(lambda j: [ask_once(j[0], j[2]) for _ in range(3)], jobs))
        cl = list(ex.map(lambda j: [ask_once(j[1], j[2]) for _ in range(2)], jobs))
    for row, p, c in zip(rows, pl, cl):
        row["planted"] = [a["answer"] for a in p]
        row["seen"] = [a["seen"] for a in p]
        row["clean"] = [a["answer"] for a in c]
        row["found_repeats"] = 0 if row["blank"] or row["ink"] is None or row["ink"] < rule.INK_LOW else sum(a == "yes" for a in row["planted"])
        print({k: row[k] for k in ("b", "obj", "f", "ink", "blank", "planted", "clean", "found_repeats")}, flush=True)
    json.dump(rows, open(AUD / "L4_soft.json", "w"), indent=1)
    n = len(rows)
    print("plants", n, "with a Found repeat", sum(r["found_repeats"] > 0 for r in rows), "Found repeats", sum(r["found_repeats"] for r in rows), "of", 3 * n)
    print("crop yes on plant", sum(a == "yes" for r in rows for a in r["planted"]), "of", 3 * n, "crop yes on clean", sum(a == "yes" for r in rows for a in r["clean"]), "of", 2 * n)
    print("ink under gate", sum(r["ink"] is not None and r["ink"] < rule.INK_LOW for r in rows), "blank", sum(r["blank"] for r in rows))
    print("spent", round(m.spent("audit2"), 4))


def compare_runs():
    L4 = json.load(open(AUD / "L4_soft.json"))
    compare.OUT = AUD
    C.OUT = AUD
    (AUD / "spark").mkdir(exist_ok=True)
    hits = [r for r in L4 if r["found_repeats"] > 0]
    rest = [r for r in L4 if r["found_repeats"] == 0]
    pick, seen = [], set()
    for r in hits + rest:
        if r["b"] not in seen and len(pick) < 4:
            pick.append(r)
            seen.add(r["b"])
    for r in hits + rest:
        if r not in pick and len(pick) < 4:
            pick.append(r)
    out = []
    for q in pick:
        b, o, f = q["b"], q["obj"], q["f"]
        key = f"{b}__soft_{o}"
        order, labels = compare.order_of(b)
        plant = AUD / f"plants/{b}__{o}/{f}.jpg"
        paths = [plant if g == f else SH / f"{b}/{g}.jpg" for g in order]
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda r: compare.run(key, paths, order, labels, r), (0, 1, 2)))
        rows, _, _ = C.rows(b, key)
        bypath = {g: (plant if g == f else SH / f"{b}/{g}.jpg") for g in range(pres.nfloors(b))}
        bx = T[b]["objects"][o]["boxes"][str(f)]
        res = []
        for i, r in enumerate(rows):
            r["id"] = i
            r["established"] = max(c["v"] for c in r["cells"].values()) >= 2
            r["ink30"], _ = C.ink_ratios(r, bypath, 30)
            _, d50 = C.ink_ratios(r, bypath, 50)
            c = r["cells"][f]
            c["blank"] = c["chk"] is None or d50.get(f, 0) < .002
            if r["fam"] != "stair" or c["chk"] is None or box_dist(c["chk"], bx) > 40:
                continue
            if not c["blank"] and (r["established"] or c["v"] >= 1):
                a = ask_once(plant, c["chk"])
                c["crop"], c["seen"] = a["answer"], a["seen"]
            state = rule.decide(dict(r, ink=r["ink30"].get(f)), c, "C")
            res.append(dict(names=r["names"][:2], v=c["v"], ink=r["ink30"].get(f), blank=c["blank"], crop=c.get("crop"), seen=c.get("seen"), state=state, dist=round(box_dist(c["chk"], bx))))
        rec = dict(key=key, b=b, obj=o, f=f, rows=res)
        out.append(rec)
        print(rec, flush=True)
    json.dump(out, open(AUD / "L4b_compare.json", "w"), indent=1, default=float)
    print("spent", round(m.spent("audit2"), 4))


if __name__ == "__main__":
    {"hard": hard, "soft": soft, "compare": compare_runs}[sys.argv[1]]()
