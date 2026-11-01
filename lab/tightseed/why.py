"""Why present floors fail, per method, and how far Spark's one-call centres sit from truth. No API."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
for me in ("hand_960", "loose_960", "snap_960", "sparkcrop_960", "d_960"):
    c = dict(unseeded=0, nomask=0, lowconf=0, misplaced=0)
    bad = []
    for st in HELD:
        tr = json.load(open(SAMD / f"{st}__{me}.json"))
        rows, _ = score(st, tr)
        for r in rows:
            if r["cls"] in SCORED and r["present"] and not r["ok"]:
                if not r["seeded"]:
                    c["unseeded"] += 1; continue
                worst = None
                for o, cl, h, conf in r["claims"]:
                    mk = tr["tracks"].get(o, {}).get(str(r["floor"]))
                    if not mk or mk["area"] == 0: worst = "nomask"
                    elif not has(mk): worst = worst or "lowconf"
                    elif not h: worst = worst or "misplaced"
                c[worst] += 1
                if me == "hand_960": bad.append((st, r["stair"], r["floor"], worst, r["claims"]))
    print(me, c, bad)
print("one-call centre error, px at stack scale, truth floors")
for st in HELD:
    T = TRUTH[st]; x1, y1, x2, y2 = T["crop"]; w, h = x2 - x1, y2 - y1
    ls = json.load(open(SPK / f"{st}_floors.json"))["out"]["stairs"]
    e = []
    for k, s in T["stairs"].items():
        if s["cls"] == "amb": continue
        best = None
        for l in ls:
            if len(l["floors"]) != T["n"]: continue
            d = [np.hypot(l["floors"][int(f)]["x"] * w / 1000 + x1 - (b[0] + b[2]) / 2, l["floors"][int(f)]["y"] * h / 1000 + y1 - (b[1] + b[3]) / 2) for f, b in s["boxes"].items() if l["floors"][int(f)]["present"]]
            if d and (best is None or np.median(d) < best):
                best = float(np.median(d))
        e.append(round(best, 1) if best is not None else None)
    print(st, "listed", len(ls), "median of medians", np.median([x for x in e if x is not None]).round(1) if any(x is not None for x in e) else None, e)
st = "leonard4"; ls = json.load(open(SPK / f"{st}_floors.json"))["out"]["stairs"]
print([(l["name"], [(f["floor"], f["present"], f["x"], f["y"]) for f in l["floors"]]) for l in ls][:6])
