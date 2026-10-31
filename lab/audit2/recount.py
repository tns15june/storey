"""Recounts with no API: presence natural absences and main-stair coverage, tightseed recipe cost, declared held-out support for each recommended Fix. Run with -I."""
import json, math, glob, collections, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/audit2")
OUT = "/Users/tsharma/Github/Meta/data/lab/presence/"
TR = json.load(open("/Users/tsharma/Github/Meta/lab/presence/truth.json")); T = TR["buildings"]
HELD = [b for b, t in T.items() if t["split"] == "heldout"]
def box_dist(p, b): return math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))
def decide(est, ink, c):
    v, crop, blank = c["v"], c.get("crop"), c.get("blank")
    if not est: return None if (v == 0 or blank or crop == "no") else "Check"
    if not c["aligned"]: return "Check"
    if v >= 2:
        if ink is not None and ink < .4: return "Check"
        if blank or crop in ("no", "unclear"): return "Check"
        return "Found"
    if blank: return "Missing"
    if crop in ("yes", "unclear"): return "Check"
    if ink is not None and ink >= .9: return "Check"
    return "Missing"
cover = collections.Counter(); byb = {}
for b in HELD:
    E = json.load(open(OUT + f"cells/{b}.json"))
    n = len(next(r for r in E["rows"])["cells"])
    for o, v in T[b]["objects"].items():
        if v["kind"] not in ("stair", "stairs") or v["cls"] != "main":
            continue
        us = {int(x) for x in v.get("unsure", [])}
        for f in range(n):
            if str(f) not in v["boxes"] or f in us:
                continue
            st = [decide(r["established"], r["ink30"].get(str(f)), r["cells"][str(f)]) for r in E["rows"] if r["fam"] == "stair" and r["cells"][str(f)]["loc"] and box_dist(r["cells"][str(f)]["loc"], v["boxes"][str(f)]) <= 40]
            s = "Found" if "Found" in st else "Check" if "Check" in st else "Missing" if "Missing" in st else "not listed"
            cover[s] += 1
            byb.setdefault(b, collections.Counter())[s] += 1
print("main stair present cells, arm C", dict(cover), "total", sum(cover.values()), {b: dict(v) for b, v in byb.items()})
# natural absences tested: a truth stair absent on floor f with a row at its registered place, using median of the row chk on that floor
import numpy as np
tested = collections.Counter(); main_tested = collections.Counter()
for b in HELD:
    E = json.load(open(OUT + f"cells/{b}.json"))
    A = json.load(open(OUT + f"align/{b}.json"))
    def app(H, p):
        q = np.array(H) @ np.array([p[0], p[1], 1.0]); return q[:2] / q[2]
    n = len(A)
    for o, v in T[b]["objects"].items():
        if v["kind"] not in ("stair", "stairs"):
            continue
        us = {int(x) for x in v.get("unsure", [])}
        pts = [app(A[f]["H"], ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)) for f, bx in v["boxes"].items() if A[f]["aligned"]]
        if not pts:
            continue
        ref = np.median(pts, axis=0)
        for f in range(n):
            if str(f) in v["boxes"] or f in us or not A[str(f)]["aligned"]:
                continue
            q = app(np.linalg.inv(np.array(A[str(f)]["H"])), ref)
            st = [decide(r["established"], r["ink30"].get(str(f)), r["cells"][str(f)]) for r in E["rows"] if r["fam"] == "stair" and r["cells"][str(f)]["chk"] and math.hypot(r["cells"][str(f)]["chk"][0] - q[0], r["cells"][str(f)]["chk"][1] - q[1]) <= 40]
            st = [s for s in st if s]
            if st:
                s = "Found" if "Found" in st else "Check" if "Check" in st else "Missing"
                tested[s] += 1
                if v["cls"] == "main": main_tested[s] += 1
print("natural absent cells with a shown row at the place (my count, 40 px from the registered truth place)", dict(tested), "main only", dict(main_tested))
print("3-run rows per truth main stair")
# tightseed cost
tot = 0; n = 0
for p in glob.glob("/Users/tsharma/Github/Meta/data/lab/tightseed/spark/*_whole*_r*.json"):
    d = json.load(open(p)); tot += d["info"]["usd"]; n += 1
print("tightseed whole-stair crop calls", n, "usd", round(tot, 3), "per call", round(tot / n, 4), "two rounds", round(2 * tot / n, 4))
f = [json.load(open(p))["info"]["usd"] for p in glob.glob("/Users/tsharma/Github/Meta/data/lab/tightseed/spark/*_f[0-9].json")]
print("per-floor grounding calls", len(f), "mean", round(sum(f) / len(f), 4))
# declared support
Y = json.load(open("/Users/tsharma/Github/Meta/data/lab/declared/yield_v2.json"))
for k, v in sorted(Y["by_sub"].items()):
    h = v["held"]
    print(f"  {k:32s} pooled {v['real']}+{v['likely']}/{v['flags']} held {h['real']}+{h['likely']}/{h['flags']} rule {v['rule']}")
