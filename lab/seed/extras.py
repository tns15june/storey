"""Tables for padding, refinement, single seed sheet, Spark variants on the four scan buildings, and location-only scoring. No API calls."""
import sys, glob; sys.path.insert(0, "lab/seed")
from common import *
import union
from union import cluster, evaluate
s = json.load(open(OUT / "sam_score.json"))
main = lambda r: [x for x in r["rows"].values() if x["cls"] == "main"]
vars_ = ["p0", "p0.1", "p0.25", "p0.5", "refine"]
pairs = sorted({t.rsplit("__", 1)[0] for t in s if t.rsplit("__", 1)[1] == "refine"})
tab = {}
for v in vars_:
    rows = [s[f"{p}__{v}"] for p in pairs]
    m = [x for r in rows for x in main(r)]
    tab[v] = {"runs": len(rows), "main_n": len(m), "seeded": sum(x["seeded"] for x in m), "e2e": sum(x["all_ok"] for x in m), "false_seeds": sum(len(r["false_seeds"]) for r in rows), "seeds": sum(r["n_seeds"] for r in rows)}
    print(v, tab[v])
single = {}
for t, r in s.items():
    if t.endswith("__p0"):
        b, f, _ = t.split("__")
        m = main(r)
        single.setdefault(b, {})[f] = [sum(x["seeded"] for x in m), sum(x["all_ok"] for x in m), len(m)]
print(single)
ab = {"ok": 0, "ghost": 0}
for t, r in s.items():
    if t.endswith("__p0"):
        bid = t.split("__")[0]
        for tt in TRUTH[bid]:
            v = r["rows"][tt["id"]]
            if v["seeded"]:
                for f, o in v["floors"].items():
                    if int(f) not in tt["floors"] and int(f) != v["seed_floor"]:
                        ab[o] = ab.get(o, 0) + 1
print("absent floors, no conf filter, p0 runs", ab)
union.CONF = 0.0
loc = {}
for bid in ["bme_f", "schenley", "habs_stelizabeths_atkins", "habs_stelizabeths_center", "mml_capitol", "nibs_clinic", "mml_baxter"]:
    cl = cluster(bid, "doc")
    tr = json.load(open(OUT / "sam" / f"union_doc__{bid}.json"))["tracks"]
    rows, _ = evaluate(bid, cl, tr, [True] * len(cl))
    m = [x for x in rows.values() if x["cls"] == "main"]
    loc[bid] = [sum(x["all_ok"] for x in m), len(m), {k: x["floors"] for k, x in rows.items() if x["cls"] == "main" and not x["all_ok"]}]
print("location only, any confidence", loc)
json.dump({"pad_refine": tab, "single_sheet": single, "absent_p0": ab, "location_any_conf": loc}, open(OUT / "extras.json", "w"), indent=1, default=int)
