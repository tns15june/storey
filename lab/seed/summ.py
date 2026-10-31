"""Pair-level comparison of SAM tracking, Spark per-sheet, and their agreement. No API calls."""
import sys, glob; sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed")
from common import *
out = {}
for var in sys.argv[1:] or ["doc", "plan_low"]:
    for mode in ("raw", "gated"):
        tot = {"pairs": 0, "sam_ok": 0, "spark_ok": 0, "agree": 0, "agree_ok": 0, "disagree": 0, "disagree_sam_right": 0}
        stacks = {"main": [0, 0, 0], "ext": [0, 0, 0]}
        false = cl = 0
        per = {}
        for p in sorted(glob.glob(str(OUT / "sam" / f"result_{var}__*.json"))):
            bid = Path(p).stem.split("__")[1]
            s = json.load(open(p))["summary"][mode]
            false += len(s["false_clusters"]); cl += s["clusters"]
            nm = [0, 0, 0]
            for tid, r in s["rows"].items():
                if r["cls"] not in stacks:
                    continue
                st = stacks[r["cls"]]
                st[0] += 1; st[1] += r["seeded"]; st[2] += r["all_ok"]
                if r["cls"] == "main":
                    nm[0] += 1; nm[1] += r["seeded"]; nm[2] += r["all_ok"]
                if not r["seeded"]:
                    continue
                for f, v in r["floors"].items():
                    a, b = v == "ok", r["spark"][f] == "ok"
                    tot["pairs"] += 1; tot["sam_ok"] += a; tot["spark_ok"] += b
                    if a == b:
                        tot["agree"] += 1; tot["agree_ok"] += a
                    else:
                        tot["disagree"] += 1; tot["disagree_sam_right"] += a
            per[bid] = nm
        print(var, mode, "clusters", cl, "false", false, "stacks[n,seeded,e2e]", stacks, tot, per)
        out[f"{var}_{mode}"] = dict(tot, clusters=cl, false=false, stacks=stacks, per=per)
json.dump(out, open(OUT / "summary.json", "w"), indent=1)
