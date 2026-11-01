"""Tables for a split, wrong passes first. No API. Usage: evaluate.py dev|heldout. Writes data/lab/presence/eval_<split>.json."""
import json
import sys

sys.path.insert(0, "lab/presence")
import numpy as np

import pres
import rule
import score as S
from pres import OUT, T, TRUTH, box_dist

split = sys.argv[1]
B = [b for b, t in T.items() if t["split"] == ("dev" if split == "dev" else "heldout")]
out = dict(split=split, clean={}, planted=[])


def sheet_cost(b):
    s = pres.load_spark()
    rows = [s[f"{b}/{r['i']}"]["info"] for r in pres.INDEX[b]] + [s[f"joint/{b}"]["info"]]
    return sum(i["usd"] for i in rows), sum(i["seconds"] for i in rows)


tot = {}
print("CLEAN STACKS. wrong = Found where the thing is absent, by a truth stair absent there; phantom = Found with no truth object near it, unadjudicated.")
for fam in ("stair", "shaft"):
    print(f"--- {fam}: arm | {'building':26s} wrong phantom excused | false findings | check (of which truly there) | found ok | truth cover found/check/missing/none")
    for b in B:
        E = json.load(open(OUT / f"cells/{b}.json"))
        bl = S.baseline(b, fam)
        sr = S.score(b, S.single_run(b, b, 0, fam), fam)
        print(f" base      {b:26s} always-present wrong {bl['wrong']} (present {bl['present']}, absent {bl['absent']}) | run0 wrong {len(sr['wrong'])} phantom {len(sr['phantom'])} false {len(sr['false'])} found ok {sr['ok']} cover {sr['cover']}")
        out["clean"].setdefault(b, {}).setdefault(fam, {})["baseline"] = bl
        out["clean"][b][fam]["run0"] = sr
        for arm in rule.ARMS:
            d = [c for c in rule.apply_rule(E, arm) if c["fam"] == fam]
            s = S.score(b, d, fam)
            print(f" arm {arm}     {b:26s} wrong {len(s['wrong'])} phantom {len(s['phantom'])} excused {s['excused']} | false {len(s['false'])} | check {s['check']} ({s['check_true']}) | found ok {s['ok']} | cover {s['cover']}")
            if s["wrong"] or s["phantom"]:
                print("        ", [(x["obj"] or "PHANTOM", x["f"], x["loc"]) for x in s["wrong"] + s["phantom"]])
            if s["false"]:
                print("         false:", [(x["obj"], x["f"], x["loc"]) for x in s["false"]])
            out["clean"][b][fam][arm] = s
            a = tot.setdefault((fam, arm), dict(wrong=0, phantom=0, false=0, check=0, found=0, ok=0))
            for k, n in (("wrong", len(s["wrong"])), ("phantom", len(s["phantom"])), ("false", len(s["false"])), ("check", s["check"]), ("found", s["found"]), ("ok", s["ok"])):
                a[k] += n
    for arm in rule.ARMS:
        print(f" TOTAL arm {arm} {fam}", tot[(fam, arm)], "per building check", round(tot[(fam, arm)]["check"] / len(B), 1), "false", round(tot[(fam, arm)]["false"] / len(B), 1))
print("\nPLANTED. An erased stair is caught when no row at its place says Found. Worst row counts.")
import planted
keys = planted.DEV_KEYS if split == "dev" else planted.HELD_KEYS
res = {}
for key, (b, cases) in keys.items():
    p = OUT / f"cells/{key}.json"
    if not p.exists():
        continue
    E = json.load(open(p))
    for kind, o, f in cases:
        bx = T[b]["objects"][o]["boxes"][str(f)]
        row = dict(key=key, b=b, kind=kind, obj=o, floor=f)
        for arm in rule.ARMS:
            here = [c for c in rule.apply_rule(E, arm) if c["fam"] == "stair" and c["f"] == f and c["chk"] and box_dist(c["chk"], bx) <= S.D]
            st = [c["state"] for c in here if c["state"]]
            row[arm] = "Found" if "Found" in st else ("Check" if "Check" in st else ("Missing" if "Missing" in st else "no row"))
            row["v"] = max([c["v"] for c in here], default=None)
            row["why"] = [c["why"] for c in here if c["state"]]
        # run 0 alone, by place
        d = [c for c in S.single_run(key, b, 0) if c["f"] == f and box_dist(c["loc"], bx) <= S.D]
        row["run0"] = "Found" if any(c["state"] == "Found" for c in d) else ("Missing" if d else "no row")
        row["always"] = "Found"
        print(f" {key:42s} {kind:9s} {o:10s} f{f} v{row['v']} A:{row['A']} B:{row['B']} C:{row['C']} run0:{row['run0']} {row['why']}")
        out["planted"].append(row)
for kind in ("middle", "discharge", "top", None):
    rs = [r for r in out["planted"] if kind in (None, r["kind"])]
    if rs:
        print(kind or "ALL", "n", len(rs), {a: {s: sum(r[a] == s for r in rs) for s in ("Found", "Check", "Missing", "no row")} for a in ("A", "B", "C", "run0")}, "always-present wrong passes", len(rs))
json.dump(out, open(OUT / f"eval_{split}.json", "w"), indent=1, default=float)
