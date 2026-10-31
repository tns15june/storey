"""Dev only, no API. Evidence by truth state for established rows: crop answers and ink at each vote level, on clean and planted stacks."""
import json
import sys
from collections import Counter

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
import pres
from pres import OUT, T
import score as S

keys = sys.argv[1:]
stats = Counter()
inks = {}
for key in keys:
    E = json.load(open(OUT / f"cells/{key}.json"))
    b = E["b"]
    plant = {}
    if "__" in key:
        for kind, o, f in pres.TRUTH["plant"][b]:
            plant[(o, f)] = kind
    print("==", key)
    for r in E["rows"]:
        if not r["established"]:
            continue
        for f, c in r["cells"].items():
            f = int(f)
            if c["chk"] is None:
                continue
            pr, un = S.near(b, f, c["chk"], r["fam"])
            isplant = any((o, f) in plant for o in pr)
            t = "planted" if isplant else ("present" if pr else ("unsure" if un else "absent"))
            ink = r["ink30"].get(str(f))
            st = (r["fam"], t, c["v"], c.get("crop"))
            stats[st] += 1
            inks.setdefault((r["fam"], t, min(c["v"], 2) if c["v"] else 0), []).append(ink)
            if isplant or (t == "absent" and c.get("crop") != "no") or (t == "present" and c.get("crop") != "yes"):
                print(f"  {r['fam'][:5]} {[round(x) for x in r['pos']] if r['pos'] else None} f{f} {t} v{c['v']} crop {c.get('crop')} ink30 {ink} ink50 {r['ink50'].get(str(f))} near {pr or un} seen {str(c.get('seen'))[:70]}")
print()
for k, v in sorted(stats.items(), key=str):
    print(k, v)
print()
for k, v in sorted(inks.items(), key=str):
    w = sorted(x for x in v if x is not None)
    print(k, "n", len(v), "none", sum(x is None for x in v), "ink30 min/median/max", (w[0], w[len(w) // 2], w[-1]) if w else None, "low5", w[:5])
