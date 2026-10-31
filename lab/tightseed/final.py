"""Last numbers from the caches, no API: masks of any confidence on absent floors, Round 1's three stacks at stair level, confidence margin on absent floors, one-call listing check, OR fusion post hoc."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import held_score as hs, floors as fl

print("masks of any confidence on absent floors, tracks seeded on the stair; scored stairs only")
for me in ("hand_960", "loose_960", "snap_960", "sparkcrop_960", "d_960", "P_960", "Ppin_960"):
    n = 0; top = []
    for st in HELD:
        tr = hs.load(f"{st}__{me}")
        rows, _ = score(st, tr)
        rows = [r for r in rows if r["cls"] in SCORED]
        n += hs.anyconf_ghosts(rows, tr, st)
        own = {}
        for o, f, b in tr["seeds"]:
            for k in assign(st, b):
                own.setdefault(k, set()).add(str(o))
        for r in rows:
            if not r["present"]:
                for o in own.get(r["stair"], ()):
                    mk = tr["tracks"].get(o, {}).get(str(r["floor"]))
                    if mk and mk["area"] > 0:
                        top.append(mk["conf"] or 0)
    print(me, "any-conf masks", n, "highest", sorted(top)[-4:])
r1 = ("bme_k", "habs_va_bldg1", "wfu_huffman")
print("Round 1 stacks, main only, tightseed frames")
for me in ("hand_960", "loose_960"):
    rows = []
    for st in r1:
        rr, _ = score(st, hs.load(f"{st}__{me}"))
        rows += hs.tag_rows(st, rr)
    print(me, summary(rows, ("main",)))
print("royce one-call listing:", [(l["name"], [(f["present"], f["x"], f["y"]) for f in l["floors"]]) for l in json.load(open(SPK / "queens_royce1_floors.json"))["out"]["stairs"]])
T = TRUTH["queens_royce1"]; c = T["crop"]
for k in ("R_G", "R_I"):
    b = T["stairs"][k]["boxes"]["0"]; print(k, "centre in 0-1000:", round(((b[0] + b[2]) / 2 - c[0]) * 1000 / (c[2] - c[0])), round(((b[1] + b[3]) / 2 - c[1]) * 1000 / (c[3] - c[1])))
print("OR fusion post hoc: Spark one-call strict OR recipe P mask")
allr = []
for st in HELD:
    F = {(r["stair"], r["floor"]): r for r in fl.score_floors(st)}
    rows, _ = score(st, hs.load(f"{st}__P_960"))
    for r in rows:
        f = F[(r["stair"], r["floor"])]
        spark_says = f["ok"] == f["present"]  # one-call says present
        sam_says = any(c[1] for c in r["claims"])
        said = spark_says or sam_says
        allr.append(dict(r, stack=st, ok=said == r["present"], ghosts=int(said and not r["present"])))
print(summary(allr))
print("spent", round(m.spent("tightseed"), 4))
