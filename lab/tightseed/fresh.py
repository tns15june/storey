"""POST HOC recipe P on a stack no model has seen. Truth was written first. Prints wrong passes first."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
import dev1, post1, post2, multi as mu, held_score as hs

st = "royce_east"
cl = dev1.loose_clusters(st); fr = frames(st)
json.dump(cl, open(SPK / f"{st}_clusters.json", "w"))
print("clusters", len(cl), [(c["seed_floor"], [round(v) for v in c["seed_box"]], c["floors"]) for c in cl], "spent", round(m.spent("tightseed"), 3), flush=True)
hand = track(f"{st}__hand_960", st, dev1.hand_seeds(st))
loose = track(f"{st}__loose_960", st, [(i + 1, c["seed_floor"], c["seed_box"]) for i, c in enumerate(cl)])
sb = post2.p_seeds(st, cl, fr)
P = track(f"{st}__P_960", st, sb)
pin = track(f"{st}__Ppin_960", st, [(o, f, b) for (o, _, b), c in zip(sb, cl) for f in c["floors"]])
print("spent", round(m.spent("tightseed"), 3), flush=True)
for name, tr in (("always", None), ("hand", hand), ("loose", loose), ("P", P), ("Ppin", pin), ("spark_members", hs.spark_member_track(st))):
    if tr is None:
        continue
    rows, false = score(st, tr)
    print(name, summary(rows), "false", len(false), "wrong passes", [(r["stair"], r["floor"], [c[3] for c in r["claims"] if c[1]]) for r in rows if r["cls"] in SCORED and not r["present"] and not r["ok"]], "misses", [(r["stair"], r["floor"]) for r in rows if r["cls"] in SCORED and r["present"] and not r["ok"]], flush=True)
print("seed IoU loose", [x[1:] for x in hs.seed_iou_rows(st, loose)])
print("seed IoU P", [x[1:] for x in hs.seed_iou_rows(st, P)])
