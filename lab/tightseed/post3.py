"""POST HOC pooled scores for recipe P on the held-out stacks, with seed IoU and the wrong passes first. No API."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
import held_score as hs, multi as mu
stacks = sys.argv[1:] or HELD
memo = {}
for me in ("P_960", "Ppin_960"):
    for lab in ("raw", "tread", "spark"):
        allr = []; false = 0; ious = []
        for st in stacks:
            tr = hs.load(f"{st}__{me}")
            gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames(st)] if lab == "tread" else None
            chk = None if lab == "raw" else (hs.tread_checker(st, gr, memo) if lab == "tread" else hs.spark_checker(st, me))
            rows, fl = score(st, tr, chk); false += len(fl)
            allr += hs.tag_rows(st, rows)
            if me == "P_960":
                ious += hs.seed_iou_rows(st, tr)
        g = [(r["stack"], r["stair"], r["floor"], [c[3] for c in r["claims"] if c[1]]) for r in allr if r["cls"] in SCORED and not r["present"] and not r["ok"]]
        print("POST HOC", me, lab, summary(allr), "false", false, "WRONG PASSES", g)
        if ious and lab == "raw":
            a = np.array([x[4] for x in ious if x[2] in SCORED])
            print("   seed IoU n", len(a), "median", round(float(np.median(a)), 3), ">=.5", int((a >= .5).sum()), ">=.3", int((a >= .3).sum()), "zero", int((a == 0).sum()))
            for st in stacks:
                b = np.array([x[4] for x in ious if x[0] == st and x[2] in SCORED])
                print("     ", st, len(b), round(float(np.median(b)), 3), int((b >= .5).sum()))
print("spent", round(m.spent("tightseed"), 4))
