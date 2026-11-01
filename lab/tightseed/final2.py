"""Cache arithmetic, no API: absent floors among seeded stairs, precision of SAM silence as an ending claim, and the second-opinion AND with Spark's one-call list."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
import held_score as hs, floors as fl


def table(me, stacks):
    rows = []
    for st in stacks:
        rr, _ = score(st, hs.load(f"{st}__{me}"))
        rows += hs.tag_rows(st, [r for r in rr if r["cls"] in SCORED])
    seeded = [r for r in rows if r["seeded"]]
    ab = [r for r in seeded if not r["present"]]; pr = [r for r in seeded if r["present"]]
    silent_true = sum(not any(c[1] for c in r["claims"]) for r in ab)
    silent_false = sum(not any(c[1] for c in r["claims"]) for r in pr)
    wrong = sum(any(c[1] for c in r["claims"]) for r in ab)
    return dict(seeded_absent_right=f"{silent_true}/{len(ab)}", wrong_passes=wrong, silence_is_end=f"{silent_true}/{silent_true + silent_false}", present_seeded_with_mask=f"{sum(any(c[1] for c in r['claims']) for r in pr)}/{len(pr)}")


for me in ("hand_960", "loose_960", "snap_960", "sparkcrop_960", "d_960", "P_960", "Ppin_960"):
    print(me, table(me, HELD))
for me in ("hand_960", "loose_960", "P_960", "Ppin_960"):
    print("fresh", me, table(me, FRESH))
print("frozen arms exclude P; P is post hoc")
print("hand seeded, held-out plus fresh", table("hand_960", HELD + FRESH))
print("P seeded, held-out plus fresh, post hoc", table("P_960", HELD + FRESH))
print("Ppin seeded, held-out plus fresh, post hoc", table("Ppin_960", HELD + FRESH))
# second opinion, post hoc: flag an ending only when Spark's one-call list says absent and SAM on the P seed is silent
tp = fp = fn = spark_only_fp = spark_only_tp = 0
for st in HELD:
    F = {(r["stair"], r["floor"]): r for r in fl.score_floors(st)}
    rows, _ = score(st, hs.load(f"{st}__P_960"))
    for r in rows:
        if r["cls"] not in SCORED or not r["seeded"]:
            continue
        f = F[(r["stair"], r["floor"])]
        spark_absent = f["ok"] != f["present"]
        sam_silent = not any(c[1] for c in r["claims"])
        flag = spark_absent and sam_silent
        tp += flag and not r["present"]; fp += flag and r["present"]; fn += (not flag) and (not r["present"])
        spark_only_tp += spark_absent and not r["present"]; spark_only_fp += spark_absent and r["present"]
print("ending flag, Spark one-call alone on seeded stairs: true", spark_only_tp, "false", spark_only_fp)
print("ending flag, Spark one-call absent AND SAM-P silent: true", tp, "false", fp, "true endings not flagged", fn)
