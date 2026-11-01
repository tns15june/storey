"""Seed IoU rows and prior exposure of the fresh window. No API. Run with -I."""
import json, sys, glob
import numpy as np
sys.path.insert(0, "lab/audit2")
import rescore_tightseed as S
for me in ("loose_960", "sparkcrop_960", "snap_960", "d_960", "P_960"):
    vals = []
    for st in S.HELD:
        tr = S.load(st, me)
        for k, s in S.ALL[st]["stairs"].items():
            if s["cls"] not in S.SC:
                continue
            hits = [(f, b) for o, f, b in tr["seeds"] if any(S.hit(b, x) for x in s["boxes"].values())]
            if not hits:
                vals.append(0.0); continue
            for f, b in hits:
                t = s["boxes"].get(str(f))
                vals.append(S.iou(b, t) if t else 0.0)
    a = np.array(vals)
    print(me, "n", len(a), "median", round(float(np.median(a)), 3), ">=.5", int((a >= .5).sum()), "zero", int((a == 0).sum()))
for st, k in (("wfu_huffman", "central"), ("mml_abbot", "ext_stair")):
    tr = S.load(st, "vector_960"); s = S.ALL[st]["stairs"][k]
    print("vector", st, [round(S.iou(b, s["boxes"][str(f)]), 2) for o, f, b in tr["seeds"] if str(f) in s["boxes"] and any(S.hit(b, x) for x in s["boxes"].values())])
for p in ("data/lab/align/spark_sheets.json", "data/lab/presence/spark_sheets.json"):
    d = json.load(open("" + p))
    print(p, "royce keys", [k for k in d if "royce" in k or "queens" in k][:10])
print(open("lab/tightseed/truth_fresh.py").read()[:900])
