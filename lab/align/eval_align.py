"""Part 2: every method on every floor of every building with verified landmarks. Error is the distance between a landmark
carried from its floor to the reference floor and the same landmark there, as a percent of sheet width."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import truth_seed as ts

truth = json.load(open(reg.OUT / "truth/truth.json"))
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(reg.METHODS)
results = {}
rows = []
for bid, lms in truth.items():
    ref = ts.REF[bid]
    shape = reg.sheet(bid, ref).shape
    width = shape[1]
    box = reg.plan_box(bid, ref, shape)
    plan_w = box[2] - box[0]
    n = len(reg.INDEX[bid])
    for f in range(n):
        if f == ref:
            continue
        c = reg.context(bid, f, ref)
        for name in names:
            t0 = time.time()
            H = reg.METHODS[name](c)
            dt = time.time() - t0
            pts = [(k, v["floors"][str(f)]["xy"], v["ref"], v["kind"]) for k, v in lms.items() if str(f) in v["floors"]]
            if H is None:
                err = [float("inf")] * len(pts)
            else:
                err = np.linalg.norm(reg.apply(H, [p[1] for p in pts]) - np.array([p[2] for p in pts], float), axis=1).tolist()
            for p, e in zip(pts, err):
                rows.append({"bid": bid, "floor": f, "method": name, "lm": p[0], "kind": p[3], "err_px": e, "err_pct": 100 * e / width, "err_plan_pct": 100 * e / plan_w, "s": dt})
json.dump(rows, open(reg.OUT / f"align_rows_{sys.argv[2] if len(sys.argv) > 2 else 'all'}.json", "w"))
b = list(truth)
print(f"{'method':20s}" + "".join(f"{x[:11]:>12s}" for x in b) + f"{'pass':>6s}{'worstfl':>9s}{'s':>7s}")
for name in names:
    meds, worst = [], []
    for x in b:
        r = [q for q in rows if q["method"] == name and q["bid"] == x]
        meds.append(np.median([q["err_pct"] for q in r]))
        worst.append(max(np.median([q["err_pct"] for q in r if q["floor"] == f]) for f in {q["floor"] for q in r}))
    s = np.mean([q["s"] for q in rows if q["method"] == name])
    print(f"{name:20s}" + "".join(f"{m:12.2f}" for m in meds) + f"{sum(m < 1 for m in meds):6d}{max(worst):9.2f}{s:7.2f}")
print("median error as percent of plan-box width")
for name in names:
    print(f"{name:20s}" + "".join(f"{np.median([q['err_plan_pct'] for q in rows if q['method'] == name and q['bid'] == x]):12.2f}" for x in b))
print("stair landmarks only / non-stair only, method blocks")
for kind in ("stair", "corner"):
    print(f"{kind:20s}" + "".join(f"{np.median([q['err_pct'] for q in rows if q['method'] == 'blocks' and q['bid'] == x and q['kind'] == kind] or [float('nan')]):12.2f}" for x in b))
print("worst floor per building (median of that floor)")
for name in names:
    print(f"{name:20s}" + "".join(f"{max(np.median([q['err_pct'] for q in rows if q['method']==name and q['bid']==x and q['floor']==f]) for f in {q['floor'] for q in rows if q['bid']==x}):12.2f}" for x in b))
