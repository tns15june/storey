"""Does the inlier gate pass wrong fits when Spark's plan box wobbles? No API. Run with -I.
For each non-reference floor of the presence held-out buildings: jitter both plan boxes by a fixed share of the sheet, refit with reg.blocks, compare with the frozen fit at the truth object centres.
A false pass is a fit at or over the gate of 25 inliers that lands more than 40 px from the frozen fit, on a floor whose frozen fit is good. Also reproduces tightseed's wfu f3 case."""
import json, sys, time
import numpy as np, cv2
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/presence")
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
import pres
from pres import T, GATE, reg, ctr
pres.load_spark()
rng = np.random.default_rng(20261031)
res = []
for b in pres.HELD + ["bme_f", "habs_stelizabeths_center", "mml_capitol"]:
    A = pres.alignment(b); ref = pres.ref_of(b)
    pts = {f: [ctr(v["boxes"][f]) for v in T[b]["objects"].values() if f in v["boxes"]] for f in A}
    for f in A:
        if int(f) == ref:
            continue
        H0 = np.array(A[f]["H"])
        for level in (.02, .05):
            for k in range(2):
                c = reg.context(b, int(f), ref)
                h, w = c["ref_ink"].shape[:2]; hm, wm = c["mov_ink"].shape[:2]
                j = lambda box, W, H: tuple(int(np.clip(v + rng.uniform(-level, level) * s, 0, s)) for v, s in zip(box, (W, H, W, H)))
                c["mov_box"] = j(c["mov_box"], wm, hm); c["ref_box"] = j(c["ref_box"], w, h)
                t = time.time()
                try:
                    H = reg.blocks(c)
                except Exception as e:
                    res.append(dict(b=b, f=f, level=level, k=k, error=repr(e)[:80])); continue
                inl = c.get("inliers")
                q = pts[f]
                dev = float(np.max(np.hypot(*(reg.apply(H, q) - reg.apply(H0, q)).T))) if q else None
                res.append(dict(b=b, f=f, level=level, k=k, inliers=inl, frozen_inliers=A[f]["inliers"], frozen_aligned=A[f]["aligned"], passed=bool(inl is not None and inl >= GATE), dev_px=None if dev is None else round(dev, 1), s=round(time.time() - t, 1)))
                print(res[-1], flush=True)
                json.dump(res, open("/Users/tsharma/Github/Meta/data/lab/audit2/gate_jitter.json", "w"), indent=0)
import stacks as st
AUD = st.AUD
ims = [cv2.imread(str(AUD / "wfu_huffman" / f"f{i}.png")) for i in range(4)]
R = (101, 101, 1919, 2727)
c = st.ctx(ims[3], ims[1], R, R); H = reg.blocks(c)
print("tightseed wfu f3 to f1 with hand box: inliers", c.get("inliers"), "of", c.get("matches"), H.round(3).tolist(), flush=True)
