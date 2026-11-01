"""Two controls, no API. (1) Is the Spark plan box needed? Compare the whole sheet and jittered boxes. (2) Does the inlier ratio catch a bad fit?
Register different buildings to each other as a negative control."""
import json
import sys

import numpy as np

sys.path.insert(0, "lab/align")
import reg
import truth_seed as ts

truth = json.load(open(reg.OUT / "truth/truth.json"))
rng = np.random.default_rng(0)


def jitter(box, shape, frac):
    d = rng.uniform(-frac, frac, 4) * max(shape[:2])
    x1, y1, x2, y2 = np.array(box, float) + d
    return (max(0, int(x1)), max(0, int(y1)), min(shape[1], int(x2)), min(shape[0], int(y2)))


rows = {}
for mode in ("spark", "whole_sheet", "jitter3pct", "jitter6pct"):
    errs, floors = {}, []
    for bid, lms in truth.items():
        ref = ts.REF[bid]
        width = reg.sheet(bid, ref).shape[1]
        e = []
        for f in range(len(reg.INDEX[bid])):
            if f == ref:
                continue
            c = reg.context(bid, f, ref)
            if mode == "whole_sheet":
                h, w = c["mov_ink"].shape
                H_, W_ = c["ref_ink"].shape
                c["mov_box"], c["ref_box"] = (0, 0, w, h), (0, 0, W_, H_)
            elif mode != "spark":
                fr = 0.03 if mode == "jitter3pct" else 0.06
                c["mov_box"], c["ref_box"] = jitter(c["mov_box"], c["mov_ink"].shape, fr), jitter(c["ref_box"], c["ref_ink"].shape, fr)
            H = reg.blocks(c)
            pts = [(v["floors"][str(f)]["xy"], v["ref"]) for v in lms.values() if str(f) in v["floors"]]
            d = np.linalg.norm(reg.apply(H, [p[0] for p in pts]) - np.array([p[1] for p in pts], float), axis=1) * 100 / width
            e += d.tolist()
            floors.append(float(np.median(d)))
        errs[bid] = round(float(np.median(e)), 2)
    rows[mode] = {"medians": errs, "pass": sum(v < 1 for v in errs.values()), "worst_floor": round(max(floors), 2), "floors_over_1pct": sum(x >= 1 for x in floors), "n_floors": len(floors)}
    print(mode, rows[mode], flush=True)

neg = []
ids = ["bme_f", "bme_k", "schenley", "habs_stelizabeths_center", "habs_va_bldg1", "mml_capitol", "nibs_clinic", "wfu_efird"]
for a in ids:
    for b in ids:
        if a == b:
            continue
        ctx = reg.context.__wrapped__ if hasattr(reg.context, "__wrapped__") else None
        A, B = reg.sheet(a, ts.REF[a]), reg.sheet(b, ts.REF[b])
        import cv2
        c = {"mov_ink": reg.ink(A), "ref_ink": reg.ink(B), "mov_box": reg.plan_box(a, ts.REF[a], A.shape), "ref_box": reg.plan_box(b, ts.REF[b], B.shape)}
        c["mov_soft"], c["ref_soft"] = cv2.GaussianBlur(c["mov_ink"], (0, 0), 1.2), cv2.GaussianBlur(c["ref_ink"], (0, 0), 1.2)
        reg.blocks(c)
        neg.append({"mov": a, "ref": b, "inliers": c.get("inliers", 0), "matches": c.get("matches", 0)})
ratio = [n["inliers"] / max(1, n["matches"]) for n in neg]
print("negative control pairs", len(neg), "ratio median", round(float(np.median(ratio)), 2), "max", round(max(ratio), 2), "inliers max", max(n["inliers"] for n in neg))
print("top", sorted(((round(n["inliers"] / max(1, n["matches"]), 2), n["inliers"], n["mov"], n["ref"]) for n in neg), reverse=True)[:5])
json.dump({"modes": rows, "negative_control": neg}, open(reg.OUT / "robustness.json", "w"), indent=1)
