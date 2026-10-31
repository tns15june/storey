"""Stair masks as alignment landmarks, from the cached unaligned runs: fit a similarity from each floor's mask centroids to the
reference prompt centres, then measure it on the verified landmarks. No API call."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import stack_sam as ss

truth = json.load(open(reg.OUT / "truth/truth.json"))
out = {}
for bid in ("bme_f", "bme_k", "schenley"):
    r = json.load(open(reg.OUT / f"sam/{bid}_none_r4_2000.json"))
    ref, boxes, absent = ss.boxes_for(bid)
    width = reg.sheet(bid, ref).shape[1]
    errs = []
    for f in range(len(reg.INDEX[bid])):
        if f == ref:
            continue
        src, dst = [], []
        for i, b in boxes.items():
            mk = r["masks"].get(f"{i}/{f}")
            if mk:
                src.append(mk["stack_xy"]); dst.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        if len(src) < 3:
            continue
        A, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst), method=cv2.RANSAC, ransacReprojThreshold=30)
        pts = [(v["floors"][str(f)]["xy"], v["ref"]) for v in truth[bid].values() if str(f) in v["floors"]]
        H = reg.h3(A)
        e = np.linalg.norm(reg.apply(H, [p[0] for p in pts]) - np.array([p[1] for p in pts], float), axis=1) * 100 / width
        errs += e.tolist()
        print(bid, f, "stairs", len(src), "inliers", int(inl.sum()), "median %", round(float(np.median(e)), 2))
    out[bid] = round(float(np.median(errs)), 3)
print(out)
json.dump(out, open(reg.OUT / "sam_landmarks.json", "w"), indent=1)
