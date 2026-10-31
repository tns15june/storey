"""Red = reference floor ink, cyan = other floor warped by blocks. Black where they agree. Also prints the fit per floor."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import truth_seed as ts

H_all = {}
strips = {}
for bid in ts.REF:
    ref = ts.REF[bid]
    n = len(reg.INDEX[bid])
    R = reg.sheet(bid, ref)
    x1, y1, x2, y2 = reg.plan_box(bid, ref, R.shape)
    tiles = []
    for f in range(n):
        if f == ref:
            H_all[f"{bid}/{f}"] = np.eye(3).tolist()
            continue
        c = reg.context(bid, f, ref)
        H = reg.blocks(c)
        H_all[f"{bid}/{f}"] = H.tolist()
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        th = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        print(f"{bid:26s} f{f} scale {s:.4f} rot {th:+.2f} shift ({H[0,2]:+.0f},{H[1,2]:+.0f}) inliers {c.get('inliers')}/{c.get('matches')} = {c.get('inliers', 0) / max(1, c.get('matches', 1)):.2f}", flush=True)
        a = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        b = reg.warp(cv2.cvtColor(reg.sheet(bid, f), cv2.COLOR_BGR2GRAY), H, a.shape)
        o = cv2.merge([a, a, b])  # BGR: reference shows cyan-free red channel from mov
        o = o[y1:y2, x1:x2]
        cv2.imwrite(str(reg.OUT / f"overlays/{bid}_f{f}.jpg"), o, [cv2.IMWRITE_JPEG_QUALITY, 85])
        tiles.append(o)
json.dump(H_all, open(reg.OUT / "H_blocks.json", "w"), indent=1)
