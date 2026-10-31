"""Crops of Spark points the IFC key calls false, for an eye check: is a door drawn there that the IFC lacks?"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, str(Path(__file__).parent))
import key as K
from spark_tiles import to_sheet, nms, arc_point, CACHE

A, L, arcs, reg = K.clinic_key("F1")
G = np.array([arc_point(l) for l in L])
outs = [json.loads(f.read_text()) for f in sorted(CACHE.glob("F1_t5low_*.json"))]
sp = np.array(nms([p for o in outs for p in to_sheet(o)], 0.45 * K.PT_PER_M))
tol = 0.8 * K.PT_PER_M
D = np.linalg.norm(sp[:, None] - G[None], axis=2)
r, c = linear_sum_assignment(np.minimum(D, tol + 1))
tp = {i for i, j in zip(r, c) if D[i, j] <= tol}
fp = [i for i in range(len(sp)) if i not in tp]
print(len(sp), len(tp), len(fp))
im = K.render("F1", 150)
z = 150 / 72
tiles_ = []
for n, i in enumerate(fp):
    x, y = (sp[i] * z).astype(int)
    pad = 100
    t = im[max(y - pad, 0):y + pad, max(x - pad, 0):x + pad].copy()
    t = cv2.copyMakeBorder(t, 0, 200 - t.shape[0], 0, 200 - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.drawMarker(t, (100, 100), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(t, str(n), (3, 14), 0, 0.5, (255, 0, 0), 1)
    tiles_.append(t)
rows = [np.hstack(tiles_[i:i + 7] + [np.full((200, 200, 3), 255, np.uint8)] * (7 - len(tiles_[i:i + 7]))) for i in range(0, len(tiles_), 7)]
cv2.imwrite(str(K.OUT / "review_fp_F1_t5low.png"), np.vstack(rows))
