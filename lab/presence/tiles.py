"""No API. The same building location on every floor, side by side, for an eye call. Usage: tiles.py <bid> <tag> <half> <zoom> x,y [x,y ...]
x,y are on the reference floor (index 1, or 0 for two floors); other floors are placed by a least-squares similarity through the draft boxes."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("/Users/tsharma/Github/Meta/data/lab/presence")
b, tag, half, zoom = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
pts = [tuple(map(float, a.split(","))) for a in sys.argv[5:]]
n = len(json.load(open(OUT / "sheets/index.json"))[b])
ref = min(1, n - 1)
draft = json.load(open(OUT / "draft.json")).get(b, {})
ims = [cv2.imread(str(OUT / f"sheets/{b}/{i}.jpg")) for i in range(n)]


def ctr(v, f):
    x1, y1, x2, y2 = v["boxes"][str(f)]
    return (x1 + x2) / 2, (y1 + y2) / 2


T = {}
for f in range(n):
    if f == ref or not draft:
        T[f] = np.eye(3)
        continue
    a = np.float32([ctr(v, ref) for v in draft.values()])
    c = np.float32([ctr(v, f) for v in draft.values()])
    A, _ = cv2.estimateAffinePartial2D(a, c, method=cv2.RANSAC, ransacReprojThreshold=30)
    T[f] = np.vstack([A, [0, 0, 1]])
rows = []
S = int(2 * half * zoom)
for k, (x, y) in enumerate(pts):
    ts = []
    for f in range(n):
        q = T[f] @ np.array([x, y, 1.0])
        cx, cy = int(q[0]), int(q[1])
        pad = cv2.copyMakeBorder(ims[f], half, half, half, half, cv2.BORDER_CONSTANT, value=(180, 180, 180))
        t = cv2.resize(pad[max(cy, 0):cy + 2 * half, max(cx, 0):cx + 2 * half], (S, S), interpolation=cv2.INTER_CUBIC)
        for g in range((cx - half) // 50 * 50 + 50, cx + half, 50):
            X = int((g - cx + half) * zoom); cv2.line(t, (X, 0), (X, 5), (0, 0, 255), 1)
            cv2.putText(t, str(g), (X + 1, 15), cv2.FONT_HERSHEY_SIMPLEX, .33, (0, 0, 255), 1, cv2.LINE_AA)
        for g in range((cy - half) // 50 * 50 + 50, cy + half, 50):
            Y = int((g - cy + half) * zoom); cv2.line(t, (0, Y), (5, Y), (0, 0, 255), 1)
            cv2.putText(t, str(g), (6, Y + 3), cv2.FONT_HERSHEY_SIMPLEX, .33, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(t, f"f{f}", (S - 30, S - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 0, 0), 1, cv2.LINE_AA)
        ts.append(cv2.copyMakeBorder(t, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    rows.append(np.hstack(ts))
p = OUT / f"truth_views/t_{b}_{tag}.jpg"
cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(p)
