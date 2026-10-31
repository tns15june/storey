"""No API. A sheet region with a pixel grid, for reading truth by eye. Usage: view.py <bid> <floor> [x1 y1 x2 y2 [zoom [step]]]"""
import sys
from pathlib import Path

import cv2
import numpy as np

SH = Path("/Users/tsharma/Github/Meta/data/lab/presence/sheets")
V = Path("/Users/tsharma/Github/Meta/data/lab/presence/truth_views")
b, f = sys.argv[1], sys.argv[2]
im = cv2.imread(str(SH / b / f"{f}.jpg"))
h, w = im.shape[:2]
a = list(map(float, sys.argv[3:]))
x1, y1, x2, y2 = (int(v) for v in a[:4]) if len(a) >= 4 else (0, 0, w, h)
z = a[4] if len(a) > 4 else 1.0
step = int(a[5]) if len(a) > 5 else 100
c = cv2.resize(im[y1:y2, x1:x2], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
o = c.copy()
for x in range((x1 // step + 1) * step, x2, step):
    X = int((x - x1) * z)
    cv2.line(o, (X, 0), (X, o.shape[0]), (0, 0, 255), 1)
    cv2.putText(o, str(x), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 255), 1, cv2.LINE_AA)
for y in range((y1 // step + 1) * step, y2, step):
    Y = int((y - y1) * z)
    cv2.line(o, (0, Y), (o.shape[1], Y), (0, 0, 255), 1)
    cv2.putText(o, str(y), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 255), 1, cv2.LINE_AA)
o = cv2.addWeighted(o, .45, c, .55, 0)
p = V / f"v_{b}_{f}.jpg"
cv2.imwrite(str(p), o, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(p, o.shape)
