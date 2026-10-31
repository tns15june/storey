"""Overlay one Spark tile (red crosses) over the key's arc points (green circles) to check the mapping by eye."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/doors")
import cv2, fitz, numpy as np
import spark_tiles as s

A, L, arcs, reg = s.K.clinic_key("F1")
G = [s.arc_point(l) for l in L]
h = np.array(G)
box = (h[:, 0].min() - 30, h[:, 1].min() - 30, h[:, 0].max() + 30, h[:, 1].max() + 30)
ts = s.tiles(box, 3)
o = s.run_tile("F1", ts[4], 1600, "probe_t3_4", None)
print(len(o["doors"]), o["count"], o["info"], o["w"], o["h"])
r = ts[4]
im = s.K.render("F1", int(round(72 * 1600 / max(r[2] - r[0], r[3] - r[1]))), clip=fitz.Rect(*r))
f = lambda p: (int((p[0] - r[0]) / (r[2] - r[0]) * o["w"]), int((p[1] - r[1]) / (r[3] - r[1]) * o["h"]))
for p in s.to_sheet(o):
    cv2.drawMarker(im, f(p), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 2)
for g in G:
    cv2.circle(im, f(g), 12, (0, 180, 0), 2)
cv2.imwrite(str(s.K.OUT / "probe_tile.png"), im)
