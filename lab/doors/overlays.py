"""Eyeball overlays from cached Spark calls: green = key, red = Spark."""
import json, sys
from pathlib import Path
import cv2, fitz, numpy as np
sys.path.insert(0, "lab"); sys.path.insert(0, str(Path(__file__).parent))
import key as K
from spark_tiles import to_sheet, nms, arc_point, CACHE

A, L, arcs, reg = K.clinic_key("F1")
G = [arc_point(l) for l in L]
h = np.array(G)
b = (h[:, 0].min() - 30, h[:, 1].min() - 30, h[:, 0].max() + 30, h[:, 1].max() + 30)
outs = [json.loads(f.read_text()) for f in sorted(CACHE.glob("F1_t5low_*.json"))]
sp = nms([p for o in outs for p in to_sheet(o)], 0.45 * K.PT_PER_M)
im = K.render("F1", 110, clip=fitz.Rect(*b))
z = 110 / 72
for g in G:
    cv2.circle(im, (int((g[0] - b[0]) * z), int((g[1] - b[1]) * z)), 11, (0, 170, 0), 2)
for p in sp:
    cv2.drawMarker(im, (int((p[0] - b[0]) * z), int((p[1] - b[1]) * z)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
cv2.imwrite(str(K.OUT / "overlay_F1_t5low.png"), im)
LAB = json.load(open(Path(__file__).parent / "labels_af_ces_admin.json"))
pg = fitz.open(str(K.ROOT / LAB["sheet"]))[LAB["page"]]
b = LAB["plan_box"]
pix = pg.get_pixmap(dpi=170, clip=fitz.Rect(*b))
im = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()
z = 170 / 72
for g in LAB["interior"] + [p for v in LAB["exterior"].values() for p in v]:
    cv2.circle(im, (int((g[0] - b[0]) * z), int((g[1] - b[1]) * z)), 11, (0, 170, 0), 2)
outs = [json.loads(f.read_text()) for f in sorted(CACHE.glob("admin_g6x3_*.json"))]
for p in nms([p for o in outs for p in to_sheet(o)], 0.45 * LAB["pt_per_m"]):
    cv2.drawMarker(im, (int((p[0] - b[0]) * z), int((p[1] - b[1]) * z)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
cv2.imwrite(str(K.OUT / "overlay_admin_g6x3.png"), im)
