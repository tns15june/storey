"""What does the concept detector see on a drawing at all? A wide word list on plans, plus a positive control on flat shapes."""
import json

import cv2
import fitz
import numpy as np

import core as c
from data import OUT, ROOT, load, select

T = select()
WORDS = ["text", "letter", "number", "circle", "line", "arrow", "toilet", "bathtub", "sink", "table", "chair", "bed", "car", "drawing", "floor plan", "blueprint",
         "square", "triangle", "shape", "paper", "window", "diagram", "stairs", "red rectangle", "blue rectangle"]


def pdf_crop(path, page, dpi, box):
    p = fitz.open(path)[page]
    pm = p.get_pixmap(dpi=dpi, clip=fitz.Rect(*box))
    return cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3], cv2.COLOR_RGB2BGR)


cap = pdf_crop(ROOT / "data/plans/mml_capitol/THE-CAPITOL.pdf", 3, 150, (1650, 480, 2150, 1300))
clinic = pdf_crop(ROOT / "data/plans/nibs_clinic/2012-03-27-ClinicRedactedDrawings/Clinic_052_Composite First Floor Plan.pdf", 0, 200, (1400, 800, 1900, 1300))
ctrl = np.full((1000, 1000, 3), 255, np.uint8)
for i, (x, y, col) in enumerate([(100, 100, (0, 0, 255)), (500, 150, (255, 0, 0)), (200, 550, (0, 0, 255)), (600, 600, (255, 0, 0))]):
    cv2.rectangle(ctrl, (x, y), (x + 280, y + 220), col, -1)
fillp = c.fill(load(T["plans"][3])[0])
images = {"ctrl": ctrl, "capitol": cap, "clinic": clinic, "cubi": load(T["plans"][3])[0], "cubi_fill": fillp}
cv2.imwrite(str(OUT / "vocab_capitol.jpg"), cap)
cv2.imwrite(str(OUT / "vocab_clinic.jpg"), clinic)
rows = c.pmap(lambda j: (j[0], j[1], [(r[1], r[2]) for r in c.sam(j[1], images[j[0]])]), [(k, w) for k in images for w in WORDS])
(OUT / "vocab.json").write_text(json.dumps(rows, default=float))
tiles = []
for k, w, rs in rows:
    if rs:
        print(k, w, len(rs), "conf", round(max(x[1] or 0 for x in rs), 2))
        if k != "ctrl" and len(tiles) < 16:
            v = images[k].copy()
            for _, _, _, mk in c.sam(w, images[k]):
                v = c.overlay(v, mk)
            tiles.append((v, f"{k} / {w}: {len(rs)}"))
if tiles:
    c.sheet(tiles, OUT / "vocab_sheet.jpg")
print("spent", round(c.m.spent("samcraft"), 3))
