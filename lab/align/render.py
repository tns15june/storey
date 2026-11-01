"""Render each stackable building's floors to 2000 px long-side JPEGs, bottom floor first, plus a contact sheet."""
import json
from pathlib import Path

import cv2
import fitz
import numpy as np

ROOT = Path(".")
OUT = ROOT / "data/lab/align/sheets"
PICK = ["bme_f", "bme_k", "schenley", "habs_stelizabeths_center", "habs_stelizabeths_atkins", "habs_va_bldg1",
        "mml_capitol", "mml_baxter", "wfu_efird", "nibs_clinic"]
LONG = 2000


def load(path, page):
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as d:
            p = d[page]
            s = LONG / max(p.rect.width, p.rect.height)
            pm = p.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
            im = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
            return im[:, :, ::-1].copy()
    im = cv2.imread(str(path))
    s = LONG / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


cat = {x["id"]: x for x in json.load(open(ROOT / "data/plans/catalog.json"))}
index = {}
for bid in PICK:
    d = OUT / bid
    d.mkdir(exist_ok=True)
    rows = []
    for i, f in enumerate(cat[bid]["floors"]):
        im = load(ROOT / "data/plans" / bid / f["file"], f["page"])
        name = f"{i}.jpg"
        cv2.imwrite(str(d / name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"i": i, "level": f["level"], "shape": im.shape[:2], "src": f["file"], "page": f["page"]})
    index[bid] = rows
    print(bid, [r["shape"] for r in rows])
    ims = [cv2.imread(str(d / f"{r['i']}.jpg")) for r in rows]
    th = [cv2.resize(x, (640, round(640 * x.shape[0] / x.shape[1]))) for x in ims]
    for t, r in zip(th, rows):
        cv2.putText(t, f"#{r['i']} L{r['level']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    H = max(t.shape[0] for t in th)
    th = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 0, 0)) for t in th]
    n = len(th)
    cols = 3 if n > 2 else 2
    gridrows = [th[i:i + cols] for i in range(0, n, cols)]
    for g in gridrows:
        while len(g) < cols:
            g.append(np.zeros_like(th[0]))
    cv2.imwrite(str(d / "contact.jpg"), np.vstack([np.hstack(g) for g in gridrows]), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump(index, open(OUT / "index.json", "w"), indent=1)
