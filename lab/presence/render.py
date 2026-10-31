"""Render candidate floors at 2000 px long side, the same way as lab/align/render.py, bottom floor first, plus a contact sheet. No API."""
import json
from pathlib import Path

import cv2
import fitz
import numpy as np

ROOT = Path("/Users/tsharma/Github/Meta")
OUT = ROOT / "data/lab/presence/sheets"
LONG = 2000
PICK = ["bme_f", "habs_stelizabeths_center", "nibs_clinic", "mml_capitol", "bme_k", "habs_va_bldg1", "wfu_huffman",
        "schenley", "habs_stelizabeths_atkins", "queens_leonard", "queens_royce", "mml_abbot"]


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
    d.mkdir(exist_ok=True, parents=True)
    rows, th = [], []
    for i, f in enumerate(cat[bid]["floors"]):
        im = load(ROOT / "data/plans" / bid / f["file"], f["page"])
        cv2.imwrite(str(d / f"{i}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"i": i, "level": f["level"], "shape": im.shape[:2], "src": f["file"], "page": f["page"]})
        t = cv2.resize(im, (640, round(640 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"#{i} L{f['level']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        th.append(t)
    index[bid] = rows
    H = max(t.shape[0] for t in th)
    th = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 0, 0)) for t in th]
    cols = 3 if len(th) > 2 else 2
    gr = [th[i:i + cols] for i in range(0, len(th), cols)]
    for g in gr:
        while len(g) < cols:
            g.append(np.zeros_like(g[0]))
    cv2.imwrite(str(d / "contact.jpg"), np.vstack([np.hstack(g) for g in gr]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(bid, [(r["level"], r["shape"]) for r in rows], flush=True)
json.dump(index, open(OUT / "index.json", "w"), indent=1)
