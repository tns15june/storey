"""Does a page have usable vector content? Rule tested on every floor sheet in the catalog plus two synthetic fall-through cases.

Run: .venv/bin/python -I lab/modern/detect.py
"""
import json, re, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern")
import pymupdf as fitz
import vec

ROOT = vec.ROOT
OUT = ROOT / "data/lab/modern"
CAT = json.loads((ROOT / "data/plans/catalog.json").read_text())


def classify(pg):
    """raster: an image covers over half the page and few paths sit on top. Else vector, split by whether text and layers exist."""
    area = abs(pg.rect) or 1.0
    img = 0.0
    for im in pg.get_images(full=True):
        for r in pg.get_image_rects(im[0]):
            img = max(img, abs(r & pg.rect) / area)
    paths = len(pg.get_drawings())
    words = len(pg.get_text("words"))
    kind = "raster" if img > 0.5 and paths < 500 else "vector" if paths >= 200 else "empty_or_raster"
    return dict(kind=kind, image_cover=round(img, 3), paths=paths, words=words, layers=len(pg.parent.get_ocgs()))


def caps(pg, c):
    """What the vector path can do on this page."""
    if c["kind"] != "vector":
        return "raster path"
    t = pg.get_text()
    scale = bool(re.search(r"\d(?:/\d+)?\"\s*=\s*1'-0\"|1\s*:\s*\d{2,3}", t))
    return ("layers" if c["layers"] else "no layers") + (", text" if c["words"] > 30 else ", no text layer") + (", printed scale" if scale else ", no scale in text")


if __name__ == "__main__":
    rows, conf = [], {}
    for e in CAT:
        if e["kind"] not in ("vector", "scan"):
            continue
        for f in e["floors"]:
            p = ROOT / "data/plans" / e["id"] / f["file"]
            if p.suffix.lower() != ".pdf":
                k = (e["kind"], "image file, raster path")
                conf[k] = conf.get(k, 0) + 1
                rows.append(dict(id=e["id"], level=str(f["level"]), truth=e["kind"], kind="raster", note="image file"))
                continue
            pg = fitz.open(p)[f["page"]]
            t = time.time()
            c = classify(pg)
            c.update(id=e["id"], level=str(f["level"]), truth=e["kind"], caps=caps(pg, c), seconds=round(time.time() - t, 3))
            rows.append(c)
            conf[(e["kind"], c["kind"])] = conf.get((e["kind"], c["kind"]), 0) + 1
    print("confusion", conf)
    for r in rows:
        print(r["id"], r["level"], r["truth"], r["kind"], {k: r.get(k) for k in ("image_cover", "paths", "words", "layers", "caps", "seconds")})
    # synthetic: a vector sheet rasterised into an image-only PDF, then with an OCR style hidden text layer
    d = fitz.open(ROOT / "data/plans/mml_capitol/THE-CAPITOL.pdf")
    pix = d[3].get_pixmap(dpi=150, alpha=False)
    tmp = fitz.open()
    pg = tmp.new_page(width=d[3].rect.width, height=d[3].rect.height)
    pg.insert_image(pg.rect, pixmap=pix)
    a = classify(pg)
    pg.insert_text((100, 100), "Main Floor Plan 1/4\" = 1'-0\" " * 20, render_mode=3, fontsize=8)
    b = classify(pg)
    print("rasterised", a, "with hidden text", b)
    assert a["kind"] == "raster" and b["kind"] == "raster"
    (OUT / "detect.json").write_text(json.dumps(dict(rows=rows, conf={f"{k[0]}->{k[1]}": v for k, v in conf.items()}, synth=[a, b]), indent=1))
