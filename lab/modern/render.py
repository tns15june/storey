"""Render every floor sheet of the modern vector sets at 150 and 200 DPI. Run: .venv/bin/python -I lab/modern/render.py"""
import json, sys, time
from pathlib import Path
import pymupdf as fitz

ROOT = Path(".")
OUT = ROOT / "data/lab/modern/render"
IDS = "nibs_clinic wfu_efird wfu_huffman mml_capitol mml_baxter mml_abbot carpinteria_adu1 usc_leconte brandeis_hall".split()

if __name__ == "__main__":
    cat = {e["id"]: e for e in json.loads((ROOT / "data/plans/catalog.json").read_text())}
    rows = []
    for id in IDS:
        for f in cat[id]["floors"]:
            pg = fitz.open(ROOT / "data/plans" / id / f["file"])[f["page"]]
            for dpi in (150, 200):
                dst = OUT / f"{id}_{f['level']}_{dpi}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                t = time.time()
                pm = pg.get_pixmap(dpi=dpi, alpha=False)
                t1 = time.time() - t
                if not dst.exists():
                    pm.save(dst)
                rows.append(dict(id=id, level=f["level"], dpi=dpi, w=pm.width, h=pm.height, render_s=round(t1, 3)))
                print(rows[-1], flush=True)
    (ROOT / "data/lab/modern/render_times.json").write_text(json.dumps(rows, indent=1))
