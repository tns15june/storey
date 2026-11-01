"""Measure every PDF under data/plans: vector vs scan, floor-plan pages, code-sheet pages. Renders plan pages to PNG.

Run: .venv/bin/python -I lab/gather_data/inventory.py
Output: data/lab/gather_data/inventory.json and data/lab/gather_data/pages/<id>/*.png
"""
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent.parent
PLANS = ROOT / "data/plans"
OUT = ROOT / "data/lab/gather_data"
FLOOR = re.compile(r"((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|GROUND|LOWER|UPPER|BASEMENT|ROOF|MEZZANINE|1ST|2ND|3RD|4TH|5TH|LEVEL\s*[-\dA-Z]{1,3})[\s-]*(?:LEVEL|FLOOR|STOREY)?[\s-]*(?:OVERALL\s+|ARCHITECTURAL\s+|LIFE SAFETY\s+|DIMENSION\s+)*(?:FLOOR\s+)?PLAN)", re.I)
CODE = re.compile(r"CODE (SUMMARY|ANALYSIS|INFORMATION|DATA|REVIEW|COMPLIANCE)|LIFE[- ]SAFETY|EGRESS", re.I)
STAIR = re.compile(r"\bSTAIR|\bSTR\b", re.I)
fitz.TOOLS.mupdf_display_errors(False)


def page_info(page):
    text = page.get_text()
    area = abs(page.rect) or 1
    img_cover = 0.0
    for im in page.get_image_info():
        x0, y0, x1, y1 = im["bbox"]
        img_cover += max(0, (x1 - x0) * (y1 - y0)) / area
    return {
        "w_in": round(page.rect.width / 72, 1), "h_in": round(page.rect.height / 72, 1),
        "chars": len(text), "drawings": len(page.get_cdrawings()), "image_cover": round(img_cover, 2),
        "floor_labels": sorted({" ".join(m.group(1).upper().split()) for m in FLOOR.finditer(text)})[:6],
        "code": sorted({m.group(0).upper() for m in CODE.finditer(text)})[:5],
        "stair_words": len(STAIR.findall(text)),
    }


def main():
    only = set(sys.argv[1:])
    report = {}
    skip = {"cubicasa5k", "swiss_dwellings"}
    for pdf in sorted(PLANS.rglob("*.pdf")):
        bid = pdf.relative_to(PLANS).parts[0]
        if bid in skip or (only and bid not in only):
            continue
        doc = fitz.open(pdf)
        pages = []
        for n, page in enumerate(doc):
            info = page_info(page)
            info["page"] = n
            pages.append(info)
            if info["floor_labels"] or len(doc) <= 8:
                d = OUT / "pages" / bid
                d.mkdir(parents=True, exist_ok=True)
                page.get_pixmap(dpi=60 if page.rect.width > 1500 else 90).save(d / f"{pdf.stem[:40]}_p{n:03d}.png")
        report.setdefault(bid, {})[str(pdf.relative_to(PLANS))] = pages
        v = sum(p["drawings"] for p in pages)
        print(bid, pdf.name[:50], len(pages), "pages", "drawings", v, "cover", max((p["image_cover"] for p in pages), default=0), "floorpages", sum(bool(p["floor_labels"]) for p in pages), "code", sum(bool(p["code"]) for p in pages), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "inventory.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(report)
    path.write_text(json.dumps(old, indent=1))


main()
