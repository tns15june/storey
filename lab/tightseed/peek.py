"""Render every page of the candidate new sets at low res to see what is in them. No API."""
import sys, json
import fitz, cv2, numpy as np
from pathlib import Path
ROOT = Path("."); OUT = ROOT / "data/lab/tightseed/raw"
CAT = {x["id"]: x for x in json.load(open(ROOT / "data/plans/catalog.json"))}
for bid in ("queens_leonard", "queens_royce", "mml_abbot"):
    print(bid, CAT[bid]["floors"])
for bid, f in (("queens_leonard", "leonard_hall.pdf"), ("queens_royce", "jean_royce_hall.pdf"), ("mml_abbot", "THE-ABBOT.pdf")):
    d = fitz.open(ROOT / "data/plans" / bid / f)
    print(bid, len(d), d[0].rect, [len(p.get_images()) for p in d][:16], [len(p.get_drawings()) for p in d][:16])
    print([p.get_text()[:60].replace("\n", " | ") for p in d][:16])
