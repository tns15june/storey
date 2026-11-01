"""Part 1: floor names from the PDF text layer, no model call. Sheets with text only."""
import collections
import json
import re
from pathlib import Path

import fitz

ROOT = Path(".")
cat = {x["id"]: x for x in json.load(open(ROOT / "data/plans/catalog.json"))}
RX = re.compile(r"\b(basement|cellar|ground|main|lower|upper|first|second|third|fourth|attic|roof)\s+(?:level\s+|floor\s+)?(?:floor\s+)?plan\b", re.I)
RANK = {"basement": -1, "cellar": -1, "lower": -1, "ground": 0, "main": 1, "first": 1, "second": 2, "upper": 2, "third": 3, "fourth": 4, "attic": 5, "roof": 6}
out = {}
for bid in ["mml_capitol", "mml_baxter", "wfu_efird", "wfu_huffman", "nibs_clinic"]:
    rows = []
    for f in cat[bid]["floors"]:
        with fitz.open(ROOT / "data/plans" / bid / f["file"]) as d:
            text = d[f["page"]].get_text()
        hits = collections.Counter(h.lower() for h in RX.findall(text))
        top = hits.most_common(1)[0][0] if hits else None
        rows.append({"level": f["level"], "chars": len(text), "hits": dict(hits), "name": top, "rank": RANK.get(top)})
    ok = all(r["rank"] is not None for r in rows) and [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)
    out[bid] = {"sheets": rows, "order_correct": ok}
    print(bid, ok, [(r["level"], r["chars"], r["name"]) for r in rows])
json.dump(out, open(ROOT / "data/lab/align/pdf_order.json", "w"), indent=1)
