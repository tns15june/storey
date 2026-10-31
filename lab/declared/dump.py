"""Dump per-page text-layer lines for the PDF sets so the reading code can be written by eye. No model."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import fitz
ROOT = Path("/Users/tsharma/Github/Meta")
OUT = ROOT / "data/lab/declared/text"; OUT.mkdir(parents=True, exist_ok=True)
SETS = {"mml_capitol": "mml_capitol/THE-CAPITOL.pdf", "mml_baxter": "mml_baxter/THE-BAXTER.pdf", "mml_abbot": "mml_abbot/THE-ABBOT.pdf",
        "usc_leconte": "usc_leconte/USC_LeConte_Bid_Drawings.pdf", "carpinteria_adu1": "carpinteria_adu1/plan1.pdf", "carpinteria_adu2": "carpinteria_adu2/plan2.pdf",
        "af_ces_admin": "af_ces_admin/CES_SD_drawings_admin_bldg.pdf", "af_ces_shops": "af_ces_shops/CES_sd_drawings_warehouse_bldg.pdf",
        "wfu_efird": "wfu_efird/Efird.pdf", "wfu_huffman": "wfu_huffman/HUFFMAN-FLOOR-PLAN-G-3.pdf", "brandeis_hall": "brandeis_hall/level-1.pdf"}
for k, f in SETS.items():
    d = fitz.open(ROOT / "data/plans" / f)
    with open(OUT / f"{k}.txt", "w") as o:
        for i, p in enumerate(d):
            o.write(f"\n#### page {i} rot={p.rotation} rect={[round(x) for x in p.rect]}\n")
            M = p.rotation_matrix
            rows = []
            for b in p.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    t = " ".join(s["text"].strip() for s in ln["spans"]).strip()
                    if t:
                        r = fitz.Rect(ln["bbox"]) * M
                        rows.append((round(r.y0), round(r.x0), round(ln["spans"][0]["size"], 1), t))
            for y, x, s, t in sorted(rows):
                o.write(f"{y}\t{x}\t{s}\t{t}\n")
