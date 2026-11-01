"""Detect stair runs on each floor of each set from vector lines alone and draw them on the 150 DPI render."""
import json, sys, time
sys.path.insert(0, "lab/modern")
import cv2, numpy as np, pymupdf as fitz
import vec
ROOT = vec.ROOT
OUT = ROOT / "data/lab/modern"
cat = {e["id"]: e for e in json.loads((ROOT / "data/plans/catalog.json").read_text())}
SETS = sys.argv[1:] or "mml_capitol mml_baxter nibs_clinic wfu_efird wfu_huffman mml_abbot carpinteria_adu1 brandeis_hall usc_leconte".split()
res = {}
for id in SETS:
    for f in cat[id]["floors"]:
        pg = vec.open_page(id, f["file"], f["page"])
        t = time.time()
        segs, curves = vec.segments(pg)
        runs = vec.stair_runs(segs)
        dt = time.time() - t
        M, Z = pg.rotation_matrix, 150 / 72
        im = cv2.imread(str(OUT / f"render/{id}_{f['level']}_150.png"))
        boxes = []
        for r in runs:
            p = (fitz.Rect(r["box"]) * M) * Z
            b = [int(p.x0), int(p.y0), int(p.x1), int(p.y1)]
            boxes.append(dict(box150=b, steps=r["steps"], tread_pt=round(r["tread"], 2), width_pt=round(r["width"], 1), angle=r["angle"]))
            cv2.rectangle(im, (b[0] - 6, b[1] - 6), (b[2] + 6, b[3] + 6), (0, 0, 255), 6)
        res[f"{id}/{f['level']}"] = dict(n=len(runs), seconds=round(dt, 3), runs=boxes)
        print(id, f["level"], "runs", len(runs), f"{dt:.2f}s", [(x["steps"], x["tread_pt"], x["width_pt"]) for x in boxes][:10], flush=True)
        s = 1500 / max(im.shape[:2])
        cv2.imwrite(str(OUT / f"preview/stairs_{id}_{f['level']}.jpg"), cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
(OUT / "stair_runs.json").write_text(json.dumps(res, indent=1))
