"""Spark door grounding on the admin sheet (raster plan, colour fills) against hand labels."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, fitz, numpy as np
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, str(Path(__file__).parent))
import metalab as m
from spark_tiles import PROMPT, SCHEMA, CACHE, nms, score, to_sheet
import key as K

LAB = json.load(open(Path(__file__).parent / "labels_af_ces_admin.json"))
pg = fitz.open(str(K.ROOT / LAB["sheet"]))[LAB["page"]]
b = LAB["plan_box"]
G = LAB["interior"] + [p for v in LAB["exterior"].values() for p in v]
PM = LAB["pt_per_m"]


def grid(nx, ny, ov=0.15):
    w = (b[2] - b[0]) / (nx - (nx - 1) * ov)
    h = (b[3] - b[1]) / (ny - (ny - 1) * ov)
    return [(b[0] + i * w * (1 - ov), b[1] + j * h * (1 - ov), b[0] + i * w * (1 - ov) + w, b[1] + j * h * (1 - ov) + h) for j in range(ny) for i in range(nx)]


def one(i, r, name, px, effort):
    f = CACHE / f"admin_{name}_{i}.json"
    if f.exists():
        return json.loads(f.read_text())
    dpi = int(round(72 * px / max(r[2] - r[0], r[3] - r[1])))
    pix = pg.get_pixmap(dpi=dpi, clip=fitz.Rect(*r))
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()
    o, info = m.spark(PROMPT, images=[im], schema=SCHEMA, effort=effort)
    out = {"rect": list(r), "w": pix.w, "h": pix.h, "doors": o["points"], "count": o["count"], "info": info}
    f.write_text(json.dumps(out))
    return out


res = {}
for name, nx, ny, px, eff in (("g1x1", 1, 1, 2000, None), ("g2x1", 2, 1, 1600, "low"), ("g3x2", 3, 2, 1600, "low"), ("g4x2", 4, 2, 1600, "low"), ("g6x3", 6, 3, 1600, "low")):
    ts = grid(nx, ny)
    with ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda it: one(it[0], it[1], name, px, eff), enumerate(ts)))
    pts = nms([p for o in outs for p in to_sheet(o)], 0.45 * PM)
    row = {"tiles": len(ts), "tile_m": (ts[0][2] - ts[0][0]) / PM, "m_per_px": (ts[0][2] - ts[0][0]) / PM / px, "pts": len(pts), "usd": sum(o["info"]["usd"] for o in outs), "s_mean": float(np.mean([o["info"]["seconds"] for o in outs]))}
    for t in (0.5, 0.8):
        tp = score(pts, G, t * PM)
        row[f"tol{t}"] = {"tp": tp, "recall": tp / len(G), "precision": tp / max(len(pts), 1)}
    res[name] = row
    print(name, json.dumps(row), flush=True)
(K.OUT / "spark_admin.json").write_text(json.dumps(res, indent=1))
