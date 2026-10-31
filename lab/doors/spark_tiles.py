"""Spark door grounding on the clinic against the IFC key: whole sheet, plan crop, and tile grids. Results are cached."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, fitz, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab"); sys.path.insert(0, str(Path(__file__).parent))
import metalab as m
import key as K

CACHE = K.OUT / "spark"
CACHE.mkdir(exist_ok=True, parents=True)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["points", "count"],
          "properties": {"count": {"type": "integer"}, "points": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["x", "y"],
                                                                 "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}}}}
PROMPT = ('Point to every instance of the following object in this floor plan drawing and count them: "door swing arc". '
          "A door swing arc is the thin quarter-circle drawn at a gap in a wall, next to the straight door leaf; a double door has two arcs. "
          'Point at the middle of each arc. Do not miss any. Answer in the format {"points": [{"x": <int>, "y": <int>}, ...], "count": <int>}. '
          "The coordinates should be in the 0-1000 range.")


def arc_point(l):
    h, o, s = (np.array(l["s_" + k]) for k in ("hinge", "open", "shut"))
    u = (o - h) / np.linalg.norm(o - h) + (s - h) / np.linalg.norm(s - h)
    return (h + u / np.linalg.norm(u) * np.linalg.norm(o - h)).tolist()


def tiles(box, n, overlap=0.15):
    x0, y0, x1, y1 = box
    w = (x1 - x0) / (n - (n - 1) * overlap)
    h = (y1 - y0) / (n - (n - 1) * overlap)
    return [(x0 + i * w * (1 - overlap), y0 + j * h * (1 - overlap), x0 + i * w * (1 - overlap) + w, y0 + j * h * (1 - overlap) + h) for j in range(n) for i in range(n)]


def run_tile(tag, rect, px, name, effort):
    f = CACHE / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    dpi = int(round(72 * px / max(rect[2] - rect[0], rect[3] - rect[1])))
    im = K.render(tag, dpi, clip=fitz.Rect(*rect))
    H, W = im.shape[:2]
    r, info = m.spark(PROMPT, images=[im], schema=SCHEMA, effort=effort)
    out = {"rect": list(rect), "w": W, "h": H, "doors": r["points"], "count": r["count"], "info": info}
    f.write_text(json.dumps(out))
    return out


def to_sheet(out):
    x0, y0, x1, y1 = out["rect"]
    return [[x0 + d["x"] / 1000 * (x1 - x0), y0 + d["y"] / 1000 * (y1 - y0)] for d in out["doors"]]


def nms(pts, r):
    kept = []
    for p in pts:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > r for q in kept):
            kept.append(p)
    return kept


def score(pts, G, tol):
    if not pts:
        return 0
    D = np.linalg.norm(np.array(pts)[:, None] - np.array(G)[None], axis=2)
    r, c = linear_sum_assignment(np.minimum(D, tol + 1))
    return int((D[r, c] <= tol).sum())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "F1"
    A, L, arcs, reg = K.clinic_key(tag)
    G = [arc_point(l) for l in L]
    h = np.array(G)
    box = (h[:, 0].min() - 30, h[:, 1].min() - 30, h[:, 0].max() + 30, h[:, 1].max() + 30)
    page = K.page(tag).rect
    runs = [("sheet", (0, 0, page.width, page.height), 1, 2000, None), ("plan", box, 1, 2000, None), ("t2", box, 2, 1600, None), ("t3", box, 3, 1600, None), ("t3low", box, 3, 1600, "low"), ("t4", box, 4, 1600, None), ("t5low", box, 5, 1600, "low"), ("t6low", box, 6, 1600, "low")]
    res = {}
    for name, b, n, px, effort in [r for r in runs if len(sys.argv) < 3 or r[0] in sys.argv[2:]]:
        ts = tiles(b, n) if n > 1 else [b]
        with ThreadPoolExecutor(6) as ex:
            outs = list(ex.map(lambda it: run_tile(tag, it[1], px, f"{tag}_{name}_{it[0]}", effort), enumerate(ts)))
        pts = [p for o in outs for p in to_sheet(o)]
        merged = nms(pts, 0.45 * K.PT_PER_M)
        row = {"tiles": len(ts), "tile_m": (ts[0][2] - ts[0][0]) / K.PT_PER_M, "px": px, "m_per_px": (ts[0][2] - ts[0][0]) / K.PT_PER_M / px, "pts_raw": len(pts), "pts": len(merged),
               "usd": sum(o["info"]["usd"] for o in outs), "s_mean": float(np.mean([o["info"]["seconds"] for o in outs])), "out_tok_mean": float(np.mean([o["info"]["output"] for o in outs]))}
        for tol_m in (0.5, 0.8):
            tp = score(merged, G, tol_m * K.PT_PER_M)
            row[f"tol{tol_m}"] = {"tp": tp, "recall": tp / len(G), "precision": tp / max(len(merged), 1)}
        res[name] = row
        print(name, json.dumps(row), flush=True)
    (K.OUT / f"spark_tiles_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
