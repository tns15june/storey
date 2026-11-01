"""Vector sheets with no text layer: Spark reads dimension numbers, vector ticks verify each number by geometry, walls give corridor width.

Run: LAB_NAME=modern LAB_BUDGET=4 LAB_TOTAL=60 .venv/bin/python lab/modern/clinic.py
Sheet 55 is 1:50, read by eye from the title and graphic scale, because the sheet has no text layer.
"""
import json, math, sys
sys.path.insert(0, "lab")
sys.path.insert(0, "lab/modern")
import cv2, numpy as np, pymupdf as fitz
import metalab as m
import vec

OUT = vec.ROOT / "data/lab/modern"
SHEET = "2012-03-27-ClinicRedactedDrawings/Clinic_055_First Floor Plan Block 1C.pdf"
MM_PER_PT = 25.4 / 72 * 50
CROPS = [(1750, 700, 2620, 1020), (500, 260, 1250, 760), (1250, 260, 2000, 760), (500, 760, 1250, 1180), (1250, 1000, 2100, 1180)]
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["dims"], "properties": {"dims": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["value", "x", "y", "vertical"],
    "properties": {"value": {"type": "integer"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "vertical": {"type": "boolean"}}}}}}
PROMPT = ("This is part of an architectural floor plan with dimensions in millimetres. Point at every dimension number printed on the drawing. "
          "Skip door tags, room numbers, equipment codes such as M1620 or A5080, and grid bubbles. For each number return its integer value, the centre "
          "of the number, and whether the text is rotated to read vertically. Coordinates are integers on a 0-1000 grid, x from the left edge, y from the top.")


def ticks_on(a, L, horiz):
    """Positions along a dimension line where a short slash crosses it."""
    d = a[:, 2:4] - a[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    s = a[(ln > 4) & (ln < 16) & (np.abs(d[:, 0]) > 1) & (np.abs(d[:, 1]) > 1)]
    if horiz:
        y = L[1]
        t = s[(np.minimum(s[:, 1], s[:, 3]) <= y) & (y <= np.maximum(s[:, 1], s[:, 3]))]
        return np.unique(np.round((t[:, 0] + t[:, 2]) / 2, 1))
    x = L[0]
    t = s[(np.minimum(s[:, 0], s[:, 2]) <= x) & (x <= np.maximum(s[:, 0], s[:, 2]))]
    return np.unique(np.round((t[:, 1] + t[:, 3]) / 2, 1))


def measure_dim(a, x, y, vertical):
    """Span in mm between the ticks around (x, y) on the nearest parallel dimension line, or None."""
    h = a[np.abs(a[:, 1] - a[:, 3]) < 0.3]
    v = a[np.abs(a[:, 0] - a[:, 2]) < 0.3]
    if not vertical:
        c = h[(np.minimum(h[:, 0], h[:, 2]) - 8 <= x) & (x <= np.maximum(h[:, 0], h[:, 2]) + 8) & (np.abs(h[:, 1] - y) < 14) & (np.abs(h[:, 2] - h[:, 0]) > 6)]
        lines = sorted(c.tolist(), key=lambda r: abs(r[1] - y))
    else:
        c = v[(np.minimum(v[:, 1], v[:, 3]) - 8 <= y) & (y <= np.maximum(v[:, 1], v[:, 3]) + 8) & (np.abs(v[:, 0] - x) < 14) & (np.abs(v[:, 3] - v[:, 1]) > 6)]
        lines = sorted(c.tolist(), key=lambda r: abs(r[0] - x))
    for L in lines[:4]:
        t = ticks_on(a, L, not vertical)
        p = x if not vertical else y
        lo, hi = t[t <= p + 1], t[t >= p - 1]
        if len(lo) and len(hi) and hi.min() > lo.max():
            return float((hi.min() - lo.max()) * MM_PER_PT)
    return None


if __name__ == "__main__":
    assert abs(MM_PER_PT - 17.639) < 0.01
    pg = vec.open_page("nibs_clinic", SHEET, 0)
    segs, _ = vec.segments(pg)
    a = vec.seg_arrays(segs)
    rows = []
    for k, c in enumerate(CROPS):
        pm = pg.get_pixmap(dpi=200, alpha=False, clip=fitz.Rect(c))
        im = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3)[:, :, ::-1].copy()
        p = OUT / f"preview/c55_crop{k}.png"
        cv2.imwrite(str(p), im)
        out, info = m.spark(PROMPT, [p], schema=SCHEMA, effort="low")
        print("crop", k, im.shape[:2], len(out["dims"]), {x: round(info[x], 3) for x in ("usd", "seconds")}, info["output"], flush=True)
        for d in out["dims"]:
            x, y = c[0] + d["x"] / 1000 * (c[2] - c[0]), c[1] + d["y"] / 1000 * (c[3] - c[1])
            mm = measure_dim(a, x, y, d["vertical"])
            rows.append(dict(crop=k, value=d["value"], x=round(x, 1), y=round(y, 1), vertical=d["vertical"], measured_mm=None if mm is None else round(mm)))
    (OUT / "clinic_dims.json").write_text(json.dumps(rows, indent=1))
    have = [r for r in rows if r["measured_mm"]]
    good = [r for r in have if abs(r["measured_mm"] - r["value"]) <= max(30, 0.03 * r["value"])]
    print("read", len(rows), "with ticks", len(have), "verified", len(good))
    print("unverified", [(r["value"], r["measured_mm"]) for r in have if r not in good][:30])
    print("spent", m.spent("modern"))
