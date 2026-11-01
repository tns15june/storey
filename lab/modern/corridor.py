"""Clinic sheet 55: corridor width from heavy wall lines on a ray, against the printed 1710 dimensions (read by Spark and by eye).

Run: .venv/bin/python -I lab/modern/corridor.py
"""
import json, sys
sys.path.insert(0, "lab/modern")
import numpy as np
import vec

SHEET = "2012-03-27-ClinicRedactedDrawings/Clinic_055_First Floor Plan Block 1C.pdf"
MM = 25.4 / 72 * 50
OUT = vec.ROOT / "data/lab/modern"
pg = vec.open_page("nibs_clinic", SHEET, 0)
segs, _ = vec.segments(pg)
a = np.array([s[:5] for s in segs])
heavy = a[np.abs(a[:, 4] - 0.79) < 0.02]


def along(h, fixed, lo, hi, vertical_ray):
    """Heavy line positions crossing a ray at `fixed`, between lo and hi."""
    if vertical_ray:
        s = h[(np.abs(h[:, 1] - h[:, 3]) < 0.3) & (np.minimum(h[:, 0], h[:, 2]) <= fixed) & (fixed <= np.maximum(h[:, 0], h[:, 2]))]
        return sorted({round(float(v), 1) for v in s[:, 1] if lo < v < hi})
    s = h[(np.abs(h[:, 0] - h[:, 2]) < 0.3) & (np.minimum(h[:, 1], h[:, 3]) <= fixed) & (fixed <= np.maximum(h[:, 1], h[:, 3]))]
    return sorted({round(float(v), 1) for v in s[:, 0] if lo < v < hi})


cases = [
    ("1CC1 east end, width across corridor", 2352, 760, 980, True, 1710),
    ("1CC1 east end, second ray", 2370, 760, 980, True, 1710),
]
rows = []
for name, fx, lo, hi, vert, printed in cases:
    p = along(heavy, fx, lo, hi, vert)
    mid = (lo + hi) / 2
    rows.append(dict(case=name, heavy_lines=p, printed_mm=printed))
    print(name, p)
    # gaps between consecutive lines in mm, the printed width should be one of them
    gaps = [round((q - r) * MM) for r, q in zip(p, p[1:])]
    rows[-1]["gaps_mm"] = gaps
    if gaps:
        g = max(gaps)
        rows[-1].update(measured_mm=g, err_pct=round(100 * (g - printed) / printed, 1))
    print("   gaps mm", gaps)
(OUT / "corridor.json").write_text(json.dumps(rows, indent=1))
