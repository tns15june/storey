"""Corridor check: at each Hall label, read the wall faces on a ray and compare clear and centre-to-centre width with the dimension chain.

Run: .venv/bin/python -I lab/modern/halls.py
"""
import json, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern")
import numpy as np, pymupdf as fitz
import vec, measure as M

CAT = M.CAT


def crossings(a, px, py, horiz):
    """Sorted coordinates where wall lines cross a ray through (px, py) along x or y."""
    if horiz:
        s = a[(np.abs(a[:, 0] - a[:, 2]) < 0.5) & (np.minimum(a[:, 1], a[:, 3]) - 0.5 <= py) & (py <= np.maximum(a[:, 1], a[:, 3]) + 0.5)]
        return np.unique(np.round(s[:, 0], 1))
    s = a[(np.abs(a[:, 1] - a[:, 3]) < 0.5) & (np.minimum(a[:, 0], a[:, 2]) - 0.5 <= px) & (px <= np.maximum(a[:, 0], a[:, 2]) + 0.5)]
    return np.unique(np.round(s[:, 1], 1))


def wall_pairs(xs, max_thick=12.0):
    """Group sorted face coordinates into walls of two faces, thicker than a line and under 12 pt (8 in at 1/4 scale)."""
    out, i = [], 0
    while i < len(xs):
        if i + 1 < len(xs) and 2.0 < xs[i + 1] - xs[i] <= max_thick:
            out.append((xs[i], xs[i + 1]))
            i += 2
        else:
            out.append((xs[i], xs[i]))
            i += 1
    return out


def run(id, f):
    pg = vec.open_page(id, f["file"], f["page"])
    segs, _ = vec.segments(pg)
    rect, note, _ = vec.plan_panel(pg, segs, rule_layer="A-anno-htch-docs")
    panel = fitz.Rect(rect * pg.rotation_matrix)
    ipp = vec.inches_per_pt(note)
    a = M.rsegs(pg, [s for s in segs if s[5] in M.WALLS])
    texts = M.rline_texts(pg)
    dims = M.dim_lines(pg, segs, panel)
    # dim lines overshoot their ticks by a fixed length, so calibrate it on the sheet and report the rest
    delta = float(np.median([d["line_pt"] * ipp - d["declared"] for d in dims if d["line_pt"]]))
    out = dict(id=id, level=str(f["level"]), overshoot_in=round(delta, 2),
               dim_resid_in=sorted(round(d["line_pt"] * ipp - delta - d["declared"], 2) for d in dims if d["line_pt"]),
               dim_outliers=[(d["text"], round(d["line_pt"] * ipp - delta, 1)) for d in dims if d["line_pt"] and abs(d["line_pt"] * ipp - delta - d["declared"]) > 1], halls=[])
    for t, q, d in texts:
        if t != "Hall" or not panel.contains(q):
            continue
        if any(u == "Stair" and abs(r.x0 - q.x0) < 60 and abs(r.y0 - q.y0) < 30 for u, r, _ in texts):
            continue
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        x = crossings(a, cx, cy, True)
        walls = wall_pairs(x)
        L = [w for w in walls if w[1] < cx][-1]
        R = [w for w in walls if w[0] > cx][0]
        clear, ctc, outer = (R[0] - L[1]) * ipp, ((R[0] + R[1]) / 2 - (L[0] + L[1]) / 2) * ipp, (R[1] - L[0]) * ipp
        # the chain dimension that overlaps this hall in x, nearest in y
        c = [e for e in dims if e["horiz"] and e["line_pt"] and e["line"][0] - 6 <= (L[1] + R[0]) / 2 <= e["line"][2] + 6 and 30 <= e["declared"] <= 80 and abs(e["line"][1] - cy) < 220]
        e = min(c, key=lambda e: abs(e["line"][1] - cy))
        tick0, tick1 = e["line"][0] + delta / ipp / 2, e["line"][2] - delta / ipp / 2
        out["halls"].append(dict(label=t, cross=[float(v) for v in x], L=[float(v) for v in L], R=[float(v) for v in R], clear_in=round(float(clear), 1), ctc_in=round(float(ctc), 1), outer_in=round(float(outer), 1),
                                 dim=e["text"], declared=e["declared"], ticks=[round(tick0, 1), round(tick1, 1)],
                                 err_clear_pct=round(100 * (clear - e["declared"]) / e["declared"], 1), err_ctc_pct=round(100 * (ctc - e["declared"]) / e["declared"], 1), err_outer_pct=round(100 * (outer - e["declared"]) / e["declared"], 1)))
    return out


if __name__ == "__main__":
    assert wall_pairs(np.array([0.0, 7.0, 50.0, 57.0])) == [(0.0, 7.0), (50.0, 57.0)]
    res = []
    for id in ("mml_capitol", "mml_baxter"):
        for f in CAT[id]["floors"]:
            r = run(id, f)
            res.append(r)
            e = np.abs(r["dim_resid_in"])
            print(id, r["level"], "overshoot", r["overshoot_in"], "n", len(e), "resid max", e.max(), "outliers", r["dim_outliers"])
            for h in r["halls"]:
                print("   ", h)
    (M.OUT / "halls.json").write_text(json.dumps(res, indent=1))
