"""Door swings from vector paths against the door schedule, by assignment of arcs to plan door marks.

Run: .venv/bin/python -I lab/modern/doors.py
"""
import json, math, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern")
import numpy as np, pymupdf as fitz
from scipy.optimize import linear_sum_assignment
import vec, measure as M


def fit_circle(pts):
    """Algebraic circle fit. Returns cx, cy, r, rms."""
    p = np.array(pts, float)
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0))
    rms = float(np.sqrt(np.mean((np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - r) ** 2)))
    return float(c[0]), float(c[1]), r, rms


def door_swings(pg, panel, ipp):
    """Arcs of 70 to 110 degrees with radius 20 to 50 real inches, from cubic curves and from polylines of short lines."""
    Mt = pg.rotation_matrix
    out = []

    def consider(pts, layer):
        if len(pts) < 5 or not fitz.Rect(panel).contains(pts[0]):
            return
        cx, cy, r, rms = fit_circle([(p.x, p.y) for p in pts])
        if not (20 <= r * ipp <= 50) or rms > 0.5:
            return
        span = float(np.degrees(np.ptp(np.unwrap(np.arctan2([p.y - cy for p in pts], [p.x - cx for p in pts])))))
        if 70 <= span <= 110:
            out.append(dict(c=[cx, cy], r_in=round(r * ipp, 2), span=round(span), layer=layer, mid=[float(np.mean([p.x for p in pts])), float(np.mean([p.y for p in pts]))]))

    for d in pg.get_drawings():
        if d.get("layer") == "A-anno-symb":
            continue
        chain = []  # arcs arrive as several cubics or short lines in a row, so flatten and chain them
        for it in d["items"]:
            if it[0] == "l" and math.dist(it[1], it[2]) < 30:
                pts = [fitz.Point(it[1]) * Mt, fitz.Point(it[2]) * Mt]
            elif it[0] == "c":
                a, b, c, e = [fitz.Point(q) * Mt for q in it[1:5]]
                pts = [fitz.Point((1-t)**3 * a.x + 3 * (1-t)**2 * t * b.x + 3 * (1-t) * t*t * c.x + t**3 * e.x, (1-t)**3 * a.y + 3 * (1-t)**2 * t * b.y + 3 * (1-t) * t*t * c.y + t**3 * e.y) for t in np.linspace(0, 1, 9)]
            else:
                consider(chain, d.get("layer"))
                chain = []
                continue
            if chain and math.dist(chain[-1], pts[0]) > 0.3:
                consider(chain, d.get("layer"))
                chain = []
            chain += pts
        consider(chain, d.get("layer"))
    # one door is sometimes drawn twice on two layers
    uniq = []
    for a in out:
        if all(math.dist(a["c"], b["c"]) > 2 or abs(a["r_in"] - b["r_in"]) > 1 for b in uniq):
            uniq.append(a)
    return uniq


def marks(pg, panel, sched, size=(9.0, 10.0)):
    """Door tags: numeric text in the plan at the tag font size. Stair risers, callouts and view numbers have other sizes."""
    Mt = pg.rotation_matrix
    out = []
    for b in pg.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                r = fitz.Rect(s["bbox"]) * Mt
                if t in sched and size[0] <= s["size"] <= size[1] and fitz.Rect(panel).contains(r):
                    out.append((t, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, round(s["size"], 1)))
    return out


def run(id, f):
    pg = vec.open_page(id, f["file"], f["page"])
    segs, _ = vec.segments(pg)
    rect, note, _ = vec.plan_panel(pg, segs, rule_layer="A-anno-htch-docs")
    panel = fitz.Rect(rect * pg.rotation_matrix)
    ipp = vec.inches_per_pt(note)
    sched = M.door_schedule(pg)
    arcs = door_swings(pg, panel, ipp)
    mk = marks(pg, panel, sched)
    pairs = []
    if arcs and mk:
        C = np.array([[math.dist(a["mid"], (m[1], m[2])) for a in arcs] for m in mk])
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < 75:
                pairs.append(dict(mark=mk[i][0], sched_in=sched[mk[i][0]], arc_in=arcs[j]["r_in"], dist_pt=round(float(C[i, j]), 1), layer=arcs[j]["layer"], mid=[round(v) for v in arcs[j]["mid"]]))
    return dict(id=id, level=str(f["level"]), n_sched=len(sched), n_marks=len(mk), mark_sizes=sorted({m[3] for m in mk}), n_arcs=len(arcs), pairs=pairs, unmatched_marks=sorted({m[0] for m in mk} - {p["mark"] for p in pairs}, key=int))


if __name__ == "__main__":
    c = fit_circle([(math.cos(t) * 5, math.sin(t) * 5) for t in np.linspace(0, 1.5, 12)])
    assert abs(c[2] - 5) < 1e-6
    res = []
    for id in ("mml_capitol", "mml_baxter"):
        for f in M.CAT[id]["floors"]:
            r = run(id, f)
            res.append(r)
            bad = [p for p in r["pairs"] if abs(p["arc_in"] - p["sched_in"]) > 1.5]
            print(id, r["level"], "sched", r["n_sched"], "marks", r["n_marks"], r["mark_sizes"], "arcs", r["n_arcs"], "pairs", len(r["pairs"]), "match<=1.5in", len(r["pairs"]) - len(bad), "bad", [(p["mark"], p["sched_in"], p["arc_in"], p["dist_pt"]) for p in bad], "unmatched", r["unmatched_marks"])
    (M.OUT / "doors.json").write_text(json.dumps(res, indent=1))
