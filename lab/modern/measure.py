"""Vector measurement on sheets with a text layer: dimension lines, hall clear widths, door arcs against the door schedule.

Run: .venv/bin/python -I lab/modern/measure.py  (no API calls). Writes data/lab/modern/measure.json and overlays.
All geometry is converted to the rendered page frame first; rotation is the trap on these PDFs.
"""
import json, math, re, statistics as st, sys, time
sys.path.insert(0, "lab/modern")
import cv2, numpy as np, pymupdf as fitz
import vec
ROOT = vec.ROOT
OUT = ROOT / "data/lab/modern"
CAT = {e["id"]: e for e in json.loads((ROOT / "data/plans/catalog.json").read_text())}
WALLS = ("A-wall-full", "A-wall-star", "A-wall-prht")


def rsegs(pg, segs):
    M = pg.rotation_matrix
    a = np.array([[*(fitz.Point(s[0], s[1]) * M), *(fitz.Point(s[2], s[3]) * M)] for s in segs], float) if segs else np.zeros((0, 4))
    return a


def rline_texts(pg):
    """(text, rect, dir) in the rendered frame. dir is the reading direction in that frame."""
    M = pg.rotation_matrix
    out = []
    for t, r, d in vec.joined_words(pg):
        q = r * M
        v = fitz.Point(d) * fitz.Matrix(M.a, M.b, M.c, M.d, 0, 0)
        out.append((t, q, (round(v.x), round(v.y))))
    return out


def dim_lines(pg, segs, panel):
    """Pair each dimension string in the panel with the dimension line beside it. Returns rows with measured inches."""
    dimseg = [s for s in segs if s[5] == "A-anno-dims" and s[4] > 0]
    a = rsegs(pg, dimseg)
    rows = []
    for t, q, d in rline_texts(pg):
        v = vec.parse_ftin(t)
        if v is None or not fitz.Rect(panel).contains(q):
            continue
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        horiz = abs(d[0]) == 1  # text reads along x, so its line runs along x
        best = None
        for x0, y0, x1, y1 in a:
            if horiz and abs(y0 - y1) < 0.3 and min(x0, x1) - 1 <= cx <= max(x0, x1) + 1 and abs(y0 - cy) < 22:
                c = (abs(y0 - cy), abs(x1 - x0), (min(x0, x1), y0, max(x0, x1), y0))
            elif not horiz and abs(x0 - x1) < 0.3 and min(y0, y1) - 1 <= cy <= max(y0, y1) + 1 and abs(x0 - cx) < 22:
                c = (abs(x0 - cx), abs(y1 - y0), (x0, min(y0, y1), x0, max(y0, y1)))
            else:
                continue
            if best is None or c[0] < best[0] - 0.5 or (abs(c[0] - best[0]) <= 0.5 and c[1] < best[1]):
                best = c
        rows.append(dict(text=t, declared=v, horiz=horiz, center=[cx, cy], line=best[2] if best else None, line_pt=best[1] if best else None))
    return rows


def ray_clear(px, py, a, horiz):
    """Nearest wall lines on both sides of a point along an axis. Returns (low, high) coordinates or None."""
    if horiz:
        sel = a[(np.abs(a[:, 0] - a[:, 2]) < 0.5) & (np.minimum(a[:, 1], a[:, 3]) - 1 <= py) & (py <= np.maximum(a[:, 1], a[:, 3]) + 1)]
        c = sel[:, 0]
        lo, hi = c[c < px], c[c > px]
    else:
        sel = a[(np.abs(a[:, 1] - a[:, 3]) < 0.5) & (np.minimum(a[:, 0], a[:, 2]) - 1 <= px) & (px <= np.maximum(a[:, 0], a[:, 2]) + 1)]
        c = sel[:, 1]
        lo, hi = c[c < py], c[c > py]
    return (lo.max(), hi.min()) if len(lo) and len(hi) else None


def halls(pg, segs, panel, ipp, dims, pattern=r"^Hall$"):
    """Clear width at each hall label from wall lines alone, then the nearest parallel dimension string."""
    a = rsegs(pg, [s for s in segs if s[5] in WALLS])
    rows = []
    for t, q, d in rline_texts(pg):
        if not re.match(pattern, t) or not fitz.Rect(panel).contains(q):
            continue
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        for horiz in (True, False):
            gap = None
            # sample along the hall, because a door opening can leave one ray with no wall
            vals = []
            for off in range(-60, 61, 6):
                r = ray_clear(cx, cy + off, a, True) if horiz else ray_clear(cx + off, cy, a, False)
                if r:
                    vals.append((r[1] - r[0], r))
            if vals:
                gap = min(vals, key=lambda v: v[0])
                gap = (st.median(v[0] for v in vals), gap[1], min(v[0] for v in vals))
            rows.append(dict(label=t, center=[cx, cy], horiz=horiz, median_in=None if not gap else gap[0] * ipp, min_in=None if not gap else gap[2] * ipp, span=None if not gap else list(gap[1])))
    keep = []
    for lbl in {tuple(r["center"]) for r in rows}:
        pair = [r for r in rows if tuple(r["center"]) == lbl and r["median_in"]]
        if not pair:
            continue
        r = min(pair, key=lambda x: x["median_in"])  # a corridor's width is its short axis
        c = [x for x in dims if x["horiz"] == r["horiz"] and 30 <= x["declared"] <= 90]
        d = min(c, key=lambda x: math.dist(x["center"], r["center"]))
        r["dim"] = d["text"]
        r["declared"] = d["declared"]
        r["dim_dist_pt"] = round(math.dist(d["center"], r["center"]), 1)
        r["err_pct"] = round(100 * (r["median_in"] - d["declared"]) / d["declared"], 1)
        keep.append(r)
    return keep


def door_arcs(pg, curves):
    """Quarter arcs from single cubic curves: tangents at the ends are perpendicular. Returns center, radius, ends."""
    M = pg.rotation_matrix
    out = []
    for p0, c1, c2, p3, w, lay in curves:
        p = [fitz.Point(q) * M for q in (p0, c1, c2, p3)]
        t0 = (p[1].x - p[0].x, p[1].y - p[0].y)
        t1 = (p[3].x - p[2].x, p[3].y - p[2].y)
        n0, n1 = math.hypot(*t0), math.hypot(*t1)
        if n0 < 1 or n1 < 1:
            continue
        cosang = (t0[0] * t1[0] + t0[1] * t1[1]) / (n0 * n1)
        if abs(cosang) > 0.12:
            continue
        # center: intersection of the normals at both ends
        A = np.array([[t0[0], t0[1]], [t1[0], t1[1]]])
        b = np.array([t0[0] * p[0].x + t0[1] * p[0].y, t1[0] * p[3].x + t1[1] * p[3].y])
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        r0, r1 = math.dist(c, (p[0].x, p[0].y)), math.dist(c, (p[3].x, p[3].y))
        if abs(r0 - r1) > 0.1 * max(r0, r1):
            continue
        out.append(dict(center=[float(c[0]), float(c[1])], r=(r0 + r1) / 2, layer=lay, p0=[p[0].x, p[0].y], p3=[p[3].x, p[3].y]))
    return out


def door_schedule(pg):
    """Mark to width (inches) from the DOOR SCHEDULE table, read off word rows in the rendered frame."""
    M = pg.rotation_matrix
    words = [(w[4], fitz.Rect(w[:4]) * M) for w in pg.get_text("words")]
    head = [r for t, r in words if t == "DOOR"]
    rows = {}
    for t, r in words:
        rows.setdefault(round(r.y0 / 3), []).append((r.x0, t, r))
    sched = {}
    heads = [r for t, r in words if t == "SCHEDULE"]
    hd = [r for r in head if any(abs(r.y0 - s.y0) < 3 and 0 < s.x0 - r.x1 < 12 for s in heads)]
    if not hd:
        return sched
    top = hd[0].y1
    left = hd[0].x0 - 200
    lines = sorted(rows.items())
    for _, line in lines:
        line.sort()
        y = line[0][2].y0
        if y < top or y > top + 260:
            continue
        txt = [x[1] for x in line if x[0] >= left]
        if not txt or not txt[0].isdigit():
            continue
        m = re.search(r"(\d+)\"\s*x\s*(\d+)\"", " ".join(txt))
        if m:
            sched[txt[0]] = int(m.group(1))
    return sched


def door_marks(pg, panel, sched):
    """Number-only text inside the plan panel that matches a schedule mark, ringed by a small circle."""
    M = pg.rotation_matrix
    out = []
    for w in pg.get_text("words"):
        r = fitz.Rect(w[:4]) * M
        if w[4] in sched and fitz.Rect(panel).contains(r):
            out.append((w[4], (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    return out


def sheet(id, f):
    pg = vec.open_page(id, f["file"], f["page"])
    t0 = time.time()
    segs, curves = vec.segments(pg)
    rect, note, title = vec.plan_panel(pg, segs, rule_layer="A-anno-htch-docs")
    panel = rect * pg.rotation_matrix
    ipp = vec.inches_per_pt(note)
    dims = dim_lines(pg, segs, panel)
    ok = [d for d in dims if d["line_pt"]]
    for d in ok:
        d["measured"] = d["line_pt"] * ipp
        d["err_pct"] = 100 * (d["measured"] - d["declared"]) / d["declared"]
    hl = halls(pg, segs, panel, ipp, ok)
    arcs = door_arcs(pg, curves)
    sched = door_schedule(pg)
    marks = door_marks(pg, panel, sched)
    pairs = []
    plan_arcs = [a for a in arcs if fitz.Rect(panel).contains(fitz.Point(a["center"])) and 12 <= a["r"] * ipp <= 60]
    for mk, x, y in marks:
        if not plan_arcs:
            break
        a = min(plan_arcs, key=lambda q: math.dist(q["center"], (x, y)) + 0.0)
        d = math.dist(a["center"], (x, y))
        if d < 75:
            pairs.append(dict(mark=mk, sched_in=sched[mk], arc_in=round(a["r"] * ipp, 1), dist_pt=round(d, 1), layer=a["layer"]))
    secs = time.time() - t0
    # overlay
    im = cv2.imread(str(OUT / f"render/{id}_{f['level']}_150.png"))
    Z = 150 / 72
    for x0, y0, x1, y1 in rsegs(pg, [s for s in segs if s[5] in WALLS]):
        cv2.line(im, (int(x0 * Z), int(y0 * Z)), (int(x1 * Z), int(y1 * Z)), (0, 0, 255), 3)
    for d in ok:
        l = d["line"]
        cv2.line(im, (int(l[0] * Z), int(l[1] * Z)), (int(l[2] * Z), int(l[3] * Z)), (0, 170, 0), 3)
    for a in plan_arcs:
        cv2.circle(im, (int(a["center"][0] * Z), int(a["center"][1] * Z)), int(a["r"] * Z), (255, 0, 255), 2)
    for r in hl:
        s = r["span"]
        p, q = ((s[0], r["center"][1]), (s[1], r["center"][1])) if r["horiz"] else ((r["center"][0], s[0]), (r["center"][0], s[1]))
        cv2.line(im, (int(p[0] * Z), int(p[1] * Z)), (int(q[0] * Z), int(q[1] * Z)), (255, 120, 0), 7)
    c = im[int(panel.y0 * Z):int(panel.y1 * Z), int(panel.x0 * Z):int(panel.x1 * Z)]
    cv2.imwrite(str(OUT / f"overlays/measure_{id}_{f['level']}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dict(id=id, level=str(f["level"]), note=note, title=title, ipp=ipp, n_dims=len(dims), n_lined=len(ok),
                dim_err=[round(d["err_pct"], 1) for d in ok], dim_bad=[(d["text"], round(d["measured"], 1)) for d in ok if abs(d["err_pct"]) > 3],
                halls=[{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k not in ("center", "span")} for r in hl],
                n_arcs=len(plan_arcs), arc_layers=sorted({a["layer"] for a in plan_arcs}), sched=sched, door_pairs=pairs, seconds=round(secs, 3))


if __name__ == "__main__":
    assert abs(vec.inches_per_pt("1/4\" = 1'-0\"") - 2 / 3) < 1e-9
    assert ray_clear(5, 5, np.array([[0, 0, 0, 10], [10, 0, 10, 10]], float), True) == (0, 10)
    res = []
    for id in ("mml_capitol", "mml_baxter"):
        for f in CAT[id]["floors"]:
            r = sheet(id, f)
            res.append(r)
            e = np.abs(r["dim_err"])
            print(id, r["level"], r["note"], "dims", r["n_dims"], "lined", r["n_lined"], f"median|err| {np.median(e):.2f}% p95 {np.percentile(e, 95):.2f}%", "bad", r["dim_bad"][:6], f"{r['seconds']}s")
            print("  halls", r["halls"])
            print("  arcs", r["n_arcs"], r["arc_layers"], "sched", r["sched"])
            print("  doors", [(p["mark"], p["sched_in"], p["arc_in"]) for p in r["door_pairs"]])
    (OUT / "measure.json").write_text(json.dumps(res, indent=1))
