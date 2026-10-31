"""Build the overlay audit test bed: 32 cases, 16 bad, 16 good, one renderer, one claim template per category.

Writes candidates only. freeze.py shuffles, renames and hashes after they are checked by eye. No API.
Run: .venv/bin/python -I lab/agent/build_cases.py [stage]   stage: stairs routes doors values all
"""
import json, math, random, sys
from pathlib import Path
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/seed"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/modern"); sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/agent")
import cv2, numpy as np, pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
import vec, measure as M  # lab/modern's vec: lab/walls has its own, so this import must come before routelib
import draw, routelib as RL

SEED = 20261031
ROOT = Path("/Users/tsharma/Github/Meta")
A = ROOT / "data/lab/audit"; MINE = ROOT / "data/lab/agent/mine"; CAND = ROOT / "data/lab/agent/cand"; CAND.mkdir(parents=True, exist_ok=True)
STAIR_T = "Storey's stair tracker placed the magenta box on this floor plan. Claim: a staircase is drawn inside the box on this floor."
ROUTE_T = "The magenta line is Storey's travel route from the green dot to the blue dot, measured as {m} m. Claim: the whole route stays on walkable floor. It leaves each room only through a doorway and crosses no wall."
DOOR_T = "Storey linked the door tag in the magenta circle, read as tag {n}, to the door whose swing arc is at the small ring at the other end of the magenta line. Claim: this tag labels that door."
VALUE_T = "A Storey finding cites the printed dimension in the magenta box as {cited} for the span between the two magenta ticks, and gives the measured span as {meas}. Claim: the box holds that printed value, and the ticks mark the span that the printed number dimensions."


def save(cases, name):
    json.dump(cases, open(CAND / f"{name}.json", "w"), indent=1)


def write(im, name):
    cv2.imwrite(str(CAND / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"{name}.jpg"


# ---------- stairs ----------
BAD_GROUPS = {"A": [("bme_k", 7, 2), ("bme_k", 7, 3)], "B": [("bme_k", 10, 2), ("bme_k", 10, 3)], "C": [("bme_k", 8, 0)], "D": [("bme_k", 9, 0)],
              "E": [("bme_k", 12, 2)], "F": [("bme_k", 13, 1)], "G": [("bme_k", 15, 0), ("bme_k", 15, 1), ("bme_k", 15, 2)], "H": [("habs_va_bldg1", 8, 4)]}


def stair_img(c, R=220, s=2.0):
    im = cv2.imread(str(A / c["b"] / f"f{c['f']}.png")); t = c["box"]
    cx, cy = (t[0] + t[2]) // 2, (t[1] + t[3]) // 2
    x0, y0 = max(cx - R, 0), max(cy - R, 0)
    v = cv2.resize(im[y0:cy + R, x0:cx + R], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    P = lambda x, y: (int((x - x0) * s), int((y - y0) * s))
    return draw.box(v, P(t[0], t[1]), P(t[2], t[3]))


def stairs():
    rng = random.Random(SEED)
    T = json.load(open(ROOT / "lab/audit/truth.json"))
    pool = json.load(open(MINE / "stair_cands.json"))
    by = {(c["b"], c["oid"], c["f"]): c for c in pool}
    out = []
    for g in rng.sample(sorted(BAD_GROUPS), 4):
        c = by[rng.choice(BAD_GROUPS[g])]
        out.append(dict(cat="stair", label="bad", source="real", note=f"SAM box at conf {c['conf']} with no stair inside, group {g}, {c['b']} o{c['oid']} f{c['f']}", claim=STAIR_T, img=write(stair_img(c), f"stair_bad_{g}")))

    def cover(c):
        for k in c["on"]:
            t = T[c["b"]][k]["boxes"][str(c["f"])]
            b = c["box"]
            iw, ih = max(0, min(b[2], t[2]) - max(b[0], t[0])), max(0, min(b[3], t[3]) - max(b[1], t[1]))
            if iw * ih >= .5 * (t[2] - t[0]) * (t[3] - t[1]) and T[c["b"]][k]["cls"] == "main":
                return True
        return False
    good = [c for c in pool if c["on"] and cover(c)]
    for b, n in (("bme_k", 2), ("habs_va_bldg1", 1), ("wfu_huffman", 1)):
        g = [c for c in good if c["b"] == b]
        rng.shuffle(g)
        seen = set()
        for c in g:  # first in seeded order; a stair is used once
            key = tuple(c["on"])
            if key in seen or n == 0:
                continue
            seen.add(key); n -= 1
            out.append(dict(cat="stair", label="good", source="real", note=f"SAM box at conf {c['conf']} on truth stair {c['on']}, {c['b']} o{c['oid']} f{c['f']}", claim=STAIR_T, img=write(stair_img(c), f"stair_good_{c['b'][:4]}_{c['oid']}_{c['f']}")))
    save(out, "stairs")
    return out


# ---------- routes ----------
def to_pt(q):
    return (q[1] / RL.K.SC + RL.K.BOX[0], q[0] / RL.K.SC + RL.K.BOX[1])


def route_img(fl, pts):
    """pts in (y, x) mask pixels, room first."""
    pg = fitz.open(RL.K.D + RL.SHEET[fl])[0]
    p = [to_pt(q) for q in pts]
    a = np.array(p); m = 3 * RL.K.PPM
    im, P, s = draw.render(pg, (a[:, 0].min() - m, a[:, 1].min() - m, a[:, 0].max() + m, a[:, 1].max() + m))
    draw.route(im, P, p)
    return im, P


def cross_xy(cr, P, im):
    """Crossing midpoints as 0-1000 grid points of the case image."""
    h, w = im.shape[:2]
    out = []
    for y, x, n in cr:
        q = P(*to_pt((y, x)))
        out.append([round(q[0] * 1000 / w), round(q[1] * 1000 / h), n])
    return out


def plant_gap(d, cum, pt, rng):
    """Find a wall of the room at `pt` with a lower exit distance on its far side and no door within 1.5 m. Returns a punched copy of walk, or None."""
    walk, lab, gt = d["walk"], d["lab"], d["gt"]
    door = gt & (lab == 0)
    x, y = pt
    if not walk[y, x]:
        return None
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(dirs)
    px_m = RL.K.PXM
    for dx, dy in dirs:
        k = 1
        while walk[y + dy * k, x + dx * k]:
            k += 1
            if k > 400:
                break
        if k > 400:
            continue
        gap = 0
        while not walk[y + dy * (k + gap), x + dx * (k + gap)] and gap < 1.3 * px_m:
            gap += 1
        fx, fy = x + dx * (k + gap), y + dy * (k + gap)
        if gap >= 1.3 * px_m or gap < 0.25 * px_m or not walk[fy, fx] or lab[fy, fx] == lab[y, x]:
            continue
        if not np.isfinite(cum[fy, fx]) or cum[fy, fx] + 5 * px_m > cum[y, x]:
            continue
        mx, my = x + dx * (k + gap // 2), y + dy * (k + gap // 2)
        R = int(1.5 * px_m)
        if door[my - R:my + R, mx - R:mx + R].any():
            continue
        w2 = walk.copy()
        cv2.line(w2 := w2.astype(np.uint8), (x + dx * (k - 2), y + dy * (k - 2)), (fx, fy), 1, int(0.5 * px_m))
        return w2.astype(bool), (mx, my)
    return None


def hits_solid(pts, solid):
    q = RL.sample(pts)
    return int(solid[q[:, 0], q[:, 1]].sum())


def routes():
    rng = random.Random(SEED + 1)
    res = json.load(open(MINE / "routes.json"))
    out = []
    # real leaks, checked by eye: rooms 100 and 72 leave through window gaps in the exterior wall. Rooms 97 and 12 are flagged by the
    # IFC mask but pass through real door or elevator openings on the sheet, so they are not used.
    byroom = {r["room"]: r for r in res["1"]}
    d1 = RL.load(1)
    for room in (100, 72):
        r = byroom[room]
        pts = [tuple(q) for q in r["pts"]][::-1]
        cr = RL.crossings(pts, d1["gt"])
        assert cr
        im, P = route_img(1, pts)
        out.append(dict(cat="route", label="bad", source="real", note=f"Storey walk mask leaks through a window gap in the exterior wall, F1 room label {room}", claim=ROUTE_T.format(m=r["m"]), img=write(im, f"route_bad_{room}"), defect=cross_xy(cr, P, im), fl=1))
    # good: no crossing of the truth mask, no heavy wall hit, 8 to 24 m
    for fl, n in ((1, 2), (2, 2)):
        d = RL.load(fl)
        g = [r for r in res[str(fl)] if not r["cross"] and 8 <= r["m"] <= 24 and r["n"] >= 3 and not hits_solid([tuple(q) for q in r["pts"]], d["solid"])]
        rng.shuffle(g)
        for r in g[:n + 3]:  # spares for the eye check
            pts = [tuple(q) for q in r["pts"]][::-1]
            im, P = route_img(fl, pts)
            out.append(dict(cat="route", label="good", source="real", note=f"Storey route, no IFC wall crossed, F{fl} room label {r['room']}", claim=ROUTE_T.format(m=r["m"]), img=write(im, f"route_good_{fl}_{r['room']}"), fl=fl, spare=True))
    # planted: same router on a mask with one gap punched where no door exists
    for fl in (1, 2):
        d = RL.load(fl)
        rows = json.load(open(RL.K.O + f"clinic_f{fl}_rows.json"), parse_constant=lambda c: float("inf"))
        seeds = [s for s in rows["seeds"][::40] if d["walk"][int(s[1]), int(s[0])]]
        snap = RL.snapper(d["walk"])
        ends = list({snap(e) for e in seeds})
        mcp = RL.MCP_Geometric(np.where(d["walk"], 1.0, -1.0), fully_connected=True)
        cum, _ = mcp.find_costs(ends)
        rooms = [r for r in res[str(fl)] if not r["cross"] and 10 <= r["m"] <= 30]
        rng.shuffle(rooms)
        made = 0
        for r in rooms:
            got = plant_gap(d, cum, tuple(r["xy"]), rng)
            if got is None:
                continue
            w2, gap = got
            mc2 = RL.MCP_Geometric(np.where(w2, 1.0, -1.0), fully_connected=True)
            c2, _ = mc2.find_costs(ends)
            x, y = r["xy"]
            if not np.isfinite(c2[y, x]) or c2[y, x] > cum[y, x] - 3 * RL.K.PXM:
                continue
            pts, t = RL.taut(w2, mc2.traceback((y, x)))
            pts = [(int(a), int(b)) for a, b in pts][::-1]
            if not hits_solid(pts, d["solid"]):
                continue
            im, P = route_img(fl, pts)
            out.append(dict(cat="route", label="bad", source="planted", note=f"gap punched at mask {gap} on F{fl}, room label {r['room']}, unplanted route {r['m']} m", claim=ROUTE_T.format(m=round(t / RL.K.PXM, 2)), img=write(im, f"route_plant_{fl}_{r['room']}"), defect=cross_xy([(gap[1], gap[0], 0)], P, im), fl=fl, spare=True))
            made += 1
            if made == 3:
                break
        print("planted on floor", fl, made, flush=True)
    save(out, "routes")
    return out


# ---------- door tags ----------
DOORS = [  # (id, level, tag text, tag xy, arc xy, label, source, note)
    ("mml_baxter", "main", "2", (1933, 881), (1905, 876), "bad", "real", "tool matched tag 2 to the entry door arc, r 33.74; tag 2 sits in the swing of the 36 in door to the living room"),
    ("mml_baxter", "main", "10", (1788, 347), (1743, 311), "bad", "real", "tool matched tag 10 to the exterior door; tag 10 sits at the Powder Room door"),
    ("mml_baxter", "upper", "10", (2033, 684), (2045, 700), "bad", "real", "the 10 is an interior elevation key inside a diamond, not a door tag"),
    ("mml_capitol", "main", "2", (1818, 1026), (1793, 1021), "bad", "planted", "leader moved from door 2 to the entry door arc of door 1"),
    ("mml_capitol", "main", "1", (1775, 1057), (1793, 1021), "good", "real", "tag 1 in the entry doorway, arc of that door"),
    ("mml_baxter", "main", "8", (1972, 278), (1976, 311), "good", "real", "tag 8 in the doorway, arc of that door"),
    ("mml_baxter", "main", "4", (1933, 619), (1969, 619), "good", "real", "tag 4 in the doorway, arc of that door"),
    ("mml_capitol", "ground", "6", (1950, 573), (1941, 538), "good", "real", "tag 6 in the doorway, arc of that door"),
]


def doors():
    res = {(r["id"], r["level"]): r for r in json.load(open(MINE / "doors.json"))}
    out = []
    for i, (id, lv, n, t, a, label, src, note) in enumerate(DOORS):
        r = res[(id, lv)]
        pg = vec.open_page(id, r["file"], r["page"])
        # exact positions from the extraction, matched to the rounded numbers above
        tag = min(r["marks"], key=lambda m: math.dist((m[1], m[2]), t))
        arc = min(r["arcs"], key=lambda q: math.dist(q["mid"], a))
        assert tag[0] == n and math.dist((tag[1], tag[2]), t) < 2 and math.dist(arc["mid"], a) < 2
        cx, cy = (tag[1] + arc["mid"][0]) / 2, (tag[2] + arc["mid"][1]) / 2
        R = 120
        im, P, s = draw.render(pg, (cx - R, cy - R, cx + R, cy + R), long=900)
        draw.leader(im, P(tag[1], tag[2]), P(*arc["mid"]), r=int(7.5 * s))
        out.append(dict(cat="door", label=label, source=src, note=f"{id} {lv}: {note}", claim=DOOR_T.format(n=n), img=write(im, f"door_{label}_{i}")))
    save(out, "doors")
    return out


# ---------- printed values ----------
def ftin(v):
    v = round(v * 2) / 2
    f, i = int(v // 12), v % 12
    if i == int(i):
        return f"{f}'-{int(i)}\""
    return f"{f}'-{int(i)} 1/2\""


def dim_rows(id, level):
    f = next(x for x in M.CAT[id]["floors"] if str(x["level"]) == level)
    pg = vec.open_page(id, f["file"], f["page"])
    segs, _ = vec.segments(pg)
    rect, note, _ = vec.plan_panel(pg, segs, rule_layer="A-anno-htch-docs")
    panel = fitz.Rect(rect * pg.rotation_matrix)
    ipp = vec.inches_per_pt(note)
    rows = M.dim_lines(pg, segs, panel)
    rects = {tuple(round(v, 1) for v in ((q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2)): q for t, q, d in M.rline_texts(pg)}
    delta = float(np.median([d["line_pt"] * ipp - d["declared"] for d in rows if d["line_pt"]]))
    out = []
    for d in rows:
        if not d["line"]:
            continue
        d["rect"] = rects[tuple(round(v, 1) for v in d["center"])]
        d["meas"] = d["line_pt"] * ipp - delta
        d["inset"] = delta / 2 / ipp
        out.append(d)
    return pg, out


def tick_ends(d):
    x0, y0, x1, y1 = d["line"]; k = d["inset"]
    return ((x0 + k, y0), (x1 - k, y1)) if d["horiz"] else ((x0, y0 + k), (x1, y1 - k))


def value_img(pg, d, new_text=None):
    a, b = tick_ends(d); q = d["rect"]
    m = 40
    clip = (min(a[0], b[0], q.x0) - m, min(a[1], b[1], q.y0) - m, max(a[0], b[0], q.x1) + m, max(a[1], b[1], q.y1) + m)
    # at least 200 pt each way so the neighbouring dimensions are in view
    cx, cy = (clip[0] + clip[2]) / 2, (clip[1] + clip[3]) / 2
    hw, hh = max((clip[2] - clip[0]) / 2, 110), max((clip[3] - clip[1]) / 2, 110)
    im, P, s = draw.render(pg, (cx - hw, cy - hh, cx + hw, cy + hh), long=1000)
    pad = 2
    p0, p1 = P(q.x0 - pad, q.y0 - pad), P(q.x1 + pad, q.y1 + pad)
    if new_text:  # a drafting error in the drawing itself, so the finding it causes is true
        im[p0[1]:p1[1], p0[0]:p1[0]] = 255
        H = (p1[1] - p0[1]) if d["horiz"] else (p1[0] - p0[0])
        size = int(H * .78)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
        w = int(font.getlength(new_text)) + 4
        tile = Image.new("L", (w, int(size * 1.25)), 255)
        ImageDraw.Draw(tile).text((2, 0), new_text, fill=0, font=font)
        t = np.array(tile)
        if not d["horiz"]:
            t = cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)
        y, x = (p0[1] + p1[1] - t.shape[0]) // 2, (p0[0] + p1[0] - t.shape[1]) // 2
        roi = im[y:y + t.shape[0], x:x + t.shape[1]]
        roi[:] = np.minimum(roi, cv2.cvtColor(t, cv2.COLOR_GRAY2BGR))
    draw.box(im, p0, p1)
    draw.ticks(im, P(*a), P(*b))
    return im


CLINIC55 = "2012-03-27-ClinicRedactedDrawings/Clinic_055_First Floor Plan Block 1C.pdf"
MM_PER_PT = 25.4 / 72 * 50


def clinic_span(a, x, y, vertical):
    """lab/modern/clinic.py measure_dim, returning where the span is: ((x0, y0), (x1, y1)) in points, or None."""
    from clinic_ticks import ticks_on
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
            return ((lo.max(), L[1]), (hi.min(), L[1])) if not vertical else ((L[0], lo.max()), (L[0], hi.min()))
    return None


def clinic_img(pg, a, r):
    sp = clinic_span(a, r["x"], r["y"], r["vertical"])
    assert sp, r
    w, h = (9, 24) if r["vertical"] else (24, 9)
    dx = -10 if r["value"] == 1763 else 0  # checked by eye: Spark's centre for this number is 10 pt right of the digits, and the box must hold the number
    q = (r["x"] - w + dx, r["y"] - h, r["x"] + w + dx, r["y"] + h)
    (ax, ay), (bx, by) = sp
    m = 40
    clip = (min(ax, bx, q[0]) - m, min(ay, by, q[1]) - m, max(ax, bx, q[2]) + m, max(ay, by, q[3]) + m)
    cx, cy = (clip[0] + clip[2]) / 2, (clip[1] + clip[3]) / 2
    hw, hh = max((clip[2] - clip[0]) / 2, 150), max((clip[3] - clip[1]) / 2, 150)
    im, P, s = draw.render(pg, (cx - hw, cy - hh, cx + hw, cy + hh), long=1100)
    draw.box(im, P(q[0], q[1]), P(q[2], q[3]))
    draw.ticks(im, P(ax, ay), P(bx, by))
    return im


def values():
    rng = random.Random(SEED + 2)
    out = []
    rows = json.load(open(ROOT / "data/lab/modern/clinic_dims.json"))
    pg55 = vec.open_page("nibs_clinic", CLINIC55, 0)
    segs, _ = vec.segments(pg55)
    a55 = vec.seg_arrays(segs)
    have = [r for r in rows if r["measured_mm"]]
    agree = [r for r in have if abs(r["measured_mm"] - r["value"]) <= max(30, .03 * r["value"])]
    wrong = [r for r in have if r not in agree and r["value"] >= 1000]
    rng.shuffle(wrong); rng.shuffle(agree)
    for i, r in enumerate(wrong):  # all kept as candidates; the printed number is checked by eye
        out.append(dict(cat="value", label="bad", source="real", note=f"clinic sheet 55 mis-pair: printed {r['value']}, ticks {r['measured_mm']} mm", claim=VALUE_T.format(cited=f"{r['value']} mm", meas=f"{r['measured_mm']} mm"), img=write(clinic_img(pg55, a55, r), f"value_wrong_{i}"), spare=True))
    big = [r for r in agree if r["value"] >= 1000][:4]
    for i, r in enumerate(big):
        out.append(dict(cat="value", label="good", source="real", note=f"clinic sheet 55 agrees: printed {r['value']}, measured {r['measured_mm']} mm", claim=VALUE_T.format(cited=f"{r['value']} mm", meas=f"{r['measured_mm']} mm"), img=write(clinic_img(pg55, a55, r), f"value_agree_{i}"), spare=True))
    # text-layer sheets: exact printed strings and exact spans
    for id, lv, tag in (("mml_capitol", "main", "cm"), ("mml_baxter", "main", "bm"), ("mml_capitol", "upper", "cu")):
        pg, ds = dim_rows(id, lv)
        ok = [d for d in ds if abs(d["meas"] - d["declared"]) <= .3 and d["declared"] >= 40]
        rng.shuffle(ok)
        for i, d in enumerate(ok[:2]):
            out.append(dict(cat="value", label="good", source="real", note=f"{id} {lv}: printed {d['text']}, measured {ftin(d['meas'])}", claim=VALUE_T.format(cited=d["text"], meas=ftin(d["meas"])), img=write(value_img(pg, d), f"value_ok_{tag}_{i}"), spare=True))
        for i, d in enumerate(ok[2:4]):
            # misread: one digit of the printed value quoted wrong, measured value true
            wrong_v = d["declared"] + rng.choice([-6, 6, 12, -12])
            out.append(dict(cat="value", label="bad", source="planted", note=f"{id} {lv}: misquote, printed {d['text']}, finding cites {ftin(wrong_v)}", claim=VALUE_T.format(cited=ftin(wrong_v), meas=ftin(d["meas"])), img=write(value_img(pg, d), f"value_misq_{tag}_{i}"), spare=True))
        for i, d in enumerate(ok[4:6]):
            # the overlay sits on the wrong dimension: finding quotes a neighbour, both numbers agree, the marked dimension is another
            n = min((x for x in ok if abs(x["declared"] - d["declared"]) >= 6), key=lambda x: math.dist(x["center"], d["center"]))
            out.append(dict(cat="value", label="bad", source="planted", note=f"{id} {lv}: overlay on {d['text']}, finding quotes {n['text']}", claim=VALUE_T.format(cited=n["text"], meas=n["text"]), img=write(value_img(pg, d), f"value_neigh_{tag}_{i}"), spare=True))
        for i, d in enumerate(ok[6:8]):
            # a true discrepancy: printed text changed in the drawing, finding quotes it right
            nv = d["declared"] + rng.choice([-5, 5, 7, -7])
            txt = ftin(nv)
            out.append(dict(cat="value", label="good", source="planted", note=f"{id} {lv}: drawing text changed from {d['text']} to {txt}, so the finding is true", claim=VALUE_T.format(cited=txt, meas=ftin(d["meas"])), img=write(value_img(pg, d, new_text=txt), f"value_true_{tag}_{i}"), spare=True))
    save(out, "values")
    return out


def contact(names, out, W=560, cols=3):
    t = []
    for n in names:
        im = cv2.imread(str(CAND / n))
        s = W / max(im.shape[:2])
        v = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        v = cv2.copyMakeBorder(v, 18, W - v.shape[0], 0, W - v.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(v, n[:40], (2, 13), 0, .42, (0, 0, 200), 1)
        t.append(v)
    while len(t) % cols:
        t.append(np.full_like(t[0], 255))
    cv2.imwrite(str(MINE / out), np.vstack([np.hstack(t[i:i + cols]) for i in range(0, len(t), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("stairs", stairs), ("routes", routes), ("doors", doors), ("values", values)):
        if stage in (name, "all"):
            c = fn()
            print(name, len(c), [x["img"] for x in c])
            for i in range(0, len(c), 9):
                contact([x["img"] for x in c[i:i + 9]], f"cand_{name}_{i // 9}.jpg")
