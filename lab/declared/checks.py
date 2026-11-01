"""Deterministic declared-against-drawn checks. Readers feed these; each check appends flags and records what it compared.

Run: env LAB_NAME=declared LAB_BUDGET=8 LAB_TOTAL=75 .venv/bin/python -I lab/declared/checks.py dev|held
Writes data/lab/declared/flags_<scope>.json. Spark reads come from data/lab/declared/spark.jsonl through read_spark.load().
"""
import collections
import difflib
import json
import math
import re
import sys

sys.path.insert(0, "lab/declared")
sys.path.insert(0, "lab/doors")
import numpy as np  # noqa: E402

import fitz  # noqa: E402
import read_tables as RT  # noqa: E402
import read_text as T  # noqa: E402
import rules as R  # noqa: E402
import vec_arcs as VA  # noqa: E402

OUT = T.ROOT / "data/lab/declared"
PT_PER_IN = 1.5  # 1/4 in = 1 ft
SWING = {"LEFT", "RIGHT"}
NO_ARC = {"BIFOLD", "POCKET", "-", ""}
FLAGS, STATS = [], collections.defaultdict(lambda: collections.defaultdict(dict))


def flag(sid, check, sub, sheet, text, boxes=(), source="text", **kw):
    FLAGS.append(dict(id=f"{sid}.{check}.{sub}.{sum(f['set'] == sid and f['sub'] == sub for f in FLAGS)}", set=sid, check=check, sub=sub, sheet=sheet, text=text, boxes=[list(b) for b in boxes], source=source, **kw))


def stat(sid, check, **kw):
    STATS[sid][check].update(kw)


# ---- check 1: sheet index against sheets present and title blocks ----
def check1(sid, tbs, rows, source="text", cover_page=0):
    """tbs: title blocks with sheet_no, title, page. rows: index rows with sheet_no, title, box."""
    have = collections.defaultdict(list)
    for t in tbs:
        if t["sheet_no"]:
            have[t["sheet_no"]].append(t)
    idx = collections.defaultdict(list)
    for r in rows:
        idx[r["sheet_no"]].append(r)
    exact = strict = 0
    for no, rs in idx.items():
        if no not in have:
            flag(sid, 1, "index_missing_sheet", no, f"index lists {no} {rs[0]['title']!r}, no sheet in the set carries that number", [(cover_page, rs[0]["box"])], source)
            continue
        ts = have[no]
        titles = [r["title"] for r in rs]
        exact += not any(a == ts[0]["title"] for a in titles)
        strict += not any(R.norm(a) == R.norm(ts[0]["title"]) for a in titles)
        if not any(R.title_same(a, ts[0]["title"]) for a in titles):
            flag(sid, 1, "title_mismatch", no, f"index {titles[0]!r} against title block {ts[0]['title']!r}", [(cover_page, rs[0]["box"]), (ts[0]["page"], ts[0]["box"])], source)
        if len(rs) > 1:
            flag(sid, 1, "index_duplicate", no, f"index lists {no} {len(rs)} times: {titles}", [(cover_page, r["box"]) for r in rs], source)
    for no, ts in have.items():
        if no not in idx:
            flag(sid, 1, "unindexed_sheet", no, f"sheet {no} {ts[0]['title']!r} is in the set but not in the index", [(ts[0]["page"], ts[0]["box"])], source)
        if len(ts) > 1:
            flag(sid, 1, "duplicate_number", no, f"{len(ts)} pages carry sheet number {no}", [(t["page"], t["box"]) for t in ts], source)
    stat(sid, 1, items=len(idx) + sum(no not in idx for no in have), pages=len(tbs), read=sum(bool(t["sheet_no"]) for t in tbs), index_rows=len(rows), baseline_exact_flags=exact, normaliser_only_flags=strict)


# ---- check 5: door and window tags against schedules ----
def inches(s):
    """Width in inches from 36" x 80" or 3' - 0"."""
    m = re.search(r'(\d+)"\s*x', s or "")
    if m:
        return int(m[1])
    m = re.search(r"(\d+)'\s*-\s*(\d+)", s or "")
    return int(m[1]) * 12 + int(m[2]) if m else None


def arcs(pg):
    """Door leaf arcs in the rendered frame, radius 24 to 40 in at 1/4 in = 1 ft. Reuses lab/doors/vec_arcs.py."""
    s = VA.page_segments(pg)
    if not len(s):
        return []
    M = pg.rotation_matrix
    out = []
    for a in VA.attach_leaf(VA.find_arcs(s, 24 * PT_PER_IN, 40 * PT_PER_IN, smin=70), s):
        p = fitz.Point(a["cx"], a["cy"]) * M
        # shortcut: one fragment means solid, so a solid arc cut by a crossing line is lost too; a real dash test needs the path's dash pattern
        if a["leaf"] and a["frags"] == 1 and not any(math.dist((p.x, p.y), q[:2]) < 3 for q in out):  # a dashed swing is an option, not a door
            mid = (fitz.Point(a["a"]) * M + fitz.Point(a["b"]) * M) / 2
            out.append((p.x, p.y, a["r"] / PT_PER_IN, mid.x, mid.y))
    return out


def pick_family(tags, marks):
    fam = collections.Counter()
    for t, _, _, f in tags:
        fam[f] += t in marks
    return fam.most_common(1)[0][0] if fam and fam.most_common(1)[0][1] else None


def check5(sid, sheet_of, pairs):
    """pairs: (schedule sheet page, table, plan page). One schedule can serve one plan page."""
    d = T.doc(sid)
    doors = windows = both = 0
    for sp, tb, pp in pairs:
        pg = d[pp]
        sheet = sheet_of(pp)
        tags = RT.shape_tags(pg)
        key = tb["cols"][0]
        marks = {r[key]: r for r in tb["rows"]}
        boxes = {r[key]: r["_box"] for r in tb["rows"]}
        fams = collections.Counter(f for *_, f in tags)
        if tb["kind"] == "window":
            kinds = [f for f in fams if f[2] in ("poly", "rect")]
            fam = next((f for f in sorted(kinds, key=lambda f: -fams[f]) if {t for t, _, _, g in tags if g == f} & set(marks)), None)
            if fam is None:
                stat(sid, 5, note_window=f"no window tags found on {sheet}")
                continue
            same = lambda f: f[2] == fam[2] and abs(f[0] - fam[0]) <= 1 and abs(f[1] - fam[1]) <= 1
            count = collections.Counter(t for t, _, _, f in tags if same(f))
            windows += len(marks)
            for m, r in marks.items():
                both += 1
                if "QTY" not in r:
                    if not count[m]:
                        flag(sid, 5, "row_not_on_plan", sheet, f"window schedule row {m} has no tag on the plan", [(sp, r["_box"])])
                    continue
                q = int(r["QTY"])
                if count[m] != q:
                    flag(sid, 5, "window_qty", sheet, f"window schedule mark {m} QTY {q}, plan shows {count[m]} tags", [(sp, r["_box"])] + [(pp, [x - 40, y - 40, x + 40, y + 40]) for t, x, y, f in tags if t == m and same(f)][:6])
            for m in set(count) - set(marks):
                flag(sid, 5, "tag_not_in_schedule", sheet, f"window tag {m} on the plan has no schedule row", [(pp, [x - 40, y - 40, x + 40, y + 40]) for t, x, y, f in tags if t == m and same(f)][:3])
            continue
        fam = next((f for f in sorted(fams, key=lambda f: -sum(t in marks for t, _, _, g in tags if g == f)) if f[2] in ("circle", "rect") and all(t.isdigit() for t, _, _, g in tags if g == f)), None)
        tg = [(t, x, y) for t, x, y, f in tags if f[:2] in ((fam[0], fam[1]), (fam[0], fam[1] - 1)) and f[2] == fam[2]] if fam else []
        on = {t for t, _, _ in tg}
        doors += len(marks)
        if not on:
            stat(sid, 5, note_door=f"no door tags found on {sheet}")
            continue
        miss = sorted(set(marks) - on, key=lambda s: (len(s), s))
        ghost = sorted(on - set(marks), key=lambda s: (len(s), s))
        if miss:
            flag(sid, 5, "row_not_on_plan", sheet, f"schedule rows {miss} have no tag on the plan", [(sp, boxes[m]) for m in miss[:3]] + [(pp, [min(x for _, x, _ in tg) - 30, min(y for _, _, y in tg) - 30, max(x for _, x, _ in tg) + 30, max(y for _, _, y in tg) + 30])], members=miss)
        if ghost:
            flag(sid, 5, "tag_not_in_schedule", sheet, f"plan tags {ghost} have no schedule row", [(pp, [x - 40, y - 40, x + 40, y + 40]) for t, x, y in tg if t in ghost][:4] + [(sp, tb["box"])], members=ghost)
        # width and type against the drawn arc
        ar = arcs(pg)
        paired = [(t, x, y) for t, x, y in tg if t in marks]
        tested = 0
        near = {}  # each arc goes to its nearest tag, and a tag keeps its nearest arc
        for a in ar:
            gap, i = min((math.dist((a[3], a[4]), (x, y)), i) for i, (_, x, y) in enumerate(paired)) if paired else (999, 0)
            if gap < 75 and (i not in near or gap < near[i][0]):
                near[i] = (gap, a)
        for i, (_, a) in near.items():
            t, x, y = paired[i]
            r = marks[t]
            swing = (r.get("SWING") or r.get("TYPE") or "").upper()
            w = inches(r.get("SIZE") or r.get("WIDTH"))
            tested += 1
            box = [(pp, [min(x, a[0]) - 60, min(y, a[1]) - 60, max(x, a[0]) + 60, max(y, a[1]) + 60]), (sp, r["_box"])]
            if swing in NO_ARC and "SWING" in r:
                flag(sid, 5, "type_vs_arc", sheet, f"tag {t}: schedule says {swing or 'none'} {r.get('SIZE')}, plan shows a swing arc of {a[2]:.1f} in", box)
            elif w and abs(a[2] - w) > R.DOOR_WIDTH_TOL_IN:
                flag(sid, 5, "width_vs_arc", sheet, f"tag {t}: schedule width {w} in, drawn arc {a[2]:.1f} in", box)
        stat(sid, 5, **{f"arc_tested_{pp}": tested})
        for t, r in marks.items():
            sw = (r.get("SWING") or "").upper()
            if "SWING" in r and sw not in SWING | NO_ARC:
                flag(sid, 5, "schedule_cell", sheet, f"schedule row {t} SWING column reads {sw!r}", [(sp, r["_box"])])
    stat(sid, 5, items=doors, window_rows=both)


# ---- check 6: floor label in the title block against the label on the drawing ----
def check6(sid, tbs):
    d = T.doc(sid)
    n = 0
    for t in tbs:
        pg = d[t["page"]]
        ft = R.floors(t["title"])
        labels = T.drawing_labels(pg)
        fl = set().union(*[R.floors(x["t"]) for x in labels]) if labels else set()
        if ft and fl:
            n += 1
            if not ft & fl:
                flag(sid, 6, "floor_label", t["sheet_no"], f"title block {t['title']!r}, drawing labels {[x['t'] for x in labels][:3]}", [(t["page"], t["box"])] + [(t["page"], [x["x0"], x["y0"], x["x1"], x["y1"]]) for x in labels[:2]])
    stat(sid, 6, items=n, pages=len(tbs))


def check6_numbering(sid):
    """Sets with no title block: the floor named on the sheet against the first digit of its room numbers."""
    d = T.doc(sid)
    want = {"GROUND": "0", "1": "1", "2": "2", "3": "3"}
    n = 0
    for i, pg in enumerate(d):
        ls = T.lines(pg)
        lab = [x for x in ls if re.search(r"floor plan", x["t"], re.I)]
        nums = [x["t"] for x in ls if re.fullmatch(r"\d{3}", x["t"])]
        if lab and nums:
            n += 1
            fl = R.floors(lab[0]["t"])
            digit = collections.Counter(x[0] for x in nums).most_common(1)[0][0]
            if fl and digit != want.get(next(iter(fl))):
                flag(sid, 6, "floor_numbering", str(i), f"label {lab[0]['t']!r}, room numbers start with {digit}", [(i, [lab[0]["x0"], lab[0]["y0"], lab[0]["x1"], lab[0]["y1"]])])
    stat(sid, 6, items=n, pages=len(d))


# ---- check 7: room names and numbers on plans against the finish schedule ----
ROOM_NO = re.compile(r"^[A-Z]?\d{3}[A-Z]?$")


def plan_rooms(pg, exclude, sheets=()):
    """Room number lines with the name printed above them, outside schedule boxes. A number with no name, or a sheet number, is not a room."""
    ls = T.lines(pg)
    inside = lambda x: any(b[0] - 5 <= x["x0"] <= b[2] + 5 and b[1] - 5 <= x["y0"] <= b[3] + 5 for b in exclude)
    out = []
    for x in ls:
        if ROOM_NO.match(x["t"]) and not inside(x):
            above = sorted((o for o in ls if 0 < x["y0"] - o["y0"] < 34 and abs((o["x0"] + o["x1"]) / 2 - (x["x0"] + x["x1"]) / 2) < 45 and re.search("[A-Za-z]{2}", o["t"]) and not inside(o)), key=lambda o: -o["y0"])
            name = " ".join(o["t"] for o in above[:2][::-1])
            if name and x["t"] not in sheets:
                out.append(dict(no=x["t"], name=name, box=[x["x0"] - 30, x["y0"] - 28, x["x1"] + 30, x["y1"] + 6]))
    return out


def room_same(a, b):
    na, nb = R.norm(a), R.norm(b)
    if not na or not nb or na == nb or na in nb or nb in na:
        return True
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.6:
        return True
    ta, tb = na.replace("/", " ").split(), nb.replace("/", " ").split()
    short, long_ = sorted((ta, tb), key=len)
    return all(any(w.startswith(s[:3]) for w in long_) for s in short)


def check7(sid, sheet_of, finish, plan_pages, source="text", sheets=()):
    """finish: rows {no, name, box, page}. plan_pages: rendered pages holding the plan labels for those rows."""
    d = T.doc(sid)
    rooms = {}
    for p, ex in plan_pages:
        for r in plan_rooms(d[p], ex, sheets):
            rooms.setdefault(r["no"], dict(r, page=p))
    n = 0
    if not rooms:  # the plan prints names only, so test the names
        for p, ex in plan_pages:
            names = {R.norm(x["t"]) for x in T.lines(d[p]) if not any(b[0] <= x["x0"] <= b[2] and b[1] <= x["y0"] <= b[3] for b in ex)}
            for r in finish:
                n += 1
                if R.norm(r["name"]) not in names:
                    flag(sid, 7, "schedule_room_not_on_plan", sheet_of(p), f"finish schedule {r['no']} {r['name']!r} is not on the plan", [(r["page"], r["box"])], source)
        stat(sid, 7, items=n, plan_rooms=0, note="plan prints names, not numbers")
        return
    for r in finish:
        n += 1
        got = rooms.get(r["no"])
        if not got:
            flag(sid, 7, "schedule_room_not_on_plan", r.get("sheet") or sheet_of(r["page"]), f"finish schedule {r['no']} {r['name']!r} is not on the plan", [(r["page"], r["box"])] if r.get("box") else [], source)
        elif not room_same(r["name"], got["name"]):
            flag(sid, 7, "room_name_mismatch", sheet_of(got["page"]), f"room {r['no']}: schedule {r['name']!r}, plan {got['name']!r}", ([(r["page"], r["box"])] if r.get("box") else []) + [(got["page"], got["box"])], source)
    have = {r["no"] for r in finish}
    hit = [g["box"] for no, g in rooms.items() if no in have]
    area = [min(b[0] for b in hit), min(b[1] for b in hit), max(b[2] for b in hit), max(b[3] for b in hit)] if hit else None
    for no, got in rooms.items():
        # rooms beyond the rooms a schedule covers are outside its scope, such as existing rooms past an area of work
        inside = area and area[0] <= got["box"][0] and got["box"][2] <= area[2] and area[1] <= got["box"][1] and got["box"][3] <= area[3]
        if no not in have and inside:
            flag(sid, 7, "plan_room_not_in_schedule", sheet_of(got["page"]), f"plan room {no} {got['name']!r} has no finish schedule row", [(got["page"], got["box"])], source)
    stat(sid, 7, items=n, plan_rooms=len(rooms))


# ---- reads that came from Spark: agree over both runs before anything rests on them ----
def agreed(a, b):
    """Rows present in both reads, keyed by a string."""
    return {k: v for k, v in a.items() if k in b and b[k] == v}


def check5_spark(sid, sheet, rows, tags, hw, sp, pp):
    """rows: {door: (width, rating, hardware)}. tags: door tags read on the plan in both reads. hw: {door: set}."""
    tb = T.page(sid, sp).rect
    table = [0.66 * tb.width, 0.62 * tb.height, 0.905 * tb.width, 0.985 * tb.height]
    miss = sorted(set(rows) - tags)
    ghost = sorted(tags - set(rows))
    if miss:
        flag(sid, 5, "row_not_on_plan", sheet, f"schedule rows {miss} have no tag on the plan", [(sp, table)], "spark", members=miss)
    if ghost:
        flag(sid, 5, "tag_not_in_schedule", sheet, f"plan tags {ghost} have no schedule row", [(sp, table)], "spark", members=ghost)
    for d, (w, r, h) in rows.items():
        if d in hw and R.norm(hw[d]).replace(" ", "") != R.norm(h).replace(" ", ""):
            flag(sid, 5, "hardware_set", sheet, f"door {d}: schedule {h}, hardware schedule {hw[d]}", [(sp, table)], "spark")
    for d in set(hw) - set(rows):
        flag(sid, 5, "hardware_set", sheet, f"hardware schedule lists {d}, no such door row", [(sp, table)], "spark")
    for d in set(rows) - set(hw):
        flag(sid, 5, "hardware_set", sheet, f"door {d} is in no hardware set", [(sp, table)], "spark")
    stat(sid, 5, items=len(rows), tags=len(tags), hardware_doors=len(hw))


# ---- check 3: code summary fields and cross-checks ----
OCC_WORDS = ("ASSEMBLY", "BUSINESS", "EDUCATIONAL", "FACTORY", "HAZARD", "INSTITUTIONAL", "MERCANTILE", "RESIDENTIAL", "STORAGE", "UTILITY")
EDITION = re.compile(r"\b((?:19|20)\d\d)\s+(?:[A-Z][A-Za-z]+\s+){0,3}(?:CBC|CRC|IBC|IRC|UBC|Code)\b|\b(?:CBC|IBC|IRC|Code)[, ]+(?:(?:19|20)\d\d)\b")


def text_edition(sid, pages):
    """A code name beside a year anywhere on the cover pages' text layer."""
    pg = [T.page(sid, p) for p in pages]
    return next((m[0] for p in pg for m in [EDITION.search(p.get_text())] if m), "")


def check3(sid, reads, sheet, page, floors_declared=None, floors_drawn=None, source="spark"):
    """reads: the code fields from each run. A flag needs both runs to say the same thing."""
    a, b = reads[0], reads[-1]
    both = lambda f: f(a) and f(b)
    items = 0
    pgr = T.page(sid, page).rect
    box = [(page, [0, 0, pgr.width, pgr.height])]
    items += 1
    has_year = lambda r: bool(re.search(r"(19|20)\d\d", r["code_edition"]))
    edition = has_year(a) or has_year(b) or text_edition(sid, [page])
    if not edition and sid != "nibs_clinic":
        edition = text_edition(sid, list(range(min(3, len(T.doc(sid))))))
    if not edition:
        flag(sid, 3, "edition_missing", sheet, f"no code edition found. Read: {a['code_edition']!r}", box, source)
    items += 1
    if both(lambda r: r["sprinklered"] == "not_stated"):
        flag(sid, 3, "sprinkler_missing", sheet, "no sprinkler statement on the code sheet", box, source)
    items += 1
    if a["sprinklered"] == b["sprinklered"]:
        m = [x for r in (a, b) for x in r["sprinkler_mentions"]]
        if a["sprinklered"] == "no" and any(re.search(r"SPRINKLER HEADS|yes box|\bYES\b", x, re.I) for x in m):
            flag(sid, 3, "sprinkler_conflict", sheet, f"sprinkler statement no, but the sheet also says: {[x for x in m if re.search(r'SPRINKLER HEADS|yes box|YES', x, re.I)][:2]}", box, source)
        if a["sprinklered"] == "yes" and any(re.search(r"not sprinklered|unsprinklered", x, re.I) for x in m):
            flag(sid, 3, "sprinkler_conflict", sheet, f"sprinkler statement yes, but the sheet also says otherwise: {m[:2]}", box, source)
    items += 1
    occ = R.norm(a["occupancy"])
    declared = {w for w in OCC_WORDS if w in occ}
    other = {w for r in (a, b) for x in r["occupancy_mentions"] for w in OCC_WORDS if w in R.norm(x)}
    other_b = {w for x in b["occupancy_mentions"] for w in OCC_WORDS if w in R.norm(x)}
    if declared and other - declared and other_b - declared:
        flag(sid, 3, "occupancy_conflict", sheet, f"occupancy {a['occupancy']!r}, also {sorted(a['occupancy_mentions'] + b['occupancy_mentions'])[:3]}", box, source)
    if floors_declared is not None and floors_drawn is not None:
        items += 1
        if floors_declared != floors_drawn:
            flag(sid, 3, "stories_mismatch", sheet, f"code data declares {floors_declared}, floor plans present {floors_drawn}", box, source)
    stat(sid, 3, items=items, unstable=[f for f in ("code_edition", "occupancy", "construction", "sprinklered", "area", "occupant_load") if R.norm(str(a[f])) != R.norm(str(b[f]))])


# ---- check 4: table arithmetic ----
def num(s):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(s))
    return float(m[0].replace(",", "")) if m else None


def check4_area_table(sid, page=0):
    """Level areas on the cover against the printed total."""
    pg = T.doc(sid)[page]
    ls = T.lines(pg)
    levels = [x for x in ls if re.match(r"^(Ground|Main|Upper|Lower|Basement)\s+Level:?$", x["t"])]
    rows = []
    for x in levels:
        v = [o for o in ls if abs(o["y0"] - x["y0"]) < 3 and "ft²" in o["t"] and o["x0"] > x["x0"]]
        if v:
            rows.append((x["t"], num(v[0]["t"]), [x["x0"], x["y0"], v[0]["x1"], v[0]["y1"]]))
    tot = [x for x in ls if re.search(r"ft².*not include", x["t"], re.I)]
    if rows and tot:
        total = num(tot[0]["t"])
        s = sum(r[1] for r in rows)
        stat(sid, 4, area_rows=len(rows))
        if abs(s - total) > R.AREA_SUM_TOL_FRAC * total:
            flag(sid, 4, "area_sum", "A1.0", f"levels sum to {s:.0f}, printed total {total:.0f}", [(page, [rows[0][2][0], rows[0][2][1], tot[0]["x1"], tot[0]["y1"]])])
        return len(rows) + 1
    return 0


def check4_light_vent(sid, sheet, page=7):
    """Light and ventilation table: area times percent against the required column, required against provided."""
    pg = T.doc(sid)[page]
    ls = T.lines(pg)
    hd = [x for x in ls if "LIGHT & VENTILATION CALCULATION" in x["t"].upper()]
    if not hd:
        return 0
    n = 0
    rows = [r for r in RT.rows_of([x for x in ls if x["y0"] > hd[0]["y0"] + 20 and x["y0"] < hd[0]["y0"] + 110 and x["x0"] > hd[0]["x0"] - 80]) if re.search(r"\d+ SF", " ".join(c["t"] for c in r))]
    for r in rows:
        cells = [c["t"] for c in r]
        name, area = cells[0], num(cells[1])
        vals = [num(c) for c in cells[2:]]
        if len(vals) != 6:
            continue
        lp, lr, lprov, vp, vr, vprov = vals
        n += 4
        box = [(page, [r[0]["x0"], r[0]["y0"], r[-1]["x1"], r[-1]["y1"]])]
        if math.ceil(area * lp / 100) != lr:
            flag(sid, 4, "light_calc", sheet, f"{name}: {area:.0f} SF at {lp:.0f}% needs {math.ceil(area * lp / 100)}, table says {lr:.0f}", box)
        if math.ceil(area * vp / 100) != vr:
            flag(sid, 4, "vent_calc", sheet, f"{name}: {area:.0f} SF at {vp:.0f}% needs {math.ceil(area * vp / 100)}, table says {vr:.0f}", box)
        if lprov < lr:
            flag(sid, 4, "light_short", sheet, f"{name}: light provided {lprov:.0f} below required {lr:.0f}", box)
        if vprov < vr:
            flag(sid, 4, "vent_short", sheet, f"{name}: vent provided {vprov:.0f} below required {vr:.0f}", box)
    return n


def check4_bcoe():
    """Sheet A012: tables parsed by lab/gather_practice/declared_a012.py, plus the plan labels and the notes."""
    sid = "bcoe_addendum3"
    sys.path.insert(0, str(T.ROOT / "lab/gather_practice"))  # reuse its row parser, not its output file
    import declared_a012 as DA
    DA_CORR, DA_rows = DA.CORR, DA.rows
    d = json.loads((T.ROOT / "data/lab/gather_practice/declared_a012.json").read_text())
    pg = T.doc(sid)[1]
    ls = T.lines(pg)
    words = pg.get_text("words")
    box_of = lambda s: next(([x["x0"], x["y0"], x["x1"], x["y1"]] for x in ls if s in x["t"]), [2000, 300, 2420, 970])
    table = [1990, 300, 2420, 970]
    exits = [1990, 1840, 2420, 2000]
    items = 0
    for r in d["rooms_where_ceiling_differs"]:
        flag(sid, 4, "occupant_calc", "A012", f"room {r['room']} {r['name']}: {r['area']} SF / {r['factor']} = {r['recomputed']}, table says {r['declared']}", [(1, table)])
    items += d["rooms_parsed"]
    if d["occupants_sum"] != d["occupants_declared_total"]:
        flag(sid, 4, "occupant_sum", "A012", f"rows sum to {d['occupants_sum']}, printed total {d['occupants_declared_total']}", [(1, table)])
    items += 1
    if abs(d["area_sum_sf"] - d["area_declared_grand_total_sf"]) > R.AREA_SUM_TOL_FRAC * d["area_declared_grand_total_sf"]:
        flag(sid, 4, "area_sum", "A012", f"rows sum to {d['area_sum_sf']} SF, printed grand total {d['area_declared_grand_total_sf']} SF", [(1, table)])
    items += 1
    for e in d["exits"]:
        items += 1
        if abs(e["required_in"] - R.exit_required_in(e["occupants"])) > R.ROUNDING_TOL_IN:
            flag(sid, 4, "exit_width_calc", "A012", f"exit {e['exit']}: {e['occupants']} x 0.2 = {e['recomputed_in']}, table says {e['required_in']}", [(1, exits)])
        if e["clear_in"] < e["required_in"]:
            flag(sid, 4, "exit_width_short", "A012", f"exit {e['exit']}: clear {e['clear_in']} in below required {e['required_in']} in", [(1, exits)])
    items += 1
    if d["worked_example_exit_1_occupants"] != d["exit_table_exit_1_occupants"]:
        flag(sid, 4, "worked_example", "A012", f"worked example exit 1 uses {d['worked_example_exit_1_occupants']} occupants and {d['worked_example_exit_1_required_in']} in, table says {d['exit_table_exit_1_occupants']} occupants and 18 in", [(1, exits)])
    items += 1
    if d["exit_occupant_sum"] != d["occupants_declared_total"]:
        flag(sid, 4, "exit_sum_vs_total", "A012", f"exit table carries {d['exit_occupant_sum']} occupants, occupant total is {d['occupants_declared_total']}", [(1, exits)])
    corr = [m.groups() for m in map(DA_CORR.match, DA_rows(words, 1990, 2420, 1340, 1436)) if m]
    for no, name, occ, x, y in corr:
        items += 1
        if abs(float(x) - int(occ) * R.EXIT_IN_PER_OCC) > R.ROUNDING_TOL_IN:
            flag(sid, 4, "corridor_calc", "A012", f"corridor {no}: {occ} x 0.2 = {int(occ) * 0.2:.1f}, table says {x}", [(1, [1990, 1340, 2420, 1436])])
    note = re.search(r"\(([\d,]+) SF\)\s*-\s*\(CIRCULATION \+ RESTROOMS\) = ([\d,]+) SF", " ".join(x["t"] for x in ls))
    if note:
        items += 1
        total, net = num(note[1]), num(note[2])
        calc = total - d["area_rooms_without_a_load_sf"]
        if abs(calc - net) > R.AREA_SUM_TOL_FRAC * net:
            flag(sid, 4, "worked_example", "A012", f"note says {total:,.0f} - (circulation + restrooms) = {net:,.0f} SF, table rows without a load sum to {d['area_rooms_without_a_load_sf']} SF, which gives {calc:,.0f} SF", [(1, [2000, 1000, 2430, 1060]), (1, table)])
    # room labels drawn on the plan against the table
    lines = DA.merge_split_rows(DA.rows(words, 1990, 2420, 300, 970))
    table_rows = {m[1]: m for m in map(DA.ROOM.match, lines) if m}
    labels = [x for x in ls if re.match(r"^\(Area:\s+(\d+) SF", x["t"]) and x["x0"] < 1980]
    seen = 0
    for x in labels:
        area = int(re.match(r"^\(Area:\s+(\d+)", x["t"])[1])
        fac = [o for o in ls if abs(o["y0"] - x["y0"] - 8) < 4 and abs(o["x0"] - x["x0"]) < 40 and "Occ. Load Factor" in o["t"]]
        occ = [o for o in ls if abs(o["y0"] - x["y0"] - 15) < 5 and abs(o["x0"] - x["x0"]) < 40 and "Occupants" in o["t"]]
        rooms = [o for o in ls if 0 < x["y0"] - o["y0"] < 34 and abs((o["x0"] + o["x1"]) / 2 - (x["x0"] + x["x1"]) / 2) < 45 and ROOM_NO.match(o["t"])]
        if not (fac and occ and rooms):
            continue
        no = min(rooms, key=lambda o: x["y0"] - o["y0"])["t"]
        f = int(re.search(r"(\d+) SF/Occ", fac[0]["t"])[1])
        o = int(re.search(r"= (\d+) Occupants", occ[0]["t"])[1])
        if no not in table_rows:
            continue
        seen += 1
        ta, tf, to = table_rows[no].groups()[2:]
        if (area, str(f), str(o)) != (int(ta), tf, to):
            flag(sid, 4, "plan_label_vs_table", "A012", f"room {no}: plan label {area} SF / {f} = {o}, table {ta} SF / {tf} = {to}", [(1, [x["x0"] - 20, x["y0"] - 30, x["x1"] + 20, occ[0]["y1"] + 4]), (1, table)])
    stat(sid, 4, items=items + seen, plan_labels=seen, rooms=d["rooms_parsed"], corridor_rows=d["corridor_rows_parsed"])


# ---- check 8: exit markers on life safety sheets ----
def dedupe_markers(reads):
    """reads: for each run, a list of (x, y, width, cap) on the 0-1000 sheet. Tile overlaps merge within a run, then a marker must appear in both runs."""
    merged = []
    for r in reads:
        keep = []
        for m in r:
            if not any(math.dist(m[:2], k[:2]) < R.TILE_DEDUP_PERMILLE and m[3] == k[3] and R.norm(m[2]) == R.norm(k[2]) for k in keep):
                keep.append(m)
        merged.append(keep)
    a, b = merged[0], merged[-1]
    both = [m for m in a if any(math.dist(m[:2], k[:2]) < 2 * R.TILE_DEDUP_PERMILLE for k in b)]
    return both, merged


def check8(sid, sheet, page, reads, load=None, factor=R.EXIT_IN_PER_OCC):
    both, merged = dedupe_markers(reads)
    rect = T.page(sid, page).rect
    tot = sum(m[3] for m in both)
    for m in both:
        x, y = m[0] / 1000 * rect.width, m[1] / 1000 * rect.height
        box = [(page, [x - 120, y - 120, x + 120, y + 120])]
        if m[3] == 0:
            flag(sid, 8, "marker_no_capacity", sheet, f"exit marker near ({m[0]:.0f},{m[1]:.0f}) prints no capacity", box, "spark")
            continue
        w = num(m[2])
        if w and R.capacity_from_mm(w, factor) != m[3]:
            flag(sid, 8, "capacity_calc", sheet, f"exit width {m[2]} carries {R.capacity_from_mm(w, factor)} at {factor} in per person, marker says {m[3]}", box, "spark")
    if load and tot < load:
        flag(sid, 8, "capacity_below_load", sheet, f"exit capacity sums to {tot}, occupant load is {load}", [(page, [0, 0, rect.width, rect.height])], "spark")
    stat(sid, 8, items=len(both), capacity_sum=tot, per_read=[len(x) for x in merged])
    return both, tot


# ---- check 2: required sheets for a commercial submission ----
def check2(sid, titles, code_read, boxes=()):
    """titles: every sheet title in the set. code_read: True when occupancy and construction were read from some sheet."""
    if sid in R.RESIDENTIAL:
        stat(sid, 2, items=0, note="residential set, the commercial list does not apply")
        return
    up = [t.upper() for t in titles]
    want = {"code summary": code_read or any("CODE" in t for t in up),
            "life safety or exit plan": any(re.search(r"LIFE SAFETY|EXIT(ING)? PLAN|EGRESS PLAN", t) for t in up),
            "door schedule": any("DOOR" in t and "SCHEDULE" in t for t in up),
            "floor plan": any(re.search(r"FLOOR PLAN|PLAN . LEVEL|STANDARD DESIGN PLAN", t) for t in up)}
    for what, ok in want.items():
        if not ok:
            flag(sid, 2, "required_absent", "set", f"no {what} sheet in the set", boxes, scope="not_a_submission" if sid in R.NOT_A_SUBMISSION else "submission")
    stat(sid, 2, items=len(want))


# ---- check 7 for keyed sets: module legends against plan keys ----
def check7_keys(sid):
    d = T.doc(sid)
    tbs = T.title_blocks(sid)
    KEY = re.compile(r"^[A-Z]\d{1,2}$")
    legend, plans, overall = {}, {}, {}
    for i, pg in enumerate(d):
        ls = [x for x in T.lines(pg) if x["x0"] < 0.9 * pg.rect.width and x["size"] >= 9]
        keys = [x for x in ls if KEY.match(x["t"])]
        leg = {x["t"]: o["t"] for x in keys for o in ls if 0 < o["x0"] - x["x1"] < 40 and abs(o["y0"] - x["y0"]) < 3 and re.search("[A-Z]{3}", o["t"])}
        if leg:
            for k, v in leg.items():
                legend[k] = (i, v)
            for x in keys:
                if x["t"] not in leg or not any(0 < o["x0"] - x["x1"] < 40 and abs(o["y0"] - x["y0"]) < 3 and re.search("[A-Z]{3}", o["t"]) for o in ls):
                    plans.setdefault(x["t"], (i, [x["x0"] - 30, x["y0"] - 30, x["x1"] + 30, x["y1"] + 30]))
        elif keys:
            for x in keys:
                overall.setdefault(x["t"], (i, [x["x0"] - 30, x["y0"] - 30, x["x1"] + 30, x["y1"] + 30]))
    sheet = lambda p: tbs[p]["sheet_no"] or str(p)
    for k, (i, name) in legend.items():
        if k not in plans:
            flag(sid, 7, "legend_key_not_on_plan", sheet(i), f"legend key {k} {name!r} has no tag on the module plan", [(i, [100, 100, d[i].rect.width - 150, d[i].rect.height - 100])])
        if overall and k not in overall:
            flag(sid, 7, "legend_key_not_on_overall", sheet(i), f"legend key {k} {name!r} is not on the overall plan", [(i, [100, 100, d[i].rect.width - 150, d[i].rect.height - 100])])
    for k, (i, b) in {**plans, **overall}.items():
        if k not in legend:
            flag(sid, 7, "plan_key_not_in_legend", sheet(i), f"plan key {k} has no legend row", [(i, b)])
    stat(sid, 7, items=len(legend), overall_keys=len(overall), module_plan_keys=len(plans))
