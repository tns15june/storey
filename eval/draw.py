"""Draws a model as a 36 by 24 in vector PDF: live text, vector lines and one layer per trade of drawing, as a CAD export has."""
import math

import numpy as np
import pymupdf
from shapely.ops import unary_union

from .model import IN_PER_M, _rect, ft_in

W, H = 2592, 1728
MARGIN = 36
# shortcut: declared.title_blocks reads only text past 90% of the sheet's width, so the strip is 3.2 in wide, the widest whose text starts past that line. At 5 in no sheet of 710 gives
# its number or title from the text layer, so all 14 go to Spark, and the free dry review stops at the first. Widen it once that reader finds the block by its rules.
STRIP = 2326
FIELD = 60, 60, 2304, 1668   # what a drawing may fill
FONT, BOLD = "helv", "hebo"
BLACK, GREY, RED = (0, 0, 0), (0.45, 0.45, 0.45), (0.8, 0, 0)
LAYERS = ("A-WALL", "A-WALL-PATT", "A-DOOR", "A-GLAZ", "A-FLOR-STRS", "A-FLOR-FIXT", "A-FLOR-OTLN", "A-ANNO-DIMS", "A-ANNO-TEXT", "A-ANNO-SYMB", "G-ANNO-TTLB", "A-LIFE-SAFE")
KAPPA = 0.5523          # a quarter circle as one Bezier
TREAD_M = 0.27
GENERATED = "GENERATED TEST SET, NOT A REAL PROJECT"
CREDIT = "Geometry: Swiss Dwellings v3.0.0, CC BY 4.0."
NOTES = ("ALL DIMENSIONS ARE TO FACE OF FINISH UNLESS NOTED.", "DO NOT SCALE THE DRAWINGS. USE FIGURED DIMENSIONS.", "DOOR AND WINDOW TAGS REFER TO SHEETS A6.1 AND A6.2.", "ROOM NAMES AND NUMBERS REFER TO SHEET A6.3.",
         "EVERY STORY IS SPRINKLERED TO NFPA 13.")


class Canvas:
    """One page. Strokes of one style go in one path, because each path is a drawing object to whoever opens the file."""

    def __init__(self, doc, oc):
        self.page = doc.new_page(width=W, height=H)
        self.shape = self.page.new_shape()
        self.oc = oc
        self.anchors = {}

    def lines(self, segments, layer, width, color=BLACK, dashes=None):
        any_line = False
        for a, b in segments:
            self.shape.draw_line(tuple(a), tuple(b))
            any_line = True
        if any_line:
            self.shape.finish(width=width, color=color, closePath=False, dashes=dashes, oc=self.oc[layer])

    def ring(self, points, layer, width, fill=None, color=BLACK, dashes=None):
        self.shape.draw_polyline([tuple(p) for p in points])
        self.shape.finish(width=width, color=color, fill=fill, closePath=True, dashes=dashes, oc=self.oc[layer])

    def box(self, x0, y0, x1, y1, layer, width, fill=None, color=BLACK, radius=None):
        self.shape.draw_rect(pymupdf.Rect(x0, y0, x1, y1), radius=radius)
        self.shape.finish(width=width, color=color, fill=fill, oc=self.oc[layer])

    def circle(self, center, r, layer, width, fill=None):
        self.shape.draw_circle(tuple(center), r)
        self.shape.finish(width=width, color=BLACK, fill=fill, oc=self.oc[layer])

    def oval(self, x0, y0, x1, y1, layer, width):
        self.shape.draw_oval(pymupdf.Rect(x0, y0, x1, y1))
        self.shape.finish(width=width, color=BLACK, oc=self.oc[layer])

    def text(self, x, y, s, size, *, font=FONT, align=0, rotate=0, color=BLACK, layer="A-ANNO-TEXT"):
        """Left, centre or right aligned on x by align 0, 0.5 or 1, with y as the baseline."""
        width = pymupdf.get_text_length(s, fontname=font, fontsize=size)
        at = (x, y + align * width) if rotate == 90 else (x - align * width, y)
        self.shape.insert_text(at, s, fontsize=size, fontname=font, rotate=rotate, color=color, oc=self.oc[layer])
        return [at[0], y - size, at[0] + width, y + 2]

    def done(self):
        self.shape.commit()


def title_block(c, m, sheet, n, scale):
    """The right-hand strip, with the same parts on every sheet."""
    b = "G-ANNO-TTLB"
    c.box(MARGIN, MARGIN, W - MARGIN, H - MARGIN, b, 1.6)
    c.lines([((STRIP, MARGIN), (STRIP, H - MARGIN))], b, 1.6)
    x, right = STRIP + 9, W - MARGIN - 9
    c.text(x, 76, "TEST DRAWING OFFICE", 15, font=BOLD, layer=b)
    c.text(x, 96, "ARCHITECTURE AND PLANNING", 9, layer=b)
    c.lines([((STRIP, 112), (W - MARGIN, 112))], b, 0.7)
    c.text(x, 130, "PROJECT", 7.5, layer=b)
    c.text(x, 150, m.name.upper(), 12, font=BOLD, layer=b)
    c.text(x, 168, m.address, 8.5, layer=b)
    c.lines([((STRIP, 182), (W - MARGIN, 182))], b, 0.7)
    c.text(x, 204, GENERATED, 8.5, font=BOLD, color=RED, layer=b)
    c.lines([((STRIP, 216), (W - MARGIN, 216))], b, 0.7)
    c.text(x, 234, "REVISIONS", 7.5, layer=b)
    c.text(x, 250, "NO.", 7.5, layer=b)
    c.text(x + 34, 250, "DATE", 7.5, layer=b)
    c.text(x + 84, 250, "DESCRIPTION", 7.5, layer=b)
    c.lines([((STRIP, 256), (W - MARGIN, 256))], b, 0.35)
    if sheet.number in {cloud["sheet"] for cloud in m.clouds} and m.revision:
        delta(c, x + 8, 269, m.revision["number"], b)
        c.text(x + 34, 272, m.revision["date"], 7.5, layer=b)
        c.text(x + 84, 272, m.revision["text"], 7.5, layer=b)
        c.anchors["revision"] = [STRIP, 258, W - MARGIN, 280]
    bottom = H - MARGIN
    c.lines([((STRIP, bottom - 330), (W - MARGIN, bottom - 330))], b, 0.7)
    c.text(x, bottom - 312, "SHEET TITLE", 7.5, layer=b)
    lines = _wrap(sheet.block_title or sheet.title, 16, right - x, BOLD)
    for k, line in enumerate(lines):
        c.text(x, bottom - 280 + k * 20, line, 16, font=BOLD, layer=b)
    c.anchors["title"] = [x, bottom - 298, right, bottom - 276 + 20 * (len(lines) - 1)]
    c.lines([((STRIP, bottom - 196), (W - MARGIN, bottom - 196))], b, 0.7)
    for k, (key, value) in enumerate((("DATE", m.date), ("SCALE", scale), ("DRAWN", "TDO"), ("PROJECT NO.", f"T-{m.building}"))):
        y = bottom - 176 + k * 19
        c.text(x, y, key, 7.5, layer=b)
        c.text(x + 66, y, value, 10, layer=b)
    c.lines([((STRIP, bottom - 104), (W - MARGIN, bottom - 104))], b, 0.7)
    c.text(x, bottom - 86, "SHEET", 7.5, layer=b)
    c.text(x, bottom - 40, sheet.number, 36, font=BOLD, layer=b)
    c.text(right, bottom - 14, f"{n} OF {len([s for s in m.sheets if s.in_set])}", 9, align=1, layer=b)


def _wrap(s, size, width, font):
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if pymupdf.get_text_length(trial, fontname=font, fontsize=size) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def delta(c, x, y, number, layer):
    c.ring([(x, y - 8), (x + 8, y + 6), (x - 8, y + 6)], layer, 0.8)
    c.text(x, y + 4, str(number), 7.5, align=0.5, layer=layer)


def view_title(c, x, y, number, title, scale):
    c.circle((x + 20, y), 20, "A-ANNO-TEXT", 1.2)
    c.lines([((x + 2, y), (x + 38, y))], "A-ANNO-TEXT", 0.6)
    c.text(x + 20, y - 4, str(number), 13, align=0.5)
    box = c.text(x + 52, y + 2, title, 20, font=BOLD)
    c.lines([((x + 52, y + 7), (x + 520, y + 7))], "A-ANNO-TEXT", 1.0)
    c.text(x + 52, y + 24, f"SCALE: {scale}", 12)
    c.anchors["label"] = box


def bar(c, x, y, pt_per_ft):
    """A graphic scale, 0 to 16 ft."""
    step = pt_per_ft * 4
    for k in range(4):
        c.box(x + k * step, y, x + (k + 1) * step, y + 6, "A-ANNO-TEXT", 0.5, fill=BLACK if k % 2 == 0 else None)
        c.text(x + k * step, y - 4, f"{k * 4}'", 7, align=0.5)
    c.text(x + 4 * step, y - 4, "16'", 7, align=0.5)


def north(c, x, y):
    c.circle((x, y), 22, "A-ANNO-TEXT", 0.8)
    c.ring([(x, y - 30), (x + 9, y + 12), (x, y + 6), (x - 9, y + 12)], "A-ANNO-TEXT", 0.5, fill=BLACK)
    c.text(x, y - 34, "N", 11, font=BOLD, align=0.5)


class Plan:
    """Plan metres to sheet points."""

    def __init__(self, extent, k, left, top, width, height):
        self.k = k
        x0, y0, x1, y1 = extent
        self.ox = left + (width - (x1 - x0) * k) / 2 - x0 * k
        self.oy = top + (height + (y1 - y0) * k) / 2 + y0 * k

    def __call__(self, p):
        return self.ox + float(p[0]) * self.k, self.oy - float(p[1]) * self.k

    def ring(self, poly):
        return [self(p) for p in poly.exterior.coords[:-1]]

    def rect(self, g):
        x0, y0, x1, y1 = g.bounds
        (a, b), (d, e) = self((x0, y1)), self((x1, y0))
        return a, b, d, e


def walls(c, polys, t, width):
    for p in polys:
        c.ring(t.ring(p), "A-WALL", width)
        for hole in p.interiors:
            c.ring([t(q) for q in hole.coords[:-1]], "A-WALL", width)


def door_symbol(c, doors, t):
    for d in doors:
        for leaf in d.leaves:
            hinge, shut, tip = (t(p) for p in leaf)
            h, s, e = np.array(hinge), np.array(shut), np.array(tip)
            c.shape.draw_line(hinge, tip)
            c.shape.draw_bezier(shut, tuple(s + KAPPA * (e - h)), tuple(e + KAPPA * (s - h)), tip)
    if doors:
        c.shape.finish(width=0.6, color=BLACK, closePath=False, oc=c.oc["A-DOOR"])


def lift_door(c, f, t):
    c.lines([(t(a), t(b)) for a, b in f.lift_doors], "A-DOOR", 1.2)


def window_symbol(c, windows, t):
    for w in windows:
        n = np.array([-w.u[1], w.u[0]])
        segments = []
        for k in (-0.5, -0.12, 0.12, 0.5):
            mid = w.center + n * k * w.depth
            segments.append((t(mid - w.u * w.width / 2), t(mid + w.u * w.width / 2)))
        c.lines(segments, "A-GLAZ", 0.45)


def stair(c, f, t, top_floor):
    """Treads across each flight, with an arrow and UP or DN, and a break line where the flight is cut. A floor with no flights has no stair drawn."""
    runs = f.flights
    if not runs and not f.landings:
        return
    for g in f.landings:
        c.ring(t.ring(g), "A-FLOR-STRS", 0.9)
    for fl in runs:
        c.ring(t.ring(fl.poly), "A-FLOR-STRS", 0.9)
        v = fl.end - fl.start
        n = _unit(np.array([-v[1], v[0]])) * _half_width(fl)
        treads = max(int(float(np.hypot(*v)) / TREAD_M), 3)
        c.lines([(t(fl.start + v * i / treads + n), t(fl.start + v * i / treads - n)) for i in range(1, treads)], "A-FLOR-STRS", 0.8)
    label = "UP" if f.number == 1 else "DN" if top_floor else None
    both = len(runs) == 1 and label is None   # one run on a typical floor carries both ways, so it takes two arrows side by side
    for k, fl in enumerate(runs):
        word = label or ("UP" if k == 0 else "DN")
        a, b = (fl.start, fl.end) if word == "UP" else (fl.end, fl.start)
        v = b - a
        n = _unit(np.array([-v[1], v[0]])) * _half_width(fl) / 2
        for w, tail, way in (((word, a + n, v), ("DN", b - n, -v)) if both else ((word, a, v),)):
            arrow(c, t(tail + way * 0.08), t(tail + way * 0.92))
            _knocked_out(c, t(tail + way * 0.72), w)
        if k == 0 and not f.landings:
            _knocked_out(c, t(a + v * (0.4 if both and abs(v[1]) > abs(v[0]) else 0.28)), "STAIR 1")   # across an upright run the words share a row, so it goes between them, clear of the break line
    if f.landings:
        x0, y0, x1, y1 = t.rect(f.landings[0])
        _knocked_out(c, ((x0 + x1) / 2, (y0 + y1) / 2), "STAIR 1")
    if both:
        fl = runs[0]
        mid = (fl.start + fl.end) / 2
        u = _unit(fl.end - fl.start)
        n = np.array([-u[1], u[0]]) * _half_width(fl)
        c.lines([(t(mid + n + u * 0.12), t(mid - n - u * 0.12))], "A-FLOR-STRS", 1.0)


def _unit(v):
    return v / float(np.hypot(*v))


def _knocked_out(c, at, word):
    """A word on the treads, with the treads cleared behind it."""
    width = pymupdf.get_text_length(word, fontname=BOLD, fontsize=10)
    c.box(at[0] - width / 2 - 3, at[1] - 8, at[0] + width / 2 + 3, at[1] + 6, "A-FLOR-STRS", 0, fill=(1, 1, 1), color=None)
    c.text(at[0], at[1] + 3, word, 10, font=BOLD, align=0.5, layer="A-FLOR-STRS")


def _half_width(fl):
    return _rect(fl.poly)[3] / 2


def arrow(c, a, b):
    a, b = np.array(a), np.array(b)
    v = (b - a) / float(np.hypot(*(b - a)))
    n = np.array([-v[1], v[0]])
    c.lines([(tuple(a), tuple(b))], "A-FLOR-STRS", 0.6)
    c.ring([tuple(b), tuple(b - 12 * v + 4 * n), tuple(b - 12 * v - 4 * n)], "A-FLOR-STRS", 0.4, fill=BLACK)


def fixtures(c, f, t):
    for kind, g in f.fixtures:
        x0, y0, x1, y1 = t.rect(g)
        c.ring(t.ring(g), "A-FLOR-FIXT", 0.4)
        pad = min(x1 - x0, y1 - y0) * 0.14
        if kind == "BATHTUB":
            c.box(x0 + pad, y0 + pad, x1 - pad, y1 - pad, "A-FLOR-FIXT", 0.3, radius=0.2)
        elif kind in ("SINK", "TOILET"):
            c.oval(x0 + pad, y0 + pad, x1 - pad, y1 - pad, "A-FLOR-FIXT", 0.3)
        elif kind == "SHOWER":
            c.lines([((x0, y0), (x1, y1)), ((x1, y0), (x0, y1))], "A-FLOR-FIXT", 0.25)
    for g in f.lift:
        x0, y0, x1, y1 = t.rect(g)
        c.ring(t.ring(g), "A-FLOR-FIXT", 0.5)
        c.lines([((x0, y0), (x1, y1)), ((x1, y0), (x0, y1))], "A-FLOR-FIXT", 0.3)
    for g in f.shafts:
        x0, y0, x1, y1 = t.rect(g)
        if x1 - x0 > 8 and y1 - y0 > 8:
            c.lines([((x0, y0), (x1, y1))], "A-FLOR-FIXT", 0.25)


def outline(c, f, t):
    for g in f.outdoors:
        for p in getattr(g, "geoms", [g]):
            c.ring(t.ring(p), "A-FLOR-OTLN", 0.35, color=GREY)
    for g in f.rails:
        for p in getattr(g, "geoms", [g]):
            c.ring(t.ring(p), "A-FLOR-OTLN", 0.35)


def door_tag(c, door, t):
    x, y = t(door.tag)
    c.circle((x, y), 11, "A-ANNO-SYMB", 0.6, fill=(1, 1, 1))
    c.text(x, y + 2.8, door.mark, 8, align=0.5, layer="A-ANNO-SYMB")


def window_tag(c, window, t):
    x, y = t(window.tag)
    r = 10
    c.ring([(x + r * math.cos(a), y + r * math.sin(a)) for a in (math.pi * k / 3 + math.pi / 6 for k in range(6))], "A-ANNO-SYMB", 0.6, fill=(1, 1, 1))
    c.text(x, y + 2.8, window.mark, 8, align=0.5, layer="A-ANNO-SYMB")


def room_label(c, room, t):
    x, y = t(room.label)
    box = c.text(x, y - 2, room.name, 10.5, font=BOLD, align=0.5)
    c.text(x, y + 9, room.number, 9, align=0.5)
    return [*box[:3], y + 11]


def dim_strings(c, m, f, t):
    """Each stop sits on a whole inch, so the printed length is the drawn one. Strings are drawn as separate lines, as dimension objects explode."""
    x0, y0, x1, y1 = m.extent
    k = t.k
    base = {"bottom": y0 - 56 / k, "top": y1 + 56 / k, "left": x0 - 56 / k, "right": x1 + 56 / k}
    extra = 34 / k
    for side, depth, a, b in f.dims:
        out = {"bottom": -1, "top": 1, "left": -1, "right": 1}[side] * depth * extra
        at = base[side] + out
        along_x = side in ("bottom", "top")
        p, q = ((a, at), (b, at)) if along_x else ((at, a), (at, b))
        _string(c, t(p), t(q), along_x, (b - a) * IN_PER_M)
        edge = _edge(f, side)
        ends = (((a, edge), p), ((b, edge), q)) if along_x else (((edge, a), p), ((edge, b), q))
        c.lines([(t(s), t(e)) for s, e in ends], "A-ANNO-DIMS", 0.2)


def _edge(f, side):
    x0, y0, x1, y1 = f.outline.bounds
    return {"bottom": y0, "top": y1, "left": x0, "right": x1}[side]


def _string(c, p, q, along_x, inches, at=None):
    """A dimension line with its ticks, and its length above it or left of it, at `at` along it or at its middle."""
    x, y = at or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    c.lines([(p, q)], "A-ANNO-DIMS", 0.35)
    c.lines([((a - 3.5, b + 3.5), (a + 3.5, b - 3.5)) for a, b in (p, q)], "A-ANNO-DIMS", 0.8)
    label = ft_in(inches)
    if along_x:
        c.text(x, y - 4, label, 9, align=0.5, layer="A-ANNO-DIMS")
    else:
        c.text(x - 4, y, label, 9, align=0.5, rotate=90, layer="A-ANNO-DIMS")


def hall_dimension(c, f, t):
    if not f.hall_width:
        return
    a, b = f.hall_width
    p, q = t(a), t(b)
    along_x = abs(p[1] - q[1]) < abs(p[0] - q[0])
    start, end = (q, p) if along_x and p[0] > q[0] or not along_x and p[1] > q[1] else (p, q)
    inches = float(np.hypot(*(np.array(a) - np.array(b)))) * IN_PER_M
    _string(c, start, end, along_x, inches, t(f.hall_text))
    c.anchors["hall"] = [min(p[0], q[0]) - 30, min(p[1], q[1]) - 30, max(p[0], q[0]) + 30, max(p[1], q[1]) + 30]


def plan_frame(m):
    x0, y0, x1, y1 = FIELD
    return Plan(m.extent, m.pt_per_m, x0 + 20, y0 + 10, x1 - x0 - 40, y1 - y0 - 130)


def life_frame(m):
    x0, y0, x1, y1 = FIELD
    return Plan(m.extent, m.ls_pt_per_m, x0 + 10, y0 + 20, 1480, y1 - y0 - 140)


def floor_plan(c, m, sheet):
    f = m.floors[sheet.floor - 1]
    t = plan_frame(m)
    outline(c, f, t)
    fixtures(c, f, t)
    walls(c, [*f.walls, *f.plan_walls], t, 1.1)
    c.lines([(t(a), t(b)) for line in f.hatch for a, b in zip(line, line[1:])], "A-WALL-PATT", 0.12)
    door_symbol(c, f.doors, t)
    lift_door(c, f, t)
    window_symbol(c, f.windows, t)
    if f.stair_on_plan:
        stair(c, f, t, f.number == len(m.floors))
    c.anchors["stair"] = list(t.rect(_stairs(f)))
    for room in f.rooms:
        c.anchors[f"room:{room.number}"] = room_label(c, room, t)
    c.anchors[f"room:{f.hall.number}"] = room_label(c, f.hall, t)
    for d in f.doors:
        if d.tag:
            door_tag(c, d, t)
            x, y = t(d.tag)
            c.anchors[f"tag:{d.mark}"] = [x - 12, y - 12, x + 12, y + 12]
    for w in f.windows:
        window_tag(c, w, t)
    hall_dimension(c, f, t)
    dim_strings(c, m, f, t)
    view_title(c, 80, FIELD[3] - 44, 1, sheet.title, m.scale[1])
    bar(c, 700, FIELD[3] - 34, t.k * 0.3048)
    north(c, 1100, FIELD[3] - 40)
    notes(c, 1220, FIELD[3] - 70)


def _stairs(f):
    return unary_union([fl.poly for fl in f.flights] + f.landings)


def notes(c, x, y):
    c.text(x, y, "GENERAL NOTES", 10, font=BOLD)
    for k, line in enumerate(NOTES):
        c.text(x, y + 16 + k * 13, f"{k + 1}. {line}", 8.5)


def life_safety(c, m, sheet):
    f = m.floors[sheet.floor - 1]
    y1 = FIELD[3]
    t = life_frame(m)
    outline(c, f, t)
    walls(c, f.walls, t, 0.9)
    door_symbol(c, f.doors, t)
    lift_door(c, f, t)
    window_symbol(c, f.windows, t)
    stair(c, f, t, f.number == len(m.floors))
    c.ring(t.ring(f.hall.poly), "A-LIFE-SAFE", 2.0, dashes="[10 4] 0")
    taken = [list(t.rect(_stairs(f))), c.text(*t(f.hall.label), "HALL", 10, font=BOLD, align=0.5)]
    for unit, room in {r.unit: r for r in sorted(f.rooms, key=lambda r: r.poly.area) if r.unit}.items():
        x, y = t(room.label)
        taken.append(c.text(x, y, f"UNIT {unit}", 11, font=BOLD, align=0.5))
    for k, marker in enumerate(m.markers[f.number]):
        exit_marker(c, marker, t, k, taken)
    view_title(c, 80, y1 - 44, 1, sheet.title, m.ls_scale[1])
    bar(c, 700, y1 - 34, t.k * 0.3048)
    north(c, 1100, y1 - 40)
    side = 1590
    row = next(r for r in m.load_rows if r["no"] == str(f.number))
    c.text(side, 110, "OCCUPANT LOAD THIS FLOOR", 16, font=BOLD)
    for k, (key, value) in enumerate((("GROSS FLOOR AREA", f"{row['area']:,} SF"), ("LOAD FACTOR", f"{row['factor']} SF PER OCCUPANT"), ("OCCUPANT LOAD", f"{row['load']} PEOPLE"), ("BUILDING OCCUPANT LOAD", f"{m.load_total['load']} PEOPLE"))):
        c.text(side, 140 + k * 22, key, 11)
        c.text(side + 240, 140 + k * 22, value, 11, font=BOLD)
    c.text(side, 260, "EXIT CAPACITY FACTORS", 16, font=BOLD)
    c.text(side, 288, "STAIRS: 0.3 IN PER OCCUPANT", 11)
    c.text(side, 308, "DOORS AND LEVEL COMPONENTS: 0.2 IN PER OCCUPANT", 11)
    c.text(side, 360, "LEGEND", 16, font=BOLD)
    c.box(side, 372, side + 44, 394, "A-LIFE-SAFE", 1.0, fill=(1, 1, 1))
    c.lines([((side + 44, 394), (side + 58, 406))], "A-LIFE-SAFE", 0.8)
    c.circle((side + 58, 406), 4, "A-LIFE-SAFE", 0.6, fill=BLACK)
    c.text(side + 76, 394, "EXIT LOCATION, WITH CLEAR WIDTH AND CAPACITY", 11)
    c.lines([((side, 430), (side + 60, 430))], "A-LIFE-SAFE", 2.0, dashes="[10 4] 0")
    c.text(side + 76, 434, "EXIT ENCLOSURE", 11)


def exit_marker(c, marker, t, k, taken):
    """A box with a leader to the exit, in the first spot round the exit that covers nothing in `taken`."""
    x, y = t(marker["point"])
    w, h = 136, 48
    spots = [(x + side * reach - (side < 0) * w, y + down) for reach in (34, 90, 150) for side, down in ((1, -74), (-1, -74), (1, 26), (-1, 26))]
    bx, by = next((s for s in spots if not any(s[0] < b[2] and b[0] < s[0] + w and s[1] < b[3] and b[1] < s[1] + h for b in taken)), spots[0])
    c.box(bx, by, bx + w, by + h, "A-LIFE-SAFE", 1.0, fill=(1, 1, 1))
    c.lines([((min(bx, bx + w, key=lambda v: abs(v - x)), min(by, by + h, key=lambda v: abs(v - y))), (x, y))], "A-LIFE-SAFE", 0.8)
    c.circle((x, y), 4, "A-LIFE-SAFE", 0.6, fill=BLACK)
    c.text(bx + 8, by + 14, marker["name"], 10.5, font=BOLD, layer="A-LIFE-SAFE")
    c.text(bx + 8, by + 28, f'{marker["width"]}" CLEAR WIDTH', 9.5, layer="A-LIFE-SAFE")
    c.text(bx + 8, by + 41, f'{marker["capacity"]} PEOPLE CAPACITY', 9.5, layer="A-LIFE-SAFE")
    c.anchors[f"marker:{k}"] = [bx, by, bx + w, by + h]


def table(c, x, y, title, cols, rows, *, title_size=14, size=9.5, pitch=17, keyed=None):
    """A ruled table. `cols` are (header, width, key). Returns the bottom edge. `keyed` names rows for anchors: a function from row to anchor name."""
    c.text(x, y, title, title_size, font=BOLD)
    width = sum(col[1] for col in cols)
    top = y + 8
    head = top + pitch + 3
    c.box(x, top, x + width, head, "A-ANNO-TEXT", 0.7)
    edges = [x]
    for _, w, _ in cols:
        edges.append(edges[-1] + w)
    centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    for (header, _, _), cx in zip(cols, centers):
        c.text(cx, head - 5.5, header, size, font=BOLD, align=0.5)
    bottom = head
    for row in rows:
        lower = bottom + pitch
        c.lines([((x, lower), (x + width, lower))], "A-ANNO-TEXT", 0.3)
        for (_, _, key), cx in zip(cols, centers):
            value = key(row) if callable(key) else row[key]
            if value not in ("", None):
                c.text(cx, lower - 5, str(value), size, align=0.5)
        if keyed:
            c.anchors[keyed(row)] = [x, bottom, x + width, lower]
        bottom = lower
    c.lines([((e, top), (e, bottom)) for e in edges], "A-ANNO-TEXT", 0.4)
    return bottom


DOOR_COLS = (("MARK", 62, "mark"), ("SIZE", 96, "size"), ("SWING", 62, "swing"), ("TYPE", 46, "type"), ("MATERIAL", 112, "material"), ("FRAME", 56, "frame"), ("HARDWARE", 92, "hardware"), ("RATING", 62, "rating"))
FINISH_COLS = (("NO.", 54, "no"), ("ROOM NAME", 150, "name"), ("FLOOR", 108, lambda r: r["finish"][0]), ("BASE", 66, lambda r: r["finish"][1]), ("WALLS", 66, lambda r: r["finish"][2]), ("CEILING", 66, lambda r: r["finish"][3]), ("AREA SF", 76, "sf"))
# shortcut: declared.schedules takes a table's columns from anything within 720 pt right of its title, so tables stand 760 pt apart. At 40 pt apart it reads 2 of 5 door and 2 of 5 finish
# schedules with no Cannot tell row: the clean findings of all 7 buildings do not change, and on 710 and 3370 tag_no_row, door_type and room_name go unraised. Close them up once that
# reader stops at the table's own width.
COLUMN_X = 70, 830, 1590


def per_floor(c, m, rows_by_floor, cols, word, key):
    rows = max(len(r) for r in rows_by_floor.values())
    slots = len(rows_by_floor)
    tiers = 1 if slots <= 3 else 2
    span = (FIELD[3] - FIELD[1] - 40) / tiers
    pitch = min(17.0, (span - 90) / max(rows, 1))
    for k, f in enumerate(m.floors):
        x = COLUMN_X[k % 3]
        y = FIELD[1] + 30 + (k // 3) * span
        table(c, x, y, f"{f.name} {word} SCHEDULE", cols, rows_by_floor[f.number], pitch=pitch, keyed=lambda r: f"{key}:{r[cols[0][2]]}")


def door_sheet(c, m, sheet):
    per_floor(c, m, m.door_rows, DOOR_COLS, "DOOR", "row")


def finish_sheet(c, m, sheet):
    per_floor(c, m, m.finish_rows, FINISH_COLS, "ROOM FINISH", "finish")


def window_sheet(c, m, sheet):
    cols = (("MARK", 62, "mark"), ("QTY", 56, "qty"), ("WIDTH", 80, "width"), ("HEIGHT", 80, "height"), ("SILL", 72, "sill"), ("OPERATION", 110, lambda r: "CASEMENT"), ("GLAZING", 110, lambda r: "INSULATED"), ("FRAME", 70, lambda r: "ALUM"))
    table(c, 70, 100, "WINDOW SCHEDULE", cols, m.window_rows, pitch=20, size=11, keyed=lambda r: f"window:{r['mark']}")
    c.text(860, 100, "WINDOW TYPES", 14, font=BOLD)
    x, y, shelf = 860, 130, 0
    k = 72 / 96 * 2   # types are drawn at 1/4 in per foot of their inches, with room for their marks
    for row in m.window_rows:
        w, h = _inches(row["width"]) * k * 0.5, _inches(row["height"]) * k * 0.5
        if x + w + 40 > FIELD[2]:
            x, y, shelf = 860, y + shelf + 70, 0
        c.box(x, y, x + w, y + h, "A-ANNO-TEXT", 0.9)
        c.lines([((x + w / 2, y), (x + w / 2, y + h))], "A-ANNO-TEXT", 0.4)
        c.text(x + w / 2, y + h + 18, f"TYPE {row['mark']}", 10, font=BOLD, align=0.5)
        c.text(x + w / 2, y + h + 32, f"{row['width']} x {row['height']}", 8.5, align=0.5)
        x += max(w, 76) + 34
        shelf = max(shelf, h)


def _inches(text):
    feet, inches = text.rstrip('"').split("'-")
    return int(feet) * 12 + int(inches)


def cover(c, m, sheet):
    c.text(70, 130, m.name.upper(), 44, font=BOLD)
    c.text(70, 166, f"{len(m.floors)} STORY APARTMENT BUILDING, {m.address.upper()}", 16)
    code = m.code
    c.text(70, 240, "CODE SUMMARY", 18, font=BOLD)
    rows = (("CODE EDITION", code["edition"]), ("OCCUPANCY", code["occupancy"]), ("CONSTRUCTION TYPE", code["construction"]), ("SPRINKLERS", code["sprinklers"]),
            ("STORIES", code["stories"]), ("BUILDING AREA", code["area"]), ("OCCUPANT LOAD", code["load"]))
    for k, (key, value) in enumerate(rows):
        y = 276 + k * 30
        c.text(70, y, f"{key}:", 16, font=BOLD)
        box = c.text(290, y, value, 16)
        c.anchors[f"code:{key}"] = [70, y - 16, box[2], y + 4]
    c.text(70, 520, "PROJECT DATA", 18, font=BOLD)
    for k, (key, value) in enumerate((("OWNER", "TEST OWNER LLC"), ("ARCHITECT", "TEST DRAWING OFFICE"), ("UNITS", str(len({r.unit for f in m.floors for r in f.rooms} - {""}))), ("BUILDING ID", f"SWISS DWELLINGS {m.building}"))):
        c.text(70, 552 + k * 24, f"{key}:", 12, font=BOLD)
        c.text(200, 552 + k * 24, value, 12)
    c.text(70, 700, CREDIT, 13, font=BOLD)
    c.text(70, 720, GENERATED, 13, font=BOLD, color=RED)
    x = 640
    c.text(x, 240, "SHEET INDEX", 18, font=BOLD)
    y = 276
    for s in m.sheets:
        if s.on_index:
            c.text(x, y, s.number, 12, font=BOLD)
            end = c.text(x + 80, y, s.title, 12)
            c.anchors[f"index:{s.number}"] = [x, y - 12, end[2], y + 4]
            y += 24
    x = 1220
    c.text(x, 240, "OCCUPANT LOAD CALCULATION", 16, font=BOLD)
    bottom = table(c, x, 256, "", (("NO.", 40, "no"), ("LEVEL", 150, "name"), ("GROSS AREA", 100, lambda r: f"{r['area']:,} SF"), ("SF/OCC", 50, "factor"), ("LOAD", 40, "load")), m.load_rows, size=11, pitch=22, keyed=lambda r: f"load:{r['no']}")
    c.text(x + 4, bottom + 22, "TOTAL", 11, font=BOLD)   # far enough under the table that a cloud round this row keeps off it
    c.text(x + 290, bottom + 22, f"{m.load_total['area']:,} SF", 11, font=BOLD, align=0.5)
    c.text(x + 360, bottom + 22, str(m.load_total["load"]), 11, font=BOLD, align=0.5)
    c.anchors["load_total"] = [x, bottom + 11, x + 380, bottom + 27]
    y = bottom + 60
    c.text(x, y, "EXIT WIDTH CALCULATION", 16, font=BOLD)
    bottom = table(c, x, y + 6, "", (("EXIT", 70, "name"), ("DOOR IN", 66, "nominal"), ("CLEAR IN", 66, "clear"), ("OCC", 50, "load"), ("REQD IN", 66, "required"), ("", 40, lambda r: "OK")), m.exit_rows, size=11, pitch=22)
    y = bottom + 50
    c.text(x, y, "STAIR AND CORRIDOR WIDTH CALCULATION", 16, font=BOLD)
    table(c, x, y + 6, "", (("SPACE", 110, "name"), ("OCC", 60, "load"), ("REQD IN", 70, "required"), ("WIDTH AT DIM", 100, lambda r: f'{r["width"]}"'), ("", 40, lambda r: "OK")), m.corridor_rows, size=11, pitch=22)
    x = 1760
    c.text(x, 240, "SYMBOLS", 18, font=BOLD)
    c.circle((x + 14, 272), 11, "A-ANNO-SYMB", 0.6)
    c.text(x + 14, 275, "1001", 7, align=0.5, layer="A-ANNO-SYMB")
    c.text(x + 44, 276, "DOOR TAG, SEE A6.1", 12)
    c.text(x + 14, 310, "A", 9, align=0.5, layer="A-ANNO-SYMB")
    c.ring([(x + 14 + 10 * math.cos(a), 306 + 10 * math.sin(a)) for a in (math.pi * k / 3 + math.pi / 6 for k in range(6))], "A-ANNO-SYMB", 0.6)
    c.text(x + 44, 310, "WINDOW TAG, SEE A6.2", 12)
    c.text(x, 380, "GENERAL NOTES", 18, font=BOLD)
    for k, line in enumerate(NOTES):
        c.text(x, 408 + k * 18, f"{k + 1}. {line}", 9.5)
    key_plan(c, m)
    section_diagram(c, m)
    abbreviations(c)


def key_plan(c, m):
    f = m.floors[min(1, len(m.floors) - 1)]
    x0, y0, x1, y1 = m.extent
    k = min(620 / (x1 - x0), 560 / (y1 - y0))
    t = Plan(m.extent, k, 70, 800, 620, 620)
    c.text(70, 790, "KEY PLAN, TYPICAL FLOOR", 18, font=BOLD)
    for p in f.walls:
        c.ring(t.ring(p), "A-WALL", 0.5, fill=(0.35, 0.35, 0.35))
    c.ring(t.ring(f.hall.poly), "A-ANNO-TEXT", 0.4, fill=(0.88, 0.88, 0.88))
    door_symbol(c, f.doors, t)
    window_symbol(c, f.windows, t)


def section_diagram(c, m):
    x, base = 820, 1500
    c.text(x, 790, "BUILDING SECTION DIAGRAM", 18, font=BOLD)
    width, story = 520, min(100, 600 / len(m.floors))
    for k, f in enumerate(m.floors):
        top = base - (k + 1) * story
        c.box(x, top, x + width, top + story, "A-ANNO-TEXT", 1.0)
        c.text(x + 16, top + story / 2 + 4, f.name, 12, font=BOLD)
        c.text(x + width - 16, top + story / 2 + 4, f"{f.gross_sf:,} SF", 12, align=1)
        c.text(x + width + 14, top + story - 6, f"EL. {ft_in(f.z * IN_PER_M)}", 10)
    c.lines([((x - 50, base), (x + width + 110, base))], "A-ANNO-TEXT", 2.0)
    c.text(x - 50, base + 18, "GRADE", 10)


ABBREVIATIONS = ("ALUM  ALUMINUM", "CLR  CLEAR", "CONC  CONCRETE", "DIM  DIMENSION", "DN  DOWN", "EL  ELEVATION", "ELEV  ELEVATOR", "FEC  FIRE EXTINGUISHER CABINET", "GWB  GYPSUM WALL BOARD", "HM  HOLLOW METAL", "LVT  LUXURY VINYL TILE",
                 "MIN  MINUTE", "OCC  OCCUPANTS", "REQD  REQUIRED", "SF  SQUARE FEET", "TYP  TYPICAL", "UNO  UNLESS NOTED OTHERWISE")


def abbreviations(c):
    x = 1560
    c.text(x, 790, "ABBREVIATIONS", 18, font=BOLD)
    for k, line in enumerate(ABBREVIATIONS):
        c.text(x, 822 + k * 21, line, 12)


SHEETS = {"cover": cover, "plan": floor_plan, "life": life_safety, "door": door_sheet, "window": window_sheet, "finish": finish_sheet}


def draw_set(m, path):
    """Writes the set and returns {sheet number: anchors}, the boxes truth points at, in sheet points."""
    doc = pymupdf.open()
    oc = {name: doc.add_ocg(name, on=True) for name in LAYERS}
    anchors = {}
    present = [s for s in m.sheets if s.in_set]
    for n, sheet in enumerate(present, 1):
        c = Canvas(doc, oc)
        SHEETS[sheet.kind](c, m, sheet)
        scale = {"plan": m.scale[1], "life": m.ls_scale[1]}.get(sheet.kind, "AS NOTED")
        title_block(c, m, sheet, n, scale)
        for cloud in m.clouds:
            if cloud["sheet"] == sheet.number:
                revision_cloud(c, cloud, m.revision["number"])
        c.done()
        anchors[sheet.number] = c.anchors
    doc.set_metadata({"title": f"{m.name} ({m.variant})", "creator": "Storey eval generator", "subject": GENERATED})
    doc.save(path, garbage=3, deflate=True, no_new_id=True)   # a random id would change the file's hash on every run
    return anchors


def revision_cloud(c, cloud, number):
    """Scallops round the box, and a delta at its corner, or beside it on a row of text, where the corner holds the row above, or over it in the title strip, where nothing lies beside it."""
    x0, y0, x1, y1 = cloud["box"]
    step = cloud["scallop"]
    for (ax, ay), (bx, by) in zip(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), ((x1, y0), (x1, y1), (x0, y1), (x0, y0))):
        length = math.hypot(bx - ax, by - ay)
        n = max(int(length / step), 2)
        out = np.array([by - ay, ax - bx]) / length * step * 0.52   # the side's outward normal, as the box runs clockwise on the page
        for k in range(n):
            a, b = (np.array([ax + (bx - ax) * j / n, ay + (by - ay) * j / n]) for j in (k, k + 1))
            mid = (a + b) / 2
            c.shape.draw_bezier(tuple(a), tuple(a * 0.6 + mid * 0.4 + out), tuple(b * 0.6 + mid * 0.4 + out), tuple(b))
    c.shape.finish(width=1.1, color=BLACK, closePath=False, oc=c.oc["A-ANNO-SYMB"])
    delta(c, *((x1 - 12, y0 - 10) if x1 > STRIP else (x1 + 16, (y0 + y1) / 2) if step < 20 else (x1 + 8, y0 - 8)), number, "A-ANNO-SYMB")
