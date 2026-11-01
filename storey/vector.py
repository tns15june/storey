"""Scale, door arcs, walls, routes and corridors from the vector content of plan sheets. This stage makes no model call.

An upload is hostile until proven otherwise, and so is every number parsed from it, so the whole stage body runs in a child process that can be killed,
as ingest does. The child reports through one file per sheet. Geometry runs in rendered page points, because rotated pages store their strokes and text
in another frame. Stage results are in sheet pixels, apart from scale_in_per_pt.
"""
import json
import math
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

from .ingest import CPU_SECONDS, MEMORY_BYTES

ROOT = Path(__file__).resolve().parent.parent
COMMAND = [sys.executable, "-m", "storey.vector"]
CHILD_SECONDS = 900
IN_PER_M = 39.3701
FT_PER_M = 3.28084
BOX_PAD = 0.02           # share of the page's long side, as in the alignment work: door arcs swing past the walls that plan_box follows
DIM_MARGIN_PT = 200      # plan_box leaves dimension strings out, and on the Capitol sheets enlarged views at other scales start over 400 pt from it
MIN_DIMS = 3             # a fitted overrun would absorb the whole error of one or two strings
MAX_OVERRUN_PT = 12      # Capitol and Baxter lines run 6 pt past their ticks. Past this, a wrong scale is being fitted away
SCALE_STOP_IN = 1.0
DOOR_RANGE_IN = 20, 67   # the 0.5 to 1.7 m radius band behind the door recall figures in the build log
QUARTER_MAX_DEG = 95     # Capitol quarter arcs measure 87 to 94 degrees, and the over-bulged ones from 97
PEN_SHARE = 0.05         # the cut-wall pen held 12 to 46% of paired length on 11 sheets, and heavier pens such as sheet borders held none
PX_PER_M = 40.0          # raster resolution of the clinic route runs
U = 0.9 * PX_PER_M       # one door width in pixels, the unit of every morphology size below
MAX_RASTER = 60_000_000  # a small page at a very large printed scale would otherwise ask for gigabytes
SNAP_M = 6.0             # the clinic's stair well boxes sit up to 4.3 m from walkable floor. Farther than this, a marker is not on this plan
MIN_WALLED = 0.8
MIN_ROOMS = 2            # a wrong pen can leave one room that everything reaches
MIN_REACH = 0.8
MIN_REGISTERED = 3
REGISTER_SHARE = 0.3
MINIMUM = "Minimum. Examiner to confirm."
UNNAMED = "Unnamed room"
MAX_NAME = 60            # this text comes from the upload and goes on screen and into the letter
BESIDE_PT = 22           # how far a dimension's text can sit from its line
STRING_LINE_PT = 2       # dimension lines this close are one line
STRING_END_PT = 1.5      # the most an end may miss in points whatever its overall prints, so a small value paired with a long line cannot take in a far string. Set after seeing the data: at 5 pt Baxter raised dozens of false pairs on party walls
STRING_GAP_PT = -20, 3   # next start minus last end. Set after seeing Capitol, where a part drawn 6 in over its neighbour reads -14.9
STRING_ROUNDING_IN = 0.5 # values print to 1/2 in, and the Capitol audit rules a miss of 1/2 in as rounding
STRING_MAX_DIMS = 2000   # shortcut: a sheet with more dimension rows gets no strings, because each row is compared with every line, 0.15 s at 2,000 and 2.7 s at 8,000, index the lines by position if a real sheet holds this many. The most seen is 133

LINE, CURVE, OUTLINE = range(3)
QUOTES = str.maketrans({"’": "'", "‘": "'", "′": "'", "”": '"', "“": '"', "″": '"'})
FT_IN = re.compile(r"^(\d+)'\s*-?\s*(\d+)(?:\s+(\d+)/([1-9]\d*))?\"$")
PAPER_SCALE = re.compile(r"(?:(\d+)\s+)?(\d+)(?:/(\d+))?\s*\"\s*=\s*(\d+)\s*'(?:\s*-?\s*(\d+)\s*\")?")
RATIO_SCALE = re.compile(r"(?<![\d/])(\d+)\s*:\s*(\d+)")
WALL_LAYER = re.compile(r"(?i)^(?!.*(door|patt|hatch|htch)).*wall")
HALL = re.compile(r"(?i)^(hall|hallway|corridor|corr\.?)$")


def inches_per_pt(note: str) -> float | None:
    """Real inches per page point from a printed scale such as 1/4" = 1'-0" or 1:150, or None."""
    note = note.translate(QUOTES)
    value = None
    if m := PAPER_SCALE.search(note):
        whole, top, bottom, feet, inches = m.groups()
        if bottom == "0":
            return None
        paper = int(whole or 0) + int(top) / int(bottom or 1)
        value = (int(feet) * 12 + int(inches or 0)) / paper / 72 if paper else None
    elif (m := RATIO_SCALE.search(note)) and int(m[1]) and int(m[2]):
        value = int(m[2]) / int(m[1]) / 72
    return value if value and value > 0 and math.isfinite(value) else None


def parse_ftin(text: str) -> float | None:
    """Feet and inches to inches, or None. 14'-6 1/2" gives 174.5."""
    m = FT_IN.match(text.strip().translate(QUOTES))
    if not m:
        return None
    feet, inches, top, bottom = m.groups()
    return int(feet) * 12 + int(inches) + (int(top) / int(bottom) if top else 0)


def strokes(page):
    """Rows of x0, y0, x1, y1, pen, layer, kind, stroked in the rendered frame, and the layer names. Curves are sampled, so one circle fit serves arcs of either encoding."""
    layers, rows = {}, []
    for d in page.get_drawings():
        tail = round(d.get("width") or 0, 2), layers.setdefault(d.get("layer") or "", len(layers))
        stroked = d["type"] != "f"
        for item in d["items"]:
            if item[0] == "l":
                points, kind = [tuple(item[1]), tuple(item[2])], LINE
            elif item[0] == "c":
                p = np.array([tuple(q) for q in item[1:5]])
                u = np.linspace(0, 1, 9)[:, None]
                points, kind = [tuple(q) for q in (1 - u) ** 3 * p[0] + 3 * (1 - u) ** 2 * u * p[1] + 3 * (1 - u) * u * u * p[2] + u ** 3 * p[3]], CURVE
            elif item[0] in ("re", "qu"):
                q = item[1] if item[0] == "qu" else item[1].quad
                points, kind = [tuple(c) for c in (q.ul, q.ur, q.lr, q.ll, q.ul)], OUTLINE
            else:
                continue
            rows.extend((*a, *b, *tail, kind, stroked) for a, b in zip(points, points[1:]))
    a = np.array(rows, float).reshape(-1, 8)
    m = page.rotation_matrix
    for x, y in ((0, 1), (2, 3)):
        a[:, x], a[:, y] = a[:, x] * m.a + a[:, y] * m.c + m.e, a[:, x] * m.b + a[:, y] * m.d + m.f
    return a, list(layers)


def text_lines(page):
    """Rows of text, x0, y0, x1, y1, runs along x, in the rendered frame."""
    m = page.rotation_matrix
    turn = pymupdf.Matrix(m.a, m.b, m.c, m.d, 0, 0)
    out = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in block.get("lines", ()):
            text = " ".join(s["text"].strip() for s in line["spans"]).strip()
            if text:
                r = pymupdf.Rect(line["bbox"]) * m
                d = pymupdf.Point(line["dir"]) * turn
                out.append((text, r.x0, r.y0, r.x1, r.y1, abs(d.x) > abs(d.y)))
    return out


def clip(segs, box):
    """Rows that touch the box."""
    x0, y0, x1, y1 = box
    return segs[(np.minimum(segs[:, 0], segs[:, 2]) <= x1) & (np.maximum(segs[:, 0], segs[:, 2]) >= x0)
                & (np.minimum(segs[:, 1], segs[:, 3]) <= y1) & (np.maximum(segs[:, 1], segs[:, 3]) >= y0)]


def dimensions(lines, segs, layers, box):
    """Each feet-and-inches string near the plan, paired with the dimension line beside it. A string with no line beside it is left out."""
    named = [i for i, name in enumerate(layers) if "dim" in name.lower()]
    a = segs[(segs[:, 6] == LINE) & (segs[:, 4] > 0) & (np.isin(segs[:, 5], named) if named else True)]
    across = a[np.abs(a[:, 1] - a[:, 3]) < 0.3][:, :4], a[np.abs(a[:, 0] - a[:, 2]) < 0.3][:, [1, 0, 3, 2]]   # lines along x, and lines along y with the axes swapped
    rows = []
    for text, x0, y0, x1, y1, along_x in lines:
        inches = parse_ftin(text)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if inches is None or not (box[0] - DIM_MARGIN_PT <= cx <= box[2] + DIM_MARGIN_PT and box[1] - DIM_MARGIN_PT <= cy <= box[3] + DIM_MARGIN_PT):
            continue
        s, u, v = (across[0], cx, cy) if along_x else (across[1], cy, cx)
        lo, hi = np.minimum(s[:, 0], s[:, 2]), np.maximum(s[:, 0], s[:, 2])
        beside = (lo - 1 <= u) & (u <= hi + 1) & (np.abs(s[:, 1] - v) < BESIDE_PT)
        if beside.any():
            off = np.abs(s[:, 1] - v)
            near = np.flatnonzero(beside & (off <= off[beside].min() + 0.5))
            k = near[(hi - lo)[near].argmin()]
            rows.append({"text": text, "inches": inches, "along_x": along_x, "start": float(lo[k]), "end": float(hi[k]), "at": float(s[k, 1])})
    return rows


def check_scale(dims, ipp, strict):
    """(strings checked, reason to reject the scale or None). Dimension lines overrun their ticks by a fixed length, so that is fitted first.
    Only a dimension layer shows that a line is a dimension line, so `strict` stops on one string. Without it the pairing is a guess, and only the median string can speak for the scale."""
    if len(dims) < MIN_DIMS:
        return 0, None
    drawn = [(d["end"] - d["start"]) * ipp for d in dims]
    fit = statistics.median(a - d["inches"] for a, d in zip(drawn, dims))
    worst = max(range(len(dims)), key=lambda i: abs(drawn[i] - fit - dims[i]["inches"]))
    if abs(fit) > MAX_OVERRUN_PT * ipp or (strict and abs(drawn[worst] - fit - dims[worst]["inches"]) > SCALE_STOP_IN):
        return len(dims), f"The printed scale disagrees with dimension {dims[worst]['text']}: drawn {drawn[worst] - fit:.1f} in, printed {dims[worst]['inches']:.1f} in."
    return len(dims), None


def dimension_strings(dims, plan):
    """Each run of shorter dimensions on one line that spans an overall and prints a different length, once per parts and overall because plans print a string twice. Needs no scale.
    Neighbours that meet overlap by the lines' overruns, and that is a drawing style: -6 pt on the Michigan sets, 0 on BCOE and the ADUs, so the sheet's median says what meeting looks like.
    Past it one length is dimensioned twice, as in audit items T37 and T38. Short of it the string has a hole, as on Baxter X1.0 inside a party wall, so its sum need not reach the overall.
    shortcut: any parallel line can hold a string, so on Capitol A2.2 a front wall string 485 pt from a kitchen width misses one of its ends by only 1 in. The nearest line alone
    would compare 1 string on Baxter A2.1 where this compares 9, so ask for it only if a real sheet raises a far one."""
    rows = {(d["along_x"], d["at"], d["start"], d["end"], d["text"]): d for d in dims}.values()   # bold text is printed on top of itself
    if len(rows) > STRING_MAX_DIMS:
        return []
    lines = []
    for d in sorted(rows, key=lambda d: (d["along_x"], d["at"])):
        if lines and lines[-1][0]["along_x"] == d["along_x"] and d["at"] - lines[-1][0]["at"] < STRING_LINE_PT:
            lines[-1].append(d)
        else:
            lines.append([d])
    for line in lines:
        line.sort(key=lambda d: d["start"])
    between = lambda parts: [b["start"] - a["end"] for a, b in zip(parts, parts[1:])]
    low, high = STRING_GAP_PT
    gaps = [gap for line in lines for gap in between(line) if low <= gap <= high]
    if not gaps:
        return []
    meet = statistics.median_high(gaps)   # an observed gap: a midpoint on a sheet with few neighbours can read a real meeting as a hole
    found = {}
    for whole in rows:
        length = whole["end"] - whole["start"]
        # Ends and holes may miss by under the rounding, by the overall's own length, so at any scale a string drawn short raises nothing. That length holds both
        # overruns, which loosens a short overall: a 1'-0" at 1/4" = 1'-0" reads 2 pt to the inch, not 1.5.
        end = min(STRING_END_PT, STRING_ROUNDING_IN * length / whole["inches"]) if whole["inches"] else STRING_END_PT
        for line in lines:
            parts = [d for d in line if d["along_x"] == whole["along_x"] and d["end"] - d["start"] < length and d["start"] > whole["start"] - end and d["end"] < whole["end"] + end]
            if len(parts) < 2 or abs(parts[0]["start"] - whole["start"]) >= end or abs(parts[-1]["end"] - whole["end"]) >= end:
                continue
            total = sum(d["inches"] for d in parts)
            if abs(total - whole["inches"]) > STRING_ROUNDING_IN and all(low <= gap <= high and gap < meet + end for gap in between(parts)):
                ends = (parts[0]["start"], parts[0]["at"] - BESIDE_PT), (parts[-1]["end"], parts[0]["at"] + BESIDE_PT)
                a, b = ends if whole["along_x"] else (ends[0][::-1], ends[1][::-1])
                found.setdefault((tuple(d["text"] for d in parts), whole["text"]), {"parts": [d["text"] for d in parts], "sum_in": total, "overall": whole["text"], "box": [*plan.px(a), *plan.px(b)]})
    return list(found.values())


def chains(segs, tol=0.13):
    """Links segments that share end points into polylines. A junction ends a chain."""
    key = lambda x, y: (round(x / tol), round(y / tol))
    ends = defaultdict(list)
    for i, s in enumerate(segs):
        if (s[0], s[1]) != (s[2], s[3]):
            ends[key(s[0], s[1])].append((i, 0))
            ends[key(s[2], s[3])].append((i, 1))
    used = np.zeros(len(segs), bool)

    def walk(k):
        points = []
        while len(ends[k]) == 2:
            free = [(j, e) for j, e in ends[k] if not used[j]]
            if not free:
                break
            j, e = free[0]
            used[j] = True
            p = segs[j][2:4] if e == 0 else segs[j][:2]
            points.append(p)
            k = key(*p)
        return points

    out = []
    for i in np.argsort([len(ends[key(s[0], s[1])]) == 2 for s in segs], kind="stable"):   # chain ends first, so a chain is walked once
        if not used[i]:
            used[i] = True
            before, after = walk(key(segs[i][0], segs[i][1])), walk(key(segs[i][2], segs[i][3]))
            out.append(np.array([*before[::-1], segs[i][:2], segs[i][2:4], *after]))
    return out


def split_corners(line, deg=40):
    if len(line) < 3:
        return [line]
    v = np.diff(line, axis=0)
    turn = np.abs(np.diff(np.unwrap(np.arctan2(v[:, 1], v[:, 0]))))
    cuts = [0, *(i + 1 for i, t in enumerate(turn) if t > math.radians(deg)), len(line) - 1]
    return [line[a:b + 1] for a, b in zip(cuts, cuts[1:]) if b > a]


def fit_circle(p):
    x, y = p[:, 0], p[:, 1]
    sol = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x * x + y * y, rcond=None)[0]
    cx, cy = sol[:2]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0))
    return cx, cy, r, np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))


def find_arcs(segs, rmin, rmax, smin=55, smax=115):
    """Quarter-ish arcs with a radius in the band. An arc gets cut at every line touching it, so fragments on one circle merge before the sweep is judged."""
    frags = []
    for line in chains(segs):
        for part in split_corners(line):
            if len(part) < 3:
                continue
            cx, cy, r, res = fit_circle(part)
            if not rmin * 0.9 <= r <= rmax * 1.1 or res > max(0.25, 0.015 * r):
                continue
            ang = np.unwrap(np.arctan2(part[:, 1] - cy, part[:, 0] - cx))
            if abs(ang[-1] - ang[0]) >= math.radians(9):
                frags.append((cx, cy, r, part))
    groups = []
    for f in sorted(frags, key=lambda q: -len(q[3])):
        for g in groups:
            if math.hypot(f[0] - g[0][0], f[1] - g[0][1]) < max(1.2, 0.1 * f[2]) and abs(f[2] - g[0][2]) < max(0.8, 0.06 * f[2]):
                g.append(f)
                break
        else:
            groups.append([f])
    arcs = []
    for g in groups:
        pts = np.vstack([q[3] for q in g])
        cx, cy, r, res = fit_circle(pts)
        if not rmin <= r <= rmax or res > max(0.4, 0.025 * r):
            continue
        ang = np.sort(np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx), 2 * math.pi))
        gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
        i = gaps.argmax()
        sweep = 2 * math.pi - gaps[i]
        if math.radians(smin) <= sweep <= math.radians(smax):
            a0, a1 = ang[(i + 1) % len(ang)], ang[i]
            arcs.append({"c": np.array([cx, cy]), "r": r, "sweep": sweep, "a": np.array([cx + r * math.cos(a0), cy + r * math.sin(a0)]), "b": np.array([cx + r * math.cos(a1), cy + r * math.sin(a1)])})
    return arcs


def cover(s, c, d, length, t):
    """Length of strokes running along c + u * d for u in [0, length], within t of that line."""
    u = d / np.linalg.norm(d)
    n = np.array([-u[1], u[0]])
    r0, r1 = s[:, :2] - c, s[:, 2:4] - c
    t0, t1 = r0 @ u, r1 @ u
    lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
    ok = (np.maximum(np.abs(r0 @ n), np.abs(r1 @ n)) < t) & (hi > 0) & (lo < length)
    return float((np.minimum(hi[ok], length) - np.maximum(lo[ok], 0)).sum()) if ok.any() else 0.0


def attach_leaf(arcs, segs):
    """The leaf is a stroke from the arc's centre to one end, and the shut end is the other, whose chord lies in the wall line so the wall goes on behind the hinge.
    A threshold line across the opening must not read as a leaf."""
    s = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) > 0]
    for arc in arcs:
        c, a, b, r = arc["c"], arc["a"], arc["b"], arc["r"]
        t = max(1.2, 0.14 * r)
        la, lb = cover(s, c, a - c, r, t) / r, cover(s, c, b - c, r, t) / r
        wa, wb = cover(s, c, c - b, r * 0.6, t) / r, cover(s, c, c - a, r * 0.6, t) / r
        arc["leaf"] = max(la, lb) >= 0.6
        arc["shut"] = b if la + wa >= lb + wb else a
    return arcs


def dedupe(arcs):
    """One arc per centre and radius, because line weights and layers can draw it twice."""
    kept = []
    for a in sorted(arcs, key=lambda x: (not x["leaf"], -x["sweep"])):
        if not any(abs(a["c"][0] - b["c"][0]) < 1 and abs(a["c"][1] - b["c"][1]) < 1 and abs(a["r"] - b["r"]) < 0.5 for b in kept):
            kept.append(a)
    return kept


def door_arcs(segs, ipp):
    """Swing door arcs from Nx4 strokes, each with a leaf. A curve drawn on top of itself makes every node a junction, so repeated strokes go first."""
    s = np.unique(segs.round(2), axis=0)
    return [a for a in dedupe(attach_leaf(find_arcs(s, DOOR_RANGE_IN[0] / ipp, DOOR_RANGE_IN[1] / ipp), s)) if a["leaf"]]


def door_width(arc):
    """The leaf length in points. Past a quarter turn an arc still ends on the shut end and the leaf tip but bulges more, so its radius reads short and the chord does not."""
    return arc["r"] if arc["sweep"] <= math.radians(QUARTER_MAX_DEG) else float(np.linalg.norm(arc["a"] - arc["b"])) / math.sqrt(2)


def wall_segments(segs, ppm, tmin=0.07, tmax=0.45, lmin=0.5, overlap=0.5):
    """True for each long segment with a parallel partner at wall thickness. segs is Nx4 in points."""
    d = segs[:, 2:] - segs[:, :2]
    length = np.hypot(*d.T)
    i = np.flatnonzero(length >= lmin * ppm)
    s, dd = segs[i], d[i] / length[i, None]
    ok = np.zeros(len(s), bool)
    th = np.round(np.degrees(np.arctan2(dd[:, 1], dd[:, 0])) % 180, 0) % 180
    for t in np.unique(th):
        same = np.flatnonzero(np.abs((th - t + 90) % 180 - 90) <= 1)
        if len(same) < 2:
            continue
        u = np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
        off = s[same, :2] @ np.array([-u[1], u[0]])
        a0, a1 = s[same, :2] @ u, s[same, 2:] @ u
        order = np.argsort(off)
        off, lo, hi, ids = off[order], np.minimum(a0, a1)[order], np.maximum(a0, a1)[order], same[order]
        for k in range(len(ids)):
            first, last = np.searchsorted(off, [off[k] + tmin * ppm, off[k] + tmax * ppm])
            shared = np.minimum(hi[k], hi[first:last]) - np.maximum(lo[k], lo[first:last])
            good = shared >= overlap * np.minimum(hi[k] - lo[k], hi[first:last] - lo[first:last])
            if good.any():
                ok[ids[k]] = True
                ok[ids[first:last][good]] = True
    out = np.zeros(len(segs), bool)
    out[i[ok]] = True
    return out


def wall_pen(segs, ppm):
    """The heaviest pen holding a real share of the strokes that have a partner at wall thickness, or None. Title blocks and borders use heavier pens but have no partner. Rows end in the pen."""
    held = {}
    for pen in np.unique(segs[:, 4]):
        s = segs[segs[:, 4] == pen][:, :4]
        held[float(pen)] = float(np.hypot(*(s[:, 2:] - s[:, :2]).T)[wall_segments(s, ppm)].sum())
    total = sum(held.values())
    return max((pen for pen, length in held.items() if length >= PEN_SHARE * total), default=None) if total else None


def is_wall_layer(name):
    return bool(WALL_LAYER.search(name))


def wall_lines(plan):
    """Wall strokes in the plan box, Nx4 in points: wall layers where the sheet has them, otherwise the wall pen."""
    lines = plan.segs[(plan.segs[:, 6] != CURVE) & (plan.segs[:, 7] == 1)]
    on_layers = lines[np.isin(lines[:, 5], [i for i, name in enumerate(plan.layers) if is_wall_layer(name)])]
    if len(on_layers):
        return on_layers[:, :4]
    pen = wall_pen(lines[:, [0, 1, 2, 3, 4]], IN_PER_M / plan.ipp)
    return lines[lines[:, 4] == pen][:, :4]


def crossings(walls, at, along_x):
    """Sorted positions where wall lines cross a ray. `at` is the ray's fixed coordinate."""
    a = walls if along_x else walls[:, [1, 0, 3, 2]]
    s = a[(np.abs(a[:, 0] - a[:, 2]) < 0.5) & (np.minimum(a[:, 1], a[:, 3]) - 0.5 <= at) & (at <= np.maximum(a[:, 1], a[:, 3]) + 0.5)]
    return np.unique(np.round(s[:, 0], 1))


def corridors(plan, walls, dims):
    """Clear width at each printed hall label, with the printed dimension that spans it. Printed widths on these sets run from wall centre to wall centre.
    A reading needs walls on both axes, since one bounded axis can be the hall's length. The width is the shorter axis, with a door gap counted as a bound, and that axis must cross no door:
    a ray through an opening runs on to a far wall, and the long span would pass as a wide hall. A label with no reading keeps its row with clear_in None, so the check says Cannot tell.
    shortcut: an opening too wide for close_doors, with no door arc, still reads as a long span, compare with a printed width if a set shows one.
    shortcut: needs a Hall or Corridor label in a text layer, so a sheet with no text layer gets none, read labels with Spark if the clinic's corridors must be checked."""
    labels = []
    for text, x0, y0, x1, y1, _ in plan.lines:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not HALL.match(text) or not (plan.box[0] <= cx <= plan.box[2] and plan.box[1] <= cy <= plan.box[3]):
            continue
        if any(t == "Stair" and abs(a - x0) * plan.ipp < 40 and abs(b - y0) * plan.ipp < 20 for t, a, b, *_ in plan.lines):
            continue   # a stair hall, which the stair check covers
        labels.append((cx, cy))
    out = [{"point": plan.px(label), "clear_in": None, "printed_in": None} for label in labels]
    raster = wall_raster(plan, walls) if labels else None
    if raster is None:
        return out
    s, solid, closed, _ = raster
    gaps = closed & ~solid
    for row, (cx, cy) in zip(out, labels):
        spans = [ray_span(plan, walls, gaps, s, along_x, u, v) for along_x, u, v in ((True, cx, cy), (False, cy, cx))]
        if all(spans):
            free, clear, along_x, middle, v = min(spans)
            if free == clear:
                near = [d for d in dims if d["along_x"] == along_x and d["start"] - 6 <= middle <= d["end"] + 6 and 30 <= d["inches"] <= 80 and abs(d["at"] - v) < 220]
                row.update(clear_in=round(float(clear * plan.ipp), 1), printed_in=min(near, key=lambda d: abs(d["at"] - v))["inches"] if near else None)
    return out


def ray_span(plan, walls, gaps, s, along_x, u, v):
    """For the ray through u at fixed coordinate v: (span to the nearest door gap or wall each side, span between the nearest walls, along_x, the walls' middle, v), or None if walls do not bound it on both sides."""
    xs = crossings(walls, v, along_x)
    before, after = xs[xs < u], xs[xs > u]
    if not len(before) or not len(after):
        return None
    lo, hi = before[-1], after[0]
    t = np.linspace(lo, hi, int((hi - lo) * s) + 2)
    x, y = np.round(to_raster(plan, np.c_[t, np.full(len(t), v)] if along_x else np.c_[np.full(len(t), v), t], s)).astype(int).T
    doors = t[gaps[np.clip(y, 0, gaps.shape[0] - 1), np.clip(x, 0, gaps.shape[1] - 1)]]
    return doors[doors > u].min(initial=hi) - doors[doors < u].max(initial=lo), hi - lo, along_x, (lo + hi) / 2, v


def square(k):
    n = max(1, int(k))
    return np.ones((n, n), np.uint8)


def wide(k):
    return np.ones((1, max(1, int(k))), np.uint8)


def tall(k):
    return np.ones((max(1, int(k)), 1), np.uint8)


def draw(segs, shape):
    """Rasterises Nx4 pixel segments to a mask."""
    im = np.zeros(shape, np.uint8)
    for a in np.clip(np.round(segs), -30000, 30000).astype(np.int32).tolist():   # a hostile page can hold any coordinate
        cv2.line(im, (a[0], a[1]), (a[2], a[3]), 1, 1)
    return im


def close_doors(w, max_gap=1.35):
    """Bridges collinear wall ends across door-sized gaps. Long runs only, so corridors between parallel walls stay open."""
    w8 = w.astype(np.uint8)
    h = cv2.morphologyEx(w8, cv2.MORPH_OPEN, wide(0.35 * U))
    v = cv2.morphologyEx(w8, cv2.MORPH_OPEN, tall(0.35 * U))
    return (w8 | cv2.morphologyEx(h, cv2.MORPH_CLOSE, wide(max_gap * U)) | cv2.morphologyEx(v, cv2.MORPH_CLOSE, tall(max_gap * U))) > 0


def to_raster(plan, points, s):
    return (np.asarray(points, float) - plan.box[:2]) * s


def wall_raster(plan, walls):
    """(pixels per point, drawn walls, the same with door openings closed, door ends) in pixels of the plan box, or None if the raster would be too large."""
    s = PX_PER_M * plan.ipp / IN_PER_M
    shape = int((plan.box[3] - plan.box[1]) * s), int((plan.box[2] - plan.box[0]) * s)
    if shape[0] * shape[1] > MAX_RASTER:
        return None
    solid = cv2.morphologyEx(draw(to_raster(plan, walls.reshape(-1, 2), s).reshape(-1, 4), shape), cv2.MORPH_CLOSE, square(0.28 * PX_PER_M)) > 0
    doors = [(to_raster(plan, a["c"], s), to_raster(plan, a["shut"], s)) for a in plan.arcs]
    bars = draw(np.array([[*h, *e] for h, e in doors]).reshape(-1, 4), shape)
    return s, solid, close_doors(solid | (cv2.dilate(bars, square(3)) > 0)), doors


def footprint(closed, gap=9.0):
    """Everything not reachable from outside once windows and doors are bridged wide."""
    p = int(gap * U) + 1   # closing treats the image edge as ink, so a plan cropped tight needs this much paper around it
    c0 = cv2.copyMakeBorder(closed.astype(np.uint8), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    c = cv2.morphologyEx(c0, cv2.MORPH_CLOSE, wide(gap * U)) | cv2.morphologyEx(c0, cv2.MORPH_CLOSE, tall(gap * U))
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, square(0.12 * U))
    _, lab = cv2.connectedComponents(1 - c, connectivity=4)
    return (lab != lab[0, 0])[p:-p, p:-p]


def room_labels(closed, fp):
    """Labels of the free regions in the footprint, with door openings bridged so each room is its own region."""
    free = cv2.morphologyEx((fp & ~closed).astype(np.uint8), cv2.MORPH_OPEN, square(0.1 * U))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    keep = stats[:, cv2.CC_STAT_AREA] >= (0.8 * U) ** 2
    keep[0] = False
    lab[~keep[lab]] = 0
    return lab


def enclosed_rooms(labels, closed):
    """Drops regions with little drawn wall around them, because the footprint bridges wide notches and the ground outside a door must not start a route.
    shortcut: 0.8 comes from one building, where rooms held 0.84 and up, so a heavily glazed room stops being a route start, re-measure on a second set with exits."""
    rooms = labels > 0
    if not rooms.any():
        return labels
    edge = rooms & (cv2.erode(rooms.astype(np.uint8), square(3)) == 0)
    walled = cv2.dilate(closed.astype(np.uint8), square(0.1 * U)) > 0
    ids = np.unique(labels[rooms])
    keep = np.zeros(labels.max() + 1, bool)
    keep[ids[ndi.mean(walled, labels * edge, ids) >= MIN_WALLED]] = True
    return labels * keep[labels]


def walkable(blocked, fp, clear=0.3):
    """Free space in the footprint, eroded by half a body so pinholes in a wall do not leak."""
    r = max(1, int(clear * U * 0.5))
    return cv2.erode((fp & ~blocked).astype(np.uint8), square(2 * r + 1)) > 0


def stair_ring(shape, box, doors):
    """A wall around a stair box, open at its doors, because a shaft's walls can be on a lighter pen. With no door on the box there is no shaft to close:
    the stair is open to its hall, or its door was missed, and a ring would seal it."""
    inside = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(inside, (x0, y0), (x1, y1), 1, -1)
    ring = cv2.dilate(inside, square(0.3 * PX_PER_M)) - inside
    band = cv2.dilate(inside, square(0.6 * PX_PER_M)) - cv2.erode(inside, square(0.6 * PX_PER_M))
    opened = False
    for hinge, shut in doors:
        x, y = (int(hinge[0]) + int(shut[0])) // 2, (int(hinge[1]) + int(shut[1])) // 2
        if 0 <= y < shape[0] and 0 <= x < shape[1] and band[y, x]:
            cv2.circle(ring, (x, y), int(0.55 * PX_PER_M), 0, -1)
            opened = True
    return ring > 0 if opened else np.zeros(shape, bool)


def taut(walk, path):
    """Any-angle route along a pixel path by greedy line-of-sight shortcuts. Returns the corner pixels and the length."""
    def clear(a, b):
        steps = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) + 1
        return walk[np.linspace(a[0], b[0], steps).round().astype(int), np.linspace(a[1], b[1], steps).round().astype(int)].all()

    points, i, last = [path[0]], 0, len(path) - 1
    while i < last:
        j = last
        while j > i + 1 and not clear(path[i], path[j]):
            j = max(i + 1, i + (j - i) // 2) if j - i > 64 else j - 1
        points.append(path[j])
        i = j
    return points, sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def offset_between(a, b):
    """The shift that lands most of the points `a` on the points `b`, and how many land. Every pair votes in 2 pt bins, then the winners are averaged."""
    diff = (b[None] - a[:, None]).reshape(-1, 2)
    bins, votes = np.unique(np.round(diff / 2).astype(int), axis=0, return_counts=True)
    near = diff[np.hypot(*(diff - bins[votes.argmax()] * 2.0).T) <= 3]
    return np.median(near, axis=0), len(near)


@dataclass
class Plan:
    """One plan sheet in the rendered frame, in points."""
    segs: np.ndarray
    layers: list
    lines: list
    px_per_pt: float
    box: tuple
    walls_box: tuple | None     # the read's box round the walls, with no pad
    entry: dict
    ipp: float | None = None
    arcs: list = field(default_factory=list)

    def px(self, point):
        return [round(float(point[0]) * self.px_per_pt), round(float(point[1]) * self.px_per_pt)]


def blank(reason=None):
    return {"scale_in_per_pt": None, "scale_checked": 0, "doors": [], "routes": [], "corridors": [], "strings": [], "skipped": reason}


def plan_boxes(row, width, height, px_per_pt):
    """(the read stage's box in points with a margin, the same box without it), or the whole page and None if it gave no usable box."""
    b = row.get("plan_box")
    if not b or len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]:
        return (0.0, 0.0, width, height), None
    pad = BOX_PAD * max(width, height)
    x0, y0, x1, y1 = (v / px_per_pt for v in b)
    return (max(0.0, x0 - pad), max(0.0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)), (x0, y0, x1, y1)


def open_plan(page, sheet, row):
    width, height = page.rect.width, page.rect.height
    px_per_pt = sheet["width"] / width   # the image's own size, so this holds whichever limit ingest rendered to
    segs, layers = strokes(page)
    return Plan(segs, layers, text_lines(page), px_per_pt, *plan_boxes(row, width, height, px_per_pt), blank())


def measure(plan, row):
    """Dimension strings, which need no scale, then scale, doors and corridors. A scale that fails its check leaves no measurement on the sheet."""
    entry = plan.entry
    dims = dimensions(plan.lines, plan.segs, plan.layers, plan.box)
    entry["strings"] = dimension_strings(dims, plan)
    plan.ipp = inches_per_pt(row.get("scale") or "")
    if plan.ipp is None:
        entry["skipped"] = "No drawing scale was read for this sheet."
        return
    entry["scale_checked"], problem = check_scale(dims, plan.ipp, any("dim" in name.lower() for name in plan.layers))
    if problem:
        entry["skipped"], plan.ipp = problem, None
        return
    entry["scale_in_per_pt"] = plan.ipp
    plan.segs = clip(plan.segs, plan.box)
    plan.arcs = door_arcs(plan.segs[plan.segs[:, 6] != OUTLINE][:, :4], plan.ipp)
    entry["doors"] = [{"point": plan.px((a["c"] + a["shut"]) / 2), "width_in": round(door_width(a) * plan.ipp, 1), "hinge": plan.px(a["c"])} for a in plan.arcs]
    entry["corridors"] = corridors(plan, wall_lines(plan), dims)
    entry["skipped"] = "Routes did not finish."   # stays if the child dies before its route step


def level_of(row):
    """shortcut: pairs a life safety sheet with its plan by level words, so 1ST FLOOR and FIRST FLOOR do not pair, match on the floors stage's order if a real set words them differently."""
    return " ".join(re.sub(r"\bplans?\b", "", (row.get("level") or "").lower()).split())   # the key the floors stage groups storeys by


def exit_name(e):
    parts = [f"{e['width_mm']:.0f} mm" if e.get("width_mm") else None, f"{e['capacity']} people" if e.get("capacity") else None]
    return ", ".join(["Exit", *filter(None, parts)])


def marker_ends(i, plans, rows, exits):
    """Declared exit markers for sheet i's floor, moved onto sheet i, and the reason if a sheet of them could not be moved."""
    level, here = level_of(rows[i]), plans[i]
    ends, problem = [], None
    for sheet in sorted({e["sheet"] for e in exits}):
        other = plans.get(sheet)
        if other is None or (sheet != i and not (level and level_of(rows[sheet]) == level)):
            continue
        ratio, shift = 1.0, np.zeros(2)
        if sheet != i:
            if other.ipp is None:
                problem = "The life safety sheet has no checked scale."
                continue
            ratio = other.ipp / here.ipp   # the two sheets can be drawn at different scales
            a = np.array([arc["c"] for arc in other.arcs]).reshape(-1, 2) * ratio
            b = np.array([arc["c"] for arc in here.arcs]).reshape(-1, 2)
            shift, landed = offset_between(a, b) if len(a) and len(b) else (shift, 0)
            if landed < max(MIN_REGISTERED, REGISTER_SHARE * min(len(a), len(b))):
                problem = "The life safety sheet does not line up with this plan."
                continue
        for e in exits:
            if e["sheet"] == sheet:
                ends.append({"name": exit_name(e), "source": "life_safety", "point": np.array(e["point"], float) / other.px_per_pt * ratio + shift, "box": None})
    return ends, problem


def stairs_on(i, plan, verticals):
    """Each stair found on sheet i, in points. A check cell may be a stair that is gone, and a route must not end on it."""
    out = []
    for v in verticals:
        cell = v.get("floors", {}).get(str(i)) or {}
        if cell.get("state") == "found" and (cell.get("box") or cell.get("point")):
            box = np.array(cell["box"], float) / plan.px_per_pt if cell.get("box") else None
            point = np.array(cell["point"], float) / plan.px_per_pt if cell.get("point") else box[:2] / 2 + box[2:] / 2
            out.append({"name": f"Stair {v['id']}", "source": "matrix", "point": point, "box": box})
    return out


def end_at_stairs(ends, stairs, reach):
    """Above the discharge level a marker names a stair but is printed beside it, so its route ends at the stair's edge, not wherever the marker lands."""
    boxes = [s["box"] for s in stairs if s["box"] is not None]
    for end in ends:
        away = [float(np.hypot(*np.maximum(np.maximum(b[:2] - end["point"], end["point"] - b[2:]), 0))) for b in boxes]
        if away and min(away) <= reach:
            end["box"] = boxes[int(np.argmin(away))]


def route_ends(i, plans, rows, floors, verticals, exits):
    """(ends, reason if there are none). Declared markers win where they exist, because a stair with no marker may not be an exit. Exit doors are never inferred from a plan."""
    ends, problem = marker_ends(i, plans, rows, exits)
    discharge = floors.get("discharge")
    stairs = stairs_on(i, plans[i], verticals)
    if ends:
        if discharge is not None and i != discharge:
            end_at_stairs(ends, stairs, SNAP_M * IN_PER_M / plans[i].ipp)
        return ends, None
    if discharge is None:
        return [], problem or "No exit markers were read, and with no discharge level a stair cannot end a route."
    if i == discharge:
        return [], problem or "No exit markers were read for the discharge level, where a stair does not end a route."
    return stairs, None if stairs else problem or "No exit markers were read and no stair was found on this floor, so a route has no end."


def trace(plan, stairs):
    """(walkable cells, room labels, pixels per point) at half raster size, or None if the raster would be too large."""
    raster = wall_raster(plan, wall_lines(plan))
    if raster is None:
        return None
    s, solid, closed, doors = raster
    fp = rooms_fp = footprint(closed)
    # shortcut: rooms past a short read box start no route, and a box short by more than the pad cuts the building (clinic F2 reads 217 ft at 31 per mille), find the box from the walls if real reads show this
    if plan.walls_box:   # section marks are drawn in the wall pen and the footprint can bridge out to them, so no route starts past the box, but exits at its edge stay on the floor
        x0, y0, x1, y1 = np.round(to_raster(plan, np.reshape(plan.walls_box, (2, 2)), s).ravel()).astype(int)
        keep = np.zeros(fp.shape, bool)
        keep[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        rooms_fp = fp & keep
    blocked = solid.copy()
    for stair in stairs:
        if stair["box"] is not None:
            blocked |= stair_ring(fp.shape, to_raster(plan, stair["box"].reshape(2, 2), s).ravel(), doors)
    walk = walkable(blocked, fp)[::2, ::2]   # routes at 5 cm per pixel cost a quarter of the time
    return walk, enclosed_rooms(room_labels(closed, rooms_fp), closed)[::2, ::2] * walk, s


def starts_for(walk, ends, cell):
    """Maps walkable cells to the end they serve. A stair serves every free cell of its box, because a tread must not hide it. A marker snaps to the nearest free cell."""
    _, lab, stats, _ = cv2.connectedComponentsWithStats(walk.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= (0.4 * U) ** 2   # the room size rule at half raster size: slivers of free space inside a wall line are no place to land
    keep[0] = False
    if not keep.any():
        return {}
    _, nearest = ndi.distance_transform_edt(~keep[lab], return_indices=True)
    starts = {}
    for end in ends:
        if end["box"] is not None:
            x0, y0, x1, y1 = (max(int(v), 0) for v in (*cell(end["box"][:2]), *cell(end["box"][2:])))
            ys, xs = np.nonzero(walk[y0:y1, x0:x1])
            starts.update(((int(y) + y0, int(x) + x0), end) for y, x in zip(ys[::4], xs[::4]))
            if len(ys):
                continue
        x, y = cell(end["point"])
        y0, x0 = int(np.clip(y, 0, walk.shape[0] - 1)), int(np.clip(x, 0, walk.shape[1] - 1))
        to = nearest[:, y0, x0]
        if math.hypot(y - to[0], x - to[1]) * 2 / PX_PER_M <= SNAP_M:
            starts[int(to[0]), int(to[1])] = end
    return starts


def longest_route(plan, ends, stairs):
    """The route from the farthest reachable point of any room to its nearest end, or a reason. Reach is what checks the wall choice: walls that came out wrong cut rooms off."""
    traced = trace(plan, stairs)
    if traced is None:
        return [], "This plan is too large to trace at its scale."
    walk, rooms, s = traced
    ids = np.unique(rooms[rooms > 0])
    if len(ids) < MIN_ROOMS:
        return [], "Rooms could not be traced on this sheet."
    x0, y0 = plan.box[:2]
    cell = lambda p: (int(round((float(p[0]) - x0) * s / 2)), int(round((float(p[1]) - y0) * s / 2)))
    starts = starts_for(walk, ends, cell)
    if not starts:
        return [], "No route end lies on this floor."
    mcp = MCP_Geometric(np.where(walk, 1.0, -1.0), fully_connected=True)
    costs = mcp.find_costs(list(starts))[0]
    open_cells = np.isfinite(costs) & walk
    reached = ndi.maximum(open_cells, rooms, ids)
    if reached.mean() < MIN_REACH:
        return [], f"Walls did not trace: only {reached.mean():.0%} of rooms reach an exit, below the {MIN_REACH:.0%} reach needed."
    far = np.unravel_index(np.where(open_cells & (rooms > 0), costs, -1.0).argmax(), costs.shape)
    points, length = taut(walk, mcp.traceback(far))
    end = starts[int(points[0][0]), int(points[0][1])]
    return [{"path": [plan.px((q[1] * 2 / s + x0, q[0] * 2 / s + y0)) for q in points[::-1]], "length_ft": round(length * 2 / PX_PER_M * FT_PER_M, 1),
             "from_room": room_name(plan, rooms, cell, rooms[far]), "to_exit": end["name"], "end_source": end["source"], "label": MINIMUM}], None


def room_name(plan, rooms, cell, room):
    """The first printed word in the room that reads as a name."""
    for text, x0, y0, x1, y1, _ in plan.lines:
        x, y = cell(((x0 + x1) / 2, (y0 + y1) / 2))
        if 0 <= y < rooms.shape[0] and 0 <= x < rooms.shape[1] and rooms[y, x] == room and sum(c.isalpha() for c in text) >= 3:
            return text[:MAX_NAME]
    return UNNAMED


def save(folder, index, entry):
    tmp = folder / f"{index}.tmp"
    tmp.write_text(json.dumps(entry, allow_nan=False))
    tmp.replace(folder / f"{index}.json")


def measure_all(job, folder):
    """The stage body, in the child. Each sheet is saved as soon as it is known, so a kill loses only what had not finished."""
    rows = {r["index"]: r for r in job["read"]}
    sheets = {s["index"]: s for s in job["sheets"]}
    floors, verticals, exits = job["floors"], job["verticals"], job["exits"]
    plans = {}
    with pymupdf.open(job["pdf"]) as doc:
        for i in job["plans"]:
            try:
                plan = open_plan(doc[i], sheets[i], rows[i])
                measure(plan, rows[i])
                save(folder, i, plan.entry)
            except Exception:   # one bad page must not cost the rest, and that includes a value json refuses
                save(folder, i, blank("This sheet could not be measured."))
                continue
            plans[i] = plan
    for i, plan in plans.items():
        if plan.ipp is not None:
            try:
                plan.entry["routes"], plan.entry["skipped"] = route_step(i, plans, rows, floors, verticals, exits)
                save(folder, i, plan.entry)
            except Exception:   # the sheet keeps its doors and corridors, and its reason already says routes did not finish
                pass


def route_step(i, plans, rows, floors, verticals, exits):
    """(routes, reason if there are none) for one sheet."""
    order = floors.get("order") or []
    if i not in order:
        return [], "Routes run only on the sheet chosen for each floor." if order else "No floor order was found, so routes did not run."
    ends, reason = route_ends(i, plans, rows, floors, verticals, exits)
    return longest_route(plans[i], ends, stairs_on(i, plans[i], verticals)) if ends else ([], reason)


def main(argv):
    """Child process. Reports through files, because a hostile PDF can flood stdout with library warnings."""
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    except (ValueError, OSError):
        pass   # shortcut: macOS refuses this limit, the wall-clock timeout in vector_stage still applies
    folder = Path(argv[0])
    measure_all(json.loads((folder / "job.json").read_text()), folder)
    return 0


def not_measured(sheet, row):
    """Why a sheet gets no measurement, or None."""
    if sheet["kind"] != "vector":
        return "Scanned sheet, so nothing can be measured from it."
    if row is None or not row.get("read_ok"):
        return "The sheet read has no result for this sheet."
    return None if row.get("cls") == "floor_plan" else "Not a floor plan."


def vector_stage(ctx) -> dict:
    """Facts for the door, travel distance and corridor checks. See the stage contract in the build plan. Keys are sheet indexes."""
    rows = ctx.stage("read").get("sheets", [])
    by_index = {r["index"]: r for r in rows}
    out = {s["index"]: blank(not_measured(s, by_index.get(s["index"]))) for s in ctx.sheets}
    plans = [i for i, entry in out.items() if not entry["skipped"]]
    if not plans:
        return {"sheets": out}
    pdfs = sorted((ctx.dir / "upload").glob("*.pdf"))
    folder = ctx.dir / "vector"
    shutil.rmtree(folder, ignore_errors=True)   # a rerun follows new stage results, so nothing old may stand
    folder.mkdir()
    exits = ctx.stage("declared").get("exits") or {}
    markers = [{**m, "sheet": s["sheet"]} for s in exits.get("sheets", []) for m in s.get("markers") or []]
    (folder / "job.json").write_text(json.dumps({
        "pdf": str(pdfs[0].resolve()) if pdfs else "", "plans": plans, "read": rows, "floors": ctx.stage("floors"), "verticals": ctx.stage("verticals").get("verticals", []),
        "exits": markers, "sheets": [{"index": s["index"], "width": s["width"]} for s in ctx.sheets]}))
    child = subprocess.Popen([*COMMAND, str(folder)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + CHILD_SECONDS
    while child.poll() is None and time.monotonic() < deadline:
        ctx.progress("Measuring vector sheets", len(list(folder.glob("*.json"))) - 1, len(plans))
        time.sleep(0.2)
    child.kill()
    child.wait()
    for i in plans:
        path = folder / f"{i}.json"
        out[i] = json.loads(path.read_text()) if path.exists() else blank("The vector content of this sheet could not be read.")
    return {"sheets": out}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
