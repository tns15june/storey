"""Vector extraction from a CAD PDF page. Everything is returned in rotated page points, the frame get_pixmap renders."""
import math
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz

ROOT = Path("/Users/tsharma/Github/Meta")


def open_page(id, file, page):
    return fitz.open(ROOT / "data/plans" / id / file)[page]


def segments(pg):
    """(x1,y1,x2,y2,width,layer,kind) for each straight line, plus curves as arcs list. Coordinates checked by overlay."""
    segs, curves = [], []
    for d in pg.get_drawings():
        w, lay = d.get("width") or 0.0, d.get("layer") or ""
        for it in d["items"]:
            if it[0] == "l":
                segs.append((it[1].x, it[1].y, it[2].x, it[2].y, w, lay))
            elif it[0] == "re":
                r = it[1]
                for a, b, c, e in ((r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)):
                    segs.append((a, b, c, e, w, lay))
            elif it[0] == "qu":
                q = it[1]
                for a, b in ((q.ul, q.ur), (q.ur, q.lr), (q.lr, q.ll), (q.ll, q.ul)):
                    segs.append((a.x, a.y, b.x, b.y, w, lay))
            elif it[0] == "c":
                curves.append((it[1], it[2], it[3], it[4], w, lay))
    return segs, curves


FT_IN = re.compile(r"^(\d+)'-(\d+)(?:\s+(\d+)/(\d+))?\"$")


def parse_ftin(s):
    """Feet-and-inches string to inches, or None. 14'-6 1/2" -> 174.5"""
    m = FT_IN.match(s.strip())
    if not m:
        return None
    f, i, n, d = m.groups()
    return int(f) * 12 + int(i) + (int(n) / int(d) if n else 0)


def joined_words(pg, gap=3.0):
    """Words on one baseline joined when close, so 14'-6 + 1/2" is one string. Handles vertical text by its line dir."""
    out = []
    for b in pg.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            txt = " ".join(s["text"].strip() for s in ln["spans"]).strip()
            if txt:
                out.append((txt, fitz.Rect(ln["bbox"]), ln["dir"]))
    return out


def dims(pg):
    """(inches, text, rect, dir) for each feet-and-inches dimension on the page."""
    r = []
    for t, rect, dr in joined_words(pg):
        v = parse_ftin(t)
        if v is not None:
            r.append((v, t, rect, dr))
    return r


if __name__ == "__main__":
    assert parse_ftin("14'-6 1/2\"") == 174.5 and parse_ftin("8'-0\"") == 96 and parse_ftin("A5.0") is None


def rotated(pg, pts):
    """Unrotated page points to rendered page points, the frame of get_pixmap."""
    M = pg.rotation_matrix
    return [fitz.Point(x, y) * M for x, y in pts]


def seg_arrays(segs):
    return np.array([s[:4] for s in segs], float) if segs else np.zeros((0, 4))


def stair_runs(segs, tread_range=(4.0, 22.0), min_steps=5, min_ratio=2.5, max_ratio=9.0):
    """Stair runs from parallel, equal-length, equally spaced segments. tread_range is in page points.

    Returns dicts with box (x0,y0,x1,y1), steps, tread (pt), width (pt), angle. Hatch rejection is the ratio test:
    a tread is 2.5 to 9 times as long as its spacing, and stairs do not repeat for more than 40 steps.
    """
    a = seg_arrays(segs)
    if not len(a):
        return []
    d = a[:, 2:4] - a[:, :2]
    L = np.hypot(*d.T)
    keep = L > 8
    a, d, L = a[keep], d[keep], L[keep]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
    out, used = [], set()
    for base in np.unique(np.round(ang / 3) * 3):
        idx = np.where(np.abs(((ang - base + 90) % 180) - 90) < 2)[0]
        if len(idx) < min_steps + 1:
            continue
        th = np.radians(base)
        u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        for lq in np.unique(np.round(L[idx] / 4) * 4):
            lidx = idx[np.abs(L[idx] - lq) < 0.2 * lq + 2]
            if len(lidx) < min_steps + 1:
                continue
            mid = (a[lidx, :2] + a[lidx, 2:]) / 2
            # split into separate columns first: side-by-side stairs share tread offsets
            cols = np.argsort(mid @ u)
            for grp in np.split(cols, np.where(np.diff((mid @ u)[cols]) > 0.5 * lq)[0] + 1):
                if len(grp) < min_steps + 1:
                    continue
                off, along = mid[grp] @ n, mid[grp] @ u
                order = np.argsort(off)
                off, along, lid = off[order], along[order], lidx[grp][order]
                k = np.r_[True, np.diff(off) > 0.8]  # two strokes on one tread edge
                off, along, lid = off[k], along[k], lid[k]
                i = 0
                while i < len(off) - min_steps:
                    s, best = None, [i]
                    for t in range(i + 1, len(off)):
                        g = off[t] - off[best[-1]]
                        if g > tread_range[1]:
                            break
                        if g < tread_range[0]:
                            continue
                        if s is None:
                            s = g
                        if abs(g - s) < 0.12 * s + 0.3:
                            best.append(t)
                    n_steps = len(best) - 1
                    if min_steps <= n_steps <= 40 and s and min_ratio <= lq / s <= max_ratio:
                        ids = lid[best]
                        pts = np.vstack([a[ids, :2], a[ids, 2:]])
                        box = (*pts.min(0), *pts.max(0))
                        key = tuple(np.round(box, 0))
                        if key not in used:
                            used.add(key)
                            out.append(dict(box=[float(x) for x in box], steps=n_steps, tread=float(s), width=float(lq), angle=float(base), ids=[int(x) for x in ids]))
                        i = best[-1]
                    i += 1
    # keep the largest of overlapping runs
    out.sort(key=lambda r: -r["steps"] * r["width"])
    kept = []
    for r in out:
        if all(_iou(r["box"], q["box"]) < 0.3 for q in kept):
            kept.append(r)
    return kept


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / min((a[2] - a[0]) * (a[3] - a[1]) + 1e-6, (b[2] - b[0]) * (b[3] - b[1]) + 1e-6)


SCALE_NOTE = re.compile(r"^(\d+)(?:/(\d+))?\"\s*=\s*1'-0\"$")


def inches_per_pt(note):
    """Real inches per page point from a note like 1/4" = 1'-0". None if it does not parse."""
    m = SCALE_NOTE.match(note.strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2) or 1)
    return 12.0 / (a / b) / 72.0


def plan_panel(pg, segs, title_re=r"(?i)floor plan", rule_layer=None):
    """Panel around the plan: ray cast from the title to the nearest long ruled lines. Returns (rect, scale note, title).

    shortcut: assumes panels are ruled with straight axis-aligned lines; rule_layer names the ruling layer, upgrade to a cell graph for sheets without one.
    """
    words = vec_lines = joined_words(pg)
    titles = [(t, r) for t, r, _ in words if re.search(title_re, t) and max(r.width, r.height) > 60]
    if not titles:
        return None, None, None
    title, tr = max(titles, key=lambda x: max(x[1].width, x[1].height))
    notes = [(t, r) for t, r, _ in words if SCALE_NOTE.match(t.strip())]
    note = min(notes, key=lambda x: abs(x[1].x0 - tr.x0) + abs(x[1].y0 - tr.y0))[0] if notes else None
    a = seg_arrays([s for s in segs if rule_layer is None or s[5] == rule_layer])
    W, H = pg.mediabox.width, pg.mediabox.height
    L = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    a = a[L > 0.15 * min(W, H)]
    cx, cy = (tr.x0 + tr.x1) / 2, (tr.y0 + tr.y1) / 2
    v = a[np.abs(a[:, 0] - a[:, 2]) < 0.5]
    h = a[np.abs(a[:, 1] - a[:, 3]) < 0.5]
    vc = v[(np.minimum(v[:, 1], v[:, 3]) <= cy) & (cy <= np.maximum(v[:, 1], v[:, 3]))][:, 0]
    hc = h[(np.minimum(h[:, 0], h[:, 2]) <= cx) & (cx <= np.maximum(h[:, 0], h[:, 2]))][:, 1]
    x0 = vc[vc < tr.x0 - 2].max(initial=0.0)
    x1 = vc[vc > tr.x1 + 2].min(initial=W)
    y0 = hc[hc < tr.y0 - 2].max(initial=0.0)
    y1 = hc[hc > tr.y1 + 2].min(initial=H)
    return fitz.Rect(x0, y0, x1, y1), note, title
