"""Render Swiss Dwellings floors as line drawings with exact truth. Each floor gets a small random shift, as real sheets do."""
import json

import cv2
import numpy as np
import shapely.wkt

from data import OUT

D = json.loads((OUT / "swiss_subset.json").read_text())
PX = 40  # px per metre


def floors(b):
    """Floor rows of one building, bottom up. Window rows sit on a half-level, so they join the floor below."""
    fl = {}
    for r in D[b]:
        fl.setdefault(r["floor_id"], []).append(r)
    return sorted(fl.values(), key=lambda rows: min(float(x["elevation"]) for x in rows))


def geom(rows, *kinds):
    out = []
    for r in rows:
        if r["entity_subtype"] in kinds:
            g = shapely.wkt.loads(r["geometry"])
            out += list(g.geoms) if hasattr(g, "geoms") else [g]
    return out


def render(rows, ox, oy, W, H, shift=(0, 0)):
    im = np.full((H, W, 3), 255, np.uint8)
    tf = lambda g: np.round((np.array(g.exterior.coords) - (ox, oy)) * PX + shift).astype(np.int32)
    for g in geom(rows, "WALL"):
        cv2.fillPoly(im, [tf(g)], (0, 0, 0))
    for g in geom(rows, "COLUMN"):
        cv2.fillPoly(im, [tf(g)], (0, 0, 0))
    for g in geom(rows, "RAILING"):
        cv2.polylines(im, [tf(g)], True, (0, 0, 0), 1)
    for g in geom(rows, "DOOR", "ENTRANCE_DOOR", "WINDOW"):  # openings cut the wall, with a thin leaf line
        cv2.fillPoly(im, [tf(g)], (255, 255, 255))
        cv2.polylines(im, [tf(g)], True, (120, 120, 120), 1)
    for g in geom(rows, "STAIRS"):  # treads every 25 cm along the long side
        p = tf(g)
        cv2.polylines(im, [p], True, (0, 0, 0), 2)
        r = cv2.minAreaRect(p)
        box = cv2.boxPoints(r)
        a, b, c = box[0], box[1], box[2]
        if np.linalg.norm(b - a) < np.linalg.norm(c - b):
            a, b, c = b, c, box[3]
        n = max(int(np.linalg.norm(b - a) / (0.25 * PX)), 2)
        for i in range(1, n):
            t = i / n
            cv2.line(im, tuple((a + (b - a) * t).astype(int)), tuple((a + (b - a) * t + (c - b)).astype(int)), (0, 0, 0), 1)
    for g in geom(rows, "ELEVATOR"):  # the X in a lift shaft
        p = tf(g)
        cv2.polylines(im, [p], True, (0, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(p)
        cv2.line(im, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.line(im, (x + w, y), (x, y + h), (0, 0, 0), 1)
    return im


def stack(b, keep=None, seed=0, jitter=30):
    fls = floors(b)
    fls = [f for f in fls if geom(f, "WALL") and sum(x["entity_subtype"] == "WALL" for x in f) >= 40]
    if keep:
        fls = fls[keep[0]:keep[1]]
    walls = [g for f in fls for g in geom(f, "WALL")]
    x = np.concatenate([np.array(g.exterior.coords) for g in walls])
    ox, oy = x.min(0) - 2
    W, H = (int((x.max(0)[0] - ox + 2) * PX), int((x.max(0)[1] - oy + 2) * PX))
    s = min(1, 1600 / max(W, H))
    rng = np.random.default_rng(seed)
    shifts = [tuple(int(v) for v in rng.integers(-jitter, jitter + 1, 2)) for _ in fls]
    return fls, (ox, oy, W, H), shifts


if __name__ == "__main__":
    for b in D:
        fls, (ox, oy, W, H), sh = stack(b)
        print(b, len(fls), W, H, [len(geom(f, "COLUMN")) for f in fls], [len(geom(f, "CORRIDOR", "LOBBY")) for f in fls], [len(geom(f, "STAIRCASE")) for f in fls])
    b = list(D)[0]
    fls, (ox, oy, W, H), sh = stack(b)
    cv2.imwrite(str(OUT / "swiss_sample.jpg"), render(fls[1], ox, oy, W, H))
