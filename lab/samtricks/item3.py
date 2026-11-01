"""Item 3: SAM concept "circle" counts door tag bubbles on a plan crop, against vector truth and a Hough no-model baseline. Usage: item3.py"""
import json
import math
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "lab/samtricks")
import api as a  # noqa: E402
import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402
import reg  # noqa: E402
import rules as R  # noqa: E402

fitz.TOOLS.mupdf_display_errors(False)
T = json.load(open(reg.TRUTH / "item3_tags.json"))


def crop(t):
    pg = fitz.open(reg.ROOT / t["file"])[t["page"]]
    # clip is in the rotated frame the truth uses, so render the page rotated and cut it.
    z = t["dpi"] / 72
    pm = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    im = cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n), cv2.COLOR_RGB2BGR)
    c = t["clip"]
    return im[int(c[1] * z):int(c[3] * z), int(c[0] * z):int(c[2] * z)].copy()


def whole(im):
    return [dict(box=r["box"], conf=r["conf"]) for r in a.sam("circle", im)]


def tiled(im, cols=2, rows=3, overlap=.12):
    h, w = im.shape[:2]
    tw, th = int(w / cols * (1 + overlap)), int(h / rows * (1 + overlap))
    jobs = [(int(x), int(y)) for x in np.linspace(0, w - tw, cols) for y in np.linspace(0, h - th, rows)]
    out = []
    per = []
    for rows_, (x, y) in zip(a.pmap(lambda j: a.sam("circle", im[j[1]:j[1] + th, j[0]:j[0] + tw]), jobs), jobs):
        per.append(len(rows_))
        for r in rows_:
            b = r["box"]
            out.append(dict(box=[b[0] + x, b[1] + y, b[2] + x, b[3] + y], conf=r["conf"]))
    uniq = []
    for r in sorted(out, key=lambda q: -(q["conf"] or 0)):
        c = ((r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2)
        if all(math.dist(c, ((u["box"][0] + u["box"][2]) / 2, (u["box"][1] + u["box"][3]) / 2)) > 8 for u in uniq):
            uniq.append(r)
    return uniq, per


def hough(im):
    g = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 3)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=18, param1=120, param2=22, minRadius=11, maxRadius=17)
    return [] if c is None else [dict(box=[float(x - r), float(y - r), float(x + r), float(y + r)], conf=None) for x, y, r in c[0]]


def score(t, preds):
    d = 28.1
    sized = [p for p in preds if .6 * d <= max(p["box"][2] - p["box"][0], p["box"][3] - p["box"][1]) <= 1.6 * d]
    tags = [q["c"] for q in t["door_tags"]]
    pairs = sorted((math.dist(((p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2), q), i, j) for i, p in enumerate(sized) for j, q in enumerate(tags))
    up, uq, hit = set(), set(), []
    for v, i, j in pairs:
        if v <= R.TAG_R and i not in up and j not in uq:
            up.add(i)
            uq.add(j)
            hit.append(j)
    missed = [t["door_tags"][j]["text"] for j in range(len(tags)) if j not in uq]
    return dict(masks=len(preds), sized=len(sized), hit=len(hit), tags=len(tags), schedule=t["schedule_marks"], false=len(sized) - len(hit), missed=missed)


if __name__ == "__main__":
    res = {}
    cells = []
    for k, t in T.items():
        im = crop(t)
        w = whole(im)
        ti, per = tiled(im)
        hg = hough(im)
        res[k] = dict(shape=im.shape[:2], whole=score(t, w), tiled=score(t, ti), hough=score(t, hg), masks_per_tile=per)
        print(k, im.shape[:2], "whole", res[k]["whole"], "\n  tiled", res[k]["tiled"], "tile masks", per, "\n  hough", res[k]["hough"])
        v = im.copy()
        for q in t["door_tags"]:
            cv2.circle(v, (int(q["c"][0]), int(q["c"][1])), 22, (255, 0, 255), 3)
        for q in t["other_circles"]:
            cv2.circle(v, (int(q["c"][0]), int(q["c"][1])), int(q["d"] / 2) + 6, (255, 160, 0), 1)
        for p in ti:
            b = [int(x) for x in p["box"]]
            cv2.rectangle(v, tuple(b[:2]), tuple(b[2:]), (0, 170, 0), 2)
        cells.append(v)
    h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cells]
    cv2.imwrite(str(reg.OUT / "item3_tags.jpg"), np.hstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (reg.OUT / "item3.json").write_text(json.dumps(res, indent=1))
    print("spent", round(a.m.spent("samtricks"), 3))
