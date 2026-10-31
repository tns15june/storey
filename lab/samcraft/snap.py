"""Spark box plus a deterministic snap of each edge to the nearest long ink line, with and without SAM after it. Reads the Spark boxes again from a repeat of the same prompt only if not cached."""
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

import core as c
import recipe as Q
import rooms as R
from data import OUT, ink_of, poly_mask

CACHE = OUT / "spark_boxes.pkl"


def snap(b, ink):
    """Move each box edge to the wall face it most likely means. Search window is a quarter of the box side."""
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = ink.shape
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    dx, dy = max(int(bw * .25), 8), max(int(bh * .25), 8)
    ys = slice(y1 + bh // 8, y2 - bh // 8 + 1)
    xs = slice(x1 + bw // 8, x2 - bw // 8 + 1)
    col = ink[ys, :].mean(0)  # share of the edge's span that is ink, per column
    row = ink[:, xs].mean(1)

    def best(prof, c0, d, inward):
        lo, hi = max(c0 - d, 0), min(c0 + d, len(prof) - 1)
        seg = prof[lo:hi + 1]
        if seg.size == 0 or seg.max() < .4:
            return c0
        p = lo + int(np.argmax(seg))
        while 0 < p + inward < len(prof) - 1 and prof[p + inward] >= .4:  # walk across the wall to its room face
            p += inward
        return p
    a, b_, c_, d = best(col, x1, dx, 1), best(row, y1, dy, 1), best(col, x2, dx, -1), best(row, y2, dy, -1)
    return a if a < c_ - 8 else x1, (b_ if b_ < d - 8 else y1), (c_ if a < c_ - 8 else x2), (d if b_ < d - 8 else y2)


def spark_boxes(pid):
    im, g = R.PLANS[pid]
    h, w = im.shape[:2]
    d, info = c.m.spark("Find every room in this floor plan. Give one box per room: x1, y1, x2, y2 on a 0-1000 grid of this image, x from the left edge, y from the top edge. "
                        "The box must touch the room's own walls. Include closets, bathrooms and corridors. Do not box the whole plan.", [im], schema=Q.SPARK, effort="low")
    return pid, [(b["x1"] * w / 1000, b["y1"] * h / 1000, b["x2"] * w / 1000, b["y2"] * h / 1000) for b in d["rooms"] if b["x2"] > b["x1"] and b["y2"] > b["y1"]]


def run(boxes):
    rows, tiles = [], []
    for pid, bs in boxes.items():
        im, g = R.PLANS[pid]
        h, w = im.shape[:2]
        ink = ink_of(im)
        sn = [snap(b, ink) for b in bs]
        objs = [dict(box=b) for b in sn]
        masks = []
        for ch in range(0, len(objs), 16):
            got = {o: mk for o, _, _, mk in c.sam(c.obj_prompt(objs[ch:ch + 16], w, h), im)}
            masks += [got.get(i + 1, np.zeros((h, w), bool)) for i in range(len(objs[ch:ch + 16]))]
        bm = [Q.box_mask(b, im.shape) for b in bs]
        sm = [Q.box_mask(b, im.shape) for b in sn]
        fm = [c.fill_holes(m) for m in masks]
        for r in [r for r in R.ROOMS if r["pid"] == pid]:
            gt = poly_mask(g["rooms"][r["i"]][1], im.shape)
            k = int(np.argmax([c.iou(b, gt) for b in bm]))
            rows.append(dict(pid=pid, i=r["i"], walled=r["walled"], box=c.iou(bm[k], gt), snap=c.iou(sm[k], gt), sam=c.iou(fm[k], gt)))
            tiles.append((c.overlay(im, sm[k], gt, (255, 0, 0)), f"{pid} box {rows[-1]['box']:.2f} snap {rows[-1]['snap']:.2f} sam {rows[-1]['sam']:.2f}"))
    return rows, tiles


if __name__ == "__main__":
    if CACHE.exists():
        boxes = pickle.loads(CACHE.read_bytes())
    else:
        boxes = dict(c.pmap(spark_boxes, R.T["plans"], n=4))  # all 12 plans
        CACHE.write_bytes(pickle.dumps(boxes))
    rows, tiles = run(boxes)
    (OUT / "snap.json").write_text(json.dumps(rows))
    for name, key in (("Spark box", "box"), ("Spark box + snap", "snap"), ("Spark box + snap + SAM", "sam")):
        a = np.array([x[key] for x in rows])
        print(name, "n", len(a), "median", round(float(np.median(a)), 3), ">=.5", round(float((a >= .5).mean()), 3), ">=.7", round(float((a >= .7).mean()), 3), ">=.8", round(float((a >= .8).mean()), 3))
    c.sheet(tiles[:16], OUT / "snap_sheet.jpg")
    print("spent", round(c.m.spent("samcraft"), 3))
