"""A controlled missing-column test: erase some columns from one floor of a Swiss stack and see whether the tracker reports them gone."""
import json

import cv2
import numpy as np

import core as c
import swiss as S
from data import OUT
from track_swiss import REPEAT, tf


def run(b, start, gone_floor, jitter=30):
    fls, (ox, oy, W, H), shifts = S.stack(b, jitter=jitter)
    fr = [S.render(f, ox, oy, W, H, s) for f, s in zip(fls, shifts)]
    cols = sorted(S.geom(fls[start], "COLUMN"), key=lambda g: -g.area)[:8]
    erased = cols[::2]
    for g in erased:  # paint the footprint and a small margin white
        q = np.round(tf(g, ox, oy, shifts[gone_floor])).astype(np.int32)
        cv2.fillPoly(fr[gone_floor], [q], (255, 255, 255))
        cv2.polylines(fr[gone_floor], [q], True, (255, 255, 255), 3)
    path = OUT / f"erase_{b}.mp4"
    w, h = c.encode(fr, path, repeat=REPEAT)
    objs = []
    for g in cols:
        q = tf(g, ox, oy, shifts[start])
        objs.append(dict(box=(q[:, 0].min() - 3, q[:, 1].min() - 3, q[:, 0].max() + 3, q[:, 1].max() + 3)))
    res = c.video(c.obj_prompt(objs, w, h, frame=start * REPEAT), path, len(fr) * REPEAT)
    last = gone_floor * REPEAT + REPEAT - 1
    rows = [dict(b=b, floor=gone_floor, obj=i + 1, erased=g in erased, got=(last, i + 1) in res, conf=res.get((last, i + 1), (0, None))[1]) for i, g in enumerate(cols)]
    v = fr[gone_floor].copy()
    for i in range(len(cols)):
        r = res.get((last, i + 1))
        if r:
            v = c.overlay(v, c.paste(r[0], r[2], v.shape))
    cv2.imwrite(str(OUT / f"erase_{b}.jpg"), v)
    return rows


rows = run("15364", 1, 2) + run("17990", 2, 3)
(OUT / "erase.json").write_text(json.dumps(rows))
e = [r for r in rows if r["erased"]]
k = [r for r in rows if not r["erased"]]
print("erased", len(e), "still reported", sum(r["got"] for r in e), "| untouched", len(k), "lost", sum(not r["got"] for r in k))
print(rows)
print("spent", round(c.m.spent("samcraft"), 3))
