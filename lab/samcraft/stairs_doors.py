"""Stair run and door recipes on CubiCasa, boxes and points taken from the truth polygons."""
import json

import cv2
import numpy as np

import core as c
from data import OUT, load, poly_mask, rep_point, select

T = select()
data = {p: load(p) for p in set(T["plans"] + T["stairs"])}
rng = np.random.default_rng(0)


def bbox(q):
    return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


def crop(im, q, pad, size=1024):
    x1, y1, x2, y2 = bbox(q)
    side = max(x2 - x1, y2 - y1) * pad
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1, X2, Y2 = max(int(cx - side / 2), 0), max(int(cy - side / 2), 0), min(int(cx + side / 2), im.shape[1]), min(int(cy + side / 2), im.shape[0])
    s = size / max(X2 - X1, Y2 - Y1)
    return cv2.resize(im[Y1:Y2, X1:X2], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), (X1, Y1), s


def jitter(b, f, w, h):
    bw, bh = b[2] - b[0], b[3] - b[1]
    d = rng.uniform(-f, f, 4) * (bw, bh, bw, bh)
    return (max(b[0] + d[0], 0), max(b[1] + d[1], 0), min(b[2] + d[2], w - 1), min(b[3] + d[3], h - 1))


def long_side(mask):
    pts = np.argwhere(mask)[:, ::-1].astype(np.float32)
    if len(pts) < 5:
        return 0.0
    (_, (a, b), _) = cv2.minAreaRect(pts)
    return float(max(a, b))


def stairs():
    jobs = []
    for pid in T["stairs"]:
        for _, q in data[pid][1]["stairs"][:3]:
            if len(q) >= 4:
                for pre in ("raw", "tn"):
                    for kind in ("BOX", "JBOX", "P"):
                        jobs.append((pid, q, pre, kind))

    def one(j):
        pid, q, pre, kind = j
        im = c.PRE[pre](data[pid][0])
        v, off, s = crop(im, q, 1.8)
        q2 = (q - off) * s
        h, w = v.shape[:2]
        b = bbox(q2)
        b = (max(b[0], 0), max(b[1], 0), min(b[2], w - 1), min(b[3], h - 1))
        o = dict(box=b) if kind == "BOX" else dict(box=jitter(b, 0.1, w, h)) if kind == "JBOX" else dict(pos=[rep_point(q2)])
        r = c.sam(c.obj_prompt([o], w, h), v)
        gt = poly_mask(q2, v.shape)
        mk = r[0][3] if r else np.zeros(gt.shape, bool)
        bm = np.zeros(gt.shape, bool)
        bm[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
        return dict(pid=pid, cfg=f"{pre}/{kind}", iou=c.iou(c.fill_holes(mk), gt) if mk.any() else 0.0, box_iou=c.iou(bm, gt), empty=not mk.any()), v, mk, gt
    return c.pmap(one, jobs)


def doors():
    jobs = []
    for pid in T["plans"]:
        for _, q in data[pid][1]["doors"][:4]:
            for kind in ("P", "BOXSWING"):
                jobs.append((pid, q, kind))

    def one(j):
        pid, q, kind = j
        v, off, s = crop(data[pid][0], q, 4.0)
        q2 = (q - off) * s
        h, w = v.shape[:2]
        L = max(np.ptp(q2[:, 0]), np.ptp(q2[:, 1]))
        cx, cy = q2.mean(0)
        o = dict(pos=[(cx, cy)]) if kind == "P" else dict(box=(max(cx - L * 1.1, 0), max(cy - L * 1.1, 0), min(cx + L * 1.1, w - 1), min(cy + L * 1.1, h - 1)))
        r = c.sam(c.obj_prompt([o], w, h), v)
        gt = poly_mask(q2, v.shape)
        mk = r[0][3] if r else np.zeros(gt.shape, bool)
        return dict(pid=pid, cfg=f"door/{kind}", L=float(L), got=long_side(mk), err=abs(long_side(mk) - float(L)) / float(L), empty=not mk.any()), v, mk, gt
    return c.pmap(one, jobs)


if __name__ == "__main__":
    a = stairs()
    (OUT / "stairs.json").write_text(json.dumps([x[0] for x in a]))
    d = {}
    for x in a:
        d.setdefault(x[0]["cfg"], []).append(x[0])
    for k, v in d.items():
        i = np.array([x["iou"] for x in v])
        print(k, "n", len(v), "median IoU", round(float(np.median(i)), 3), ">=.7", round(float((i >= .7).mean()), 3), "box itself median", round(float(np.median([x["box_iou"] for x in v])), 3), "empty", np.mean([x["empty"] for x in v]))
    c.sheet([(c.overlay(x[1], x[2], x[3]), f'{x[0]["cfg"]} {x[0]["iou"]:.2f}') for x in a if x[0]["cfg"] == "raw/BOX"][:12], OUT / "stairs_sheet.jpg")
    b = doors()
    (OUT / "doors.json").write_text(json.dumps([x[0] for x in b]))
    d = {}
    for x in b:
        d.setdefault(x[0]["cfg"], []).append(x[0])
    for k, v in d.items():
        e = np.array([x["err"] for x in v])
        print(k, "n", len(v), "median width error", round(float(np.median(e)), 3), "within 15%", round(float((e <= .15).mean()), 3), "empty", np.mean([x["empty"] for x in v]))
    c.sheet([(c.overlay(x[1], x[2], x[3]), f'{x[0]["cfg"]} err {x[0]["err"]:.2f}') for x in b][:16], OUT / "doors_sheet.jpg")
    print("spent", round(c.m.spent("samcraft"), 3))
