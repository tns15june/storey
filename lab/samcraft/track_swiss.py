"""Tracker on rendered Swiss Dwellings stacks: stairs, corridors, shafts, lifts and columns floor to floor, with exact truth."""
import json

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box as sbox

import core as c
import swiss as S
from data import OUT

REPEAT = 4


def tf(g, ox, oy, shift):
    return (np.array(g.exterior.coords) - (ox, oy)) * S.PX + shift


def run(b, kinds, start, jitter=30, keep=None, tag="", tol=0.3):
    """kinds: list of (name, subtypes, max objects). Prompts on floor `start`. Truth on another floor is the same-subtype polygon with the largest overlap."""
    fls, (ox, oy, W, H), shifts = S.stack(b, keep=keep, jitter=jitter)
    fr = [S.render(f, ox, oy, W, H, s) for f, s in zip(fls, shifts)]
    scale = min(1, 1400 / max(W, H))
    if scale < 1:
        fr = [cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in fr]
    path = OUT / f"sw_{b}{tag}.mp4"
    w, h = c.encode(fr, path, repeat=REPEAT)
    objs, meta = [], []
    for name, subs, mx in kinds:
        gs = sorted(S.geom(fls[start], *subs), key=lambda g: -g.area)[:mx]
        for g in gs:
            q = tf(g, ox, oy, shifts[start]) * scale
            x1, y1, x2, y2 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
            objs.append(dict(id=len(objs) + 1, box=(max(x1 - 3, 0), max(y1 - 3, 0), min(x2 + 3, w - 1), min(y2 + 3, h - 1))))
            meta.append((name, subs, g))
    frames = len(fr) * REPEAT
    res = c.video(c.obj_prompt(objs, w, h, frame=start * REPEAT), path, frames)
    rows = []
    tiles = []
    for fi, f in enumerate(fls):
        last = fi * REPEAT + REPEAT - 1
        v = fr[fi].copy()
        for oi, (name, subs, g0) in enumerate(meta, 1):
            cand = S.geom(f, *subs)
            best = max(cand, key=lambda g: g.intersection(g0).area, default=None)
            near = best is not None and best.intersection(g0).area > 0.2 * min(best.area, g0.area)
            if not near:  # same subtype nearby, for columns that moved a little
                d = [(g.distance(g0), g) for g in cand]
                if d and min(x[0] for x in d) < tol:
                    best, near = min(d, key=lambda x: x[0])[1], True
            rec = res.get((last, oi))
            mk = c.paste(rec[0], rec[2], fr[fi].shape) if rec else np.zeros(fr[fi].shape[:2], bool)
            if near:
                gt = np.zeros(fr[fi].shape[:2], np.uint8)
                cv2.fillPoly(gt, [np.round(tf(best, ox, oy, shifts[fi]) * scale).astype(np.int32)], 1)
                gt = gt.astype(bool)
            else:
                gt = np.zeros(fr[fi].shape[:2], bool)
            foot = np.zeros(fr[fi].shape[:2], np.uint8)
            cv2.fillPoly(foot, [np.round(tf(g0, ox, oy, shifts[fi]) * scale).astype(np.int32)], 1)
            drawn = float((fr[fi][foot.astype(bool)].mean(1) < 128).mean())  # is the prompted footprint solid ink on this floor
            rows.append(dict(b=b, tag=tag, floor=fi, prompted=fi == start, obj=oi, name=name, truth=bool(near), drawn=drawn, got=bool(mk.any()), iou=c.iou(c.fill_holes(mk), gt) if near else None,
                             conf=rec[1] if rec else None, centre_px=float(np.hypot(*(np.argwhere(mk).mean(0) - np.argwhere(gt).mean(0)))) if near and mk.any() else None))
            if mk.any():
                v = c.overlay(v, mk, gt if near else None, color=(0, 0, 255) if near else (255, 0, 255))
        tiles.append((v, f"{b}{tag} floor {fi}{' PROMPT' if fi == start else ''}"))
    c.sheet(tiles, OUT / f"sw_{b}{tag}.jpg", cols=4, w=480)
    return rows


def summary(rows):
    d = {}
    for r in rows:
        if r["prompted"]:
            continue
        d.setdefault((r["b"], r["tag"], r["name"]), []).append(r)
    for k, v in d.items():
        t = [x for x in v if x["truth"]]
        tp = sum(x["got"] for x in t)
        fp = sum(x["got"] for x in v if not x["truth"])
        ious = [x["iou"] for x in t if x["got"]]
        print(k, "pairs", len(v), "present", len(t), "kept", tp, "absent", len(v) - len(t), "false", fp, "median IoU", round(float(np.median(ious)), 3) if ious else None,
              "median centre px", round(float(np.median([x["centre_px"] for x in t if x["got"]])), 1) if ious else None)
    col = [r for r in rows if r["name"] == "column" and not r["prompted"]]
    ink = [r for r in col if r["drawn"] >= .5]
    bare = [r for r in col if r["drawn"] < .5]
    print("columns, truth is ink under the footprint: ink pairs", len(ink), "kept", sum(r["got"] for r in ink), "bare pairs", len(bare), "false", sum(r["got"] for r in bare))
    for b in sorted({r["b"] for r in col}):
        i = [r for r in ink if r["b"] == b]
        z = [r for r in bare if r["b"] == b]
        print(" ", b, "ink", len(i), "kept", sum(r["got"] for r in i), "bare", len(z), "false", sum(r["got"] for r in z))


if __name__ == "__main__":
    KINDS = [("stair", ("STAIRCASE",), 2), ("flight", ("STAIRS",), 2), ("corridor", ("CORRIDOR", "LOBBY"), 3), ("shaft", ("SHAFT",), 2), ("lift", ("ELEVATOR",), 1), ("column", ("COLUMN",), 6)]
    rows = []
    rows += run("15364", KINDS, 1)
    rows += run("17990", KINDS, 2)
    rows += run("18612", KINDS, 1)
    rows += run("16172", [("column", ("COLUMN",), 16)], 3, keep=(2, 6), tag="_col")
    (OUT / "track_swiss.json").write_text(json.dumps(rows))
    summary(rows)
    print("spent", round(c.m.spent("samcraft"), 3))
