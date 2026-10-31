"""Room masks: every prompt kind x preprocessing x framing, scored by IoU against CubiCasa room polygons.
Boxes and points come from the ground truth, so these are upper bounds on prompt quality. Usage: rooms.py [grid|top]."""
import json
import sys

import cv2
import numpy as np

import core as c
from data import OUT, load, poly_mask, rep_point, select

T = select()
PLANS = {p: load(p) for p in T["plans"]}
ROOMS = T["rooms"]
SUB = ROOMS[::3]  # 23 rooms for the crop grid


def bbox(q):
    return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


def prompt_obj(kind, q, others, box=None):
    """others: polygons of the other rooms on the plan. box overrides the true box (for jitter tests)."""
    b = box if box is not None else bbox(q)
    pos = [rep_point(q)]
    neg = [rep_point(o) for o in others][:6]
    bw, bh = b[2] - b[0], b[3] - b[1]
    neg_out = [(b[0] - 12, b[1] + bh / 2), (b[2] + 12, b[1] + bh / 2), (b[0] + bw / 2, b[1] - 12), (b[0] + bw / 2, b[3] + 12)]
    return {"P": dict(pos=pos), "PN": dict(pos=pos, neg=neg + neg_out), "BOX": dict(box=b), "BOXPN": dict(box=b, pos=pos, neg=neg + neg_out)}[kind]


def score(mask, gt):
    filled = c.fill_holes(mask) if mask.any() else mask
    return dict(iou=c.iou(mask, gt), iou_f=c.iou(filled, gt), empty=not mask.any(), ratio=float(filled.sum() / gt.sum()))


def plan_cfg(pre, kind):
    """All sampled rooms of each plan in one request, on the whole plan."""
    rows = []

    def one(pid):
        im, g = PLANS[pid]
        rs = [r for r in ROOMS if r["pid"] == pid]
        polys = [q for _, q in g["rooms"]]
        v = c.PRE[pre](im)
        h, w = im.shape[:2]
        for ch in range(0, len(rs), 16):
            objs = [prompt_obj(kind, polys[r["i"]], [q for j, q in enumerate(polys) if j != r["i"]]) for r in rs[ch:ch + 16]]
            for o in objs:
                for k in ("pos", "neg"):
                    if k in o:
                        o[k] = [p for p in o[k] if 0 <= p[0] < w and 0 <= p[1] < h]
                if "box" in o:
                    b = o["box"]
                    o["box"] = (max(b[0], 0), max(b[1], 0), min(b[2], w - 1), min(b[3], h - 1))
            got = {oid: mk for oid, _, _, mk in c.sam(c.obj_prompt(objs, w, h), v)}
            out = []
            for i, r in enumerate(rs[ch:ch + 16], 1):
                gt = poly_mask(polys[r["i"]], im.shape)
                mk = got.get(i, np.zeros(gt.shape, bool))
                out.append(dict(pid=pid, i=r["i"], cfg=f"plan/{pre}/{kind}", walled=r["walled"], cls=r["cls"], **score(mk, gt)))
            return out
    for o in c.pmap(one, T["plans"]):
        rows += o
    return rows


def crop_of(im, q, margin=0.3, size=1024):
    x1, y1, x2, y2 = bbox(q)
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = max(int(cx - side / 2), 0), max(int(cy - side / 2), 0)
    X2, Y2 = min(int(cx + side / 2), im.shape[1]), min(int(cy + side / 2), im.shape[0])
    s = size / max(X2 - X1, Y2 - Y1)
    return (X1, Y1, X2, Y2), s


def crop_cfg(pre, kind, rooms, margin=0.3, size=1024, box_fn=None, tag=None):
    def one(r):
        im, g = PLANS[r["pid"]]
        polys = [q for _, q in g["rooms"]]
        q = polys[r["i"]]
        (X1, Y1, X2, Y2), s = crop_of(im, q, margin, size)
        v = cv2.resize(c.PRE[pre](im)[Y1:Y2, X1:X2], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = v.shape[:2]
        tf = lambda P: (np.asarray(P, float) - (X1, Y1)) * s
        q2 = tf(q)
        others = [tf(o) for j, o in enumerate(polys) if j != r["i"] and X1 < rep_point(o)[0] < X2 and Y1 < rep_point(o)[1] < Y2]
        box = box_fn(r, q2, w, h) if box_fn else None
        o = prompt_obj(kind, q2, others, box)
        for k in ("pos", "neg"):
            if k in o:
                o[k] = [p for p in o[k] if 0 <= p[0] < w and 0 <= p[1] < h]
        if "box" in o:
            b = o["box"]
            o["box"] = (max(b[0], 0), max(b[1], 0), min(b[2], w - 1), min(b[3], h - 1))
        res = c.sam(c.obj_prompt([o], w, h), v)
        gt = poly_mask(q2, v.shape)
        mk = res[0][3] if res else np.zeros(gt.shape, bool)
        return dict(pid=r["pid"], i=r["i"], cfg=tag or f"crop{size}/{pre}/{kind}", walled=r["walled"], cls=r["cls"], **score(mk, gt))
    return c.pmap(one, rooms)


def baselines(rooms=ROOMS):
    """No API. The box itself, and the sealed region under the point, at several seal radii."""
    rows = []
    labs = {}
    for r in rooms:
        im, g = PLANS[r["pid"]]
        q = g["rooms"][r["i"]][1]
        gt = poly_mask(q, im.shape)
        x1, y1, x2, y2 = (int(round(v)) for v in bbox(q))
        bm = np.zeros(gt.shape, bool)
        bm[y1:y2 + 1, x1:x2 + 1] = True
        rows.append(dict(pid=r["pid"], i=r["i"], cfg="base/box_itself", walled=r["walled"], cls=r["cls"], **score(bm, gt)))
        for rad in (6, 10, 14, 20, 28):
            key = (r["pid"], rad)
            if key not in labs:
                labs[key] = c.regions(im, rad)[0]
            lab = labs[key]
            x, y = (int(v) for v in rep_point(q))
            v = lab[y, x]
            if v == 0:
                w = lab[max(y - 15, 0):y + 15, max(x - 15, 0):x + 15]
                w = w[w > 0]
                v = np.bincount(w).argmax() if w.size else -1
            rows.append(dict(pid=r["pid"], i=r["i"], cfg=f"base/seal{rad}", walled=r["walled"], cls=r["cls"], **score(lab == v, gt)))
    return rows


def table(rows):
    cfgs = {}
    for r in rows:
        cfgs.setdefault(r["cfg"], []).append(r)
    out = []
    for k, v in cfgs.items():
        a = np.array([x["iou_f"] for x in v])
        w = np.array([x["iou_f"] for x in v if x["walled"]])
        out.append(dict(cfg=k, n=len(v), med=round(float(np.median(a)), 3), ge5=round(float((a >= .5).mean()), 3), ge7=round(float((a >= .7).mean()), 3), ge8=round(float((a >= .8).mean()), 3),
                        ge9=round(float((a >= .9).mean()), 3), walled_n=len(w), walled_ge7=round(float((w >= .7).mean()), 3) if len(w) else None, empty=round(float(np.mean([x["empty"] for x in v])), 3),
                        leak=round(float(np.mean([x["ratio"] > 2 for x in v])), 3)))
    return sorted(out, key=lambda x: -x["ge7"])


if __name__ == "__main__":
    step = sys.argv[1]
    rows = []
    if step == "grid":
        rows += baselines()
        print("baselines done", flush=True)
        for pre in ("raw", "inv", "tn", "dt", "fill", "photo"):
            for kind in ("P", "PN", "BOX"):
                rows += plan_cfg(pre, kind)
                print(pre, kind, flush=True)
        print("plan done", flush=True)
        (OUT / "rooms_grid.json").write_text(json.dumps(rows))
        for pre in ("raw", "tn", "dt", "fill"):
            for kind in ("P", "PN", "BOX"):
                rows += crop_cfg(pre, kind, SUB)
                print(pre, kind, flush=True)
        (OUT / "rooms_grid.json").write_text(json.dumps(rows))
    for t in table(rows):
        print(t)
    print("spent", round(c.m.spent("samcraft"), 3))
