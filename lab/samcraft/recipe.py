"""Does SAM add anything to a box? Noisy boxes, non-rectangular rooms, and real Muse Spark boxes, all scored against the box alone."""
import json
import sys

import cv2
import numpy as np

import core as c
import rooms as R
from data import OUT, poly_mask

SPARK = {"type": "object", "additionalProperties": False, "required": ["rooms"], "properties": {"rooms": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["label", "x1", "y1", "x2", "y2"],
    "properties": {"label": {"type": "string"}, "x1": {"type": "integer"}, "y1": {"type": "integer"}, "x2": {"type": "integer"}, "y2": {"type": "integer"}}}}}}


def box_mask(b, shape):
    m = np.zeros(shape[:2], bool)
    m[max(int(b[1]), 0):int(b[3]) + 1, max(int(b[0]), 0):int(b[2]) + 1] = True
    return m


def noisy(f):
    """Each edge of the true box moved by up to f of the room size, as a model box would be. Same seed per room."""
    rows = []

    def one(pid):
        im, g = R.PLANS[pid]
        h, w = im.shape[:2]
        rs = [r for r in R.ROOMS if r["pid"] == pid]
        objs, gts = [], []
        for r in rs:
            q = g["rooms"][r["i"]][1]
            b = np.array(R.bbox(q))
            rng = np.random.default_rng(int(pid) * 100 + r["i"])
            d = rng.uniform(-f, f, 4) * np.array([b[2] - b[0], b[3] - b[1], b[2] - b[0], b[3] - b[1]])
            j = b + d
            objs.append(dict(box=(max(j[0], 0), max(j[1], 0), min(j[2], w - 1), min(j[3], h - 1))))
            gts.append(poly_mask(q, im.shape))
        got = {o: mk for o, _, _, mk in c.sam(c.obj_prompt(objs, w, h), im)}
        return [dict(pid=pid, i=r["i"], walled=r["walled"], box=c.iou(box_mask(o["box"], im.shape), gt), sam=c.iou(c.fill_holes(got[k + 1]), gt) if k + 1 in got else 0.0)
                for k, (r, o, gt) in enumerate(zip(rs, objs, gts))]
    for x in c.pmap(one, R.T["plans"]):
        rows += x
    return rows


def spark(pid):
    im, g = R.PLANS[pid]
    h, w = im.shape[:2]
    d, info = c.m.spark("Find every room in this floor plan. Give one box per room: x1, y1, x2, y2 on a 0-1000 grid of this image, x from the left edge, y from the top edge. "
                        "The box must touch the room's own walls. Include closets, bathrooms and corridors. Do not box the whole plan.", [im], schema=SPARK, effort="low")
    boxes = [(b["x1"] * w / 1000, b["y1"] * h / 1000, b["x2"] * w / 1000, b["y2"] * h / 1000) for b in d["rooms"] if b["x2"] > b["x1"] and b["y2"] > b["y1"]]
    masks = []
    for ch in range(0, len(boxes), 16):
        objs = [dict(box=(max(b[0], 0), max(b[1], 0), min(b[2], w - 1), min(b[3], h - 1))) for b in boxes[ch:ch + 16]]
        got = {o: mk for o, _, _, mk in c.sam(c.obj_prompt(objs, w, h), im)}
        masks += [got.get(i + 1, np.zeros(im.shape[:2], bool)) for i in range(len(objs))]
    rows = []
    bm = [box_mask(b, im.shape) for b in boxes]
    fm = [c.fill_holes(m) for m in masks]
    tiles = []
    for r in [r for r in R.ROOMS if r["pid"] == pid]:
        gt = poly_mask(g["rooms"][r["i"]][1], im.shape)
        if not boxes:
            rows.append(dict(pid=pid, i=r["i"], walled=r["walled"], box=0.0, sam=0.0))
            continue
        k = int(np.argmax([c.iou(b, gt) for b in bm]))
        rows.append(dict(pid=pid, i=r["i"], walled=r["walled"], box=c.iou(bm[k], gt), sam=c.iou(fm[k], gt), n_boxes=len(boxes), usd=info["usd"], seconds=info["seconds"]))
        tiles.append((c.overlay(im, fm[k], gt), f"{pid} box {rows[-1]['box']:.2f} sam {rows[-1]['sam']:.2f}"))
    return rows, tiles


def show(name, rows):
    b = np.array([x["box"] for x in rows])
    s = np.array([x["sam"] for x in rows])
    print(name, "n", len(rows), "box median", round(float(np.median(b)), 3), ">=.7", round(float((b >= .7).mean()), 3), ">=.8", round(float((b >= .8).mean()), 3),
          "| sam median", round(float(np.median(s)), 3), ">=.7", round(float((s >= .7).mean()), 3), ">=.8", round(float((s >= .8).mean()), 3), "| sam better on", round(float((s > b).mean()), 3), flush=True)


if __name__ == "__main__":
    res = {}
    for f in (0.1, 0.2):
        res[f"noise{f}"] = noisy(f)
        show(f"noise {f}", res[f"noise{f}"])
    # Non-rectangular rooms, from the grid already scored: rooms the true box fits badly
    g = json.loads((OUT / "rooms_grid.json").read_text())
    base = {(r["pid"], r["i"]): r["iou_f"] for r in g if r["cfg"] == "base/box_itself"}
    sam = {(r["pid"], r["i"]): r["iou_f"] for r in g if r["cfg"] == "plan/raw/BOX"}
    nr = [dict(box=base[k], sam=sam[k]) for k in base if base[k] < 0.9 and k in sam]
    show("non-rectangular rooms, true box", nr)
    show("all rooms, true box", [dict(box=base[k], sam=sam[k]) for k in base if k in sam])
    if len(sys.argv) > 1:
        plans = R.T["plans"][:int(sys.argv[1])]
        out = c.pmap(spark, plans, n=3)
        res["spark"] = sum((o[0] for o in out), [])
        show("Spark boxes", res["spark"])
        show("Spark boxes, walled", [x for x in res["spark"] if x["walled"]])
        print("spark boxes per plan", [o[0][0].get("n_boxes") for o in out if o[0]], "usd", round(sum(o[0][0].get("usd", 0) for o in out if o[0]), 3))
        c.sheet(sum((o[1] for o in out), [])[:16], OUT / "spark_sam_sheet.jpg")
    (OUT / "recipe.json").write_text(json.dumps(res, default=float))
    print("spent", round(c.m.spent("samcraft"), 3))
