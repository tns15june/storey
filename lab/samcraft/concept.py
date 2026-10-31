"""Why concept prompts fail: prompt wording x preprocessing x scale, scored against CubiCasa polygons.
A hit is a returned mask with IoU >= 0.5 to a true object of the matching class, or box overlap for doors."""
import json
import zipfile

import cv2
import numpy as np

import core as c
from data import OUT, ROOT, load, poly_mask, select

T = select()
WORDS = ["room", "stairs", "staircase", "staircase symbol", "door", "wall", "parallel lines", "rectangle", "empty space", "hallway"]
CLASS = {"room": "rooms", "empty space": "rooms", "rectangle": "rooms", "hallway": "rooms", "stairs": "stairs", "staircase": "stairs", "staircase symbol": "stairs", "parallel lines": "stairs", "door": "doors"}
PLANS = T["plans"][:3] + T["stairs"][:3]
data = {p: load(p) for p in PLANS}


def tile(im, g, cls, size):
    """A native-size window of `size` px centred on the first object of the class that fits."""
    objs = g[cls] or g["rooms"]
    q = objs[0][1]
    cx, cy = q.mean(0)
    x1 = int(np.clip(cx - size / 2, 0, max(im.shape[1] - size, 0)))
    y1 = int(np.clip(cy - size / 2, 0, max(im.shape[0] - size, 0)))
    return x1, y1, min(x1 + size, im.shape[1]), min(y1 + size, im.shape[0])


def run(pid, word, pre, frame):
    im, g = data[pid]
    cls = CLASS.get(word, "rooms")
    if frame == "whole":
        win = (0, 0, im.shape[1], im.shape[0])
        s = 1.0
    else:
        size, s = {"w512": (512, 1.0), "w1024": (1024, 1.0), "up512x2": (512, 2.0), "up256x4": (256, 4.0)}[frame]
        win = tile(im, g, cls, size)
    v = c.PRE[pre](im)[win[1]:win[3], win[0]:win[2]]
    if s != 1:
        v = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    res = c.sam(word, v)
    off = np.array(win[:2], float)
    gts = [poly_mask((q - off) * s, v.shape) for _, q in g[cls]]
    gts = [m for m in gts if m.sum() > 200]
    hits = 0
    for _, _, _, mk in res:
        if any(c.iou(mk, gt) >= 0.5 for gt in gts):
            hits += 1
    found = sum(any(c.iou(mk, gt) >= 0.5 for _, _, _, mk in res) for gt in gts)
    return dict(pid=pid, word=word, pre=pre, frame=frame, n=len(res), hits=hits, found=found, gt=len(gts), conf=max([r[2] or 0 for r in res], default=None))


if __name__ == "__main__":
    jobs = [(p, w, pre, "w1024") for p in PLANS[:2] + PLANS[3:4] for w in WORDS for pre in ("raw", "inv", "thick", "notext", "blue", "fill", "photo")]
    jobs += [(p, w, "raw", f) for p in PLANS for w in ("room", "stairs", "door", "rectangle") for f in ("whole", "w512", "up512x2", "up256x4")]
    rows = c.pmap(lambda j: run(*j), jobs)
    (OUT / "concept.json").write_text(json.dumps(rows))
    print(len(rows), "calls, spent", round(c.m.spent("samcraft"), 3))
    def agg(key):
        d = {}
        for r in rows:
            d.setdefault(key(r), []).append(r)
        for k, v in sorted(d.items(), key=lambda x: -sum(a["n"] for a in x[1])):
            print(k, "calls", len(v), "any", sum(a["n"] > 0 for a in v), "masks", sum(a["n"] for a in v), "true", sum(a["hits"] for a in v), "found", sum(a["found"] for a in v), "of", sum(a["gt"] for a in v))
    agg(lambda r: r["word"]); print(); agg(lambda r: r["pre"]); print(); agg(lambda r: r["frame"]); print(); agg(lambda r: (r["word"], r["pre"], r["frame"]))
