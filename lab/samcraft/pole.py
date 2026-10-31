"""Why points fail: same rooms, the point moved off printed text to the emptiest pixel of the room, alone and with 3 more empty points."""
import json

import cv2
import numpy as np
from scipy import ndimage as ndi

import core as c
import rooms as R
from data import OUT, ink_of, poly_mask


def poles(gt, ink, n):
    d = ndi.distance_transform_edt(~ink & gt)
    pts = []
    for _ in range(n):
        y, x = np.unravel_index(d.argmax(), d.shape)
        if d[y, x] < 3:
            break
        pts.append((float(x), float(y)))
        yy, xx = np.ogrid[:d.shape[0], :d.shape[1]]
        d[(yy - y) ** 2 + (xx - x) ** 2 < (2 * d[y, x]) ** 2] = 0
    return pts or [(float(np.argwhere(gt)[:, 1].mean()), float(np.argwhere(gt)[:, 0].mean()))]


rows = []
for n in (1, 4):
    def one(pid):
        im, g = R.PLANS[pid]
        h, w = im.shape[:2]
        ink = ink_of(im)
        rs = [r for r in R.ROOMS if r["pid"] == pid]
        gts = [poly_mask(g["rooms"][r["i"]][1], im.shape) for r in rs]
        objs = [dict(pos=poles(gt, ink, n)) for gt in gts]
        got = {o: mk for o, _, _, mk in c.sam(c.obj_prompt(objs, w, h), im)}
        return [dict(pid=pid, i=r["i"], n=n, iou=c.iou(c.fill_holes(got[k + 1]), gt) if k + 1 in got else 0.0, ratio=float(got[k + 1].sum() / gt.sum()) if k + 1 in got else 0.0) for k, (r, gt) in enumerate(zip(rs, gts))]
    x = sum(c.pmap(one, R.T["plans"]), [])
    rows += x
    a = np.array([r["iou"] for r in x])
    print("points", n, "n", len(a), "median IoU", round(float(np.median(a)), 3), ">=.5", round(float((a >= .5).mean()), 3), ">=.7", round(float((a >= .7).mean()), 3), ">=.8", round(float((a >= .8).mean()), 3), "empty", round(float(np.mean([r["ratio"] == 0 for r in x])), 3), "leak x2", round(float(np.mean([r["ratio"] > 2 for r in x])), 3))
(OUT / "pole.json").write_text(json.dumps(rows))
print("spent", round(c.m.spent("samcraft"), 3))
