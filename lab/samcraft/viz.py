"""Contact sheets of point failures, read from cache only."""
import numpy as np

import core as c
import rooms as R
from data import OUT, poly_mask, rep_point

t = []
for r in R.ROOMS[::6]:
    im, g = R.PLANS[r["pid"]]
    h, w = im.shape[:2]
    rs = [x for x in R.ROOMS if x["pid"] == r["pid"]]
    polys = [q for _, q in g["rooms"]]
    objs = [R.prompt_obj("P", polys[x["i"]], []) for x in rs]
    got = {o: mk for o, _, _, mk in c.sam(c.obj_prompt(objs, w, h), im)}
    k = rs.index(r) + 1
    gt = poly_mask(polys[r["i"]], im.shape)
    mk = got.get(k, np.zeros(gt.shape, bool))
    v = c.overlay(im, mk, gt)
    t.append((v, f"plan/raw/P iou {c.iou(c.fill_holes(mk), gt):.2f} area x{mk.sum() / gt.sum():.2f}"))
c.sheet(t[:12], OUT / "point_fail_sheet.jpg")
