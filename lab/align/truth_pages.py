"""Pick up to 8 spread landmarks per building from the seeds (more candidates) and draw a near 1:1 page to verify by eye."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/align")
import reg
import truth_seed as ts

ts.autos.__defaults__ = (5, 4)
for bid in ts.REF:
    r = ts.seed(bid)
    n = len(reg.INDEX[bid]) - 1
    scored = sorted(r.items(), key=lambda kv: -np.median([v["ncc"] for v in kv[1]["floors"].values()] or [0]) * (len(kv[1]["floors"]) == n))
    keep = {}
    for name, l in scored:
        if len(keep) >= 8:
            break
        if all(np.hypot(l["ref"][0] - k["ref"][0], l["ref"][1] - k["ref"][1]) > 170 for k in keep.values()):
            keep[name] = l
    # always show the hand-picked stairs too, they are the landmarks the product cares about
    for name, l in r.items():
        if l["kind"] == "stair" and name not in keep and len(keep) < 10:
            keep[name] = l
    json.dump(keep, open(reg.OUT / f"truth/{bid}.seed.json", "w"), indent=1)
    ts.gallery(bid, keep, reg.OUT / f"truth/{bid}.jpg", half=75, zoom=1.4)
    print(bid, list(keep))
