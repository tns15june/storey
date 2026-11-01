"""Dev only, no API. For each planted cell in the dev stacks: votes in the row at that place, and the ink ratio at two box sizes."""
import sys

sys.path.insert(0, "lab/presence")
import numpy as np

import cells as C
import planted
import pres
from pres import T, ctr, box_dist

keys = planted.DEV_KEYS if len(sys.argv) < 2 else planted.HELD_KEYS
for key, (b, cases) in keys.items():
    rows, usd, secs = C.rows(b, key)
    order = list(range(pres.nfloors(b)))
    p = dict(zip(order, planted.paths(b, key, order)))
    print("==", key, "rows", len(rows), "usd", round(usd, 3))
    for kind, o, f in cases:
        bx = T[b]["objects"][o]["boxes"][str(f)]
        # the row holding this stair on other floors
        best = None
        for r in rows:
            if r["fam"] != "stair":
                continue
            ds = [box_dist(c["loc"], T[b]["objects"][o]["boxes"][str(g)]) for g, c in r["cells"].items() if c["loc"] and str(g) in T[b]["objects"][o]["boxes"]]
            if ds and (best is None or np.median(ds) < best[0]):
                best = (float(np.median(ds)), r)
        d, r = best
        ra, de = C.ink_ratios(r, p)
        ra2, _ = C.ink_ratios(r, p, 30)
        print(f" {kind:9s} {o:10s} f{f} row at {[round(x) for x in r['pos']]} d{d:.0f} votes {[c['v'] for c in r['cells'].values()]} here v{r['cells'][f]['v']} ink50 {ra[f]} ink30 {ra2[f]} names {r['names'][:2]}")
