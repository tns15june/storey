"""Dev only, no API. Print rows, votes, ink ratios and where each row sits against truth, to set radius, tolerance and the ink threshold."""
import sys

sys.path.insert(0, "lab/presence")
import numpy as np

import cells as C
import pres
from pres import T, SH, box_dist, DEV, fam


def nearest(b, f, loc, family):
    """Closest truth object of the family drawn on this floor, or None."""
    best = (None, 1e9, None)
    for o, v in T[b]["objects"].items():
        if fam(v["kind"]) != family or str(f) not in v["boxes"]:
            continue
        d = box_dist(loc, v["boxes"][str(f)])
        if d < best[1]:
            best = (o, d, str(f) in map(str, v.get("unsure", [])))
    return best


for b in (sys.argv[1:] or DEV):
    rows, usd, secs = C.rows(b, b)
    paths = {f: SH / f"{b}/{f}.jpg" for f in range(pres.nfloors(b))}
    print("==", b, "rows", len(rows), "usd", round(usd, 3), "s", [round(s) for s in secs])
    for r in rows:
        ra, d = C.ink_ratios(r, paths)
        cs = []
        for f, c in r["cells"].items():
            n = nearest(b, f, c["loc"], r["fam"]) if c["loc"] else (None, 1e9, None)
            cs.append(f"f{f}:v{c['v']} r{ra[f]} {n[0]}:{n[1]:.0f}" if c["loc"] else f"f{f}:v{c['v']} noloc")
        print(f" {r['fam'][:5]} runs{r['runs']} pos{None if r['pos'] is None else [round(x) for x in r['pos']]} {r['names'][:2]} | " + " | ".join(cs))
