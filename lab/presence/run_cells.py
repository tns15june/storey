"""Build the evidence for every cell of a stack: votes, ink at the registered place, and the crop answer. Writes data/lab/presence/cells/<key>.json. Spends only on crops not yet cached.
Usage: run_cells.py <mode> <key ...>, mode spec = crops where runs disagree or say absent, mode all = crops on every cell of a row seen by two runs or more on some floor.
A cell whose registered place is blank paper gets no crop: nothing is drawn there, so the question cannot change the answer."""
import json
import sys

sys.path.insert(0, "lab/presence")
import cells as C
import crop
import planted
import pres
from pres import OUT, SH
import metalab as m

BLANK = 0.002
(OUT / "cells").mkdir(exist_ok=True)


def stack(key):
    if "__" in key:
        b = key.split("__")[0]
        order = list(range(pres.nfloors(b)))
        return b, dict(zip(order, planted.paths(b, key, order)))
    return key, {f: SH / f"{key}/{f}.jpg" for f in range(pres.nfloors(key))}


def evidence(key, mode):
    b, paths = stack(key)
    rows, usd, secs = C.rows(b, key)
    jobs = []
    for i, r in enumerate(rows):
        r["id"] = i
        r["established"] = max(c["v"] for c in r["cells"].values()) >= 2
        r["ink30"], d30 = C.ink_ratios(r, paths, 30)
        r["ink50"], d50 = C.ink_ratios(r, paths, 50)
        r["dens50"] = d50
        for f, c in r["cells"].items():
            c["blank"] = c["chk"] is None or d50.get(f, 0) < BLANK
            want = c["v"] < 3 if r["established"] else c["v"] >= 1
            if mode == "all" and r["established"]:
                want = True
            if want and not c["blank"]:
                jobs.append((r, f, paths[f], c["chk"], r["fam"]))
    n0 = m.spent("presence")
    ans = crop.ask_many([j[2:] for j in jobs])
    new = [a for a in ans if not a.get("cached")]
    for (r, f, *_), a in zip(jobs, ans):
        r["cells"][f]["crop"] = a["answer"]
        r["cells"][f]["seen"] = a["seen"]
    o = dict(key=key, b=b, rows=rows, compare_usd=usd, compare_seconds=secs, crops=len(jobs), crops_new=len(new), crop_usd=round(sum(a["usd"] for a in ans), 5), crop_seconds=round(sum(a["seconds"] for a in ans), 1), crop_new_usd=round(sum(a["usd"] for a in new), 5))
    json.dump(o, open(OUT / f"cells/{key}.json", "w"), indent=1, default=float)
    print(key, "rows", len(rows), "crops", len(jobs), "new", len(new), "usd new", o["crop_new_usd"], "spent", round(m.spent("presence"), 4), flush=True)
    return o


if __name__ == "__main__":
    for k in sys.argv[2:]:
        evidence(k, sys.argv[1])
