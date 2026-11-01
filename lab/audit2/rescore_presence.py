"""Independent rescore of presence from cells/*.json with the rule rewritten here. No API, imports nothing from lab/presence.
Also: how sensitive is 0 of 15 to the 40 px place test, and what stood between each plant and a Found. Run: .venv/bin/python -I lab/audit2/rescore_presence.py"""
import json
import math
from collections import Counter
from pathlib import Path

OUT = Path("data/lab/presence")
TR = json.load(open("lab/presence/truth.json"))
T = TR["buildings"]
HELD = [b for b, t in T.items() if t["split"] == "heldout"]
INK_LOW, INK_HIGH = .40, .90


def decide(est, fam, ink, c, arm):
    v, crop, blank = c["v"], c.get("crop"), c.get("blank")
    ink = ink if fam == "stair" else None
    if not est:
        if v == 0 or blank or crop == "no":
            return None
        return "Check"
    if not c["aligned"]:
        return "Check"
    if v >= 2:
        guarded = v < 3 or arm in "BC"
        if guarded and ink is not None and ink < INK_LOW:
            return "Check"
        asks = v < 3 or arm == "C"
        if asks and blank:
            return "Check"
        if asks and crop in ("no", "unclear"):
            return "Check"
        return "Found"
    if blank:
        return "Missing"
    if crop in ("yes", "unclear"):
        return "Check"
    if ink is not None and ink >= INK_HIGH:
        return "Check"
    return "Missing"


def box_dist(p, b):
    return math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))


def cells_of(key, arm):
    E = json.load(open(OUT / f"cells/{key}.json"))
    out = []
    for r in E["rows"]:
        for f, c in r["cells"].items():
            out.append(dict(f=int(f), state=decide(r["established"], r["fam"], r["ink30"].get(f), c, arm), v=c["v"], crop=c.get("crop"), seen=c.get("seen"), ink=r["ink30"].get(f), blank=c.get("blank"), aligned=c["aligned"], loc=c["loc"], chk=c["chk"], row=r["id"], fam=r["fam"], names=r["names"][:2]))
    return out


def main():
    print("PLANTED, rewritten rule; D is the place test in px")
    rows = []
    for b in HELD:
        for kind, o, f in TR["plant"][b]:
            key = f"{b}__{kind}"
            bx = T[b]["objects"][o]["boxes"][str(f)]
            line = dict(key=key, obj=o, f=f, box_side=max(bx[2] - bx[0], bx[3] - bx[1]))
            for arm in "ABC":
                cs = [c for c in cells_of(key, arm) if c["fam"] == "stair" and c["f"] == f and c["state"] and (c["chk"] or c["loc"])]
                for D in (40, 60, 80, 120, 200):
                    st = [c["state"] for c in cs if (c["chk"] and box_dist(c["chk"], bx) <= D) or (c["loc"] and box_dist(c["loc"], bx) <= D)]
                    line[f"{arm}{D}"] = "Found" if "Found" in st else ("Check" if "Check" in st else ("Missing" if "Missing" in st else "no row"))
                if arm == "C":
                    near = sorted((round(min(box_dist(c["chk"], bx) if c["chk"] else 1e9, box_dist(c["loc"], bx) if c["loc"] else 1e9)), c["state"], c["v"], c["crop"], c["ink"], c["names"]) for c in cs)
                    line["near"] = near[:3]
            rows.append(line)
            print(f" {key:42s} {o:9s} f{f} side {line['box_side']:4d} | C40 {line['C40']:7s} C60 {line['C60']:7s} C80 {line['C80']:7s} C120 {line['C120']:7s} C200 {line['C200']:7s} | A40 {line['A40']:7s} B40 {line['B40']:7s}")
            for n in line["near"]:
                print("      nearest", n)
    for arm in "ABC":
        for D in (40, 60, 80, 120, 200):
            print(f" arm {arm} D {D}: wrong passes {sum(r[f'{arm}{D}'] == 'Found' for r in rows)} of {len(rows)}", Counter(r[f"{arm}{D}"] for r in rows))
    print("\nCLEAN arm C by building: Found / Check / Missing stair cells, from cells/*.json")
    tot = Counter()
    for b in HELD:
        c = Counter(x["state"] for x in cells_of(b, "C") if x["fam"] == "stair")
        tot.update(c)
        print(" ", b, dict(c))
    print(" total", dict(tot))
    print("\nhow many clean stair cells have v>=2 and rest on a single crop 'yes' as their only local confirmation")
    n = Counter()
    for b in HELD:
        for x in cells_of(b, "C"):
            if x["fam"] == "stair" and x["v"] >= 2 and x["state"]:
                n[(x["state"], x["crop"])] += 1
    print(" ", dict(n))
    json.dump(rows, open("data/lab/audit2/presence_planted.json", "w"), indent=1)


main()
