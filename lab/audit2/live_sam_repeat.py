"""L1: repeat the exact P and Ppin requests and read the absent floors that sat close to the .5 line. Spends about 0.09 USD. Results cached in data/lab/audit2/sam."""
import json
import sys

sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
import core

SRC = core.SAMD
AUD = core.ROOT / "data/lab/audit2"
core.SAMD = AUD / "sam"
core.SAMD.mkdir(exist_ok=True)
JOBS = (("bme_k", "P_960", "svc_SW", 2), ("queens_royce1", "P_960", "R_G", 2), ("mml_abbot", "Ppin_960", "ext_stair", 3))


def owners(st, tr, k):
    return sorted({str(o) for o, f, b in tr["seeds"] if any(core.hit(b, x) for x in core.TRUTH[st]["stairs"][k]["boxes"].values())})


for st, me, k, f in JOBS:
    base = json.load(open(SRC / f"{st}__{me}.json"))
    seeds = [(o, g, b) for o, g, b in base["seeds"]]
    ow = owners(st, base, k)
    print(st, me, k, "floor", f, "cached", [base["tracks"].get(o, {}).get(str(f), {}).get("conf") for o in ow], "seeds", len(seeds), flush=True)
    for r in range(3):
        tr = core.track(f"{st}__{me}__rep{r}", st, seeds)
        print("   repeat", r, [tr["tracks"].get(o, {}).get(str(f), {}).get("conf") for o in ow], "all floors", {o: {g: v["conf"] for g, v in tr["tracks"].get(o, {}).items()} for o in ow}, flush=True)
print("spent", round(core.m.spent("audit2"), 4))
