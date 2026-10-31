"""The flag lost between v2a and v2, plus a seeded sample of held-out flags for a second reader. No API."""
import json, random, sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/declared")
import verdicts as V
D = "/Users/tsharma/Github/Meta/data/lab/declared/"
a = json.load(open(D + "flags_held_v2a.json")); b = json.load(open(D + "flags_held_v2.json"))
ib = {f["id"] for f in b["flags"]}
for f in a["flags"]:
    if f["id"] not in ib:
        print("LOST FLAG", json.dumps({k: f[k] for k in ("id", "sheet", "text", "source")}))
rows = [f for f in b["flags"] if f["check"] != 2]
by = {"real": [], "likely": [], "false": []}
for f in rows:
    by[V.held_verdict(f)[0]].append(f)
random.seed(20261031)
pick = random.sample(by["real"], 5) + random.sample(by["false"], 5) + random.sample(by["likely"], 2)
for f in pick:
    print(f["id"], "|", f["sheet"], "|", f["text"][:170])
json.dump([f["id"] for f in pick], open("/Users/tsharma/Github/Meta/data/lab/audit2/declared_pick.json", "w"))
