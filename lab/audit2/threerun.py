"""Truth stairs with a row seen by all 3 runs within 40 px of a truth box on a shared floor; Richmond input rule; declared frozen exit sums. No API."""
import json, math
O = "/Users/tsharma/Github/Meta/data/lab/presence/"
T = json.load(open("/Users/tsharma/Github/Meta/lab/presence/truth.json"))["buildings"]
def d(p, b): return math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))
tot = dict(all=[0, 0], main=[0, 0])
for b, t in T.items():
    if t["split"] != "heldout":
        continue
    E = json.load(open(O + f"cells/{b}.json"))
    for o, v in t["objects"].items():
        if v["kind"] not in ("stair", "stairs"):
            continue
        hit = False
        for r in E["rows"]:
            if r["fam"] != "stair" or len(r["runs"]) != 3:
                continue
            for f, bx in v["boxes"].items():
                c = r["cells"][f]
                if c["loc"] and d(c["loc"], bx) <= 40:
                    hit = True
        tot["all"][1] += 1; tot["all"][0] += hit
        if v["cls"] == "main":
            tot["main"][1] += 1; tot["main"][0] += hit
print(tot)
r = json.load(open("/Users/tsharma/Github/Meta/data/lab/realsets/va_trb/richmond_truth.json"))
print(r["input_rule"]); print(r["page_base"], r["limits"][:400] if isinstance(r["limits"], str) else r["limits"])
print(sorted({c["page"] for c in r["comments"]}))
f = json.load(open("/Users/tsharma/Github/Meta/data/lab/declared/final.json"))
print(list(f)[:12])
for k, v in f.items():
    if "exit" in k.lower() or "capac" in k.lower():
        print(k, str(v)[:500])
