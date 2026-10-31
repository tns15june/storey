"""Run 0 alone on the 15 plants, from the raw Spark run, nearest entity claim within 40 px of the erased box. No API."""
import json, math
O = "/Users/tsharma/Github/Meta/data/lab/presence/"
TR = json.load(open("/Users/tsharma/Github/Meta/lab/presence/truth.json")); T = TR["buildings"]
IDX = json.load(open(O + "sheets/index.json"))
def d(p, b): return math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))
found = 0; rows = []
for b, t in T.items():
    if t["split"] != "heldout":
        continue
    for kind, o, f in TR["plant"][b]:
        r = json.load(open(O + f"spark/{b}__{kind}_r0.json"))
        order = r["order"]; h, w = IDX[b][f]["shape"]; bx = t["objects"][o]["boxes"][str(f)]
        pos = order.index(f) + 1
        hit = False
        for s in r["out"]["stairs"]:
            for fl in s["floors"]:
                if fl["floor"].strip().upper().startswith(f"F{pos}") and fl["present"] and d((fl["x"] * w / 1000, fl["y"] * h / 1000), bx) <= 40:
                    hit = True
        found += hit; rows.append((b[:8], kind, hit))
print("run 0 Found on erased cell", found, "of", len(rows), rows)
e = json.load(open(O + "eval_heldout.json"))["planted"]
print("frozen eval run0 Found", sum(x["run0"] == "Found" for x in e))
