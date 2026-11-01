"""Comparator: Round 1 one-call stair list on the held-out frames. Strict: every listed stair within 120 px of a truth stair must be right; best-listed beside it, marked. Cached."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *

ORDER = {"bme_k": "lower ground, ground, first, second", "habs_va_bldg1": "ground, first, second, third, fourth", "wfu_huffman": "ground, first, second, third",
         "queens_royce1": "first, second, third, fourth", "leonard4": "first, second, third, fourth", "mml_abbot": "foundation, main level, upper level, attic and roof"}
P = """These images are floor plans of one building, one image per floor, in this order: {order}. Staircases in the same place on consecutive floors are the same staircase.
List every distinct staircase in the building. For each give a short name, and for every floor in the list say whether it is drawn on that floor (present) and its centre x, y in the 0-1000 range of that floor's image (0, 0 if absent). A stair drawn struck through or filled in counts as absent. Compare the floors carefully, stair by stair."""
STR = {"type": "string"}; INT = {"type": "integer"}
def obj(**k): return {"type": "object", "additionalProperties": False, "required": list(k), "properties": k}
def arr(i): return {"type": "array", "items": i}
S = obj(stairs=arr(obj(name=STR, floors=arr(obj(floor=STR, present={"type": "boolean"}, x=INT, y=INT)))))


def call(st):
    p = SPK / f"{st}_floors.json"
    if p.exists():
        return json.load(open(p))
    ims = [str(R / "crops" / f"{st}_f{i}.png") for i in range(TRUTH[st]["n"])]
    r, info = m.spark(P.format(order=ORDER[st]), images=ims, schema=S, effort="medium", max_tokens=20000)
    o = dict(out=r, info=info); json.dump(o, open(p, "w")); print(st, round(info["usd"], 4), flush=True)
    return o


def score_floors(st):
    T = TRUTH[st]; n = T["n"]; x1, y1, x2, y2 = T["crop"]; w, h = x2 - x1, y2 - y1
    ls = call(st)["out"]["stairs"]
    rows = []
    for k, s in T["stairs"].items():
        if s["cls"] == "amb":
            continue
        bx = {int(f): b for f, b in s["boxes"].items()}
        near = []
        for l in ls:
            if len(l["floors"]) != n:
                continue
            d = [np.hypot(l["floors"][f]["x"] * w / 1000 + x1 - (bx[f][0] + bx[f][2]) / 2, l["floors"][f]["y"] * h / 1000 + y1 - (bx[f][1] + bx[f][3]) / 2) for f in bx if l["floors"][f]["present"]]
            if d and np.median(d) < 120:
                near.append((float(np.median(d)), l))
        near.sort(key=lambda z: z[0])
        for f in range(n):
            if f in s.get("exclude", ()):
                continue
            say = [bool(l["floors"][f]["present"]) for _, l in near]
            strict = bool(near) and all(v == (f in bx) for v in say)
            best = bool(near) and say[0] == (f in bx)
            rows.append(dict(stack=st, stair=k, cls=s["cls"], floor=f, present=f in bx, seeded=bool(near), ok=strict, best=best, ghosts=sum(say) if f not in bx else 0, claims=[]))
    return rows


if __name__ == "__main__":
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(call, HELD))
    allr = []
    for st in HELD:
        r = score_floors(st); allr += r
        print(st, "strict", summary(r), "best-listed pairs", sum(x["best"] for x in r if x["cls"] in SCORED), "/", sum(x["cls"] in SCORED for x in r), [(x["stair"], x["floor"], x["present"]) for x in r if not x["ok"] and x["cls"] in SCORED])
    print("POOLED strict", summary(allr), "best-listed", sum(x["best"] for x in allr if x["cls"] in SCORED), "/", sum(x["cls"] in SCORED for x in allr), "best-listed absent", sum(x["best"] for x in allr if x["cls"] in SCORED and not x["present"]), "/", sum(x["cls"] in SCORED and not x["present"] for x in allr))
    print("spent", round(m.spent("tightseed"), 4))
