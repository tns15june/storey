"""L7 for sheet c162 only, with an empty model answer retried once and counted. Writes to data/lab/audit2."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "lab"); sys.path.insert(0, "lab/declared")
import metalab as m
import read_spark as RS
SID, k, want = "nibs_clinic", "c162", 294
s = RS.CL[k]; clips = RS.tiles(SID, s); empties = []
def one(a):
    i, c, run = a
    for t in range(2):
        try:
            r, info = m.spark(RS.EXIT_P, images=[RS.render(SID, s, 2000, c)], schema=RS.EXIT_S, effort="medium", max_tokens=6000)
            return i, run, r
        except (TypeError, json.JSONDecodeError):
            empties.append((run, i, t))
    return i, run, dict(exits=[])
with ThreadPoolExecutor(4) as ex:
    ans = list(ex.map(one, [(i, c, run) for run in (0, 1) for i, c in enumerate(clips)]))
res = {}
for run in (0, 1):
    kept, raw = [], 0
    for i, rr, r in ans:
        if rr != run:
            continue
        c = clips[i]
        for e in r["exits"]:
            x, y = c[0] + e["x"] / 1000 * (c[2] - c[0]), c[1] + e["y"] / 1000 * (c[3] - c[1])
            raw += e["capacity"] or 0
            if e["capacity"] and (x < .5) == (c[0] < .25) and (y < .5) == (c[1] < .25):
                kept.append((round(x * 1000), round(y * 1000), e["width"], e["capacity"]))
    res[run] = dict(sum=sum(q[3] for q in kept), want=want, raw_sum=raw, kept=kept)
    print(k, "read", run, "half-sheet sum", sum(q[3] for q in kept), "want", want, "raw", raw, "kept", kept, flush=True)
print("empty answers", empties)
json.dump(dict(res=res, empties=empties), open("data/lab/audit2/L7_exit162.json", "w"), indent=1)
print("spent", round(m.spent("audit2"), 4))
