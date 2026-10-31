"""POST HOC comparator: no SAM. The P box, unmoved, on every floor of the registered stack, with one Spark verdict per floor. Only clusters seeded on a truth stair: spend control."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import multi as mu, held_score as hs

for st in sys.argv[1:]:
    tr = json.load(open(SAMD / f"{st}__P_960.json")); fr = frames(st); n = TRUTH[st]["n"]
    own = [(o, f, b) for o, f, b in tr["seeds"] if [k for k in assign(st, b) if TRUTH[st]["stairs"][k]["cls"] != "amb"]]
    jobs = [(st, f"NS_{o}_{g}", fr[g], b) for o, f, b in own for g in range(n)]
    with ThreadPoolExecutor(8) as ex:
        ans = list(ex.map(mu.spark_yes, jobs))
    tracks = {}
    i = 0
    for o, f, b in own:
        tracks[str(o)] = {}
        for g in range(n):
            if ans[i]:
                tracks[str(o)][str(g)] = dict(box=b, area=1, conf=1.0)
            i += 1
    d = dict(seeds=[[o, f, b] for o, f, b in own], tracks=tracks)
    json.dump(d, open(SAMD / f"{st}__NS.json", "w"))
    rows, false = score(st, d)
    print("POST HOC no SAM, Spark verdict", st, summary(rows), "wrong passes", [(r["stair"], r["floor"]) for r in rows if r["cls"] in SCORED and not r["present"] and not r["ok"]], "misses", [(r["stair"], r["floor"]) for r in rows if r["cls"] in SCORED and r["present"] and not r["ok"]], flush=True)
    print("spent", round(m.spent("tightseed"), 4), flush=True)
