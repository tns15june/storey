"""Spark crop verdict on mask boxes of tracks seeded on a truth stair. Tags on the command line. Cached per mask."""
import sys
sys.path.insert(0, "/Users/tsharma/Github/Meta/lab/tightseed")
from core import *
import multi as mu

for me in sys.argv[1:]:
    jobs = []
    for st in HELD:
        p = SAMD / f"{st}__{me}.json"
        if not p.exists():
            continue
        tr = json.load(open(p)); fr = frames(st)
        own = set()
        for o, f, b in tr["seeds"]:
            if [k for k in assign(st, b) if TRUTH[st]["stairs"][k]["cls"] != "amb"]:
                own.add(str(o))
        for o in own:
            for f, mk in tr["tracks"].get(o, {}).items():
                if has(mk):
                    jobs.append((st, f"{me}_{o}_{f}", fr[int(f)], mk["box"]))
    print(me, "verdicts", len(jobs), flush=True)
    with ThreadPoolExecutor(8) as ex:
        r = list(ex.map(mu.spark_yes, jobs))
    print(me, "yes", sum(r), "of", len(r), "spent", round(m.spent("tightseed"), 4), flush=True)
