"""POST HOC, marked as such in every output. Recipe P: Spark clusters, then two rounds of zoomed Spark crops asking for the whole stair, then the tracker; P pin puts the same box on every floor the cluster saw."""
import sys
sys.path.insert(0, "lab/tightseed")
from core import *
import dev1, post1


def p_seeds(st, cl, fr):
    jobs = [(st, str(i + 1), fr[c["seed_floor"]], c["seed_box"]) for i, c in enumerate(cl)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(post1.spark_whole, jobs))
    return [(i + 1, c["seed_floor"], r["box"]) for i, (c, r) in enumerate(zip(cl, res))]


def run(st):
    cl = dev1.loose_clusters(st); fr = frames(st)
    json.dump(cl, open(SPK / f"{st}_clusters.json", "w"))
    sb = p_seeds(st, cl, fr)
    track(f"{st}__P_960", st, sb)
    pin = [(o, f, b) for (o, _, b), c in zip(sb, cl) for f in c["floors"]]
    track(f"{st}__Ppin_960", st, pin)


if __name__ == "__main__":
    for st in sys.argv[1:] or HELD:
        run(st)
        for me in ("P_960", "Ppin_960"):
            tr = json.load(open(SAMD / f"{st}__{me}.json"))
            rows, false = score(st, tr)
            print("POST HOC", st, me, summary(rows), "false", len(false), flush=True)
    print("spent", round(m.spent("tightseed"), 4))
